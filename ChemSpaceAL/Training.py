import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data.dataloader import DataLoader

from typing import Optional

from ChemSpaceAL.Dataset import SMILESDataset
from ChemSpaceAL.Configuration import Config, ModelConfig
from ChemSpaceAL.Model import GPT

import wandb
from tqdm import tqdm

# Setting seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class Trainer:
    """Class to handle training and testing of GPT model."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        model_config: ModelConfig,
        train_dataset: SMILESDataset,
        valid_dataset: Optional[SMILESDataset] = None,
        wandb=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.model_config = model_config
        self.stoi = train_dataset.stoi
        self.itos = train_dataset.itos
        self.wandb = wandb

    def run_epoch(self, split: str, epoch: int):
        """Run one epoch of training or validation."""
        assert split in {
            "train",
            "valid",
        }, "Only `train` and `valid` splits are supported"
        is_train = split == "train"
        self.model.train(is_train)
        data = self.train_dataset if is_train else self.valid_dataset

        # (mypy isn't smart enough to recognize that SMILESDataset inherits from Dataset)
        loader = DataLoader(
            data,  # type:ignore
            shuffle=True,
            pin_memory=True,
            batch_size=self.model_config.batch_size,
            num_workers=self.model_config.num_workers,
        )
        losses = []
        pbar = (
            tqdm(enumerate(loader), total=len(loader))
            if is_train
            else enumerate(loader)
        )

        # Iterate over batches
        for it, (x, y) in pbar:
            x, y = x.to(self.model_config.device), y.to(self.model_config.device)
            if self.model_config.device == "cuda":
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        logits, loss = self.model(x, y)
                        loss = loss.mean()
                        losses.append(loss.item())
            else:
                with torch.cpu.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        logits, loss = self.model(x, y)
                        loss = loss.mean()
                        losses.append(loss.item())

            if is_train:
                # Gradient accumulation and optimization
                self.model.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Learning rate adjustment
                if self.model_config.lr_decay:
                    self.tokens += (y >= 0).sum()
                    if (
                        self.model_config.train_params["lr_warmup"]
                        and self.tokens < self.model_config.num_warmup_tokens
                    ):
                        lr_mult = float(self.tokens) / float(
                            max(1, self.model_config.num_warmup_tokens)
                        )
                    else:
                        baseline = (
                            self.model_config.num_warmup_tokens
                            if self.model_config.train_params["lr_warmup"]
                            else 0
                        )
                        progress = float(self.tokens - baseline) / float(
                            max(1, self.model_config.total_num_tokens - baseline)
                        )
                        lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
                    lr = self.model_config.train_params["learning_rate"] * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                else:
                    lr = self.model_config.train_params["lr"]

                # Log to wandb, if enabled
                if self.wandb:
                    wandb.log(
                        {
                            "step_train_loss": loss,
                            "train_step": it + epoch * len(loader),
                            "learning_rate": lr,
                        }
                    )
                pbar.set_description(
                    f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                )

        # Return mean loss
        return float(np.mean(losses))

    def train(self):
        """Training loop for the model."""
        self.scaler = GradScaler()
        self.tokens = 0

        best_loss = float("inf")
        for epoch in range(self.model_config.train_params["epochs"]):
            train_loss = self.run_epoch("train", epoch)
            log_dict = {"epoch_train_loss": train_loss, "epoch": epoch + 1}

            if self.valid_dataset is not None:
                valid_loss = self.run_epoch("valid", epoch)
                log_dict["epoch_valid_loss"] = valid_loss

            if self.wandb:
                wandb.log(log_dict)

            good_model = False
            if self.valid_dataset is None:
                good_model = True
            else:
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    good_model = True

            if good_model:
                torch.save(
                    self.model.state_dict(),
                    self.model_config.train_params["save_model_weight"],
                )


def train_GPT(
    config: Config,
    training_dataset: SMILESDataset,
    validation_dataset: Optional[SMILESDataset] = None,
    load_checkpoint: bool = False,
    log_wandb: bool = False,
):
    """
    Trains the GPT model using the provided training_dataset, configuration dictionary,
    and optional validation_dataset.

    Args:
    - training_dataset: Dataset used for training.
    - config_dict: Dictionary containing configuration parameters for the GPT model and training loop.
    - validation_dataset: Optional dataset used for validation.
    - load_checkpoint: If True, loads the model from a saved checkpoint (Note: this isn't implemented in the code provided).
    - log_wandb: If True, logs training metrics to wandb.

    Returns:
    - model: The trained GPT model.
    - trainer: The Trainer object that trained the model.
    - wandb: wandb logger object.
    """
    mconf = config.model_config
    assert {"epochs", "learning_rate", "lr_warmup"} - set(
        mconf.train_params.keys()
    ) == set(), "Please call .set_training_parameters on config before trying to train the model"

    if log_wandb:
        assert (
            "wandb_runname" in mconf.train_params
        ), "if you want to log training run to wandb, please provide wandb_project_name and wandb_runname when calling .set_training_parameters on config"

    if load_checkpoint:
        assert (
            ckpt := mconf.train_params["load_model_weight"]
        ) is not None, (
            "please provide load_weight_path to .set_training_parameters of the config"
        )

    total_num_tokens = (
        mconf.train_params["epochs"]
        * training_dataset.len_data
        * training_dataset.block_size
    )

    mconf.set_dataset_attributes(
        vocab_size=training_dataset.vocab_size,
        block_size=training_dataset.block_size,
        num_warmup_tokens=int(
            0.1 * training_dataset.len_data * training_dataset.block_size
        ),
        total_num_tokens=total_num_tokens,
        loss_ignore_index=training_dataset.stoi["<"],
    )

    model = GPT(mconf)
    if load_checkpoint:
        model.load_state_dict(torch.load(ckpt))
    optimizer = model.configure_optimizers(
        weight_decay=mconf.weight_decay,
        lr=mconf.train_params["learning_rate"],
        betas=mconf.betas,
        rho=mconf.rho,
    )
    model.to(mconf.device)
    torch.compile(model)

    if log_wandb:
        wandb.init(
            project=mconf.train_params["wandb_project_name"],
            name=mconf.train_params["wandb_runname"],
        )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        model_config=mconf,
        train_dataset=training_dataset,
        valid_dataset=validation_dataset,
        wandb=wandb if wandb.run is not None else None,
    )
    trainer.train()
    if log_wandb:
        wandb.finish()
    return model, trainer
