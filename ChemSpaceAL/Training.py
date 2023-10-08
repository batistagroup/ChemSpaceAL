import numpy as np
import pandas as pd
import re
import wandb
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import os

from .Model import *
from .Dataset import *
from .Configuration import *

# Setting seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class Trainer:
    """Class to handle training and testing of GPT model."""
    
    def __init__(self, model, train_dataset, test_dataset=None, wandb=False):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = model.config
        self.stoi = train_dataset.stoi
        self.itos = train_dataset.itos
        self.wandb = wandb

    def train(self):
        """Training loop for the model."""
        model, config = self.model, self.config
        optimizer = model.configure_optimizers(config)
        scaler = GradScaler()
        self.tokens = 0 

        def run_epoch(split):
            """Run one epoch of training or validation."""
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            
            # Initialize data loader
            loader = DataLoader(data, shuffle=True, pin_memory=True, batch_size=config.batch_size, num_workers=config.num_workers)
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            
            # Iterate over batches
            for it, (x, y) in pbar:
                x, y = x.to(config.device), y.to(config.device)
                if config.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        with torch.set_grad_enabled(is_train):
                            logits, loss = model(x, y)
                            loss = loss.mean()
                            losses.append(loss.item())
                else:
                    with torch.cpu.amp.autocast():
                        with torch.set_grad_enabled(is_train):
                            logits, loss = model(x, y)
                            loss = loss.mean()
                            losses.append(loss.item())

                if is_train:
                    # Gradient accumulation and optimization
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Learning rate adjustment
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if config.lr_warmup and self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens)) 
                        else:
                            baseline = config.warmup_tokens if config.lr_warmup else 0
                            progress = float(self.tokens - baseline) / float(max(1, config.final_tokens - baseline))
                            lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    
                    # Log to wandb, if enabled
                    if self.wandb:
                        wandb.log({'step_train_loss': loss, 'train_step': it + epoch*len(loader), 'learning_rate': lr})
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            
            # Return mean loss
            return float(np.mean(losses))

        best_loss = float('inf')
        for epoch in range(config.epochs):
            print(f'{epoch=}')
            train_loss = run_epoch('train')
            log_dict = {'epoch_train_loss': train_loss, 'epoch': epoch + 1}
            
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                log_dict['epoch_valid_loss'] = test_loss
            
            if self.wandb:
                wandb.log(log_dict)

            good_model = False
            if self.test_dataset is None:
                good_model = True
            else:
                if test_loss < best_loss:
                    best_loss = test_loss
                    good_model = True

            # Save model checkpoint based on mode
            if good_model:
                if config.mode == 'Pretraining':
                    torch.save(self.model.state_dict(), self.config.pretrining_checkpoint_path)
                elif config.mode == 'Active Learning':
                    torch.save(self.model.state_dict(), self.config.al_checkpoint_path)


def train_GPT(training_dataset, 
              config_dict, 
              validation_dataset=None, 
              load_checkpoint=False, 
              log_wandb=False):
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

    # Set up model configuration
    mconf = GPTConfig(vocab_size=training_dataset.vocab_size, 
                      block_size=training_dataset.block_size,
                      warmup_tokens=0.1*training_dataset.len_data*training_dataset.block_size,
                      final_tokens=config_dict["epochs"]*training_dataset.len_data*training_dataset.block_size,
                      loss_ignore_index=training_dataset.stoi['<'],
                      **config_dict)
    
    # Initialize model and move it to the specified device
    model = GPT(mconf)
    model.to(config_dict["device"])

    # Initialize the Trainer with model and datasets
    trainer = Trainer(model, training_dataset, validation_dataset, wandb=log_wandb)

    # If using wandb, initialize the wandb logger
    if log_wandb: 
        wandb.init(project=config_dict["wandb_project"], name=config_dict["wandb_runname"])
        
    # Start training
    trainer.train()

    # Return trained model, trainer instance, and wandb logger
    return model, trainer, wandb
