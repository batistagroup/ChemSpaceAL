import yaml  # type:ignore
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from Sophia.sophia import SophiaG

from typing import Tuple


class GPTConfig:
    """GPT Configuration class."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        num_warmup_tokens: int,
        total_num_tokens: int,
        loss_ignore_index: int,
        batch_size: int,
        num_workers: int,
        device: str,
        lr: float,
        lr_decay: bool,
        lr_warmup: bool,
        save_model_path: str,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_warmup_tokens = num_warmup_tokens
        self.total_num_tokens = total_num_tokens
        self.loss_ignore_index = loss_ignore_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_warmup = lr_warmup
        self.save_model_path = save_model_path
        # for k, v in kwargs.items():
        #     setattr(self, k, v)

    def export_attributes(self, export_path: str):
        """Export model attributes to a yaml file."""
        with open(export_path, "w") as f:
            yaml.dump(vars(self), f)

    def load_attributes(self, load_path: str):
        """Load model attributes from a yaml file."""
        with open(load_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.__dict__.update(config_dict)


class SelfAttention(nn.Module):
    """Self-attention block for the GPT architecture."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.config = config
        # Defining the query, key, and value linear transformations
        self.query = nn.Linear(config.n_embed, config.n_embed, bias=config.att_bias)
        self.key = nn.Linear(config.n_embed, config.n_embed, bias=config.att_bias)
        self.value = nn.Linear(config.n_embed, config.n_embed, bias=config.att_bias)

        # Dropout layers
        self.attn_drop = nn.Dropout(config.att_drop_rate)
        self.resid_drop = nn.Dropout(config.att_drop_rate)

        # Projection layer
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head

        # Registering a lower-triangular mask for causal attention
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, C // self.n_head)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head)

        # Flash-based implementation for scaled dot-product attention
        if self.config.do_flash:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.config.att_drop_rate if self.training else 0,
                is_causal=True,
            )
            y = y.transpose(1, 2)
        else:
            # Standard self-attention implementation
            att = torch.einsum("bths,bihs->bhti", q, k) / np.sqrt(k.size(-1))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = torch.einsum("bhtq,bqhs->bths", att, v)
            self.att_weights = att

        self.attended = y
        y = y.contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        self.out = y
        return y


class Block(nn.Module):
    """GPT block that consists of one self-attention module and one MLP."""

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = SelfAttention(config)
        # Feed-forward neural network
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, config.ff_mult * config.n_embed),
            nn.GELU() if config.doGELU else nn.ReLU(),
            nn.Linear(config.ff_mult * config.n_embed, config.n_embed),
            nn.Dropout(config.att_drop_rate),
        )

    def forward(self, x):
        y = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Main GPT model."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)

        # Type embeddings, typically used for distinguishing between different kinds of input
        self.type_emb = nn.Embedding(2, config.n_embed)

        # Position embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embed))

        # Dropout layer
        self.drop = nn.Dropout(config.gpt_drop_rate)

        # GPT blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # Final layer normalization and the output head
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=config.gpt_bias)
        self.block_size = config.block_size

        # Initializing weights
        self.apply(self._init_weights)

    def get_block_size(self):
        """Returns the block size."""
        return self.block_size

    def _init_weights(self, module):
        """Initialize weights of the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(
        self, weight_decay: float, lr: float, betas: Tuple[float, float], rho: float
    ):
        """Configure optimizers based on training configuration."""

        decay, no_decay = set(), set()
        whitelist_weight_modules = torch.nn.Linear
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # Separating parameters that decay and those that don't
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias") or ("bias" in pn):
                    no_decay.add(fpn)
                elif (pn.endswith("weight") or ("weight" in pn)) and isinstance(
                    m, whitelist_weight_modules
                ):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        no_decay.add("pos_emb")

        # Checking consistency in parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0

        # Creating optimizer groups and initializing the SophiaG optimizer
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = SophiaG(
            optim_groups, lr=lr, betas=betas, rho=rho, weight_decay=weight_decay
        )
        return optimizer

    def forward(self, idx, targets=None):
        """Forward pass for the GPT model."""
        b, t = idx.size()
        assert t <= self.block_size

        # Token, position, and type embeddings
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        type_embeddings = self.type_emb(
            torch.ones((b, t), dtype=torch.long, device=idx.device)
        )
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)

        # Passing through the GPT blocks
        for layer in self.blocks:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)

        # Compute loss if targets are provided
        loss = (
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.loss_ignore_index,
            )
            if targets is not None
            else None
        )
        return logits, loss
