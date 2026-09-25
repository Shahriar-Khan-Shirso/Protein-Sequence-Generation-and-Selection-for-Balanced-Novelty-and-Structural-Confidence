"""Denoising transformer.

Predicts the clean latent z0 directly rather than the noise. The tangent
schedule is designed around z0-prediction so that reconstruction loss
rises roughly linearly in t; under noise-prediction the allocation
inverts and the high-noise region the sampler traverses first, which
sets the global fold, receives almost none of the gradient.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ScoreEstimator(nn.Module):
    """Pre-norm transformer with trainable positional encodings, per-block
    time gating and long skip connections.

    Positional encodings matter here: at high noise the input carries no
    positional information at all, so without them the network is
    permutation-equivariant over residues and cannot represent register
    or helix periodicity.
    """

    def __init__(
        self,
        latent_dim=480,
        hidden_dim=512,
        n_layers=6,
        n_heads=8,
        max_len=256,
        dropout=0.0,
        use_grad_ckpt=True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.use_grad_ckpt = use_grad_ckpt

        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        self.time_emb = SinusoidalTimeEmb(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_gates = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.skip_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.deep_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_t, t, mask, x_self_cond=None):
        """x_t (B, L, D); t (B,) in [0, 1]; mask (B, L) with True = real residue.

        Returns the predicted clean latent, zeroed at padded positions.
        """
        _, L, _ = x_t.shape
        if x_self_cond is None:
            x_self_cond = torch.zeros_like(x_t)

        h = torch.cat([x_t, x_self_cond], dim=-1)
        h = self.input_proj(h) + self.pos_emb[:, :L, :]

        t_emb = F.silu(self.time_proj(self.time_emb(t))).unsqueeze(1)
        pad_mask = ~mask

        shallow_out = deep_out = None
        n = len(self.blocks)

        for i, (block, gate) in enumerate(zip(self.blocks, self.time_gates)):
            h = h + gate(t_emb)
            # The self-conditioning pass runs under no_grad, where
            # checkpointing is pure overhead.
            if self.use_grad_ckpt and self.training and torch.is_grad_enabled():
                h = grad_ckpt(block, h, None, pad_mask, use_reentrant=False)
            else:
                h = block(h, src_key_padding_mask=pad_mask)

            if i == n // 2 - 1:
                shallow_out = h
            if i == n - 2:
                deep_out = h

        if shallow_out is not None and deep_out is not None:
            h = self.skip_proj(torch.cat([h, shallow_out], dim=-1))
            h = self.deep_proj(torch.cat([h, deep_out], dim=-1))

        out = self.out_proj(self.out_norm(h))
        return out * mask.unsqueeze(-1).float()
