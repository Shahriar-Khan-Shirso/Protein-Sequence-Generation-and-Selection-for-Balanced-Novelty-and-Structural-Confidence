"""Sequence decoder.

A bidirectional transformer over the full latent sequence, emitting
logits over the 20 canonical amino acids. It is trained from scratch
rather than reusing the encoder's masked-language-model head, so it
decodes into a 20-class space directly and the encoder's 33-token
vocabulary is never involved.

Attention across residue positions is what distinguishes this from a
per-position linear head: a corrupted position can be constrained by
its confident neighbours.
"""

import torch
import torch.nn as nn

from ..constants import AA_STR, VOCAB_SIZE


class SequenceDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=480,
        hidden_dim=480,
        n_layers=3,
        n_heads=8,
        vocab_size=VOCAB_SIZE,
        max_len=256,
        dropout=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, mask=None):
        # Guard against being called on flattened single positions, which
        # would run the decoder at sequence length 1 and leave every
        # positional embedding beyond index 0 untrained.
        assert z.dim() == 3, (
            f"SequenceDecoder expects (B, L, {self.latent_dim}) full-sequence "
            f"input, got {tuple(z.shape)}"
        )
        _, L, _ = z.shape
        assert L <= self.max_len, f"L={L} exceeds decoder max_len={self.max_len}"

        h = self.in_proj(z) + self.pos_emb[:, :L, :]
        h = self.transformer(h, src_key_padding_mask=(~mask) if mask is not None else None)
        return self.fc(self.norm(h))

    @torch.no_grad()
    def decode_to_strings(self, z_norm, mask, lengths):
        """Decode normalised latents to sequences of the given lengths."""
        idx = self.forward(z_norm, mask=mask).argmax(dim=-1).cpu()
        return [
            "".join(AA_STR[i] for i in idx[b, : int(lengths[b])].tolist())
            for b in range(idx.shape[0])
        ]
