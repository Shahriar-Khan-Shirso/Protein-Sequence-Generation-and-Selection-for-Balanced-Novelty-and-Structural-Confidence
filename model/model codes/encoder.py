"""Frozen protein language model encoder.

The encoder is never fine-tuned. It runs once over a corpus to produce a
per-residue latent tensor, which is cached; training, generation and
evaluation all operate in that latent space. Because it carries no
gradients and sits outside the training loop, its size affects only
preprocessing cost and latent width, not training-time memory.
"""

import torch
import torch.nn as nn

from ..constants import DEVICE, resolve_encoder


class ProteinEncoder(nn.Module):
    """Wraps an ESM-2 or ProtBERT model and returns padded per-residue
    representations together with a validity mask."""

    def __init__(self, encoder="35M", max_len=256, device=DEVICE):
        super().__init__()
        self.name, self.repr_layer, self.latent_dim = resolve_encoder(encoder)
        self.encoder_key = encoder
        self.max_len = max_len
        self.device = device
        self.is_protbert = self.repr_layer is None

        if self.is_protbert:
            from transformers import BertModel, BertTokenizer

            self.tokenizer = BertTokenizer.from_pretrained(
                self.name, do_lower_case=False
            )
            self.model = BertModel.from_pretrained(self.name)
        else:
            import esm

            self.model, self.alphabet = getattr(esm.pretrained, self.name)()
            self.batch_converter = self.alphabet.get_batch_converter()

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, sequences):
        """Return (reps, mask) of shape (B, max_len, latent_dim) and (B, max_len)."""
        seqs = [s[: self.max_len] for s in sequences]

        if self.is_protbert:
            spaced = [" ".join(list(s)) for s in seqs]
            batch = self.tokenizer(
                spaced, return_tensors="pt", padding=True, add_special_tokens=True
            ).to(self.device)
            out = self.model(**batch)
            # Strip [CLS] and the trailing [SEP]/padding.
            reps_full = out.last_hidden_state[:, 1:-1, :]
        else:
            data = [(f"p{i}", s) for i, s in enumerate(seqs)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)
            out = self.model(
                tokens, repr_layers=[self.repr_layer], return_contacts=False
            )
            reps_full = out["representations"][self.repr_layer][:, 1:-1, :]

        B = reps_full.shape[0]
        reps = torch.zeros(B, self.max_len, self.latent_dim, device=self.device)
        mask = torch.zeros(B, self.max_len, dtype=torch.bool, device=self.device)
        for i, s in enumerate(seqs):
            n = min(len(s), self.max_len)
            reps[i, :n, :] = reps_full[i, :n, :]
            mask[i, :n] = True
        return reps, mask
