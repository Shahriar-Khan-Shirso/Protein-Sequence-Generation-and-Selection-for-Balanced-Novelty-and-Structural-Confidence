"""Learned confidence predictor used by the proxy foldability objective.

The natural supervisory signal for foldability, pLDDT, comes from a
structure predictor that is not differentiable with respect to the
latent. This fits a small differentiable model to latent-pLDDT pairs
once, freezes it, and uses it as a stand-in during training.

The label set is deliberately heterogeneous. Real sequences occupy a
narrow band near 80, and a predictor fitted only to those converges to a
constant function whose gradient is zero. Composition-matched shuffled
sequences supply the low end and are the informative ones: preserving
composition exactly and destroying only residue order forces the
predictor to read structural arrangement rather than count residue types.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


class PlddtPredictor(nn.Module):
    """Masked latent sequence to a scalar on the normalised pLDDT scale."""

    def __init__(self, latent_dim, hidden=256, n_layers=3, n_heads=8, max_len=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, z, mask):
        h = self.in_proj(z) + self.pos_emb[:, : z.shape[1], :]
        h = self.norm(self.tr(h, src_key_padding_mask=~mask))
        mf = mask.unsqueeze(-1).float()
        return self.head((h * mf).sum(1) / mf.sum(1).clamp(min=1)).squeeze(-1)


def make_negative_examples(sequences, n_shuffled=150, n_random=100, seed=0):
    """Composition-matched shuffles and random sequences.

    Shuffles preserve amino acid composition and destroy only order, so a
    predictor cannot separate them from real sequences by counting
    residues.
    """
    from ..constants import AA_STR

    rng = random.Random(seed)
    out = []
    pool = rng.sample(sequences, min(n_shuffled, len(sequences)))
    for s in pool:
        chars = list(s)
        rng.shuffle(chars)
        out.append(("shuffled", "".join(chars)))

    lengths = [len(s) for s in rng.sample(sequences, min(n_random, len(sequences)))]
    for L in lengths:
        out.append(("random", "".join(rng.choice(AA_STR) for _ in range(L))))
    return out


def fit_predictor(
    latents,
    masks,
    plddt_labels,
    latent_dim,
    max_len=256,
    epochs=40,
    batch_size=32,
    lr=3e-4,
    noise_aug_max=0.40,
    val_frac=0.15,
    device=None,
    seed=0,
):
    """Fit and freeze the predictor. Returns (predictor, metrics).

    Latents are perturbed with Gaussian noise of randomly sampled scale
    during fitting, because at training time the predictor is applied to
    the denoiser's prediction rather than to a clean latent.
    """
    from scipy.stats import spearmanr

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = PlddtPredictor(latent_dim, max_len=max_len).to(device)

    Y = torch.tensor(np.asarray(plddt_labels) / 100.0, dtype=torch.float32)
    ds = TensorDataset(latents, masks, Y)

    perm = torch.randperm(
        len(ds), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    n_val = max(48, int(len(ds) * val_frac))
    tr_ld = DataLoader(
        Subset(ds, perm[n_val:]), batch_size=batch_size, shuffle=True, drop_last=True
    )
    va_ld = DataLoader(Subset(ds, perm[:n_val]), batch_size=batch_size)

    opt = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * max(1, len(tr_ld)), eta_min=1e-6
    )

    @torch.no_grad()
    def evaluate():
        predictor.eval()
        P, T = [], []
        for z, m, y in va_ld:
            P.append(predictor(z.to(device).float(), m.to(device)).cpu())
            T.append(y)
        predictor.train()
        P = torch.cat(P).numpy() * 100
        T = torch.cat(T).numpy() * 100
        rho = spearmanr(P, T).correlation
        return float(np.abs(P - T).mean()), (0.0 if np.isnan(rho) else float(rho))

    for ep in range(epochs):
        for z, m, y in tr_ld:
            z, m, y = z.to(device).float(), m.to(device), y.to(device)
            sigma = torch.rand(z.shape[0], 1, 1, device=device) * noise_aug_max
            z = (z + sigma * torch.randn_like(z)) * m.unsqueeze(-1).float()
            loss = F.mse_loss(predictor(z, m), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt.step()
            sch.step()
        if (ep + 1) % 10 == 0:
            mae, rho = evaluate()
            print(f"  epoch {ep+1}/{epochs} | val MAE {mae:.2f} | Spearman {rho:.3f}")

    mae, rho = evaluate()
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad = False

    if rho < 0.5:
        print(
            "[warning] rank correlation below 0.5: the predictor barely orders "
            "sequences, so the proxy objective would mostly inject noise"
        )

    return predictor, {"val_mae": mae, "val_spearman": rho, "n_labels": len(ds)}
