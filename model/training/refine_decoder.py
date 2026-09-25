"""Decoder refinement.

An optional stage after joint training, fine-tuning the decoder alone on
the denoiser's own output.

Two things make this different from naive noise augmentation. The
corruption comes from the EMA denoiser rather than from added Gaussian
noise, because diffusion error is correlated across positions and
dimensions and lies off the data manifold, so Gaussian noise is not the
error distribution the decoder actually faces. And the timestep is
capped: the decoder is invoked once at the end of sampling, so it is
refined only in the regime it is used in.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, TensorDataset

from ..constants import PAD_IDX, VOCAB_SIZE
from .objectives import decode_accuracy


@torch.no_grad()
def corrupt_via_denoiser(ema_denoiser, diffusion, x0, mask, t_ceiling=250, clamp=3.0):
    """The EMA denoiser's prediction at a randomly sampled low timestep."""
    B = x0.shape[0]
    t_idx = torch.randint(0, t_ceiling, (B,), device=x0.device)
    t_cont = t_idx.float() / diffusion.n_steps
    mask_f = mask.unsqueeze(-1).float()

    x_t, _ = diffusion.q_sample(x0, t_idx)
    x_t = x_t * mask_f
    sc = (ema_denoiser(x_t, t_cont, mask, None).clamp(-clamp, clamp)) * mask_f
    return (ema_denoiser(x_t, t_cont, mask, sc).clamp(-clamp, clamp)) * mask_f


def refine(
    decoder,
    ema_denoiser,
    diffusion,
    latents,
    masks,
    aa_targets,
    epochs=30,
    lr=5e-5,
    batch_size=48,
    t_ceiling=250,
    val_frac=0.05,
    device=None,
    seed=0,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ema_denoiser.eval()
    for p in ema_denoiser.parameters():
        p.requires_grad = False

    ds = TensorDataset(latents, masks, aa_targets)
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = max(32, int(len(ds) * val_frac))
    tr_ld = DataLoader(
        Subset(ds, perm[n_val:]), batch_size=batch_size, shuffle=True, drop_last=True
    )
    va_ld = DataLoader(Subset(ds, perm[:n_val]), batch_size=batch_size)

    opt = AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * max(1, len(tr_ld)), eta_min=1e-6
    )

    @torch.no_grad()
    def evaluate(loader):
        decoder.eval()
        clean_acc = hat_acc = 0.0
        n = 0
        for x0, mask, tgt in loader:
            x0, mask, tgt = x0.to(device).float(), mask.to(device), tgt.to(device)
            clean_acc += decode_accuracy(decoder(x0, mask=mask), tgt)
            x0_hat = corrupt_via_denoiser(
                ema_denoiser, diffusion, x0, mask, t_ceiling
            )
            hat_acc += decode_accuracy(decoder(x0_hat, mask=mask), tgt)
            n += 1
        decoder.train()
        return clean_acc / max(1, n), hat_acc / max(1, n)

    c0, h0 = evaluate(va_ld)
    print(f"before refinement | clean {c0*100:.1f}% | denoiser output {h0*100:.1f}%")

    for ep in range(epochs):
        for x0, mask, tgt in tr_ld:
            x0, mask, tgt = x0.to(device).float(), mask.to(device), tgt.to(device)
            x0_hat = corrupt_via_denoiser(ema_denoiser, diffusion, x0, mask, t_ceiling)
            logits = decoder(x0_hat, mask=mask)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                tgt.reshape(-1),
                ignore_index=PAD_IDX,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            c, h = evaluate(va_ld)
            print(
                f"  epoch {ep+1}/{epochs} | clean {c*100:.1f}% | "
                f"denoiser output {h*100:.1f}%"
            )

    return decoder
