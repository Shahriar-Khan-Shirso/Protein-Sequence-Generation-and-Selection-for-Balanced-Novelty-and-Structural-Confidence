"""Training loop.

Denoiser and decoder are optimised together in a single step, so the
decoder never learns from a stale denoiser, but their gradients stay
separate (see objectives.py).

Self-conditioning runs the denoiser once without its own estimate, then
feeds that estimate back, detached, on the gradient-carrying pass. It is
applied on half of steps: the unconditional path is exactly what the
first sampling step from pure noise uses, so it has to be trained too.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..constants import PAD_IDX
from . import checkpoint as ckpt_utils
from .aux_losses import proxy_foldability_loss
from .objectives import decode_accuracy, decoder_cross_entropy, masked_latent_mse


def build_optimizer(denoiser, decoder, cfg, steps_per_epoch):
    optimizer = AdamW(
        list(denoiser.parameters()) + list(decoder.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    warmup = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=cfg.train.warmup_steps,
    )
    cawr = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.train.cawr_period_epochs * steps_per_epoch,
        T_mult=1,
        eta_min=cfg.train.lr * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cawr],
        milestones=[cfg.train.warmup_steps],
    )
    return optimizer, scheduler


def train(
    cfg,
    denoiser,
    decoder,
    ema,
    diffusion,
    dataset,
    latent_mean,
    latent_std,
    composition_objectives=None,
    plddt_predictor=None,
    device=None,
    resume=True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    steps_per_epoch = len(loader)
    optimizer, scheduler = build_optimizer(denoiser, decoder, cfg, steps_per_epoch)

    start_epoch, global_step = 0, 0
    if resume:
        last_epoch, path = ckpt_utils.find_latest(
            f"{cfg.train.checkpoint_name.replace('.pt', '')}_epoch*.pt"
        )
        if path is not None:
            state = torch.load(path, map_location=device, weights_only=False)
            ckpt_utils.assert_compatible(state, cfg)
            start_epoch, global_step = ckpt_utils.restore(
                state, denoiser, decoder, ema, optimizer, scheduler
            )
            print(f"resuming from {path} after epoch {start_epoch}")

    latent_dim = cfg.model.latent_dim
    base = cfg.train.checkpoint_name.replace(".pt", "")

    print(
        f"training {cfg.name}: {cfg.train.epochs} epochs | "
        f"batch {cfg.train.batch_size} | {steps_per_epoch} steps/epoch"
    )

    for epoch in range(start_epoch, cfg.train.epochs):
        epoch_num = epoch + 1
        totals = {"mse": 0.0, "ce_clean": 0.0, "ce_hat": 0.0}
        acc = {"clean": 0.0, "hat": 0.0}
        aux_totals = {"kmer": 0.0, "polar": 0.0, "entropy": 0.0, "proxy": 0.0}
        n_batches = 0

        composition_active = (
            composition_objectives is not None
            and cfg.aux.composition_enabled
            and epoch_num > cfg.aux.composition_warmup_epochs
        )
        proxy_active = (
            plddt_predictor is not None
            and cfg.aux.proxy_enabled
            and epoch_num > cfg.aux.proxy_warmup_epochs
        )

        for x0, mask, aa_tgt in tqdm(
            loader, desc=f"epoch {epoch_num}/{cfg.train.epochs}", leave=False
        ):
            x0 = x0.to(device, non_blocking=True).float()
            mask = mask.to(device, non_blocking=True)
            aa_tgt = aa_tgt.to(device, non_blocking=True)
            B = x0.shape[0]
            mask_f = mask.unsqueeze(-1).float()

            t_idx = torch.randint(0, diffusion.n_steps, (B,), device=device)
            t_cont = t_idx.float() / diffusion.n_steps
            x_t, _ = diffusion.q_sample(x0, t_idx)
            x_t = x_t * mask_f

            x_self_cond = None
            if torch.rand(1).item() < cfg.train.self_cond_prob:
                with torch.no_grad():
                    x_self_cond = denoiser(x_t, t_cont, mask, None)
                    x_self_cond = (
                        x_self_cond.clamp(
                            -cfg.diffusion.clamp_val, cfg.diffusion.clamp_val
                        )
                        * mask_f
                    ).detach()

            x0_hat = denoiser(x_t, t_cont, mask, x_self_cond)
            mse = masked_latent_mse(x0_hat, x0, mask_f, latent_dim)

            signal_w = diffusion.sqrt_alphas[t_idx]
            ce, ce_clean, ce_hat, logits_clean, logits_hat = decoder_cross_entropy(
                decoder, x0, x0_hat, mask, aa_tgt, signal_w
            )

            loss = mse + cfg.train.ce_weight * ce

            if composition_active:
                valid = (aa_tgt != PAD_IDX).float()
                l_k, l_p, l_e, _ = composition_objectives(logits_hat, valid)
                loss = loss + (
                    cfg.aux.kmer_weight * l_k
                    + cfg.aux.polar_weight * l_p
                    + cfg.aux.entropy_weight * l_e
                )
                aux_totals["kmer"] += float(l_k.detach())
                aux_totals["polar"] += float(l_p.detach())
                aux_totals["entropy"] += float(l_e.detach())

            if proxy_active:
                l_proxy, _ = proxy_foldability_loss(
                    plddt_predictor, x0_hat, mask, cfg.aux.proxy_target
                )
                loss = loss + cfg.aux.proxy_weight * l_proxy
                aux_totals["proxy"] += float(l_proxy.detach())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(denoiser.parameters()) + list(decoder.parameters()),
                cfg.train.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            ema.update()
            global_step += 1

            totals["mse"] += mse.item()
            totals["ce_clean"] += ce_clean.item()
            totals["ce_hat"] += ce_hat.item()
            acc["clean"] += decode_accuracy(logits_clean, aa_tgt)
            acc["hat"] += decode_accuracy(logits_hat, aa_tgt)
            n_batches += 1

        n = max(1, n_batches)
        line = (
            f"epoch {epoch_num}/{cfg.train.epochs} | "
            f"z0-MSE {totals['mse']/n:.4f} | "
            f"CE(clean) {totals['ce_clean']/n:.4f} [acc {acc['clean']/n*100:.1f}%] | "
            f"CE(z0_hat) {totals['ce_hat']/n:.4f} [acc {acc['hat']/n*100:.1f}%]"
        )
        if composition_active:
            line += (
                f" | bio {aux_totals['kmer']/n:.4f}/"
                f"{aux_totals['polar']/n:.4f}/{aux_totals['entropy']/n:.4f}"
            )
        if proxy_active:
            line += f" | proxy {aux_totals['proxy']/n:.4f}"
        print(line)

        state = ckpt_utils.build_state(
            epoch_num,
            global_step,
            denoiser,
            decoder,
            ema,
            optimizer,
            scheduler,
            latent_mean,
            latent_std,
            cfg,
        )
        ckpt_utils.atomic_save(state, f"{base}_latest.pt")
        if epoch_num % cfg.train.checkpoint_every == 0:
            ckpt_utils.atomic_save(state, f"{base}_epoch{epoch_num}.pt")

    ckpt_utils.atomic_save(state, cfg.train.checkpoint_name)
    print(f"training complete -> {cfg.train.checkpoint_name}")
    return denoiser, decoder, ema
