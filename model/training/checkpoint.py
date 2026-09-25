"""Checkpointing and resume.

Writes are atomic: the file is written to a temporary path and renamed,
so an interruption mid-write cannot leave a truncated checkpoint that
looks valid. Long runs on a single machine are interrupted often enough
for this to matter.

On resume, the configuration recorded in the checkpoint is asserted
against the current session. Normalisation statistics in particular are
restored rather than recomputed: recomputing them on a slightly
different subset would shift the latent space out from under a
half-trained denoiser.
"""

import glob
import os
import re

import torch


def atomic_save(obj, path):
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def find_latest(pattern="checkpoint_epoch*.pt"):
    """Return (epoch, path) for the highest-numbered checkpoint, or (0, None)."""
    found = {}
    for path in glob.glob(pattern):
        m = re.search(r"epoch(\d+)\.pt$", path)
        if m:
            found[int(m.group(1))] = path
    if not found:
        return 0, None
    latest = max(found)
    return latest, found[latest]


def build_state(
    epoch,
    global_step,
    denoiser,
    decoder,
    ema,
    optimizer,
    scheduler,
    latent_mean,
    latent_std,
    config,
):
    return {
        "epoch_num": epoch,
        "global_step": global_step,
        "denoiser": denoiser.state_dict(),
        "denoiser_ema": ema.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "latent_mean": latent_mean.cpu(),
        "latent_std": latent_std.cpu(),
        "parameterisation": "z0",
        "encoder": config.model.encoder,
        "latent_dim": config.model.latent_dim,
        "l_max": config.data.max_len,
        "schedule": config.diffusion.schedule,
        "steepness": config.diffusion.steepness,
        "ce_weight": config.train.ce_weight,
        "aux": {
            "proxy": config.aux.proxy_enabled,
            "proxy_weight": config.aux.proxy_weight,
            "proxy_target": config.aux.proxy_target,
            "composition": config.aux.composition_enabled,
        },
        "config_name": config.name,
    }


def assert_compatible(ckpt, config):
    """Fail loudly when a checkpoint does not match the current session.

    A silent mismatch here produces a model that trains without error and
    generates nonsense, which is expensive to diagnose after the fact.
    """
    checks = [
        ("parameterisation", ckpt.get("parameterisation"), "z0"),
        ("l_max", ckpt.get("l_max"), config.data.max_len),
        ("encoder", ckpt.get("encoder"), config.model.encoder),
        ("schedule", ckpt.get("schedule"), config.diffusion.schedule),
    ]
    for name, got, want in checks:
        if got is not None and got != want:
            raise ValueError(
                f"checkpoint {name}={got!r} does not match this session "
                f"({want!r}); start from a fresh checkpoint or fix the config"
            )


def restore(ckpt, denoiser, decoder, ema, optimizer=None, scheduler=None):
    denoiser.load_state_dict(ckpt["denoiser"])
    decoder.load_state_dict(ckpt["decoder"])
    ema.load_state_dict(ckpt["denoiser_ema"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch_num", 0), ckpt.get("global_step", 0)
