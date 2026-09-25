"""Generate candidate sequences from a trained checkpoint.

    python scripts/generate.py --config configs/swissprot_225ep.yaml --checkpoint swissprot_225ep.pt
"""

import argparse

import pandas as pd
import torch

from sdima.config import ExperimentConfig
from sdima.data import length_pool
from sdima.diffusion import LatentDiffusion
from sdima.generation import generate, validity_report
from sdima.models import ScoreEstimator, SequenceDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="generated_raw.csv")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("l_max") not in (None, cfg.data.max_len):
        raise ValueError("checkpoint max_len does not match this config")

    denoiser = ScoreEstimator(
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.denoiser_hidden,
        n_layers=cfg.model.denoiser_layers,
        max_len=cfg.data.max_len,
    ).to(device)
    decoder = SequenceDecoder(
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.decoder_hidden,
        n_layers=cfg.model.decoder_layers,
        max_len=cfg.data.max_len,
    ).to(device)

    # Generation uses the averaged denoiser weights.
    denoiser.load_state_dict(ckpt["denoiser_ema"])
    decoder.load_state_dict(ckpt["decoder"])

    diffusion = LatentDiffusion(
        n_steps=cfg.diffusion.n_steps,
        schedule=cfg.diffusion.schedule,
        steepness=cfg.diffusion.steepness,
        spacing=cfg.diffusion.spacing,
        device=device,
    )

    train = pd.read_csv(cfg.data.train_csv)
    pool = length_pool(train, cfg.data.min_len, cfg.data.max_len)

    df = generate(
        denoiser, decoder, diffusion, pool,
        n_sequences=cfg.generation.n_candidates,
        batch_size=cfg.generation.batch_size,
        max_len=cfg.data.max_len,
        latent_dim=cfg.model.latent_dim,
        n_sample_steps=cfg.diffusion.sample_steps,
        temperature=cfg.generation.temperature,
        seed=cfg.generation.seed,
        device=device,
    )
    df.to_csv(args.out, index=False)

    rep = validity_report(df, train["sequence"].dropna().tolist())
    print(f"\nvalidity | lengths match: {rep['length_matches']} | "
          f"non-canonical residues: {rep['non_canonical'] or 'none'} | "
          f"duplicate fraction: {rep['duplicate_frac']:.4f}")
    print(f"saved {len(df)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
