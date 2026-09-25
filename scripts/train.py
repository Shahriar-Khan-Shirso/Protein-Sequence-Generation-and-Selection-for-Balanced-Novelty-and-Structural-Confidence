"""Train one configuration.

    python scripts/train.py --config configs/baseline_pdb_curated_35m_180ep.yaml
"""

import argparse

import pandas as pd
import torch

from sdima.config import ExperimentConfig
from sdima.data import (
    SeqLatentDataset,
    encode_corpus,
    latent_stats,
    normalise,
    residue_targets,
)
from sdima.diffusion import LatentDiffusion
from sdima.models import EMA, ProteinEncoder, ScoreEstimator, SequenceDecoder
from sdima.training import train as run_training
from sdima.training.aux_losses import CompositionObjectives, measure_reference_levels
from sdima.training.aux_losses import build_kmer_table, polar_mask
from sdima.training.refine_decoder import refine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(cfg.summary())

    sequences = pd.read_csv(cfg.data.train_csv)["sequence"].dropna().tolist()
    print(f"{len(sequences):,} training sequences")

    encoder = ProteinEncoder(cfg.model.encoder, cfg.data.max_len, device).to(device)
    latents, masks = encode_corpus(encoder, sequences, cfg.data.max_len)
    mean, std = latent_stats(latents, masks)
    latents = normalise(latents, masks, mean, std)
    targets = residue_targets(sequences, cfg.data.max_len)

    dataset = SeqLatentDataset(latents, masks, targets)

    denoiser = ScoreEstimator(
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.denoiser_hidden,
        n_layers=cfg.model.denoiser_layers,
        n_heads=cfg.model.denoiser_heads,
        max_len=cfg.data.max_len,
        dropout=cfg.model.dropout_denoiser,
    ).to(device)
    decoder = SequenceDecoder(
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.decoder_hidden,
        n_layers=cfg.model.decoder_layers,
        n_heads=cfg.model.decoder_heads,
        max_len=cfg.data.max_len,
        dropout=cfg.model.dropout_decoder,
    ).to(device)
    ema = EMA(denoiser, cfg.train.ema_decay)
    diffusion = LatentDiffusion(
        n_steps=cfg.diffusion.n_steps,
        schedule=cfg.diffusion.schedule,
        steepness=cfg.diffusion.steepness,
        spacing=cfg.diffusion.spacing,
        device=device,
    )

    n_den = sum(p.numel() for p in denoiser.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(
        f"trainable: denoiser {n_den/1e6:.1f}M + decoder {n_dec/1e6:.2f}M "
        f"= {(n_den+n_dec)/1e6:.1f}M (encoder frozen)"
    )

    composition = None
    if cfg.aux.composition_enabled:
        table = build_kmer_table(sequences, device=device)
        pm = polar_mask(device)
        levels = measure_reference_levels(
            sequences, table, pm, cfg.data.max_len, device, strict=cfg.aux.strict_refs
        )
        print("composition reference levels:", {k: v for k, v in levels.items() if k != "violation_rates"})
        print("real-sequence violation rates:", levels["violation_rates"])
        composition = CompositionObjectives(table, pm, levels)

    predictor = None
    if cfg.aux.proxy_enabled:
        state = torch.load(cfg.aux.proxy_predictor_path, map_location=device, weights_only=False)
        from sdima.training import PlddtPredictor

        predictor = PlddtPredictor(cfg.model.latent_dim, max_len=cfg.data.max_len).to(device)
        predictor.load_state_dict(state["state_dict"])
        predictor.eval()
        for p in predictor.parameters():
            p.requires_grad = False
        print(f"proxy predictor loaded | val Spearman {state.get('val_spearman', float('nan')):.3f}")

    denoiser, decoder, ema = run_training(
        cfg, denoiser, decoder, ema, diffusion, dataset, mean, std,
        composition_objectives=composition, plddt_predictor=predictor,
        device=device, resume=not args.no_resume,
    )

    if cfg.train.refine_decoder:
        refine(
            decoder, ema.get_model(), diffusion, latents, masks, targets,
            epochs=cfg.train.refine_epochs, lr=cfg.train.refine_lr,
            t_ceiling=cfg.train.refine_t_ceiling, device=device,
        )


if __name__ == "__main__":
    main()
