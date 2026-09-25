"""Score candidates and apply selection.

    python scripts/select.py --config configs/swissprot_225ep.yaml

Scores every candidate on the composite, then takes a global top-K.
Also writes a matched random sample of the same size, which is what
makes the contribution of selection measurable: the same trained model
is evaluated on both.
"""

import argparse

import pandas as pd
import torch

from sdima.config import ExperimentConfig
from sdima.models import ProteinEncoder
from sdima.selection import (
    esm2_pll,
    normalise_pll,
    score_all,
    score_frame,
    select_top_k,
    unselected_sample,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--candidates", default="generated_raw.csv")
    ap.add_argument("--out", default="scored_sequences.csv")
    ap.add_argument("--selected-out", default="selected.csv")
    ap.add_argument("--unselected-out", default="unselected.csv")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seqs = pd.read_csv(args.candidates)["sequence"].tolist()
    train_seqs = pd.read_csv(cfg.data.train_csv)["sequence"].dropna().tolist()
    print(f"scoring {len(seqs)} candidates")

    bio = score_all(
        seqs, train_seqs,
        weights=(cfg.selection.kmer_sub_weight,
                 cfg.selection.polar_sub_weight,
                 cfg.selection.entropy_sub_weight),
    )

    # The scorer is fixed across every experiment, independent of which
    # encoder the model was trained on, so selection is identical when
    # the representation model is ablated.
    scorer = ProteinEncoder("35M", cfg.data.max_len, device).to(device)
    raw_pll = esm2_pll(
        seqs, scorer.model, scorer.alphabet,
        max_len=cfg.data.max_len, stride=cfg.selection.pll_stride, device=device,
    )
    proxy = normalise_pll(raw_pll)

    scored = score_frame(seqs, bio, proxy, cfg.selection.bio_weight, cfg.selection.pll_weight)
    scored.to_csv(args.out, index=False)

    unselected_sample(scored, n=cfg.selection.top_k, seed=cfg.generation.seed).to_csv(
        args.unselected_out, index=False
    )

    if cfg.selection.enabled:
        if cfg.selection.cluster_rule != "none":
            print(
                f"[note] cluster_rule={cfg.selection.cluster_rule} is an ablation; "
                "run scripts/cluster_ablation.py for that path"
            )
        select_top_k(scored, cfg.selection.top_k).to_csv(args.selected_out, index=False)
    else:
        print("selection disabled for this configuration")


if __name__ == "__main__":
    main()
