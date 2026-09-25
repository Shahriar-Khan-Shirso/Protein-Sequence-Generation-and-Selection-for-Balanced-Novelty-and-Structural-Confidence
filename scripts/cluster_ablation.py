"""Reproduce the cluster-aware selection ablation.

    python scripts/cluster_ablation.py --config configs/cluster_uniform.yaml

This is not part of the pipeline. The stage was built, measured and
removed: diversity is resolved upstream by corpus deduplication, at a
fraction of the complexity and with no quality penalty. The code is kept
so the two reported cluster rows can be reproduced.
"""

import argparse

import pandas as pd
import torch

from model.config import ExperimentConfig
from model.metrics import embed_sequences
from model.models import ProteinEncoder
from model.selection.cluster_ablation import (
    cluster_embeddings,
    cluster_stats,
    pool_concentration,
    retain_score_scaled,
    retain_uniform,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scored", default="scored_sequences.csv")
    ap.add_argument("--out", default="cluster_selected.csv")
    ap.add_argument("--subsample", type=int, default=10000)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    rule = cfg.selection.cluster_rule
    if rule == "none":
        raise SystemExit("this config does not use a cluster rule")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scored = pd.read_csv(args.scored).head(args.subsample).reset_index(drop=True)

    encoder = ProteinEncoder("35M", cfg.data.max_len, device).to(device)
    emb = embed_sequences(scored["sequence"].tolist(), encoder)

    labels, info = cluster_embeddings(emb)
    scored["cluster"] = labels
    stats = cluster_stats(scored)
    print("\n" + stats.head(10).to_string(index=False))

    if rule == "score_scaled":
        selected = retain_score_scaled(scored, stats)
    elif rule == "uniform":
        selected = retain_uniform(scored, stats)
    else:
        raise ValueError(f"unknown cluster rule {rule}")

    conc = pool_concentration(selected)
    print(
        f"\npool concentration | {conc['from_top_n']} of {conc['total']} "
        f"({conc['fraction']*100:.0f}%) from the top {conc['top_n']} clusters"
    )
    if conc["fraction"] > 0.7:
        print(
            "A pool this concentrated cannot differ much from global ranking, "
            "which is why score-scaled retention reproduces it."
        )

    selected.to_csv(args.out, index=False)
    print(f"saved {len(selected)} -> {args.out}")


if __name__ == "__main__":
    main()
