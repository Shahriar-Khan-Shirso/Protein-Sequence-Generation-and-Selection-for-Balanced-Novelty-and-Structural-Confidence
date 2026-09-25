"""Evaluate a folded set: confidence, diversity, novelty, FD-seq.

    python scripts/evaluate.py --config configs/swissprot_225ep.yaml --folded folded_selected.csv

All four are reported together on purpose. Any one of them can be
improved at the expense of the others, so a single headline number does
not describe the system.
"""

import argparse
import json

import pandas as pd
import torch

from model.config import ExperimentConfig
from model.folding import threshold_table
from model.metrics import diversity_report, fdseq, novelty_report
from model.models import ProteinEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folded", required=True)
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--skip-fdseq", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folded = pd.read_csv(args.folded)
    # Diversity and novelty are computed on sequences that fold above the
    # threshold: embeddings of unfolded sequences describe noise.
    kept = folded[folded["plddt"] > cfg.evaluation.plddt_threshold]
    seqs = kept["sequence"].tolist()
    print(
        f"{len(kept)} of {len(folded)} sequences above pLDDT "
        f"{cfg.evaluation.plddt_threshold}"
    )

    result = {
        "config": cfg.name,
        "n_folded": int(len(folded)),
        "n_above_threshold": int(len(kept)),
        "plddt_mean": float(folded["plddt"].mean()),
        "plddt_median": float(folded["plddt"].median()),
        "plddt_std": float(folded["plddt"].std()),
        "plddt_min": float(folded["plddt"].min()),
        "plddt_max": float(folded["plddt"].max()),
        "thresholds": threshold_table(folded["plddt"]).to_dict("records"),
    }

    result.update(diversity_report(seqs, cfg.evaluation.cd_thresholds, cfg.evaluation.coverage))

    train_seqs = pd.read_csv(cfg.data.train_csv)["sequence"].dropna().tolist()
    test_seqs = pd.read_csv(cfg.data.test_csv)["sequence"].dropna().tolist()
    result.update(novelty_report(seqs, train_seqs, test_seqs))

    if not args.skip_fdseq:
        encoder = ProteinEncoder("35M", cfg.data.max_len, device).to(device)
        result.update(fdseq(seqs, train_seqs[: len(seqs) * 2], encoder))

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\npLDDT mean {result['plddt_mean']:.2f} | "
          f"CD@0.5 {result.get('cd@0.5', float('nan')):.4f} | "
          f"novelty L1 {result.get('novelty_l1_mean', float('nan')):.4f}")
    if "fdseq_normalised" in result:
        print(f"FD-seq {result['fdseq_normalised']:.4f} (normalised scale); "
              f"raw {result['fdseq_raw']:.4f}")
        print("FD-seq measures coverage, not quality; read it beside CD@0.5.")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
