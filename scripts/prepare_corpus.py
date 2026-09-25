"""Build a training corpus and report its redundancy.

    python scripts/prepare_corpus.py --config configs/baseline_pdb_curated_35m_180ep.yaml
"""

import argparse
from pathlib import Path

from sdima.config import ExperimentConfig
from sdima.data import (
    CORPUS_PRESETS,
    corpus_repetition,
    load_pdb,
    load_swissprot,
    preprocess_corpus,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--merged-csv", default="merged_protein_data.csv")
    ap.add_argument("--fasta", default="uniprot_sprot.fasta")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.data.corpus.startswith("pdb"):
        preset = CORPUS_PRESETS[cfg.data.corpus]
        train_raw, test_raw = load_pdb(
            args.merged_csv,
            train_years=preset["train_years"],
            test_years=preset["test_years"],
        )
    elif cfg.data.corpus == "swissprot":
        train_raw, test_raw = load_swissprot(fasta=args.fasta)
    else:
        raise ValueError(f"unknown corpus {cfg.data.corpus}")

    train = preprocess_corpus(
        train_raw, cfg.data.min_len, cfg.data.max_len, cfg.data.length_mode
    )
    test = preprocess_corpus(
        test_raw, cfg.data.min_len, cfg.data.max_len, cfg.data.length_mode
    )

    train.to_csv(cfg.data.train_csv or out / f"{cfg.data.corpus}_train.csv", index=False)
    test.to_csv(cfg.data.test_csv or out / f"{cfg.data.corpus}_test.csv", index=False)

    stats = corpus_repetition(train["sequence"].tolist())
    print(f"\nRep(train) = {stats['rep']:.4f} over {stats['n_sampled']} sequences")
    for n, r in stats["uniqueness"].items():
        print(f"  unique {n:>2}-mers: {r*100:7.3f}%")
    print(
        "\nRedundancy present at every scale indicates repeated whole domains "
        "rather than shared local motifs."
    )


if __name__ == "__main__":
    main()
