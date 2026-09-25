"""Fold a sequence set with ESMFold. Resumable.

    python scripts/fold.py --input selected.csv --out folded_selected.csv
"""

import argparse

import pandas as pd

from sdima.folding import fold_sequences, threshold_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="folded.csv")
    ap.add_argument("--pdb-dir", default=None)
    ap.add_argument("--save-pdbs", action="store_true")
    args = ap.parse_args()

    seqs = pd.read_csv(args.input)["sequence"].dropna().tolist()
    df = fold_sequences(
        seqs, out_csv=args.out, pdb_dir=args.pdb_dir, save_pdbs=args.save_pdbs
    )
    print("\n" + threshold_table(df["plddt"]).to_string(index=False))


if __name__ == "__main__":
    main()
