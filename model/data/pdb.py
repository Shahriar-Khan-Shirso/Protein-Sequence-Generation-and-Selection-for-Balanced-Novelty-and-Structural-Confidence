"""Structural corpora from the Protein Data Bank.

The PDB export arrives as two files: per-structure metadata and the
corresponding sequences, joined on structureId. An inner join keeps only
entries present in both.

Splits are temporal, by publication year. This is worth stating plainly
as a limitation rather than a design choice: a temporal split leaves
homologues on both sides of the boundary, so held-out evaluation on
these corpora is not a clean test of generalisation. Identity- or
cluster-disjoint splitting would fix it.

Two corpora are built from this source. The curated one restricts to a
narrow publication window and a 128-256 residue band. The whole archive
spans the full history at 40-512 residues and is roughly three times
larger, which serves as the negative control for the redundancy
argument: it adds data without reducing repetition.
"""

import pandas as pd


def merge_export(metadata_csv, sequence_csv, key="structureId", out_csv=None):
    """Join the metadata and sequence files on structureId."""
    meta = pd.read_csv(metadata_csv)
    seqs = pd.read_csv(sequence_csv)
    merged = meta.merge(seqs, on=key, how="inner")
    print(
        f"merged {len(meta):,} metadata rows and {len(seqs):,} sequence rows "
        f"-> {len(merged):,}"
    )
    if out_csv:
        merged.to_csv(out_csv, index=False)
    return merged


def load_pdb(
    merged_csv,
    train_years=(2000, 2015),
    test_years=(2016, 2018),
    year_col="publicationYear",
):
    """Split a merged PDB export into train and test by publication year."""
    merged = pd.read_csv(merged_csv)
    merged[year_col] = pd.to_numeric(merged[year_col], errors="coerce")
    merged = merged.drop_duplicates(subset=["sequence"])

    train_raw = merged[merged[year_col].between(*train_years)].copy()
    test_raw = merged[merged[year_col].between(*test_years)].copy()

    print(
        f"loaded {len(merged):,} unique sequences | "
        f"train {train_years[0]}-{train_years[1]}: {len(train_raw):,} | "
        f"test {test_years[0]}-{test_years[1]}: {len(test_raw):,}"
    )
    print(
        "[note] the split is temporal, so homologues appear on both sides of "
        "the boundary; held-out novelty on this corpus is indicative only"
    )
    return train_raw, test_raw


CORPUS_PRESETS = {
    "pdb_curated": {
        "train_years": (2000, 2015),
        "test_years": (2016, 2018),
        "min_len": 128,
        "max_len": 256,
        "description": "curated structural subset, narrow length band",
    },
    "pdb_whole": {
        "train_years": (1971, 2015),
        "test_years": (2016, 2018),
        "min_len": 40,
        "max_len": 512,
        "description": "whole structural archive, wide length band",
    },
}
