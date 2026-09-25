"""Corpus preprocessing.

Sequences are filtered to a length band and to the 20 canonical residues.
Two points of policy matter for what the model learns:

Sequences outside the band are dropped rather than truncated. A chain cut
mid-domain exposes its hydrophobic core and cannot fold, so truncation
would train the model to produce unfoldable fragments.

Sequences containing non-canonical residues are dropped rather than
repaired. Those positions are not among the 20 classes the decoder
predicts, so they would become ignored targets that the diffusion loss
still trains on.
"""

import numpy as np
import pandas as pd

from ..constants import AA_STR

VALID_AA = set(AA_STR)


def clean_sequence(seq, min_len, max_len, length_mode="drop"):
    """Return the cleaned sequence, or None if it fails the policy."""
    if not isinstance(seq, str):
        return None
    seq = seq.upper().strip()
    if set(seq) - VALID_AA:
        return None
    if len(seq) < min_len:
        return None
    if len(seq) > max_len:
        if length_mode == "drop":
            return None
        seq = seq[:max_len]
    return seq


def length_policy_report(lengths, min_len, max_len, length_mode="drop"):
    """What the length policy costs, before it is applied."""
    lengths = np.asarray(lengths)
    n = len(lengths)
    n_short = int((lengths < min_len).sum())
    n_long = int((lengths > max_len).sum())
    kept = n - n_short - (n_long if length_mode == "drop" else 0)
    return {
        "total": n,
        "below_min": n_short,
        "above_max": n_long,
        "kept": kept,
        "kept_if_trim": n - n_short,
        "mode": length_mode,
    }


def filter_proteins(df, type_col="macromoleculeType_x"):
    if type_col not in df.columns:
        return df.copy()
    mask = df[type_col].str.upper().str.contains("PROTEIN", na=False)
    return df[mask].copy()


def preprocess_corpus(
    df,
    min_len=128,
    max_len=256,
    length_mode="drop",
    dedup_keys=("structureId", "chainId"),
    verbose=True,
):
    """Apply the full preprocessing path to one split.

    Returns the cleaned frame with a seq_len column added.
    """
    out = filter_proteins(df)

    keys = [k for k in dedup_keys if k in out.columns]
    if keys:
        out = out.drop_duplicates(subset=keys)

    if verbose:
        raw_lens = out["sequence"].dropna().astype(str).str.len()
        rep = length_policy_report(raw_lens, min_len, max_len, length_mode)
        print(
            f"length policy {min_len}-{max_len} aa (mode={length_mode}): "
            f"{rep['kept']:,} of {rep['total']:,} kept "
            f"({rep['below_min']:,} too short, {rep['above_max']:,} too long)"
        )

    out = out.copy()
    out["sequence"] = out["sequence"].apply(
        lambda s: clean_sequence(s, min_len, max_len, length_mode)
    )
    out = out.dropna(subset=["sequence"])
    out["seq_len"] = out["sequence"].str.len()

    if verbose:
        print(
            f"kept {len(out):,} sequences | "
            f"length p10={np.percentile(out['seq_len'], 10):.0f} "
            f"p50={np.percentile(out['seq_len'], 50):.0f} "
            f"p90={np.percentile(out['seq_len'], 90):.0f}"
        )

    if len(out) < 5000:
        print(
            "[warning] fewer than 5,000 sequences; a 22M-parameter denoiser "
            "will overfit at this scale"
        )

    return out.reset_index(drop=True)


def length_pool(df, min_len, max_len):
    """Empirical length distribution used to sample target lengths at
    generation time."""
    lens = df["seq_len"].to_numpy()
    return lens[(lens >= min_len) & (lens <= max_len)]
