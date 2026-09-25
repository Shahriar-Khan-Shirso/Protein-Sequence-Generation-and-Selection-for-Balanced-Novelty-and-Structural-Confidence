"""Novelty against a reference corpus.

For each generated sequence, one minus its maximum normalised similarity
to any sequence in the reference set. A value near 1 means the sequence
resembles nothing in the reference; near 0 means it is close to
something already known.

Two reference sets are used. L1 is against the training corpus and
answers whether the model is reproducing what it was shown. L2 is
against a held-out set and answers whether it is reproducing proteins it
never saw.

A caution on L2 for the structural corpora: their splits are temporal,
and a temporal split leaves homologues on both sides of the boundary, so
a sequence can resemble a held-out protein simply because a relative of
it was in training. This is why the held-out figure is not reported as a
headline number, and why cluster-disjoint splitting is recommended.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def max_similarity(query, reference, scorer=None):
    """Highest normalised similarity between one sequence and a reference set."""
    from rapidfuzz import fuzz, process

    scorer = scorer or fuzz.ratio
    match = process.extractOne(query, reference, scorer=scorer)
    return (match[1] / 100.0) if match else 0.0


def novelty(
    generated,
    reference,
    batch_size=64,
    show_progress=True,
    workers=-1,
):
    """Per-sequence novelty in [0, 1] against a reference corpus."""
    from rapidfuzz import fuzz, process

    ref = list(reference)
    out = np.zeros(len(generated), dtype=float)

    rng = range(0, len(generated), batch_size)
    if show_progress:
        rng = tqdm(rng, desc="novelty")

    for start in rng:
        chunk = generated[start : start + batch_size]
        sims = process.cdist(
            chunk, ref, scorer=fuzz.ratio, workers=workers
        )
        out[start : start + len(chunk)] = 1.0 - sims.max(axis=1) / 100.0

    return out


def novelty_report(generated, train_reference, test_reference=None, **kwargs):
    """Novelty against training (L1) and optionally held-out (L2)."""
    result = {}
    l1 = novelty(generated, train_reference, **kwargs)
    result["novelty_l1_mean"] = float(l1.mean())
    result["novelty_l1_median"] = float(np.median(l1))
    result["novel_frac_l1"] = float((l1 > 0.5).mean())

    if test_reference is not None:
        l2 = novelty(generated, test_reference, **kwargs)
        result["novelty_l2_mean"] = float(l2.mean())
        result["novelty_l2_median"] = float(np.median(l2))
        result["note"] = (
            "L2 uses a temporal split for the structural corpora; homologues "
            "appear on both sides of the boundary, so treat it as indicative"
        )

    result["n"] = len(generated)
    return result


def nearest_neighbour_table(generated, reference, top_n=10):
    """The closest reference match for the least novel sequences.

    Useful as a memorisation check: if the least novel outputs are near
    exact copies, the model is reproducing training data rather than
    generating.
    """
    from rapidfuzz import fuzz, process

    rows = []
    for seq in generated:
        match = process.extractOne(seq, reference, scorer=fuzz.ratio)
        rows.append(
            {
                "sequence": seq,
                "closest": match[0] if match else None,
                "identity": (match[1] / 100.0) if match else 0.0,
            }
        )
    df = pd.DataFrame(rows).sort_values("identity", ascending=False)
    return df.head(top_n).reset_index(drop=True)
