"""Sequence diversity.

Cluster density is the fraction of a pool that survives as distinct
clusters under greedy single-linkage clustering at an identity
threshold. A value near 1 means almost every sequence is distinct; a low
value means the pool has collapsed onto a few repeated designs.

Two thresholds are reported. CD@0.5 is the informative one for
generated pools, since near-duplicates differing in a few positions are
still the same design. CD@0.95 catches only near-exact repetition and
saturates at 1 for most pools.

Diversity is computed on folded sequences above a confidence threshold.
Embeddings and identities of sequences that do not fold describe noise,
so including them inflates the measure.
"""

import numpy as np
from tqdm import tqdm


def cluster_density(sequences, sim_threshold=0.5, cov_threshold=0.80, show_progress=True):
    """Greedy single-linkage clustering at an identity threshold.

    Returns (density, n_clusters). Pairs whose lengths differ by more
    than the coverage threshold are not compared, so a short fragment is
    not absorbed into a long sequence it happens to prefix.
    """
    n = len(sequences)
    if n < 2:
        return (1.0 if n == 1 else 0.0), n

    assigned = np.zeros(n, dtype=bool)
    n_clusters = 0
    rng = range(n)
    if show_progress:
        rng = tqdm(rng, desc=f"CD@{sim_threshold}", leave=False)

    for i in rng:
        if assigned[i]:
            continue
        n_clusters += 1
        assigned[i] = True
        s1 = sequences[i]
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            s2 = sequences[j]
            max_len = max(len(s1), len(s2))
            min_len = min(len(s1), len(s2))
            if min_len < cov_threshold * max_len:
                continue
            matches = sum(a == b for a, b in zip(s1[:min_len], s2[:min_len]))
            if matches / min_len >= sim_threshold:
                assigned[j] = True

    return n_clusters / n, n_clusters


def diversity_report(sequences, thresholds=(0.5, 0.95), cov_threshold=0.80):
    out = {}
    for t in thresholds:
        density, n_cl = cluster_density(sequences, t, cov_threshold)
        out[f"cd@{t}"] = density
        out[f"n_clusters@{t}"] = n_cl
    out["n"] = len(sequences)
    return out
