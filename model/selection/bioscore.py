"""Biological plausibility score.

Three closed-form statistics of a sequence, combined into one value in
[0, 1]. None requires a GPU, so this term is effectively free and runs
over the whole candidate pool.

Its role is to remove obvious structural abnormalities — highly
repetitive strings, uniformly hydrophobic stretches, compositionally
collapsed sequences — before the more expensive likelihood term is
applied. It does not by itself indicate foldability.
"""

import math
import random
from collections import Counter

from ..constants import POLAR_SET

KMER_WEIGHT = 0.34
POLAR_WEIGHT = 0.33
ENTROPY_WEIGHT = 0.33


def build_kmer_freq(train_sequences, k=3, sample_size=10000, seed=0):
    """Reference k-mer frequencies from the training corpus."""
    rng = random.Random(seed)
    sample = rng.sample(train_sequences, min(sample_size, len(train_sequences)))
    counts = Counter()
    for s in sample:
        for i in range(len(s) - k + 1):
            counts[s[i : i + k]] += 1
    total = sum(counts.values()) + 1e-8
    return {km: c / total for km, c in counts.items()}


def kmer_score(seq, train_freq, k=3):
    """Agreement between the sequence's k-mer usage and the corpus's.

    A short window is used because local residue interactions are what
    drive secondary structure formation. Sequences using k-mers at
    frequencies far from natural usage score low.
    """
    seq_kmers = Counter(seq[i : i + k] for i in range(len(seq) - k + 1))
    total = sum(seq_kmers.values()) + 1e-8
    divergence = 0.0
    for km, cnt in seq_kmers.items():
        p_seq = cnt / total
        p_train = train_freq.get(km, 1e-6)
        divergence += p_seq * math.log(p_seq / p_train + 1e-8)
    return max(0.0, 1.0 - abs(divergence) / 5.0)


def polar_fraction_score(seq, lo=0.35, hi=0.65):
    """Balance of polar and non-polar residues.

    Soluble globular proteins sit near 40-60% polar. Too few and the
    sequence aggregates; too many and it cannot form a buried
    hydrophobic core.
    """
    frac = sum(1 for aa in seq if aa in POLAR_SET) / len(seq)
    if lo <= frac <= hi:
        return 1.0
    return max(0.0, 1.0 - abs(frac - 0.5) * 4)


def entropy_score(seq):
    """Shannon entropy of residue composition, normalised by log 20.

    Natural proteins occupy a middle ground: a low-entropy sequence such
    as a poly-alanine chain has no distinct fold, and a maximum-entropy
    one is compositionally disordered.
    """
    counts = Counter(seq)
    n = len(seq)
    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-8) for p in probs)
    return min(1.0, entropy / math.log(20))


def bio_score(seq, train_freq, weights=(KMER_WEIGHT, POLAR_WEIGHT, ENTROPY_WEIGHT)):
    wk, wp, we = weights
    return (
        wk * kmer_score(seq, train_freq)
        + wp * polar_fraction_score(seq)
        + we * entropy_score(seq)
    )


def score_all(sequences, train_sequences, weights=None, k=3, sample_size=10000):
    train_freq = build_kmer_freq(train_sequences, k=k, sample_size=sample_size)
    w = weights or (KMER_WEIGHT, POLAR_WEIGHT, ENTROPY_WEIGHT)
    return [bio_score(s, train_freq, w) for s in sequences]
