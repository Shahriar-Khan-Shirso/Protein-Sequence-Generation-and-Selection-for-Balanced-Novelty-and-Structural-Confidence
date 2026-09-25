"""Corpus redundancy measurement and deduplication.

Redundancy is the property this work identifies as governing generated
diversity. A protein database is an archive of what has been deposited,
not a sample of sequence space: well-studied folds appear many times,
and a model trained on such a corpus learns that compressed distribution
and reproduces it.

Rep is one minus the product, over several n-mer lengths, of the
fraction of n-mers that are unique. Taking several scales matters for
interpretation: redundancy concentrated at short n indicates shared
local motifs, while redundancy present at every length up to 64
indicates recurring whole domains and chains, which is a different
problem and needs a different fix.

The same measure is applied to generated pools, so corpus repetition and
output repetition are directly comparable.
"""

from collections import Counter

import numpy as np

DEFAULT_NS = (8, 16, 32, 64)


def rep_score(sequences, ns=DEFAULT_NS):
    """Repetition in [0, 1]. Lower is less repetitive."""
    out = 1.0
    for n in ns:
        total, uniq = 0, set()
        for s in sequences:
            for i in range(len(s) - n + 1):
                uniq.add(s[i : i + n])
                total += 1
        out *= len(uniq) / max(1, total)
    return 1.0 - out


def rep_breakdown(sequences, ns=DEFAULT_NS):
    """Rep together with the per-scale uniqueness ratio.

    The breakdown is what says whether repetition lives in short motifs
    or in whole repeated chains.
    """
    ratios = {}
    out = 1.0
    for n in ns:
        total, uniq = 0, set()
        for s in sequences:
            for i in range(len(s) - n + 1):
                uniq.add(s[i : i + n])
                total += 1
        r = len(uniq) / max(1, total)
        ratios[n] = r
        out *= r
    return 1.0 - out, ratios


def corpus_repetition(sequences, sample=2000, ns=DEFAULT_NS, mode="head", seed=42):
    """Rep over a fixed-size sample.

    The sample size must be equal across corpora being compared: Rep
    depends on how many sequences are pooled, so unequal samples are not
    comparable.
    """
    seqs = list(sequences)
    n_use = min(sample, len(seqs))
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(seqs))[:n_use]
        chosen = [seqs[i] for i in idx]
    else:
        chosen = seqs[:n_use]

    rep, ratios = rep_breakdown(chosen, ns)
    lengths = np.array([len(s) for s in chosen])
    return {
        "rep": rep,
        "uniqueness": ratios,
        "n_pool": len(seqs),
        "n_sampled": n_use,
        "mean_length": float(lengths.mean()),
    }


def aa_frequency(sequences, cap=30000):
    """Residue composition, counting non-canonical residues separately."""
    from ..constants import AA_STR

    counts = Counter()
    for s in sequences[:cap]:
        counts.update(s)
    total = sum(counts.values()) or 1
    freq = {a: counts.get(a, 0) / total for a in AA_STR}
    other = sum(v for k, v in counts.items() if k not in AA_STR) / total
    return freq, other


def minhash_dedup(
    sequences, k=5, n_hash=128, bands=16, seed=42, keep="first"
):
    """Bucket near-duplicate sequences by MinHash and keep one per bucket.

    This is an approximate method chosen because it runs in minutes on a
    single machine over a corpus of this size, where exhaustive pairwise
    identity would not.

    Note on reproducibility: MinHash over Python string hashes depends on
    hash randomisation, so PYTHONHASHSEED must be fixed for the partition
    to be reproducible. The released corpora are provided as a cached
    partition for exactly this reason.
    """
    import hashlib

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2 ** 31 - 1, size=n_hash)

    def kmer_hashes(seq):
        kmers = {seq[i : i + k] for i in range(len(seq) - k + 1)}
        if not kmers:
            return None
        digests = np.array(
            [int(hashlib.md5(km.encode()).hexdigest()[:8], 16) for km in kmers],
            dtype=np.int64,
        )
        return np.array([np.min((digests ^ int(s)) & 0x7FFFFFFF) for s in seeds])

    rows_per_band = max(1, n_hash // bands)
    buckets = {}
    keep_idx = []

    for i, seq in enumerate(sequences):
        sig = kmer_hashes(seq)
        if sig is None:
            continue
        found = False
        for b in range(bands):
            band = tuple(sig[b * rows_per_band : (b + 1) * rows_per_band].tolist())
            key = (b, band)
            if key in buckets:
                found = True
                break
        if not found:
            keep_idx.append(i)
            for b in range(bands):
                band = tuple(
                    sig[b * rows_per_band : (b + 1) * rows_per_band].tolist()
                )
                buckets[(b, band)] = i

    print(
        f"MinHash dedup | kept {len(keep_idx)} of {len(sequences)} "
        f"({len(keep_idx)/max(1,len(sequences))*100:.1f}%)"
    )
    return keep_idx


def compare_corpora(corpora, sample=2000, ns=DEFAULT_NS):
    """Rep for several corpora side by side.

    The sample size is held equal across corpora so the comparison is
    valid.
    """
    import pandas as pd

    rows = []
    for name, seqs in corpora.items():
        stats = corpus_repetition(seqs, sample=sample, ns=ns)
        rows.append(
            {
                "corpus": name,
                "n_train": stats["n_pool"],
                "n_sampled": stats["n_sampled"],
                "mean_len": round(stats["mean_length"], 1),
                "rep": round(stats["rep"], 4),
                **{
                    f"uniq@{n}": round(stats["uniqueness"][n], 4) for n in ns
                },
            }
        )
    return pd.DataFrame(rows)
