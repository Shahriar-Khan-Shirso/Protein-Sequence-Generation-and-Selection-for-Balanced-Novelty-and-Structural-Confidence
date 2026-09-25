"""SwissProt corpus.

UniProtKB/Swiss-Prot is the manually reviewed section of UniProt. It is
used here because it differs from the structural archive on the axis
that matters: its entries are curated and non-redundant at the record
level, and it spans a far wider range of protein families.

The corpus is clustered at 50% sequence identity and one sequence is
kept per cluster, which is what makes it the low-redundancy arm of the
comparison. The held-out set is drawn from clusters disjoint from
training, so unlike the structural corpora its novelty figure is a
clean measurement.
"""

import gzip
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

FASTA_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_sprot.fasta.gz"
)


def parse_fasta(path):
    """Read a FASTA file into (entry_id, sequence) pairs."""
    ids, seqs = [], []
    cur_id, cur = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur))
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line.strip())
    if cur_id is not None:
        ids.append(cur_id)
        seqs.append("".join(cur))
    return pd.DataFrame({"entry_id": ids, "sequence": seqs})


def ensure_fasta(fasta="uniprot_sprot.fasta", fasta_gz="uniprot_sprot.fasta.gz"):
    if os.path.exists(fasta):
        return fasta
    if not os.path.exists(fasta_gz):
        raise FileNotFoundError(
            f"neither {fasta} nor {fasta_gz} found. Download SwissProt from\n  {FASTA_URL}"
        )
    with gzip.open(fasta_gz, "rb") as fin, open(fasta, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return fasta


def cluster_mmseqs(
    frame,
    min_seq_id=0.5,
    coverage=0.8,
    tmp_dir="mmseqs_tmp",
    prefix="mmseqs_out",
):
    """Cluster with MMseqs2 and return one representative per cluster.

    Requires mmseqs on PATH. The identity threshold of 50% is what makes
    this corpus non-redundant in the sense the thesis uses.
    """
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "mmseqs not found on PATH; install MMseqs2 or use the cached "
            "partition shipped with the released corpora"
        )

    os.makedirs(tmp_dir, exist_ok=True)
    fasta_in = "mmseqs_input.fasta"
    with open(fasta_in, "w") as fh:
        for eid, seq in zip(frame.entry_id, frame.sequence):
            fh.write(f">{eid}\n{seq}\n")

    subprocess.run(
        [
            "mmseqs",
            "easy-cluster",
            fasta_in,
            prefix,
            tmp_dir,
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
        ],
        check=True,
    )

    tsv = f"{prefix}_cluster.tsv"
    clusters = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "member"])
    reps = clusters["rep"].unique()
    print(f"MMseqs2 | {len(frame):,} sequences -> {len(reps):,} clusters")
    return frame[frame.entry_id.isin(reps)].copy()


def load_swissprot(
    fasta="uniprot_sprot.fasta",
    cache_csv="swissprot_clustered.csv",
    prefilter=(40, 512),
    min_seq_id=0.5,
    coverage=0.8,
    test_n=5000,
    seed=42,
):
    """Build the clustered SwissProt corpus, reusing the cache if present.

    The cache is preferred whenever it exists: clustering is expensive
    and its output is the partition the released results were trained on.
    """
    if os.path.exists(cache_csv):
        print(f"reusing cached partition {cache_csv}")
        clustered = pd.read_csv(cache_csv)
    else:
        path = ensure_fasta(fasta)
        frame = parse_fasta(path)
        lengths = frame.sequence.str.len()
        frame = frame[lengths.between(*prefilter)].reset_index(drop=True)
        print(f"prefiltered to {prefilter[0]}-{prefilter[1]} aa: {len(frame):,}")
        clustered = cluster_mmseqs(frame, min_seq_id, coverage)
        clustered.to_csv(cache_csv, index=False)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(clustered))
    test_idx = perm[:test_n]
    train_idx = perm[test_n:]

    test_raw = clustered.iloc[test_idx].copy()
    train_raw = clustered.iloc[train_idx].copy()

    for frame in (train_raw, test_raw):
        frame["macromoleculeType_x"] = "Protein"
        frame["structureId"] = frame["entry_id"]
        frame["chainId"] = "A"
        frame["seq_len"] = frame["sequence"].str.len()

    print(
        f"train {len(train_raw):,} | test {len(test_raw):,} "
        f"(cluster-disjoint from train)"
    )
    return train_raw, test_raw
