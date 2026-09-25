"""Structure prediction with ESMFold.

This is the most expensive stage in the pipeline, at roughly 1.5-2.7
seconds per sequence on one GPU, which is why selection exists at all.

Folding is resumable. A run over a few thousand sequences takes hours
and is interrupted by power loss and disk exhaustion often enough that
restarting from scratch is not viable; results are appended to a
checkpoint file and already-folded sequences are skipped on resume.

Failed predictions are omitted rather than recorded as zero. A zero is
not a measurement of a poor structure, it is the absence of one, and
averaging it in understates the mean.
"""

import gc
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_esmfold(device=None, chunk_size=128):
    """Load ESMFold. Needs roughly 16 GB of VRAM."""
    import esm

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    if chunk_size is not None:
        model.set_chunk_size(chunk_size)
    return model


def fold_one(model, sequence, pdb_path=None):
    """Fold a single sequence and return its mean pLDDT, or None on failure."""
    import biotite.structure.io as bsio

    with torch.no_grad():
        pdb_string = model.infer_pdb(sequence)

    if pdb_path is None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
            fh.write(pdb_string)
            path = fh.name
        struct = bsio.load_structure(path, extra_fields=["b_factor"])
        os.unlink(path)
    else:
        with open(pdb_path, "w") as fh:
            fh.write(pdb_string)
        struct = bsio.load_structure(pdb_path, extra_fields=["b_factor"])

    return float(struct.b_factor.mean())


def fold_sequences(
    sequences,
    model=None,
    out_csv="folded.csv",
    pdb_dir=None,
    save_pdbs=False,
    resume=True,
    flush_every=25,
    device=None,
):
    """Fold a list of sequences, resuming from out_csv if it exists.

    Returns a DataFrame of sequence and plddt. Sequences whose structure
    prediction failed are absent from the result rather than scored zero.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    done = {}
    if resume and os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = dict(zip(prev["sequence"], prev["plddt"]))
        print(f"resuming | {len(done)} sequences already folded")

    todo = [s for s in sequences if s not in done]
    if not todo:
        print("nothing to fold")
        return pd.DataFrame(
            {"sequence": list(done), "plddt": [done[s] for s in done]}
        )

    if model is None:
        model = load_esmfold(device=device)

    if save_pdbs and pdb_dir:
        os.makedirs(pdb_dir, exist_ok=True)

    n_failed = 0
    for i, seq in enumerate(tqdm(todo, desc="folding")):
        pdb_path = (
            os.path.join(pdb_dir, f"prediction_{len(done)}.pdb")
            if (save_pdbs and pdb_dir)
            else None
        )
        try:
            done[seq] = fold_one(model, seq, pdb_path)
        except Exception as exc:  # noqa: BLE001 - a failure here is data, not a bug
            n_failed += 1
            print(f"[failed] length {len(seq)}: {exc}")

        if (i + 1) % flush_every == 0:
            _flush(done, out_csv)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _flush(done, out_csv)
    df = pd.DataFrame({"sequence": list(done), "plddt": [done[s] for s in done]})
    print(
        f"folded {len(df)} sequences | mean pLDDT {df['plddt'].mean():.2f} | "
        f"{n_failed} failed"
    )
    return df


def _flush(done, out_csv):
    tmp = f"{out_csv}.tmp"
    pd.DataFrame({"sequence": list(done), "plddt": [done[s] for s in done]}).to_csv(
        tmp, index=False
    )
    os.replace(tmp, out_csv)


def threshold_table(plddt_values, thresholds=(30, 40, 50, 60, 70, 80)):
    """Counts above each confidence threshold.

    The mean alone hides the shape of the distribution: two pools with
    the same mean can differ entirely in how many sequences are
    confidently folded.
    """
    arr = np.asarray(plddt_values, dtype=float)
    n = len(arr)
    return pd.DataFrame(
        {
            "threshold": list(thresholds),
            "count": [int((arr > t).sum()) for t in thresholds],
            "fraction": [float((arr > t).mean()) for t in thresholds],
        }
    ).assign(total=n)
