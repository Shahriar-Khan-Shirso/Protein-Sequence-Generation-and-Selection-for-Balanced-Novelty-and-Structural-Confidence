"""Candidate generation.

Three things determine whether generation produces usable sequences, and
each was a source of error worth stating explicitly.

The target length is sampled first and used as the attention mask during
the reverse pass, so the sequence is denoised under the length it will
have. Generating at full padded width and truncating the decoded string
afterwards is not length conditioning: it produces a sequence whose
ending was never part of what the model was resolving.

The latent is decoded in normalised space. The decoder was trained on
normalised latents, so denormalising before decoding hands it inputs
from a distribution it has never seen.

Decoding indexes the 20-class alphabet directly. Passing 20-class
indices through the encoder's 33-token vocabulary substitutes a
different residue at every position.
"""

import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def sample_lengths(length_pool, n, rng=None):
    """Draw target lengths from the empirical training distribution."""
    rng = rng or random
    return np.array(rng.choices(list(length_pool), k=n), dtype=int)


def build_length_mask(lengths, max_len, device):
    mask = torch.zeros(len(lengths), max_len, dtype=torch.bool, device=device)
    for i, L in enumerate(lengths):
        mask[i, : int(L)] = True
    return mask


@torch.no_grad()
def generate(
    denoiser,
    decoder,
    diffusion,
    length_pool,
    n_sequences=10000,
    batch_size=512,
    max_len=256,
    latent_dim=480,
    n_sample_steps=250,
    temperature=1.0,
    seed=42,
    device=None,
):
    """Generate candidate sequences. Returns a DataFrame of sequence and length."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    denoiser.eval()
    decoder.eval()

    alloc = diffusion.step_allocation(n_sample_steps)
    print(
        f"sampling | {alloc['total']} steps, "
        f"{alloc['below_t_cut']} of them below t={alloc['t_cut']}"
    )

    seqs, lens = [], []
    for start in tqdm(range(0, n_sequences, batch_size), desc="sampling"):
        b = min(batch_size, n_sequences - start)
        batch_lens = sample_lengths(length_pool, b, rng)
        mask = build_length_mask(batch_lens, max_len, device)

        z0 = diffusion.sample(
            denoiser,
            (b, max_len, latent_dim),
            mask,
            n_sample_steps=n_sample_steps,
            temperature=temperature,
        )
        seqs.extend(decoder.decode_to_strings(z0, mask, batch_lens))
        lens.extend(batch_lens.tolist())

    df = pd.DataFrame({"sequence": seqs, "length": lens})
    print(f"generated {len(df)} sequences | {df['length'].min()}-{df['length'].max()} aa")
    return df


def validity_report(df, train_sequences=None, max_len=256):
    """Checks that catch decoding faults before anything expensive runs.

    A residue composition far from natural, or lengths that do not match
    what was requested, indicates a broken decode path rather than a
    poorly trained model.
    """
    from collections import Counter

    from ..constants import AA_STR

    seqs = df["sequence"].tolist()
    joined = "".join(seqs)
    counts = Counter(joined)
    total = max(1, len(joined))

    report = {
        "n": len(seqs),
        "n_empty": sum(1 for s in seqs if not s),
        "length_matches": bool((df["sequence"].str.len() == df["length"]).all()),
        "non_canonical": sorted(set(joined) - set(AA_STR)),
        "duplicate_frac": 1 - len(set(seqs)) / max(1, len(seqs)),
        "composition": {aa: counts.get(aa, 0) / total for aa in AA_STR},
    }

    if train_sequences:
        train_joined = "".join(train_sequences[:5000])
        tcounts = Counter(train_joined)
        ttotal = max(1, len(train_joined))
        report["train_composition"] = {
            aa: tcounts.get(aa, 0) / ttotal for aa in AA_STR
        }

    return report
