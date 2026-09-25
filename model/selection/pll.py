"""ESM-2 pseudo-log-likelihood.

Masks every stride-th position and reads how confidently the language
model recovers the true residue there. A sequence the model finds
unsurprising is one whose local patterns resemble natural proteins,
which correlates with structural confidence without running a structure
predictor.

This is the expensive half of the selection score, and it is still far
cheaper than folding: it replaces a full structure prediction per
candidate with a handful of masked forward passes.

All masked variants of a sequence are batched into one forward call.
Doing this position by position at batch size one is what makes a naive
implementation take hours rather than minutes.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def esm2_pll(
    sequences,
    model,
    alphabet,
    max_len=256,
    stride=5,
    pos_batch=64,
    device=None,
    show_progress=True,
):
    """Mean per-masked-position log-likelihood for each sequence."""
    device = device or next(model.parameters()).device
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    out = []
    iterator = tqdm(sequences, desc="ESM-2 PLL") if show_progress else sequences

    for seq in iterator:
        seq_t = seq[:max_len]
        _, _, tokens = batch_converter([("p", seq_t)])
        tokens = tokens.to(device)
        n_res = tokens.shape[1] - 2  # strip BOS/EOS

        positions = list(range(0, n_res, stride))
        total_logp = 0.0
        n_masked = 0

        for start in range(0, len(positions), pos_batch):
            chunk = positions[start : start + pos_batch]
            masked = tokens.repeat(len(chunk), 1)
            for row, pos in enumerate(chunk):
                masked[row, pos + 1] = alphabet.mask_idx  # +1 for BOS

            logits = model(masked, repr_layers=[], return_contacts=False)["logits"]
            for row, pos in enumerate(chunk):
                logp = F.log_softmax(logits[row, pos + 1, :], dim=-1)
                total_logp += logp[tokens[0, pos + 1].item()].item()
                n_masked += 1

        out.append(total_logp / max(1, n_masked))

    return out


def normalise_pll(raw_plls):
    """Min-max scale to [0, 1] across the pool.

    Normalisation is within a run, so the resulting value is a ranking
    quantity only and is not comparable across runs. Absolute thresholds
    on it are meaningless for the same reason.
    """
    arr = np.asarray(raw_plls, dtype=float)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)
