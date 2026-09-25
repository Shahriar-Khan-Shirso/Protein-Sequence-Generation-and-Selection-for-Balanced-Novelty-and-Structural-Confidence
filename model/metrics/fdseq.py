"""Frechet distance over sequence embeddings.

The Frechet distance between the Gaussian fits of two embedding sets,
computed on mask-aware mean-pooled encoder embeddings.

Two cautions, both of which cost time to discover and both of which
apply to any use of this metric.

It is a coverage statistic, not a quality one, and it moves opposite to
quality under selection. The lowest distances in this work belong to
unfiltered pools and to the worst-performing configurations; the highest
belongs to the best. Selection narrows a pool onto high-likelihood
modes, which raises the distance while raising quality. Reported alone
it is not interpretable; reported beside cluster diversity it usefully
separates matching a distribution from matching its dominant modes.

It is homogeneous of degree two in the embeddings, so normalised and raw
values differ by the squared mean embedding norm. Cross-study comparison
is invalid unless the scale is stated. Both are reported here for that
reason.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from tqdm import tqdm


@torch.no_grad()
def embed_sequences(
    sequences,
    encoder,
    batch_size=32,
    normalise=True,
    show_progress=True,
):
    """Mask-aware mean-pooled embeddings, optionally L2-normalised."""
    out = []
    rng = range(0, len(sequences), batch_size)
    if show_progress:
        rng = tqdm(rng, desc="embedding")

    for start in rng:
        chunk = sequences[start : start + batch_size]
        reps, mask = encoder.encode(chunk)
        mf = mask.unsqueeze(-1).float()
        pooled = (reps * mf).sum(1) / mf.sum(1).clamp(min=1)
        if normalise:
            pooled = F.normalize(pooled, dim=-1)
        out.append(pooled.cpu())

    return torch.cat(out).numpy()


def frechet_distance(emb1, emb2):
    """Frechet distance between the Gaussian fits of two embedding sets."""
    mu1, mu2 = emb1.mean(axis=0), emb2.mean(axis=0)
    cov1 = np.cov(emb1, rowvar=False)
    cov2 = np.cov(emb2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2 * covmean))


def fdseq(
    generated,
    reference,
    encoder,
    batch_size=32,
    report_both_scales=True,
):
    """FD-seq between a generated pool and a reference corpus.

    Returns both the normalised and raw-scale values, together with the
    mean embedding norm that converts between them. Publishing both is
    what makes the number comparable to other work.
    """
    gen_norm = embed_sequences(generated, encoder, batch_size, normalise=True)
    ref_norm = embed_sequences(reference, encoder, batch_size, normalise=True)
    fd_norm = frechet_distance(gen_norm, ref_norm)

    result = {"fdseq_normalised": fd_norm, "n_gen": len(generated), "n_ref": len(reference)}

    if report_both_scales:
        gen_raw = embed_sequences(
            generated, encoder, batch_size, normalise=False, show_progress=False
        )
        ref_raw = embed_sequences(
            reference, encoder, batch_size, normalise=False, show_progress=False
        )
        mean_norm = float(
            np.linalg.norm(np.concatenate([gen_raw, ref_raw]), axis=1).mean()
        )
        result["fdseq_raw"] = frechet_distance(gen_raw, ref_raw)
        result["mean_embedding_norm"] = mean_norm
        result["scale_factor"] = mean_norm ** 2

    return result
