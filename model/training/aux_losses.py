"""Auxiliary training objectives.

Both objectives here were measured and neither is enabled in the final
design. They are kept because the reasons they failed are specific and
the design rules that follow from them are reusable.

Composition objectives attach to the decoder branch on a detached
latent, so they shape the readout only. The proxy foldability objective
deliberately does not detach, so its gradient reaches the denoiser.

Two rules govern both:

Score discrete sequences, not probability distributions. A statistic
computed under expectation lets a model reduce the loss by becoming less
certain rather than more correct; a maximally hedged decoder scores
better on 3-mer plausibility than real proteins do.

Hinge against a level measured on real data, not a theoretical optimum.
A k-mer divergence of zero means using k-mers more typically than real
proteins, which is repetition. Thresholds are set at quantiles of the
real distribution so a meaningful fraction of real sequences remain in
violation and the term keeps gradient.
"""

import math
from collections import Counter

import torch
import torch.nn.functional as F

from ..constants import AA_INDEX, POLAR_SET, VOCAB_SIZE


# ----------------------------------------------------------------------
# Composition statistics
# ----------------------------------------------------------------------

def build_kmer_table(sequences, k=3, sample=10000, device=None, seed=0):
    """Reference k-mer distribution estimated from the training corpus."""
    if k != 3:
        raise NotImplementedError("the contraction below is written for k=3")
    import random

    rng = random.Random(seed)
    subset = rng.sample(sequences, min(sample, len(sequences)))

    counts = Counter()
    for s in subset:
        for i in range(len(s) - 2):
            counts[s[i : i + 3]] += 1
    total = sum(counts.values()) + 1e-8

    bins = VOCAB_SIZE ** 3
    vec = torch.full((bins,), 1e-6, dtype=torch.float32)
    for km, n in counts.items():
        if all(c in AA_INDEX for c in km):
            idx = AA_INDEX[km[0]] * 400 + AA_INDEX[km[1]] * 20 + AA_INDEX[km[2]]
            vec[idx] = n / total
    return vec.to(device) if device else vec


def polar_mask(device=None):
    m = torch.zeros(VOCAB_SIZE)
    for aa in POLAR_SET:
        if aa in AA_INDEX:
            m[AA_INDEX[aa]] = 1.0
    return m.to(device) if device else m


def sequence_stats(p, valid, kmer_table, polar_m):
    """Per-sequence (kmer KL, polar fraction, normalised entropy).

    p is (B, L, 20) probabilities; valid is (B, L) as float.
    """
    B, L, _ = p.shape
    pv = p * valid.unsqueeze(-1)
    n = valid.sum(1, keepdim=True).clamp(min=1.0)

    q_aa = pv.sum(1) / n
    f_pol = (q_aa * polar_m).sum(1)
    ent = -(q_aa * (q_aa + 1e-8).log()).sum(1) / math.log(VOCAB_SIZE)

    w = valid[:, :-2] * valid[:, 1:-1] * valid[:, 2:]
    p1, p2, p3 = p[:, :-2], p[:, 1:-1], p[:, 2:]
    bij = torch.einsum("bli,blj->blij", p1, p2)
    q3 = torch.einsum("blij,blk,bl->bijk", bij, p3, w)
    q3 = q3.reshape(B, VOCAB_SIZE ** 3) / w.sum(1, keepdim=True).clamp(min=1.0)

    # This direction penalises using 3-mers the corpus does not use. The
    # reverse would require one sequence of a few hundred residues to
    # cover all 8,000 bins, which is unsatisfiable.
    kl = (q3 * ((q3 / kmer_table.unsqueeze(0)) + 1e-8).log()).sum(1)
    return kl, f_pol, ent


def straight_through(logits):
    """Forward pass reads the discrete sequence the decoder would emit;
    gradients flow through the softmax."""
    p = F.softmax(logits, dim=-1)
    hard = F.one_hot(p.argmax(-1), VOCAB_SIZE).to(p.dtype)
    return hard + p - p.detach()


def measure_reference_levels(
    sequences, kmer_table, polar_m, max_len, device, sample=2000, strict=True, seed=0
):
    """Thresholds measured on real sequences.

    With strict thresholds roughly 90%, 50% and 90% of real sequences
    remain in violation of the three terms respectively, so none of the
    hinges saturates early. A previous configuration used the means
    instead, and the decoder cleared all three within five epochs through
    reconstruction learning alone, leaving the terms reading zero for the
    rest of training.
    """
    import random

    rng = random.Random(seed)
    subset = rng.sample(sequences, min(sample, len(sequences)))

    kls, pols, ents = [], [], []
    with torch.no_grad():
        for i in range(0, len(subset), 64):
            chunk = subset[i : i + 64]
            p = torch.zeros(len(chunk), max_len, VOCAB_SIZE, device=device)
            v = torch.zeros(len(chunk), max_len, device=device)
            for b, s in enumerate(chunk):
                for j, aa in enumerate(s[:max_len]):
                    if aa in AA_INDEX:
                        p[b, j, AA_INDEX[aa]] = 1.0
                        v[b, j] = 1.0
            k, f, e = sequence_stats(p, v, kmer_table, polar_m)
            kls.append(k.cpu())
            pols.append(f.cpu())
            ents.append(e.cpu())

    kls = torch.cat(kls)
    pols = torch.cat(pols)
    ents = torch.cat(ents)

    if strict:
        levels = {
            "kmer_ref": float(kls.quantile(0.10)),
            "entropy_ref": float(ents.quantile(0.90)),
            "polar_lo": float(pols.quantile(0.25)),
            "polar_hi": float(pols.quantile(0.75)),
        }
    else:
        levels = {
            "kmer_ref": float(kls.mean()),
            "entropy_ref": float(ents.mean()),
            "polar_lo": 0.35,
            "polar_hi": 0.65,
        }

    levels["violation_rates"] = {
        "kmer": float((kls > levels["kmer_ref"]).float().mean()),
        "entropy": float((ents < levels["entropy_ref"]).float().mean()),
        "polar": float(
            ((pols < levels["polar_lo"]) | (pols > levels["polar_hi"]))
            .float()
            .mean()
        ),
    }
    return levels


class CompositionObjectives:
    """Three one-sided hinges on emitted-sequence composition."""

    def __init__(self, kmer_table, polar_m, levels):
        self.kmer_table = kmer_table
        self.polar_m = polar_m
        self.kmer_ref = levels["kmer_ref"]
        self.entropy_ref = levels["entropy_ref"]
        self.polar_lo = levels["polar_lo"]
        self.polar_hi = levels["polar_hi"]

    def __call__(self, logits, valid):
        p = straight_through(logits)
        kl, f_pol, ent = sequence_stats(p, valid, self.kmer_table, self.polar_m)

        l_kmer = F.relu(kl - self.kmer_ref).mean()
        l_polar = (
            F.relu(self.polar_lo - f_pol) + F.relu(f_pol - self.polar_hi)
        ).mean()
        l_ent = F.relu(self.entropy_ref - ent).mean()

        diag = {
            "kmer_kl": float(kl.mean().detach()),
            "polar": float(f_pol.mean().detach()),
            "entropy": float(ent.mean().detach()),
        }
        return l_kmer, l_polar, l_ent, diag


# ----------------------------------------------------------------------
# Proxy foldability
# ----------------------------------------------------------------------

def proxy_foldability_loss(predictor, x0_hat, mask, target=93.0):
    """One-sided hinge against a frozen confidence predictor.

    The gradient is deliberately not detached, so it reaches the
    denoiser. Reconstruction can only say "go to this target"; this term
    says "go where proteins fold well", which is a direction
    reconstruction cannot express.

    The risk is stated plainly: the predictor is a proxy, and a model
    optimised against a proxy may learn to satisfy it rather than the
    property it stands for. The weight is kept small and the objective is
    judged by folding the output, never by the predictor's own readings.
    """
    pred = predictor(x0_hat, mask)
    return F.relu(target / 100.0 - pred).mean(), float(pred.mean().detach()) * 100
