"""The base training objective.

Two terms, routed to disjoint parameter sets:

    L = L_MSE + lambda_CE * L_CE

L_MSE is masked mean squared error between the predicted and true clean
latent. It supervises the denoiser and nothing else.

L_CE trains the decoder on two inputs at once: the clean latent, which
gives a stationary learning problem, and the denoiser's own prediction,
which is what the decoder actually receives at generation time. The
second branch is detached, so no decoder gradient reaches the denoiser.
Without that detachment the denoiser could reduce total loss by emitting
latents that are easy to decode rather than latents that are accurate.

The second branch is also weighted by signal level. The decoder is
invoked once, at the end of sampling, on a latent that has already
passed through the low-noise regime; residue identity predicted from a
heavily noised latent is not recoverable in principle, and fitting it
would spend capacity on an unsolvable sub-problem.
"""

import torch
import torch.nn.functional as F

from ..constants import PAD_IDX, VOCAB_SIZE


def masked_latent_mse(x0_hat, x0, mask_f, latent_dim):
    """Per-element masked MSE.

    Normalising by (valid positions x latent dim) rather than by sequence
    count means the loss is an average over scalars, so batches are not
    weighted by their length composition.
    """
    n_elem = mask_f.sum().clamp(min=1) * latent_dim
    return ((x0_hat - x0) ** 2 * mask_f).sum() / n_elem


def decoder_cross_entropy(decoder, x0, x0_hat, mask, aa_targets, signal_weight):
    """Returns (ce_total, ce_clean, ce_hat, logits_clean, logits_hat)."""
    B, L = aa_targets.shape
    valid = (aa_targets != PAD_IDX).float()

    logits_clean = decoder(x0, mask=mask)
    ce_clean = F.cross_entropy(
        logits_clean.reshape(-1, VOCAB_SIZE),
        aa_targets.reshape(-1),
        ignore_index=PAD_IDX,
    )

    logits_hat = decoder(x0_hat.detach(), mask=mask)
    ce_pos = F.cross_entropy(
        logits_hat.reshape(-1, VOCAB_SIZE),
        aa_targets.reshape(-1),
        ignore_index=PAD_IDX,
        reduction="none",
    ).view(B, L)

    w = signal_weight.view(B, 1)
    ce_hat = (ce_pos * valid * w).sum() / (valid * w).sum().clamp(min=1e-8)

    return 0.5 * ce_clean + 0.5 * ce_hat, ce_clean, ce_hat, logits_clean, logits_hat


@torch.no_grad()
def decode_accuracy(logits, aa_targets):
    valid = (aa_targets != PAD_IDX).float()
    correct = (logits.argmax(-1) == aa_targets).float() * valid
    return (correct.sum() / valid.sum().clamp(min=1)).item()


def null_model_mse(latents, masks, latent_dim):
    """MSE of a model that emits zeros, for scale.

    Normalised latents have unit variance over real positions, so this
    is the reference against which a trained denoiser's MSE is read.
    """
    mask_f = masks.unsqueeze(-1).float()
    n_elem = mask_f.sum().clamp(min=1) * latent_dim
    return ((latents ** 2) * mask_f).sum() / n_elem
