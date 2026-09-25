"""sDiMA: latent diffusion for protein sequence generation under a
single-GPU compute budget, with a post-generation selection stage.

Pipeline stages, in order:

    data        corpus construction, preprocessing, redundancy, latents
    models      frozen encoder, denoiser, decoder
    diffusion   forward process and DDIM sampler
    training    joint objective, auxiliary objectives, checkpointing
    generation  length-conditioned sampling
    selection   biological score, pseudo-log-likelihood, global top-K
    folding     ESMFold structure prediction
    metrics     confidence, diversity, novelty, distributional distance
"""

__version__ = "1.0.0"

from .config import ExperimentConfig

__all__ = ["ExperimentConfig", "__version__"]
