"""Latent cache and dataset.

The encoder is frozen, so its output for a given corpus never changes.
Encoding once and caching the result removes the encoder from the
training loop entirely: training reads tensors, not sequences.

Latents are normalised per dimension over real residue positions only.
Padded positions carry no signal and would otherwise drag the statistics.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..constants import AA_INDEX, PAD_IDX


def encode_corpus(encoder, sequences, max_len, batch_size=64, dtype=torch.float32):
    """Run the frozen encoder over a corpus and return (latents, masks)."""
    encoder.eval()
    lat_chunks, mask_chunks = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="encoding"):
            reps, mask = encoder.encode(sequences[i : i + batch_size])
            lat_chunks.append(reps.to(dtype).cpu())
            mask_chunks.append(mask.cpu())
    return torch.cat(lat_chunks), torch.cat(mask_chunks)


def latent_stats(latents, masks, eps=0.01):
    """Per-dimension mean and standard deviation over real positions."""
    real = latents[masks]
    return real.mean(0), real.std(0).clamp(min=eps)


def normalise(latents, masks, mean, std):
    out = latents.clone()
    out[masks] = (latents[masks] - mean.cpu()) / std.cpu()
    return out


def denormalise(latents, mean, std):
    return latents * std + mean


def residue_targets(sequences, max_len):
    """Residue indices in the decoder's 20-class space, padded with PAD_IDX."""
    targets = torch.full((len(sequences), max_len), PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_len]):
            if aa in AA_INDEX:
                targets[i, j] = AA_INDEX[aa]
    return targets


class SeqLatentDataset(Dataset):
    def __init__(self, latents, masks, aa_targets):
        self.latents = latents
        self.masks = masks
        self.aa_targets = aa_targets

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, i):
        return self.latents[i], self.masks[i], self.aa_targets[i]


class MemmapLatents:
    """Memory-mapped latent cache.

    Used for the larger corpora, where holding the full tensor in RAM is
    not possible within the machine's budget.
    """

    def __init__(self, path, shape, dtype=np.float16, mode="r"):
        self.path = path
        self.shape = shape
        self.array = np.memmap(path, dtype=dtype, mode=mode, shape=shape)

    @classmethod
    def create(cls, path, n, max_len, latent_dim, dtype=np.float16):
        return cls(path, (n, max_len, latent_dim), dtype=dtype, mode="w+")

    def write(self, start, tensor):
        self.array[start : start + tensor.shape[0]] = tensor.numpy()

    def flush(self):
        self.array.flush()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.asarray(self.array[idx]))
