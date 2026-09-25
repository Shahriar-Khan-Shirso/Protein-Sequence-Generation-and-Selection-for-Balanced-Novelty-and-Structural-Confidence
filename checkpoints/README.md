# Checkpoints

Hosted on Hugging Face rather than tracked in git — each is roughly 470 MB.

**[huggingface.co/shirshokhan/cheap-protein-diffusion-model](https://huggingface.co/shirshokhan/cheap-protein-diffusion-model)**

```bash
# everything (6.09 GB)
hf download shirshokhan/cheap-protein-diffusion-model --local-dir checkpoints

# or a single checkpoint
hf download shirshokhan/cheap-protein-diffusion-model \
    "swissprot 106k 225 epoch checkpoint.pt" --local-dir checkpoints
```

Place the files in this directory, or pass an explicit path to
`scripts/generate.py --checkpoint`.

## Mapping

Every row corresponds to one config in `configs/` and one row in the
results tables in the main README, so a reported number can be traced to
the weights that produced it.

| Config | File | pLDDT |
|:--|:--|--:|
| `baseline_pdb_curated_35m_180ep` | `esm2 35 M 180 epoch.pt` | 72.89 |
| `encoder_esm2_8m` | `esm2 8m checkpoint.pt` | 71.83 |
| `encoder_esm2_150m` | `esm2 150M checkpoint.pt` | 73.45 |
| `encoder_protbert` | `probert checkpoint.pt` | 68.26 |
| `duration_270ep` | `esm2 35M 270 epoch.pt` | 78.61 |
| `duration_360ep` | `esm2 35M 360 epoch.pt` | 80.52 |
| `schedule_cosine` | `cos_checkpoint_epoch180.pt` | 54.02 |
| `aux_proxy_foldability` | `esm2 35M plddt loss.pt` | 74.56 |
| `aux_composition` | `bio_checkpoint_epoch180.pt` | 40.69 |
| `pdb_whole_180ep` | `whole pdb dataset epoch 180.pt` | 71.49 |
| `swissprot_135ep` | `swissprot_checkpoint for 135M epoch.pt` | 66.67 |
| `swissprot_180ep` | `swissprot_checkpoint_epoch180.pt` | 68.64 |
| `swissprot_225ep` | `swissprot 106k 225 epoch checkpoint.pt` | 71.73 |

The two cluster ablation rows reuse the baseline checkpoint; they differ
only in the selection rule applied after generation.

Filenames carry spaces, so quote them in shell commands.

## What a checkpoint contains

```python
{
    "denoiser":         trained denoiser weights,
    "denoiser_ema":     averaged weights — this is what generation uses,
    "decoder":          decoder weights,
    "optimizer":        optimiser state, for resuming,
    "scheduler":        LR scheduler state,
    "latent_mean":      per-dimension normalisation mean,
    "latent_std":       per-dimension normalisation standard deviation,
    "parameterisation": "z0",
    "encoder":          encoder key, e.g. "35M",
    "l_max":            maximum sequence length,
    "schedule":         "tan" or "cosine",
    "epoch_num":        epoch at which it was written,
}
```

Generation loads `denoiser_ema`, not `denoiser`. Under cosine annealing
with warm restarts, the instantaneous weights at the end of a cycle are
not necessarily the best point on the trajectory.

The normalisation statistics travel with the checkpoint deliberately.
Recomputing them on a slightly different subset would shift the latent
space out from under a trained denoiser, which fails silently rather
than loudly.

`scripts/train.py` asserts on resume that the checkpoint's
parameterisation, sequence length, encoder and schedule match the
current config, because a mismatch otherwise trains without error and
generates nonsense.

## Note on the length band

The whole-PDB checkpoint was trained at 40–512 residues; every other
checkpoint uses 128–256. The corresponding config carries the right
values, so use the config that matches the checkpoint.
