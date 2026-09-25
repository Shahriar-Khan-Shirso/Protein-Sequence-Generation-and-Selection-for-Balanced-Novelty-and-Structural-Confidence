---
license: mit
tags:
  - protein
  - protein-design
  - diffusion
  - latent-diffusion
  - biology
library_name: pytorch
base_model: facebook/esm2_t12_35M_UR50D
---

# Compute-Constrained Protein Sequence Diffusion

Checkpoints for **Protein Sequence Generation and Selection for Balanced
Novelty and Structural Confidence**, a B.Sc. thesis at Brac University.

Latent diffusion over frozen ESM-2 representations with a post-generation
selection stage. Every model here was trained, sampled and evaluated on a
single 24 GB consumer GPU.

**Code and full results:** [GitHub](https://github.com/Shahriar-Khan-Shirso/Protein-Sequence-Generation-and-Selection-for-Balanced-Novelty-and-Structural-Confidence)

## What is in this repository

Thirteen checkpoints, one per reported configuration. Each corresponds to
a row in the thesis results tables, so a published number can be traced
back to the weights that produced it.

| File | Configuration | pLDDT |
|:--|:--|--:|
| `esm2 35 M 180 epoch.pt` | baseline — curated PDB, 35M encoder, 180 ep | 72.89 |
| `esm2 8m checkpoint.pt` | ESM-2 8M encoder | 71.83 |
| `esm2 150M checkpoint.pt` | ESM-2 150M encoder | 73.45 |
| `probert checkpoint.pt` | ProtBERT encoder | 68.26 |
| `esm2 35M 270 epoch.pt` | 270 epochs | 78.61 |
| `esm2 35M 360 epoch.pt` | 360 epochs | 80.52 |
| `cos_checkpoint_epoch180.pt` | cosine noise schedule | 54.02 |
| `esm2 35M plddt loss.pt` | proxy foldability objective | 74.56 |
| `bio_checkpoint_epoch180.pt` | composition objectives | 40.69 |
| `whole pdb dataset epoch 180.pt` | whole PDB archive | 71.49 |
| `swissprot_checkpoint for 135M epoch.pt` | SwissProt, 135 ep | 66.67 |
| `swissprot_checkpoint_epoch180.pt` | SwissProt, 180 ep | 68.64 |
| `swissprot 106k 225 epoch checkpoint.pt` | SwissProt, 225 ep | 71.73 |

Reference pLDDT for real sequences under the same predictor: 78.87 for
the structural corpora, 83.9 for SwissProt.

**Start with `swissprot 106k 225 epoch checkpoint.pt`** unless you have a
reason not to. It reaches 71.73 pLDDT while keeping cluster diversity at
0.9920 and novelty at 0.5425 — the best joint position of the set. The
360-epoch structural checkpoint scores higher on confidence alone but at
0.1936 diversity.

## Model

| Component | Configuration | Trainable parameters |
|:--|:--|--:|
| Encoder | ESM-2 35M, frozen | — |
| Denoiser | 6 layers, width 512, 8 heads | 22.7M |
| Decoder | 3 layers, width 480, 8 heads | 8.68M |

The encoder is frozen and runs once over the corpus to build a latent
cache, so it never enters the training loop. The denoiser predicts the
clean latent rather than the noise, under a tangent noise schedule with
DDIM sampling spaced uniformly in signal level.

## Usage

```bash
pip install torch fair-esm numpy pandas pyyaml tqdm
hf download shirshokhan/cheap-protein-diffusion-model --local-dir checkpoints
```

Then, with the code from the GitHub repository:

```bash
python scripts/generate.py \
    --config configs/swissprot_225ep.yaml \
    --checkpoint "checkpoints/swissprot 106k 225 epoch checkpoint.pt"
```

Generation loads `denoiser_ema`, the averaged weights, not `denoiser`.
The normalisation statistics travel inside the checkpoint and must not be
recomputed — doing so shifts the latent space out from under a trained
model, which fails silently.

Filenames contain spaces; quote them in shell commands.

## Limitations

Every configuration was run once, which forces a claim band of roughly
two pLDDT points. All confidence figures come from a single structure
predictor whose biases may overlap with those of the language model
driving selection. The structural corpora use temporal splits, so
held-out novelty on them is indicative rather than a clean test of
generalisation.

## Intended use

A research artifact for studying compute-constrained generative
modelling. These models generate unconditionally: there is no functional,
family or structural conditioning of any kind, training used public
corpora, and no sequences were synthesised or experimentally
characterised. Adding conditioning on function or fold would change that
assessment materially.

## Citation

```bibtex
@thesis{shirso2026protein,
  title   = {Protein Sequence Generation and Selection for Balanced
             Novelty and Structural Confidence},
  author  = {Khan Shirso, Shahriar and Biswas Mugdha, Suprio and
             Anis Trishna, Anika and Siddiki Aishi, Sinka},
  school  = {Brac University},
  type    = {B.Sc. thesis},
  address = {Dhaka, Bangladesh},
  year    = {2026},
  month   = {February}
}
```

Built on [ESM-2 and ESMFold](https://github.com/facebookresearch/esm)
(Meta AI) and [ProtBERT](https://huggingface.co/Rostlab/prot_bert)
(Rostlab). The diffusion framework follows
[DiMA](https://arxiv.org/abs/2403.03726).
