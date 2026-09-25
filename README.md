<div align="center">

# Protein Sequence Generation and Selection<br>for Balanced Novelty and Structural Confidence

Latent diffusion over frozen protein language model representations,
with a post-generation selection stage.<br>
Training, generation, folding and evaluation run on a single 24 GB consumer GPU.

<br>

[![Thesis](https://img.shields.io/badge/Thesis-PDF-8B0000?style=flat-square&logo=adobeacrobatreader&logoColor=white)](docs/thesis.pdf)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-Hugging%20Face-FFB000?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/shirshokhan/cheap-protein-diffusion-model)
[![License](https://img.shields.io/badge/License-MIT-4A5568?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

<br>

**Shahriar Khan Shirso** · **Suprio Biswas Mugdha** · **Anika Anis Trishna** · **Sinka Siddiki Aishi**

Department of Computer Science and Engineering, Brac University<br>
Supervised by Dr. Amitabha Chakrabarty · Co-supervised by Dr. Swakkhar Shatabda

</div>

---

## Overview

A generated protein sequence has to satisfy two conditions at once. It must fold
into a stable structure, and it must be new rather than a near copy of something
already in the databases. These pull against each other: the most direct route to
structural confidence — training longer, or steering towards high-likelihood
sequences — is also the route back towards the training data.

Systems that manage this trade-off well are typically trained on multi-GPU
clusters. This work asks how much of the balance survives on the hardware an
individual researcher actually owns, and which parts of such a system are worth
investing in. It reimplements and extends
[DiMA](https://arxiv.org/abs/2403.03726) rather than proposing a new generative
paradigm; the contribution is a characterised operating point under a fixed
compute budget, together with the full ablation record behind it.

<table>
<tr><td width="33%" valign="top">

**Selection carries the result**

Removing it costs **32.26** pLDDT on the curated structural corpus, **23.43** on
the whole archive and **19.56** on SwissProt — roughly four times what doubling
the training budget buys.

</td><td width="33%" valign="top">

**Encoder scale barely matters**

A nineteen-fold parameter range moves mean pLDDT by **1.62** points. The
bottleneck is the denoiser, which recovers about a third of the available latent
signal.

</td><td width="33%" valign="top">

**Redundancy governs diversity**

Reducing corpus repetition from **0.7466** to **0.0873** raises generated cluster
diversity from **0.3129** to **0.9975**. Adding redundant data changes neither.

</td></tr>
</table>

That last finding changes what training duration costs. On the redundant
structural corpus, buying +7.63 pLDDT costs 38% of diversity; on the deduplicated
corpus, buying +5.06 costs 0.8%.

---

## Results

Real-sequence reference under the same structure predictor: **78.87** pLDDT for
the structural corpora, **83.9** for SwissProt.

<details open>
<summary><b>Curated PDB</b> — 2000–2015, 128–256 aa, 18,349 sequences</summary>

<br>

| Configuration | pLDDT | FD-seq | Novelty (L1) | CD@0.5 |
|:--|--:|--:|--:|--:|
| **Baseline** — 35M, tangent, 180 ep | **72.89** | 0.2661 | 0.3027 | 0.3129 |
| ESM-2 8M encoder *(d = 320)* | 71.83 | 0.2502 | 0.3090 | 0.3089 |
| ESM-2 150M encoder *(d = 640)* | 73.45 | 0.1539 | 0.2598 | 0.2706 |
| ProtBERT encoder *(d = 1024)* | 68.26 | 0.1462 | 0.2880 | 0.2920 |
| 270 epochs | 78.61 | 0.3004 | 0.2119 | 0.2025 |
| 360 epochs | 80.52 | 0.2921 | 0.1895 | 0.1936 |
| Cosine noise schedule | 54.02 | 0.1335 | 0.5251 | 1.0000 |
| Proxy foldability objective | 74.56 | 0.3269 | 0.2830 | 0.2415 |
| Cluster selection — score-scaled | 71.34 | 0.2518 | 0.3106 | 0.3159 |
| Cluster selection — uniform 20% | 43.20 | 0.0704 | 0.6004 | 0.8201 |
| Composition objectives *(unselected)* | 40.69 | 0.0841 | 0.5918 | 0.8216 |
| No selection *(unfiltered 2,000)* | 40.63 | 0.0768 | 0.5969 | 0.8307 |

</details>

<details>
<summary><b>Whole PDB archive</b> — 1971–2015, 40–512 aa, 51,829 sequences</summary>

<br>

| Configuration | pLDDT | FD-seq | Novelty (L1) | CD@0.5 |
|:--|--:|--:|--:|--:|
| **180 epochs, selected** | **71.49** | 0.1442 | 0.4296 | 0.6578 |
| 180 epochs, unfiltered | 48.06 | 0.0413 | 0.6325 | 0.9287 |

</details>

<details>
<summary><b>SwissProt</b> — 50% identity clustered, 128–256 aa, 106,034 sequences</summary>

<br>

| Configuration | pLDDT | FD-seq | Novelty (L1) | CD@0.5 |
|:--|--:|--:|--:|--:|
| 135 epochs, selected | 66.67 | 0.0521 | 0.5906 | 1.0000 |
| 180 epochs, selected | 68.64 | 0.0600 | 0.5633 | 0.9975 |
| **225 epochs, selected** | **71.73** | 0.0650 | 0.5425 | 0.9920 |
| 180 epochs, unfiltered | 50.35 | 0.0361 | 0.6633 | 1.0000 |
| 225 epochs, unfiltered | 52.17 | 0.0290 | 0.6533 | 1.0000 |

</details>

> **On FD-seq.** It measures distributional *coverage*, not quality, and moves
> opposite to quality under selection — the lowest values here belong to
> unfiltered pools and the worst configurations. It is also homogeneous of degree
> two in its embeddings, so normalised and raw values differ by the squared mean
> embedding norm; cross-study comparison is invalid unless the scale is stated.
> Read it beside CD@0.5, never alone.

---

## Model

| Component | Configuration | Trainable parameters |
|:--|:--|--:|
| Encoder | ESM-2 35M, frozen | — |
| Denoiser | 6 layers, width 512, 8 heads | 22.7M |
| Decoder | 3 layers, width 480, 8 heads | 8.68M |
| | | **31.4M** |

The encoder is frozen and runs once over the corpus to build a latent cache, so it
never enters the training loop. The objective is

```
L    = L_MSE + 0.2 · L_CE
L_CE = 0.5 · CE(g(z₀), y) + 0.5 · CE_w(g(sg[ẑ₀]), y)
```

The stop-gradient keeps the two components separable: the denoiser is supervised
by reconstruction alone, the decoder by cross-entropy alone. Without it the
denoiser could reduce total loss by emitting latents that are easy to decode
rather than latents that are accurate. The second cross-entropy branch is weighted
by signal level, because the decoder is invoked once at the end of sampling and
residue identity predicted from a heavily noised latent is not recoverable in
principle.

---

## Installation

```bash
git clone https://github.com/Shahriar-Khan-Shirso/Protein-Sequence-Generation-and-Selection-for-Balanced-Novelty-and-Structural-Confidence.git
cd Protein-Sequence-Generation-and-Selection-for-Balanced-Novelty-and-Structural-Confidence
pip install -r requirements.txt
```

ESMFold needs roughly 16 GB of VRAM. MMseqs2 is required only to rebuild the
SwissProt corpus from scratch; the cached cluster partition makes that optional.

**Checkpoints** are hosted on Hugging Face:

```bash
hf download shirshokhan/cheap-protein-diffusion-model --local-dir checkpoints
```

See [`checkpoints/README.md`](checkpoints/README.md) for the mapping from each
configuration to its checkpoint file.

---

## Running the pipeline

```bash
# 1 · Build a corpus and measure its redundancy
python scripts/prepare_corpus.py  --config configs/swissprot_225ep.yaml

# 2 · Train — resumable, atomic checkpoints every 5 epochs
python scripts/train.py           --config configs/swissprot_225ep.yaml

# 3 · Generate candidates
python scripts/generate.py        --config configs/swissprot_225ep.yaml \
                                  --checkpoint swissprot_225ep.pt

# 4 · Score and select
python scripts/select.py          --config configs/swissprot_225ep.yaml

# 5 · Fold — resumable, the expensive stage
python scripts/fold.py            --input selected.csv --out folded_selected.csv

# 6 · Evaluate
python scripts/evaluate.py        --config configs/swissprot_225ep.yaml \
                                  --folded folded_selected.csv
```

Step 4 also writes a matched random sample of the same size. Folding both is what
makes the contribution of selection measurable: generation and selection are
separate stages, and the same trained model is evaluated on each.

---

## Repository layout

```
model/
├── config.py              experiment configuration
├── constants.py           alphabet, encoder registry, fixed scorer
├── data/
│   ├── pdb.py             structural corpora, temporal splits
│   ├── swissprot.py       FASTA parsing, MMseqs2 clustering
│   ├── preprocess.py      length banding, canonical residues
│   ├── redundancy.py      Rep(train), MinHash deduplication
│   └── latents.py         encoding, normalisation, caching
├── models/
│   ├── encoder.py         frozen PLM — ESM-2 or ProtBERT
│   ├── denoiser.py        6-layer transformer, predicts z₀
│   ├── decoder.py         3-layer transformer, 20 classes
│   └── ema.py
├── diffusion/schedule.py  tangent and cosine, DDIM sampler
├── training/
│   ├── trainer.py         joint loop
│   ├── objectives.py      masked MSE + signal-weighted CE
│   ├── aux_losses.py      composition and proxy objectives
│   ├── plddt_predictor.py frozen confidence head
│   ├── refine_decoder.py  denoiser-realistic refinement
│   └── checkpoint.py      atomic writes, resume assertions
├── generation/sample.py   length-conditioned sampling
├── selection/
│   ├── bioscore.py        k-mer, polar fraction, entropy
│   ├── pll.py             ESM-2 pseudo-log-likelihood
│   ├── composite.py       0.3 / 0.7 combination, global top-K
│   └── cluster_ablation.py   measured and removed — see below
├── folding/esmfold.py     resumable structure prediction
└── metrics/               pLDDT, CD@k, novelty, FD-seq

configs/     one YAML per reported configuration
scripts/     pipeline entry points
results/     reported tables as CSV
docs/        thesis PDF
```

---

## On the cluster ablation

[`model/selection/cluster_ablation.py`](model/selection/cluster_ablation.py) sits
deliberately outside the pipeline. Two retention rules were built and measured —
retention scaled by cluster quality, and a uniform 20% per cluster — and the stage
was then removed, because diversity is resolved upstream by corpus deduplication
at a fraction of the complexity and with no quality penalty. The code is kept so
the two cluster rows in the results table can be reproduced.

The ordinary runs also contain clustering steps, but with retention set to 0.99
for every cluster, which keeps essentially everything and is equivalent to global
ranking. Those are not reproduced here.

---

## Reported negative results

Both auxiliary objectives failed, and for different reasons.

The **proxy foldability objective** satisfied its hinge within a few epochs and
thereafter contributed a loss of approximately 0.003 for the rest of training.
Across the full run it bought 1.67 pLDDT points at a cost of 0.0714 in cluster
diversity — a worse exchange than simply training longer.

The **composition objectives** produced a null result against a matched control
(40.69 against 40.63), despite being designed around the two failure modes they
were meant to avoid: they score discrete sequences through a straight-through
estimator, and hinge against empirical quantiles of the real distribution so that
90%, 50% and 90% of real sequences respectively remain in violation. That the
remedies were applied and the objective still failed is the informative part — it
locates the constraint on foldability somewhere other than amino acid composition
and 3-mer statistics.

---

## Limitations

**Every configuration was run once.** This forces a claim band of roughly two
pLDDT points and leaves the 8M-versus-35M encoder comparison formally
inconclusive. Multi-seed replication is the most consequential missing check.

**The structural corpora use temporal splits**, which leave homologues on both
sides of the boundary. Held-out novelty on those corpora is indicative rather than
a clean generalisation test. The SwissProt split is cluster-disjoint and does not
have this problem.

**All confidence figures come from one structure predictor**, whose biases may
overlap with those of the language model driving selection. Evaluating the same
pools with an independent predictor is the single most valuable check available
without wet-lab work.

**The cross-system comparison is indicative, not controlled**, since baseline
figures are drawn from those authors' own evaluation corpora.

---

## Intended use

This is a research artifact for studying compute-constrained generative modelling.
The model generates unconditionally: it has no functional, family or structural
conditioning of any kind, it was trained on public corpora, and no sequences were
synthesised or experimentally characterised. Adding conditioning on function or
fold would change that assessment materially and should be reviewed on that basis
before rather than after it is built.

---

## Citation

```bibtex
@thesis{shirso2026protein,
  title  = {Protein Sequence Generation and Selection for Balanced
            Novelty and Structural Confidence},
  author = {Khan Shirso, Shahriar and Biswas Mugdha, Suprio and
            Anis Trishna, Anika and Siddiki Aishi, Sinka},
  school = {Brac University},
  type   = {B.Sc. thesis},
  address = {Dhaka, Bangladesh},
  year   = {2026},
  month  = {February}
}
```

---

## Acknowledgements

Built on [ESM-2 and ESMFold](https://github.com/facebookresearch/esm) (Meta AI)
and [ProtBERT](https://huggingface.co/Rostlab/prot_bert) (Rostlab). The diffusion
framework follows [DiMA](https://arxiv.org/abs/2403.03726). We thank our research
assistant Azwad Aziz for support with the experiments.

Released under the [MIT License](LICENSE).
