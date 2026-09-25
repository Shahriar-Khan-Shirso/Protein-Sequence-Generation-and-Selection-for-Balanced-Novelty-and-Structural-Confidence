"""Experiment configuration.

Every run reported in the thesis is one YAML file under configs/. The
fields below are the complete set of things that vary between them;
anything not listed here was held constant across all experiments.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .constants import resolve_encoder


@dataclass
class DataConfig:
    corpus: str = "pdb_curated"          # pdb_curated | pdb_whole | swissprot
    min_len: int = 128
    max_len: int = 256
    length_mode: str = "drop"            # drop | trim
    train_csv: str = ""
    test_csv: str = ""
    latent_cache: str = ""


@dataclass
class ModelConfig:
    encoder: str = "35M"
    denoiser_hidden: int = 512
    denoiser_layers: int = 6
    denoiser_heads: int = 8
    decoder_hidden: int = 480
    decoder_layers: int = 3
    decoder_heads: int = 8
    dropout_denoiser: float = 0.0
    dropout_decoder: float = 0.1

    @property
    def latent_dim(self):
        return resolve_encoder(self.encoder)[2]


@dataclass
class DiffusionConfig:
    schedule: str = "tan"                # tan | cosine
    steepness: float = 10.0              # d in the tangent schedule
    n_steps: int = 1000
    sample_steps: int = 250
    spacing: str = "signal"              # signal | index
    clamp_val: float = 3.0


@dataclass
class AuxLossConfig:
    """Auxiliary objectives. Both were measured and neither is enabled
    in the final design; see the thesis Chapter 5."""
    proxy_enabled: bool = False
    proxy_weight: float = 0.05
    proxy_target: float = 93.0
    proxy_warmup_epochs: int = 30
    proxy_predictor_path: str = "plddt_predictor.pt"

    composition_enabled: bool = False
    kmer_weight: float = 0.10
    polar_weight: float = 0.10
    entropy_weight: float = 0.10
    composition_warmup_epochs: int = 0
    strict_refs: bool = True


@dataclass
class TrainConfig:
    epochs: int = 180
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    ce_weight: float = 0.2
    self_cond_prob: float = 0.5
    grad_clip: float = 2.0
    warmup_steps: int = 500
    cawr_period_epochs: int = 15
    ema_decay: float = 0.9999
    seed: int = 42
    checkpoint_every: int = 5
    checkpoint_name: str = "checkpoint.pt"
    refine_decoder: bool = True
    refine_epochs: int = 30
    refine_lr: float = 5e-5
    refine_t_ceiling: int = 250


@dataclass
class GenerationConfig:
    n_candidates: int = 10000
    batch_size: int = 1024
    seed: int = 42
    temperature: float = 1.0


@dataclass
class SelectionConfig:
    enabled: bool = True
    top_k: int = 2000
    bio_weight: float = 0.3
    pll_weight: float = 0.7
    kmer_sub_weight: float = 0.34
    polar_sub_weight: float = 0.33
    entropy_sub_weight: float = 0.33
    pll_stride: int = 5
    # Cluster-aware retention was measured and removed; see
    # sdima.selection.cluster_ablation.
    cluster_rule: str = "none"           # none | score_scaled | uniform


@dataclass
class EvalConfig:
    plddt_threshold: float = 30.0
    cd_thresholds: tuple = (0.5, 0.95)
    coverage: float = 0.8
    fdseq_normalised: bool = True


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    description: str = ""
    checkpoint: str = ""                 # name of the released checkpoint
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    aux: AuxLossConfig = field(default_factory=AuxLossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path):
        raw = yaml.safe_load(Path(path).read_text())
        sections = {
            "data": DataConfig,
            "model": ModelConfig,
            "diffusion": DiffusionConfig,
            "aux": AuxLossConfig,
            "train": TrainConfig,
            "generation": GenerationConfig,
            "selection": SelectionConfig,
            "evaluation": EvalConfig,
        }
        kwargs = {k: raw[k] for k in ("name", "description", "checkpoint") if k in raw}
        for key, klass in sections.items():
            if key in raw and raw[key] is not None:
                kwargs[key] = klass(**raw[key])
        return cls(**kwargs)

    def to_yaml(self, path):
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    def summary(self):
        return (
            f"{self.name}: encoder {self.model.encoder} | "
            f"{self.diffusion.schedule} schedule | "
            f"{self.train.epochs} epochs | "
            f"selection {'on' if self.selection.enabled else 'off'}"
        )
