"""Constants shared across the pipeline.

The decoder emits exactly the 20 canonical amino acids, indexed by AA_STR.
The encoder's own 33-token vocabulary is never used for decoding: mapping
20-class indices through it silently substitutes residues.
"""

AA_STR = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA_STR)}
VOCAB_SIZE = len(AA_STR)

# Ignored by F.cross_entropy.
PAD_IDX = -100

# Residues classified as polar for the biological plausibility score.
POLAR_SET = set("DERKHNSQT")

def get_device():
    """Resolve the compute device.

    Imported lazily so that configuration and corpus statistics can be
    inspected without a torch installation.
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _LazyDevice:
    """Resolves to the real device on first use."""

    _resolved = None

    def _get(self):
        if _LazyDevice._resolved is None:
            _LazyDevice._resolved = get_device()
        return _LazyDevice._resolved

    def __repr__(self):
        return repr(self._get())

    def __str__(self):
        return str(self._get())

    def __getattr__(self, item):
        return getattr(self._get(), item)


DEVICE = _LazyDevice()

# Frozen encoders available as the representation model. Each entry is
# (model name, representation layer, latent width). ProtBERT has no
# intermediate layer selection, so its layer is None.
ESM_ENCODERS = {
    "8M": ("esm2_t6_8M_UR50D", 6, 320),
    "35M": ("esm2_t12_35M_UR50D", 12, 480),
    "150M": ("esm2_t30_150M_UR50D", 30, 640),
    "650M": ("esm2_t33_650M_UR50D", 33, 1280),
}

PROTBERT_ENCODERS = {
    "protbert": ("Rostlab/prot_bert", None, 1024),
    "protbert-bfd": ("Rostlab/prot_bert_bfd", None, 1024),
}

# The scorer is held fixed across every experiment so that the selection
# stage is identical when the representation model is ablated.
SCORER_NAME = "esm2_t12_35M_UR50D"
SCORER_LAYER = 12
SCORER_DIM = 480


def resolve_encoder(name):
    """Return (model_name, repr_layer, latent_dim) for an encoder key."""
    if name in ESM_ENCODERS:
        return ESM_ENCODERS[name]
    if name in PROTBERT_ENCODERS:
        return PROTBERT_ENCODERS[name]
    raise KeyError(
        f"unknown encoder '{name}'; expected one of "
        f"{sorted(ESM_ENCODERS) + sorted(PROTBERT_ENCODERS)}"
    )
