"""feecbench: FEEC-native PDE benchmark prototype (2D acoustic pilot).

Public API (the golden path is ``FEECDataset`` -> ``collate_feec`` -> ``evaluate_profile``).
"""

from .dataset import FEECBatch, FEECDataset, collate_feec
from .evaluator import evaluate_profile, harmonic_basis, register_energy
from .io import load_sample, save_sample, write_index
from .normalization import Normalizer, compute_stats, load_stats, write_stats
from .oracle import DEFAULT_CONFIG, Oracle, ToyAcousticOracle, resolve_config
from .sampler import sample_configs
from .schema import FEECSample, Field, assert_samples_equal

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CONFIG",
    "FEECBatch",
    "FEECDataset",
    "FEECSample",
    "Field",
    "Normalizer",
    "Oracle",
    "ToyAcousticOracle",
    "assert_samples_equal",
    "collate_feec",
    "compute_stats",
    "evaluate_profile",
    "harmonic_basis",
    "load_sample",
    "load_stats",
    "register_energy",
    "resolve_config",
    "sample_configs",
    "save_sample",
    "write_index",
    "write_stats",
]
