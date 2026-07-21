from .base import RegionConfig, TROPOMIConfig
from .us import USConfig
from .world import WorldConfig


def get_config(region: str) -> RegionConfig:
    region = region.lower()
    if region == "us":
        return USConfig()
    if region == "world":
        return WorldConfig()
    raise ValueError(f"Unknown region: {region!r}. Use 'us' or 'world'.")


# Training hyperparameters (re-exported from the legacy config.py since this
# package shadows the module path). 6_training/train_mlp.py imports
# `from config import TRAINING`.
TRAINING = {
    "hidden_dims":          [256, 128, 64, 32],
    "dropout":              0.3,
    "batch_size":           32,
    "num_epochs":           100,
    "patience":             5,
    "optimizer":            "adam",
    "loss":                 "BCEWithLogitsLoss",
    "prediction_threshold": 0.5,
    "learning_rates":       [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "num_runs":             5,
    "default_lr":           1e-3,
    "train_ratio":          0.6,
    "val_ratio":            0.2,
    "test_ratio":           0.2,
    "simple_test_size":     0.2,
    "split_seed_simple":    42,
    "split_seed_sweep":     345,
    "oversample_seed_base": 42,
    "qa_threshold":         0.75,
}


__all__ = ["RegionConfig", "USConfig", "WorldConfig", "TROPOMIConfig",
           "get_config", "TRAINING"]
