from typing import Callable

_DATASET_REGISTRY: dict[str, type] = {}
_FORMATTER_REGISTRY: dict[str, object] = {}


def register(env_family: str, dataset_cls, formatter_module):
    _DATASET_REGISTRY[env_family] = dataset_cls
    _FORMATTER_REGISTRY[env_family] = formatter_module


def get_dataset(env_family: str):
    if env_family not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown env_family: {env_family}. Register it in data/registry.py.")
    return _DATASET_REGISTRY[env_family]


def get_formatter(env_family: str):
    if env_family not in _FORMATTER_REGISTRY:
        raise ValueError(f"Unknown env_family: {env_family}. Register it in data/registry.py.")
    return _FORMATTER_REGISTRY[env_family]


# ── Registration ──────────────────────────────────────────────────────────────
from data.pointmaze.dataset import PointMazeDataset
from data.pointmaze import formatting as pointmaze_formatting

register("pointmaze", PointMazeDataset, pointmaze_formatting)
