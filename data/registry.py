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


def get_action_dim(env_family: str, variants: list[str]) -> int:
    dataset_cls = get_dataset(env_family)
    if not hasattr(dataset_cls, "get_action_dim"):
        raise ValueError(f"Dataset for env_family={env_family!r} does not expose get_action_dim().")
    action_dim = int(dataset_cls.get_action_dim(list(variants)))
    if action_dim < 1:
        raise ValueError(f"Invalid action_dim={action_dim} for env_family={env_family!r}.")
    return action_dim


# ── Registration ──────────────────────────────────────────────────────────────
from data.pointmaze.dataset import PointMazeDataset
from data.pointmaze import formatting as pointmaze_formatting

register("pointmaze", PointMazeDataset, pointmaze_formatting)
