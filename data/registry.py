_DATASET_REGISTRY: dict[str, type] = {}
_FORMATTER_REGISTRY: dict[str, object] = {}
_VARIANT_REGISTRY: dict[str, dict[str, dict]] = {}


def register(env_family: str, dataset_cls, formatter_module, variants: dict[str, dict]):
    _DATASET_REGISTRY[env_family] = dataset_cls
    _FORMATTER_REGISTRY[env_family] = formatter_module
    _VARIANT_REGISTRY[env_family] = variants


def get_dataset(env_family: str):
    if env_family not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown env_family: {env_family}. Register it in data/registry.py.")
    return _DATASET_REGISTRY[env_family]


def get_formatter(env_family: str):
    if env_family not in _FORMATTER_REGISTRY:
        raise ValueError(f"Unknown env_family: {env_family}. Register it in data/registry.py.")
    return _FORMATTER_REGISTRY[env_family]


def get_variants(env_family: str) -> dict[str, dict]:
    if env_family not in _VARIANT_REGISTRY:
        raise ValueError(f"Unknown env_family: {env_family}. Register it in data/registry.py.")
    return _VARIANT_REGISTRY[env_family]


def get_variant(env_family: str, variant: str) -> dict:
    variants = get_variants(env_family)
    if variant not in variants:
        raise ValueError(
            f"Unknown variant {variant!r} for env_family={env_family!r}. "
            f"Available: {list(variants)}"
        )
    return variants[variant]


def resolve_variant_env_spec(env_family: str, variant: str) -> tuple[dict, str, dict]:
    meta = get_variant(env_family, variant)
    if "env_paras" in meta:
        env_kwargs = dict(meta["env_paras"])
        env_id = env_kwargs.pop("id")
    else:
        env_id = meta["env_id"]
        env_kwargs = dict(meta.get("env_kwargs") or {})
    return meta, env_id, env_kwargs


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
from data.pointmaze.variants import POINTMAZE_VARIANTS
from data.antmaze.dataset import AntMazeDataset
from data.antmaze import formatting as antmaze_formatting
from data.antmaze.variants import ANTMAZE_VARIANTS

register("pointmaze", PointMazeDataset, pointmaze_formatting, POINTMAZE_VARIANTS)
register("antmaze", AntMazeDataset, antmaze_formatting, ANTMAZE_VARIANTS)
