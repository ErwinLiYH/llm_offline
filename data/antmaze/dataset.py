from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import h5py
import minari
from minari import MinariDataset
import numpy as np

from data.antmaze.variants import (
    ANTMAZE_VARIANTS,
    get_antmaze_variant_type,
    resolve_local_dataset_path,
)
from data.pointmaze.dataset import PointMazeDataset


DEFAULT_ANTMAZE_DATA_CONFIG = {
    "filter_success": False,
    "truncate": False,
    "truncate_holding": 0,
}


def _normalize_antmaze_data_config(config) -> dict:
    if config is None:
        raw_config = {}
    elif isinstance(config, Mapping):
        raw_config = dict(config)
    else:
        raise ValueError(
            "antmaze_data_config must be a mapping with keys "
            "filter_success, truncate, and truncate_holding."
        )

    unknown_keys = sorted(set(raw_config) - set(DEFAULT_ANTMAZE_DATA_CONFIG))
    if unknown_keys:
        raise ValueError(f"Unknown antmaze_data_config keys: {unknown_keys}")

    normalized = dict(DEFAULT_ANTMAZE_DATA_CONFIG)
    normalized.update(raw_config)

    for key in ("filter_success", "truncate"):
        if not isinstance(normalized[key], bool):
            raise ValueError(f"antmaze_data_config.{key} must be a bool, got {normalized[key]!r}")

    truncate_holding = normalized["truncate_holding"]
    if isinstance(truncate_holding, bool) or not isinstance(truncate_holding, int):
        raise ValueError(
            "antmaze_data_config.truncate_holding must be an integer >= 0, "
            f"got {truncate_holding!r}"
        )
    if truncate_holding < 0:
        raise ValueError(
            "antmaze_data_config.truncate_holding must be >= 0, "
            f"got {truncate_holding}"
        )
    normalized["truncate_holding"] = int(truncate_holding)
    return normalized


def _read_hdf5_tree(node):
    if isinstance(node, h5py.Dataset):
        return node[()]
    return {name: _read_hdf5_tree(node[name]) for name in node.keys()}


def _load_local_antmaze_hdf5_episodes(data_path):
    h5_path = data_path / "main_data.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Local AntMaze data file not found at {h5_path}. "
            "Generate it with local_antmaze_gen.py first."
        )
    episodes = []
    with h5py.File(h5_path, "r") as f:
        episode_names = sorted(
            (name for name in f.keys() if name.startswith("episode_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        for name in episode_names:
            group = f[name]
            kwargs = {
                "observations": _read_hdf5_tree(group["observations"]),
                "actions": group["actions"][()],
            }
            for field in ("rewards", "terminations", "truncations", "infos"):
                if field in group:
                    kwargs[field] = _read_hdf5_tree(group[field])
            episodes.append(SimpleNamespace(**kwargs))
    if not episodes:
        raise ValueError(f"No episodes found in local AntMaze data file {h5_path}")
    return episodes


def _get_episode_field(episode, name: str, default=None):
    if isinstance(episode, Mapping):
        return episode.get(name, default)
    return getattr(episode, name, default)


def _get_mapping_value(value, key: str, default=None):
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _success_values_from_infos(infos):
    if infos is None:
        return None
    success = _get_mapping_value(infos, "success")
    if success is not None:
        return success
    if isinstance(infos, (list, tuple)):
        values = []
        found = False
        for item in infos:
            item_success = _get_mapping_value(item, "success")
            if item_success is not None:
                found = True
            values.append(bool(item_success) if item_success is not None else False)
        if found:
            return values
    return None


def _transition_mask_from_values(values, action_len: int) -> np.ndarray:
    mask = np.zeros(action_len, dtype=bool)
    if values is None or action_len < 1:
        return mask
    array = np.asarray(values)
    if array.ndim == 0:
        return np.full(action_len, bool(array), dtype=bool)
    flat = array.reshape(-1)
    if flat.shape[0] >= action_len + 1:
        aligned = flat[1 : action_len + 1]
    elif flat.shape[0] >= action_len:
        aligned = flat[:action_len]
    else:
        aligned = flat
    mask[: aligned.shape[0]] = aligned.astype(bool)
    return mask


def _raw_success_any(episode) -> bool:
    values = _success_values_from_infos(_get_episode_field(episode, "infos"))
    if values is None:
        return False
    return bool(np.asarray(values).astype(bool).any())


def _success_transition_mask(episode, action_len: int) -> np.ndarray:
    values = _success_values_from_infos(_get_episode_field(episode, "infos"))
    return _transition_mask_from_values(values, action_len)


def _aligned_post_action_observations(observation_values, action_len: int) -> np.ndarray | None:
    if observation_values is None or action_len < 1:
        return None
    array = np.asarray(observation_values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim < 2:
        return None
    if array.shape[0] >= action_len + 1:
        states = array[1 : action_len + 1]
    elif array.shape[0] >= action_len:
        states = array[:action_len]
    else:
        states = array
    if states.shape[0] < action_len:
        padded = np.zeros((action_len, states.shape[1]), dtype=states.dtype)
        padded[: states.shape[0]] = states
        states = padded
    return states


def _fall_transition_mask(episode, action_len: int) -> np.ndarray:
    observations = _get_episode_field(episode, "observations", {})
    observation_values = _get_mapping_value(observations, "observation")
    states = _aligned_post_action_observations(observation_values, action_len)
    fall = np.zeros(action_len, dtype=bool)
    if states is None or states.shape[1] < 1:
        return fall

    z = states[:, 0].astype(float)
    finite_z = np.isfinite(z)
    if states.shape[1] < 5:
        return finite_z & (z < 0.30)

    quat = states[:, 1:5].astype(float)
    quat_norm = np.linalg.norm(quat, axis=1)
    valid_quat = np.isfinite(quat).all(axis=1) & np.isfinite(quat_norm) & (quat_norm > 1e-8)
    fallback = finite_z & (z < 0.30)
    fall[~valid_quat] = fallback[~valid_quat]
    if valid_quat.any():
        quat_unit = quat[valid_quat] / quat_norm[valid_quat, None]
        qx = quat_unit[:, 1]
        qy = quat_unit[:, 2]
        body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        fall[valid_quat] = finite_z[valid_quat] & (z[valid_quat] < 0.35) & (body_up_z < 0.0)
    return fall


def _slice_aligned_value(value, old_action_len: int, new_action_len: int):
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {
            key: _slice_aligned_value(item, old_action_len, new_action_len)
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.copy()
        if value.shape[0] == old_action_len + 1:
            return value[: new_action_len + 1].copy()
        if value.shape[0] == old_action_len:
            return value[:new_action_len].copy()
        return value.copy()
    if isinstance(value, list):
        if len(value) == old_action_len + 1:
            return list(value[: new_action_len + 1])
        if len(value) == old_action_len:
            return list(value[:new_action_len])
        return list(value)
    if isinstance(value, tuple):
        if len(value) == old_action_len + 1:
            return tuple(value[: new_action_len + 1])
        if len(value) == old_action_len:
            return tuple(value[:new_action_len])
        return tuple(value)
    return value


def _set_last_truncation_true(truncations):
    if truncations is None:
        return None
    if isinstance(truncations, np.ndarray):
        result = truncations.copy()
        if result.ndim > 0 and result.shape[0] > 0:
            result[-1] = True
        return result
    if isinstance(truncations, list):
        result = list(truncations)
        if result:
            result[-1] = True
        return result
    if isinstance(truncations, tuple):
        result = list(truncations)
        if result:
            result[-1] = True
        return tuple(result)
    return truncations


def _slice_episode(episode, action_end: int, *, mark_truncated: bool):
    actions = _get_episode_field(episode, "actions")
    old_action_len = len(actions)
    kwargs = {
        "observations": _slice_aligned_value(
            _get_episode_field(episode, "observations"),
            old_action_len,
            action_end,
        ),
        "actions": np.asarray(actions)[:action_end].copy(),
    }
    for field in ("rewards", "terminations", "truncations", "infos"):
        value = _get_episode_field(episode, field)
        if value is None:
            continue
        kwargs[field] = _slice_aligned_value(value, old_action_len, action_end)
    if mark_truncated and "truncations" in kwargs:
        kwargs["truncations"] = _set_last_truncation_true(kwargs["truncations"])
    return SimpleNamespace(**kwargs)


def _truncate_episode(episode, config: dict):
    actions = _get_episode_field(episode, "actions")
    action_len = len(actions)
    if action_len < 1:
        return episode
    success_mask = _success_transition_mask(episode, action_len)
    fall_mask = _fall_transition_mask(episode, action_len)
    event_indices = np.flatnonzero(success_mask | fall_mask)
    if event_indices.size == 0:
        return episode
    action_end = min(
        action_len,
        int(event_indices[0]) + 1 + int(config["truncate_holding"]),
    )
    if action_end >= action_len:
        return episode
    return _slice_episode(episode, action_end, mark_truncated=True)


def _apply_antmaze_data_config(episodes: list, config: dict, *, variant: str) -> list:
    processed = list(episodes)
    if config["filter_success"]:
        processed = [episode for episode in processed if _raw_success_any(episode)]

    if config["truncate"]:
        processed = [_truncate_episode(episode, config) for episode in processed]

    if config["filter_success"]:
        processed = [episode for episode in processed if _raw_success_any(episode)]
        if not processed:
            raise ValueError(
                "AntMaze data preprocessing removed all episodes for "
                f"variant={variant!r}. Check the dataset success rate, disable "
                "antmaze_data_config.filter_success, or regenerate the data."
            )

    return processed


def _local_antmaze_dataset_step_signature(
    meta: dict,
    local_dataset_root: str | None = None,
) -> str:
    dataset_root = resolve_local_dataset_path(
        meta["dataset_path"],
        local_dataset_root=local_dataset_root,
    )
    data_path = dataset_root / "data"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Local AntMaze dataset not found at {data_path}. "
            "Generate it with local_antmaze_gen.py first."
        )
    try:
        dataset = MinariDataset(data_path)
        total_steps = int(dataset.total_steps)
    except ValueError as exc:
        if "No data found in data path" not in str(exc):
            raise
        h5_path = data_path / "main_data.hdf5"
        if not h5_path.exists():
            raise FileNotFoundError(
                f"Local AntMaze data file not found at {h5_path}. "
                "Generate it with local_antmaze_gen.py first."
            )
        with h5py.File(h5_path, "r") as f:
            if "total_steps" in f.attrs:
                total_steps = int(f.attrs["total_steps"])
            else:
                total_steps = sum(
                    int(f[name]["actions"].shape[0])
                    for name in f.keys()
                    if name.startswith("episode_")
                )
    return f"localsteps{total_steps}"


class AntMazeDataset(PointMazeDataset):
    """D4RL AntMaze dataset using the shared goal-maze tokenization pipeline."""

    ENV_FAMILY = "antmaze"
    VARIANTS = ANTMAZE_VARIANTS
    ACTION_DIM = 8
    CACHE_FORMAT = "antmaze_hash_signature_v7"

    @classmethod
    def _normalize_family_data_config(cls, family_data_config: dict | None):
        return _normalize_antmaze_data_config(family_data_config)

    @classmethod
    def _normalize_request(cls, request):
        config = super()._normalize_request(request)
        return replace(
            config,
            family_data_config=_normalize_antmaze_data_config(request.family_data_config),
        )

    @classmethod
    def _load_variant_episodes(
        cls,
        variant: str,
        family_data_config: dict | None = None,
        local_dataset_root: str | None = None,
    ):
        if variant not in cls.VARIANTS:
            raise ValueError(f"Unknown AntMaze variant: {variant}")
        data_config = _normalize_antmaze_data_config(family_data_config)
        meta = cls.VARIANTS[variant]
        if cls._get_variant_type(meta) == "local":
            dataset_root = resolve_local_dataset_path(
                meta["dataset_path"],
                local_dataset_root=local_dataset_root,
            )
            data_path = dataset_root / "data"
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Local AntMaze dataset for variant={variant!r} not found at {data_path}. "
                    "Generate it with local_antmaze_gen.py first."
                )
            try:
                dataset = MinariDataset(data_path)
                episodes = list(dataset.iterate_episodes())
            except ValueError as exc:
                if "No data found in data path" not in str(exc):
                    raise
                episodes = _load_local_antmaze_hdf5_episodes(data_path)
        else:
            dataset = minari.load_dataset(meta["dataset_id"], download=True)
            episodes = list(dataset.iterate_episodes())
        episodes = _apply_antmaze_data_config(episodes, data_config, variant=variant)
        step_counts = [len(episode.actions) for episode in episodes]
        return meta, episodes, step_counts

    @classmethod
    def _get_variant_type(cls, meta: dict) -> str:
        return get_antmaze_variant_type(meta)

    @classmethod
    def _local_data_signature(
        cls,
        meta: dict,
        local_dataset_root: str | None = None,
    ) -> str | None:
        if cls._get_variant_type(meta) != "local":
            return None
        return _local_antmaze_dataset_step_signature(
            meta,
            local_dataset_root=local_dataset_root,
        )
