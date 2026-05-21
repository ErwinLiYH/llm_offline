import json
import multiprocessing
import os
import pickle
import hashlib
import math
import signal
import time
import gc
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import h5py
import minari
from minari import MinariDataset
import numpy as np
import torch

from transformers import AutoTokenizer

from data.base_dataset import BaseOfflineDataset, DatasetBuildRequest, TensorSample, VariantEpisodeStats
from data.pointmaze import formatting
from data.pointmaze.variants import (
    POINTMAZE_VARIANTS,
    get_pointmaze_variant_type,
    resolve_local_dataset_path,
)
from utils import action_bins as action_bins_module
from utils import chat_template as chat_template_module
from utils import prompt_loader as prompt_loader_module
from utils.action_bins import (
    get_action_bin_range,
    get_action_bin_codec,
    get_action_num_bins,
    get_action_token_mode,
)
from utils.chat_template import build_generation_prompt, build_training_conversation
from utils.file_progress import (
    MultiWorkerFileProgress,
    get_sub_progress_process,
)
from utils.prompt_loader import load_named_templates, load_template_names, render_template

DEFAULT_TRAIN_DATA_RATIO = 0.9
DEFAULT_EPISODE_KEEP_NUM = None
DEFAULT_SAMPLING_SEED = 0
HUMAN_READABLE_CACHE_EPISODE_LIMIT = 3

_POINTMAZE_WORKER_TOKENIZER = None
_POINTMAZE_WORKER_CONFIGS: dict[str, dict] | None = None
_POINTMAZE_WORKER_SHARED_CONFIG: dict | None = None
_POINTMAZE_WORKER_ACTION_CODEC = None
_POINTMAZE_WORKER_DONE = 0
_POINTMAZE_WORKER_START_TIME = 0.0
_LINUX_PR_SET_PDEATHSIG = 1


def _pointmaze_action_config(config: dict) -> dict:
    return {
        "action_token_mode": config["action_token_mode"],
        "action_num_bins": config["action_num_bins"],
        "action_bin_min": config["action_bin_min"],
        "action_bin_max": config["action_bin_max"],
        "new_token": config.get("new_token", False),
    }


def _terminate_worker_when_parent_dies():
    """Ask Linux to SIGTERM this worker if its parent process exits.

    `ProcessPoolExecutor` does not expose a standard "kill workers when parent
    dies" option. With the spawn context, each worker runs this initializer, so
    setting PR_SET_PDEATHSIG here prevents tokenization workers from surviving
    as PPID=1 orphans after the training process is killed.
    """
    if os.name != "posix" or not hasattr(os, "getppid"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        result = libc.prctl(_LINUX_PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        return
    if result != 0:
        return

    # Race guard: the parent can die between process spawn and prctl().
    if os.getppid() == 1:
        os._exit(1)


def _init_pointmaze_tokenization_worker(progress_initializer, progress_initargs: tuple, worker_config: dict):
    global _POINTMAZE_WORKER_TOKENIZER
    global _POINTMAZE_WORKER_CONFIGS
    global _POINTMAZE_WORKER_SHARED_CONFIG
    global _POINTMAZE_WORKER_ACTION_CODEC
    global _POINTMAZE_WORKER_DONE
    global _POINTMAZE_WORKER_START_TIME

    _terminate_worker_when_parent_dies()
    progress_initializer(*progress_initargs)
    _POINTMAZE_WORKER_SHARED_CONFIG = dict(worker_config["shared"])
    _POINTMAZE_WORKER_CONFIGS = dict(worker_config["plans"])
    _POINTMAZE_WORKER_DONE = 0
    _POINTMAZE_WORKER_START_TIME = time.monotonic()
    _POINTMAZE_WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        _POINTMAZE_WORKER_SHARED_CONFIG["tokenizer_name_or_path"],
        trust_remote_code=True,
    )
    action_config = _pointmaze_action_config(_POINTMAZE_WORKER_SHARED_CONFIG)
    if action_config["action_token_mode"] != "text":
        _POINTMAZE_WORKER_ACTION_CODEC = get_action_bin_codec(
            _POINTMAZE_WORKER_TOKENIZER,
            action_config,
            ensure_registered=True,
        )
    else:
        _POINTMAZE_WORKER_ACTION_CODEC = None


def _format_pointmaze_action_texts(action: np.ndarray, config: dict) -> dict:
    if config["action_token_mode"] == "text":
        action_text = formatting.format_action(action)
        return {
            "model_text": action_text,
            "display_text": action_text,
            "bin_indices": [],
            "token_ids": [],
        }

    if _POINTMAZE_WORKER_TOKENIZER is None or _POINTMAZE_WORKER_ACTION_CODEC is None:
        raise RuntimeError("PointMaze action-bin codec was not initialized.")
    bin_indices = _POINTMAZE_WORKER_ACTION_CODEC.bin_indices_for_action(
        action,
        low=config["action_bin_min"],
        high=config["action_bin_max"],
    )
    return {
        "model_text": _POINTMAZE_WORKER_ACTION_CODEC.model_text_for_bins(
            _POINTMAZE_WORKER_TOKENIZER,
            bin_indices,
        ),
        "display_text": _POINTMAZE_WORKER_ACTION_CODEC.display_text_for_bins(bin_indices),
        "bin_indices": bin_indices,
        "token_ids": _POINTMAZE_WORKER_ACTION_CODEC.token_ids_for_bins(bin_indices),
    }


def _find_subsequence(values: list[int], needle: list[int], start: int) -> int | None:
    if not needle:
        return None
    max_start = len(values) - len(needle)
    for pos in range(max(start, 0), max_start + 1):
        if values[pos : pos + len(needle)] == needle:
            return pos
    return None


def _tokenize_pointmaze_sample(
    prompt: str,
    action_text: str,
    config: dict,
    *,
    expected_action_token_ids: list[int] | None = None,
    expected_action_bin_indices: list[int] | None = None,
) -> dict:
    tok = _POINTMAZE_WORKER_TOKENIZER
    if tok is None:
        raise RuntimeError("PointMaze tokenization worker was not initialized.")

    prompt_text = build_generation_prompt(tok, prompt)
    full_text = build_training_conversation(tok, prompt, action_text)

    prompt_ids = tok(text=prompt_text, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)

    full_enc = tok(
        text=full_text,
        add_special_tokens=False,
        max_length=config["max_length"],
        truncation=True,
    )
    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    action_bin_labels = [-1] * len(input_ids)
    if config["action_token_mode"] != "text":
        if not expected_action_token_ids or expected_action_bin_indices is None:
            raise RuntimeError("Missing expected action-bin token IDs for PointMaze tokenization.")
        action_start = _find_subsequence(input_ids, expected_action_token_ids, prompt_len)
        if action_start is None:
            raise ValueError(
                "Tokenized PointMaze sample does not contain the expected action-bin token ID sequence. "
                f"expected_ids={expected_action_token_ids}, prompt_len={prompt_len}, "
                f"seq_len={len(input_ids)}, max_length={config['max_length']}."
            )
        for offset, bin_idx in enumerate(expected_action_bin_indices):
            action_bin_labels[action_start + offset] = int(bin_idx)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "action_bin_labels": action_bin_labels,
    }


def _process_pointmaze_episode(payload: dict) -> list[tuple[int, dict | None, dict]]:
    global _POINTMAZE_WORKER_DONE

    configs = _POINTMAZE_WORKER_CONFIGS
    shared_config = _POINTMAZE_WORKER_SHARED_CONFIG
    if configs is None or shared_config is None:
        raise RuntimeError("PointMaze tokenization worker was not initialized.")

    job_id = str(payload["job_id"])
    config = configs[job_id]
    episode_idx = int(payload["episode_idx"])
    obs_arr = payload["observations"]
    goal_arr = payload["goals"]
    actions = payload["actions"]
    templates = config["templates"]
    prompt_vars = config["prompt_vars"]
    include_text_records = bool(payload.get("include_text_records", False))
    worker_total = int(shared_config["worker_total"])
    split = config["split"]
    variant = config["variant"]

    sub_progress = get_sub_progress_process()
    sub_progress.update(
        f"Tokenizing {variant} [{split}]",
        _POINTMAZE_WORKER_DONE,
        worker_total,
        _POINTMAZE_WORKER_START_TIME,
        extra=f"episode={episode_idx} steps={len(actions)}",
    )

    results = []
    for t, action in enumerate(actions):
        obs_vec = obs_arr[t].astype(np.float32)
        goal = goal_arr[t].astype(np.float32)
        action = action.astype(np.float32)
        obs = {
            "observation": obs_vec,
            "desired_goal": goal,
        }
        action_texts = _format_pointmaze_action_texts(action, config)
        history_entries = []
        if config["history_num"] > 0:
            history_indices = []
            hist_idx = t - 1
            while hist_idx >= 0 and len(history_indices) < config["history_num"]:
                history_indices.append(hist_idx)
                hist_idx -= config["history_stride"]
            history_indices.reverse()
            for hist_t in history_indices:
                history_entries.append(
                    {
                        "observation": obs_arr[hist_t].astype(np.float32),
                        "action_text": _format_pointmaze_action_texts(
                            actions[hist_t].astype(np.float32),
                            config,
                        )["display_text"],
                        "steps_ago": t - hist_t,
                    }
                )
        obs_payload = formatting.format_obs(obs, prompt_vars)
        history_payload = formatting.format_history(history_entries, prompt_vars)
        for template in templates:
            prompt = render_template(template, prompt_vars, **obs_payload, **history_payload)
            token_sample = _tokenize_pointmaze_sample(
                prompt,
                action_texts["model_text"],
                config,
                expected_action_token_ids=action_texts["token_ids"],
                expected_action_bin_indices=action_texts["bin_indices"],
            )
            text_record = None
            if include_text_records:
                text_record = {
                    "episode_idx": episode_idx,
                    "timestep": t,
                    "prompt": prompt,
                    "action": action_texts["display_text"],
                }
            results.append((episode_idx, text_record, token_sample))

    _POINTMAZE_WORKER_DONE += 1
    sub_progress.update(
        f"Tokenizing {variant} [{split}]",
        _POINTMAZE_WORKER_DONE,
        worker_total,
        _POINTMAZE_WORKER_START_TIME,
        extra=f"episode={episode_idx} samples={len(results)}",
    )
    sub_progress.increment_total(1)
    return results


def _load_variant_episodes(variant: str):
    meta = POINTMAZE_VARIANTS[variant]
    if get_pointmaze_variant_type(meta) == "local":
        dataset_root = resolve_local_dataset_path(meta["dataset_path"])
        data_path = dataset_root / "data"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Local PointMaze dataset for variant={variant!r} not found at {data_path}. "
                "Generate it with local_varient_gen.py first."
            )
        try:
            dataset = MinariDataset(data_path)
        except ValueError as exc:
            if "No data found in data path" not in str(exc):
                raise
            episodes = _load_local_hdf5_episodes(data_path)
            step_counts = [len(episode.actions) for episode in episodes]
            return meta, episodes, step_counts
    else:
        dataset = minari.load_dataset(meta["dataset_id"], download=True)
    episodes = list(dataset.iterate_episodes())
    step_counts = [len(episode.actions) for episode in episodes]
    return meta, episodes, step_counts


def _load_local_hdf5_episodes(data_path):
    h5_path = data_path / "main_data.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Local PointMaze data file not found at {h5_path}. "
            "Generate it with local_varient_gen.py first."
        )
    episodes = []
    with h5py.File(h5_path, "r") as f:
        episode_names = sorted(
            (name for name in f.keys() if name.startswith("episode_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        for name in episode_names:
            group = f[name]
            obs_group = group["observations"]
            observations = {
                obs_name: obs_group[obs_name][()]
                for obs_name in obs_group.keys()
            }
            episodes.append(
                SimpleNamespace(
                    observations=observations,
                    actions=group["actions"][()],
                )
            )
    if not episodes:
        raise ValueError(f"No episodes found in local PointMaze data file {h5_path}")
    return episodes


def _local_dataset_step_signature(meta: dict) -> str:
    dataset_root = resolve_local_dataset_path(meta["dataset_path"])
    data_path = dataset_root / "data"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Local PointMaze dataset not found at {data_path}. "
            "Generate it with local_varient_gen.py first."
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
                f"Local PointMaze data file not found at {h5_path}. "
                "Generate it with local_varient_gen.py first."
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


def _normalize_episode_keep_num(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "episode_keep_num must be an integer episode count or null/omitted to use all episodes, "
            f"got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"episode_keep_num must be >= 1 when provided, got {value}")
    return value


def _compute_sampled_episode_target(total_episodes: int, episode_keep_num: int | None) -> int:
    if total_episodes < 1:
        raise ValueError("Offline dataset contains no episodes.")
    episode_keep_num = _normalize_episode_keep_num(episode_keep_num)
    if episode_keep_num is None:
        return total_episodes
    return min(total_episodes, episode_keep_num)


def _variant_sampling_seed(variant: str, sampling_seed: int) -> int:
    digest = hashlib.sha256(f"{variant}:{sampling_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _collect_variant_episode_stats(variant: str, episode_keep_num: int | None) -> VariantEpisodeStats:
    _, episodes, step_counts = _load_variant_episodes(variant)
    total_episodes = len(episodes)
    total_steps = sum(step_counts)
    sampled_episode_target = _compute_sampled_episode_target(total_episodes, episode_keep_num)
    return {
        "variant": variant,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "initial_train_target": sampled_episode_target,
        "sampled_episode_target": sampled_episode_target,
    }


def select_variant_episode_indices(
    variant: str,
    train_data_ratio: float,
    episode_keep_num: int | None,
    sampling_seed: int,
    balanced_train_target: int | None = None,
) -> dict:
    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(
            "Invalid train_data_ratio: expected 0 < train_data_ratio < 1, "
            f"got train_data_ratio={train_data_ratio}"
        )
    if not isinstance(sampling_seed, int):
        raise ValueError(f"sampling_seed must be an int, got {type(sampling_seed).__name__}")

    _, episodes, step_counts = _load_variant_episodes(variant)
    total_episodes = len(episodes)
    total_steps = sum(step_counts)
    initial_sampled_target = _compute_sampled_episode_target(total_episodes, episode_keep_num)
    sampled_target = initial_sampled_target if balanced_train_target is None else balanced_train_target
    sampled_target = min(total_episodes, sampled_target)
    if sampled_target < 1:
        raise ValueError(f"sampled_target must be >= 1 for variant={variant}, got {sampled_target}")

    rng = np.random.default_rng(_variant_sampling_seed(variant, sampling_seed))
    permutation = rng.permutation(total_episodes).tolist()
    sampled_indices = permutation[:sampled_target]
    train_target = math.floor(sampled_target * train_data_ratio)
    if train_target < 1:
        raise ValueError(
            "train_data_ratio and episode_keep_num selected zero train episodes: "
            f"sampled_target={sampled_target}, train_data_ratio={train_data_ratio}"
        )
    val_target = sampled_target - train_target
    train_indices = sorted(sampled_indices[:train_target])
    val_indices = sorted(sampled_indices[train_target:])

    train_steps = sum(step_counts[idx] for idx in train_indices)
    val_steps = sum(step_counts[idx] for idx in val_indices)

    return {
        "variant": variant,
        "episodes": episodes,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "initial_train_target": initial_sampled_target,
        "initial_sampled_target": initial_sampled_target,
        "sampled_episode_count": sampled_target,
        "balanced_train_target": balanced_train_target,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_episode_count": len(train_indices),
        "val_episode_count": len(val_indices),
        "train_steps": train_steps,
        "val_steps": val_steps,
        "val_target": val_target,
        "val_shortfall_reason": None,
    }


@dataclass(frozen=True)
class PointMazeBuildConfig:
    variant: str
    split: str
    tokenizer_name_or_path: str
    max_length: int
    num_workers: int
    cache_dir: str | None
    max_data_num: int | None
    prompt_template_count: int
    prompt_templete_index: list[str] | None
    train_data_ratio: float
    episode_keep_num: int | None
    balance_variant_episode_count: bool
    balanced_train_episode_count: int | None
    sampling_seed: int
    history_num: int
    history_stride: int
    action_token_mode: str
    action_num_bins: int
    action_bin_min: float
    action_bin_max: float
    new_token: bool
    action_token_schema_hash: str
    progress_interval_seconds: float


@dataclass
class PointMazeTokenizationJob:
    job_id: str
    config: PointMazeBuildConfig
    cache_path: str | None
    total_episodes: int
    episode_indices: list[int]
    episode_payloads: list[dict]
    worker_config: dict
    shared_config: dict


class PointMazeDataset(BaseOfflineDataset):
    """Loaded tokenized PointMaze behavior cloning dataset."""

    def __init__(self, variant: str, split: str, samples: list[dict]):
        super().__init__()
        self.variant = variant
        self.split = split
        self._samples = samples

    @classmethod
    def collect_variant_episode_stats(cls, variant: str, episode_keep_num: int | None) -> VariantEpisodeStats:
        return _collect_variant_episode_stats(variant, episode_keep_num)

    @classmethod
    def build_batch(cls, requests: list[DatasetBuildRequest]) -> list["PointMazeDataset"]:
        configs = [cls._normalize_request(request) for request in requests]
        datasets: list[PointMazeDataset | None] = [None] * len(configs)
        jobs: list[PointMazeTokenizationJob] = []
        pending_variant_indices: dict[str, list[int]] = {}
        selections_by_variant: dict[str, dict] = {}

        for idx, config in enumerate(configs):
            cls._validate_config(config)
            pending_variant_indices.setdefault(config.variant, []).append(idx)

        for variant, request_indices in pending_variant_indices.items():
            variant_configs = [configs[idx] for idx in request_indices]
            cls._validate_variant_request_group(variant_configs)
            base_config = variant_configs[0]
            selection = select_variant_episode_indices(
                variant=base_config.variant,
                train_data_ratio=base_config.train_data_ratio,
                episode_keep_num=base_config.episode_keep_num,
                sampling_seed=base_config.sampling_seed,
                balanced_train_target=base_config.balanced_train_episode_count,
            )
            selections_by_variant[variant] = selection
            cls._print_selection_summary(base_config, selection)
            cache_path = cls._cache_path(base_config)
            cached_episodes = cls._load_cached_episodes(base_config, cache_path, selection)
            if cached_episodes is not None:
                cls._fill_datasets_from_episode_cache(
                    datasets=datasets,
                    request_indices=request_indices,
                    configs=configs,
                    selection=selection,
                    episode_samples=cached_episodes,
                )
                selection.pop("episodes", None)
                continue

            job = cls._create_tokenization_job(base_config, cache_path, selection)
            selection.pop("episodes", None)
            jobs.append(job)

        cls._print_total_selection_summary(list(selections_by_variant.values()))

        if jobs:
            num_workers = max(job.config.num_workers for job in jobs)
            progress_interval_seconds = min(job.config.progress_interval_seconds for job in jobs)
            results_by_job = cls._execute_tokenization_jobs(
                jobs,
                num_workers=num_workers,
                progress_interval_seconds=progress_interval_seconds,
            )
            for job in jobs:
                episode_samples = cls._finalize_tokenization_job(job, results_by_job[job.job_id])
                results_by_job.pop(job.job_id, None)
                request_indices = pending_variant_indices[job.config.variant]
                cls._fill_datasets_from_episode_cache(
                    datasets=datasets,
                    request_indices=request_indices,
                    configs=configs,
                    selection=selections_by_variant[job.config.variant],
                    episode_samples=episode_samples,
                )
                job.episode_payloads.clear()
                del episode_samples
                gc.collect()
            results_by_job.clear()
            jobs.clear()
            gc.collect()

        if any(dataset is None for dataset in datasets):
            raise RuntimeError("PointMazeDataset.build_batch did not construct every requested dataset.")
        return [dataset for dataset in datasets if dataset is not None]

    @classmethod
    def _normalize_request(cls, request: DatasetBuildRequest) -> PointMazeBuildConfig:
        tokenizer_name_or_path = request.tokenizer_name_or_path or getattr(request.tokenizer, "name_or_path", None)
        if not tokenizer_name_or_path:
            raise ValueError(
                "PointMazeDataset requires tokenizer_name_or_path when tokenizer does not expose name_or_path."
            )
        action_config = {
            "action_token_mode": request.action_token_mode,
            "action_num_bins": request.action_num_bins,
            "action_bin_min": request.action_bin_min,
            "action_bin_max": request.action_bin_max,
            "new_token": request.new_token,
        }
        action_token_mode = get_action_token_mode(action_config)
        new_token = bool(request.new_token) if action_token_mode != "text" else False
        action_config["new_token"] = new_token
        action_num_bins = get_action_num_bins(action_config)
        action_bin_min, action_bin_max = get_action_bin_range(action_config)
        if action_token_mode == "text":
            action_token_schema_hash = "text"
        else:
            action_token_schema_hash = get_action_bin_codec(
                request.tokenizer,
                action_config,
                ensure_registered=True,
            ).mapping_hash
        return PointMazeBuildConfig(
            variant=request.variant,
            split=request.split,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_length=request.max_length,
            num_workers=max(int(request.num_workers), 1),
            cache_dir=request.cache_dir,
            max_data_num=request.max_data_num,
            prompt_template_count=request.prompt_template_count,
            prompt_templete_index=cls._normalize_prompt_templete_index(request.prompt_templete_index),
            train_data_ratio=request.train_data_ratio,
            episode_keep_num=_normalize_episode_keep_num(request.episode_keep_num),
            balance_variant_episode_count=request.balance_variant_episode_count,
            balanced_train_episode_count=request.balanced_train_episode_count,
            sampling_seed=request.sampling_seed,
            history_num=request.history_num,
            history_stride=request.history_stride,
            action_token_mode=action_token_mode,
            action_num_bins=action_num_bins,
            action_bin_min=action_bin_min,
            action_bin_max=action_bin_max,
            new_token=new_token,
            action_token_schema_hash=action_token_schema_hash,
            progress_interval_seconds=float(request.progress_interval_seconds),
        )

    @staticmethod
    def _normalize_prompt_templete_index(value) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(
                f"prompt_templete_index must be a list of prompt names, got {type(value).__name__}"
            )
        names = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"prompt_templete_index must contain non-empty strings, got {item!r}")
            names.append(item.strip())
        if not names:
            raise ValueError("prompt_templete_index must not be empty when provided")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"prompt_templete_index contains duplicate prompt names: {duplicates}")
        return names

    @classmethod
    def _resolve_prompt_names(cls, config: PointMazeBuildConfig) -> list[str]:
        available_names = load_template_names("pointmaze")
        if config.prompt_templete_index is not None:
            missing = [name for name in config.prompt_templete_index if name not in available_names]
            if missing:
                available = ", ".join(available_names)
                raise ValueError(
                    f"Unknown prompt template names for pointmaze: {missing}. Available: {available}"
                )
            return list(config.prompt_templete_index)

        if config.prompt_template_count < 1:
            raise ValueError(
                f"prompt_template_count must be >= 1, got {config.prompt_template_count}"
            )
        if config.prompt_template_count > len(available_names):
            raise ValueError(
                "prompt_template_count exceeds available templates: "
                f"requested {config.prompt_template_count}, available {len(available_names)}"
            )
        return available_names[: config.prompt_template_count]

    @staticmethod
    def _source_path_hash(path) -> str:
        if not path or not os.path.exists(path):
            return "missing"
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @classmethod
    def _source_file_hash(cls, module) -> str:
        return cls._source_path_hash(getattr(module, "__file__", None))

    @classmethod
    def _cache_signature_payload(cls, config: PointMazeBuildConfig) -> dict:
        meta = POINTMAZE_VARIANTS[config.variant]
        variant_type = get_pointmaze_variant_type(meta)
        prompt_names = cls._resolve_prompt_names(config)
        templates = load_named_templates("pointmaze", prompt_names)
        local_data_signature = None
        if variant_type == "local":
            local_data_signature = _local_dataset_step_signature(meta)
        return {
            "env_family": "pointmaze",
            "cache_kind": "episode_tokenized_samples",
            "variant": config.variant,
            "variant_type": variant_type,
            "variant_metadata": {
                "dataset_id": meta.get("dataset_id"),
                "dataset_path": meta.get("dataset_path"),
                "env_id": meta.get("env_id"),
                "env_paras": meta.get("env_paras"),
                "prompt_vars": meta["prompt_vars"],
            },
            "local_data_signature": local_data_signature,
            "tokenizer_name_or_path": config.tokenizer_name_or_path,
            "max_length": config.max_length,
            "prompt_names": prompt_names,
            "templates": templates,
            "history_num": config.history_num,
            "history_stride": config.history_stride,
            "action_token_mode": config.action_token_mode,
            "action_num_bins": config.action_num_bins,
            "action_bin_min": config.action_bin_min,
            "action_bin_max": config.action_bin_max,
            "new_token": config.new_token,
            "action_token_schema_hash": config.action_token_schema_hash,
            "source_hashes": {
                "data.pointmaze.dataset": cls._source_path_hash(__file__),
                "data.pointmaze.formatting": cls._source_file_hash(formatting),
                "utils.action_bins": cls._source_file_hash(action_bins_module),
                "utils.chat_template": cls._source_file_hash(chat_template_module),
                "utils.prompt_loader": cls._source_file_hash(prompt_loader_module),
            },
        }

    @staticmethod
    def _hash_json_payload(payload: dict, *, length: int = 32) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]

    @classmethod
    def _cache_signature_hash(cls, config: PointMazeBuildConfig) -> str:
        return cls._hash_json_payload(cls._cache_signature_payload(config))

    @classmethod
    def _cache_path(cls, config: PointMazeBuildConfig) -> str | None:
        if config.cache_dir is None:
            return None
        return os.path.join(config.cache_dir, f"{cls._cache_signature_hash(config)}.pkl")

    @classmethod
    def _load_cached_episodes(
        cls,
        config: PointMazeBuildConfig,
        cache_path: str | None,
        selection: dict,
    ) -> dict[int, list[dict]] | None:
        if not cache_path or not os.path.exists(cache_path):
            return None
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        metadata = cache.get("metadata", {})
        expected_signature_hash = cls._cache_signature_hash(config)
        if metadata.get("cache_signature_hash") != expected_signature_hash:
            raise ValueError(
                f"Dataset cache schema mismatch at {cache_path}: "
                f"cache_signature_hash={metadata.get('cache_signature_hash')!r}, "
                f"expected {expected_signature_hash!r}. "
                "Remove the stale cache file or rebuild with matching tokenization settings."
            )
        episode_samples = {
            int(episode_idx): samples
            for episode_idx, samples in cache.get("episodes", {}).items()
        }
        required_indices = set(selection["train_indices"]) | set(selection["val_indices"])
        cached_indices = set(episode_samples)
        if not required_indices.issubset(cached_indices):
            missing_count = len(required_indices - cached_indices)
            print(
                f"[dataset] Cache at {cache_path} does not cover current sampled episodes "
                f"for variant {config.variant} (missing {missing_count}); rebuilding cache."
            )
            return None
        print(
            f"[dataset] Loading cached tokenized episodes from {cache_path} "
            f"(using {len(required_indices)} / cached {len(cached_indices)} episodes)"
        )
        return episode_samples

    @classmethod
    def _create_tokenization_job(
        cls,
        config: PointMazeBuildConfig,
        cache_path: str | None,
        selection: dict,
    ) -> PointMazeTokenizationJob:
        meta = POINTMAZE_VARIANTS[config.variant]
        prompt_names = cls._resolve_prompt_names(config)
        templates = load_named_templates("pointmaze", prompt_names)
        prompt_vars = meta["prompt_vars"]

        all_episodes = selection["episodes"]
        episode_indices = sorted(set(selection["train_indices"]) | set(selection["val_indices"]))

        job_id = f"{config.variant}:episodes:{len(episode_indices)}:{id(config)}"
        human_readable_episode_indices = set(
            episode_indices[:HUMAN_READABLE_CACHE_EPISODE_LIMIT]
        ) if cache_path else set()
        episode_payloads = [
            {
                "job_id": job_id,
                "episode_idx": episode_idx,
                "include_text_records": episode_idx in human_readable_episode_indices,
                "observations": episode.observations["observation"],
                "goals": episode.observations["desired_goal"],
                "actions": episode.actions,
            }
            for episode_idx in episode_indices
            for episode in [all_episodes[episode_idx]]
        ]
        worker_config = {
            "variant": config.variant,
            "split": "cache",
            "max_length": config.max_length,
            "templates": templates,
            "prompt_vars": prompt_vars,
            "history_num": config.history_num,
            "history_stride": config.history_stride,
            "action_token_mode": config.action_token_mode,
            "action_num_bins": config.action_num_bins,
            "action_bin_min": config.action_bin_min,
            "action_bin_max": config.action_bin_max,
            "new_token": config.new_token,
            "action_token_schema_hash": config.action_token_schema_hash,
        }
        shared_config = {
            "tokenizer_name_or_path": config.tokenizer_name_or_path,
            "action_token_mode": config.action_token_mode,
            "action_num_bins": config.action_num_bins,
            "action_bin_min": config.action_bin_min,
            "action_bin_max": config.action_bin_max,
            "new_token": config.new_token,
            "action_token_schema_hash": config.action_token_schema_hash,
        }
        return PointMazeTokenizationJob(
            job_id=job_id,
            config=config,
            cache_path=cache_path,
            total_episodes=selection["total_episodes"],
            episode_indices=episode_indices,
            episode_payloads=episode_payloads,
            worker_config=worker_config,
            shared_config=shared_config,
        )

    @staticmethod
    def _validate_config(config: PointMazeBuildConfig):
        if not (0.0 < config.train_data_ratio < 1.0):
            raise ValueError(
                "Invalid train_data_ratio: expected 0 < train_data_ratio < 1, "
                f"got train_data_ratio={config.train_data_ratio}"
            )
        if not isinstance(config.sampling_seed, int):
            raise ValueError(f"sampling_seed must be an int, got {type(config.sampling_seed).__name__}")
        if config.history_num < 0:
            raise ValueError(f"history_num must be >= 0, got {config.history_num}")
        if config.history_stride < 1:
            raise ValueError(f"history_stride must be >= 1, got {config.history_stride}")

    @staticmethod
    def _validate_variant_request_group(configs: list[PointMazeBuildConfig]):
        if not configs:
            return
        base = configs[0]
        for config in configs[1:]:
            fields = (
                "tokenizer_name_or_path",
                "max_length",
                "num_workers",
                "cache_dir",
                "prompt_template_count",
                "prompt_templete_index",
                "train_data_ratio",
                "episode_keep_num",
                "balance_variant_episode_count",
                "balanced_train_episode_count",
                "sampling_seed",
                "history_num",
                "history_stride",
                "action_token_mode",
                "action_num_bins",
                "action_bin_min",
                "action_bin_max",
                "new_token",
                "action_token_schema_hash",
            )
            for field in fields:
                if getattr(config, field) != getattr(base, field):
                    raise ValueError(
                        "PointMazeDataset.build_batch requires requests for the same variant "
                        f"to share {field}; got {getattr(base, field)!r} and {getattr(config, field)!r}."
                    )

    @classmethod
    def _fill_datasets_from_episode_cache(
        cls,
        *,
        datasets: list["PointMazeDataset | None"],
        request_indices: list[int],
        configs: list[PointMazeBuildConfig],
        selection: dict,
        episode_samples: dict[int, list[dict]],
    ):
        for request_idx in request_indices:
            config = configs[request_idx]
            if config.split == "train":
                selected_indices = selection["train_indices"]
            elif config.split == "val":
                selected_indices = selection["val_indices"]
            else:
                raise ValueError(f"Unknown split: {config.split!r}. Expected 'train' or 'val'.")

            samples = []
            for episode_idx in selected_indices:
                samples.extend(episode_samples[int(episode_idx)])

            if config.max_data_num is not None:
                samples = samples[: config.max_data_num]
                print(f"[dataset] max_data_num={config.max_data_num}: using {len(samples)} samples")
            datasets[request_idx] = cls(config.variant, config.split, samples)

    @staticmethod
    def _print_selection_summary(config: PointMazeBuildConfig, selection: dict):
        balanced_target = selection["balanced_train_target"]
        balance_text = (
            "not applied"
            if balanced_target is None
            else (
                f"sampled pool clipped to {selection['sampled_episode_count']}"
                if balanced_target < selection["initial_sampled_target"]
                else "enabled, unchanged"
            )
        )
        if config.split == "train":
            print(
                f"[dataset] Variant {config.variant}: total_episodes={selection['total_episodes']}, "
                f"total_steps={selection['total_steps']}, initial_sampled_target={selection['initial_sampled_target']}, "
                f"balance={balance_text}, sampled_episodes={selection['sampled_episode_count']}, "
                f"final_train_episodes={selection['train_episode_count']}, "
                f"train_steps={selection['train_steps']}, final_val_episodes={selection['val_episode_count']}, "
                f"val_steps={selection['val_steps']}"
            )
        if config.split == "val" and selection["val_episode_count"] == 0:
            print(
                f"[dataset] WARNING: Variant {config.variant} val split is empty "
                f"(sampled_episodes={selection['sampled_episode_count']}, "
                f"train_data_ratio={config.train_data_ratio})."
            )

    @staticmethod
    def _print_total_selection_summary(selections: list[dict]):
        if not selections:
            return
        train_episodes = sum(selection["train_episode_count"] for selection in selections)
        val_episodes = sum(selection["val_episode_count"] for selection in selections)
        train_steps = sum(selection["train_steps"] for selection in selections)
        val_steps = sum(selection["val_steps"] for selection in selections)
        sampled_episodes = sum(selection["sampled_episode_count"] for selection in selections)
        print(
            "[dataset] Total selected across variants: "
            f"sampled_episodes={sampled_episodes}, "
            f"train_episodes={train_episodes}, train_steps={train_steps}, "
            f"val_episodes={val_episodes}, val_steps={val_steps}"
        )

    @classmethod
    def _execute_tokenization_jobs(
        cls,
        jobs: list[PointMazeTokenizationJob],
        *,
        num_workers: int,
        progress_interval_seconds: float = 5.0,
    ) -> dict[str, list[list[tuple[int, dict | None, dict]]]]:
        pending_jobs = [job for job in jobs if job.episode_payloads]
        if not pending_jobs:
            return {job.job_id: [] for job in jobs}

        shared_config = dict(pending_jobs[0].shared_config)
        for job in pending_jobs[1:]:
            if dict(job.shared_config) != shared_config:
                raise ValueError(
                    "PointMazeDataset.build_batch requires all pending datasets to use the same "
                    "tokenizer and action-bin configuration."
                )

        episode_payloads = []
        job_configs = {}
        for job in pending_jobs:
            job_configs[job.job_id] = job.worker_config
            episode_payloads.extend(job.episode_payloads)

        num_workers = min(os.cpu_count() or 1, max(int(num_workers), 1), max(len(episode_payloads), 1))
        worker_config = {
            "shared": {
                **shared_config,
                "worker_total": max(math.ceil(len(episode_payloads) / max(num_workers, 1)), 1),
            },
            "plans": job_configs,
        }
        results_by_job: dict[str, list[list[tuple[int, dict | None, dict]]]] = {
            job.job_id: [] for job in jobs
        }
        ctx = multiprocessing.get_context("spawn")
        with MultiWorkerFileProgress(
            desc="Tokenizing pointmaze datasets",
            total=len(episode_payloads),
            interval_seconds=progress_interval_seconds,
            cleanup_on_success=True,
        ) as progress:
            print(f"[dataset] Tokenization progress in file: {progress.path.resolve()}")
            progress_initializer, progress_initargs = progress.process_initializer(ctx)
            chunksize = max(1, math.ceil(len(episode_payloads) / max(num_workers * 8, 1)))
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=_init_pointmaze_tokenization_worker,
                initargs=(progress_initializer, progress_initargs, worker_config),
            ) as executor:
                for idx, episode_results in enumerate(
                    executor.map(
                        _process_pointmaze_episode,
                        episode_payloads,
                        chunksize=chunksize,
                    )
                ):
                    payload = episode_payloads[idx]
                    results_by_job[str(payload["job_id"])].append(episode_results)
                    episode_payloads[idx] = None
        episode_payloads.clear()
        gc.collect()
        return results_by_job

    @classmethod
    def _finalize_tokenization_job(
        cls,
        job: PointMazeTokenizationJob,
        episode_results_list: list[list[tuple[int, dict | None, dict]]],
    ) -> dict[int, list[dict]]:
        episode_samples: dict[int, list[dict]] = {}
        text_records = []
        for episode_results in episode_results_list:
            episode_idx = None
            samples = []
            for result_episode_idx, text_record, token_sample in episode_results:
                episode_idx = int(result_episode_idx)
                if text_record is not None:
                    text_records.append(text_record)
                samples.append(token_sample)
            if episode_idx is not None:
                episode_samples[episode_idx] = samples

        if job.cache_path:
            os.makedirs(job.config.cache_dir, exist_ok=True)
            cache_signature_payload = cls._cache_signature_payload(job.config)
            cache_signature_hash = cls._hash_json_payload(cache_signature_payload)
            cache = {
                "metadata": {
                    "cache_format": "pointmaze_hash_signature_v1",
                    "cache_signature_hash": cache_signature_hash,
                    "cache_signature_payload": cache_signature_payload,
                    "total_episodes": job.total_episodes,
                    "episode_indices": job.episode_indices,
                },
                "episodes": episode_samples,
            }
            with open(job.cache_path, "wb") as f:
                pickle.dump(cache, f)
            print(f"[dataset] Saved dataset cache to {job.cache_path}")
            jsonl_path = job.cache_path.replace(".pkl", ".jsonl")
            with open(jsonl_path, "w") as f:
                for record in text_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[dataset] Saved human-readable cache to {jsonl_path}")

        return episode_samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> TensorSample:
        sample = self._samples[idx]
        action_bin_labels = sample.get("action_bin_labels")
        if action_bin_labels is None:
            action_bin_labels = [-1] * len(sample["input_ids"])
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
            "action_bin_labels": torch.tensor(action_bin_labels, dtype=torch.long),
        }
