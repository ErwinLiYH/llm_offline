import json
import multiprocessing
import os
import pickle
import hashlib
import math
import time
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
from utils.action_bins import (
    get_action_bin_range,
    get_action_bin_token_ids,
    get_action_num_bins,
    get_action_token_mode,
    register_action_tokens,
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

_POINTMAZE_WORKER_TOKENIZER = None
_POINTMAZE_WORKER_CONFIGS: dict[str, dict] | None = None
_POINTMAZE_WORKER_SHARED_CONFIG: dict | None = None
_POINTMAZE_WORKER_TOKEN_ID_TO_BIN: dict[int, int] | None = None
_POINTMAZE_WORKER_DONE = 0
_POINTMAZE_WORKER_START_TIME = 0.0


def _pointmaze_action_config(config: dict) -> dict:
    return {
        "action_token_mode": config["action_token_mode"],
        "action_num_bins": config["action_num_bins"],
        "action_bin_min": config["action_bin_min"],
        "action_bin_max": config["action_bin_max"],
    }


def _init_pointmaze_tokenization_worker(progress_initializer, progress_initargs: tuple, worker_config: dict):
    global _POINTMAZE_WORKER_TOKENIZER
    global _POINTMAZE_WORKER_CONFIGS
    global _POINTMAZE_WORKER_SHARED_CONFIG
    global _POINTMAZE_WORKER_TOKEN_ID_TO_BIN
    global _POINTMAZE_WORKER_DONE
    global _POINTMAZE_WORKER_START_TIME

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
        register_action_tokens(_POINTMAZE_WORKER_TOKENIZER, action_config)
        _POINTMAZE_WORKER_TOKEN_ID_TO_BIN = {
            token_id: bin_idx
            for bin_idx, token_id in enumerate(
                get_action_bin_token_ids(_POINTMAZE_WORKER_TOKENIZER, action_config)
            )
        }
    else:
        _POINTMAZE_WORKER_TOKEN_ID_TO_BIN = None


def _format_pointmaze_action_for_mode(action: np.ndarray, config: dict) -> str:
    if config["action_token_mode"] == "text":
        return formatting.format_action(action)
    return formatting.format_action_bin_tokens(
        action,
        num_bins=config["action_num_bins"],
        low=config["action_bin_min"],
        high=config["action_bin_max"],
    )


def _tokenize_pointmaze_sample(prompt: str, action_text: str, config: dict) -> dict:
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
        token_id_to_bin = _POINTMAZE_WORKER_TOKEN_ID_TO_BIN or {}
        for pos in range(min(prompt_len, len(input_ids)), len(input_ids)):
            bin_idx = token_id_to_bin.get(input_ids[pos])
            if bin_idx is not None:
                action_bin_labels[pos] = bin_idx

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "action_bin_labels": action_bin_labels,
    }


def _process_pointmaze_episode(payload: dict) -> list[tuple[dict, dict]]:
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
        action_text = _format_pointmaze_action_for_mode(action, config)
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
                        "action_text": _format_pointmaze_action_for_mode(
                            actions[hist_t].astype(np.float32),
                            config,
                        ),
                        "steps_ago": t - hist_t,
                    }
                )
        obs_payload = formatting.format_obs(obs, prompt_vars)
        history_payload = formatting.format_history(history_entries, prompt_vars)
        for template in templates:
            prompt = render_template(template, prompt_vars, **obs_payload, **history_payload)
            token_sample = _tokenize_pointmaze_sample(prompt, action_text, config)
            text_record = {"prompt": prompt, "action": action_text}
            results.append((text_record, token_sample))

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
    progress_interval_seconds: float


@dataclass
class PointMazeTokenizationJob:
    job_id: str
    dataset: "PointMazeDataset"
    config: PointMazeBuildConfig
    cache_path: str | None
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
        datasets: list[PointMazeDataset] = []
        jobs: list[PointMazeTokenizationJob] = []

        for config in configs:
            cache_path = cls._cache_path(config)
            cached_dataset = cls._load_cached_dataset(config, cache_path)
            if cached_dataset is not None:
                datasets.append(cached_dataset)
                continue

            dataset, job = cls._create_tokenization_job(config, cache_path)
            datasets.append(dataset)
            jobs.append(job)

        if jobs:
            num_workers = max(job.config.num_workers for job in jobs)
            progress_interval_seconds = min(job.config.progress_interval_seconds for job in jobs)
            results_by_job = cls._execute_tokenization_jobs(
                jobs,
                num_workers=num_workers,
                progress_interval_seconds=progress_interval_seconds,
            )
            for job in jobs:
                cls._finalize_tokenization_job(job, results_by_job[job.job_id])

        return datasets

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
        }
        action_token_mode = get_action_token_mode(action_config)
        action_num_bins = get_action_num_bins(action_config)
        action_bin_min, action_bin_max = get_action_bin_range(action_config)
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

    @classmethod
    def _prompt_cache_tag(cls, config: PointMazeBuildConfig) -> str:
        prompt_names = cls._resolve_prompt_names(config)
        joined = "+".join(prompt_names)
        if len(joined) <= 80 and all(ch.isalnum() or ch in "._+-" for ch in joined):
            return joined
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
        return f"hash-{digest}"

    @staticmethod
    def _split_tag(config: PointMazeBuildConfig) -> str:
        train_pct = int(round(config.train_data_ratio * 100))
        return f"split{train_pct:02d}"

    @classmethod
    def _cache_path(cls, config: PointMazeBuildConfig) -> str | None:
        if config.cache_dir is None:
            return None
        meta = POINTMAZE_VARIANTS[config.variant]
        data_signature = ""
        if get_pointmaze_variant_type(meta) == "local":
            data_signature = f"-{_local_dataset_step_signature(meta)}"
        fname = (
            f"pointmaze-{config.variant}-{config.split}{data_signature}-"
            f"prompts-{cls._prompt_cache_tag(config)}-"
            f"hist{config.history_num}-stride{config.history_stride}-{cls._split_tag(config)}-"
            f"action-{config.action_token_mode}-bins{config.action_num_bins}-"
            f"range{config.action_bin_min:g}to{config.action_bin_max:g}.pkl"
        )
        return os.path.join(config.cache_dir, fname)

    @classmethod
    def _load_cached_dataset(
        cls,
        config: PointMazeBuildConfig,
        cache_path: str | None,
    ) -> "PointMazeDataset | None":
        if not cache_path or not os.path.exists(cache_path):
            return None
        print(f"[dataset] Loading cached dataset from {cache_path}")
        print(
            "[dataset] Cached split bypasses episode sampling settings: "
            f"episode_keep_num={config.episode_keep_num}, "
            f"balance_variant_episode_count={config.balance_variant_episode_count}, "
            f"sampling_seed={config.sampling_seed}. This run did not apply them."
        )
        with open(cache_path, "rb") as f:
            samples = pickle.load(f)
        if config.max_data_num is not None:
            samples = samples[: config.max_data_num]
            print(f"[dataset] max_data_num={config.max_data_num}: using {len(samples)} samples")
        return cls(config.variant, config.split, samples)

    @classmethod
    def _create_tokenization_job(
        cls,
        config: PointMazeBuildConfig,
        cache_path: str | None,
    ) -> tuple["PointMazeDataset", PointMazeTokenizationJob]:
        cls._validate_config(config)

        meta = POINTMAZE_VARIANTS[config.variant]
        prompt_names = cls._resolve_prompt_names(config)
        templates = load_named_templates("pointmaze", prompt_names)
        prompt_vars = meta["prompt_vars"]

        selection = select_variant_episode_indices(
            variant=config.variant,
            train_data_ratio=config.train_data_ratio,
            episode_keep_num=config.episode_keep_num,
            sampling_seed=config.sampling_seed,
            balanced_train_target=config.balanced_train_episode_count,
        )
        all_episodes = selection["episodes"]
        cls._print_selection_summary(config, selection)

        if config.split == "train":
            episodes = [all_episodes[idx] for idx in selection["train_indices"]]
        elif config.split == "val":
            episodes = [all_episodes[idx] for idx in selection["val_indices"]]
        else:
            raise ValueError(f"Unknown split: {config.split!r}. Expected 'train' or 'val'.")

        job_id = f"{config.variant}:{config.split}:{len(episodes)}:{id(config)}"
        episode_payloads = [
            {
                "job_id": job_id,
                "episode_idx": idx,
                "observations": episode.observations["observation"],
                "goals": episode.observations["desired_goal"],
                "actions": episode.actions,
            }
            for idx, episode in enumerate(episodes)
        ]
        worker_config = {
            "variant": config.variant,
            "split": config.split,
            "max_length": config.max_length,
            "templates": templates,
            "prompt_vars": prompt_vars,
            "history_num": config.history_num,
            "history_stride": config.history_stride,
            "action_token_mode": config.action_token_mode,
            "action_num_bins": config.action_num_bins,
            "action_bin_min": config.action_bin_min,
            "action_bin_max": config.action_bin_max,
        }
        shared_config = {
            "tokenizer_name_or_path": config.tokenizer_name_or_path,
            "action_token_mode": config.action_token_mode,
            "action_num_bins": config.action_num_bins,
            "action_bin_min": config.action_bin_min,
            "action_bin_max": config.action_bin_max,
        }
        dataset = cls(config.variant, config.split, [])
        job = PointMazeTokenizationJob(
            job_id=job_id,
            dataset=dataset,
            config=config,
            cache_path=cache_path,
            episode_payloads=episode_payloads,
            worker_config=worker_config,
            shared_config=shared_config,
        )
        return dataset, job

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

    @classmethod
    def _execute_tokenization_jobs(
        cls,
        jobs: list[PointMazeTokenizationJob],
        *,
        num_workers: int,
        progress_interval_seconds: float = 5.0,
    ) -> dict[str, list[list[tuple[dict, dict]]]]:
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
        results_by_job: dict[str, list[list[tuple[dict, dict]]]] = {
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
                futures = list(
                    executor.map(
                        _process_pointmaze_episode,
                        episode_payloads,
                        chunksize=chunksize,
                    )
                )
        for payload, episode_results in zip(episode_payloads, futures):
            results_by_job[str(payload["job_id"])].append(episode_results)
        return results_by_job

    @classmethod
    def _finalize_tokenization_job(
        cls,
        job: PointMazeTokenizationJob,
        episode_results_list: list[list[tuple[dict, dict]]],
    ):
        samples = []
        text_records = []
        for episode_results in episode_results_list:
            for text_record, token_sample in episode_results:
                text_records.append(text_record)
                samples.append(token_sample)

        if job.cache_path:
            os.makedirs(job.config.cache_dir, exist_ok=True)
            with open(job.cache_path, "wb") as f:
                pickle.dump(samples, f)
            print(f"[dataset] Saved dataset cache to {job.cache_path}")
            jsonl_path = job.cache_path.replace(".pkl", ".jsonl")
            with open(jsonl_path, "w") as f:
                for record in text_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[dataset] Saved human-readable cache to {jsonl_path}")

        if job.config.max_data_num is not None:
            samples = samples[: job.config.max_data_num]
            print(f"[dataset] max_data_num={job.config.max_data_num}: using {len(samples)} samples")
        job.dataset._samples = samples

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
