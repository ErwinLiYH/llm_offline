"""Estimate one-epoch dataset size and tokenized cache footprint.

Usage:
    python estimate_dataset.py --config config.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import pickle
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from data.base_dataset import DatasetBuildRequest
from data.pointmaze.dataset import (
    _compute_sampled_episode_target,
    _variant_sampling_seed,
    select_variant_episode_indices,
    split_episode_segments_for_partitions,
)
from data.registry import get_action_dim, get_dataset
from model.mtp_bin import (
    resolve_mtp_k,
    resolve_mtp_lcm_weight,
    resolve_mtp_quadratic_decoding,
)
from utils.action_bins import get_action_token_mode
from utils.config_loader import load_merged_config
from utils.distributed import DistributedContext
from utils.episode_keep import (
    RESOLVED_EPISODE_KEEP_PER_VARIANT_KEY,
    effective_episode_keep_num,
    has_episode_keep_per_variant,
    resolve_episode_keep_per_variant,
)
from utils.prompt_loader import load_template_names
from utils.variant_selection import get_available_variants, resolve_selection


BYTES_PER_GB = 1_000_000_000


@dataclass
class VariantData:
    variant: str
    episodes: list[Any]
    step_counts: list[int]
    selection: dict
    prompt_count: int

    @property
    def total_episodes(self) -> int:
        return int(self.selection["total_episodes"])

    @property
    def total_steps(self) -> int:
        return int(self.selection["total_steps"])

    @property
    def train_steps(self) -> int:
        return int(self.selection["train_steps"])

    @property
    def val_steps(self) -> int:
        return int(self.selection["val_steps"])

    @property
    def train_indices(self) -> list[int]:
        return [int(index) for index in self.selection["train_indices"]]

    @property
    def val_indices(self) -> list[int]:
        return [int(index) for index in self.selection["val_indices"]]


@dataclass
class SampleFootprint:
    variant: str
    sampled_episodes: int
    sampled_steps: int
    sampled_samples: int
    sampled_pickle_bytes: int
    sampled_memory_bytes: int
    sampled_tokens: int

    @property
    def bytes_per_step(self) -> float:
        if self.sampled_steps < 1:
            return 0.0
        return self.sampled_pickle_bytes / self.sampled_steps

    @property
    def memory_bytes_per_step(self) -> float:
        if self.sampled_steps < 1:
            return 0.0
        return self.sampled_memory_bytes / self.sampled_steps

    @property
    def tokens_per_step(self) -> float:
        if self.sampled_steps < 1:
            return 0.0
        return self.sampled_tokens / self.sampled_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=["config.yaml"])
    parser.add_argument(
        "--sample-episodes-per-variant",
        type=int,
        default=4,
        help="Number of complete selected episodes to tokenize per selected variant.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Episode sampling seed for the estimator; defaults to sampling_seed from config.",
    )
    parser.add_argument(
        "--world_size",
        "--world-size",
        dest="world_size",
        type=int,
        default=1,
        help="Estimated DDP world size for batch-count math only; does not initialize DDP.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final estimate as JSON. Progress from dataset builders is sent to stderr.",
    )
    return parser.parse_args()


def estimate_context(world_size: int) -> DistributedContext:
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError(f"--world_size must be >= 1, got {world_size}")
    return DistributedContext(
        backend="ddp" if world_size > 1 else "single",
        rank=0,
        world_size=world_size,
        local_rank=0,
        is_distributed=world_size > 1,
        device=torch.device("cpu"),
    )


def _normalize_prompt_names(value, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of prompt names, got {type(value).__name__}")
    names = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings, got {item!r}")
        names.append(item.strip())
    if not names:
        raise ValueError(f"{field_name} must not be empty")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate prompt names: {duplicates}")
    return names


def normalize_prompt_config(config: dict) -> None:
    primary_key = "prompt_templete_index"
    legacy_key = "prompt_template_index"
    primary_value = config.get(primary_key)
    legacy_value = config.get(legacy_key)
    if primary_value is not None and legacy_value is not None and primary_value != legacy_value:
        raise ValueError(
            f"{primary_key} and {legacy_key} both exist but differ; keep only {primary_key}."
        )

    raw_names = primary_value if primary_value is not None else legacy_value
    available_names = load_template_names(config["env_family"])
    if raw_names is None:
        prompt_template_count = int(config.get("prompt_template_count", 1))
        if prompt_template_count < 1:
            raise ValueError(f"prompt_template_count must be >= 1, got {prompt_template_count}")
        if prompt_template_count > len(available_names):
            raise ValueError(
                "prompt_template_count exceeds available templates: "
                f"requested {prompt_template_count}, available {len(available_names)}"
            )
        names = available_names[:prompt_template_count]
    else:
        names = _normalize_prompt_names(
            raw_names,
            field_name=primary_key if primary_value is not None else legacy_key,
        )

    missing = [name for name in names if name not in available_names]
    if missing:
        available = ", ".join(available_names)
        raise ValueError(
            f"Unknown prompt template names for {config['env_family']}: {missing}. "
            f"Available: {available}"
        )

    config[primary_key] = names
    config.pop(legacy_key, None)


def resolve_train_selection(config: dict, available_variants: list[str]):
    train_variants = config.get("train_varients", config.get("variants"))
    return resolve_selection(
        mode=config["train_mode"],
        variants=train_variants,
        available_variants=available_variants,
        field_name="train_varients",
    )


def resolve_dataset_load_partitions(config: dict, dist_context: DistributedContext | None = None) -> int:
    partitions = int(config.get("dataset_load_partitions", 1) or 1)
    if partitions < 1:
        raise ValueError(f"dataset_load_partitions must be >= 1, got {partitions}")
    if partitions > 1 and not config.get("dataset_cache_dir"):
        raise ValueError(
            "dataset_load_partitions > 1 requires dataset_cache_dir so train tokenized shards "
            "can be written and reloaded without keeping all samples in memory."
        )
    if partitions > 1 and dist_context is not None and dist_context.is_distributed:
        world_size = int(dist_context.world_size)
        if partitions < world_size or partitions % world_size != 0:
            raise ValueError(
                "DDP partitioned training requires dataset_load_partitions to be "
                f">= world_size and divisible by world_size; got "
                f"dataset_load_partitions={partitions}, world_size={world_size}."
            )
    return partitions


def _family_data_config(config: dict) -> dict | None:
    env_family = config.get("env_family")
    if env_family == "antmaze":
        return config.get("antmaze_data_config")
    if env_family == "pointmaze":
        return config.get("pointmaze_data_config")
    return None


def _local_dataset_root(config: dict) -> str | None:
    root = config.get("local_dataset_root")
    alias = config.get("local_dataset_path")
    if root is not None and alias is not None and root != alias:
        raise ValueError(
            "Use only one local dataset path override: local_dataset_root and "
            "local_dataset_path were both set to different values."
        )
    value = root if root is not None else alias
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(
            "local_dataset_root must be a path string or null/omitted to use variant defaults, "
            f"got {type(value).__name__}"
        )
    path_text = os.fspath(value)
    if not path_text:
        raise ValueError("local_dataset_root must not be an empty path string")
    return path_text


def build_dataset_request(
    config: dict,
    tokenizer,
    variant: str,
    split: str,
    *,
    episode_segments: list[dict] | None = None,
    episode_payloads: list[dict] | None = None,
    partition_plan_hash: str | None = None,
) -> DatasetBuildRequest:
    return DatasetBuildRequest(
        variant=variant,
        split=split,
        tokenizer=tokenizer,
        tokenizer_name_or_path=config["model_name"],
        max_length=config["max_length"],
        num_workers=config.get("dataset_workers", 8),
        cache_dir=config.get("dataset_cache_dir"),
        max_data_num=config.get("max_data_num"),
        dataset_partition_count=config.get("dataset_partition_count", 1),
        dataset_partition_index=config.get("dataset_partition_index"),
        episode_segments=episode_segments,
        episode_payloads=episode_payloads,
        partition_plan_hash=partition_plan_hash,
        prompt_template_count=config.get("prompt_template_count", 1),
        prompt_templete_index=config.get("prompt_templete_index"),
        train_data_ratio=config.get("train_data_ratio", 0.9),
        episode_keep_num=effective_episode_keep_num(config, variant),
        balance_variant_episode_count=config.get("balance_variant_episode_count", False),
        balanced_train_episode_count=config.get("balanced_train_episode_count"),
        sampling_seed=config.get("sampling_seed", 0),
        family_data_config=_family_data_config(config),
        local_dataset_root=_local_dataset_root(config),
        history_num=config.get("history_num", 0),
        history_stride=config.get("history_stride", 1),
        action_token_mode=config.get("action_token_mode", "text"),
        action_num_bins=config.get("action_num_bins", 10),
        action_bin_min=config.get("action_bin_min", -1.0),
        action_bin_max=config.get("action_bin_max", 1.0),
        new_token=config.get("new_token", False),
        action_dim=config.get("action_dim"),
        mtp_k=config.get("mtp_k"),
        progress_interval_seconds=config.get("progress_interval_seconds", 5.0),
    )


def _resolve_training_config(config: dict, world_size: int) -> tuple[dict, Any, DistributedContext]:
    if "episode_keep_ratio" in config:
        raise ValueError("episode_keep_ratio is no longer supported; use episode_keep_num instead.")

    config = dict(config)
    normalize_prompt_config(config)
    available_variants = get_available_variants(config["env_family"])
    train_selection = resolve_train_selection(config, available_variants)
    action_dim = get_action_dim(config["env_family"], train_selection.selected_variants)
    config["train_varients"] = train_selection.configured_variants
    config.pop("variants", None)
    config["action_dim"] = action_dim
    config[RESOLVED_EPISODE_KEEP_PER_VARIANT_KEY] = resolve_episode_keep_per_variant(
        config,
        train_selection.selected_variants,
        available_variants=available_variants,
    )

    action_token_mode = get_action_token_mode(config)
    if action_token_mode == "mtp_bin":
        config["mtp_k"] = resolve_mtp_k(action_dim, config.get("mtp_k"))
        config["mtp_lcm_weight"] = resolve_mtp_lcm_weight(config)
        config["mtp_quadratic_decoding"] = resolve_mtp_quadratic_decoding(config)
    elif action_token_mode == "simple_mtp_bin":
        config.pop("mtp_k", None)
        config["mtp_lcm_weight"] = resolve_mtp_lcm_weight(config)

    dist_context = estimate_context(world_size)
    config["dataset_load_partitions"] = resolve_dataset_load_partitions(config, dist_context)
    return config, train_selection, dist_context


def _balanced_train_episode_count(
    config: dict,
    selected_variants: list[str],
    loaded_by_variant: dict[str, tuple[Any, list[Any], list[int]]],
) -> int | None:
    balance_enabled = bool(config.get("balance_variant_episode_count", False))
    if not balance_enabled:
        return None
    if has_episode_keep_per_variant(config):
        print(
            "[estimate] WARNING: episode_keep_per_varient is configured; "
            "ignoring balance_variant_episode_count=true because per-variant episode_keep values take precedence.",
            file=sys.stderr,
        )
        return None
    if len(selected_variants) <= 1:
        return None
    keep_num = config.get("episode_keep_num")
    targets = [
        _compute_sampled_episode_target(len(loaded_by_variant[variant][1]), keep_num)
        for variant in selected_variants
    ]
    return min(targets)


def load_variant_data(config: dict, selected_variants: list[str]) -> list[VariantData]:
    dataset_cls = get_dataset(config["env_family"])
    family_data_config = _family_data_config(config)
    local_dataset_root = _local_dataset_root(config)
    if RESOLVED_EPISODE_KEEP_PER_VARIANT_KEY in config:
        episode_keep_by_variant = {
            variant: effective_episode_keep_num(config, variant)
            for variant in selected_variants
        }
    else:
        episode_keep_by_variant = resolve_episode_keep_per_variant(config, selected_variants)
    loaded_by_variant = {}
    for variant in selected_variants:
        meta, episodes, step_counts = dataset_cls._load_variant_episodes(
            variant,
            family_data_config=family_data_config,
            local_dataset_root=local_dataset_root,
        )
        loaded_by_variant[variant] = (meta, list(episodes), [int(count) for count in step_counts])

    balanced_target = _balanced_train_episode_count(config, selected_variants, loaded_by_variant)
    prompt_count = len(config["prompt_templete_index"])
    variant_data = []
    for variant in selected_variants:
        meta, episodes, step_counts = loaded_by_variant[variant]

        def loader(requested_variant: str, family_data_config: dict | None = None):
            if requested_variant != variant:
                raise ValueError(f"Unexpected variant request {requested_variant!r}; expected {variant!r}")
            return meta, episodes, step_counts

        selection = select_variant_episode_indices(
            variant=variant,
            train_data_ratio=config.get("train_data_ratio", 0.9),
            episode_keep_num=episode_keep_by_variant[variant],
            sampling_seed=config.get("sampling_seed", 0),
            balanced_train_target=balanced_target,
            episode_loader=loader,
            family_data_config=family_data_config,
            local_dataset_root=local_dataset_root,
        )
        variant_data.append(
            VariantData(
                variant=variant,
                episodes=episodes,
                step_counts=step_counts,
                selection=selection,
                prompt_count=prompt_count,
            )
        )
    return variant_data


def _split_sample_count(steps: int, prompt_count: int, max_data_num: int | None) -> int:
    samples = int(steps) * int(prompt_count)
    if max_data_num is not None:
        samples = min(samples, int(max_data_num))
    return samples


def _effective_steps_for_size(steps: int, prompt_count: int, max_data_num: int | None) -> float:
    if max_data_num is None:
        return float(steps)
    return min(float(steps), float(max_data_num) / float(prompt_count))


def estimate_bytes_for_steps(sampled_bytes: int, sampled_steps: int, target_steps: float) -> float:
    if sampled_steps < 1:
        raise ValueError("sampled_steps must be >= 1")
    return float(sampled_bytes) * float(target_steps) / float(sampled_steps)


def deep_getsizeof(obj: Any, seen: set[int] | None = None) -> int:
    """Return recursive CPython object size, counting shared references once."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset, deque)):
        size += sum(deep_getsizeof(item, seen) for item in obj)
    elif isinstance(obj, np.ndarray):
        if not obj.flags.owndata:
            size += int(obj.nbytes)
    elif isinstance(obj, torch.Tensor):
        size += int(obj.element_size() * obj.nelement())
    elif hasattr(obj, "__dict__"):
        size += deep_getsizeof(vars(obj), seen)
    return int(size)


def _ceil_div(numerator: int, denominator: int) -> int:
    return int(math.ceil(int(numerator) / int(denominator)))


def estimate_epoch_batches(
    variant_data: list[VariantData],
    config: dict,
    *,
    partition_count: int,
    world_size: int,
) -> dict:
    batch_size = int(config["batch_size"])
    prompt_count = len(config["prompt_templete_index"])
    max_data_num = config.get("max_data_num")

    if partition_count <= 1:
        selected_train_samples = sum(
            _split_sample_count(data.train_steps, prompt_count, max_data_num)
            for data in variant_data
        )
        per_rank_samples = (
            _ceil_div(selected_train_samples, world_size)
            if world_size > 1
            else selected_train_samples
        )
        train_batches = _ceil_div(per_rank_samples, batch_size) if per_rank_samples else 0
        return {
            "partitioned": False,
            "selected_train_samples": int(selected_train_samples),
            "per_rank_samples_per_epoch": int(per_rank_samples),
            "sampler_samples_per_epoch": int(per_rank_samples * world_size),
            "train_batches_per_epoch": int(train_batches),
            "partition_stats": [],
            "round_stats": [],
        }

    partition_variant_segments: dict[str, list[list[dict]]] = {}
    for data in variant_data:
        partition_variant_segments[data.variant] = split_episode_segments_for_partitions(
            data.train_indices,
            data.step_counts,
            partition_count=partition_count,
            prompt_count=prompt_count,
            variant=data.variant,
            sampling_seed=int(config.get("sampling_seed", 0)),
        )

    partition_stats = []
    selected_train_samples = 0
    for partition_index in range(partition_count):
        shard_steps = 0
        shard_samples = 0
        for data in variant_data:
            segments = partition_variant_segments[data.variant][partition_index]
            variant_steps = sum(int(segment["step_count"]) for segment in segments)
            variant_samples = sum(int(segment["sample_count"]) for segment in segments)
            if max_data_num is not None:
                variant_samples = min(variant_samples, int(max_data_num))
            shard_steps += variant_steps
            shard_samples += variant_samples
        selected_train_samples += shard_samples
        partition_stats.append(
            {
                "partition_index": int(partition_index),
                "train_steps": int(shard_steps),
                "train_samples": int(shard_samples),
                "train_batches": _ceil_div(shard_samples, batch_size),
            }
        )

    round_stats = []
    per_rank_samples_per_epoch = 0
    for round_index, start in enumerate(range(0, partition_count, world_size)):
        shard_stats = partition_stats[start : start + world_size]
        if len(shard_stats) != world_size:
            raise ValueError(
                "dataset_load_partitions must be divisible by --world_size for DDP batch estimation: "
                f"partitions={partition_count}, world_size={world_size}"
            )
        target_batches = max(int(stat["train_batches"]) for stat in shard_stats)
        target_samples = target_batches * batch_size
        per_rank_samples_per_epoch += target_samples
        round_stats.append(
            {
                "round_index": int(round_index),
                "partition_indices": [int(stat["partition_index"]) for stat in shard_stats],
                "target_batches": int(target_batches),
                "target_samples_per_rank": int(target_samples),
                "shard_train_samples": [int(stat["train_samples"]) for stat in shard_stats],
                "shard_train_batches": [int(stat["train_batches"]) for stat in shard_stats],
            }
        )

    return {
        "partitioned": True,
        "selected_train_samples": int(selected_train_samples),
        "per_rank_samples_per_epoch": int(per_rank_samples_per_epoch),
        "sampler_samples_per_epoch": int(per_rank_samples_per_epoch * world_size),
        "train_batches_per_epoch": int(sum(stat["target_batches"] for stat in round_stats)),
        "partition_stats": partition_stats,
        "round_stats": round_stats,
    }


def _sample_episode_indices(data: VariantData, count: int, sample_seed: int) -> list[int]:
    candidates = sorted(set(data.train_indices + data.val_indices))
    candidates = [index for index in candidates if int(data.step_counts[index]) > 0]
    if not candidates:
        raise ValueError(f"Variant {data.variant!r} has no selected non-empty episodes to sample.")
    if len(candidates) <= count:
        return candidates
    rng = np.random.default_rng(_variant_sampling_seed(f"{data.variant}:estimate", sample_seed))
    chosen = rng.choice(candidates, size=count, replace=False).tolist()
    return sorted(int(index) for index in chosen)


def tokenize_sample_footprint(
    config: dict,
    tokenizer,
    data: VariantData,
    *,
    sample_episodes_per_variant: int,
    sample_seed: int,
) -> SampleFootprint:
    if sample_episodes_per_variant < 1:
        raise ValueError(
            "--sample-episodes-per-variant must be >= 1, "
            f"got {sample_episodes_per_variant}"
        )
    dataset_cls = get_dataset(config["env_family"])
    episode_indices = _sample_episode_indices(data, sample_episodes_per_variant, sample_seed)
    segments = []
    for segment_key, episode_idx in enumerate(episode_indices):
        step_count = int(data.step_counts[episode_idx])
        segments.append(
            {
                "segment_key": int(segment_key),
                "episode_idx": int(episode_idx),
                "start_t": 0,
                "end_t": step_count,
                "episode_len": step_count,
                "step_count": step_count,
                "sample_count": step_count * data.prompt_count,
                "variant": data.variant,
            }
        )
    payloads = dataset_cls.payloads_for_segments(data.episodes, segments)

    sample_config = dict(config)
    sample_config["dataset_cache_dir"] = None
    sample_config["dataset_partition_count"] = 2
    sample_config["dataset_partition_index"] = 0
    sample_config["max_data_num"] = None
    request = build_dataset_request(
        sample_config,
        tokenizer,
        data.variant,
        "train",
        episode_segments=segments,
        episode_payloads=payloads,
        partition_plan_hash="estimate",
    )
    dataset = dataset_cls.build_batch([request])[0]
    samples = list(getattr(dataset, "_samples"))
    cache_like = {
        "metadata": {
            "estimated": True,
            "variant": data.variant,
            "sampled_episode_indices": episode_indices,
        },
        "samples": samples,
    }
    sampled_pickle_bytes = len(pickle.dumps(cache_like))
    sampled_memory_bytes = deep_getsizeof(samples)
    sampled_tokens = sum(len(sample["input_ids"]) for sample in samples)
    sampled_steps = sum(int(segment["step_count"]) for segment in segments)
    return SampleFootprint(
        variant=data.variant,
        sampled_episodes=len(episode_indices),
        sampled_steps=int(sampled_steps),
        sampled_samples=len(samples),
        sampled_pickle_bytes=int(sampled_pickle_bytes),
        sampled_memory_bytes=int(sampled_memory_bytes),
        sampled_tokens=int(sampled_tokens),
    )


def _train_effective_steps_for_size(data: VariantData, config: dict, partition_count: int) -> float:
    prompt_count = len(config["prompt_templete_index"])
    max_data_num = config.get("max_data_num")
    if partition_count <= 1:
        return _effective_steps_for_size(data.train_steps, prompt_count, max_data_num)

    shards = split_episode_segments_for_partitions(
        data.train_indices,
        data.step_counts,
        partition_count=partition_count,
        prompt_count=prompt_count,
        variant=data.variant,
        sampling_seed=int(config.get("sampling_seed", 0)),
    )
    effective_steps = 0.0
    for segments in shards:
        shard_steps = sum(int(segment["step_count"]) for segment in segments)
        effective_steps += _effective_steps_for_size(shard_steps, prompt_count, max_data_num)
    return effective_steps


def _partition_train_memory_bytes(
    variant_data: list[VariantData],
    footprints_by_variant: dict[str, SampleFootprint],
    config: dict,
    partition_count: int,
) -> list[float]:
    if partition_count <= 1:
        return []

    prompt_count = len(config["prompt_templete_index"])
    max_data_num = config.get("max_data_num")
    partition_bytes = [0.0 for _ in range(partition_count)]
    for data in variant_data:
        footprint = footprints_by_variant[data.variant]
        shards = split_episode_segments_for_partitions(
            data.train_indices,
            data.step_counts,
            partition_count=partition_count,
            prompt_count=prompt_count,
            variant=data.variant,
            sampling_seed=int(config.get("sampling_seed", 0)),
        )
        for partition_index, segments in enumerate(shards):
            shard_steps = sum(int(segment["step_count"]) for segment in segments)
            effective_steps = _effective_steps_for_size(shard_steps, prompt_count, max_data_num)
            partition_bytes[partition_index] += estimate_bytes_for_steps(
                footprint.sampled_memory_bytes,
                footprint.sampled_steps,
                effective_steps,
            )
    return partition_bytes


def build_estimate(
    config: dict,
    variant_data: list[VariantData],
    footprints: list[SampleFootprint],
    *,
    partition_count: int,
    world_size: int,
    sample_seed: int,
) -> dict:
    prompt_count = len(config["prompt_templete_index"])
    max_data_num = config.get("max_data_num")
    footprints_by_variant = {footprint.variant: footprint for footprint in footprints}
    variants = []
    total_train_bytes = 0.0
    total_val_bytes = 0.0
    total_sampled_steps = 0
    total_sampled_samples = 0
    total_sampled_tokens = 0
    total_sampled_bytes = 0
    total_sampled_memory_bytes = 0
    total_train_memory_bytes = 0.0
    total_val_memory_bytes = 0.0

    for data in variant_data:
        footprint = footprints_by_variant[data.variant]
        train_effective_steps = _train_effective_steps_for_size(data, config, partition_count)
        val_effective_steps = _effective_steps_for_size(data.val_steps, prompt_count, max_data_num)
        train_bytes = estimate_bytes_for_steps(
            footprint.sampled_pickle_bytes,
            footprint.sampled_steps,
            train_effective_steps,
        )
        val_bytes = estimate_bytes_for_steps(
            footprint.sampled_pickle_bytes,
            footprint.sampled_steps,
            val_effective_steps,
        )
        train_memory_bytes = estimate_bytes_for_steps(
            footprint.sampled_memory_bytes,
            footprint.sampled_steps,
            train_effective_steps,
        )
        val_memory_bytes = estimate_bytes_for_steps(
            footprint.sampled_memory_bytes,
            footprint.sampled_steps,
            val_effective_steps,
        )
        total_train_bytes += train_bytes
        total_val_bytes += val_bytes
        total_train_memory_bytes += train_memory_bytes
        total_val_memory_bytes += val_memory_bytes
        total_sampled_steps += footprint.sampled_steps
        total_sampled_samples += footprint.sampled_samples
        total_sampled_tokens += footprint.sampled_tokens
        total_sampled_bytes += footprint.sampled_pickle_bytes
        total_sampled_memory_bytes += footprint.sampled_memory_bytes
        variants.append(
            {
                "variant": data.variant,
                "total_episodes": data.total_episodes,
                "total_steps": data.total_steps,
                "sampled_episode_count": int(data.selection["sampled_episode_count"]),
                "train_episodes": int(data.selection["train_episode_count"]),
                "train_steps": data.train_steps,
                "val_episodes": int(data.selection["val_episode_count"]),
                "val_steps": data.val_steps,
                "train_samples": _split_sample_count(data.train_steps, prompt_count, max_data_num),
                "val_samples": _split_sample_count(data.val_steps, prompt_count, max_data_num),
                "size_target_train_steps": train_effective_steps,
                "size_target_val_steps": val_effective_steps,
                "sampled_episodes": footprint.sampled_episodes,
                "sampled_steps": footprint.sampled_steps,
                "sampled_samples": footprint.sampled_samples,
                "sampled_pickle_bytes": footprint.sampled_pickle_bytes,
                "sampled_memory_bytes": footprint.sampled_memory_bytes,
                "sampled_tokens": footprint.sampled_tokens,
                "bytes_per_step": footprint.bytes_per_step,
                "memory_bytes_per_step": footprint.memory_bytes_per_step,
                "tokens_per_step": footprint.tokens_per_step,
                "estimated_train_bytes": train_bytes,
                "estimated_val_bytes": val_bytes,
                "estimated_train_gb": train_bytes / BYTES_PER_GB,
                "estimated_val_gb": val_bytes / BYTES_PER_GB,
                "estimated_train_memory_bytes": train_memory_bytes,
                "estimated_val_memory_bytes": val_memory_bytes,
                "estimated_train_memory_gb": train_memory_bytes / BYTES_PER_GB,
                "estimated_val_memory_gb": val_memory_bytes / BYTES_PER_GB,
            }
        )

    batch_estimate = estimate_epoch_batches(
        variant_data,
        config,
        partition_count=partition_count,
        world_size=world_size,
    )
    partition_memory_bytes = _partition_train_memory_bytes(
        variant_data,
        footprints_by_variant,
        config,
        partition_count,
    )
    peak_train_partition_bytes = (
        max(partition_memory_bytes) if partition_memory_bytes else total_train_memory_bytes
    )
    return {
        "config": {
            "env_family": config["env_family"],
            "model_name": config["model_name"],
            "action_token_mode": get_action_token_mode(config),
            "prompt_templete_index": list(config["prompt_templete_index"]),
            "prompt_count": prompt_count,
            "dataset_load_partitions": int(partition_count),
            "world_size": int(world_size),
            "batch_size": int(config["batch_size"]),
            "max_data_num": max_data_num,
            "sample_seed": int(sample_seed),
        },
        "variants": variants,
        "sampling": {
            "sampled_steps": int(total_sampled_steps),
            "sampled_samples": int(total_sampled_samples),
            "sampled_tokens": int(total_sampled_tokens),
            "sampled_pickle_bytes": int(total_sampled_bytes),
            "sampled_memory_bytes": int(total_sampled_memory_bytes),
            "tokens_per_step": (
                float(total_sampled_tokens) / float(total_sampled_steps)
                if total_sampled_steps
                else 0.0
            ),
            "bytes_per_step": (
                float(total_sampled_bytes) / float(total_sampled_steps)
                if total_sampled_steps
                else 0.0
            ),
            "memory_bytes_per_step": (
                float(total_sampled_memory_bytes) / float(total_sampled_steps)
                if total_sampled_steps
                else 0.0
            ),
        },
        "epoch": batch_estimate,
        "size": {
            "train_bytes": total_train_bytes,
            "val_bytes": total_val_bytes,
            "total_bytes": total_train_bytes + total_val_bytes,
            "train_gb": total_train_bytes / BYTES_PER_GB,
            "val_gb": total_val_bytes / BYTES_PER_GB,
            "total_gb": (total_train_bytes + total_val_bytes) / BYTES_PER_GB,
        },
        "memory": {
            "method": "recursive sys.getsizeof(dataset._samples); shared objects counted once",
            "scope": (
                "loaded tokenized sample Python objects only; excludes raw episodes, tokenizer, "
                "DataLoader worker copies/prefetch batches, model, optimizer, gradients, and activations"
            ),
            "train_bytes": total_train_memory_bytes,
            "val_bytes": total_val_memory_bytes,
            "total_bytes": total_train_memory_bytes + total_val_memory_bytes,
            "train_gb": total_train_memory_bytes / BYTES_PER_GB,
            "val_gb": total_val_memory_bytes / BYTES_PER_GB,
            "total_gb": (total_train_memory_bytes + total_val_memory_bytes) / BYTES_PER_GB,
            "peak_train_partition_bytes": peak_train_partition_bytes,
            "peak_train_partition_gb": peak_train_partition_bytes / BYTES_PER_GB,
            "peak_train_partition_plus_val_bytes": (
                peak_train_partition_bytes + total_val_memory_bytes
            ),
            "peak_train_partition_plus_val_gb": (
                peak_train_partition_bytes + total_val_memory_bytes
            )
            / BYTES_PER_GB,
            "partition_train_bytes": partition_memory_bytes,
            "partition_train_gb": [value / BYTES_PER_GB for value in partition_memory_bytes],
        },
    }


def _format_gb(bytes_value: float) -> str:
    return f"{bytes_value / BYTES_PER_GB:.4f} GB"


def print_text_report(estimate: dict) -> None:
    cfg = estimate["config"]
    size = estimate["size"]
    memory = estimate["memory"]
    epoch = estimate["epoch"]
    sampling = estimate["sampling"]
    print("=" * 112)
    print("[estimate] DATASET SIZE ESTIMATE")
    print(
        f"[estimate] env_family={cfg['env_family']}, model={cfg['model_name']}, "
        f"action_token_mode={cfg['action_token_mode']}"
    )
    print(
        f"[estimate] prompts={cfg['prompt_templete_index']} "
        f"(prompt_count={cfg['prompt_count']}), partitions={cfg['dataset_load_partitions']}, "
        f"world_size={cfg['world_size']}"
    )
    if cfg["max_data_num"] is not None:
        print(
            "[estimate] max_data_num is set; sample counts and size targets are capped "
            "the same way as dataset construction."
        )
    print("-" * 112)
    header = (
        "variant",
        "train_steps",
        "val_steps",
        "sample_steps",
        "tokens/step",
        "pkl_B/step",
        "mem_B/step",
        "pkl_train_GB",
        "mem_train_GB",
    )
    print(
        f"{header[0]:<22} {header[1]:>12} {header[2]:>10} {header[3]:>12} "
        f"{header[4]:>12} {header[5]:>12} {header[6]:>12} {header[7]:>12} "
        f"{header[8]:>12}"
    )
    for row in estimate["variants"]:
        print(
            f"{row['variant']:<22} {row['train_steps']:>12} {row['val_steps']:>10} "
            f"{row['sampled_steps']:>12} {row['tokens_per_step']:>12.2f} "
            f"{row['bytes_per_step']:>12.2f} {row['memory_bytes_per_step']:>12.2f} "
            f"{row['estimated_train_gb']:>12.4f} {row['estimated_train_memory_gb']:>12.4f}"
        )
    print("-" * 112)
    print(
        "[estimate] One epoch: "
        f"selected_train_samples={epoch['selected_train_samples']}, "
        f"sampler_samples_per_epoch={epoch['sampler_samples_per_epoch']}, "
        f"per_rank_samples={epoch['per_rank_samples_per_epoch']}, "
        f"train_batches_per_epoch={epoch['train_batches_per_epoch']}"
    )
    print(
        "[estimate] Sampled for tokenization: "
        f"steps={sampling['sampled_steps']}, samples={sampling['sampled_samples']}, "
        f"tokens/step={sampling['tokens_per_step']:.2f}, "
        f"pkl_bytes/step={sampling['bytes_per_step']:.2f}, "
        f"memory_bytes/step={sampling['memory_bytes_per_step']:.2f}"
    )
    print(
        "[estimate] Tokenized .pkl size estimate: "
        f"train={_format_gb(size['train_bytes'])}, "
        f"val={_format_gb(size['val_bytes'])}, "
        f"total={_format_gb(size['total_bytes'])}"
    )
    print(
        "[estimate] Python in-memory tokenized samples estimate: "
        f"train={_format_gb(memory['train_bytes'])}, "
        f"val={_format_gb(memory['val_bytes'])}, "
        f"total={_format_gb(memory['total_bytes'])}"
    )
    if cfg["dataset_load_partitions"] > 1:
        print(
            "[estimate] Partitioned training resident estimate: "
            f"peak_train_shard={_format_gb(memory['peak_train_partition_bytes'])}, "
            f"peak_train_shard_plus_val={_format_gb(memory['peak_train_partition_plus_val_bytes'])}"
        )
    print(
        "[estimate] Memory scope: tokenized dataset._samples Python objects only; excludes raw data, "
        "tokenizer/model/optimizer, DataLoader worker copies, prefetch batches, and activations."
    )
    print("=" * 112)


def main() -> None:
    args = parse_args()
    raw_config = load_merged_config(args.config)
    raw_config["estimate_config_source"] = (
        args.config[0] if len(args.config) == 1 else list(args.config)
    )
    raw_config["config_sources"] = list(args.config)
    config, train_selection, dist_context = _resolve_training_config(raw_config, args.world_size)
    sample_seed = int(args.sample_seed if args.sample_seed is not None else config.get("sampling_seed", 0))
    partition_count = int(config["dataset_load_partitions"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
    )
    variant_data = load_variant_data(config, train_selection.selected_variants)

    footprints = []
    output_redirect = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    with output_redirect:
        for data in variant_data:
            footprints.append(
                tokenize_sample_footprint(
                    config,
                    tokenizer,
                    data,
                    sample_episodes_per_variant=int(args.sample_episodes_per_variant),
                    sample_seed=sample_seed,
                )
            )

    estimate = build_estimate(
        config,
        variant_data,
        footprints,
        partition_count=partition_count,
        world_size=dist_context.world_size,
        sample_seed=sample_seed,
    )
    if args.json:
        print(json.dumps(estimate, ensure_ascii=False, indent=2))
    else:
        print_text_report(estimate)


if __name__ == "__main__":
    main()
