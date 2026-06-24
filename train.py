"""Training entry point for LLM offline RL (behavior cloning).

Usage:
    python train.py --config config.yaml
"""

import argparse
import contextlib
import io
import uuid
import os
import json
import time
import math
import sys
import gc
import subprocess

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

with contextlib.redirect_stdout(io.StringIO()):
    from unsloth import FastLanguageModel

from data.base_dataset import DatasetBuildRequest
from data.registry import get_action_dim, get_dataset
from model.continuous_action import (
    ensure_continuous_action_decoder,
    resolve_action_head_dropout,
    resolve_gaussian_log_std_bounds,
    resolve_gaussian_log_std_init,
    resolve_student_t_df,
    resolve_action_head_num_blocks,
    resolve_action_query_len,
    save_continuous_action_decoder,
    squashed_gaussian_negative_log_likelihood,
    student_t_negative_log_likelihood,
    unpatch_continuous_action_forward,
)
from model.mtp_bin import (
    ensure_mtp_bin_decoder,
    mtp_bin_action_loss,
    mtp_bin_equivalent_l1,
    mtp_bin_equivalent_l1_by_path,
    resolve_mtp_k,
    resolve_mtp_lcm_weight,
    resolve_mtp_quadratic_decoding,
    save_mtp_bin_decoder,
    unpatch_mtp_bin_forward,
    uses_mtp_bin,
)
from model.policy import (
    load_model_and_tokenizer,
    load_model_and_tokenizer_for_training_checkpoint,
    get_model_slug,
)
from utils.action_bins import (
    action_bin_equivalent_l1,
    gaussian_action_loss,
    get_action_bin_range,
    get_action_bin_token_ids,
    get_action_num_bins,
    get_action_token_mode,
    uses_continuous_actions,
)
from utils.distributed import (
    DistributedContext,
    all_gather_objects,
    barrier,
    broadcast_object,
    cleanup_distributed,
    init_distributed_context,
    rank_zero_print,
    reduce_mean,
    resolve_parallel_backend,
    scatter_object,
    unwrap_model,
)
from utils.distributed_sampler import DistributedWeightedSampler, LocalShardPaddingSampler
from utils.experiment_config import save_experiment_config_snapshot
from utils.file_progress import FileProgress
from utils.lr_scheduler import (
    get_optimizer_lr,
    lr_scale_for_step,
    normalize_lr_scheduler_type,
    resolve_lr_decay_steps,
    resolve_min_lr_ratio,
    resolve_warmup_steps,
    set_optimizer_lr,
)
from utils.prompt_loader import load_template_names
from utils.resource_monitor import ResourceMonitor, resource_monitor_path
from utils.variant_selection import resolve_selection, VariantSelection, get_available_variants
from utils.wandb_logging import (
    WandbLogger,
    global_batch_sample_count,
    init_wandb_logger,
    prompt_template_multiplier,
    wandb_enabled,
    wandb_log_interval,
    wandb_step_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--parallel_backend", type=str, choices=["single", "ddp"], default=None)
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Override experiment_id from the config; useful for scheduler job ids.",
    )
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="Build/load all selected train/val tokenized dataset caches, then exit before training.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory containing trainer_state.pt.",
    )
    return parser.parse_args()


def _as_bool(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Expected a boolean value, got {value!r}")
    return bool(value)


def resolve_dataloader_config(config: dict) -> dict:
    raw_config = config.get("dataloader_config")
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError("dataloader_config must be a mapping")

    allowed_keys = {
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "non_blocking",
    }
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown dataloader_config keys: {unknown_keys}. "
            f"Supported keys: {sorted(allowed_keys)}"
        )

    raw_num_workers = raw_config.get("num_workers", 0)
    if isinstance(raw_num_workers, bool):
        raise ValueError("dataloader_config.num_workers must be an integer >= 0")
    num_workers = int(raw_num_workers)
    if num_workers < 0:
        raise ValueError(
            f"dataloader_config.num_workers must be >= 0, got {num_workers}"
        )

    pin_memory = _as_bool(raw_config.get("pin_memory", False))
    persistent_workers = _as_bool(raw_config.get("persistent_workers", False))
    non_blocking = _as_bool(raw_config.get("non_blocking", False))

    raw_prefetch_factor = raw_config.get("prefetch_factor")
    prefetch_factor = None
    if raw_prefetch_factor is not None:
        if isinstance(raw_prefetch_factor, bool):
            raise ValueError(
                "dataloader_config.prefetch_factor must be an integer >= 1 or null"
            )
        prefetch_factor = int(raw_prefetch_factor)
        if prefetch_factor < 1:
            raise ValueError(
                "dataloader_config.prefetch_factor must be >= 1, "
                f"got {prefetch_factor}"
            )

    if num_workers == 0:
        if persistent_workers:
            raise ValueError(
                "dataloader_config.persistent_workers requires num_workers > 0"
            )
        if prefetch_factor is not None:
            raise ValueError(
                "dataloader_config.prefetch_factor requires num_workers > 0"
            )

    resolved = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "non_blocking": non_blocking,
    }
    config["dataloader_config"] = resolved
    return resolved


def _build_dataloader(
    config: dict,
    dataset,
    *,
    shuffle: bool = False,
    sampler=None,
    collate_fn=None,
) -> DataLoader:
    dataloader_config = resolve_dataloader_config(config)
    kwargs = {
        "num_workers": dataloader_config["num_workers"],
        "pin_memory": dataloader_config["pin_memory"],
    }
    if dataloader_config["num_workers"] > 0:
        kwargs["persistent_workers"] = dataloader_config["persistent_workers"]
        if dataloader_config["prefetch_factor"] is not None:
            kwargs["prefetch_factor"] = dataloader_config["prefetch_factor"]

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        **kwargs,
    )


def resolve_step_eval_skip(config: dict) -> int:
    raw_value = config.get("step_eval_skip", 1)
    if isinstance(raw_value, bool):
        raise ValueError(f"step_eval_skip must be an integer >= 1, got {raw_value!r}")
    if isinstance(raw_value, int):
        step_eval_skip = raw_value
    elif isinstance(raw_value, str):
        try:
            step_eval_skip = int(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"step_eval_skip must be an integer >= 1, got {raw_value!r}") from exc
    else:
        raise ValueError(f"step_eval_skip must be an integer >= 1, got {raw_value!r}")
    if step_eval_skip < 1:
        raise ValueError(f"step_eval_skip must be >= 1, got {step_eval_skip}")
    config["step_eval_skip"] = step_eval_skip
    return step_eval_skip


def resolve_dataset_load_partitions(
    config: dict,
    dist_context: DistributedContext | None = None,
) -> int:
    partitions = int(config.get("dataset_load_partitions", 1) or 1)
    if partitions < 1:
        raise ValueError(f"dataset_load_partitions must be >= 1, got {partitions}")
    if partitions > 1 and not config.get("dataset_cache_dir"):
        raise ValueError(
            "dataset_load_partitions > 1 requires dataset_cache_dir so train tokenized shards "
            "can be written and reloaded without keeping all samples in memory."
        )
    if (
        partitions > 1
        and dist_context is not None
        and dist_context.is_distributed
    ):
        world_size = int(dist_context.world_size)
        if partitions < world_size or partitions % world_size != 0:
            raise ValueError(
                "DDP partitioned training requires dataset_load_partitions to be "
                f">= world_size and divisible by world_size; got "
                f"dataset_load_partitions={partitions}, world_size={world_size}."
            )
    return partitions


def resolve_continuous_mean_l1_weight(config: dict) -> float:
    raw_weight = config.get("continuous_mean_l1_weight", 0.0)
    weight = 0.0 if raw_weight is None else float(raw_weight)
    if weight < 0:
        raise ValueError(f"continuous_mean_l1_weight must be >= 0, got {weight}")
    return weight


def resolve_action_head_weight_decay(config: dict) -> float | None:
    if "action_head_weight_decay" not in config:
        return None
    raw_weight_decay = config.get("action_head_weight_decay")
    weight_decay = 0.0 if raw_weight_decay is None else float(raw_weight_decay)
    if weight_decay < 0:
        raise ValueError(f"action_head_weight_decay must be >= 0, got {weight_decay}")
    return weight_decay


def ensure_experiment_id(config: dict) -> str:
    experiment_id = config.get("experiment_id")
    if experiment_id:
        return str(experiment_id)

    experiment_id = uuid.uuid4().hex[:8]
    config["experiment_id"] = experiment_id
    return experiment_id


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


def normalize_prompt_config(config: dict):
    """Persist the exact training prompt names in checkpoint config.yaml."""
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
        raise ValueError(f"Unknown prompt template names for {config['env_family']}: {missing}. Available: {available}")

    config[primary_key] = names
    config.pop(legacy_key, None)


def resolve_train_selection(config: dict, available_variants: list[str]) -> VariantSelection:
    train_variants = config.get("train_varients", config.get("variants"))
    return resolve_selection(
        mode=config["train_mode"],
        variants=train_variants,
        available_variants=available_variants,
        field_name="train_varients",
    )



def resolve_epoch_eval_selection(
    config: dict,
    available_variants: list[str],
    train_selection: VariantSelection,
) -> VariantSelection:
    eval_mode = config.get("eval_mode")
    eval_variants = config.get("eval_variants")
    if not eval_mode and not eval_variants:
        return train_selection

    resolved_eval_mode = eval_mode or train_selection.mode
    default_variants = None
    if resolved_eval_mode == "single":
        default_variants = train_selection.selected_variants
    elif resolved_eval_mode == "except":
        default_variants = train_selection.configured_variants

    return resolve_selection(
        mode=resolved_eval_mode,
        variants=eval_variants,
        available_variants=available_variants,
        field_name="eval_variants",
        default_variants=default_variants,
    )



def get_checkpoint_dir(
    config: dict,
    selection_tag: str,
    epoch: int | None = None,
    step: int | None = None,
) -> str:
    slug = get_model_slug(config["model_name"])
    root = config.get("checkpoint_root", "checkpoints")
    experiment_id = config["experiment_id"]
    if epoch is not None and step is not None:
        raise ValueError("checkpoint path can be keyed by epoch or step, not both")
    if step is not None:
        tag = f"step{step}"
    elif epoch is not None:
        tag = f"ep{epoch}"
    else:
        tag = "final"
    return os.path.join(root, config["env_family"], slug, selection_tag, experiment_id, tag)



def get_train_results_base_dir(config: dict, train_selection_tag: str) -> str:
    slug = get_model_slug(config["model_name"])
    env_family = config["env_family"]
    experiment_id = config["experiment_id"]
    result_root = config.get("result_root", "results")
    train_tag = f"train={env_family}-{train_selection_tag}"
    exp_tag = f"exp={experiment_id}"
    return os.path.join(result_root, slug, train_tag, exp_tag)


def get_eval_epoch_results_dir(config: dict, train_selection_tag: str, epoch: int) -> str:
    return os.path.join(get_train_results_base_dir(config, train_selection_tag), f"epoch_{epoch}")


def get_eval_step_results_dir(config: dict, train_selection_tag: str, step: int) -> str:
    return os.path.join(get_train_results_base_dir(config, train_selection_tag), f"step{step}")


def get_eval_results_dir(
    config: dict,
    train_selection_tag: str,
    epoch: int | None = None,
    step: int | None = None,
) -> str:
    if epoch is not None and step is not None:
        raise ValueError("eval results path can be keyed by epoch or step, not both")
    if step is not None:
        return get_eval_step_results_dir(config, train_selection_tag, step)
    if epoch is not None:
        return get_eval_epoch_results_dir(config, train_selection_tag, epoch)
    raise ValueError("eval results path requires epoch or step")


def get_eval_variant_results_dir(
    config: dict,
    train_selection_tag: str,
    variant: str,
    epoch: int | None = None,
    step: int | None = None,
) -> str:
    return os.path.join(
        get_eval_results_dir(config, train_selection_tag, epoch=epoch, step=step),
        f"eval={config['env_family']}-{variant}",
    )



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
        episode_keep_num=config.get("episode_keep_num"),
        balance_variant_episode_count=config.get("balance_variant_episode_count", False),
        balanced_train_episode_count=config.get("balanced_train_episode_count"),
        sampling_seed=config.get("sampling_seed", 0),
        family_data_config=(
            config.get("antmaze_data_config")
            if config.get("env_family") == "antmaze"
            else None
        ),
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


def _resolve_balanced_train_episode_count(
    config: dict,
    dataset_cls,
    selected_variants: list[str],
    dist_context: DistributedContext,
) -> int | None:
    balance_enabled = config.get("balance_variant_episode_count", False)
    if len(selected_variants) <= 1:
        if balance_enabled:
            rank_zero_print(
                dist_context,
                "[train] balance_variant_episode_count=true but only one variant is selected; skipping balancing.",
            )
        return None

    if not balance_enabled:
        return None

    keep_num = config.get("episode_keep_num")
    family_data_config = (
        config.get("antmaze_data_config")
        if config.get("env_family") == "antmaze"
        else None
    )
    variant_stats = [
        dataset_cls.collect_variant_episode_stats(
            variant,
            keep_num,
            family_data_config=family_data_config,
        )
        for variant in selected_variants
    ]
    balanced_target = min(stat["sampled_episode_target"] for stat in variant_stats)
    stats_text = ", ".join(
        f"{stat['variant']}: total_episodes={stat['total_episodes']}, "
        f"sampled_episode_target={stat['sampled_episode_target']}"
        for stat in variant_stats
    )
    rank_zero_print(dist_context, f"[train] Multi-variant episode balance stats -> {stats_text}")
    rank_zero_print(dist_context, f"[train] Balanced sampled episode target across variants: {balanced_target}")
    return balanced_target


def _build_data_loaders_once(
    config: dict,
    tokenizer,
    selected_variants: list[str],
    dist_context: DistributedContext,
    splits: tuple[str, ...] = ("train", "val"),
):
    requested_splits = tuple(splits)
    if not requested_splits:
        raise ValueError("At least one dataset split must be requested.")
    unknown_splits = sorted(set(requested_splits) - {"train", "val"})
    if unknown_splits:
        raise ValueError(f"Unknown dataset splits requested: {unknown_splits}")

    train_datasets = []
    val_datasets = []
    dataset_cls = get_dataset(config["env_family"])
    dataset_config = dict(config)
    dataset_config["balanced_train_episode_count"] = _resolve_balanced_train_episode_count(
        config, dataset_cls, selected_variants, dist_context
    )

    dataset_jobs = []
    dataset_requests = []

    for variant in selected_variants:
        rank_zero_print(dist_context, f"[train] Loading data for variant: {variant}")
        for split in requested_splits:
            dataset_jobs.append((variant, split))
            dataset_requests.append(build_dataset_request(dataset_config, tokenizer, variant, split))

    datasets = dataset_cls.build_batch(dataset_requests)
    collate_fn = dataset_cls.collate_fn

    for (_, split), dataset in zip(dataset_jobs, datasets):
        if split == "train":
            train_datasets.append(dataset)
        elif split == "val":
            val_datasets.append(dataset)

    include_train = "train" in requested_splits
    include_val = "val" in requested_splits
    train_loader = None
    val_loader = None

    if len(selected_variants) == 1:
        train_dataset = train_datasets[0] if include_train else None
        val_dataset = val_datasets[0] if include_val else None
        train_count = len(train_dataset) if train_dataset is not None else 0
        val_count = len(val_dataset) if val_dataset is not None else 0
        rank_zero_print(
            dist_context,
            f"[train] Train samples: {train_count}, Val samples: {val_count}",
        )
        if train_dataset is not None:
            if len(train_dataset) < 1:
                raise ValueError(
                    "Selected train dataset is empty; reduce dataset_load_partitions "
                    "or increase episode_keep_num."
                )
            train_sampler = None
            train_shuffle = True
            if dist_context.is_distributed:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=dist_context.world_size,
                    rank=dist_context.rank,
                    shuffle=True,
                    seed=int(config.get("sampling_seed", 0)),
                    drop_last=False,
                )
                train_shuffle = False
            train_loader = _build_dataloader(
                config,
                train_dataset,
                shuffle=train_shuffle,
                sampler=train_sampler,
                collate_fn=collate_fn,
            )
        if val_dataset is not None:
            val_loader = _build_dataloader(
                config,
                val_dataset,
                shuffle=False,
                collate_fn=collate_fn,
            )
        return train_loader, val_loader

    combined_train = None
    combined_val = None
    if include_train:
        weights = []
        for ds in train_datasets:
            n = len(ds)
            w = 1.0 / n if n > 0 else 0.0
            weights.extend([w] * n)

        combined_train = ConcatDataset(train_datasets)
        if len(combined_train) < 1:
            raise ValueError(
                "Selected train dataset partition is empty; reduce dataset_load_partitions "
                "or increase episode_keep_num."
            )

        if dist_context.is_distributed:
            sampler = DistributedWeightedSampler(
                weights=weights,
                num_replicas=dist_context.world_size,
                rank=dist_context.rank,
                num_samples=len(combined_train),
                replacement=True,
                seed=int(config.get("sampling_seed", 0)),
            )
        else:
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(combined_train),
                replacement=True,
            )
        train_loader = _build_dataloader(
            config,
            combined_train,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    if include_val:
        combined_val = ConcatDataset(val_datasets)
        val_loader = _build_dataloader(
            config,
            combined_val,
            shuffle=False,
            collate_fn=collate_fn,
        )

    train_count = len(combined_train) if combined_train is not None else 0
    val_count = len(combined_val) if combined_val is not None else 0
    rank_zero_print(
        dist_context,
        f"[train] Joint train samples: {train_count}, Val samples: {val_count}",
    )
    return train_loader, val_loader


def build_data_loaders(
    config: dict,
    tokenizer,
    selected_variants: list[str],
    dist_context: DistributedContext,
    splits: tuple[str, ...] = ("train", "val"),
):
    if dist_context.is_distributed and config.get("dataset_cache_dir"):
        loaders = None
        if dist_context.is_main_process:
            loaders = _build_data_loaders_once(config, tokenizer, selected_variants, dist_context, splits)
        barrier(dist_context)
        if not dist_context.is_main_process:
            loaders = _build_data_loaders_once(config, tokenizer, selected_variants, dist_context, splits)
        barrier(dist_context)
        if loaders is None:
            raise RuntimeError("Failed to build distributed data loaders.")
        return loaders

    if dist_context.is_distributed:
        rank_zero_print(
            dist_context,
            "[train] WARNING: dataset_cache_dir is not configured; each DDP rank will build datasets independently.",
        )
    return _build_data_loaders_once(config, tokenizer, selected_variants, dist_context, splits)


def _partition_config(config: dict, partition_count: int, partition_index: int) -> dict:
    partition_config = dict(config)
    partition_config["dataset_partition_count"] = partition_count
    partition_config["dataset_partition_index"] = partition_index
    return partition_config


def _validation_cache_config(config: dict, partition_count: int) -> dict:
    if partition_count <= 1:
        return dict(config)
    # Val remains logically unpartitioned, but the partition marker makes the
    # dataset cache split-specific instead of reusing the train+val sampled pool.
    return _partition_config(config, partition_count, 0)


def _build_partition_data_loaders(
    config: dict,
    tokenizer,
    selected_variants: list[str],
    dist_context: DistributedContext,
    *,
    partition_count: int,
    partition_index: int,
    splits: tuple[str, ...] = ("train", "val"),
):
    return build_data_loaders(
        _partition_config(config, partition_count, partition_index),
        tokenizer,
        selected_variants,
        dist_context,
        splits=splits,
    )


def _build_partition_plan_requests(
    config: dict,
    tokenizer,
    selected_variants: list[str],
    dist_context: DistributedContext,
) -> list[DatasetBuildRequest]:
    dataset_cls = get_dataset(config["env_family"])
    dataset_config = dict(config)
    dataset_config["balanced_train_episode_count"] = _resolve_balanced_train_episode_count(
        config, dataset_cls, selected_variants, dist_context
    )
    return [
        build_dataset_request(dataset_config, tokenizer, variant, "train")
        for variant in selected_variants
    ]


def _compute_partition_round_stats(
    partition_stats: list[dict],
    *,
    world_size: int,
    batch_size: int,
) -> list[dict]:
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if len(partition_stats) % world_size != 0:
        raise ValueError(
            "partition_stats length must be divisible by world_size: "
            f"partitions={len(partition_stats)}, world_size={world_size}"
        )
    rounds = []
    for round_index, start in enumerate(range(0, len(partition_stats), world_size)):
        shard_stats = partition_stats[start : start + world_size]
        shard_batches = [
            int(math.ceil(int(stat["train_samples"]) / int(batch_size)))
            for stat in shard_stats
        ]
        target_batches = max(shard_batches)
        rounds.append(
            {
                "round_index": round_index,
                "partition_indices": [int(stat["partition_index"]) for stat in shard_stats],
                "target_batches": int(target_batches),
                "target_samples": int(target_batches) * int(batch_size),
                "shard_train_batches": shard_batches,
                "shard_train_samples": [
                    int(stat["train_samples"]) for stat in shard_stats
                ],
            }
        )
    return rounds


def _partition_plan_metadata(plan: dict, *, dist_context: DistributedContext, batch_size: int) -> dict:
    partition_stats = [
        {
            "partition_index": int(shard["partition_index"]),
            "train_samples": int(shard["sample_count"]),
            "train_batches": int(math.ceil(int(shard["sample_count"]) / int(batch_size))),
            "train_steps": int(shard["step_count"]),
        }
        for shard in plan["shards"]
    ]
    round_stats = _compute_partition_round_stats(
        partition_stats,
        world_size=int(dist_context.world_size),
        batch_size=int(batch_size),
    )
    return {
        "env_family": plan["env_family"],
        "partition_count": int(plan["partition_count"]),
        "plan_hash": plan["plan_hash"],
        "variants": list(plan["variants"]),
        "selections": list(plan["selections"]),
        "shards": list(plan["shards"]),
        "partition_stats": partition_stats,
        "round_stats": round_stats,
    }


def _prepare_partition_plan(
    config: dict,
    tokenizer,
    selected_variants: list[str],
    dist_context: DistributedContext,
    *,
    partition_count: int,
) -> tuple[dict | None, dict]:
    full_plan = None
    metadata = None
    if dist_context.is_main_process:
        dataset_cls = get_dataset(config["env_family"])
        if not hasattr(dataset_cls, "plan_train_shards"):
            raise ValueError(
                f"Dataset family {config['env_family']!r} does not support partitioned shard planning."
            )
        plan_requests = _build_partition_plan_requests(
            config,
            tokenizer,
            selected_variants,
            dist_context,
        )
        full_plan = dataset_cls.plan_train_shards(
            plan_requests,
            partition_count=partition_count,
        )
        metadata = _partition_plan_metadata(
            full_plan,
            dist_context=dist_context,
            batch_size=int(config["batch_size"]),
        )
    metadata = broadcast_object(metadata, dist_context)
    if metadata is None:
        raise RuntimeError("Failed to broadcast partition shard plan metadata.")
    return full_plan, metadata


def _round_order(round_count: int, epoch: int, sampling_seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(int(sampling_seed) + int(epoch))
    return torch.randperm(round_count, generator=generator).tolist()


def _assignment_for_round(full_plan: dict, plan_metadata: dict, round_index: int) -> list[dict]:
    dataset_cls = get_dataset(full_plan["env_family"])
    round_stat = plan_metadata["round_stats"][round_index]
    assignments = []
    shards_by_index = {
        int(shard["partition_index"]): shard
        for shard in full_plan["shards"]
    }
    for rank_offset, partition_index in enumerate(round_stat["partition_indices"]):
        shard = shards_by_index[int(partition_index)]
        segments_by_variant = {
            variant: list(segments)
            for variant, segments in shard["segments_by_variant"].items()
        }
        payloads_by_variant = {}
        for variant, segments in segments_by_variant.items():
            episodes = full_plan["episodes_by_variant"][variant]
            payloads_by_variant[variant] = dataset_cls.payloads_for_segments(episodes, segments)
        assignments.append(
            {
                "round_index": int(round_index),
                "rank_offset": int(rank_offset),
                "partition_index": int(partition_index),
                "partition_count": int(plan_metadata["partition_count"]),
                "target_batches": int(round_stat["target_batches"]),
                "target_samples": int(round_stat["target_samples"]),
                "plan_hash": plan_metadata["plan_hash"],
                "segments_by_variant": segments_by_variant,
                "episode_payloads_by_variant": payloads_by_variant,
            }
        )
    return assignments


def _scatter_round_assignment(
    full_plan: dict | None,
    plan_metadata: dict,
    round_index: int,
    dist_context: DistributedContext,
) -> dict:
    assignments = None
    if dist_context.is_main_process:
        if full_plan is None:
            raise RuntimeError("Main process is missing the full partition plan.")
        assignments = _assignment_for_round(full_plan, plan_metadata, round_index)
    assignment = scatter_object(assignments, dist_context)
    if assignment is None:
        raise RuntimeError("Failed to scatter partition round assignment.")
    return assignment


def _partition_shard_config(config: dict, assignment: dict) -> dict:
    shard_config = dict(config)
    shard_config["dataset_partition_count"] = int(assignment["partition_count"])
    shard_config["dataset_partition_index"] = int(assignment["partition_index"])
    return shard_config


def _build_partition_shard_train_loader(
    config: dict,
    tokenizer,
    selected_variants: list[str],
    dist_context: DistributedContext,
    *,
    assignment: dict,
    sampler_seed: int,
):
    dataset_cls = get_dataset(config["env_family"])
    shard_config = _partition_shard_config(config, assignment)
    dataset_requests = []
    dataset_jobs = []
    for variant in selected_variants:
        segments = list(assignment["segments_by_variant"].get(variant, []))
        if not segments:
            continue
        payloads = list(assignment["episode_payloads_by_variant"].get(variant, []))
        dataset_jobs.append((variant, "train"))
        dataset_requests.append(
            build_dataset_request(
                shard_config,
                tokenizer,
                variant,
                "train",
                episode_segments=segments,
                episode_payloads=payloads,
                partition_plan_hash=assignment["plan_hash"],
            )
        )
    if not dataset_requests:
        raise ValueError(
            "Selected train dataset shard is empty; reduce dataset_load_partitions "
            "or increase episode_keep_num."
        )

    datasets = dataset_cls.build_batch(dataset_requests)
    train_datasets = [
        dataset
        for (_, split), dataset in zip(dataset_jobs, datasets)
        if split == "train" and len(dataset) > 0
    ]
    if not train_datasets:
        raise ValueError(
            "Selected train dataset shard is empty after tokenization; reduce dataset_load_partitions "
            "or increase episode_keep_num."
        )
    collate_fn = dataset_cls.collate_fn
    target_samples = int(assignment["target_samples"])

    if len(train_datasets) == 1 and len(selected_variants) == 1:
        train_dataset = train_datasets[0]
        if not dist_context.is_distributed:
            loader = _build_dataloader(
                shard_config,
                train_dataset,
                shuffle=True,
                collate_fn=collate_fn,
            )
            rank_zero_print(
                dist_context,
                f"[train] Shard train samples: {len(train_dataset)}, "
                f"train_batches={len(loader)}",
            )
            return loader
        sampler = LocalShardPaddingSampler(
            len(train_dataset),
            num_samples=target_samples,
            seed=sampler_seed,
            shuffle=True,
        )
        loader = _build_dataloader(
            shard_config,
            train_dataset,
            sampler=sampler,
            collate_fn=collate_fn,
        )
        rank_zero_print(
            dist_context,
            f"[train] Shard train samples: {len(train_dataset)}, "
            f"target_batches={assignment['target_batches']}",
        )
        return loader

    weights = []
    for ds in train_datasets:
        n = len(ds)
        if n < 1:
            continue
        w = 1.0 / n
        weights.extend([w] * n)
    combined_train = ConcatDataset(train_datasets)
    if not dist_context.is_distributed:
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(combined_train),
            replacement=True,
        )
        loader = _build_dataloader(
            shard_config,
            combined_train,
            sampler=sampler,
            collate_fn=collate_fn,
        )
        rank_zero_print(
            dist_context,
            f"[train] Joint shard train samples: {len(combined_train)}, "
            f"train_batches={len(loader)}",
        )
        return loader
    sampler = LocalShardPaddingSampler(
        len(combined_train),
        num_samples=target_samples,
        seed=sampler_seed,
        weights=weights,
    )
    loader = _build_dataloader(
        shard_config,
        combined_train,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    rank_zero_print(
        dist_context,
        f"[train] Joint shard train samples: {len(combined_train)}, "
        f"target_batches={assignment['target_batches']}",
    )
    return loader


def _loader_sample_count(loader) -> int:
    if loader is None:
        return 0
    return len(loader.dataset)


def _release_loaders(*loaders) -> None:
    for loader in loaders:
        if loader is not None:
            del loader
    gc.collect()


def _assert_model_on_device(model, device: torch.device) -> None:
    mismatches = []
    for name, parameter in model.named_parameters():
        if parameter.device != device:
            mismatches.append((name, parameter.device))
            if len(mismatches) >= 5:
                break
    if not mismatches:
        return

    details = ", ".join(f"{name} on {param_device}" for name, param_device in mismatches)
    raise RuntimeError(
        "DDP requires every model parameter to be on the rank-local device "
        f"{device}, but found: {details}. Check Unsloth device_map and "
        "CUDA_VISIBLE_DEVICES."
    )


def _prewarm_partition_caches(
    config: dict,
    tokenizer,
    selected_variants: list[str],
    dist_context: DistributedContext,
    *,
    partition_count: int,
) -> tuple[object, list[dict], dict | None, dict]:
    full_plan, plan_metadata = _prepare_partition_plan(
        config,
        tokenizer,
        selected_variants,
        dist_context,
        partition_count=partition_count,
    )
    rank_zero_print(
        dist_context,
        "[train] Prepared train shard plan: "
        f"partitions={plan_metadata['partition_count']}, "
        f"rounds={len(plan_metadata['round_stats'])}, "
        f"plan_hash={plan_metadata['plan_hash']}",
    )

    val_loader = None
    if dist_context.is_main_process:
        rank_zero_print(dist_context, "[train] Preparing full validation dataset on rank0")
        val_config = _validation_cache_config(config, partition_count)
        _, val_loader = _build_data_loaders_once(
            val_config,
            tokenizer,
            selected_variants,
            dist_context,
            splits=("val",),
        )
    barrier(dist_context)
    val_batches = len(val_loader) if val_loader is not None else 0
    val_samples = _loader_sample_count(val_loader)
    rank_zero_print(
        dist_context,
        f"[train] Prepared full validation dataset: "
        f"val_samples={val_samples}, val_batches={val_batches}",
    )

    stats = list(plan_metadata["partition_stats"])
    for round_stat in plan_metadata["round_stats"]:
        round_index = int(round_stat["round_index"])
        assignment = _scatter_round_assignment(
            full_plan,
            plan_metadata,
            round_index,
            dist_context,
        )
        rank_zero_print(
            dist_context,
            f"[train] Preparing train shard round {round_index + 1}/"
            f"{len(plan_metadata['round_stats'])}: "
            f"partitions={round_stat['partition_indices']}, "
            f"target_batches={round_stat['target_batches']}",
        )
        train_loader = _build_partition_shard_train_loader(
            config,
            tokenizer,
            selected_variants,
            dist_context,
            assignment=assignment,
            sampler_seed=int(config.get("sampling_seed", 0)) + int(assignment["partition_index"]),
        )
        local_stat = {
            "partition_index": int(assignment["partition_index"]),
            "train_batches": len(train_loader),
            "train_samples": _loader_sample_count(train_loader),
            "target_batches": int(assignment["target_batches"]),
        }
        gathered_stats = all_gather_objects(local_stat, dist_context)
        rank_zero_print(
            dist_context,
            "[train] Prepared train shard round "
            f"{round_index + 1}/{len(plan_metadata['round_stats'])}: "
            f"local_stats={gathered_stats}",
        )
        train_loader = None
        _release_loaders()
        barrier(dist_context)
    return val_loader, stats, full_plan, plan_metadata


def _partition_order(partition_count: int, epoch: int, sampling_seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(int(sampling_seed) + int(epoch))
    return torch.randperm(partition_count, generator=generator).tolist()



def _build_training_eval_config(config: dict) -> dict:
    action_token_mode = config.get("action_token_mode", "text")
    eval_config = {
        "env_family": config["env_family"],
        "num_episodes": config["eval_num_episodes"],
        "seed": config.get("eval_seed", 1),
        "parse_retry_limit": config.get("parse_retry_limit", 3),
        "env_kwargs": config.get("eval_env_kwargs", {"continuing_task": False}),
        "history_num": config.get("history_num", 0),
        "history_stride": config.get("history_stride", 1),
        "record_video": config.get("record_video", False),
        "record_all": config.get("record_all", False),
        "video_episode_index": config.get("video_episode_index", 0),
        "video_fps": config.get("video_fps", 20),
        "video_format": config.get("video_format", "gif"),
        "video_save_workers": config.get("video_save_workers", 1),
        "video_save_max_pending": config.get("video_save_max_pending"),
        "mujoco_gl": config.get("mujoco_gl"),
        "record_step_logs": config.get("record_step_logs", True),
        "eval_parallel_episodes": config.get("eval_parallel_episodes", 1),
        "eval_distribute_variants": config.get("eval_distribute_variants", True),
        "action_sampling": config.get("action_sampling", False),
        "action_temperature": config.get("action_temperature", 1.0),
        "action_top_p": config.get("action_top_p", 1.0),
        "action_top_k": config.get("action_top_k", 0),
        "action_token_mode": config.get("action_token_mode", "text"),
        "action_num_bins": config.get("action_num_bins", 10),
        "action_bin_min": config.get("action_bin_min", -1.0),
        "action_bin_max": config.get("action_bin_max", 1.0),
        "new_token": config.get("new_token", False),
        "action_dim": config.get("action_dim"),
        "mtp_k": config.get("mtp_k"),
        "mtp_quadratic_decoding": config.get("mtp_quadratic_decoding", True),
        "action_query_len": config.get("action_query_len"),
        "action_head_num_blocks": config.get("action_head_num_blocks"),
        "action_head_dropout": config.get("action_head_dropout"),
        "action_head_weight_decay": config.get("action_head_weight_decay"),
        "max_length": config.get("max_length"),
    }
    if action_token_mode in {"parallel_gaussian", "parallel_t"}:
        eval_config["gaussian_log_std_min"] = config.get("gaussian_log_std_min")
        eval_config["gaussian_log_std_max"] = config.get("gaussian_log_std_max")
    if action_token_mode == "parallel_gaussian":
        eval_config["gaussian_log_std_init"] = config.get("gaussian_log_std_init")
    if action_token_mode == "parallel_t":
        eval_config["student_t_df"] = config.get("student_t_df")
        eval_config["continuous_mean_l1_weight"] = config.get("continuous_mean_l1_weight")
    return eval_config


def _build_training_eval_context(
    config: dict,
    *,
    eval_type: str,
    train_loss,
    val_loss,
    val_metrics: dict | None,
    checkpoint_dir: str,
    epoch: int | None,
    batch_step: int | None,
    epoch_step: int | None,
    optimizer_step: int | None,
    scheduled_step: int | None,
    scheduled_epoch_step: int | None,
    eval_rank: int | None = None,
    eval_world_size: int | None = None,
    eval_distribute_variants: bool | None = None,
) -> dict:
    context = {
        "eval_type": eval_type,
        "epoch": epoch,
        "batch_step": batch_step,
        "epoch_step": epoch_step,
        "optimizer_step": optimizer_step,
        "scheduled_step": scheduled_step,
        "scheduled_epoch_step": scheduled_epoch_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_metrics": val_metrics or {},
        "checkpoint_path": checkpoint_dir,
        "experiment_id": config["experiment_id"],
    }
    if eval_rank is not None:
        context["eval_rank"] = eval_rank
    if eval_world_size is not None:
        context["eval_world_size"] = eval_world_size
    if eval_distribute_variants is not None:
        context["eval_distribute_variants"] = eval_distribute_variants
    return context


def _isolated_eval_root_dir(
    config: dict,
    train_selection_tag: str,
    *,
    eval_type: str,
    epoch: int | None,
    batch_step: int | None,
) -> str:
    if eval_type == "step":
        return get_eval_results_dir(config, train_selection_tag, step=batch_step)
    return get_eval_results_dir(config, train_selection_tag, epoch=epoch)


def _isolated_eval_rank_dir(
    config: dict,
    train_selection_tag: str,
    *,
    eval_type: str,
    epoch: int | None,
    batch_step: int | None,
    rank: int,
) -> str:
    return os.path.join(
        _isolated_eval_root_dir(
            config,
            train_selection_tag,
            eval_type=eval_type,
            epoch=epoch,
            batch_step=batch_step,
        ),
        "isolated_eval",
        f"rank_{rank}",
    )


_DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
)


def _isolated_eval_subprocess_env(dist_context: DistributedContext) -> dict:
    env = os.environ.copy()
    for key in _DISTRIBUTED_ENV_KEYS:
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"

    if dist_context.is_distributed:
        local_rank = int(dist_context.local_rank)
        visible_devices = env.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            devices = [item.strip() for item in visible_devices.split(",") if item.strip()]
            if local_rank < len(devices):
                env["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    return env


def _build_isolated_training_eval_config(
    config: dict,
    eval_config: dict,
    *,
    local_variants: list[str],
    training_eval_context: dict,
    distribute_variants: bool,
) -> dict:
    child_config = dict(eval_config)
    child_config.update(
        {
            "model_path": os.path.abspath(training_eval_context["checkpoint_path"]),
            "result_root": os.path.abspath(config.get("result_root", "results")),
            "load_in_4bit": config.get("load_in_4bit"),
            "eval_mode": "all",
            "variants": list(local_variants),
            "parallel_backend": "single",
            "eval_distribute_variants": distribute_variants,
            "eval_output_mode": "training",
            "training_eval_context": training_eval_context,
            "prompt_templete_index": [config["prompt_templete_index"][0]],
            "wandb_enabled": False,
        }
    )
    return child_config


def _read_isolated_eval_results(
    config: dict,
    train_selection_tag: str,
    variants: list[str],
    *,
    eval_type: str,
    epoch: int | None,
    batch_step: int | None,
) -> list[dict]:
    results = []
    for variant in variants:
        result_dir = get_eval_variant_results_dir(
            config,
            train_selection_tag,
            variant,
            epoch=epoch if eval_type == "epoch" else None,
            step=batch_step if eval_type == "step" else None,
        )
        result_path = os.path.join(result_dir, "result.json")
        if not os.path.exists(result_path):
            raise FileNotFoundError(
                f"Isolated eval completed but result file is missing: {result_path}"
            )
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        results.append(result)
    return results


def _isolated_eval_attempt_configs(child_config: dict) -> list[tuple[str, dict]]:
    configured = dict(child_config)
    configured["isolated_eval_attempt_mode"] = "configured"
    attempts = [("configured", configured)]

    try:
        requested_parallel_episodes = int(child_config.get("eval_parallel_episodes", 1))
    except (TypeError, ValueError):
        requested_parallel_episodes = 1

    if requested_parallel_episodes > 1:
        serial_fallback = dict(child_config)
        serial_fallback["eval_parallel_episodes"] = 1
        serial_fallback["isolated_eval_attempt_mode"] = "serial_fallback"
        serial_fallback["isolated_eval_fallback_reason"] = (
            "previous_attempt_failed"
        )
        serial_fallback["isolated_eval_original_eval_parallel_episodes"] = (
            requested_parallel_episodes
        )
        attempts.append(("serial_fallback", serial_fallback))

    return attempts


def _run_isolated_eval_subprocess_for_rank(
    config: dict,
    eval_config: dict,
    train_selection_tag: str,
    local_variants: list[str],
    *,
    training_eval_context: dict,
    distribute_variants: bool,
    dist_context: DistributedContext,
) -> tuple[list[dict], list[dict]]:
    if not local_variants:
        return [], []

    rank_dir = _isolated_eval_rank_dir(
        config,
        train_selection_tag,
        eval_type=training_eval_context["eval_type"],
        epoch=training_eval_context.get("epoch"),
        batch_step=training_eval_context.get("batch_step"),
        rank=dist_context.rank,
    )
    os.makedirs(rank_dir, exist_ok=True)

    child_config = _build_isolated_training_eval_config(
        config,
        eval_config,
        local_variants=local_variants,
        training_eval_context=training_eval_context,
        distribute_variants=distribute_variants,
    )
    repo_root = os.path.dirname(os.path.abspath(__file__))
    command = [
        sys.executable,
        os.path.join(repo_root, "evaluate.py"),
        "--config",
        "",
        "--parallel_backend",
        "single",
        "-y",
    ]
    env = _isolated_eval_subprocess_env(dist_context)
    attempt_configs = _isolated_eval_attempt_configs(child_config)
    max_attempts = len(attempt_configs)
    last_error = None

    for attempt, (attempt_mode, attempt_config) in enumerate(attempt_configs, start=1):
        attempt_prefix = os.path.join(rank_dir, f"attempt_{attempt}")
        config_path = f"{attempt_prefix}.yaml"
        stdout_path = f"{attempt_prefix}.stdout"
        stderr_path = f"{attempt_prefix}.stderr"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(attempt_config, f, sort_keys=False, allow_unicode=True)

        attempt_command = list(command)
        attempt_command[attempt_command.index("--config") + 1] = config_path
        print(
            f"[eval][rank {dist_context.rank}] isolated attempt "
            f"{attempt}/{max_attempts} mode={attempt_mode} variants={local_variants} "
            f"parallel_episodes={attempt_config.get('eval_parallel_episodes')}"
        )
        with open(stdout_path, "w", encoding="utf-8") as stdout, open(
            stderr_path,
            "w",
            encoding="utf-8",
        ) as stderr:
            completed = subprocess.run(
                attempt_command,
                cwd=repo_root,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )

        if completed.returncode == 0:
            try:
                return _read_isolated_eval_results(
                    config,
                    train_selection_tag,
                    local_variants,
                    eval_type=training_eval_context["eval_type"],
                    epoch=training_eval_context.get("epoch"),
                    batch_step=training_eval_context.get("batch_step"),
                ), []
            except Exception as exc:  # noqa: BLE001 - retry missing/malformed artifacts.
                last_error = str(exc)
        else:
            last_error = f"returncode={completed.returncode}"

        if attempt < max_attempts:
            print(
                f"[eval][rank {dist_context.rank}] WARNING: isolated eval "
                f"attempt {attempt}/{max_attempts} failed for variants={local_variants}; "
                "retrying with eval_parallel_episodes=1. "
                f"stdout={stdout_path} stderr={stderr_path} error={last_error}"
            )

    failure = {
        "variants": list(local_variants),
        "rank": dist_context.rank,
        "attempts": max_attempts,
        "error": last_error,
        "log_dir": rank_dir,
    }
    print(
        f"[eval][rank {dist_context.rank}] WARNING: isolated eval failed after "
        f"{max_attempts} attempt(s) for variants={local_variants}; "
        f"training will continue. log_dir={rank_dir} error={last_error}"
    )
    return [], [failure]


def _write_isolated_eval_config_snapshot(
    config: dict,
    eval_config: dict,
    train_selection_tag: str,
    variants,
    *,
    training_eval_context: dict,
    assignments: dict,
    parallel_episodes: int,
    distribute_variants: bool,
) -> None:
    snapshot_context = dict(training_eval_context)
    snapshot_context.pop("eval_rank", None)
    snapshot_context["eval_world_size"] = training_eval_context.get("eval_world_size")
    snapshot_context["eval_distribute_variants"] = distribute_variants
    snapshot = _build_isolated_training_eval_config(
        config,
        eval_config,
        local_variants=list(variants),
        training_eval_context=snapshot_context,
        distribute_variants=distribute_variants,
    )
    snapshot["resolved_eval_variants"] = list(variants)
    snapshot["resolved_eval_variant_assignments"] = assignments
    snapshot["eval_world_size"] = training_eval_context.get("eval_world_size")
    snapshot["eval_parallel_episodes"] = parallel_episodes
    snapshot["isolated_eval_fallback_on_failure"] = parallel_episodes > 1
    if parallel_episodes > 1:
        snapshot["isolated_eval_fallback_eval_parallel_episodes"] = 1

    run_results_dir = _isolated_eval_root_dir(
        config,
        train_selection_tag,
        eval_type=training_eval_context["eval_type"],
        epoch=training_eval_context.get("epoch"),
        batch_step=training_eval_context.get("batch_step"),
    )
    os.makedirs(run_results_dir, exist_ok=True)
    eval_config_path = os.path.join(run_results_dir, "eval_config.yaml")
    with open(eval_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)
    print(f"[eval] Isolated eval config saved to: {eval_config_path}")


def _log_training_eval_wandb_metrics(
    *,
    variants,
    results_by_variant: dict,
    failures_by_variant: dict,
    wandb_logger: WandbLogger | None,
    train_env_steps: float | None,
    wandb_batch_step: int | None,
    batch_step: int | None,
    optimizer_step: int | None,
    epoch: int | None,
) -> None:
    if wandb_logger is None or not wandb_logger.enabled or train_env_steps is None:
        return
    step_metrics = wandb_step_metrics(
        env_steps=train_env_steps,
        batch_step=wandb_batch_step if wandb_batch_step is not None else batch_step,
        optimizer_step=optimizer_step,
        epoch=epoch,
    )
    for variant in variants:
        result = results_by_variant.get(variant)
        if result is not None:
            wandb_logger.log(
                {
                    **step_metrics,
                    f"eval/{variant}/success_rate": float(result["success_rate"]),
                    f"eval/{variant}/mean_episode_steps": float(
                        result["mean_episode_steps"]
                    ),
                }
            )
            continue
        failure = failures_by_variant.get(variant)
        if failure is not None:
            wandb_logger.log(
                {
                    **step_metrics,
                    f"eval/{variant}/rollout_failed": 1.0,
                    f"eval/{variant}/isolated_attempts": float(
                        failure.get("attempts", 0)
                    ),
                }
            )


def _run_eval_isolated(
    config,
    train_selection_tag: str,
    variants,
    eval_type: str,
    train_loss,
    val_loss,
    checkpoint_dir: str,
    val_metrics: dict | None = None,
    epoch: int | None = None,
    batch_step: int | None = None,
    epoch_step: int | None = None,
    optimizer_step: int | None = None,
    scheduled_step: int | None = None,
    scheduled_epoch_step: int | None = None,
    wandb_logger: WandbLogger | None = None,
    train_env_steps: float | None = None,
    wandb_batch_step: int | None = None,
    dist_context: DistributedContext | None = None,
):
    from utils.eval_parallel import (
        assigned_eval_variants,
        eval_variant_assignments,
        resolve_eval_distribute_variants,
        resolve_eval_parallel_episodes,
    )

    if dist_context is None:
        dist_context = DistributedContext(backend="single", device=torch.device("cpu"))

    eval_config = _build_training_eval_config(config)
    distribute_variants = resolve_eval_distribute_variants(eval_config)
    parallel_episodes = resolve_eval_parallel_episodes(eval_config)
    assignments = eval_variant_assignments(
        variants,
        dist_context,
        distribute_variants=distribute_variants,
    )
    local_variants = assigned_eval_variants(
        variants,
        dist_context,
        distribute_variants=distribute_variants,
    )
    training_eval_context = _build_training_eval_context(
        config,
        eval_type=eval_type,
        train_loss=train_loss,
        val_loss=val_loss,
        val_metrics=val_metrics,
        checkpoint_dir=checkpoint_dir,
        epoch=epoch,
        batch_step=batch_step,
        epoch_step=epoch_step,
        optimizer_step=optimizer_step,
        scheduled_step=scheduled_step,
        scheduled_epoch_step=scheduled_epoch_step,
        eval_rank=dist_context.rank,
        eval_world_size=dist_context.world_size,
        eval_distribute_variants=distribute_variants,
    )

    label = f"Step {batch_step}" if eval_type == "step" else f"Epoch {epoch}"
    if dist_context.is_main_process:
        print(
            f"[eval] {label} | isolated rollout enabled | "
            f"variant assignments={assignments} | "
            f"parallel_episodes={parallel_episodes} | "
            "fallback_on_failure="
            f"{parallel_episodes > 1}"
        )

    local_results, local_failures = _run_isolated_eval_subprocess_for_rank(
        config,
        eval_config,
        train_selection_tag,
        local_variants,
        training_eval_context=training_eval_context,
        distribute_variants=distribute_variants,
        dist_context=dist_context,
    )
    gathered = all_gather_objects(
        {"results": local_results, "failures": local_failures},
        dist_context,
    )
    if dist_context.is_main_process:
        _write_isolated_eval_config_snapshot(
            config,
            eval_config,
            train_selection_tag,
            variants,
            training_eval_context=training_eval_context,
            assignments=assignments,
            parallel_episodes=parallel_episodes,
            distribute_variants=distribute_variants,
        )
        results_by_variant = {
            result["variant"]: result
            for rank_payload in gathered
            for result in rank_payload["results"]
        }
        failures_by_variant = {}
        for rank_payload in gathered:
            for failure in rank_payload["failures"]:
                for variant in failure.get("variants", []):
                    failures_by_variant[variant] = failure

        for variant in variants:
            result = results_by_variant.get(variant)
            if result is not None:
                print(
                    f"[eval] {variant}: success_rate={result['success_rate']:.2%}, "
                    f"mean_steps={result['mean_episode_steps']:.1f}, "
                    f"rank={result.get('eval_rank')}"
                )
            else:
                failure = failures_by_variant.get(variant, {})
                print(
                    f"[eval] WARNING: isolated rollout failed for variant={variant}; "
                    f"training will continue. attempts={failure.get('attempts')} "
                    f"log_dir={failure.get('log_dir')} error={failure.get('error')}"
                )

        _log_training_eval_wandb_metrics(
            variants=variants,
            results_by_variant=results_by_variant,
            failures_by_variant=failures_by_variant,
            wandb_logger=wandb_logger,
            train_env_steps=train_env_steps,
            wandb_batch_step=wandb_batch_step,
            batch_step=batch_step,
            optimizer_step=optimizer_step,
            epoch=epoch,
        )


def _format_loss_value(value) -> str:
    if value is None:
        return "nan"
    try:
        if not math.isfinite(value):
            return "nan"
    except TypeError:
        return str(value)
    return f"{value:.4f}"


def _format_optional_metric(name: str, value) -> str:
    if value is None:
        return ""
    return f"  {name}={_format_loss_value(value)}  "


def _run_eval(
    config,
    model,
    tokenizer,
    device,
    train_selection_tag: str,
    variants,
    eval_type: str,
    train_loss,
    val_loss,
    checkpoint_dir: str,
    val_metrics: dict | None = None,
    epoch: int | None = None,
    batch_step: int | None = None,
    epoch_step: int | None = None,
    optimizer_step: int | None = None,
    scheduled_step: int | None = None,
    scheduled_epoch_step: int | None = None,
    wandb_logger: WandbLogger | None = None,
    train_env_steps: float | None = None,
    wandb_batch_step: int | None = None,
    dist_context: DistributedContext | None = None,
):
    if _as_bool(config.get("training_eval_rollout_isolated", False)):
        return _run_eval_isolated(
            config,
            train_selection_tag,
            variants,
            eval_type,
            train_loss,
            val_loss,
            checkpoint_dir,
            val_metrics=val_metrics,
            epoch=epoch,
            batch_step=batch_step,
            epoch_step=epoch_step,
            optimizer_step=optimizer_step,
            scheduled_step=scheduled_step,
            scheduled_epoch_step=scheduled_epoch_step,
            wandb_logger=wandb_logger,
            train_env_steps=train_env_steps,
            wandb_batch_step=wandb_batch_step,
            dist_context=dist_context,
        )

    import gymnasium_robotics  # noqa: F401
    from evaluate import (
        apply_training_eval_context_to_result,
        configure_mujoco_gl,
        evaluate_variant,
    )
    from utils.eval_parallel import (
        assigned_eval_variants,
        eval_variant_assignments,
        resolve_eval_distribute_variants,
        resolve_eval_parallel_episodes,
    )
    from utils.prompt_loader import load_named_templates

    if dist_context is None:
        dist_context = DistributedContext(backend="single", device=device)
    eval_config = _build_training_eval_config(config)
    configure_mujoco_gl(eval_config)
    distribute_variants = resolve_eval_distribute_variants(eval_config)
    parallel_episodes = resolve_eval_parallel_episodes(eval_config)
    assignments = eval_variant_assignments(
        variants,
        dist_context,
        distribute_variants=distribute_variants,
    )
    local_variants = assigned_eval_variants(
        variants,
        dist_context,
        distribute_variants=distribute_variants,
    )
    training_eval_context = _build_training_eval_context(
        config,
        eval_type=eval_type,
        train_loss=train_loss,
        val_loss=val_loss,
        val_metrics=val_metrics,
        checkpoint_dir=checkpoint_dir,
        epoch=epoch,
        batch_step=batch_step,
        epoch_step=epoch_step,
        optimizer_step=optimizer_step,
        scheduled_step=scheduled_step,
        scheduled_epoch_step=scheduled_epoch_step,
    )
    prompt_name = config["prompt_templete_index"][0]
    template = load_named_templates(config["env_family"], [prompt_name])[0]
    label = f"Step {batch_step}" if eval_type == "step" else f"Epoch {epoch}"
    if eval_type == "step" and scheduled_step is not None and scheduled_step != batch_step:
        label = f"{label} (scheduled at batch step {scheduled_step})"
    if eval_type == "step" and epoch_step is not None:
        label = f"{label}, epoch step {epoch_step}"
        if scheduled_epoch_step is not None and scheduled_epoch_step != epoch_step:
            label = f"{label} (scheduled at epoch step {scheduled_epoch_step})"

    model.eval()
    unpatch_continuous_action_forward(model)
    unpatch_mtp_bin_forward(model)
    FastLanguageModel.for_inference(model)
    ensure_continuous_action_decoder(model, config)
    ensure_mtp_bin_decoder(model, config)

    try:
        if dist_context.is_main_process:
            print(
                f"[eval] {label} | variant assignments={assignments} | "
                f"parallel_episodes={parallel_episodes}"
            )
        local_results = []
        for variant in local_variants:
            print(
                f"[eval][rank {dist_context.rank}] "
                f"{label} | variant: {variant}"
            )
            results_dir = get_eval_variant_results_dir(
                config,
                train_selection_tag,
                variant,
                epoch=epoch if eval_type == "epoch" else None,
                step=batch_step if eval_type == "step" else None,
            )
            os.makedirs(results_dir, exist_ok=True)
            result_path = os.path.join(results_dir, "result.json")

            result = evaluate_variant(
                eval_config,
                variant,
                model,
                tokenizer,
                device,
                template,
                variant_results_dir=results_dir,
            )
            result["prompt_template_name"] = prompt_name
            result["result_path"] = result_path
            apply_training_eval_context_to_result(result, training_eval_context)
            result["eval_rank"] = dist_context.rank
            result["eval_world_size"] = dist_context.world_size
            result["eval_distribute_variants"] = distribute_variants

            print(
                f"[eval][rank {dist_context.rank}] "
                f"{variant}: mean_return={result['mean_return']:.4f}, "
                f"success_rate={result['success_rate']:.2%}, "
                f"mean_steps={result['mean_episode_steps']:.1f}, "
                f"train_loss={_format_loss_value(train_loss)}, "
                f"val_loss={_format_loss_value(val_loss)}"
                f"{_format_optional_metric('val_mae', (val_metrics or {}).get('mae'))}"
            )

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(
                f"[eval][rank {dist_context.rank}] Saved: {result_path}"
            )
            local_results.append(result)

        gathered_results = all_gather_objects(local_results, dist_context)
        if dist_context.is_main_process:
            results_by_variant = {
                result["variant"]: result
                for rank_results in gathered_results
                for result in rank_results
            }
            _log_training_eval_wandb_metrics(
                variants=variants,
                results_by_variant=results_by_variant,
                failures_by_variant={},
                wandb_logger=wandb_logger,
                train_env_steps=train_env_steps,
                wandb_batch_step=wandb_batch_step,
                batch_step=batch_step,
                optimizer_step=optimizer_step,
                epoch=epoch,
            )
    finally:
        model.train()
        unpatch_continuous_action_forward(model)
        unpatch_mtp_bin_forward(model)
        FastLanguageModel.for_training(model)
        ensure_continuous_action_decoder(model, config)
        ensure_mtp_bin_decoder(model, config)


def _compute_batch_loss(model, batch, device, loss_context: dict):
    non_blocking = loss_context["dataloader_non_blocking"]
    input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
    attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
    labels = batch["labels"].to(device, non_blocking=non_blocking)

    if loss_context["action_token_mode"] in {"mtp_bin", "simple_mtp_bin"}:
        action_bin_labels = batch["action_bin_labels"].to(
            device, non_blocking=non_blocking
        )
        action_query_mask = batch["action_query_mask"].to(
            device, non_blocking=non_blocking
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=batch["position_ids"].to(
                device, non_blocking=non_blocking
            ),
            action_query_mask=action_query_mask,
            action_query_offsets=batch["action_query_offsets"].to(
                device, non_blocking=non_blocking
            ),
            action_query_source_positions=batch["action_query_source_positions"].to(
                device, non_blocking=non_blocking
            ),
            action_query_prev_token_ids=batch["action_query_prev_token_ids"].to(
                device, non_blocking=non_blocking
            ),
            mtp_bin=True,
        )
        loss, metrics = mtp_bin_action_loss(
            outputs,
            action_bin_labels,
            action_query_mask,
            batch["action_query_anchor_positions"].to(
                device, non_blocking=non_blocking
            ),
            loss_context["bin_token_ids"],
            lcm_weight=loss_context["mtp_lcm_weight"],
            base_loss_on_queries=loss_context["action_token_mode"] != "simple_mtp_bin",
        )
        metrics["bin_l1"] = mtp_bin_equivalent_l1(
            outputs,
            action_bin_labels,
            action_query_mask,
            loss_context["bin_token_ids"],
            loss_context["action_num_bins"],
            loss_context["action_bin_min"],
            loss_context["action_bin_max"],
        )
        metrics.update(
            mtp_bin_equivalent_l1_by_path(
                outputs,
                action_bin_labels,
                action_query_mask,
                loss_context["bin_token_ids"],
                loss_context["action_num_bins"],
                loss_context["action_bin_min"],
                loss_context["action_bin_max"],
            )
        )
        return loss, metrics

    if loss_context["action_token_mode"] == "gaussian_bin":
        action_bin_labels = batch["action_bin_labels"].to(
            device, non_blocking=non_blocking
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        loss, metrics = gaussian_action_loss(
            outputs.logits,
            labels,
            action_bin_labels,
            loss_context["bin_token_ids"],
            loss_context["action_num_bins"],
            loss_context["action_sigma"],
            action_loss_weight=loss_context["action_loss_weight"],
            stop_loss_weight=loss_context["action_stop_loss_weight"],
            soft_label_radius=loss_context["action_soft_label_radius"],
        )
        metrics["bin_l1"] = action_bin_equivalent_l1(
            outputs.logits,
            action_bin_labels,
            loss_context["bin_token_ids"],
            loss_context["action_num_bins"],
            loss_context["action_bin_min"],
            loss_context["action_bin_max"],
            causal_shift=True,
        )
        return loss, metrics

    if loss_context["action_token_mode"] == "parallel_l1":
        action_values = batch["action_values"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
        predicted_actions = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            continuous_action=True,
        )
        loss = F.l1_loss(predicted_actions.float(), action_values, reduction="mean")
        return loss, {"l1_loss": float(loss.detach().item())}

    if loss_context["action_token_mode"] == "parallel_gaussian":
        action_values = batch["action_values"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
        action_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            continuous_action=True,
        )
        mean = action_output.mean.float()
        if action_output.latent_mean is None:
            raise RuntimeError("parallel_gaussian requires latent_mean from the continuous decoder")
        latent_mean = action_output.latent_mean.float()
        log_std = action_output.log_std.float()
        std = action_output.std.float()
        nll = squashed_gaussian_negative_log_likelihood(
            action_values,
            latent_mean,
            log_std,
        )
        loss = nll.mean()
        mae = F.l1_loss(mean, action_values, reduction="mean")
        return loss, {
            "gaussian_nll_loss": float(loss.detach().item()),
            "gaussian_mae": float(mae.detach().item()),
            "gaussian_mean_std": float(std.detach().mean().item()),
        }

    if loss_context["action_token_mode"] == "parallel_t":
        action_values = batch["action_values"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
        action_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            continuous_action=True,
        )
        mean = action_output.mean.float()
        log_scale = action_output.log_scale.float()
        scale = action_output.scale.float()
        df = loss_context["student_t_df"]
        nll = student_t_negative_log_likelihood(action_values, mean, log_scale, df)
        nll_loss = nll.mean()
        mae = F.l1_loss(mean, action_values, reduction="mean")
        mean_l1_weight = loss_context["continuous_mean_l1_weight"]
        mean_l1_aux = mean_l1_weight * mae
        loss = nll_loss + mean_l1_aux
        return loss, {
            "student_t_nll_loss": float(nll_loss.detach().item()),
            "student_t_mae": float(mae.detach().item()),
            "student_t_mean_l1_aux": float(mean_l1_aux.detach().item()),
            "student_t_mean_l1_weight": float(mean_l1_weight),
            "student_t_mean_scale": float(scale.detach().mean().item()),
            "student_t_df": float(df),
        }

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    if loss_context["action_token_mode"] == "bin":
        action_bin_labels = batch["action_bin_labels"].to(
            device, non_blocking=non_blocking
        )
        return outputs.loss, {
            "bin_l1": action_bin_equivalent_l1(
                outputs.logits,
                action_bin_labels,
                loss_context["bin_token_ids"],
                loss_context["action_num_bins"],
                loss_context["action_bin_min"],
                loss_context["action_bin_max"],
                causal_shift=True,
            )
        }
    return outputs.loss, None


def _format_loss_extra(loss, loss_parts, *, display_loss: float | None = None) -> str:
    loss_value = loss.item() if display_loss is None else float(display_loss)
    if loss_parts is None:
        return f"loss={loss_value:.4f}"
    def _append_bin_l1(text: str) -> str:
        if "bin_l1" not in loss_parts:
            return text
        extra = f"{text} bin_l1={float(loss_parts['bin_l1']):.6f}"
        if "mtp_bin_l1" in loss_parts:
            extra = f"{extra} mtp_bin_l1={float(loss_parts['mtp_bin_l1']):.6f}"
        if "ntp_bin_l1" in loss_parts:
            extra = f"{extra} ntp_bin_l1={float(loss_parts['ntp_bin_l1']):.6f}"
        return extra

    if "bin_l1" in loss_parts and "action_loss" not in loss_parts:
        return _append_bin_l1(f"loss={loss_value:.4f}")
    if "l1_loss" in loss_parts:
        return f"loss={loss_value:.4f} l1={loss_parts['l1_loss']:.6f}"
    if "gaussian_nll_loss" in loss_parts:
        return (
            f"loss={loss_value:.4f} nll={loss_parts['gaussian_nll_loss']:.6f} "
            f"mae={loss_parts['gaussian_mae']:.6f} "
            f"std={loss_parts['gaussian_mean_std']:.6f}"
        )
    if "student_t_nll_loss" in loss_parts:
        return (
            f"loss={loss_value:.4f} tnll={loss_parts['student_t_nll_loss']:.6f} "
            f"mae={loss_parts['student_t_mae']:.6f} "
            f"aux_l1={loss_parts['student_t_mean_l1_aux']:.6f} "
            f"l1w={loss_parts['student_t_mean_l1_weight']:.3f} "
            f"scale={loss_parts['student_t_mean_scale']:.6f} "
            f"df={loss_parts['student_t_df']:.2f}"
        )
    if "base_loss" in loss_parts:
        return _append_bin_l1(
            f"loss={loss_value:.4f} base={loss_parts['base_loss']:.6f} "
            f"sampler={loss_parts['sampler_loss']:.6f} "
            f"lcm={loss_parts['lcm_loss']:.6f} "
            f"tokens={int(loss_parts.get('action_tokens', 0))} "
            f"aqt={int(loss_parts.get('action_query_tokens', 0))}"
        )
    if int(loss_parts.get("stop_tokens", 0)) == 0:
        return _append_bin_l1(
            f"loss={loss_value:.4f} action={loss_parts['action_loss']:.6f} "
            f"tokens={int(loss_parts.get('action_tokens', 0))}"
        )
    return _append_bin_l1(
        f"loss={loss_value:.4f} action={loss_parts['action_loss']:.6f} "
        f"stop={loss_parts['stop_loss']:.4f}"
    )


def _loss_parts_to_wandb_metrics(loss_parts, *, prefix: str) -> dict[str, float]:
    if not loss_parts:
        return {}
    if "l1_loss" in loss_parts:
        return {f"{prefix}/l1": float(loss_parts["l1_loss"])}
    if "gaussian_nll_loss" in loss_parts:
        return {
            f"{prefix}/nll": float(loss_parts["gaussian_nll_loss"]),
            f"{prefix}/mae": float(loss_parts["gaussian_mae"]),
            f"{prefix}/std": float(loss_parts["gaussian_mean_std"]),
        }
    if "student_t_nll_loss" in loss_parts:
        return {
            f"{prefix}/tnll": float(loss_parts["student_t_nll_loss"]),
            f"{prefix}/mae": float(loss_parts["student_t_mae"]),
            f"{prefix}/mean_l1_aux": float(loss_parts["student_t_mean_l1_aux"]),
            f"{prefix}/mean_l1_weight": float(loss_parts["student_t_mean_l1_weight"]),
            f"{prefix}/scale": float(loss_parts["student_t_mean_scale"]),
            f"{prefix}/df": float(loss_parts["student_t_df"]),
        }
    metrics = {}
    if "base_loss" in loss_parts:
        metrics[f"{prefix}/base_loss"] = float(loss_parts["base_loss"])
        metrics[f"{prefix}/sampler_loss"] = float(loss_parts["sampler_loss"])
        metrics[f"{prefix}/lcm_loss"] = float(loss_parts["lcm_loss"])
    if "action_loss" in loss_parts:
        metrics[f"{prefix}/action_loss"] = float(loss_parts["action_loss"])
    if "stop_loss" in loss_parts and int(loss_parts.get("stop_tokens", 1)) > 0:
        metrics[f"{prefix}/stop_loss"] = float(loss_parts["stop_loss"])
    if "bin_l1" in loss_parts:
        metrics[f"{prefix}/bin_l1"] = float(loss_parts["bin_l1"])
    if "mtp_bin_l1" in loss_parts:
        metrics[f"{prefix}/mtp_bin_l1"] = float(loss_parts["mtp_bin_l1"])
    if "ntp_bin_l1" in loss_parts:
        metrics[f"{prefix}/ntp_bin_l1"] = float(loss_parts["ntp_bin_l1"])
    return metrics


def _validation_mae_from_loss_parts(loss_parts) -> float | None:
    if not loss_parts:
        return None
    if "l1_loss" in loss_parts:
        return float(loss_parts["l1_loss"])
    if "gaussian_mae" in loss_parts:
        return float(loss_parts["gaussian_mae"])
    if "student_t_mae" in loss_parts:
        return float(loss_parts["student_t_mae"])
    return None


def _reduce_wandb_loss_part_metrics(
    loss_parts,
    *,
    prefix: str,
    dist_context: DistributedContext,
    device: torch.device,
) -> dict[str, float]:
    metrics = _loss_parts_to_wandb_metrics(loss_parts, prefix=prefix)
    return {
        key: reduce_mean(value, dist_context, device)
        for key, value in metrics.items()
    }


def _run_validation(
    model,
    val_loader,
    device,
    loss_context: dict,
    progress: FileProgress | None,
    desc: str,
    warning_label: str,
):
    model.eval()
    val_total = len(val_loader)
    if val_total == 0:
        print(f"[{warning_label}] WARNING: val_loader is empty; val_loss will be reported as NaN.")
        return math.nan, {}

    val_loss = 0.0
    val_loss_parts: dict[str, float] = {}
    val_batches = 0
    val_start = time.monotonic()
    with torch.no_grad():
        for step, batch in enumerate(val_loader, start=1):
            loss, loss_parts = _compute_batch_loss(model, batch, device, loss_context)
            val_loss += loss.item()
            if loss_parts:
                for key, value in loss_parts.items():
                    val_loss_parts[key] = val_loss_parts.get(key, 0.0) + float(value)
            val_batches += 1
            if progress is not None:
                progress.update(
                    desc,
                    step,
                    val_total,
                    val_start,
                    extra=_format_loss_extra(loss, loss_parts),
                )

    avg_loss_parts = {
        key: value / max(val_batches, 1)
        for key, value in val_loss_parts.items()
    }
    val_metrics = {}
    val_mae = _validation_mae_from_loss_parts(avg_loss_parts)
    if val_mae is not None:
        val_metrics["mae"] = val_mae
    return val_loss / max(val_batches, 1), val_metrics


def _create_training_progress_path(config: dict, dist_context: DistributedContext) -> str | None:
    if not dist_context.is_main_process:
        return None
    experiment_id = str(config.get("experiment_id") or uuid.uuid4().hex[:8])
    progress_path = os.path.join("progress", f"{experiment_id}.txt")
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    print(f"[train] Training progress in file: {os.path.abspath(progress_path)}")
    return progress_path


def _finish_training_progress(progress_path: str | None) -> None:
    if progress_path is None:
        return
    try:
        with open(progress_path, "r", encoding="utf-8") as progress_file:
            final_progress = progress_file.read().rstrip("\n")
    except FileNotFoundError:
        return
    if final_progress:
        print(final_progress, flush=True)
    try:
        os.unlink(progress_path)
    except FileNotFoundError:
        pass


def _create_resource_monitor(config: dict, dist_context: DistributedContext) -> ResourceMonitor:
    enabled = _as_bool(config.get("resource_monitor_enabled", False))
    interval_seconds = float(config.get("resource_monitor_interval_seconds", 1.0))
    if interval_seconds <= 0:
        raise ValueError(
            "resource_monitor_interval_seconds must be > 0, "
            f"got {interval_seconds}"
        )
    config["resource_monitor_enabled"] = enabled
    config["resource_monitor_interval_seconds"] = interval_seconds
    return ResourceMonitor(
        resource_monitor_path(str(config["experiment_id"])),
        interval_seconds=interval_seconds,
        enabled=enabled and dist_context.is_main_process,
    )


def _maybe_prompt_eval_step_interval(
    config: dict,
    train_loader_or_batches,
    dist_context: DistributedContext,
) -> None:
    if not dist_context.is_main_process:
        config["eval_step_interval"] = broadcast_object(None, dist_context)
        return

    eval_step_interval = int(config.get("eval_step_interval", 0) or 0)
    if eval_step_interval != 0:
        config["eval_step_interval"] = broadcast_object(eval_step_interval, dist_context)
        return

    if isinstance(train_loader_or_batches, int):
        batches_per_epoch = train_loader_or_batches
    else:
        batches_per_epoch = len(train_loader_or_batches)
    num_epochs = int(config["num_epochs"])
    total_batches = batches_per_epoch * num_epochs
    print(
        "[train] eval_step_interval=0. "
        f"train batches per epoch={batches_per_epoch}, total train batches={total_batches}."
    )
    if not sys.stdin.isatty():
        print("[train] Non-interactive stdin detected; keeping eval_step_interval disabled.")
        config["eval_step_interval"] = broadcast_object(0, dist_context)
        return

    answer = input(
        "[train] Enter eval_step_interval to enable step eval, "
        "or press Enter/0 to keep disabled: "
    ).strip()
    if not answer or answer == "0":
        print("[train] Keeping eval_step_interval disabled.")
        config["eval_step_interval"] = broadcast_object(0, dist_context)
        return
    try:
        selected_interval = int(answer)
    except ValueError:
        print(f"[train] Invalid eval_step_interval {answer!r}; keeping disabled.")
        config["eval_step_interval"] = broadcast_object(0, dist_context)
        return
    if selected_interval == 0:
        print("[train] Keeping eval_step_interval disabled.")
        config["eval_step_interval"] = broadcast_object(0, dist_context)
        return
    if selected_interval < 0:
        print(f"[train] eval_step_interval must be >= 0, got {selected_interval}; keeping disabled.")
        config["eval_step_interval"] = broadcast_object(0, dist_context)
        return

    config["eval_step_interval"] = selected_interval
    print(f"[train] Using eval_step_interval={selected_interval}.")
    config["eval_step_interval"] = broadcast_object(selected_interval, dist_context)


_STEP_EVAL_EPOCH_SKIP_RATIO = 0.25
TRAINER_STATE_FILENAME = "trainer_state.pt"


def _initial_epoch_step_eval_at(eval_step_interval: int) -> int | None:
    return eval_step_interval if eval_step_interval > 0 else None


def _consume_epoch_step_eval_trigger(
    *,
    next_step_eval_at: int | None,
    epoch_batch_step: int,
    eval_step_interval: int,
) -> tuple[int | None, int | None]:
    if (
        next_step_eval_at is None
        or eval_step_interval <= 0
        or epoch_batch_step < next_step_eval_at
    ):
        return None, next_step_eval_at

    scheduled_epoch_step = next_step_eval_at
    while next_step_eval_at <= epoch_batch_step:
        next_step_eval_at += eval_step_interval
    return scheduled_epoch_step, next_step_eval_at


def _step_eval_epoch_skip_reason(
    *,
    epoch: int,
    epoch_batch_step: int,
    train_batches_per_epoch: int,
    eval_step_interval: int,
) -> str | None:
    if eval_step_interval <= 0:
        return None

    skip_window = eval_step_interval * _STEP_EVAL_EPOCH_SKIP_RATIO
    distance_before_epoch_eval = train_batches_per_epoch - epoch_batch_step
    if 0 <= distance_before_epoch_eval <= skip_window:
        if distance_before_epoch_eval == 0:
            return (
                f"coincides with epoch {epoch} end; epoch eval will run instead."
            )
        return (
            f"runs {distance_before_epoch_eval} train batches before epoch {epoch} end, "
            f"within {skip_window:g}=0.25*eval_step_interval; epoch eval will run instead."
        )

    distance_after_previous_epoch_eval = epoch_batch_step
    if epoch > 1 and distance_after_previous_epoch_eval <= skip_window:
        return (
            f"runs {distance_after_previous_epoch_eval} train batches after epoch {epoch - 1} end, "
            f"within {skip_window:g}=0.25*eval_step_interval; previous epoch eval already ran."
        )

    return None


def _step_eval_checkpoint_only_reason(
    *,
    trigger_count: int,
    step_eval_skip: int,
    epoch_skip_reason: str | None,
) -> str | None:
    if epoch_skip_reason is not None:
        return f"near epoch eval: {epoch_skip_reason}"
    if step_eval_skip <= 1:
        return None
    if trigger_count % step_eval_skip == 0:
        return None
    return (
        f"step_eval_skip={step_eval_skip}: step eval trigger {trigger_count} "
        "this epoch is checkpoint-only"
    )


def _trainer_state_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, TRAINER_STATE_FILENAME)


def _load_trainer_state(checkpoint_dir: str | None) -> dict | None:
    if not checkpoint_dir:
        return None
    state_path = _trainer_state_path(checkpoint_dir)
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Cannot resume from {checkpoint_dir}: missing {TRAINER_STATE_FILENAME}. "
            "Resume is supported only for checkpoints saved with trainer state."
        )
    return torch.load(state_path, map_location="cpu")


def _optimizer_param_group_signature(optimizer: torch.optim.Optimizer) -> list[dict]:
    signature = []
    for group in optimizer.param_groups:
        signature.append(
            {
                "param_count": len(group.get("params", [])),
                "weight_decay": float(group.get("weight_decay", 0.0) or 0.0),
            }
        )
    return signature


def _partition_stats_signature(partition_stats: list[dict] | None) -> list[dict]:
    if not partition_stats:
        return []
    return [
        {
            "partition_index": int(stat["partition_index"]),
            "train_batches": int(stat["train_batches"]),
            "train_samples": int(stat["train_samples"]),
            **(
                {"train_steps": int(stat["train_steps"])}
                if "train_steps" in stat
                else {}
            ),
        }
        for stat in partition_stats
    ]


def _round_stats_signature(round_stats: list[dict] | None) -> list[dict]:
    if not round_stats:
        return []
    return [
        {
            "round_index": int(stat["round_index"]),
            "partition_indices": [int(index) for index in stat["partition_indices"]],
            "target_batches": int(stat["target_batches"]),
            "target_samples": int(stat["target_samples"]),
            "shard_train_batches": [int(value) for value in stat["shard_train_batches"]],
            "shard_train_samples": [int(value) for value in stat["shard_train_samples"]],
        }
        for stat in round_stats
    ]


def _resume_compat_metadata(
    *,
    config: dict,
    selected_variants: list[str],
    dist_context: DistributedContext,
    optimizer: torch.optim.Optimizer,
    train_batches_per_epoch: int,
    partition_count: int,
    partition_stats: list[dict] | None,
    partition_plan_hash: str | None = None,
    round_stats: list[dict] | None = None,
) -> dict:
    metadata = {
        "train_variants": list(selected_variants),
        "world_size": int(dist_context.world_size),
        "batch_size": int(config["batch_size"]),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
        "action_token_mode": get_action_token_mode(config),
        "action_dim": int(config.get("action_dim", 0) or 0),
        "dataset_load_partitions": int(partition_count),
        "train_batches_per_epoch": int(train_batches_per_epoch),
        "optimizer_param_groups": _optimizer_param_group_signature(optimizer),
        "partition_stats": _partition_stats_signature(partition_stats),
    }
    if int(partition_count) > 1:
        metadata["partition_plan_hash"] = partition_plan_hash
        metadata["round_stats"] = _round_stats_signature(round_stats)
    return metadata


def _validate_resume_compatibility(saved: dict, current: dict) -> None:
    mismatches = []
    for key, current_value in current.items():
        saved_value = saved.get(key)
        if saved_value != current_value:
            mismatches.append((key, saved_value, current_value))
    if mismatches:
        lines = [
            "Resume checkpoint is incompatible with the current training setup:"
        ]
        for key, saved_value, current_value in mismatches:
            lines.append(f"  {key}: checkpoint={saved_value!r}, current={current_value!r}")
        raise ValueError("\n".join(lines))


def _scheduler_metadata(
    *,
    scheduler_type: str,
    base_learning_rate: float,
    warmup_steps: int,
    lr_decay_steps: int,
    min_lr_ratio: float,
    total_training_steps: int,
    updates_per_epoch: int,
    optimizer_step: int,
) -> dict:
    return {
        "scheduler_type": scheduler_type,
        "base_learning_rate": float(base_learning_rate),
        "warmup_steps": int(warmup_steps),
        "lr_decay_steps": int(lr_decay_steps),
        "min_lr_ratio": float(min_lr_ratio),
        "total_training_steps": int(total_training_steps),
        "updates_per_epoch": int(updates_per_epoch),
        "optimizer_step": int(optimizer_step),
    }


def _loop_state(
    *,
    epoch: int,
    completed_epoch_batch_step: int,
    train_batches_per_epoch: int,
    global_batch_step: int,
    epoch_step_eval_count: int,
    next_step_eval_at: int | None,
    partition_order_position: int | None = None,
    partition_index: int | None = None,
    completed_partition_batch_step: int | None = None,
) -> dict:
    return {
        "current_epoch": int(epoch),
        "completed_epoch_batch_step": int(completed_epoch_batch_step),
        "train_batches_per_epoch": int(train_batches_per_epoch),
        "global_batch_step": int(global_batch_step),
        "epoch_step_eval_count": int(epoch_step_eval_count),
        "next_step_eval_at": next_step_eval_at,
        "partition_order_position": partition_order_position,
        "partition_index": partition_index,
        "completed_partition_batch_step": completed_partition_batch_step,
    }


def _build_trainer_state(
    *,
    config: dict,
    optimizer: torch.optim.Optimizer,
    scheduler_meta: dict,
    loop_state: dict,
    compat_metadata: dict,
) -> dict:
    source_checkpoint = config.get("resume_from_checkpoint")
    return {
        "version": 1,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": dict(scheduler_meta),
        "loop_state": dict(loop_state),
        "compat": dict(compat_metadata),
        "source_checkpoint_path": str(source_checkpoint) if source_checkpoint else None,
        "source_experiment_id": config.get("resume_source_experiment_id"),
        "experiment_id": config.get("experiment_id"),
    }


def _resume_epoch_plan(
    *,
    additional_epochs: int,
    train_batches_per_epoch: int,
    resume_loop_state: dict | None,
) -> tuple[int, int, int, int]:
    if additional_epochs < 0:
        raise ValueError(f"num_epochs must be >= 0 when resuming, got {additional_epochs}")
    if resume_loop_state is None:
        return 1, int(additional_epochs), 0, 0

    resume_epoch = int(resume_loop_state["current_epoch"])
    completed = int(resume_loop_state.get("completed_epoch_batch_step", 0) or 0)
    if completed < 0 or completed > int(train_batches_per_epoch):
        raise ValueError(
            "Invalid checkpoint loop state: completed_epoch_batch_step="
            f"{completed}, train_batches_per_epoch={train_batches_per_epoch}"
        )
    end_epoch = resume_epoch + int(additional_epochs)
    start_epoch = resume_epoch if completed < int(train_batches_per_epoch) else resume_epoch + 1
    return start_epoch, end_epoch, resume_epoch, completed


def _completed_optimizer_steps_in_epoch(
    completed_epoch_batch_step: int,
    gradient_accumulation_steps: int,
    train_batches_per_epoch: int,
) -> int:
    completed = int(completed_epoch_batch_step)
    if completed <= 0:
        return 0
    if completed >= int(train_batches_per_epoch):
        return int(math.ceil(int(train_batches_per_epoch) / int(gradient_accumulation_steps)))
    if completed % int(gradient_accumulation_steps) != 0:
        raise ValueError(
            "Cannot resume from a checkpoint saved mid gradient-accumulation window: "
            f"completed_epoch_batch_step={completed}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}"
        )
    return completed // int(gradient_accumulation_steps)


def _build_loss_context(config: dict, tokenizer) -> dict:
    action_token_mode = get_action_token_mode(config)
    dataloader_config = resolve_dataloader_config(config)
    bin_token_ids = None
    action_num_bins = None
    action_bin_min = None
    action_bin_max = None
    action_sigma = None
    action_loss_weight = None
    action_stop_loss_weight = None
    action_soft_label_radius = None
    mtp_lcm_weight = None
    student_t_df = None
    continuous_mean_l1_weight = 0.0
    if action_token_mode in {"bin", "gaussian_bin", "mtp_bin", "simple_mtp_bin"}:
        bin_token_ids = get_action_bin_token_ids(tokenizer, config)
        action_num_bins = get_action_num_bins(config)
        action_bin_min, action_bin_max = get_action_bin_range(config)
    if action_token_mode in {"mtp_bin", "simple_mtp_bin"}:
        mtp_lcm_weight = resolve_mtp_lcm_weight(config)
    if action_token_mode == "gaussian_bin":
        action_sigma = float(config.get("action_soft_label_sigma", 1.0))
        action_loss_weight = float(config.get("action_loss_weight", 1.0))
        action_stop_loss_weight = float(config.get("action_stop_loss_weight", 1.0))
        action_soft_label_radius = config.get("action_soft_label_radius")
        if action_soft_label_radius is not None:
            action_soft_label_radius = int(action_soft_label_radius)
    if action_token_mode == "parallel_t":
        student_t_df = resolve_student_t_df(config)
        continuous_mean_l1_weight = resolve_continuous_mean_l1_weight(config)
    return {
        "action_token_mode": action_token_mode,
        "bin_token_ids": bin_token_ids,
        "action_num_bins": action_num_bins,
        "action_bin_min": action_bin_min,
        "action_bin_max": action_bin_max,
        "action_sigma": action_sigma,
        "action_loss_weight": action_loss_weight,
        "action_stop_loss_weight": action_stop_loss_weight,
        "action_soft_label_radius": action_soft_label_radius,
        "mtp_lcm_weight": mtp_lcm_weight,
        "student_t_df": student_t_df,
        "continuous_mean_l1_weight": continuous_mean_l1_weight,
        "dataloader_non_blocking": dataloader_config["non_blocking"],
    }


def _build_optimizer_params(model, raw_model, config: dict):
    action_head_weight_decay = resolve_action_head_weight_decay(config)
    if action_head_weight_decay is None or not uses_continuous_actions(config):
        return filter(lambda p: p.requires_grad, model.parameters()), None

    decoder = getattr(raw_model, "continuous_action_decoder", None)
    if decoder is None:
        raise ValueError("action_head_weight_decay requires an attached continuous_action_decoder")

    decay_param_ids = {
        id(param)
        for _, param in decoder.action_head.named_parameters()
        if param.requires_grad and param.ndim >= 2
    }
    decay_params = []
    no_decay_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in decay_param_ids:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = []
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": action_head_weight_decay})
    optimizer_info = {
        "action_head_weight_decay": action_head_weight_decay,
        "action_head_decay_param_count": len(decay_params),
        "non_decay_param_count": len(no_decay_params),
    }
    return param_groups, optimizer_info


def _run_training_partitioned(
    config,
    model,
    tokenizer,
    selected_variants: list[str],
    val_loader,
    device,
    selection_tag: str,
    progress_interval_seconds: float,
    dist_context: DistributedContext,
    *,
    partition_count: int,
    partition_stats: list[dict],
    full_partition_plan: dict | None,
    partition_plan_metadata: dict,
    eval_variants=None,
    wandb_logger: WandbLogger | None = None,
):
    if wandb_logger is None:
        wandb_logger = WandbLogger()
    raw_model = unwrap_model(model)
    base_learning_rate = float(config["learning_rate"])
    optimizer_params, optimizer_info = _build_optimizer_params(model, raw_model, config)
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=base_learning_rate,
    )
    if optimizer_info is not None:
        rank_zero_print(
            dist_context,
            "[train] Continuous action MLP optimizer: "
            f"action_head_weight_decay={optimizer_info['action_head_weight_decay']}, "
            f"decay_params={optimizer_info['action_head_decay_param_count']}, "
            f"non_decay_params={optimizer_info['non_decay_param_count']}",
        )

    num_epochs = int(config["num_epochs"])
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be >= 1, "
            f"got {gradient_accumulation_steps}"
        )
    round_stats = list(partition_plan_metadata["round_stats"])
    train_batches_per_epoch = sum(int(stat["target_batches"]) for stat in round_stats)
    val_batches_per_epoch = len(val_loader) if val_loader is not None else 0
    if train_batches_per_epoch < 1:
        raise ValueError("Partitioned training has zero train batches.")
    updates_per_epoch = math.ceil(train_batches_per_epoch / gradient_accumulation_steps)
    total_training_steps = max(updates_per_epoch * num_epochs, 1)
    lr_scheduler_type = normalize_lr_scheduler_type(config.get("lr_scheduler_type", "constant"))
    warmup_steps = resolve_warmup_steps(
        config,
        total_training_steps,
        steps_per_epoch=updates_per_epoch,
    )
    lr_decay_steps = resolve_lr_decay_steps(
        config,
        total_training_steps=total_training_steps,
        warmup_steps=warmup_steps,
        steps_per_epoch=updates_per_epoch,
    )
    min_lr_ratio = resolve_min_lr_ratio(config)
    initial_lr = set_optimizer_lr(
        optimizer,
        base_learning_rate,
        lr_scale_for_step(
            step_index=1,
            total_training_steps=total_training_steps,
            warmup_steps=warmup_steps,
            decay_steps=lr_decay_steps,
            scheduler_type=lr_scheduler_type,
            min_lr_ratio=min_lr_ratio,
        ),
    )
    compat_metadata = _resume_compat_metadata(
        config=config,
        selected_variants=selected_variants,
        dist_context=dist_context,
        optimizer=optimizer,
        train_batches_per_epoch=train_batches_per_epoch,
        partition_count=partition_count,
        partition_stats=partition_stats,
        partition_plan_hash=partition_plan_metadata["plan_hash"],
        round_stats=round_stats,
    )
    resume_trainer_state = _load_trainer_state(config.get("resume_from_checkpoint"))
    resume_loop_state = None
    optimizer_step = 0
    global_batch_step = 0
    if resume_trainer_state is not None:
        _validate_resume_compatibility(resume_trainer_state.get("compat", {}), compat_metadata)
        optimizer.load_state_dict(resume_trainer_state["optimizer_state_dict"])
        saved_scheduler = resume_trainer_state["lr_scheduler"]
        lr_scheduler_type = normalize_lr_scheduler_type(saved_scheduler["scheduler_type"])
        base_learning_rate = float(saved_scheduler["base_learning_rate"])
        warmup_steps = int(saved_scheduler["warmup_steps"])
        lr_decay_steps = int(saved_scheduler["lr_decay_steps"])
        min_lr_ratio = float(saved_scheduler["min_lr_ratio"])
        total_training_steps = int(saved_scheduler["total_training_steps"])
        optimizer_step = int(saved_scheduler["optimizer_step"])
        resume_loop_state = resume_trainer_state["loop_state"]
        global_batch_step = int(resume_loop_state["global_batch_step"])
        initial_lr = get_optimizer_lr(optimizer)
        rank_zero_print(
            dist_context,
            "[train] Resuming partitioned training from "
            f"{config['resume_from_checkpoint']} at epoch "
            f"{resume_loop_state['current_epoch']} batch "
            f"{resume_loop_state['completed_epoch_batch_step']}/"
            f"{train_batches_per_epoch}, optimizer_step={optimizer_step}, "
            f"global_batch_step={global_batch_step}",
        )
    eval_step_interval = int(config.get("eval_step_interval", 0) or 0)
    if eval_step_interval < 0:
        raise ValueError(f"eval_step_interval must be >= 0, got {eval_step_interval}")
    step_eval_skip = resolve_step_eval_skip(config)
    wandb_tracking_enabled = wandb_enabled(config)
    wandb_log_every = wandb_log_interval(config) if wandb_tracking_enabled else 10
    env_step_prompt_multiplier = prompt_template_multiplier(config)
    global_prompt_samples = 0
    current_env_steps = 0.0
    global_effective_batch_size = (
        int(config["batch_size"]) * gradient_accumulation_steps * dist_context.world_size
    )
    start_epoch, end_epoch, resume_epoch, resume_completed_epoch_step = _resume_epoch_plan(
        additional_epochs=num_epochs,
        train_batches_per_epoch=train_batches_per_epoch,
        resume_loop_state=resume_loop_state,
    )
    scheduler_meta = _scheduler_metadata(
        scheduler_type=lr_scheduler_type,
        base_learning_rate=base_learning_rate,
        warmup_steps=warmup_steps,
        lr_decay_steps=lr_decay_steps,
        min_lr_ratio=min_lr_ratio,
        total_training_steps=total_training_steps,
        updates_per_epoch=updates_per_epoch,
        optimizer_step=optimizer_step,
    )
    rank_zero_print(
        dist_context,
        "[train] Partitioned optimizer setup: "
        f"dataset_load_partitions={partition_count}, "
        f"train_batches_per_epoch={train_batches_per_epoch}, "
        f"val_batches_per_epoch={val_batches_per_epoch}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"updates_per_epoch={updates_per_epoch}, total_updates={total_training_steps}, "
        f"learning_rate={base_learning_rate}, "
        f"initial_lr={initial_lr:.6g}, lr_scheduler_type={lr_scheduler_type}, "
        f"warmup_steps={warmup_steps}, lr_decay_steps={lr_decay_steps}, "
        f"min_lr_ratio={min_lr_ratio}, "
        f"eval_step_interval={eval_step_interval}, "
        f"step_eval_skip={step_eval_skip}, "
        f"parallel_backend={dist_context.backend}, world_size={dist_context.world_size}, "
        f"global_effective_batch_size={global_effective_batch_size}"
    )
    loss_context = _build_loss_context(config, tokenizer)
    should_run_eval = bool(config.get("eval_num_episodes", 0) > 0 and tokenizer is not None and eval_variants)
    progress_path = _create_training_progress_path(config, dist_context)
    last_loop_state = _loop_state(
        epoch=resume_epoch,
        completed_epoch_batch_step=resume_completed_epoch_step,
        train_batches_per_epoch=train_batches_per_epoch,
        global_batch_step=global_batch_step,
        epoch_step_eval_count=int((resume_loop_state or {}).get("epoch_step_eval_count", 0) or 0),
        next_step_eval_at=(resume_loop_state or {}).get("next_step_eval_at"),
        partition_order_position=(resume_loop_state or {}).get("partition_order_position"),
        partition_index=(resume_loop_state or {}).get("partition_index"),
        completed_partition_batch_step=(resume_loop_state or {}).get("completed_partition_batch_step"),
    )
    for epoch in range(start_epoch, end_epoch + 1):
        progress_context = (
            FileProgress(
                path=progress_path,
                interval_seconds=progress_interval_seconds,
                cleanup_on_success=False,
                print_on_success=False,
            )
            if dist_context.is_main_process
            else contextlib.nullcontext(None)
        )
        with progress_context as progress:
            if progress is not None:
                progress.update(
                    f"Epoch {epoch}/{end_epoch} [starting]",
                    0,
                    train_batches_per_epoch,
                    time.monotonic(),
                    extra="starting epoch",
                    force=True,
                )
            model.train()
            unpatch_continuous_action_forward(raw_model)
            unpatch_mtp_bin_forward(raw_model)
            FastLanguageModel.for_training(raw_model)
            ensure_continuous_action_decoder(raw_model, config)
            ensure_mtp_bin_decoder(raw_model, config)
            total_loss = 0.0
            num_batches = 0
            active_resume = resume_loop_state if epoch == resume_epoch else None
            completed_epoch_step = (
                int(active_resume.get("completed_epoch_batch_step", 0) or 0)
                if active_resume is not None
                else 0
            )
            epoch_optimizer_step = _completed_optimizer_steps_in_epoch(
                completed_epoch_step,
                gradient_accumulation_steps,
                train_batches_per_epoch,
            )
            epoch_batch_step = completed_epoch_step
            next_step_eval_at = (
                active_resume.get("next_step_eval_at")
                if active_resume is not None
                else _initial_epoch_step_eval_at(eval_step_interval)
            )
            epoch_step_eval_count = (
                int(active_resume.get("epoch_step_eval_count", 0) or 0)
                if active_resume is not None
                else 0
            )
            train_start = None
            train_desc = f"Epoch {epoch}/{end_epoch} [train]"
            optimizer.zero_grad(set_to_none=True)
            round_order = _round_order(
                len(round_stats),
                epoch,
                int(config.get("sampling_seed", 0)),
            )
            resume_partition_order_position = (
                active_resume.get("partition_order_position")
                if active_resume is not None
                else None
            )
            resume_completed_partition_step = int(
                (active_resume or {}).get("completed_partition_batch_step", 0) or 0
            )
            for partition_order_position, round_index in enumerate(round_order):
                if active_resume is not None and completed_epoch_step >= train_batches_per_epoch:
                    break
                if (
                    active_resume is not None
                    and resume_partition_order_position is not None
                    and partition_order_position < int(resume_partition_order_position)
                ):
                    continue
                round_stat = round_stats[int(round_index)]
                assignment = _scatter_round_assignment(
                    full_partition_plan,
                    partition_plan_metadata,
                    int(round_index),
                    dist_context,
                )
                partition_index = int(assignment["partition_index"])
                rank_zero_print(
                    dist_context,
                    f"[train] Epoch {epoch}/{end_epoch}: loading shard round "
                    f"{int(round_index) + 1}/{len(round_stats)} "
                    f"partitions={round_stat['partition_indices']} "
                    f"target_batches={round_stat['target_batches']}",
                )
                loading_start = train_start if train_start is not None else time.monotonic()
                if progress is not None:
                    progress.update(
                        f"Epoch {epoch}/{end_epoch} "
                        f"[loading data shard round {int(round_index) + 1}/{len(round_stats)}]",
                        epoch_batch_step,
                        train_batches_per_epoch,
                        loading_start,
                        extra="loading data ...",
                        force=True,
                    )
                train_loader = _build_partition_shard_train_loader(
                    config,
                    tokenizer,
                    selected_variants,
                    dist_context,
                    assignment=assignment,
                    sampler_seed=(
                        int(config.get("sampling_seed", 0))
                        + epoch * partition_count
                        + partition_index
                    ),
                )
                sampler = getattr(train_loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch * partition_count + partition_index)
                else:
                    torch.manual_seed(
                        int(config.get("sampling_seed", 0)) + epoch * partition_count + partition_index
                    )

                for partition_batch_step, batch in enumerate(train_loader, start=1):
                    if (
                        active_resume is not None
                        and resume_partition_order_position is not None
                        and partition_order_position == int(resume_partition_order_position)
                        and partition_batch_step <= resume_completed_partition_step
                    ):
                        continue
                    if train_start is None:
                        train_start = time.monotonic()
                    epoch_batch_step += 1
                    global_batch_step += 1
                    loss, loss_parts = _compute_batch_loss(model, batch, device, loss_context)
                    should_step = (
                        epoch_batch_step % gradient_accumulation_steps == 0
                        or epoch_batch_step == train_batches_per_epoch
                    )
                    sync_context = (
                        model.no_sync()
                        if dist_context.is_distributed and hasattr(model, "no_sync") and not should_step
                        else contextlib.nullcontext()
                    )
                    with sync_context:
                        (loss / gradient_accumulation_steps).backward()
                    if should_step:
                        current_lr = set_optimizer_lr(
                            optimizer,
                            base_learning_rate,
                            lr_scale_for_step(
                                step_index=optimizer_step + 1,
                                total_training_steps=total_training_steps,
                                warmup_steps=warmup_steps,
                                decay_steps=lr_decay_steps,
                                scheduler_type=lr_scheduler_type,
                                min_lr_ratio=min_lr_ratio,
                            ),
                        )
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1
                        epoch_optimizer_step += 1
                    else:
                        current_lr = get_optimizer_lr(optimizer)

                    loss_value = reduce_mean(float(loss.detach().item()), dist_context, device)
                    if wandb_tracking_enabled:
                        global_prompt_samples += global_batch_sample_count(batch, dist_context, device)
                        current_env_steps = global_prompt_samples / env_step_prompt_multiplier
                    total_loss += loss_value
                    num_batches += 1
                    accum_step = ((epoch_batch_step - 1) % gradient_accumulation_steps) + 1
                    loss_extra = _format_loss_extra(loss, loss_parts, display_loss=loss_value)
                    loss_extra += (
                        f" lr={current_lr:.2e} opt_step={epoch_optimizer_step}/{updates_per_epoch} "
                        f"batch_step={global_batch_step} accum={accum_step}/{gradient_accumulation_steps} "
                        f"round={int(round_index) + 1}/{len(round_stats)} "
                        f"partition={partition_index + 1}/{partition_count}"
                    )
                    if progress is not None:
                        progress.update(
                            train_desc,
                            epoch_batch_step,
                            train_batches_per_epoch,
                            train_start,
                            extra=loss_extra,
                        )

                    should_log_wandb_batch = (
                        wandb_tracking_enabled
                        and (
                            global_batch_step == 1
                            or global_batch_step % wandb_log_every == 0
                            or epoch_batch_step == train_batches_per_epoch
                        )
                    )
                    wandb_loss_part_metrics = {}
                    if should_log_wandb_batch:
                        wandb_loss_part_metrics = _reduce_wandb_loss_part_metrics(
                            loss_parts,
                            prefix="train",
                            dist_context=dist_context,
                            device=device,
                        )
                    if wandb_logger.enabled and should_log_wandb_batch:
                        wandb_logger.log(
                            {
                                **wandb_step_metrics(
                                    env_steps=current_env_steps,
                                    batch_step=global_batch_step,
                                    optimizer_step=optimizer_step,
                                    epoch=epoch,
                                ),
                                "train/loss": loss_value,
                                "train/learning_rate": current_lr,
                                **wandb_loss_part_metrics,
                            }
                        )

                    scheduled_epoch_step = None
                    if should_step:
                        scheduled_epoch_step, next_step_eval_at = _consume_epoch_step_eval_trigger(
                            next_step_eval_at=next_step_eval_at,
                            epoch_batch_step=epoch_batch_step,
                            eval_step_interval=eval_step_interval,
                        )
                    if scheduled_epoch_step is not None:
                        scheduled_step = global_batch_step - epoch_batch_step + scheduled_epoch_step
                        skip_reason = _step_eval_epoch_skip_reason(
                            epoch=epoch,
                            epoch_batch_step=epoch_batch_step,
                            train_batches_per_epoch=train_batches_per_epoch,
                            eval_step_interval=eval_step_interval,
                        )
                        epoch_step_eval_count += 1
                        checkpoint_only_reason = _step_eval_checkpoint_only_reason(
                            trigger_count=epoch_step_eval_count,
                            step_eval_skip=step_eval_skip,
                            epoch_skip_reason=skip_reason,
                        )

                        step_train_loss = total_loss / max(num_batches, 1)
                        step_ckpt_dir = get_checkpoint_dir(config, selection_tag, step=global_batch_step)
                        step_val_loss = math.nan
                        step_val_metrics = {}
                        if dist_context.is_main_process:
                            if checkpoint_only_reason is None:
                                step_val_loss, step_val_metrics = _run_validation(
                                    raw_model,
                                    val_loader,
                                    device,
                                    loss_context,
                                    progress,
                                    desc=f"Step {global_batch_step} [val]",
                                    warning_label=f"step {global_batch_step}",
                                )
                                print(
                                    f"[step {global_batch_step}] train_loss={step_train_loss:.4f}  "
                                    f"epoch_step={epoch_batch_step}/{train_batches_per_epoch}  "
                                    f"step_eval_trigger={epoch_step_eval_count}  "
                                    f"val_loss={_format_loss_value(step_val_loss)}  "
                                    f"{_format_optional_metric('val_mae', step_val_metrics.get('mae'))}"
                                    f"optimizer_step={optimizer_step}"
                                )
                                if wandb_logger.enabled:
                                    wandb_logger.log(
                                        {
                                            **wandb_step_metrics(
                                                env_steps=current_env_steps,
                                                batch_step=global_batch_step,
                                                optimizer_step=optimizer_step,
                                                epoch=epoch,
                                            ),
                                            "train/step_loss": step_train_loss,
                                            "val/loss": step_val_loss,
                                            **{
                                                f"val/{key}": float(value)
                                                for key, value in step_val_metrics.items()
                                            },
                                            "train/learning_rate": get_optimizer_lr(optimizer),
                                        }
                                    )
                            else:
                                print(
                                    f"[step {global_batch_step}] checkpoint-only step eval: "
                                    f"train_loss={step_train_loss:.4f}  "
                                    f"epoch_step={epoch_batch_step}/{train_batches_per_epoch}  "
                                    f"step_eval_trigger={epoch_step_eval_count}  "
                                    f"optimizer_step={optimizer_step}  "
                                    f"reason={checkpoint_only_reason}"
                                )
                            scheduler_meta["optimizer_step"] = optimizer_step
                            step_loop_state = _loop_state(
                                epoch=epoch,
                                completed_epoch_batch_step=epoch_batch_step,
                                train_batches_per_epoch=train_batches_per_epoch,
                                global_batch_step=global_batch_step,
                                epoch_step_eval_count=epoch_step_eval_count,
                                next_step_eval_at=next_step_eval_at,
                                partition_order_position=partition_order_position,
                                partition_index=partition_index,
                                completed_partition_batch_step=partition_batch_step,
                            )
                            last_loop_state = step_loop_state
                            _save_checkpoint(
                                config,
                                raw_model,
                                tokenizer,
                                step_ckpt_dir,
                                trainer_state=_build_trainer_state(
                                    config=config,
                                    optimizer=optimizer,
                                    scheduler_meta=scheduler_meta,
                                    loop_state=step_loop_state,
                                    compat_metadata=compat_metadata,
                                ),
                            )
                        if checkpoint_only_reason is None:
                            step_val_loss = broadcast_object(
                                step_val_loss,
                                dist_context,
                            )
                            step_val_metrics = broadcast_object(
                                step_val_metrics,
                                dist_context,
                            )
                            if should_run_eval:
                                _run_eval(
                                    config,
                                    raw_model,
                                    tokenizer,
                                    device,
                                    selection_tag,
                                    eval_variants,
                                    eval_type="step",
                                    train_loss=step_train_loss,
                                    val_loss=step_val_loss,
                                    val_metrics=step_val_metrics,
                                    checkpoint_dir=step_ckpt_dir,
                                    epoch=epoch,
                                    batch_step=global_batch_step,
                                    epoch_step=epoch_batch_step,
                                    optimizer_step=optimizer_step,
                                    scheduled_step=scheduled_step,
                                    scheduled_epoch_step=scheduled_epoch_step,
                                    wandb_logger=wandb_logger,
                                    train_env_steps=current_env_steps,
                                    wandb_batch_step=global_batch_step,
                                    dist_context=dist_context,
                                )
                        barrier(dist_context)
                        model.train()
                        unpatch_continuous_action_forward(raw_model)
                        unpatch_mtp_bin_forward(raw_model)
                        FastLanguageModel.for_training(raw_model)
                        ensure_continuous_action_decoder(raw_model, config)
                        ensure_mtp_bin_decoder(raw_model, config)

                train_loader = None
                _release_loaders()
                barrier(dist_context)

            train_loss = total_loss / max(num_batches, 1)
            val_loss = math.nan
            val_metrics = {}
            if dist_context.is_main_process:
                val_loss, val_metrics = _run_validation(
                    raw_model,
                    val_loader,
                    device,
                    loss_context,
                    progress,
                    desc=f"Epoch {epoch}/{end_epoch} [val]",
                    warning_label=f"epoch {epoch}/{end_epoch}",
                )

        if dist_context.is_main_process:
            print(
                f"[epoch {epoch}/{end_epoch}] train_loss={train_loss:.4f}  "
                f"val_loss={_format_loss_value(val_loss)}"
                f"{_format_optional_metric('val_mae', val_metrics.get('mae'))}"
            )
            if wandb_logger.enabled:
                wandb_logger.log(
                    {
                        **wandb_step_metrics(
                            env_steps=current_env_steps,
                            batch_step=global_batch_step,
                            optimizer_step=optimizer_step,
                            epoch=epoch,
                        ),
                        "train/epoch_loss": train_loss,
                        "val/loss": val_loss,
                        **{
                            f"val/{key}": float(value)
                            for key, value in val_metrics.items()
                        },
                        "train/learning_rate": get_optimizer_lr(optimizer),
                    }
                )

        epoch_ckpt_dir = get_checkpoint_dir(config, selection_tag, epoch=epoch)
        scheduler_meta["optimizer_step"] = optimizer_step
        epoch_loop_state = _loop_state(
            epoch=epoch,
            completed_epoch_batch_step=train_batches_per_epoch,
            train_batches_per_epoch=train_batches_per_epoch,
            global_batch_step=global_batch_step,
            epoch_step_eval_count=epoch_step_eval_count,
            next_step_eval_at=next_step_eval_at,
            partition_order_position=len(round_order),
            partition_index=None,
            completed_partition_batch_step=None,
        )
        last_loop_state = epoch_loop_state
        if dist_context.is_main_process:
            _save_checkpoint(
                config,
                raw_model,
                tokenizer,
                epoch_ckpt_dir,
                trainer_state=_build_trainer_state(
                    config=config,
                    optimizer=optimizer,
                    scheduler_meta=scheduler_meta,
                    loop_state=epoch_loop_state,
                    compat_metadata=compat_metadata,
                ),
            )
        val_loss = broadcast_object(val_loss, dist_context)
        val_metrics = broadcast_object(val_metrics, dist_context)
        if should_run_eval:
            _run_eval(
                config,
                raw_model,
                tokenizer,
                device,
                selection_tag,
                eval_variants,
                eval_type="epoch",
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                checkpoint_dir=epoch_ckpt_dir,
                epoch=epoch,
                optimizer_step=optimizer_step,
                wandb_logger=wandb_logger,
                train_env_steps=current_env_steps,
                wandb_batch_step=global_batch_step,
                dist_context=dist_context,
            )
        barrier(dist_context)
        model.train()
        unpatch_continuous_action_forward(raw_model)
        unpatch_mtp_bin_forward(raw_model)
        FastLanguageModel.for_training(raw_model)
        ensure_continuous_action_decoder(raw_model, config)
        ensure_mtp_bin_decoder(raw_model, config)
    final_ckpt_dir = get_checkpoint_dir(config, selection_tag)
    scheduler_meta["optimizer_step"] = optimizer_step
    if dist_context.is_main_process:
        _save_checkpoint(
            config,
            raw_model,
            tokenizer,
            final_ckpt_dir,
            trainer_state=_build_trainer_state(
                config=config,
                optimizer=optimizer,
                scheduler_meta=scheduler_meta,
                loop_state=last_loop_state,
                compat_metadata=compat_metadata,
            ),
        )
    barrier(dist_context)
    return progress_path


def _run_training(
    config,
    model,
    train_loader,
    val_loader,
    device,
    selected_variants: list[str],
    selection_tag: str,
    progress_interval_seconds: float,
    dist_context: DistributedContext,
    tokenizer=None,
    eval_variants=None,
    wandb_logger: WandbLogger | None = None,
):
    if wandb_logger is None:
        wandb_logger = WandbLogger()
    raw_model = unwrap_model(model)
    base_learning_rate = float(config["learning_rate"])
    optimizer_params, optimizer_info = _build_optimizer_params(model, raw_model, config)
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=base_learning_rate,
    )
    if optimizer_info is not None:
        rank_zero_print(
            dist_context,
            "[train] Continuous action MLP optimizer: "
            f"action_head_weight_decay={optimizer_info['action_head_weight_decay']}, "
            f"decay_params={optimizer_info['action_head_decay_param_count']}, "
            f"non_decay_params={optimizer_info['non_decay_param_count']}",
        )

    num_epochs = int(config["num_epochs"])
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be >= 1, "
            f"got {gradient_accumulation_steps}"
        )
    updates_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_training_steps = max(updates_per_epoch * num_epochs, 1)
    lr_scheduler_type = normalize_lr_scheduler_type(config.get("lr_scheduler_type", "constant"))
    warmup_steps = resolve_warmup_steps(
        config,
        total_training_steps,
        steps_per_epoch=updates_per_epoch,
    )
    lr_decay_steps = resolve_lr_decay_steps(
        config,
        total_training_steps=total_training_steps,
        warmup_steps=warmup_steps,
        steps_per_epoch=updates_per_epoch,
    )
    min_lr_ratio = resolve_min_lr_ratio(config)
    initial_lr = set_optimizer_lr(
        optimizer,
        base_learning_rate,
        lr_scale_for_step(
            step_index=1,
            total_training_steps=total_training_steps,
            warmup_steps=warmup_steps,
            decay_steps=lr_decay_steps,
            scheduler_type=lr_scheduler_type,
            min_lr_ratio=min_lr_ratio,
        ),
    )
    train_total = len(train_loader)
    compat_metadata = _resume_compat_metadata(
        config=config,
        selected_variants=selected_variants,
        dist_context=dist_context,
        optimizer=optimizer,
        train_batches_per_epoch=train_total,
        partition_count=1,
        partition_stats=None,
    )
    resume_trainer_state = _load_trainer_state(config.get("resume_from_checkpoint"))
    resume_loop_state = None
    optimizer_step = 0
    global_batch_step = 0
    if resume_trainer_state is not None:
        _validate_resume_compatibility(resume_trainer_state.get("compat", {}), compat_metadata)
        optimizer.load_state_dict(resume_trainer_state["optimizer_state_dict"])
        saved_scheduler = resume_trainer_state["lr_scheduler"]
        lr_scheduler_type = normalize_lr_scheduler_type(saved_scheduler["scheduler_type"])
        base_learning_rate = float(saved_scheduler["base_learning_rate"])
        warmup_steps = int(saved_scheduler["warmup_steps"])
        lr_decay_steps = int(saved_scheduler["lr_decay_steps"])
        min_lr_ratio = float(saved_scheduler["min_lr_ratio"])
        total_training_steps = int(saved_scheduler["total_training_steps"])
        optimizer_step = int(saved_scheduler["optimizer_step"])
        resume_loop_state = resume_trainer_state["loop_state"]
        global_batch_step = int(resume_loop_state["global_batch_step"])
        initial_lr = get_optimizer_lr(optimizer)
        rank_zero_print(
            dist_context,
            "[train] Resuming training from "
            f"{config['resume_from_checkpoint']} at epoch "
            f"{resume_loop_state['current_epoch']} batch "
            f"{resume_loop_state['completed_epoch_batch_step']}/"
            f"{train_total}, optimizer_step={optimizer_step}, "
            f"global_batch_step={global_batch_step}",
        )
    eval_step_interval = int(config.get("eval_step_interval", 0) or 0)
    if eval_step_interval < 0:
        raise ValueError(f"eval_step_interval must be >= 0, got {eval_step_interval}")
    step_eval_skip = resolve_step_eval_skip(config)
    wandb_tracking_enabled = wandb_enabled(config)
    wandb_log_every = wandb_log_interval(config) if wandb_tracking_enabled else 10
    env_step_prompt_multiplier = prompt_template_multiplier(config)
    global_prompt_samples = 0
    current_env_steps = 0.0
    global_effective_batch_size = (
        int(config["batch_size"]) * gradient_accumulation_steps * dist_context.world_size
    )
    start_epoch, end_epoch, resume_epoch, resume_completed_epoch_step = _resume_epoch_plan(
        additional_epochs=num_epochs,
        train_batches_per_epoch=train_total,
        resume_loop_state=resume_loop_state,
    )
    scheduler_meta = _scheduler_metadata(
        scheduler_type=lr_scheduler_type,
        base_learning_rate=base_learning_rate,
        warmup_steps=warmup_steps,
        lr_decay_steps=lr_decay_steps,
        min_lr_ratio=min_lr_ratio,
        total_training_steps=total_training_steps,
        updates_per_epoch=updates_per_epoch,
        optimizer_step=optimizer_step,
    )
    rank_zero_print(
        dist_context,
        "[train] Optimizer setup: "
        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"updates_per_epoch={updates_per_epoch}, total_updates={total_training_steps}, "
        f"learning_rate={base_learning_rate}, "
        f"initial_lr={initial_lr:.6g}, lr_scheduler_type={lr_scheduler_type}, "
        f"warmup_steps={warmup_steps}, lr_decay_steps={lr_decay_steps}, "
        f"min_lr_ratio={min_lr_ratio}, "
        f"eval_step_interval={eval_step_interval}, "
        f"step_eval_skip={step_eval_skip}, "
        f"parallel_backend={dist_context.backend}, world_size={dist_context.world_size}, "
        f"global_effective_batch_size={global_effective_batch_size}"
    )
    loss_context = _build_loss_context(config, tokenizer)
    should_run_eval = bool(config.get("eval_num_episodes", 0) > 0 and tokenizer is not None and eval_variants)
    progress_path = _create_training_progress_path(config, dist_context)
    last_loop_state = _loop_state(
        epoch=resume_epoch,
        completed_epoch_batch_step=resume_completed_epoch_step,
        train_batches_per_epoch=train_total,
        global_batch_step=global_batch_step,
        epoch_step_eval_count=int((resume_loop_state or {}).get("epoch_step_eval_count", 0) or 0),
        next_step_eval_at=(resume_loop_state or {}).get("next_step_eval_at"),
    )
    for epoch in range(start_epoch, end_epoch + 1):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        else:
            torch.manual_seed(int(config.get("sampling_seed", 0)) + epoch)

        progress_context = (
            FileProgress(
                path=progress_path,
                interval_seconds=progress_interval_seconds,
                cleanup_on_success=False,
                print_on_success=False,
            )
            if dist_context.is_main_process
            else contextlib.nullcontext(None)
        )
        with progress_context as progress:
            model.train()
            unpatch_continuous_action_forward(raw_model)
            unpatch_mtp_bin_forward(raw_model)
            FastLanguageModel.for_training(raw_model)
            ensure_continuous_action_decoder(raw_model, config)
            ensure_mtp_bin_decoder(raw_model, config)
            total_loss = 0.0
            num_batches = 0
            active_resume = resume_loop_state if epoch == resume_epoch else None
            completed_epoch_step = (
                int(active_resume.get("completed_epoch_batch_step", 0) or 0)
                if active_resume is not None
                else 0
            )
            epoch_optimizer_step = _completed_optimizer_steps_in_epoch(
                completed_epoch_step,
                gradient_accumulation_steps,
                train_total,
            )
            next_step_eval_at = (
                active_resume.get("next_step_eval_at")
                if active_resume is not None
                else _initial_epoch_step_eval_at(eval_step_interval)
            )
            epoch_step_eval_count = (
                int(active_resume.get("epoch_step_eval_count", 0) or 0)
                if active_resume is not None
                else 0
            )
            train_start = time.monotonic()
            train_desc = f"Epoch {epoch}/{end_epoch} [train]"
            if progress is not None:
                progress.update(
                    train_desc,
                    completed_epoch_step,
                    train_total,
                    train_start,
                    extra="starting epoch",
                    force=True,
                )
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                if active_resume is not None and step <= completed_epoch_step:
                    continue
                global_batch_step += 1
                loss, loss_parts = _compute_batch_loss(model, batch, device, loss_context)
                should_step = step % gradient_accumulation_steps == 0 or step == train_total
                sync_context = (
                    model.no_sync()
                    if dist_context.is_distributed and hasattr(model, "no_sync") and not should_step
                    else contextlib.nullcontext()
                )
                with sync_context:
                    (loss / gradient_accumulation_steps).backward()
                if should_step:
                    current_lr = set_optimizer_lr(
                        optimizer,
                        base_learning_rate,
                        lr_scale_for_step(
                            step_index=optimizer_step + 1,
                            total_training_steps=total_training_steps,
                            warmup_steps=warmup_steps,
                            decay_steps=lr_decay_steps,
                            scheduler_type=lr_scheduler_type,
                            min_lr_ratio=min_lr_ratio,
                        ),
                    )
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    epoch_optimizer_step += 1
                else:
                    current_lr = get_optimizer_lr(optimizer)

                loss_value = reduce_mean(float(loss.detach().item()), dist_context, device)
                if wandb_tracking_enabled:
                    global_prompt_samples += global_batch_sample_count(batch, dist_context, device)
                    current_env_steps = global_prompt_samples / env_step_prompt_multiplier
                total_loss += loss_value
                num_batches += 1
                accum_step = ((step - 1) % gradient_accumulation_steps) + 1
                loss_extra = _format_loss_extra(loss, loss_parts, display_loss=loss_value)
                loss_extra += (
                    f" lr={current_lr:.2e} opt_step={epoch_optimizer_step}/{updates_per_epoch} "
                    f"batch_step={global_batch_step} accum={accum_step}/{gradient_accumulation_steps}"
                )
                if progress is not None:
                    progress.update(
                        train_desc,
                        step,
                        train_total,
                        train_start,
                        extra=loss_extra,
                    )

                should_log_wandb_batch = (
                    wandb_tracking_enabled
                    and (
                        global_batch_step == 1
                        or global_batch_step % wandb_log_every == 0
                        or step == train_total
                    )
                )
                wandb_loss_part_metrics = {}
                if should_log_wandb_batch:
                    wandb_loss_part_metrics = _reduce_wandb_loss_part_metrics(
                        loss_parts,
                        prefix="train",
                        dist_context=dist_context,
                        device=device,
                    )
                if wandb_logger.enabled and should_log_wandb_batch:
                    wandb_logger.log(
                        {
                            **wandb_step_metrics(
                                env_steps=current_env_steps,
                                batch_step=global_batch_step,
                                optimizer_step=optimizer_step,
                                epoch=epoch,
                            ),
                            "train/loss": loss_value,
                            "train/learning_rate": current_lr,
                            **wandb_loss_part_metrics,
                        }
                    )

                scheduled_epoch_step = None
                if should_step:
                    scheduled_epoch_step, next_step_eval_at = _consume_epoch_step_eval_trigger(
                        next_step_eval_at=next_step_eval_at,
                        epoch_batch_step=step,
                        eval_step_interval=eval_step_interval,
                    )
                if scheduled_epoch_step is not None:
                    scheduled_step = global_batch_step - step + scheduled_epoch_step
                    skip_reason = _step_eval_epoch_skip_reason(
                        epoch=epoch,
                        epoch_batch_step=step,
                        train_batches_per_epoch=train_total,
                        eval_step_interval=eval_step_interval,
                    )
                    epoch_step_eval_count += 1
                    checkpoint_only_reason = _step_eval_checkpoint_only_reason(
                        trigger_count=epoch_step_eval_count,
                        step_eval_skip=step_eval_skip,
                        epoch_skip_reason=skip_reason,
                    )

                    step_train_loss = total_loss / max(num_batches, 1)
                    step_ckpt_dir = get_checkpoint_dir(config, selection_tag, step=global_batch_step)
                    step_val_loss = math.nan
                    step_val_metrics = {}
                    if dist_context.is_main_process:
                        if checkpoint_only_reason is None:
                            step_val_loss, step_val_metrics = _run_validation(
                                raw_model,
                                val_loader,
                                device,
                                loss_context,
                                progress,
                                desc=f"Step {global_batch_step} [val]",
                                warning_label=f"step {global_batch_step}",
                            )
                            print(
                                f"[step {global_batch_step}] train_loss={step_train_loss:.4f}  "
                                f"epoch_step={step}/{train_total}  "
                                f"step_eval_trigger={epoch_step_eval_count}  "
                                f"val_loss={_format_loss_value(step_val_loss)}  "
                                f"{_format_optional_metric('val_mae', step_val_metrics.get('mae'))}"
                                f"optimizer_step={optimizer_step}"
                            )
                            if wandb_logger.enabled:
                                wandb_logger.log(
                                    {
                                        **wandb_step_metrics(
                                            env_steps=current_env_steps,
                                            batch_step=global_batch_step,
                                            optimizer_step=optimizer_step,
                                            epoch=epoch,
                                        ),
                                        "train/step_loss": step_train_loss,
                                        "val/loss": step_val_loss,
                                        **{
                                            f"val/{key}": float(value)
                                            for key, value in step_val_metrics.items()
                                        },
                                        "train/learning_rate": get_optimizer_lr(optimizer),
                                    }
                                )
                        else:
                            print(
                                f"[step {global_batch_step}] checkpoint-only step eval: "
                                f"train_loss={step_train_loss:.4f}  "
                                f"epoch_step={step}/{train_total}  "
                                f"step_eval_trigger={epoch_step_eval_count}  "
                                f"optimizer_step={optimizer_step}  "
                                f"reason={checkpoint_only_reason}"
                            )
                        scheduler_meta["optimizer_step"] = optimizer_step
                        step_loop_state = _loop_state(
                            epoch=epoch,
                            completed_epoch_batch_step=step,
                            train_batches_per_epoch=train_total,
                            global_batch_step=global_batch_step,
                            epoch_step_eval_count=epoch_step_eval_count,
                            next_step_eval_at=next_step_eval_at,
                        )
                        last_loop_state = step_loop_state
                        _save_checkpoint(
                            config,
                            raw_model,
                            tokenizer,
                            step_ckpt_dir,
                            trainer_state=_build_trainer_state(
                                config=config,
                                optimizer=optimizer,
                                scheduler_meta=scheduler_meta,
                                loop_state=step_loop_state,
                                compat_metadata=compat_metadata,
                            ),
                        )
                    if checkpoint_only_reason is None:
                        step_val_loss = broadcast_object(
                            step_val_loss,
                            dist_context,
                        )
                        step_val_metrics = broadcast_object(
                            step_val_metrics,
                            dist_context,
                        )
                        if should_run_eval:
                            _run_eval(
                                config,
                                raw_model,
                                tokenizer,
                                device,
                                selection_tag,
                                eval_variants,
                                eval_type="step",
                                train_loss=step_train_loss,
                                val_loss=step_val_loss,
                                val_metrics=step_val_metrics,
                                checkpoint_dir=step_ckpt_dir,
                                epoch=epoch,
                                batch_step=global_batch_step,
                                epoch_step=step,
                                optimizer_step=optimizer_step,
                                scheduled_step=scheduled_step,
                                scheduled_epoch_step=scheduled_epoch_step,
                                wandb_logger=wandb_logger,
                                train_env_steps=current_env_steps,
                                wandb_batch_step=global_batch_step,
                                dist_context=dist_context,
                            )
                    barrier(dist_context)
                    model.train()
                    unpatch_continuous_action_forward(raw_model)
                    unpatch_mtp_bin_forward(raw_model)
                    FastLanguageModel.for_training(raw_model)
                    ensure_continuous_action_decoder(raw_model, config)
                    ensure_mtp_bin_decoder(raw_model, config)

            train_loss = total_loss / max(num_batches, 1)

            val_loss = math.nan
            val_metrics = {}
            if dist_context.is_main_process:
                val_loss, val_metrics = _run_validation(
                    raw_model,
                    val_loader,
                    device,
                    loss_context,
                    progress,
                    desc=f"Epoch {epoch}/{end_epoch} [val]",
                    warning_label=f"epoch {epoch}/{end_epoch}",
                )

        if dist_context.is_main_process:
            print(
                f"[epoch {epoch}/{end_epoch}] train_loss={train_loss:.4f}  "
                f"val_loss={_format_loss_value(val_loss)}"
                f"{_format_optional_metric('val_mae', val_metrics.get('mae'))}"
            )
            if wandb_logger.enabled:
                wandb_logger.log(
                    {
                        **wandb_step_metrics(
                            env_steps=current_env_steps,
                            batch_step=global_batch_step,
                            optimizer_step=optimizer_step,
                            epoch=epoch,
                        ),
                        "train/epoch_loss": train_loss,
                        "val/loss": val_loss,
                        **{
                            f"val/{key}": float(value)
                            for key, value in val_metrics.items()
                        },
                        "train/learning_rate": get_optimizer_lr(optimizer),
                    }
                )

        epoch_ckpt_dir = get_checkpoint_dir(config, selection_tag, epoch=epoch)
        scheduler_meta["optimizer_step"] = optimizer_step
        epoch_loop_state = _loop_state(
            epoch=epoch,
            completed_epoch_batch_step=train_total,
            train_batches_per_epoch=train_total,
            global_batch_step=global_batch_step,
            epoch_step_eval_count=epoch_step_eval_count,
            next_step_eval_at=next_step_eval_at,
        )
        last_loop_state = epoch_loop_state
        if dist_context.is_main_process:
            _save_checkpoint(
                config,
                raw_model,
                tokenizer,
                epoch_ckpt_dir,
                trainer_state=_build_trainer_state(
                    config=config,
                    optimizer=optimizer,
                    scheduler_meta=scheduler_meta,
                    loop_state=epoch_loop_state,
                    compat_metadata=compat_metadata,
                ),
            )
        val_loss = broadcast_object(val_loss, dist_context)
        val_metrics = broadcast_object(val_metrics, dist_context)
        if should_run_eval:
            _run_eval(
                config,
                raw_model,
                tokenizer,
                device,
                selection_tag,
                eval_variants,
                eval_type="epoch",
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                checkpoint_dir=epoch_ckpt_dir,
                epoch=epoch,
                optimizer_step=optimizer_step,
                wandb_logger=wandb_logger,
                train_env_steps=current_env_steps,
                wandb_batch_step=global_batch_step,
                dist_context=dist_context,
            )
        barrier(dist_context)
        model.train()
        unpatch_continuous_action_forward(raw_model)
        unpatch_mtp_bin_forward(raw_model)
        FastLanguageModel.for_training(raw_model)
        ensure_continuous_action_decoder(raw_model, config)
        ensure_mtp_bin_decoder(raw_model, config)
    final_ckpt_dir = get_checkpoint_dir(config, selection_tag)
    scheduler_meta["optimizer_step"] = optimizer_step
    if dist_context.is_main_process:
        _save_checkpoint(
            config,
            raw_model,
            tokenizer,
            final_ckpt_dir,
            trainer_state=_build_trainer_state(
                config=config,
                optimizer=optimizer,
                scheduler_meta=scheduler_meta,
                loop_state=last_loop_state,
                compat_metadata=compat_metadata,
            ),
        )
    barrier(dist_context)
    return progress_path



def _save_checkpoint(config, model, tokenizer, checkpoint_dir, trainer_state: dict | None = None):
    model = unwrap_model(model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    if uses_continuous_actions(config):
        save_continuous_action_decoder(model, checkpoint_dir)
    if uses_mtp_bin(config):
        save_mtp_bin_decoder(model, checkpoint_dir)
    config_dst = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_dst, "w") as f:
        yaml.dump(config, f)
    if trainer_state is not None:
        torch.save(trainer_state, _trainer_state_path(checkpoint_dir))

    adapter_cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        adapter_cfg["base_model_name_or_path"] = config["model_name"]
        with open(adapter_cfg_path, "w") as f:
            json.dump(adapter_cfg, f, indent=2)

    print(f"[train] Checkpoint saved to: {checkpoint_dir}")


def _print_tokenize_only_step_summary(
    config: dict,
    dist_context: DistributedContext,
    *,
    train_batches_per_epoch: int,
) -> None:
    num_epochs = int(config["num_epochs"])
    total_train_batch_steps = train_batches_per_epoch * num_epochs
    global_samples_per_batch_step = int(config["batch_size"]) * dist_context.world_size
    separator = "=" * 72
    rank_zero_print(dist_context, separator)
    rank_zero_print(dist_context, "[tokenize-only] STEP EVAL INTERVAL PLANNING")
    rank_zero_print(
        dist_context,
        f"[tokenize-only] Train batch steps per epoch: {train_batches_per_epoch}",
    )
    rank_zero_print(
        dist_context,
        f"[tokenize-only] Total train batch steps across {num_epochs} epoch(s): "
        f"{total_train_batch_steps}",
    )
    rank_zero_print(
        dist_context,
        "[tokenize-only] eval_step_interval counts the per-epoch batch steps above "
        "and resets at every epoch.",
    )
    rank_zero_print(
        dist_context,
        "[tokenize-only] Each batch step processes approximately "
        f"batch_size({config['batch_size']}) * world_size({dist_context.world_size}) "
        f"= {global_samples_per_batch_step} global samples.",
    )
    rank_zero_print(dist_context, separator)


def prepare_tokenized_data(
    config: dict,
    tokenizer,
    train_selection: VariantSelection,
    dist_context: DistributedContext,
) -> None:
    if not config.get("dataset_cache_dir"):
        raise ValueError("--tokenize-only requires dataset_cache_dir")

    selected_variants = train_selection.selected_variants
    partition_count = resolve_dataset_load_partitions(config, dist_context)
    rank_zero_print(
        dist_context,
        "[tokenize-only] Preparing tokenized dataset caches for variants: "
        f"{selected_variants}",
    )

    if partition_count > 1:
        val_loader, partition_stats, _full_plan, plan_metadata = _prewarm_partition_caches(
            config,
            tokenizer,
            selected_variants,
            dist_context,
            partition_count=partition_count,
        )
        val_samples = _loader_sample_count(val_loader)
        train_samples = sum(int(stat["train_samples"]) for stat in partition_stats)
        train_batches = sum(
            int(stat["target_batches"]) for stat in plan_metadata["round_stats"]
        )
        val_batches = len(val_loader) if val_loader is not None else 0
        _release_loaders(val_loader)
        rank_zero_print(
            dist_context,
            "[tokenize-only] Complete: "
            f"partitions={partition_count}, train_samples={train_samples}, "
            f"train_batches={train_batches}, val_samples={val_samples}, "
            f"val_batches={val_batches}",
        )
        _print_tokenize_only_step_summary(
            config,
            dist_context,
            train_batches_per_epoch=train_batches,
        )
        barrier(dist_context)
        return

    train_loader, val_loader = build_data_loaders(
        config,
        tokenizer,
        selected_variants,
        dist_context,
    )
    train_samples = _loader_sample_count(train_loader)
    val_samples = _loader_sample_count(val_loader)
    train_batches = len(train_loader) if train_loader is not None else 0
    val_batches = len(val_loader) if val_loader is not None else 0
    _release_loaders(train_loader, val_loader)
    rank_zero_print(
        dist_context,
        "[tokenize-only] Complete: "
        f"partitions=1, train_samples={train_samples}, train_batches={train_batches}, "
        f"val_samples={val_samples}, val_batches={val_batches}",
    )
    _print_tokenize_only_step_summary(
        config,
        dist_context,
        train_batches_per_epoch=train_batches,
    )
    barrier(dist_context)



def train_with_selection(
    config: dict,
    train_selection: VariantSelection,
    eval_selection: VariantSelection,
    model,
    tokenizer,
    device: torch.device,
    dist_context: DistributedContext,
):
    rank_zero_print(dist_context, f"[train] Resolved train variants: {train_selection.selected_variants}")
    rank_zero_print(dist_context, f"[train] Resolved train tag: {train_selection.selection_tag}")
    rank_zero_print(dist_context, f"[train] Resolved eval variants: {eval_selection.selected_variants}")

    dataset_load_partitions = resolve_dataset_load_partitions(config, dist_context)
    progress_interval = float(config.get("progress_interval_seconds", 5.0))
    if dataset_load_partitions > 1:
        val_loader, partition_stats, full_plan, plan_metadata = _prewarm_partition_caches(
            config,
            tokenizer,
            train_selection.selected_variants,
            dist_context,
            partition_count=dataset_load_partitions,
        )
        train_batches_per_epoch = sum(
            int(stat["target_batches"]) for stat in plan_metadata["round_stats"]
        )
        _maybe_prompt_eval_step_interval(config, train_batches_per_epoch, dist_context)
        wandb_logger = init_wandb_logger(config, dist_context)
        try:
            progress_path = _run_training_partitioned(
                config,
                model,
                tokenizer,
                train_selection.selected_variants,
                val_loader,
                device,
                selection_tag=train_selection.selection_tag,
                progress_interval_seconds=progress_interval,
                dist_context=dist_context,
                partition_count=dataset_load_partitions,
                partition_stats=partition_stats,
                full_partition_plan=full_plan,
                partition_plan_metadata=plan_metadata,
                eval_variants=eval_selection.selected_variants,
                wandb_logger=wandb_logger,
            )

            _finish_training_progress(progress_path)
        finally:
            wandb_logger.finish()
        return

    train_loader, val_loader = build_data_loaders(
        config,
        tokenizer,
        train_selection.selected_variants,
        dist_context,
    )
    _maybe_prompt_eval_step_interval(config, train_loader, dist_context)
    wandb_logger = init_wandb_logger(config, dist_context)
    try:
        progress_path = _run_training(
            config,
            model,
            train_loader,
            val_loader,
            device,
            selected_variants=train_selection.selected_variants,
            selection_tag=train_selection.selection_tag,
            progress_interval_seconds=progress_interval,
            dist_context=dist_context,
            tokenizer=tokenizer,
            eval_variants=eval_selection.selected_variants,
            wandb_logger=wandb_logger,
        )

        _finish_training_progress(progress_path)
    finally:
        wandb_logger.finish()



def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.experiment_id is not None:
        experiment_id_override = str(args.experiment_id).strip()
        if not experiment_id_override:
            raise ValueError("--experiment_id must not be empty when provided")
        config["experiment_id"] = experiment_id_override
    if args.resume_from_checkpoint is not None:
        resume_override = str(args.resume_from_checkpoint).strip()
        if not resume_override:
            raise ValueError("--resume_from_checkpoint must not be empty when provided")
        config["resume_from_checkpoint"] = resume_override
    config["train_config_source"] = args.config
    config["tokenize_only"] = bool(args.tokenize_only)
    parallel_backend = resolve_parallel_backend(config, args.parallel_backend)
    dist_context = init_distributed_context(config, parallel_backend)
    resource_monitor = None
    try:
        if "episode_keep_ratio" in config:
            raise ValueError("episode_keep_ratio is no longer supported; use episode_keep_num instead.")
        resume_from_checkpoint = config.get("resume_from_checkpoint")
        if resume_from_checkpoint:
            resume_from_checkpoint = os.path.abspath(str(resume_from_checkpoint))
            config["resume_from_checkpoint"] = resume_from_checkpoint
            if args.tokenize_only:
                raise ValueError("--tokenize-only cannot be combined with resume_from_checkpoint")
            saved_config_path = os.path.join(resume_from_checkpoint, "config.yaml")
            if not os.path.exists(saved_config_path):
                raise FileNotFoundError(
                    f"Cannot resume from {resume_from_checkpoint}: missing config.yaml"
                )
            with open(saved_config_path) as saved_config_file:
                saved_checkpoint_config = yaml.safe_load(saved_config_file) or {}
            config["resume_source_experiment_id"] = saved_checkpoint_config.get("experiment_id")

        if dist_context.is_main_process:
            experiment_id = ensure_experiment_id(config)
        else:
            experiment_id = config.get("experiment_id")
        experiment_id = broadcast_object(experiment_id, dist_context)
        config["experiment_id"] = experiment_id

        resource_monitor = _create_resource_monitor(config, dist_context)
        resource_monitor.start()
        if resource_monitor.enabled:
            rank_zero_print(
                dist_context,
                f"[sys-info] Writing latest system status to: {resource_monitor.path.resolve()}",
            )

        normalize_prompt_config(config)
        dataloader_config = resolve_dataloader_config(config)
        config["dataset_load_partitions"] = resolve_dataset_load_partitions(config, dist_context)
        resolve_step_eval_skip(config)
        config["training_eval_rollout_isolated"] = _as_bool(
            config.get("training_eval_rollout_isolated", False)
        )
        available_variants = get_available_variants(config["env_family"])
        train_selection = resolve_train_selection(config, available_variants)
        eval_selection = resolve_epoch_eval_selection(config, available_variants, train_selection)
        action_dim = get_action_dim(config["env_family"], train_selection.selected_variants)

        config["train_varients"] = train_selection.configured_variants
        config.pop("variants", None)
        config["action_dim"] = action_dim
        action_token_mode = get_action_token_mode(config)
        if action_token_mode == "mtp_bin":
            config["mtp_k"] = resolve_mtp_k(action_dim, config.get("mtp_k"))
            config["mtp_lcm_weight"] = resolve_mtp_lcm_weight(config)
            config["mtp_quadratic_decoding"] = resolve_mtp_quadratic_decoding(config)
        elif action_token_mode == "simple_mtp_bin":
            config.pop("mtp_k", None)
            config["mtp_lcm_weight"] = resolve_mtp_lcm_weight(config)
        if uses_continuous_actions(config):
            config["action_query_len"] = resolve_action_query_len(
                action_dim,
                config.get("action_query_len"),
            )
            config["action_head_num_blocks"] = resolve_action_head_num_blocks(
                config.get("action_head_num_blocks")
            )
            config["action_head_dropout"] = resolve_action_head_dropout(
                config.get("action_head_dropout")
            )
            action_head_weight_decay = resolve_action_head_weight_decay(config)
            if action_head_weight_decay is not None:
                config["action_head_weight_decay"] = action_head_weight_decay
            if action_token_mode in {"parallel_gaussian", "parallel_t"}:
                gaussian_log_std_min, gaussian_log_std_max = resolve_gaussian_log_std_bounds(config)
                config["gaussian_log_std_min"] = gaussian_log_std_min
                config["gaussian_log_std_max"] = gaussian_log_std_max
            if action_token_mode == "parallel_gaussian":
                gaussian_log_std_init = resolve_gaussian_log_std_init(config)
                config["gaussian_log_std_init"] = max(
                    gaussian_log_std_min,
                    min(gaussian_log_std_init, gaussian_log_std_max),
                )
            if action_token_mode == "parallel_t":
                config["student_t_df"] = resolve_student_t_df(config)
                config["continuous_mean_l1_weight"] = resolve_continuous_mean_l1_weight(config)
        config["resolved_train_variants"] = train_selection.selected_variants
        config["train_selection_tag"] = train_selection.selection_tag
        config["resolved_eval_mode"] = eval_selection.mode
        config["resolved_eval_variants"] = eval_selection.selected_variants
        config["world_size"] = dist_context.world_size
        config["global_effective_batch_size"] = (
            int(config["batch_size"])
            * int(config.get("gradient_accumulation_steps", 1))
            * dist_context.world_size
        )

        device = dist_context.device
        rank_zero_print(dist_context, f"[train] Using device: {device}")
        rank_zero_print(dist_context, f"[train] Experiment ID: {experiment_id}")
        rank_zero_print(dist_context, f"[train] Resolved action_dim: {action_dim}")
        rank_zero_print(
            dist_context,
            "[train] DataLoader config: "
            f"num_workers={dataloader_config['num_workers']}, "
            f"pin_memory={dataloader_config['pin_memory']}, "
            f"persistent_workers={dataloader_config['persistent_workers']}, "
            f"prefetch_factor={dataloader_config['prefetch_factor']}, "
            f"non_blocking={dataloader_config['non_blocking']}",
        )
        rank_zero_print(
            dist_context,
            "[train] Dataset tokenization workers per rank: "
            f"{int(config.get('dataset_workers', 8))}",
        )
        if uses_continuous_actions(config):
            continuous_decoder_info = (
                "[train] Resolved continuous decoder: "
                f"mode={action_token_mode}, "
                f"action_query_len={config['action_query_len']}, "
                f"action_head_num_blocks={config['action_head_num_blocks']}, "
                f"action_head_dropout={config['action_head_dropout']}"
            )
            if "action_head_weight_decay" in config:
                continuous_decoder_info = (
                    f"{continuous_decoder_info}, "
                    f"action_head_weight_decay={config['action_head_weight_decay']}"
                )
            if action_token_mode == "parallel_gaussian":
                continuous_decoder_info = (
                    f"{continuous_decoder_info}, "
                    f"gaussian_log_std_init={config['gaussian_log_std_init']}, "
                    f"gaussian_log_std_bounds=("
                    f"{config['gaussian_log_std_min']}, {config['gaussian_log_std_max']})"
                )
            if action_token_mode == "parallel_t":
                continuous_decoder_info = (
                    f"{continuous_decoder_info}, student_t_df={config['student_t_df']}, "
                    f"continuous_mean_l1_weight={config['continuous_mean_l1_weight']}"
                )
            rank_zero_print(
                dist_context,
                continuous_decoder_info,
            )
        if uses_mtp_bin(config):
            if action_token_mode == "simple_mtp_bin":
                rank_zero_print(
                    dist_context,
                    "[train] Resolved simple_mtp_bin decoder: "
                    f"action_queries={action_dim}, mtp_lcm_weight={config['mtp_lcm_weight']}",
                )
            else:
                rank_zero_print(
                    dist_context,
                    "[train] Resolved mtp_bin decoder: "
                    f"mtp_k={config['mtp_k']}, mtp_lcm_weight={config['mtp_lcm_weight']}, "
                    f"mtp_quadratic_decoding={config['mtp_quadratic_decoding']}",
                )
        rank_zero_print(
            dist_context,
            "[train] Parallel setup: "
            f"backend={dist_context.backend}, world_size={dist_context.world_size}, "
            f"rank={dist_context.rank}, local_rank={dist_context.local_rank}",
        )
        if dist_context.is_main_process:
            exp_snapshot_paths = save_experiment_config_snapshot(config)
            print(f"[train] Experiment config saved: {exp_snapshot_paths['config']}")
            print(f"[train] Experiment git metadata saved: {exp_snapshot_paths['git']}")
            print(f"[train] Experiment dirty patch saved: {exp_snapshot_paths['patch']}")
        barrier(dist_context)

        if config.get("resume_from_checkpoint"):
            model, tokenizer = load_model_and_tokenizer_for_training_checkpoint(
                config["resume_from_checkpoint"],
                config,
                load_in_4bit=config.get("load_in_4bit"),
            )
        else:
            model, tokenizer = load_model_and_tokenizer(config)
        if args.tokenize_only:
            prepare_tokenized_data(
                config,
                tokenizer,
                train_selection,
                dist_context,
            )
            if resource_monitor is not None:
                resource_monitor.stop(final_status="stopped")
            return

        model.to(device)

        if dist_context.is_distributed:
            _assert_model_on_device(model, device)
            model = DistributedDataParallel(
                model,
                device_ids=[dist_context.local_rank],
                output_device=dist_context.local_rank,
                find_unused_parameters=_as_bool(config.get("ddp_find_unused_parameters", False)),
            )

        train_with_selection(
            config,
            train_selection,
            eval_selection,
            model,
            tokenizer,
            device,
            dist_context,
        )
        if resource_monitor is not None:
            resource_monitor.stop(final_status="stopped")
    finally:
        if resource_monitor is not None:
            resource_monitor.stop(final_status=None)
        cleanup_distributed(dist_context)


if __name__ == "__main__":
    main()
