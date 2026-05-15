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

import yaml
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset

with contextlib.redirect_stdout(io.StringIO()):
    from unsloth import FastLanguageModel

from data.base_dataset import DatasetBuildRequest
from data.registry import get_dataset
from model.policy import load_model_and_tokenizer, get_model_slug
from utils.action_bins import (
    gaussian_action_loss,
    get_action_bin_token_ids,
    get_action_num_bins,
    get_action_token_mode,
)
from utils.file_progress import FileProgress
from utils.prompt_loader import load_template_names
from utils.variant_selection import resolve_selection, VariantSelection, get_available_variants


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()



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



def build_dataset_request(config: dict, tokenizer, variant: str, split: str) -> DatasetBuildRequest:
    return DatasetBuildRequest(
        variant=variant,
        split=split,
        tokenizer=tokenizer,
        tokenizer_name_or_path=config["model_name"],
        max_length=config["max_length"],
        num_workers=config.get("dataset_workers", 8),
        cache_dir=config.get("dataset_cache_dir"),
        max_data_num=config.get("max_data_num"),
        prompt_template_count=config.get("prompt_template_count", 1),
        prompt_templete_index=config.get("prompt_templete_index"),
        train_data_ratio=config.get("train_data_ratio", 0.9),
        episode_keep_num=config.get("episode_keep_num"),
        balance_variant_episode_count=config.get("balance_variant_episode_count", False),
        balanced_train_episode_count=config.get("balanced_train_episode_count"),
        sampling_seed=config.get("sampling_seed", 0),
        history_num=config.get("history_num", 0),
        history_stride=config.get("history_stride", 1),
        action_token_mode=config.get("action_token_mode", "text"),
        action_num_bins=config.get("action_num_bins", 10),
        action_bin_min=config.get("action_bin_min", -1.0),
        action_bin_max=config.get("action_bin_max", 1.0),
        new_token=config.get("new_token", False),
        progress_interval_seconds=config.get("progress_interval_seconds", 5.0),
    )


def _resolve_balanced_train_episode_count(config: dict, dataset_cls, selected_variants: list[str]) -> int | None:
    balance_enabled = config.get("balance_variant_episode_count", False)
    if len(selected_variants) <= 1:
        if balance_enabled:
            print("[train] balance_variant_episode_count=true but only one variant is selected; skipping balancing.")
        return None

    if not balance_enabled:
        return None

    keep_num = config.get("episode_keep_num")
    variant_stats = [
        dataset_cls.collect_variant_episode_stats(variant, keep_num)
        for variant in selected_variants
    ]
    balanced_target = min(stat["sampled_episode_target"] for stat in variant_stats)
    stats_text = ", ".join(
        f"{stat['variant']}: total_episodes={stat['total_episodes']}, "
        f"sampled_episode_target={stat['sampled_episode_target']}"
        for stat in variant_stats
    )
    print(f"[train] Multi-variant episode balance stats -> {stats_text}")
    print(f"[train] Balanced sampled episode target across variants: {balanced_target}")
    return balanced_target


def build_data_loaders(config: dict, tokenizer, selected_variants: list[str]):
    train_datasets = []
    val_datasets = []
    dataset_cls = get_dataset(config["env_family"])
    dataset_config = dict(config)
    dataset_config["balanced_train_episode_count"] = _resolve_balanced_train_episode_count(
        config, dataset_cls, selected_variants
    )

    dataset_jobs = []
    dataset_requests = []

    for variant in selected_variants:
        print(f"[train] Loading data for variant: {variant}")
        for split in ("train", "val"):
            dataset_jobs.append((variant, split))
            dataset_requests.append(build_dataset_request(dataset_config, tokenizer, variant, split))

    datasets = dataset_cls.build_batch(dataset_requests)
    collate_fn = dataset_cls.collate_fn

    for (_, split), dataset in zip(dataset_jobs, datasets):
        if split == "train":
            train_datasets.append(dataset)
        elif split == "val":
            val_datasets.append(dataset)

    if len(selected_variants) == 1:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
        print(f"[train] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader

    weights = []
    for ds in train_datasets:
        n = len(ds)
        w = 1.0 / n if n > 0 else 0.0
        weights.extend([w] * n)

    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(combined_train),
        replacement=True,
    )
    train_loader = DataLoader(
        combined_train,
        batch_size=config["batch_size"],
        sampler=sampler,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"[train] Joint train samples: {len(combined_train)}, Val samples: {len(combined_val)}")
    return train_loader, val_loader



def _build_training_eval_config(config: dict) -> dict:
    return {
        "env_family": config["env_family"],
        "num_episodes": config["eval_num_episodes"],
        "parse_retry_limit": config.get("parse_retry_limit", 3),
        "env_kwargs": config.get("eval_env_kwargs", {"continuing_task": False}),
        "history_num": config.get("history_num", 0),
        "history_stride": config.get("history_stride", 1),
        "record_video": config.get("record_video", False),
        "record_all": config.get("record_all", False),
        "video_episode_index": config.get("video_episode_index", 0),
        "video_fps": config.get("video_fps", 20),
        "video_format": config.get("video_format", "gif"),
        "mujoco_gl": config.get("mujoco_gl"),
        "record_step_logs": config.get("record_step_logs", True),
        "action_sampling": config.get("action_sampling", False),
        "action_temperature": config.get("action_temperature", 1.0),
        "action_top_p": config.get("action_top_p", 1.0),
        "action_top_k": config.get("action_top_k", 0),
        "action_token_mode": config.get("action_token_mode", "text"),
        "action_num_bins": config.get("action_num_bins", 10),
        "action_bin_min": config.get("action_bin_min", -1.0),
        "action_bin_max": config.get("action_bin_max", 1.0),
        "new_token": config.get("new_token", False),
    }


def _format_loss_value(value) -> str:
    if value is None:
        return "nan"
    try:
        if not math.isfinite(value):
            return "nan"
    except TypeError:
        return str(value)
    return f"{value:.4f}"


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
    epoch: int | None = None,
    batch_step: int | None = None,
    optimizer_step: int | None = None,
    scheduled_step: int | None = None,
):
    import gymnasium_robotics  # noqa: F401
    from evaluate import configure_mujoco_gl, evaluate_variant
    from utils.prompt_loader import load_named_templates

    eval_config = _build_training_eval_config(config)
    configure_mujoco_gl(eval_config)
    prompt_name = config["prompt_templete_index"][0]
    template = load_named_templates(config["env_family"], [prompt_name])[0]
    eval_tag = f"step{batch_step}" if eval_type == "step" else f"epoch_{epoch}"
    label = f"Step {batch_step}" if eval_type == "step" else f"Epoch {epoch}"
    if eval_type == "step" and scheduled_step is not None and scheduled_step != batch_step:
        label = f"{label} (scheduled at batch step {scheduled_step})"

    model.eval()
    FastLanguageModel.for_inference(model)

    try:
        for variant in variants:
            print(f"[eval] {label} | variant: {variant}")
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
            result["train_loss"] = train_loss
            result["val_loss"] = val_loss
            result["experiment_id"] = config["experiment_id"]
            result["result_path"] = result_path
            result["eval_type"] = eval_type
            result["eval_tag"] = eval_tag
            result["epoch"] = epoch
            result["batch_step"] = batch_step
            result["optimizer_step"] = optimizer_step
            result["scheduled_step"] = scheduled_step
            result["checkpoint_path"] = checkpoint_dir

            print(
                f"[eval] {variant}: mean_return={result['mean_return']:.4f}, "
                f"success_rate={result['success_rate']:.2%}, "
                f"mean_steps={result['mean_episode_steps']:.1f}, "
                f"train_loss={_format_loss_value(train_loss)}, "
                f"val_loss={_format_loss_value(val_loss)}"
            )

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[eval] Saved: {result_path}")
    finally:
        model.train()
        FastLanguageModel.for_training(model)


def _compute_batch_loss(model, batch, device, loss_context: dict):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    if loss_context["action_token_mode"] == "gaussian_bin":
        action_bin_labels = batch["action_bin_labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return gaussian_action_loss(
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

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    return outputs.loss, None


def _format_loss_extra(loss, loss_parts) -> str:
    if loss_parts is None:
        return f"loss={loss.item():.4f}"
    return (
        f"loss={loss.item():.4f} action={loss_parts['action_loss']:.6f} "
        f"stop={loss_parts['stop_loss']:.4f}"
    )


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
        return math.nan

    val_loss = 0.0
    val_batches = 0
    val_start = time.monotonic()
    with torch.no_grad():
        for step, batch in enumerate(val_loader, start=1):
            loss, loss_parts = _compute_batch_loss(model, batch, device, loss_context)
            val_loss += loss.item()
            val_batches += 1
            if progress is not None:
                progress.update(
                    desc,
                    step,
                    val_total,
                    val_start,
                    extra=_format_loss_extra(loss, loss_parts),
                )

    return val_loss / max(val_batches, 1)


def _maybe_prompt_eval_step_interval(config: dict, train_loader) -> None:
    eval_step_interval = int(config.get("eval_step_interval", 0) or 0)
    if eval_step_interval != 0:
        return

    batches_per_epoch = len(train_loader)
    num_epochs = int(config["num_epochs"])
    total_batches = batches_per_epoch * num_epochs
    print(
        "[train] eval_step_interval=0. "
        f"train batches per epoch={batches_per_epoch}, total train batches={total_batches}."
    )
    if not sys.stdin.isatty():
        print("[train] Non-interactive stdin detected; keeping eval_step_interval disabled.")
        return

    answer = input(
        "[train] Enter eval_step_interval to enable step eval, "
        "or press Enter/0 to keep disabled: "
    ).strip()
    if not answer or answer == "0":
        print("[train] Keeping eval_step_interval disabled.")
        return
    try:
        selected_interval = int(answer)
    except ValueError:
        print(f"[train] Invalid eval_step_interval {answer!r}; keeping disabled.")
        return
    if selected_interval == 0:
        print("[train] Keeping eval_step_interval disabled.")
        return
    if selected_interval < 0:
        print(f"[train] eval_step_interval must be >= 0, got {selected_interval}; keeping disabled.")
        return

    config["eval_step_interval"] = selected_interval
    print(f"[train] Using eval_step_interval={selected_interval}.")



def _run_training(config, model, train_loader, val_loader, device,
                  selection_tag: str, progress_interval_seconds: float, tokenizer=None, eval_variants=None):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )

    num_epochs = config["num_epochs"]
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be >= 1, "
            f"got {gradient_accumulation_steps}"
        )
    updates_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_training_steps = max(updates_per_epoch * num_epochs, 1)
    optimizer_step = 0
    global_batch_step = 0
    eval_step_interval = int(config.get("eval_step_interval", 0) or 0)
    if eval_step_interval < 0:
        raise ValueError(f"eval_step_interval must be >= 0, got {eval_step_interval}")
    next_step_eval_at = eval_step_interval if eval_step_interval > 0 else None
    print(
        "[train] Optimizer setup: "
        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"updates_per_epoch={updates_per_epoch}, total_updates={total_training_steps}, "
        f"learning_rate={config['learning_rate']}, "
        f"eval_step_interval={eval_step_interval}"
    )
    action_token_mode = get_action_token_mode(config)
    bin_token_ids = None
    action_num_bins = None
    action_sigma = None
    action_loss_weight = None
    action_stop_loss_weight = None
    action_soft_label_radius = None
    if action_token_mode == "gaussian_bin":
        bin_token_ids = get_action_bin_token_ids(tokenizer, config)
        action_num_bins = get_action_num_bins(config)
        action_sigma = float(config.get("action_soft_label_sigma", 1.0))
        action_loss_weight = float(config.get("action_loss_weight", 1.0))
        action_stop_loss_weight = float(config.get("action_stop_loss_weight", 1.0))
        action_soft_label_radius = config.get("action_soft_label_radius")
        if action_soft_label_radius is not None:
            action_soft_label_radius = int(action_soft_label_radius)
    loss_context = {
        "action_token_mode": action_token_mode,
        "bin_token_ids": bin_token_ids,
        "action_num_bins": action_num_bins,
        "action_sigma": action_sigma,
        "action_loss_weight": action_loss_weight,
        "action_stop_loss_weight": action_stop_loss_weight,
        "action_soft_label_radius": action_soft_label_radius,
    }
    should_run_eval = bool(config.get("eval_num_episodes", 0) > 0 and tokenizer is not None and eval_variants)
    for epoch in range(1, num_epochs + 1):
        with FileProgress(interval_seconds=progress_interval_seconds) as progress:
            print(f"[train] Epoch {epoch}/{num_epochs} progress in file: {progress.path.resolve()}")
            model.train()
            total_loss = 0.0
            num_batches = 0
            epoch_optimizer_step = 0
            train_total = len(train_loader)
            train_start = time.monotonic()
            train_desc = f"Epoch {epoch}/{num_epochs} [train]"
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                global_batch_step += 1
                loss, loss_parts = _compute_batch_loss(model, batch, device, loss_context)
                (loss / gradient_accumulation_steps).backward()
                should_step = step % gradient_accumulation_steps == 0 or step == train_total
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    epoch_optimizer_step += 1

                total_loss += loss.item()
                num_batches += 1
                accum_step = ((step - 1) % gradient_accumulation_steps) + 1
                loss_extra = _format_loss_extra(loss, loss_parts)
                loss_extra += (
                    f" lr={config['learning_rate']:.2e} opt_step={epoch_optimizer_step}/{updates_per_epoch} "
                    f"batch_step={global_batch_step} accum={accum_step}/{gradient_accumulation_steps}"
                )
                progress.update(
                    train_desc,
                    step,
                    train_total,
                    train_start,
                    extra=loss_extra,
                )

                if (
                    should_step
                    and next_step_eval_at is not None
                    and global_batch_step >= next_step_eval_at
                ):
                    scheduled_step = next_step_eval_at
                    while next_step_eval_at <= global_batch_step:
                        next_step_eval_at += eval_step_interval
                    if step == train_total:
                        print(
                            f"[step {global_batch_step}] step eval skipped because it coincides "
                            f"with epoch {epoch} end; epoch eval will run instead."
                        )
                        continue

                    step_train_loss = total_loss / max(num_batches, 1)
                    step_val_loss = _run_validation(
                        model,
                        val_loader,
                        device,
                        loss_context,
                        progress,
                        desc=f"Step {global_batch_step} [val]",
                        warning_label=f"step {global_batch_step}",
                    )
                    print(
                        f"[step {global_batch_step}] train_loss={step_train_loss:.4f}  "
                        f"val_loss={_format_loss_value(step_val_loss)}  "
                        f"optimizer_step={optimizer_step}"
                    )
                    step_ckpt_dir = get_checkpoint_dir(config, selection_tag, step=global_batch_step)
                    _save_checkpoint(config, model, tokenizer, step_ckpt_dir)
                    if should_run_eval:
                        _run_eval(
                            config,
                            model,
                            tokenizer,
                            device,
                            selection_tag,
                            eval_variants,
                            eval_type="step",
                            train_loss=step_train_loss,
                            val_loss=step_val_loss,
                            checkpoint_dir=step_ckpt_dir,
                            epoch=epoch,
                            batch_step=global_batch_step,
                            optimizer_step=optimizer_step,
                            scheduled_step=scheduled_step,
                        )
                    else:
                        model.train()

            train_loss = total_loss / max(num_batches, 1)

            val_loss = _run_validation(
                model,
                val_loader,
                device,
                loss_context,
                progress,
                desc=f"Epoch {epoch}/{num_epochs} [val]",
                warning_label=f"epoch {epoch}/{num_epochs}",
            )

        print(
            f"[epoch {epoch}/{num_epochs}] train_loss={train_loss:.4f}  "
            f"val_loss={_format_loss_value(val_loss)}"
        )

        epoch_ckpt_dir = get_checkpoint_dir(config, selection_tag, epoch=epoch)
        _save_checkpoint(config, model, tokenizer, epoch_ckpt_dir)
        if should_run_eval:
            _run_eval(
                config,
                model,
                tokenizer,
                device,
                selection_tag,
                eval_variants,
                eval_type="epoch",
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_dir=epoch_ckpt_dir,
                epoch=epoch,
                optimizer_step=optimizer_step,
            )
        else:
            model.train()



def _save_checkpoint(config, model, tokenizer, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    config_dst = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_dst, "w") as f:
        yaml.dump(config, f)

    adapter_cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        adapter_cfg["base_model_name_or_path"] = config["model_name"]
        with open(adapter_cfg_path, "w") as f:
            json.dump(adapter_cfg, f, indent=2)

    print(f"[train] Checkpoint saved to: {checkpoint_dir}")



def train_with_selection(
    config: dict,
    train_selection: VariantSelection,
    eval_selection: VariantSelection,
    model,
    tokenizer,
    device: torch.device,
):
    print(f"[train] Resolved train variants: {train_selection.selected_variants}")
    print(f"[train] Resolved train tag: {train_selection.selection_tag}")
    print(f"[train] Resolved eval variants: {eval_selection.selected_variants}")

    train_loader, val_loader = build_data_loaders(config, tokenizer, train_selection.selected_variants)
    _maybe_prompt_eval_step_interval(config, train_loader)
    progress_interval = float(config.get("progress_interval_seconds", 5.0))
    _run_training(
        config,
        model,
        train_loader,
        val_loader,
        device,
        selection_tag=train_selection.selection_tag,
        progress_interval_seconds=progress_interval,
        tokenizer=tokenizer,
        eval_variants=eval_selection.selected_variants,
    )

    checkpoint_dir = get_checkpoint_dir(config, train_selection.selection_tag)
    _save_checkpoint(config, model, tokenizer, checkpoint_dir)



def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if "episode_keep_ratio" in config:
        raise ValueError("episode_keep_ratio is no longer supported; use episode_keep_num instead.")

    experiment_id = ensure_experiment_id(config)
    normalize_prompt_config(config)
    available_variants = get_available_variants(config["env_family"])
    train_selection = resolve_train_selection(config, available_variants)
    eval_selection = resolve_epoch_eval_selection(config, available_variants, train_selection)

    config["train_varients"] = train_selection.configured_variants
    config.pop("variants", None)
    config["resolved_train_variants"] = train_selection.selected_variants
    config["train_selection_tag"] = train_selection.selection_tag
    config["resolved_eval_mode"] = eval_selection.mode
    config["resolved_eval_variants"] = eval_selection.selected_variants

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")
    print(f"[train] Experiment ID: {experiment_id}")

    model, tokenizer = load_model_and_tokenizer(config)
    model.to(device)

    train_with_selection(config, train_selection, eval_selection, model, tokenizer, device)


if __name__ == "__main__":
    main()
