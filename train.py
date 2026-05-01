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
    uses_action_bins,
)
from utils.file_progress import FileProgress
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



def get_checkpoint_dir(config: dict, selection_tag: str, epoch: int | None = None) -> str:
    slug = get_model_slug(config["model_name"])
    root = config.get("checkpoint_root", "checkpoints")
    experiment_id = config["experiment_id"]
    tag = f"ep{epoch}" if epoch is not None else "final"
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


def get_eval_variant_results_dir(config: dict, train_selection_tag: str, variant: str, epoch: int) -> str:
    return os.path.join(
        get_eval_epoch_results_dir(config, train_selection_tag, epoch),
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
        prompt_templete_index=config.get("prompt_templete_index", config.get("prompt_template_index")),
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



def _run_epoch_eval(config, model, tokenizer, device, train_selection_tag: str, variants, epoch, train_loss, val_loss):
    import gymnasium_robotics  # noqa: F401
    from evaluate import configure_mujoco_gl, evaluate_variant
    from utils.prompt_loader import load_templates

    eval_config = {
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
        "action_token_mode": config.get("action_token_mode", "text"),
        "action_num_bins": config.get("action_num_bins", 10),
        "action_bin_min": config.get("action_bin_min", -1.0),
        "action_bin_max": config.get("action_bin_max", 1.0),
    }

    configure_mujoco_gl(eval_config)
    model.eval()
    FastLanguageModel.for_inference(model)

    for variant in variants:
        print(f"[eval] Epoch {epoch} | variant: {variant}")
        templates = load_templates(config["env_family"])
        results_dir = get_eval_variant_results_dir(config, train_selection_tag, variant, epoch)
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, "result.json")

        result = evaluate_variant(
            eval_config,
            variant,
            model,
            tokenizer,
            device,
            templates[0],
            variant_results_dir=results_dir,
        )
        result["train_loss"] = train_loss
        result["val_loss"] = val_loss
        result["experiment_id"] = config["experiment_id"]
        result["result_path"] = result_path

        print(
            f"[eval] {variant}: mean_return={result['mean_return']:.4f}, "
            f"success_rate={result['success_rate']:.2%}, "
            f"mean_steps={result['mean_episode_steps']:.1f}, "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval] Saved: {result_path}")

    model.train()
    FastLanguageModel.for_training(model)



def _run_training(config, model, train_loader, val_loader, device,
                  selection_tag: str, progress: FileProgress, tokenizer=None, eval_variants=None):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )

    num_epochs = config["num_epochs"]
    action_token_mode = get_action_token_mode(config)
    bin_token_ids = None
    if action_token_mode == "gaussian_bin":
        bin_token_ids = get_action_bin_token_ids(tokenizer, config)
        action_num_bins = get_action_num_bins(config)
        action_sigma = float(config.get("action_soft_label_sigma", 1.0))
        action_loss_weight = float(config.get("action_loss_weight", 1.0))
        action_stop_loss_weight = float(config.get("action_stop_loss_weight", 1.0))
        action_soft_label_radius = config.get("action_soft_label_radius")
        if action_soft_label_radius is not None:
            action_soft_label_radius = int(action_soft_label_radius)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        train_total = len(train_loader)
        train_start = time.monotonic()
        train_desc = f"Epoch {epoch}/{num_epochs} [train]"
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            if action_token_mode == "gaussian_bin":
                action_bin_labels = batch["action_bin_labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss, loss_parts = gaussian_action_loss(
                    outputs.logits,
                    labels,
                    action_bin_labels,
                    bin_token_ids,
                    action_num_bins,
                    action_sigma,
                    action_loss_weight=action_loss_weight,
                    stop_loss_weight=action_stop_loss_weight,
                    soft_label_radius=action_soft_label_radius,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss_parts = None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress.update(
                train_desc,
                step,
                train_total,
                train_start,
                extra=(
                    f"loss={loss.item():.4f}"
                    if loss_parts is None
                    else (
                        f"loss={loss.item():.4f} action={loss_parts['action_loss']:.6f} "
                        f"stop={loss_parts['stop_loss']:.4f}"
                    )
                ),
            )

        train_loss = total_loss / max(num_batches, 1)

        model.eval()
        val_total = len(val_loader)
        if val_total == 0:
            print(f"[epoch {epoch}/{num_epochs}] WARNING: val_loader is empty; val_loss will be reported as NaN.")
            val_loss = math.nan
            val_batches = 0
        else:
            val_loss = 0.0
            val_batches = 0
        val_start = time.monotonic()
        val_desc = f"Epoch {epoch}/{num_epochs} [val]"
        with torch.no_grad():
            for step, batch in enumerate(val_loader, start=1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                if action_token_mode == "gaussian_bin":
                    action_bin_labels = batch["action_bin_labels"].to(device)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss, loss_parts = gaussian_action_loss(
                        outputs.logits,
                        labels,
                        action_bin_labels,
                        bin_token_ids,
                        action_num_bins,
                        action_sigma,
                        action_loss_weight=action_loss_weight,
                        stop_loss_weight=action_stop_loss_weight,
                        soft_label_radius=action_soft_label_radius,
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    loss_parts = None
                val_loss += loss.item()
                val_batches += 1
                progress.update(
                    val_desc,
                    step,
                    val_total,
                    val_start,
                    extra=(
                        f"loss={loss.item():.4f}"
                        if loss_parts is None
                        else (
                            f"loss={loss.item():.4f} action={loss_parts['action_loss']:.6f} "
                            f"stop={loss_parts['stop_loss']:.4f}"
                        )
                    ),
                )

        if val_batches > 0:
            val_loss = val_loss / val_batches
        print(f"[epoch {epoch}/{num_epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        epoch_ckpt_dir = get_checkpoint_dir(config, selection_tag, epoch=epoch)
        _save_checkpoint(config, model, tokenizer, epoch_ckpt_dir)

        if config.get("eval_num_episodes", 0) > 0 and tokenizer is not None and eval_variants:
            _run_epoch_eval(
                config,
                model,
                tokenizer,
                device,
                selection_tag,
                eval_variants,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )



def _save_checkpoint(config, model, tokenizer, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if uses_action_bins(config):
        model.save_pretrained(checkpoint_dir, save_embedding_layers=False)
    else:
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
    progress_interval = float(config.get("progress_interval_seconds", 5.0))
    with FileProgress(interval_seconds=progress_interval) as progress:
        print(f"[train] Progress in file: {progress.path.resolve()}")
        _run_training(
            config,
            model,
            train_loader,
            val_loader,
            device,
            selection_tag=train_selection.selection_tag,
            progress=progress,
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
