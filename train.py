"""Training entry point for LLM offline RL (behavior cloning).

Usage:
    python train.py --config config.yaml
"""

from unsloth import FastLanguageModel
import argparse
import uuid
import os
import json
import time

import yaml
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset

from data.registry import get_dataset
from data.pointmaze.dataset import collate_fn
from model.policy import load_model_and_tokenizer, get_model_slug
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


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _render_progress(desc: str, current: int, total: int, start_time: float, extra: str = "") -> str:
    total = max(total, 1)
    current = min(current, total)
    ratio = current / total
    width = 24
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.monotonic() - start_time, 0.0)
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 else 0.0
    elapsed_text = _format_duration(elapsed)
    eta_text = _format_duration(remaining)
    suffix = f" {extra}" if extra else ""
    return (
        f"{desc} [{bar}] {ratio * 100:6.2f}% "
        f"{current}/{total} elapsed={elapsed_text} eta={eta_text}{suffix}"
    )


def _print_progress(
    desc: str,
    current: int,
    total: int,
    start_time: float,
    *,
    extra: str = "",
    done: bool = False,
):
    line = _render_progress(desc, current, total, start_time, extra=extra)
    print(line, end="\n" if done else "\r", flush=True)


def _should_log_progress(step: int, total_steps: int, last_log_time: float, interval_seconds: float) -> bool:
    if step == 1 or step == total_steps:
        return True
    return (time.monotonic() - last_log_time) >= interval_seconds



def resolve_train_selection(config: dict, available_variants: list[str]) -> VariantSelection:
    return resolve_selection(
        mode=config["train_mode"],
        variants=config.get("variants"),
        available_variants=available_variants,
        field_name="variants",
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



def get_eval_results_dir(config: dict, train_selection_tag: str, variant: str) -> str:
    slug = get_model_slug(config["model_name"])
    env_family = config["env_family"]
    experiment_id = config["experiment_id"]
    train_tag = f"train={env_family}-{train_selection_tag}"
    exp_tag = f"exp={experiment_id}"
    eval_tag = f"eval={env_family}-{variant}"
    return os.path.join("results", slug, train_tag, exp_tag, eval_tag)



def build_dataset(config: dict, tokenizer, variant: str, split: str):
    dataset_cls = get_dataset(config["env_family"])
    return dataset_cls(
        variant=variant,
        split=split,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        num_workers=config.get("dataset_workers", 8),
        cache_dir=config.get("dataset_cache_dir"),
        max_data_num=config.get("max_data_num"),
        prompt_template_count=config.get("prompt_template_count", 1),
        train_data_ratio=config.get("train_data_ratio", 0.9),
        history_num=config.get("history_num", 0),
        history_stride=config.get("history_stride", 1),
    )



def build_data_loaders(config: dict, tokenizer, selected_variants: list[str]):
    train_datasets = []
    val_datasets = []

    for variant in selected_variants:
        print(f"[train] Loading data for variant: {variant}")
        train_datasets.append(build_dataset(config, tokenizer, variant, "train"))
        val_datasets.append(build_dataset(config, tokenizer, variant, "val"))

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
    from evaluate import evaluate_variant
    from utils.prompt_loader import load_templates

    eval_config = {
        "env_family": config["env_family"],
        "num_episodes": config["eval_num_episodes"],
        "parse_retry_limit": config.get("parse_retry_limit", 3),
        "env_kwargs": config.get("eval_env_kwargs", {"continuing_task": False}),
        "history_num": config.get("history_num", 0),
        "history_stride": config.get("history_stride", 1),
    }

    model.eval()
    FastLanguageModel.for_inference(model)

    for variant in variants:
        print(f"[eval] Epoch {epoch} | variant: {variant}")
        templates = load_templates(config["env_family"])
        result = evaluate_variant(eval_config, variant, model, tokenizer, device, templates[0])
        result["train_loss"] = train_loss
        result["val_loss"] = val_loss
        result["experiment_id"] = config["experiment_id"]

        print(
            f"[eval] {variant}: mean_return={result['mean_return']:.4f}, "
            f"success_rate={result['success_rate']:.2%}, "
            f"mean_steps={result['mean_episode_steps']:.1f}, "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        results_dir = get_eval_results_dir(config, train_selection_tag, variant)
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, f"result_ep{epoch}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval] Saved: {result_path}")

    model.train()
    FastLanguageModel.for_training(model)



def _run_training(config, model, train_loader, val_loader, device,
                  selection_tag: str, tokenizer=None, eval_variants=None):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )

    num_epochs = config["num_epochs"]
    progress_interval = float(config.get("progress_interval_seconds", 5.0))
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        train_total = len(train_loader)
        train_start = time.monotonic()
        train_last_log = train_start - progress_interval
        train_desc = f"Epoch {epoch}/{num_epochs} [train]"
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if _should_log_progress(step, train_total, train_last_log, progress_interval):
                _print_progress(
                    train_desc,
                    step,
                    train_total,
                    train_start,
                    extra=f"loss={loss.item():.4f}",
                    done=step == train_total,
                )
                train_last_log = time.monotonic()

        train_loss = total_loss / max(num_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_total = len(val_loader)
        val_start = time.monotonic()
        val_last_log = val_start - progress_interval
        val_desc = f"Epoch {epoch}/{num_epochs} [val]"
        with torch.no_grad():
            for step, batch in enumerate(val_loader, start=1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
                val_batches += 1
                if _should_log_progress(step, val_total, val_last_log, progress_interval):
                    _print_progress(
                        val_desc,
                        step,
                        val_total,
                        val_start,
                        extra=f"loss={outputs.loss.item():.4f}",
                        done=step == val_total,
                    )
                    val_last_log = time.monotonic()

        val_loss = val_loss / max(val_batches, 1)
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
    _run_training(
        config,
        model,
        train_loader,
        val_loader,
        device,
        selection_tag=train_selection.selection_tag,
        tokenizer=tokenizer,
        eval_variants=eval_selection.selected_variants,
    )

    checkpoint_dir = get_checkpoint_dir(config, train_selection.selection_tag)
    _save_checkpoint(config, model, tokenizer, checkpoint_dir)



def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    experiment_id = ensure_experiment_id(config)
    available_variants = get_available_variants(config["env_family"])
    train_selection = resolve_train_selection(config, available_variants)
    eval_selection = resolve_epoch_eval_selection(config, available_variants, train_selection)

    config["variants"] = train_selection.configured_variants
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
