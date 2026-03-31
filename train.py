"""Training entry point for LLM offline RL (behavior cloning).

Usage:
    python train.py --config config.yaml
"""

import argparse
import os
import json

import yaml
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from tqdm import tqdm

from data.registry import get_dataset
from data.pointmaze.variants import POINTMAZE_VARIANTS
from data.pointmaze.dataset import collate_fn
from model.policy import load_model_and_tokenizer, get_model_slug


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def get_checkpoint_dir(config: dict, variant: str) -> str:
    slug = get_model_slug(config["model_name"])
    root = config.get("checkpoint_root", "checkpoints")
    return os.path.join(
        root,
        config["env_family"],
        slug,
        config["train_mode"],
        variant,
        "final",
    )


def train_single_variant(config: dict, variant: str, model, tokenizer, device: torch.device):
    DatasetCls = get_dataset(config["env_family"])

    print(f"\n[train] Loading data for variant: {variant}")
    train_dataset = DatasetCls(
        variant=variant,
        split="train",
        tokenizer=tokenizer,
        max_length=config["max_length"],
        num_workers=config.get("dataset_workers", 8),
        cache_dir=config.get("dataset_cache_dir"),
    )
    val_dataset = DatasetCls(
        variant=variant,
        split="val",
        tokenizer=tokenizer,
        max_length=config["max_length"],
        num_workers=config.get("dataset_workers", 8),
        cache_dir=config.get("dataset_cache_dir"),
    )
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

    eval_variants = config.get("eval_variants") or [variant]
    _run_training(config, model, train_loader, val_loader, device,
                  tokenizer=tokenizer, eval_variants=eval_variants)

    checkpoint_dir = get_checkpoint_dir(config, variant)
    _save_checkpoint(config, model, tokenizer, checkpoint_dir)


def train_all_variants(config: dict, model, tokenizer, device: torch.device):
    DatasetCls = get_dataset(config["env_family"])
    all_variants = list(POINTMAZE_VARIANTS.keys())

    train_datasets = []
    val_datasets = []
    for variant in all_variants:
        print(f"[train] Loading data for variant: {variant}")
        train_datasets.append(
            DatasetCls(
                variant=variant,
                split="train",
                tokenizer=tokenizer,
                max_length=config["max_length"],
                num_workers=config.get("dataset_workers", 8),
            )
        )
        val_datasets.append(
            DatasetCls(
                variant=variant,
                split="val",
                tokenizer=tokenizer,
                max_length=config["max_length"],
                num_workers=config.get("dataset_workers", 8),
            )
        )

    # Weighted sampling: weight each sample by 1/variant_size so all variants
    # contribute equally regardless of dataset size.
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

    print(f"[train] Total train samples: {len(combined_train)}, val: {len(combined_val)}")
    eval_variants = config.get("eval_variants") or []
    _run_training(config, model, train_loader, val_loader, device,
                  tokenizer=tokenizer, eval_variants=eval_variants)

    checkpoint_dir = get_checkpoint_dir(config, "all")
    _save_checkpoint(config, model, tokenizer, checkpoint_dir)


def get_eval_results_dir(config: dict, variant: str) -> str:
    slug = get_model_slug(config["model_name"])
    env_family = config["env_family"]
    train_mode = config["train_mode"]
    train_variant = config.get("variant", "all")
    train_tag = f"train={env_family}-{train_variant}-{train_mode}"
    eval_tag = f"eval={env_family}-{variant}"
    return os.path.join("results", slug, train_tag, eval_tag)


def _run_epoch_eval(config, model, tokenizer, device, variants, epoch, train_loss, val_loss):
    import gymnasium_robotics  # noqa: F401
    from evaluate import evaluate_variant
    from utils.prompt_loader import load_templates

    eval_config = {
        "env_family": config["env_family"],
        "num_episodes": config["eval_num_episodes"],
        "parse_retry_limit": config.get("parse_retry_limit", 3),
        "env_kwargs": config.get("eval_env_kwargs", {"continuing_task": False}),
    }

    for variant in variants:
        print(f"[eval] Epoch {epoch} | variant: {variant}")
        templates = load_templates(config["env_family"], variant)
        model.eval()
        result = evaluate_variant(eval_config, variant, model, tokenizer, device, templates[0])
        result["train_loss"] = train_loss
        result["val_loss"] = val_loss

        print(
            f"[eval] {variant}: mean_return={result['mean_return']:.4f}, "
            f"success_rate={result['success_rate']:.2%}, "
            f"mean_steps={result['mean_episode_steps']:.1f}, "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        results_dir = get_eval_results_dir(config, variant)
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, f"result_ep{epoch}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval] Saved: {result_path}")


def _run_training(config, model, train_loader, val_loader, device,
                  tokenizer=None, eval_variants=None):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )

    num_epochs = config["num_epochs"]
    for epoch in range(1, num_epochs + 1):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False)
        for batch in pbar:
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
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(num_batches, 1)

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]", leave=False):
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

        val_loss = val_loss / max(val_batches, 1)
        print(f"[epoch {epoch}/{num_epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if config.get("eval_num_episodes", 0) > 0 and tokenizer is not None and eval_variants:
            _run_epoch_eval(config, model, tokenizer, device, eval_variants,
                            epoch=epoch, train_loss=train_loss, val_loss=val_loss)


def _save_checkpoint(config, model, tokenizer, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    config_dst = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_dst, "w") as f:
        yaml.dump(config, f)

    # Unsloth rewrites base_model_name_or_path to "unsloth/<model>" in adapter_config.json.
    # Patch it back to the original model name so offline checkpoint loading works.
    adapter_cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        adapter_cfg["base_model_name_or_path"] = config["model_name"]
        with open(adapter_cfg_path, "w") as f:
            json.dump(adapter_cfg, f, indent=2)

    print(f"[train] Checkpoint saved to: {checkpoint_dir}")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(config)
    model.to(device)

    train_mode = config["train_mode"]
    if train_mode == "single":
        variant = config["variant"]
        train_single_variant(config, variant, model, tokenizer, device)
    elif train_mode == "all":
        train_all_variants(config, model, tokenizer, device)
    else:
        raise ValueError(f"Unknown train_mode: {train_mode!r}. Expected 'single' or 'all'.")


if __name__ == "__main__":
    main()
