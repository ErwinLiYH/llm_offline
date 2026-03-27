"""Training entry point for LLM offline RL (behavior cloning).

Usage:
    python train.py --config config.yaml
"""

import argparse
import os
import shutil

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
    )
    val_dataset = DatasetCls(
        variant=variant,
        split="val",
        tokenizer=tokenizer,
        max_length=config["max_length"],
        num_workers=config.get("dataset_workers", 8),
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

    _run_training(config, model, train_loader, val_loader, device)

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
    _run_training(config, model, train_loader, val_loader, device)

    checkpoint_dir = get_checkpoint_dir(config, "all")
    _save_checkpoint(config, model, tokenizer, checkpoint_dir)


def _run_training(config, model, train_loader, val_loader, device):
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


def _save_checkpoint(config, model, tokenizer, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    config_dst = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_dst, "w") as f:
        yaml.dump(config, f)
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
