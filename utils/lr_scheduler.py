"""Learning-rate schedule helpers for optimizer-step based training."""

from __future__ import annotations

import math

import torch


def normalize_lr_scheduler_type(value) -> str:
    scheduler_type = str(value or "constant").strip().lower()
    aliases = {
        "none": "constant",
        "off": "constant",
        "constant_with_warmup": "constant",
    }
    scheduler_type = aliases.get(scheduler_type, scheduler_type)
    valid = {"constant", "linear", "cosine"}
    if scheduler_type not in valid:
        raise ValueError(
            f"Invalid lr_scheduler_type={value!r}; expected one of {sorted(valid)} "
            "or constant_with_warmup"
        )
    return scheduler_type


def normalize_warmup_ratio_basis(value) -> str:
    """Normalize legacy warmup_ratio_basis values."""
    basis = str(value or "total").strip().lower()
    aliases = {
        "run": "total",
        "training": "total",
        "all": "total",
        "epoch_steps": "epoch",
    }
    basis = aliases.get(basis, basis)
    valid = {"total", "epoch"}
    if basis not in valid:
        raise ValueError(f"warmup_ratio_basis must be one of {sorted(valid)}, got {value!r}")
    return basis


def resolve_warmup_ratio_ref_epoch(config: dict) -> float | None:
    if "warmup_ratio_ref_epoch" not in config or config.get("warmup_ratio_ref_epoch") is None:
        return None
    ref_epoch = float(config["warmup_ratio_ref_epoch"])
    if ref_epoch < 0.0:
        raise ValueError(f"warmup_ratio_ref_epoch must be >= 0, got {ref_epoch}")
    return ref_epoch


def resolve_lr_decay_ref_epochs(config: dict) -> float | None:
    if "lr_decay_ref_epochs" in config and config.get("lr_decay_ref_epochs") is not None:
        ref_epochs = float(config["lr_decay_ref_epochs"])
    elif "lr_decay_epochs" in config and config.get("lr_decay_epochs") is not None:
        ref_epochs = float(config["lr_decay_epochs"])
    else:
        return None
    if ref_epochs < 0.0:
        raise ValueError(f"lr_decay_ref_epochs must be >= 0, got {ref_epochs}")
    return ref_epochs


def resolve_warmup_steps(
    config: dict,
    total_training_steps: int,
    steps_per_epoch: int | None = None,
) -> int:
    if "warmup_steps" in config and config.get("warmup_steps") is not None:
        warmup_steps = int(config["warmup_steps"])
    else:
        warmup_ratio = float(config.get("warmup_ratio", 0.0) or 0.0)
        if warmup_ratio < 0.0 or warmup_ratio > 1.0:
            raise ValueError(f"warmup_ratio must satisfy 0 <= warmup_ratio <= 1, got {warmup_ratio}")
        warmup_ratio_ref_epoch = resolve_warmup_ratio_ref_epoch(config)
        if warmup_ratio_ref_epoch is not None:
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch is required when warmup_ratio_ref_epoch is set")
            reference_steps = int(steps_per_epoch) * warmup_ratio_ref_epoch
        else:
            basis = normalize_warmup_ratio_basis(config.get("warmup_ratio_basis", "total"))
            reference_steps = total_training_steps
            if basis == "epoch":
                if steps_per_epoch is None:
                    raise ValueError("steps_per_epoch is required when warmup_ratio_basis='epoch'")
                reference_steps = int(steps_per_epoch)
        warmup_steps = int(math.ceil(reference_steps * warmup_ratio))
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    return min(warmup_steps, max(int(total_training_steps), 0))


def resolve_min_lr_ratio(config: dict) -> float:
    min_lr_ratio = float(config.get("min_lr_ratio", 0.0) or 0.0)
    if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
        raise ValueError(f"min_lr_ratio must satisfy 0 <= min_lr_ratio <= 1, got {min_lr_ratio}")
    return min_lr_ratio


def resolve_lr_decay_steps(
    config: dict,
    *,
    total_training_steps: int,
    warmup_steps: int,
    steps_per_epoch: int | None = None,
) -> int:
    if "lr_decay_steps" in config and config.get("lr_decay_steps") is not None:
        decay_steps = int(config["lr_decay_steps"])
    else:
        lr_decay_ref_epochs = resolve_lr_decay_ref_epochs(config)
        if lr_decay_ref_epochs is not None:
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch is required when lr_decay_ref_epochs is set")
            decay_steps = int(math.ceil(int(steps_per_epoch) * lr_decay_ref_epochs))
        else:
            decay_steps = int(total_training_steps) - int(warmup_steps)

    if decay_steps < 0:
        raise ValueError(f"lr_decay_steps must be >= 0, got {decay_steps}")
    return max(decay_steps, 1)


def lr_scale_for_step(
    *,
    step_index: int,
    total_training_steps: int,
    warmup_steps: int,
    decay_steps: int | None = None,
    scheduler_type: str,
    min_lr_ratio: float,
) -> float:
    """Return LR scale for the 1-based optimizer step about to run."""
    step_index = max(int(step_index), 1)
    total_training_steps = max(int(total_training_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    if warmup_steps > 0 and step_index <= warmup_steps:
        return float(step_index) / float(warmup_steps)
    if scheduler_type == "constant":
        return 1.0

    if decay_steps is None:
        decay_steps = total_training_steps - warmup_steps
    decay_steps = max(int(decay_steps), 1)
    progress = (step_index - warmup_steps) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    if scheduler_type == "linear":
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
    if scheduler_type == "cosine":
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_scale
    raise ValueError(f"Unsupported lr_scheduler_type={scheduler_type!r}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, base_lr: float, scale: float) -> float:
    lr = float(base_lr) * float(scale)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])
