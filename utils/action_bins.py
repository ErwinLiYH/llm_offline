"""Utilities for discrete action-bin tokens and losses."""

from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn.functional as F


VALID_ACTION_TOKEN_MODES = {"text", "bin", "gaussian_bin"}


def get_action_token_mode(config: dict) -> str:
    mode = str(config.get("action_token_mode", "text"))
    if mode not in VALID_ACTION_TOKEN_MODES:
        raise ValueError(
            f"Invalid action_token_mode={mode!r}; expected one of {sorted(VALID_ACTION_TOKEN_MODES)}"
        )
    return mode


def uses_action_bins(config: dict) -> bool:
    return get_action_token_mode(config) in {"bin", "gaussian_bin"}


def get_action_num_bins(config: dict) -> int:
    num_bins = int(config.get("action_num_bins", 10))
    if num_bins < 2:
        raise ValueError(f"action_num_bins must be >= 2, got {num_bins}")
    return num_bins


def get_action_bin_range(config: dict) -> tuple[float, float]:
    low = float(config.get("action_bin_min", -1.0))
    high = float(config.get("action_bin_max", 1.0))
    if not low < high:
        raise ValueError(f"Expected action_bin_min < action_bin_max, got {low} >= {high}")
    return low, high


def get_action_token_width(num_bins: int) -> int:
    return max(2, len(str(num_bins - 1)))


def get_action_bin_tokens(num_bins: int) -> list[str]:
    width = get_action_token_width(num_bins)
    return [f"<act_{idx:0{width}d}>" for idx in range(num_bins)]


def register_action_tokens(tokenizer, config: dict) -> int:
    """Register action-bin tokens on a tokenizer when a bin mode is enabled."""
    if not uses_action_bins(config):
        return 0
    tokens = get_action_bin_tokens(get_action_num_bins(config))
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    merged = existing + [token for token in tokens if token not in existing]
    return tokenizer.add_special_tokens({"additional_special_tokens": merged})


def get_action_bin_token_ids(tokenizer, config: dict) -> list[int]:
    tokens = get_action_bin_tokens(get_action_num_bins(config))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    missing = [token for token, token_id in zip(tokens, token_ids) if token_id is None or token_id < 0]
    if missing:
        raise ValueError(f"Tokenizer is missing action-bin tokens: {missing}")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None:
        missing = [token for token, token_id in zip(tokens, token_ids) if token_id == unk_id]
        if missing:
            raise ValueError(f"Tokenizer maps action-bin tokens to unk_token_id: {missing}")
    return [int(token_id) for token_id in token_ids]


def continuous_to_bin(value: float, num_bins: int, low: float = -1.0, high: float = 1.0) -> int:
    value = float(np.clip(value, low, high))
    scaled = (value - low) / (high - low)
    return int(np.clip(np.round(scaled * (num_bins - 1)), 0, num_bins - 1))


def bin_to_continuous(index: int, num_bins: int, low: float = -1.0, high: float = 1.0) -> float:
    if not 0 <= int(index) < num_bins:
        raise ValueError(f"Action bin index out of range: {index}")
    return low + (high - low) * float(index) / float(num_bins - 1)


def format_action_bins(
    action: np.ndarray,
    num_bins: int,
    low: float = -1.0,
    high: float = 1.0,
) -> str:
    tokens = get_action_bin_tokens(num_bins)
    return "".join(tokens[continuous_to_bin(float(value), num_bins, low, high)] for value in action)


def action_to_bin_indices(
    action: np.ndarray,
    num_bins: int,
    low: float = -1.0,
    high: float = 1.0,
) -> list[int]:
    return [continuous_to_bin(float(value), num_bins, low, high) for value in action]


def parse_action_bins(
    text: str,
    action_dim: int,
    num_bins: int,
    low: float = -1.0,
    high: float = 1.0,
) -> tuple[np.ndarray, bool]:
    width = get_action_token_width(num_bins)
    pattern = re.compile(rf"<act_(\d{{{width},}})>")
    matches = pattern.findall(text)
    if len(matches) < action_dim:
        return np.zeros(action_dim, dtype=np.float32), False

    try:
        indices = [int(raw) for raw in matches[:action_dim]]
        if any(index < 0 or index >= num_bins for index in indices):
            return np.zeros(action_dim, dtype=np.float32), False
        action = np.array(
            [bin_to_continuous(index, num_bins, low, high) for index in indices],
            dtype=np.float32,
        )
        return action, True
    except ValueError:
        return np.zeros(action_dim, dtype=np.float32), False


def gaussian_bin_targets(bin_indices: torch.Tensor, num_bins: int, sigma: float) -> torch.Tensor:
    if not sigma > 0:
        raise ValueError(f"action_soft_label_sigma must be > 0, got {sigma}")
    centers = torch.arange(num_bins, device=bin_indices.device, dtype=torch.float32)
    dist2 = (centers.unsqueeze(0) - bin_indices.float().unsqueeze(1)).pow(2)
    targets = torch.exp(-dist2 / (2.0 * sigma * sigma))
    return targets / targets.sum(dim=-1, keepdim=True)


def gaussian_action_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_bin_labels: torch.Tensor,
    bin_token_ids: list[int],
    num_bins: int,
    sigma: float,
    action_loss_weight: float = 1.0,
    stop_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Causal loss with Gaussian soft CE for action-bin tokens and hard CE elsewhere."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_action_bins = action_bin_labels[:, 1:].contiguous()

    action_mask = shift_action_bins >= 0
    labelled_mask = shift_labels != -100
    stop_mask = labelled_mask & ~action_mask

    total_loss = logits.sum() * 0.0
    metrics = {
        "action_loss": math.nan,
        "stop_loss": math.nan,
        "action_tokens": int(action_mask.sum().item()),
        "stop_tokens": int(stop_mask.sum().item()),
    }

    if action_mask.any():
        token_ids = torch.tensor(bin_token_ids, device=logits.device, dtype=torch.long)
        action_logits = shift_logits[action_mask].index_select(dim=-1, index=token_ids)
        target_bins = shift_action_bins[action_mask]
        targets = gaussian_bin_targets(target_bins, num_bins, sigma)
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_loss = -(targets * log_probs).sum(dim=-1).mean()
        total_loss = total_loss + float(action_loss_weight) * action_loss
        metrics["action_loss"] = float(action_loss.detach().item())

    if stop_mask.any():
        stop_loss = F.cross_entropy(shift_logits[stop_mask], shift_labels[stop_mask])
        total_loss = total_loss + float(stop_loss_weight) * stop_loss
        metrics["stop_loss"] = float(stop_loss.detach().item())

    return total_loss, metrics
