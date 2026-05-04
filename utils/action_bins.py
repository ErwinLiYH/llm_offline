"""Utilities for discrete action-bin tokens and losses."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


VALID_ACTION_TOKEN_MODES = {"text", "bin", "gaussian_bin"}


def get_tokenizer_backend(tokenizer_or_processor):
    """Return the actual tokenizer object, unwrapping HF processors when needed."""
    if hasattr(tokenizer_or_processor, "add_special_tokens"):
        return tokenizer_or_processor
    inner = getattr(tokenizer_or_processor, "tokenizer", None)
    if inner is not None and hasattr(inner, "add_special_tokens"):
        return inner
    raise AttributeError(
        f"{type(tokenizer_or_processor).__name__} does not expose add_special_tokens "
        "or a tokenizer with add_special_tokens"
    )


def get_action_token_mode(config: dict) -> str:
    mode = str(config.get("action_token_mode", "text"))
    if mode not in VALID_ACTION_TOKEN_MODES:
        raise ValueError(
            f"Invalid action_token_mode={mode!r}; expected one of {sorted(VALID_ACTION_TOKEN_MODES)}"
        )
    return mode


def uses_action_bins(config: dict) -> bool:
    return get_action_token_mode(config) in {"bin", "gaussian_bin"}


def action_bins_use_new_tokens(config: dict) -> bool:
    """Return whether bin modes should add readable action tokens to the tokenizer."""
    value = config.get("new_token", False)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid new_token={value!r}; expected a boolean")
    return bool(value)


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


def _decode_token_ids(tokenizer, token_ids: list[int] | tuple[int, ...]) -> str:
    ids = [int(token_id) for token_id in token_ids]
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids, skip_special_tokens=False)


def _encode_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text=text, add_special_tokens=False)
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        input_ids = encoded["input_ids"]
    return [int(token_id) for token_id in input_ids]


def _sequence_roundtrips(tokenizer, token_ids: list[int] | tuple[int, ...]) -> bool:
    text = _decode_token_ids(tokenizer, token_ids)
    return _encode_text(tokenizer, text) == [int(token_id) for token_id in token_ids]


def _select_existing_action_token_ids(tokenizer, num_bins: int) -> tuple[list[int], list[str]]:
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    special_ids = {
        int(token_id)
        for token_id in (getattr(tokenizer, "all_special_ids", []) or [])
        if token_id is not None
    }
    selected_ids: list[int] = []
    selected_texts: list[str] = []

    for token_id in range(vocab_size - 1, -1, -1):
        if token_id in special_ids:
            continue
        text = _decode_token_ids(tokenizer, [token_id])
        if not text or text.isspace() or "\ufffd" in text:
            continue
        if _encode_text(tokenizer, text) != [token_id]:
            continue

        # PointMaze actions are two-token sequences. Pair validation prevents
        # context-sensitive BPE merges from changing the target IDs.
        pair_stable = True
        for existing_id in selected_ids + [token_id]:
            if not _sequence_roundtrips(tokenizer, [token_id, existing_id]):
                pair_stable = False
                break
            if not _sequence_roundtrips(tokenizer, [existing_id, token_id]):
                pair_stable = False
                break
        if not pair_stable:
            continue

        selected_ids.append(token_id)
        selected_texts.append(text)
        if len(selected_ids) == num_bins:
            return selected_ids, selected_texts

    raise ValueError(
        "Could not find enough stable existing tokenizer tokens for action bins: "
        f"needed {num_bins}, found {len(selected_ids)}."
    )


def _mapping_hash(*, new_token: bool, token_ids: list[int], display_tokens: list[str]) -> str:
    payload = {
        "new_token": bool(new_token),
        "token_ids": [int(token_id) for token_id in token_ids],
        "display_tokens": list(display_tokens),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class ActionBinCodec:
    """Mapping between display action bins and the tokenizer IDs used by the model."""

    num_bins: int
    new_token: bool
    model_token_ids: tuple[int, ...]
    display_tokens: tuple[str, ...]
    mapping_hash: str

    def token_id_to_bin(self) -> dict[int, int]:
        return {int(token_id): bin_idx for bin_idx, token_id in enumerate(self.model_token_ids)}

    def token_ids_for_bins(self, bin_indices: list[int] | tuple[int, ...]) -> list[int]:
        ids = []
        for bin_idx in bin_indices:
            bin_idx = int(bin_idx)
            if bin_idx < 0 or bin_idx >= self.num_bins:
                raise ValueError(f"Action bin index out of range: {bin_idx}")
            ids.append(int(self.model_token_ids[bin_idx]))
        return ids

    def display_text_for_bins(self, bin_indices: list[int] | tuple[int, ...]) -> str:
        pieces = []
        for bin_idx in bin_indices:
            bin_idx = int(bin_idx)
            if bin_idx < 0 or bin_idx >= self.num_bins:
                raise ValueError(f"Action bin index out of range: {bin_idx}")
            pieces.append(self.display_tokens[bin_idx])
        return "".join(pieces)

    def model_text_for_bins(self, tokenizer, bin_indices: list[int] | tuple[int, ...]) -> str:
        if self.new_token:
            return self.display_text_for_bins(bin_indices)
        tok = get_tokenizer_backend(tokenizer)
        token_ids = self.token_ids_for_bins(bin_indices)
        text = _decode_token_ids(tok, token_ids)
        encoded_ids = _encode_text(tok, text)
        if encoded_ids != token_ids:
            raise ValueError(
                "Action-bin model text did not roundtrip to the selected token IDs: "
                f"expected={token_ids}, encoded={encoded_ids}."
            )
        return text

    def bin_indices_from_token_ids(
        self,
        token_ids: list[int] | tuple[int, ...],
        action_dim: int,
    ) -> list[int]:
        token_id_to_bin = self.token_id_to_bin()
        indices = []
        for token_id in token_ids:
            bin_idx = token_id_to_bin.get(int(token_id))
            if bin_idx is not None:
                indices.append(bin_idx)
                if len(indices) == action_dim:
                    break
        return indices

    def display_text_for_token_ids(self, tokenizer, token_ids: list[int] | tuple[int, ...]) -> str:
        tok = get_tokenizer_backend(tokenizer)
        token_id_to_bin = self.token_id_to_bin()
        special_ids = {
            int(token_id)
            for token_id in (getattr(tok, "all_special_ids", []) or [])
            if token_id is not None
        }
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            bin_idx = token_id_to_bin.get(token_id)
            if bin_idx is not None:
                pieces.append(self.display_tokens[bin_idx])
            elif token_id not in special_ids:
                pieces.append(_decode_token_ids(tok, [token_id]))
        return "".join(pieces)


def register_action_tokens(tokenizer, config: dict) -> int:
    """Register action-bin tokens on a tokenizer when a bin mode is enabled."""
    if not uses_action_bins(config) or not action_bins_use_new_tokens(config):
        return 0
    tok = get_tokenizer_backend(tokenizer)
    tokens = get_action_bin_tokens(get_action_num_bins(config))
    existing = list(getattr(tok, "additional_special_tokens", []) or [])
    merged = existing + [token for token in tokens if token not in existing]
    return tok.add_special_tokens({"additional_special_tokens": merged})


def get_action_bin_codec(tokenizer, config: dict, *, ensure_registered: bool = False) -> ActionBinCodec:
    tok = get_tokenizer_backend(tokenizer)
    num_bins = get_action_num_bins(config)
    display_tokens = get_action_bin_tokens(num_bins)
    new_token = action_bins_use_new_tokens(config)

    if new_token:
        if ensure_registered:
            register_action_tokens(tok, config)
        token_ids = tok.convert_tokens_to_ids(display_tokens)
        missing = [
            token
            for token, token_id in zip(display_tokens, token_ids)
            if token_id is None or int(token_id) < 0
        ]
        if missing:
            raise ValueError(f"Tokenizer is missing action-bin tokens: {missing}")
        unk_id = getattr(tok, "unk_token_id", None)
        if unk_id is not None:
            missing = [
                token
                for token, token_id in zip(display_tokens, token_ids)
                if int(token_id) == int(unk_id)
            ]
            if missing:
                raise ValueError(f"Tokenizer maps action-bin tokens to unk_token_id: {missing}")
        token_ids = [int(token_id) for token_id in token_ids]
    else:
        token_ids, _ = _select_existing_action_token_ids(tok, num_bins)

    return ActionBinCodec(
        num_bins=num_bins,
        new_token=new_token,
        model_token_ids=tuple(token_ids),
        display_tokens=tuple(display_tokens),
        mapping_hash=_mapping_hash(
            new_token=new_token,
            token_ids=token_ids,
            display_tokens=display_tokens,
        ),
    )


def get_action_bin_token_ids(tokenizer, config: dict) -> list[int]:
    return list(get_action_bin_codec(tokenizer, config).model_token_ids)


def get_action_bin_mapping_hash(tokenizer, config: dict) -> str:
    if not uses_action_bins(config):
        return "text"
    return get_action_bin_codec(tokenizer, config).mapping_hash


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


def parse_action_bin_token_ids(
    token_ids: list[int] | tuple[int, ...],
    tokenizer,
    config: dict,
    action_dim: int,
    low: float = -1.0,
    high: float = 1.0,
) -> tuple[np.ndarray, bool]:
    codec = get_action_bin_codec(tokenizer, config)
    indices = codec.bin_indices_from_token_ids(token_ids, action_dim)
    if len(indices) < action_dim:
        return np.zeros(action_dim, dtype=np.float32), False
    action = np.array(
        [bin_to_continuous(index, codec.num_bins, low, high) for index in indices],
        dtype=np.float32,
    )
    return action, True


def gaussian_bin_targets(bin_indices: torch.Tensor, num_bins: int, sigma: float) -> torch.Tensor:
    if not sigma > 0:
        raise ValueError(f"action_soft_label_sigma must be > 0, got {sigma}")
    centers = torch.arange(num_bins, device=bin_indices.device, dtype=torch.float32)
    dist2 = (centers.unsqueeze(0) - bin_indices.float().unsqueeze(1)).pow(2)
    targets = torch.exp(-dist2 / (2.0 * sigma * sigma))
    return targets / targets.sum(dim=-1, keepdim=True)


def gaussian_window_targets(bin_indices: torch.Tensor, candidate_bins: torch.Tensor, sigma: float) -> torch.Tensor:
    if not sigma > 0:
        raise ValueError(f"action_soft_label_sigma must be > 0, got {sigma}")
    dist2 = (candidate_bins.float() - bin_indices.float().unsqueeze(1)).pow(2)
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
    soft_label_radius: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Causal loss with Gaussian soft CE for action-bin tokens and hard CE elsewhere."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_action_bins = action_bin_labels[:, 1:].contiguous()

    action_mask = shift_action_bins >= 0
    labelled_mask = shift_labels != -100
    stop_mask = labelled_mask & ~action_mask

    total_loss = logits.float().sum() * 0.0
    metrics = {
        "action_loss": math.nan,
        "stop_loss": math.nan,
        "action_tokens": int(action_mask.sum().item()),
        "stop_tokens": int(stop_mask.sum().item()),
    }

    if action_mask.any():
        token_ids = torch.tensor(bin_token_ids, device=logits.device, dtype=torch.long)
        target_bins = shift_action_bins[action_mask]
        selected_logits = shift_logits[action_mask].float()
        selected_log_probs = F.log_softmax(selected_logits, dim=-1)
        if soft_label_radius is None or soft_label_radius >= num_bins:
            action_log_probs = selected_log_probs.index_select(dim=-1, index=token_ids)
            targets = gaussian_bin_targets(target_bins, num_bins, sigma)
        else:
            if soft_label_radius < 0:
                raise ValueError(f"action_soft_label_radius must be >= 0, got {soft_label_radius}")
            offsets = torch.arange(
                -soft_label_radius,
                soft_label_radius + 1,
                device=logits.device,
                dtype=torch.long,
            )
            candidate_bins = target_bins.unsqueeze(1) + offsets.unsqueeze(0)
            valid = (candidate_bins >= 0) & (candidate_bins < num_bins)
            safe_candidate_bins = candidate_bins.clamp(0, num_bins - 1)
            candidate_token_ids = token_ids[safe_candidate_bins]
            action_log_probs = selected_log_probs.gather(dim=1, index=candidate_token_ids)
            action_log_probs = action_log_probs.masked_fill(~valid, 0.0)
            targets = gaussian_window_targets(target_bins, safe_candidate_bins, sigma)
            targets = targets.masked_fill(~valid, 0.0)
            targets = targets / targets.sum(dim=-1, keepdim=True)
        action_loss = -(targets * action_log_probs).sum(dim=-1).mean()
        total_loss = total_loss + float(action_loss_weight) * action_loss
        metrics["action_loss"] = float(action_loss.detach().item())

    if stop_mask.any():
        stop_loss = F.cross_entropy(shift_logits[stop_mask].float(), shift_labels[stop_mask])
        total_loss = total_loss + float(stop_loss_weight) * stop_loss
        metrics["stop_loss"] = float(stop_loss.detach().item())

    return total_loss, metrics
