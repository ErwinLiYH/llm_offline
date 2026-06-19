"""MTP-style action-bin decoder with trainable Action Query Tokens."""

from __future__ import annotations

import os
import types
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from utils.action_bins import bin_to_continuous, get_action_token_mode


MTP_BIN_DECODER_FILENAME = "mtp_bin_decoder.pt"
MTP_BIN_MODE = "mtp_bin"
SIMPLE_MTP_BIN_MODE = "simple_mtp_bin"
MTP_BIN_MODES = {MTP_BIN_MODE, SIMPLE_MTP_BIN_MODE}


def uses_mtp_bin(config: dict) -> bool:
    return get_action_token_mode(config) in MTP_BIN_MODES


def uses_simple_mtp_bin(config: dict) -> bool:
    return get_action_token_mode(config) == SIMPLE_MTP_BIN_MODE


def resolve_mtp_decoder_mode(config: dict) -> str:
    mode = get_action_token_mode(config)
    if mode not in MTP_BIN_MODES:
        raise ValueError(f"Expected an MTP action-bin mode, got action_token_mode={mode!r}")
    return mode


def resolve_mtp_query_count(action_dim: int, config: dict) -> int:
    action_dim = int(action_dim)
    mode = resolve_mtp_decoder_mode(config)
    if mode == SIMPLE_MTP_BIN_MODE:
        if action_dim < 1:
            raise ValueError(f"simple_mtp_bin requires action_dim >= 1, got {action_dim}")
        return action_dim
    return resolve_mtp_k(action_dim, config.get("mtp_k"))


def resolve_mtp_k(action_dim: int, mtp_k: Any | None = None) -> int:
    action_dim = int(action_dim)
    if action_dim < 2:
        raise ValueError(f"mtp_bin requires action_dim >= 2, got {action_dim}")
    resolved = action_dim - 1 if mtp_k is None else int(mtp_k)
    if resolved < 1:
        raise ValueError(f"mtp_k must be >= 1, got {resolved}")
    if resolved != action_dim - 1:
        raise ValueError(
            "mtp_bin v1 requires mtp_k == action_dim - 1: "
            f"mtp_k={resolved}, action_dim={action_dim}"
        )
    return resolved


def resolve_mtp_lcm_weight(config: dict | None = None) -> float:
    config = config or {}
    value = float(config.get("mtp_lcm_weight", 1.0))
    if value < 0:
        raise ValueError(f"mtp_lcm_weight must be >= 0, got {value}")
    return value


def resolve_mtp_quadratic_decoding(config: dict | None = None) -> bool:
    config = config or {}
    value = config.get("mtp_quadratic_decoding", True)
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(
        "mtp_quadratic_decoding must be a boolean-like value, "
        f"got {value!r}"
    )


def _resolve_hidden_size(model) -> int:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is None:
        raise ValueError("Cannot resolve model hidden size for mtp_bin decoder")
    return int(weight.shape[1])


def _module_weight_dtype(module, fallback: torch.dtype) -> torch.dtype:
    weight = getattr(module, "weight", None)
    if torch.is_tensor(weight):
        return weight.dtype
    for parameter in module.parameters(recurse=True):
        return parameter.dtype
    return fallback


def build_mtp_bin_attention_mask(
    attention_mask: torch.Tensor,
    action_query_mask: torch.Tensor,
    action_query_source_positions: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a 4D additive attention mask for NTP tokens plus AQT blocks."""
    if attention_mask.dim() != 2:
        raise ValueError(
            "attention_mask must be 2D [batch, seq_len], "
            f"got shape={tuple(attention_mask.shape)}"
        )
    if action_query_mask.shape != attention_mask.shape:
        raise ValueError(
            "action_query_mask shape must match attention_mask shape, "
            f"got action_query_mask={tuple(action_query_mask.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}"
        )
    if action_query_source_positions.shape != attention_mask.shape:
        raise ValueError(
            "action_query_source_positions shape must match attention_mask shape, "
            f"got source_positions={tuple(action_query_source_positions.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}"
        )
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32

    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape
    visible = attention_mask.to(dtype=torch.bool)
    query = action_query_mask.to(device=device, dtype=torch.bool) & visible
    ntp = visible & ~query
    seq_idx = torch.arange(seq_len, device=device)
    causal = seq_idx.view(1, -1, 1) >= seq_idx.view(1, 1, -1)

    allowed = torch.zeros((batch_size, seq_len, seq_len), device=device, dtype=torch.bool)
    allowed |= ntp.unsqueeze(2) & ntp.unsqueeze(1) & causal

    source = action_query_source_positions.to(device=device, dtype=torch.long)
    key_idx = seq_idx.view(1, 1, seq_len)
    query_source = source.unsqueeze(2)
    allowed |= query.unsqueeze(2) & ntp.unsqueeze(1) & (key_idx <= query_source)

    same_block = (
        source.unsqueeze(2) >= 0
    ) & (source.unsqueeze(2) == source.unsqueeze(1))
    allowed |= query.unsqueeze(2) & query.unsqueeze(1) & same_block

    blocked_value = torch.finfo(dtype).min
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype)
    return mask.masked_fill(~allowed.unsqueeze(1), blocked_value)


class MTPSamplerHead(nn.Module):
    """Two-layer sampler head conditioned on AQT hidden state and prior token."""

    def __init__(self, hidden_size: int):
        super().__init__()
        hidden_size = int(hidden_size)
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        self.hidden_size = hidden_size
        self.input_norm = nn.LayerNorm(hidden_size * 2)
        self.input_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.SiLU()
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, action_query_hidden: torch.Tensor, previous_token_embedding: torch.Tensor):
        if action_query_hidden.shape != previous_token_embedding.shape:
            raise ValueError(
                "AQT hidden states and previous-token embeddings must have the same shape: "
                f"hidden={tuple(action_query_hidden.shape)}, "
                f"prev={tuple(previous_token_embedding.shape)}"
            )
        x = torch.cat([action_query_hidden, previous_token_embedding], dim=-1)
        x = self.activation(self.input_proj(self.input_norm(x)))
        return self.output_norm(self.output_proj(x))


@dataclass(frozen=True)
class MTPBinOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    sampler_logits: torch.Tensor
    sampler_query_mask: torch.Tensor


class MTPBinDecoder(nn.Module):
    """Append trainable AQT embeddings and predict action-bin tokens."""

    def __init__(self, mtp_k: int, hidden_size: int, mode: str = MTP_BIN_MODE):
        super().__init__()
        mtp_k = int(mtp_k)
        hidden_size = int(hidden_size)
        if mode not in MTP_BIN_MODES:
            raise ValueError(f"Unknown MTP decoder mode: {mode!r}")
        if mtp_k < 1:
            raise ValueError(f"mtp_k must be >= 1, got {mtp_k}")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        self.mtp_k = mtp_k
        self.hidden_size = hidden_size
        self.mode = mode
        self.action_query_embeddings = nn.Embedding(mtp_k, hidden_size)
        self.simple_prev_embedding = nn.Parameter(torch.empty(hidden_size))
        self.sampler_head = MTPSamplerHead(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.action_query_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.simple_prev_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        model,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_query_mask: torch.Tensor,
        action_query_offsets: torch.Tensor,
        action_query_source_positions: torch.Tensor,
        action_query_prev_token_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> MTPBinOutput:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D [batch, seq_len], got {tuple(input_ids.shape)}")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        if action_query_mask.shape != input_ids.shape:
            raise ValueError("action_query_mask must match input_ids shape")
        if action_query_offsets.shape != input_ids.shape:
            raise ValueError("action_query_offsets must match input_ids shape")
        if action_query_prev_token_ids.shape != input_ids.shape:
            raise ValueError("action_query_prev_token_ids must match input_ids shape")

        embeddings = model.get_input_embeddings()
        inputs_embeds = embeddings(input_ids)
        query_mask = action_query_mask.to(device=input_ids.device, dtype=torch.bool) & attention_mask.to(
            device=input_ids.device, dtype=torch.bool
        )
        if query_mask.any():
            offsets = action_query_offsets.to(device=input_ids.device, dtype=torch.long)[query_mask]
            if offsets.min().item() < 0 or offsets.max().item() >= self.mtp_k:
                raise ValueError(
                    "AQT offset out of range for mtp_bin decoder: "
                    f"min={int(offsets.min().item())}, max={int(offsets.max().item())}, "
                    f"mtp_k={self.mtp_k}"
                )
            query_embeds = self.action_query_embeddings(offsets).to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[query_mask] = query_embeds

        mask_dtype = inputs_embeds.dtype if torch.is_floating_point(inputs_embeds) else torch.float32
        additive_mask = build_mtp_bin_attention_mask(
            attention_mask,
            action_query_mask,
            action_query_source_positions,
            dtype=mask_dtype,
        )
        forward_fn = getattr(model, "_llm_offline_mtp_original_forward", model.forward)
        outputs = forward_fn(
            inputs_embeds=inputs_embeds,
            attention_mask=additive_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]

        sampler_logits = hidden.new_zeros((0, outputs.logits.shape[-1]))
        if query_mask.any():
            query_hidden = hidden[query_mask]
            prev_ids = action_query_prev_token_ids.to(device=input_ids.device, dtype=torch.long)[
                query_mask
            ]
            sampler_dtype = _module_weight_dtype(self.sampler_head, query_hidden.dtype)
            query_hidden_for_sampler = query_hidden.to(dtype=sampler_dtype)
            if self.mode == SIMPLE_MTP_BIN_MODE:
                prev_embeds = self.simple_prev_embedding.to(
                    device=query_hidden.device,
                    dtype=sampler_dtype,
                ).expand_as(query_hidden_for_sampler)
            else:
                prev_embeds = embeddings(prev_ids).to(dtype=sampler_dtype)
            sampler_hidden = self.sampler_head(query_hidden_for_sampler, prev_embeds)
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is None:
                raise ValueError("mtp_bin requires a model output embedding/lm_head")
            output_dtype = _module_weight_dtype(output_embeddings, query_hidden.dtype)
            sampler_hidden = sampler_hidden.to(dtype=output_dtype)
            sampler_logits = output_embeddings(sampler_hidden)

        return MTPBinOutput(
            logits=outputs.logits,
            hidden_states=hidden,
            sampler_logits=sampler_logits,
            sampler_query_mask=query_mask,
        )


def _mtp_bin_forward_patched(self, *args, mtp_bin: bool = False, **kwargs):
    if not mtp_bin:
        return self._llm_offline_mtp_original_forward(*args, **kwargs)

    if args:
        if "input_ids" in kwargs:
            raise ValueError("input_ids was provided both positionally and by keyword")
        kwargs["input_ids"] = args[0]
        args = args[1:]
    if args:
        raise ValueError("mtp_bin forward accepts only input_ids as a positional argument")

    required = (
        "input_ids",
        "attention_mask",
        "action_query_mask",
        "action_query_offsets",
        "action_query_source_positions",
        "action_query_prev_token_ids",
    )
    missing = [key for key in required if key not in kwargs]
    if missing:
        raise ValueError(f"Missing mtp_bin forward kwargs: {missing}")
    input_ids = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")
    action_query_mask = kwargs.pop("action_query_mask")
    action_query_offsets = kwargs.pop("action_query_offsets")
    action_query_source_positions = kwargs.pop("action_query_source_positions")
    action_query_prev_token_ids = kwargs.pop("action_query_prev_token_ids")
    position_ids = kwargs.pop("position_ids", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise ValueError(f"Unexpected mtp_bin forward kwargs: {unexpected}")
    return self.mtp_bin_decoder(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        action_query_mask=action_query_mask,
        action_query_offsets=action_query_offsets,
        action_query_source_positions=action_query_source_positions,
        action_query_prev_token_ids=action_query_prev_token_ids,
        position_ids=position_ids,
    )


def _is_mtp_bin_forward_patched(model) -> bool:
    return getattr(getattr(model, "forward", None), "__func__", None) is _mtp_bin_forward_patched


def attach_mtp_bin_decoder(
    model,
    action_dim: int,
    *,
    mtp_k: int | None = None,
    hidden_size: int | None = None,
    mode: str = MTP_BIN_MODE,
) -> MTPBinDecoder:
    action_dim = int(action_dim)
    if mode not in MTP_BIN_MODES:
        raise ValueError(f"Unknown MTP decoder mode: {mode!r}")
    mtp_k = action_dim if mode == SIMPLE_MTP_BIN_MODE else resolve_mtp_k(action_dim, mtp_k)
    hidden_size = _resolve_hidden_size(model) if hidden_size is None else int(hidden_size)

    decoder = getattr(model, "mtp_bin_decoder", None)
    if decoder is None:
        decoder = MTPBinDecoder(mtp_k=mtp_k, hidden_size=hidden_size, mode=mode)
        model.add_module("mtp_bin_decoder", decoder)
    else:
        decoder_mode = getattr(decoder, "mode", MTP_BIN_MODE)
        if decoder_mode != mode:
            raise ValueError(
                "Existing mtp_bin_decoder mode does not match config: "
                f"decoder={decoder_mode}, config={mode}"
            )
        if int(decoder.mtp_k) != mtp_k:
            raise ValueError(
                "Existing mtp_bin_decoder mtp_k does not match config: "
                f"decoder={decoder.mtp_k}, config={mtp_k}"
            )
        if int(decoder.hidden_size) != hidden_size:
            raise ValueError(
                "Existing mtp_bin_decoder hidden_size does not match model: "
                f"decoder={decoder.hidden_size}, model={hidden_size}"
            )

    if not _is_mtp_bin_forward_patched(model):
        model._llm_offline_mtp_original_forward = model.forward
        model.forward = types.MethodType(_mtp_bin_forward_patched, model)
    return decoder


def unpatch_mtp_bin_forward(model):
    if _is_mtp_bin_forward_patched(model) and hasattr(model, "_llm_offline_mtp_original_forward"):
        model.forward = model._llm_offline_mtp_original_forward


def ensure_mtp_bin_decoder(model, config: dict) -> MTPBinDecoder | None:
    if not uses_mtp_bin(config):
        return None
    if "action_dim" not in config:
        raise ValueError("MTP action-bin modes require config['action_dim']")
    action_dim = int(config["action_dim"])
    mode = resolve_mtp_decoder_mode(config)
    return attach_mtp_bin_decoder(
        model,
        action_dim,
        mtp_k=resolve_mtp_query_count(action_dim, config),
        mode=mode,
    )


def save_mtp_bin_decoder(model, checkpoint_dir: str):
    decoder = getattr(model, "mtp_bin_decoder", None)
    if decoder is None:
        raise ValueError("Cannot save mtp_bin checkpoint without mtp_bin_decoder")
    payload = {
        "mode": getattr(decoder, "mode", MTP_BIN_MODE),
        "mtp_k": int(decoder.mtp_k),
        "hidden_size": int(decoder.hidden_size),
        "state_dict": decoder.state_dict(),
    }
    torch.save(payload, os.path.join(checkpoint_dir, MTP_BIN_DECODER_FILENAME))


def load_mtp_bin_decoder(
    model,
    checkpoint_dir: str,
    *,
    expected_action_dim: int | None = None,
    expected_mtp_k: int | None = None,
    expected_mode: str | None = None,
):
    path = os.path.join(checkpoint_dir, MTP_BIN_DECODER_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing mtp_bin decoder checkpoint: {path}. "
            "mtp_bin checkpoints must include this file."
        )
    payload: dict[str, Any] = torch.load(path, map_location="cpu")
    mode = str(payload.get("mode", MTP_BIN_MODE))
    if mode not in MTP_BIN_MODES:
        raise ValueError(f"Unknown mtp_bin_decoder.pt mode: {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(
            "mtp_bin_decoder.pt mode does not match config.yaml: "
            f"decoder={mode}, config={expected_mode}"
        )
    mtp_k = int(payload["mtp_k"])
    hidden_size = int(payload["hidden_size"])
    model_hidden_size = _resolve_hidden_size(model)
    if hidden_size != model_hidden_size:
        raise ValueError(
            "mtp_bin_decoder.pt hidden_size does not match model config: "
            f"decoder={hidden_size}, model={model_hidden_size}"
        )
    if expected_action_dim is not None:
        if mode == SIMPLE_MTP_BIN_MODE:
            expected_resolved = int(expected_action_dim)
        else:
            expected_resolved = resolve_mtp_k(int(expected_action_dim), expected_mtp_k)
        if mtp_k != expected_resolved:
            raise ValueError(
                "mtp_bin_decoder.pt mtp_k does not match config.yaml: "
                f"decoder={mtp_k}, config={expected_resolved}"
            )
        action_dim = int(expected_action_dim)
    else:
        action_dim = mtp_k if mode == SIMPLE_MTP_BIN_MODE else mtp_k + 1
    decoder = attach_mtp_bin_decoder(
        model,
        action_dim=action_dim,
        mtp_k=mtp_k,
        hidden_size=hidden_size,
        mode=mode,
    )
    state_dict = payload["state_dict"]
    if mode == MTP_BIN_MODE and "simple_prev_embedding" not in state_dict:
        state_dict = dict(state_dict)
        state_dict["simple_prev_embedding"] = decoder.simple_prev_embedding.detach().cpu()
    decoder.load_state_dict(state_dict)
    return decoder


def mtp_bin_action_loss(
    output: MTPBinOutput,
    action_bin_labels: torch.Tensor,
    action_query_mask: torch.Tensor,
    action_query_anchor_positions: torch.Tensor,
    bin_token_ids: list[int],
    *,
    lcm_weight: float = 1.0,
    base_loss_on_queries: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if output.logits.dim() != 3:
        raise ValueError(f"logits must be 3D [batch, seq_len, vocab], got {tuple(output.logits.shape)}")
    if action_bin_labels.shape != output.logits.shape[:2]:
        raise ValueError("action_bin_labels must match logits batch/sequence dimensions")
    action_mask = action_bin_labels >= 0
    total_loss = output.logits.float().sum() * 0.0
    metrics = {
        "base_loss": 0.0,
        "sampler_loss": 0.0,
        "lcm_loss": 0.0,
        "action_tokens": int(action_mask.sum().item()),
        "action_query_tokens": int((action_mask & action_query_mask.bool()).sum().item()),
    }
    if not action_mask.any():
        return total_loss, metrics

    token_ids = torch.tensor(bin_token_ids, device=output.logits.device, dtype=torch.long)
    query_mask = action_query_mask.to(device=action_bin_labels.device, dtype=torch.bool)
    base_mask = action_mask if base_loss_on_queries else (action_mask & ~query_mask)
    target_bins = action_bin_labels[action_mask]
    if target_bins.min().item() < 0 or target_bins.max().item() >= token_ids.numel():
        raise ValueError(
            "mtp_bin target bin index out of range: "
            f"min={int(target_bins.min().item())}, max={int(target_bins.max().item())}, "
            f"num_bins={token_ids.numel()}"
        )
    if base_mask.any():
        base_target_token_ids = token_ids[action_bin_labels[base_mask]]
        base_loss = F.cross_entropy(output.logits[base_mask].float(), base_target_token_ids)
        total_loss = total_loss + base_loss
        metrics["base_loss"] = float(base_loss.detach().item())

    all_query_labels = action_bin_labels[query_mask]
    valid_query = all_query_labels >= 0
    if valid_query.any():
        query_target_token_ids = token_ids[all_query_labels[valid_query]]
        sampler_logits = output.sampler_logits[valid_query].float()
        sampler_loss = F.cross_entropy(sampler_logits, query_target_token_ids)
        total_loss = total_loss + sampler_loss
        metrics["sampler_loss"] = float(sampler_loss.detach().item())

        batch_idx, seq_idx = torch.nonzero(query_mask & (action_bin_labels >= 0), as_tuple=True)
        anchor_pos = action_query_anchor_positions.to(
            device=action_bin_labels.device,
            dtype=torch.long,
        )[batch_idx, seq_idx]
        if anchor_pos.min().item() < 0 or anchor_pos.max().item() >= output.hidden_states.shape[1]:
            raise ValueError("mtp_bin LCM anchor position out of range")
        query_hidden = output.hidden_states[batch_idx, seq_idx].float()
        anchor_hidden = output.hidden_states[batch_idx, anchor_pos].detach().float()
        lcm_loss = F.mse_loss(query_hidden, anchor_hidden, reduction="mean")
        weighted_lcm = float(lcm_weight) * lcm_loss
        total_loss = total_loss + weighted_lcm
        metrics["lcm_loss"] = float(lcm_loss.detach().item())

    return total_loss, metrics


def mtp_bin_equivalent_l1(
    output: MTPBinOutput,
    action_bin_labels: torch.Tensor,
    action_query_mask: torch.Tensor,
    bin_token_ids: list[int],
    num_bins: int,
    low: float = -1.0,
    high: float = 1.0,
) -> float:
    action_mask = action_bin_labels >= 0
    if not action_mask.any():
        return float("nan")
    token_ids = torch.tensor(bin_token_ids, device=output.logits.device, dtype=torch.long)
    query_mask = action_query_mask.to(device=action_bin_labels.device, dtype=torch.bool)
    ntp_mask = action_mask & ~query_mask

    pred_bins = []
    target_bins = []
    if ntp_mask.any():
        ntp_logits = output.logits[ntp_mask].float().index_select(dim=-1, index=token_ids)
        pred_bins.append(ntp_logits.argmax(dim=-1))
        target_bins.append(action_bin_labels[ntp_mask])
    if (query_mask & action_mask).any():
        all_query_labels = action_bin_labels[query_mask]
        valid_query = all_query_labels >= 0
        query_logits = output.sampler_logits[valid_query].float().index_select(
            dim=-1,
            index=token_ids,
        )
        pred_bins.append(query_logits.argmax(dim=-1))
        target_bins.append(all_query_labels[valid_query])
    if not pred_bins:
        return float("nan")

    predicted = torch.cat(pred_bins).detach().cpu().tolist()
    target = torch.cat(target_bins).detach().cpu().tolist()
    errors = [
        abs(
            bin_to_continuous(int(pred), num_bins, low, high)
            - bin_to_continuous(int(label), num_bins, low, high)
        )
        for pred, label in zip(predicted, target)
    ]
    return float(sum(errors) / len(errors)) if errors else float("nan")


def mtp_bin_equivalent_l1_by_path(
    output: MTPBinOutput,
    action_bin_labels: torch.Tensor,
    action_query_mask: torch.Tensor,
    bin_token_ids: list[int],
    num_bins: int,
    low: float = -1.0,
    high: float = 1.0,
) -> dict[str, float]:
    action_mask = action_bin_labels >= 0
    if not action_mask.any():
        return {"ntp_bin_l1": float("nan"), "mtp_bin_l1": float("nan")}
    token_ids = torch.tensor(bin_token_ids, device=output.logits.device, dtype=torch.long)
    query_mask = action_query_mask.to(device=action_bin_labels.device, dtype=torch.bool)

    def _l1_for(pred: torch.Tensor, target: torch.Tensor) -> float:
        predicted = pred.detach().cpu().tolist()
        labels = target.detach().cpu().tolist()
        errors = [
            abs(
                bin_to_continuous(int(pred_bin), num_bins, low, high)
                - bin_to_continuous(int(label_bin), num_bins, low, high)
            )
            for pred_bin, label_bin in zip(predicted, labels)
        ]
        return float(sum(errors) / len(errors)) if errors else float("nan")

    metrics = {"ntp_bin_l1": float("nan"), "mtp_bin_l1": float("nan")}
    ntp_mask = action_mask & ~query_mask
    if ntp_mask.any():
        ntp_logits = output.logits[ntp_mask].float().index_select(dim=-1, index=token_ids)
        metrics["ntp_bin_l1"] = _l1_for(ntp_logits.argmax(dim=-1), action_bin_labels[ntp_mask])
    mtp_mask = action_mask & query_mask
    if mtp_mask.any():
        all_query_labels = action_bin_labels[query_mask]
        valid_query = all_query_labels >= 0
        query_logits = output.sampler_logits[valid_query].float().index_select(
            dim=-1,
            index=token_ids,
        )
        metrics["mtp_bin_l1"] = _l1_for(query_logits.argmax(dim=-1), all_query_labels[valid_query])
    return metrics
