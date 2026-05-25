"""Continuous action decoder for prompt-conditioned regression policies."""

from __future__ import annotations

import os
import types
from typing import Any

import torch
from torch import nn


CONTINUOUS_ACTION_DECODER_FILENAME = "continuous_action_decoder.pt"


def build_parallel_action_attention_mask(
    attention_mask: torch.Tensor,
    query_len: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return 4D additive mask for prompt tokens plus learned action queries."""
    if attention_mask.dim() != 2:
        raise ValueError(
            "attention_mask must be 2D [batch, prompt_len], "
            f"got shape={tuple(attention_mask.shape)}"
        )
    query_len = int(query_len)
    if query_len < 1:
        raise ValueError(f"action_query_len must be >= 1, got {query_len}")
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32

    batch_size, prompt_len = attention_mask.shape
    total_len = prompt_len + query_len
    device = attention_mask.device
    prompt_keys_visible = attention_mask.to(dtype=torch.bool)
    causal = torch.tril(torch.ones((prompt_len, prompt_len), device=device, dtype=torch.bool))

    allowed = torch.zeros((batch_size, total_len, total_len), device=device, dtype=torch.bool)
    allowed[:, :prompt_len, :prompt_len] = causal.unsqueeze(0) & prompt_keys_visible.unsqueeze(1)
    allowed[:, prompt_len:, :prompt_len] = prompt_keys_visible.unsqueeze(1).expand(
        batch_size,
        query_len,
        prompt_len,
    )
    allowed[:, prompt_len:, prompt_len:] = True

    blocked_value = torch.finfo(dtype).min
    mask = torch.zeros((batch_size, 1, total_len, total_len), device=device, dtype=dtype)
    return mask.masked_fill(~allowed.unsqueeze(1), blocked_value)


def resolve_action_query_len(action_dim: int, action_query_len: Any | None = None) -> int:
    action_dim = int(action_dim)
    if action_dim < 1:
        raise ValueError(f"action_dim must be >= 1, got {action_dim}")
    if action_query_len is None:
        return action_dim
    action_query_len = int(action_query_len)
    if action_query_len < 1:
        raise ValueError(f"action_query_len must be >= 1, got {action_query_len}")
    return action_query_len


def resolve_action_head_num_blocks(action_head_num_blocks: Any | None = None) -> int:
    if action_head_num_blocks is None:
        return 2
    action_head_num_blocks = int(action_head_num_blocks)
    if action_head_num_blocks < 1:
        raise ValueError(
            f"action_head_num_blocks must be >= 1, got {action_head_num_blocks}"
        )
    return action_head_num_blocks


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x) + x


class MLPResNetActionHead(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        action_query_len: int | None = None,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.action_query_len = resolve_action_query_len(self.action_dim, action_query_len)
        self.num_blocks = resolve_action_head_num_blocks(num_blocks)
        input_dim = self.action_query_len * self.hidden_size
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, self.hidden_size)
        self.input_activation = nn.SiLU()
        self.blocks = nn.ModuleList(
            [MLPResNetBlock(self.hidden_size) for _ in range(self.num_blocks)]
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.action_dim)
        self.output_activation = nn.Tanh()

    def forward(self, action_hidden: torch.Tensor) -> torch.Tensor:
        if action_hidden.dim() != 3:
            raise ValueError(
                "action_hidden must be 3D [batch, action_query_len, hidden_size], "
                f"got {tuple(action_hidden.shape)}"
            )
        if tuple(action_hidden.shape[1:]) != (self.action_query_len, self.hidden_size):
            raise ValueError(
                "action_hidden shape does not match action head: "
                f"got {tuple(action_hidden.shape[1:])}, "
                f"expected {(self.action_query_len, self.hidden_size)}"
            )
        x = action_hidden.reshape(action_hidden.shape[0], self.action_query_len * self.hidden_size)
        x = self.input_activation(self.input_proj(self.input_norm(x)))
        for block in self.blocks:
            x = block(x)
        return self.output_activation(self.output_proj(self.output_norm(x)))


class ContinuousActionDecoder(nn.Module):
    """Append learned action queries and regress continuous actions."""

    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        action_query_len: int | None = None,
        action_head_num_blocks: int | None = None,
    ):
        super().__init__()
        action_dim = int(action_dim)
        hidden_size = int(hidden_size)
        action_query_len = resolve_action_query_len(action_dim, action_query_len)
        action_head_num_blocks = resolve_action_head_num_blocks(action_head_num_blocks)
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")

        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.action_query_len = action_query_len
        self.action_head_num_blocks = action_head_num_blocks
        self.action_queries = nn.Parameter(torch.empty(action_query_len, hidden_size))
        self.action_head = MLPResNetActionHead(
            action_dim=action_dim,
            hidden_size=hidden_size,
            action_query_len=action_query_len,
            num_blocks=action_head_num_blocks,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)

    def forward(
        self,
        model,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("ContinuousActionDecoder.forward requires input_ids")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D [batch, seq_len], got {tuple(input_ids.shape)}")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask shape must match input_ids shape, "
                f"got attention_mask={tuple(attention_mask.shape)}, input_ids={tuple(input_ids.shape)}"
            )

        prompt_embeds = model.get_input_embeddings()(input_ids)
        batch_size = int(prompt_embeds.shape[0])
        query_embeds = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        query_embeds = query_embeds.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        inputs_embeds = torch.cat([prompt_embeds, query_embeds], dim=1)
        action_attention_mask = build_parallel_action_attention_mask(
            attention_mask.to(device=prompt_embeds.device),
            self.action_query_len,
            dtype=prompt_embeds.dtype,
        )

        base_forward = getattr(model, "_llm_offline_original_forward", None)
        if base_forward is None:
            base_forward = model.forward
        outputs = base_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=action_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Continuous action forward requires model outputs.hidden_states")
        action_hidden = hidden_states[-1][:, -self.action_query_len :, :]
        head_dtype = next(self.action_head.parameters()).dtype
        action_hidden = action_hidden.to(dtype=head_dtype)
        return self.action_head(action_hidden)


def _resolve_hidden_size(model) -> int:
    config = getattr(model, "config", None)
    candidates = [config, getattr(config, "text_config", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(candidate, attr, None)
            if value is not None:
                return int(value)
    raise ValueError("Could not resolve model hidden size for ContinuousActionDecoder.")


def _is_continuous_forward_patched(model) -> bool:
    return getattr(getattr(model, "forward", None), "__func__", None) is _continuous_forward_patched


def _continuous_forward_patched(self, *args, continuous_action: bool = False, **kwargs):
    if not continuous_action:
        return self._llm_offline_original_forward(*args, **kwargs)

    if args:
        if "input_ids" in kwargs:
            raise ValueError("input_ids was provided both positionally and by keyword")
        kwargs["input_ids"] = args[0]
        args = args[1:]
    if args:
        raise ValueError("continuous_action forward accepts only input_ids as a positional argument")

    input_ids = kwargs.pop("input_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise ValueError(f"Unexpected continuous_action forward kwargs: {unexpected}")
    return self.continuous_action_decoder(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )


def attach_continuous_action_decoder(
    model,
    action_dim: int,
    *,
    hidden_size: int | None = None,
    action_query_len: int | None = None,
    action_head_num_blocks: int | None = None,
) -> ContinuousActionDecoder:
    """Attach and register a continuous decoder, then patch model.forward."""
    action_dim = int(action_dim)
    action_query_len = resolve_action_query_len(action_dim, action_query_len)
    action_head_num_blocks = resolve_action_head_num_blocks(action_head_num_blocks)
    hidden_size = _resolve_hidden_size(model) if hidden_size is None else int(hidden_size)

    decoder = getattr(model, "continuous_action_decoder", None)
    if decoder is None:
        decoder = ContinuousActionDecoder(
            action_dim=action_dim,
            hidden_size=hidden_size,
            action_query_len=action_query_len,
            action_head_num_blocks=action_head_num_blocks,
        )
        model.add_module("continuous_action_decoder", decoder)
    else:
        if int(decoder.action_dim) != action_dim:
            raise ValueError(
                "Existing continuous_action_decoder action_dim does not match config: "
                f"decoder={decoder.action_dim}, config={action_dim}"
            )
        if int(decoder.hidden_size) != hidden_size:
            raise ValueError(
                "Existing continuous_action_decoder hidden_size does not match model: "
                f"decoder={decoder.hidden_size}, model={hidden_size}"
            )
        if int(decoder.action_query_len) != action_query_len:
            raise ValueError(
                "Existing continuous_action_decoder action_query_len does not match config: "
                f"decoder={decoder.action_query_len}, config={action_query_len}"
            )
        if int(decoder.action_head_num_blocks) != action_head_num_blocks:
            raise ValueError(
                "Existing continuous_action_decoder action_head_num_blocks does not match config: "
                f"decoder={decoder.action_head_num_blocks}, config={action_head_num_blocks}"
            )

    if not _is_continuous_forward_patched(model):
        model._llm_offline_original_forward = model.forward
        model.forward = types.MethodType(_continuous_forward_patched, model)
    return decoder


def unpatch_continuous_action_forward(model):
    """Restore the underlying LM forward before external libraries patch it."""
    if _is_continuous_forward_patched(model) and hasattr(model, "_llm_offline_original_forward"):
        model.forward = model._llm_offline_original_forward


def ensure_continuous_action_decoder(model, config: dict) -> ContinuousActionDecoder | None:
    if str(config.get("action_token_mode", "text")) != "parallel_l1":
        return None
    if "action_dim" not in config:
        raise ValueError("action_token_mode='parallel_l1' requires config['action_dim']")
    action_dim = int(config["action_dim"])
    return attach_continuous_action_decoder(
        model,
        action_dim,
        action_query_len=resolve_action_query_len(action_dim, config.get("action_query_len")),
        action_head_num_blocks=resolve_action_head_num_blocks(
            config.get("action_head_num_blocks")
        ),
    )


def save_continuous_action_decoder(model, checkpoint_dir: str):
    decoder = getattr(model, "continuous_action_decoder", None)
    if decoder is None:
        raise ValueError("Cannot save parallel_l1 checkpoint without continuous_action_decoder")
    payload = {
        "action_dim": int(decoder.action_dim),
        "action_query_len": int(decoder.action_query_len),
        "action_head_num_blocks": int(decoder.action_head_num_blocks),
        "hidden_size": int(decoder.hidden_size),
        "state_dict": decoder.state_dict(),
    }
    torch.save(payload, os.path.join(checkpoint_dir, CONTINUOUS_ACTION_DECODER_FILENAME))


def load_continuous_action_decoder(
    model,
    checkpoint_dir: str,
    *,
    expected_action_dim: int | None = None,
    expected_action_query_len: int | None = None,
    expected_action_head_num_blocks: int | None = None,
):
    path = os.path.join(checkpoint_dir, CONTINUOUS_ACTION_DECODER_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing continuous action decoder checkpoint: {path}. "
            "parallel_l1 checkpoints must include this file."
        )
    payload: dict[str, Any] = torch.load(path, map_location="cpu")
    action_dim = int(payload["action_dim"])
    action_query_len = resolve_action_query_len(action_dim, payload.get("action_query_len"))
    action_head_num_blocks = resolve_action_head_num_blocks(payload.get("action_head_num_blocks"))
    hidden_size = int(payload["hidden_size"])
    if expected_action_dim is not None and action_dim != int(expected_action_dim):
        raise ValueError(
            "continuous_action_decoder.pt action_dim does not match config.yaml: "
            f"decoder={action_dim}, config={expected_action_dim}"
        )
    if (
        expected_action_query_len is not None
        and action_query_len != resolve_action_query_len(action_dim, expected_action_query_len)
    ):
        raise ValueError(
            "continuous_action_decoder.pt action_query_len does not match config.yaml: "
            f"decoder={action_query_len}, config={expected_action_query_len}"
        )
    if (
        expected_action_head_num_blocks is not None
        and action_head_num_blocks
        != resolve_action_head_num_blocks(expected_action_head_num_blocks)
    ):
        raise ValueError(
            "continuous_action_decoder.pt action_head_num_blocks does not match config.yaml: "
            f"decoder={action_head_num_blocks}, config={expected_action_head_num_blocks}"
        )
    model_hidden_size = _resolve_hidden_size(model)
    if hidden_size != model_hidden_size:
        raise ValueError(
            "continuous_action_decoder.pt hidden_size does not match model config: "
            f"decoder={hidden_size}, model={model_hidden_size}"
        )
    decoder = attach_continuous_action_decoder(
        model,
        action_dim=action_dim,
        hidden_size=hidden_size,
        action_query_len=action_query_len,
        action_head_num_blocks=action_head_num_blocks,
    )
    decoder.load_state_dict(payload["state_dict"])
    return decoder
