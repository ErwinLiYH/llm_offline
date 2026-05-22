"""Shared rollout helpers for evaluation-style model action selection."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from utils.action_bins import (
    bin_to_continuous,
    get_action_bin_codec,
    get_action_bin_range,
    get_action_num_bins,
    get_action_token_mode,
    uses_action_bins,
    uses_continuous_actions,
)
from utils.chat_template import build_generation_prompt
from utils.prompt_loader import render_template


class AllowedTokenIdsLogitsProcessor(LogitsProcessor):
    """Mask generation logits so only the provided token IDs remain valid."""

    def __init__(self, allowed_token_ids):
        token_ids = sorted({int(token_id) for token_id in allowed_token_ids})
        if not token_ids:
            raise ValueError("allowed_token_ids must contain at least one token ID")
        self.allowed_token_ids = tuple(token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        allowed = torch.tensor(self.allowed_token_ids, device=scores.device, dtype=torch.long)
        masked_scores = scores.new_full(scores.shape, -float("inf"))
        masked_scores[:, allowed] = scores[:, allowed]
        return masked_scores


@dataclass(frozen=True)
class ActionRolloutContext:
    action_token_mode: str
    action_generation_config: dict
    action_codec: object | None
    collect_bin_probabilities: bool
    generation_max_new_tokens: int
    allowed_token_ids: tuple[int, ...] | None


@dataclass(frozen=True)
class GeneratedActionResult:
    action: np.ndarray
    executed_action_text: str
    generated_attempts: list[str]
    generated_probability_logs: list[str]
    attempt_count: int
    parse_status: str
    parse_failures: int
    fallback_count: int
    action_time_seconds: float
    generation_count: int
    raw_continuous_action: list[float] | None = None


def resolve_action_generation_config(config: dict) -> dict:
    action_sampling = bool(config.get("action_sampling", False))
    action_temperature = float(config.get("action_temperature", 1.0))
    action_top_p = float(config.get("action_top_p", 1.0))
    action_top_k = int(config.get("action_top_k", 0))

    if action_temperature <= 0:
        raise ValueError(f"action_temperature must be > 0, got {action_temperature}")
    if action_top_p <= 0 or action_top_p > 1:
        raise ValueError(f"action_top_p must satisfy 0 < action_top_p <= 1, got {action_top_p}")
    if action_top_k < 0:
        raise ValueError(f"action_top_k must be >= 0, got {action_top_k}")

    return {
        "action_sampling": action_sampling,
        "action_temperature": action_temperature,
        "action_top_p": action_top_p,
        "action_top_k": action_top_k,
    }


def validate_history_config(config: dict) -> tuple[int, int]:
    history_num = int(config.get("history_num", 0))
    history_stride = int(config.get("history_stride", 1))
    if history_num < 0:
        raise ValueError(f"history_num must be >= 0, got {history_num}")
    if history_stride < 1:
        raise ValueError(f"history_stride must be >= 1, got {history_stride}")
    return history_num, history_stride


def sample_history_entries(
    history_buffer: list[dict],
    *,
    history_num: int,
    history_stride: int,
) -> list[dict]:
    if history_num <= 0:
        return []

    sampled_history = []
    hist_idx = len(history_buffer) - 1
    while hist_idx >= 0 and len(sampled_history) < history_num:
        sampled_entry = dict(history_buffer[hist_idx])
        sampled_entry["steps_ago"] = len(history_buffer) - hist_idx
        sampled_history.append(sampled_entry)
        hist_idx -= history_stride
    sampled_history.reverse()
    return sampled_history


def render_policy_prompt(
    *,
    formatter,
    template: str,
    prompt_vars: dict,
    obs,
    history_buffer: list[dict],
    history_num: int,
    history_stride: int,
) -> str:
    sampled_history = sample_history_entries(
        history_buffer,
        history_num=history_num,
        history_stride=history_stride,
    )
    history_payload = formatter.format_history(sampled_history, prompt_vars)
    obs_payload = formatter.format_obs(obs, prompt_vars)
    return render_template(template, prompt_vars, **obs_payload, **history_payload)


def generate_action(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 20,
    skip_special_tokens: bool = True,
    collect_scores: bool = False,
    action_codec=None,
    action_sampling: bool = False,
    action_temperature: float = 1.0,
    action_top_p: float = 1.0,
    action_top_k: int = 0,
    allowed_token_ids=None,
) -> tuple[str, list[int], tuple[torch.Tensor, ...] | None]:
    """Run inference and return display text plus generated action token IDs."""
    encoded = tokenizer(
        text=build_generation_prompt(tokenizer, prompt),
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    generate_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": action_sampling,
    }
    if action_sampling:
        generate_kwargs.update(
            {
                "temperature": action_temperature,
                "top_p": action_top_p,
                "top_k": action_top_k,
            }
        )
    if allowed_token_ids is not None:
        generate_kwargs["logits_processor"] = LogitsProcessorList(
            [AllowedTokenIdsLogitsProcessor(allowed_token_ids)]
        )
    if eos_token_id is None:
        warnings.warn(
            "Tokenizer/model does not define eos_token_id; generation will stop only at max_new_tokens.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        generate_kwargs["eos_token_id"] = eos_token_id
        generate_kwargs["pad_token_id"] = eos_token_id
    if collect_scores:
        generate_kwargs.update(
            {
                "return_dict_in_generate": True,
                "output_scores": True,
            }
        )

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            **generate_kwargs,
        )
    output_ids = outputs.sequences if collect_scores else outputs
    new_tokens = output_ids[0, input_ids.shape[1]:]
    generated_token_ids = [int(token_id) for token_id in new_tokens.detach().cpu().tolist()]
    if action_codec is None:
        text = tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
    else:
        text = action_codec.display_text_for_token_ids(tokenizer, generated_token_ids)
    if not collect_scores:
        return text, generated_token_ids, None
    return text, generated_token_ids, outputs.scores


def collect_action_bin_probabilities(
    scores,
    action_codec,
    action_dim: int,
) -> list[list[float]]:
    if not scores:
        return []
    bin_token_ids = torch.tensor(action_codec.model_token_ids, device=scores[0].device)
    distributions = []
    for score in scores[:action_dim]:
        bin_logits = score[0].index_select(dim=-1, index=bin_token_ids).float()
        bin_probs = torch.softmax(bin_logits, dim=-1)
        distributions.append([float(value) for value in bin_probs.detach().cpu().tolist()])
    return distributions


def format_action_bin_probability_log(distributions: list[list[float]], config: dict, action_codec) -> str:
    if not distributions:
        return ""
    num_bins = get_action_num_bins(config)
    low, high = get_action_bin_range(config)
    lines = []
    for dim_idx, probs in enumerate(distributions):
        lines.append(f"dim={dim_idx}")
        for bin_idx, prob in enumerate(probs):
            center = bin_to_continuous(bin_idx, num_bins, low, high)
            lines.append(
                f"  {action_codec.display_tokens[bin_idx]} "
                f"token_id={action_codec.model_token_ids[bin_idx]} "
                f"center={center:.6f} prob={prob:.8f}"
            )
    return "\n".join(lines)


def format_action_for_mode(formatter, action: np.ndarray, config: dict, action_codec=None) -> str:
    if not uses_action_bins(config):
        return formatter.format_action(action)
    if action_codec is None:
        raise RuntimeError("Action-bin display formatting requires an initialized action codec.")
    low, high = get_action_bin_range(config)
    return action_codec.display_text_for_action(action, low, high)


def parse_action_for_mode(
    formatter,
    text: str,
    token_ids: list[int],
    config: dict,
    *,
    action_dim: int,
    action_codec=None,
) -> tuple[np.ndarray, bool]:
    if get_action_token_mode(config) == "text":
        return formatter.parse_action(text)
    if action_codec is None:
        raise RuntimeError("Action-bin eval requires an initialized action codec.")
    low, high = get_action_bin_range(config)
    return action_codec.action_from_token_ids(token_ids, action_dim, low, high)


def build_action_rollout_context(
    *,
    config: dict,
    tokenizer,
    action_dim: int,
    collect_bin_probabilities: bool,
) -> ActionRolloutContext:
    action_token_mode = get_action_token_mode(config)
    action_generation_config = resolve_action_generation_config(config)
    action_codec = None
    if uses_action_bins(config):
        action_codec = get_action_bin_codec(tokenizer, config, ensure_registered=True)

    constrain_action_tokens = action_generation_config["action_sampling"] and uses_action_bins(config)
    generation_max_new_tokens = action_dim if constrain_action_tokens else 20
    allowed_token_ids = (
        tuple(int(token_id) for token_id in action_codec.model_token_ids)
        if constrain_action_tokens
        else None
    )
    return ActionRolloutContext(
        action_token_mode=action_token_mode,
        action_generation_config=action_generation_config,
        action_codec=action_codec,
        collect_bin_probabilities=collect_bin_probabilities,
        generation_max_new_tokens=generation_max_new_tokens,
        allowed_token_ids=allowed_token_ids,
    )


def generate_valid_action(
    *,
    model,
    tokenizer,
    device: torch.device,
    formatter,
    prompt: str,
    config: dict,
    action_context: ActionRolloutContext,
    action_shape: tuple[int, ...],
    action_dim: int,
    parse_retry_limit: int,
    action_low=None,
    action_high=None,
) -> GeneratedActionResult:
    if uses_continuous_actions(config):
        t0 = time.perf_counter()
        encoded = tokenizer(
            text=build_generation_prompt(tokenizer, prompt),
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        with torch.no_grad():
            predicted = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                continuous_action=True,
            )
        action_time_seconds = time.perf_counter() - t0
        raw_action = predicted[0].detach().float().cpu().numpy().astype(np.float32)
        if raw_action.shape != (action_dim,):
            raise ValueError(
                "Continuous action decoder returned unexpected shape: "
                f"got {raw_action.shape}, expected {(action_dim,)}"
            )
        low = (
            np.full(action_shape, -1.0, dtype=np.float32)
            if action_low is None
            else np.asarray(action_low, dtype=np.float32)
        )
        high = (
            np.full(action_shape, 1.0, dtype=np.float32)
            if action_high is None
            else np.asarray(action_high, dtype=np.float32)
        )
        action = np.clip(raw_action.reshape(action_shape), low, high).astype(np.float32)
        valid = formatter.validate_action(action.reshape(-1))
        if valid:
            parse_status = "success"
            fallback_count = 0
        else:
            action = np.clip(np.zeros(action_shape, dtype=np.float32), low, high).astype(np.float32)
            parse_status = "fallback"
            fallback_count = 1
        executed_action_text = formatter.format_action(action.reshape(-1))
        return GeneratedActionResult(
            action=action,
            executed_action_text=executed_action_text,
            generated_attempts=[executed_action_text],
            generated_probability_logs=[],
            attempt_count=1,
            parse_status=parse_status,
            parse_failures=0,
            fallback_count=fallback_count,
            action_time_seconds=action_time_seconds,
            generation_count=1,
            raw_continuous_action=[float(value) for value in raw_action.tolist()],
        )

    action = None
    executed_action_text = None
    generated_attempts = []
    generated_probability_logs = []
    attempt_count = 0
    parse_failures = 0
    action_time_seconds = 0.0
    generation_count = 0

    for _attempt in range(parse_retry_limit + 1):
        attempt_count += 1
        t0 = time.perf_counter()
        generated, generated_token_ids, generation_scores = generate_action(
            model,
            tokenizer,
            prompt,
            device,
            skip_special_tokens=action_context.action_token_mode == "text",
            collect_scores=action_context.collect_bin_probabilities,
            action_codec=action_context.action_codec,
            max_new_tokens=action_context.generation_max_new_tokens,
            allowed_token_ids=action_context.allowed_token_ids,
            **action_context.action_generation_config,
        )
        action_time_seconds += time.perf_counter() - t0
        generation_count += 1
        generated_attempts.append(generated)

        if action_context.collect_bin_probabilities:
            distributions = collect_action_bin_probabilities(
                generation_scores,
                action_context.action_codec,
                action_dim=action_dim,
            )
            probability_log = format_action_bin_probability_log(
                distributions,
                config,
                action_context.action_codec,
            )
            generated_probability_logs.append(
                f"[Attempt {attempt_count}]\n{probability_log}"
                if probability_log
                else f"[Attempt {attempt_count}]"
            )

        parsed_action, success = parse_action_for_mode(
            formatter,
            generated,
            generated_token_ids,
            config,
            action_dim=action_dim,
            action_codec=action_context.action_codec,
        )
        if success and formatter.validate_action(parsed_action):
            action = np.clip(parsed_action, -1.0, 1.0)
            executed_action_text = format_action_for_mode(
                formatter,
                action,
                config,
                action_context.action_codec,
            )
            break
        parse_failures += 1

    if action is None:
        action = np.zeros(action_shape, dtype=np.float32)
        executed_action_text = format_action_for_mode(
            formatter,
            action,
            config,
            action_context.action_codec,
        )
        fallback_count = 1
        parse_status = "fallback"
    else:
        fallback_count = 0
        parse_status = "success"

    return GeneratedActionResult(
        action=action,
        executed_action_text=executed_action_text,
        generated_attempts=generated_attempts,
        generated_probability_logs=generated_probability_logs,
        attempt_count=attempt_count,
        parse_status=parse_status,
        parse_failures=parse_failures,
        fallback_count=fallback_count,
        action_time_seconds=action_time_seconds,
        generation_count=generation_count,
    )
