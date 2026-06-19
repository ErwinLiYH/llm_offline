"""Shared rollout helpers for evaluation-style model action selection."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from model.continuous_action import resolve_student_t_df
from model.mtp_bin import (
    _module_weight_dtype,
    resolve_mtp_k,
    resolve_mtp_quadratic_decoding,
    uses_simple_mtp_bin,
    uses_mtp_bin,
)
from utils.action_bins import (
    bin_to_continuous,
    get_action_bin_codec,
    get_action_bin_range,
    get_action_num_bins,
    get_action_token_mode,
    get_tokenizer_backend,
    uses_action_bins,
    uses_continuous_actions,
    uses_gaussian_continuous_actions,
    uses_student_t_continuous_actions,
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
    gaussian_action_mean: list[float] | None = None
    gaussian_action_std: list[float] | None = None
    student_t_action_mean: list[float] | None = None
    student_t_action_scale: list[float] | None = None


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


def _filtered_bin_logits(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0 and top_k < filtered.shape[-1]:
        kth_values = torch.topk(filtered, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        filtered = filtered.masked_fill(filtered < kth_values, -float("inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove_sorted = cumulative > top_p
        remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
        remove_sorted[:, 0] = False
        remove = torch.zeros_like(remove_sorted)
        remove.scatter_(dim=-1, index=sorted_indices, src=remove_sorted)
        filtered = filtered.masked_fill(remove, -float("inf"))
    return filtered


def _select_action_bin_indices(
    bin_logits: torch.Tensor,
    action_generation_config: dict,
) -> list[int]:
    logits = bin_logits.float()
    if not action_generation_config["action_sampling"]:
        return [int(value) for value in torch.argmax(logits, dim=-1).detach().cpu().tolist()]

    logits = logits / float(action_generation_config["action_temperature"])
    logits = _filtered_bin_logits(
        logits,
        top_k=int(action_generation_config["action_top_k"]),
        top_p=float(action_generation_config["action_top_p"]),
    )
    probs = torch.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return [int(value) for value in sampled.detach().cpu().tolist()]


def _get_mtp_decoder(model):
    decoder = getattr(model, "mtp_bin_decoder", None)
    if decoder is None and hasattr(model, "module"):
        decoder = getattr(model.module, "mtp_bin_decoder", None)
    if decoder is None:
        raise RuntimeError("mtp_bin eval requires an initialized mtp_bin_decoder.")
    return decoder


def _build_mtp_eval_tensors(
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prefix_token_ids: list[int],
    *,
    query_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    if prompt_input_ids.dim() != 2 or prompt_input_ids.shape[0] != 1:
        raise ValueError("mtp_bin eval currently expects a single prompt batch.")
    prefix = torch.tensor([prefix_token_ids], device=device, dtype=prompt_input_ids.dtype)
    prefix_attention = torch.ones_like(prefix, dtype=prompt_attention_mask.dtype)
    query_ids = torch.zeros((1, query_count), device=device, dtype=prompt_input_ids.dtype)
    query_attention = torch.ones_like(query_ids, dtype=prompt_attention_mask.dtype)
    input_ids = torch.cat([prompt_input_ids, prefix, query_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, prefix_attention, query_attention], dim=1)

    seq_len = input_ids.shape[1]
    prompt_len = int(prompt_input_ids.shape[1])
    source_pos = prompt_len + len(prefix_token_ids) - 1 if prefix_token_ids else prompt_len - 1
    if source_pos < 0:
        raise ValueError("mtp_bin eval requires a non-empty prompt.")
    action_query_mask = torch.zeros((1, seq_len), device=device, dtype=torch.bool)
    action_query_offsets = torch.full((1, seq_len), -1, device=device, dtype=torch.long)
    action_query_source_positions = torch.full((1, seq_len), -1, device=device, dtype=torch.long)
    action_query_anchor_positions = torch.full((1, seq_len), -1, device=device, dtype=torch.long)
    action_query_prev_token_ids = torch.zeros((1, seq_len), device=device, dtype=torch.long)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    for offset in range(query_count):
        pos = prompt_len + len(prefix_token_ids) + offset
        action_query_mask[0, pos] = True
        action_query_offsets[0, pos] = offset
        action_query_source_positions[0, pos] = source_pos
        action_query_anchor_positions[0, pos] = source_pos + 1 + offset
        position_ids[0, pos] = source_pos + 1 + offset

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "action_query_mask": action_query_mask,
        "action_query_offsets": action_query_offsets,
        "action_query_source_positions": action_query_source_positions,
        "action_query_anchor_positions": action_query_anchor_positions,
        "action_query_prev_token_ids": action_query_prev_token_ids,
        "source_pos": source_pos,
    }


def _build_simple_mtp_eval_tensors(
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    *,
    action_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    if prompt_input_ids.dim() != 2 or prompt_input_ids.shape[0] != 1:
        raise ValueError("simple_mtp_bin eval currently expects a single prompt batch.")
    query_ids = torch.zeros((1, action_dim), device=device, dtype=prompt_input_ids.dtype)
    query_attention = torch.ones_like(query_ids, dtype=prompt_attention_mask.dtype)
    input_ids = torch.cat([prompt_input_ids, query_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, query_attention], dim=1)

    seq_len = input_ids.shape[1]
    prompt_len = int(prompt_input_ids.shape[1])
    source_pos = prompt_len - 1
    if source_pos < 0:
        raise ValueError("simple_mtp_bin eval requires a non-empty prompt.")
    action_query_mask = torch.zeros((1, seq_len), device=device, dtype=torch.bool)
    action_query_offsets = torch.full((1, seq_len), -1, device=device, dtype=torch.long)
    action_query_source_positions = torch.full((1, seq_len), -1, device=device, dtype=torch.long)
    action_query_anchor_positions = torch.full((1, seq_len), -1, device=device, dtype=torch.long)
    action_query_prev_token_ids = torch.zeros((1, seq_len), device=device, dtype=torch.long)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    for dim_idx in range(action_dim):
        pos = prompt_len + dim_idx
        action_query_mask[0, pos] = True
        action_query_offsets[0, pos] = dim_idx
        action_query_source_positions[0, pos] = source_pos
        action_query_anchor_positions[0, pos] = source_pos + dim_idx
        position_ids[0, pos] = source_pos + dim_idx

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "action_query_mask": action_query_mask,
        "action_query_offsets": action_query_offsets,
        "action_query_source_positions": action_query_source_positions,
        "action_query_anchor_positions": action_query_anchor_positions,
        "action_query_prev_token_ids": action_query_prev_token_ids,
        "source_pos": source_pos,
    }


def generate_simple_mtp_bin_action(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    action_dim: int,
    config: dict,
    action_codec,
    collect_probabilities: bool,
    action_generation_config: dict,
) -> tuple[str, list[int], list[list[float]]]:
    if action_codec is None:
        raise RuntimeError("simple_mtp_bin eval requires an initialized action codec.")
    if action_generation_config.get("action_sampling", False):
        raise ValueError("simple_mtp_bin eval currently requires action_sampling: false.")
    _get_mtp_decoder(model)
    action_dim = int(action_dim)

    tokenizer_kwargs = {
        "text": build_generation_prompt(tokenizer, prompt),
        "return_tensors": "pt",
        "add_special_tokens": False,
    }
    if config.get("max_length") is not None:
        tokenizer_kwargs["max_length"] = max(int(config["max_length"]) - action_dim, 1)
        tokenizer_kwargs["truncation"] = True
    encoded = tokenizer(**tokenizer_kwargs)
    prompt_input_ids = encoded.input_ids.to(device)
    prompt_attention_mask = encoded.attention_mask.to(device)
    bin_token_ids = torch.tensor(action_codec.model_token_ids, device=device, dtype=torch.long)

    tensors = _build_simple_mtp_eval_tensors(
        prompt_input_ids,
        prompt_attention_mask,
        action_dim=action_dim,
        device=device,
    )
    with torch.no_grad():
        output = model(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            position_ids=tensors["position_ids"],
            action_query_mask=tensors["action_query_mask"],
            action_query_offsets=tensors["action_query_offsets"],
            action_query_source_positions=tensors["action_query_source_positions"],
            action_query_prev_token_ids=tensors["action_query_prev_token_ids"],
            mtp_bin=True,
        )
    query_logits = output.sampler_logits.float().index_select(dim=-1, index=bin_token_ids)
    selected_bins = [int(value) for value in query_logits.argmax(dim=-1).detach().cpu().tolist()]
    selected_token_ids = action_codec.token_ids_for_bins(selected_bins)
    distributions = []
    if collect_probabilities:
        distributions = [
            [float(value) for value in row]
            for row in torch.softmax(query_logits.float(), dim=-1).detach().cpu().tolist()
        ]
    return action_codec.display_text_for_bins(selected_bins), selected_token_ids, distributions


def generate_mtp_bin_action(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    action_dim: int,
    config: dict,
    action_codec,
    collect_probabilities: bool,
    action_generation_config: dict,
) -> tuple[str, list[int], list[list[float]]]:
    if action_codec is None:
        raise RuntimeError("mtp_bin eval requires an initialized action codec.")
    if action_generation_config.get("action_sampling", False):
        raise ValueError("mtp_bin eval currently requires action_sampling: false.")
    use_quadratic_decoding = resolve_mtp_quadratic_decoding(config)
    decoder = _get_mtp_decoder(model)
    action_dim = int(action_dim)
    mtp_k = resolve_mtp_k(action_dim, config.get("mtp_k"))
    if mtp_k != action_dim - 1:
        raise ValueError("mtp_bin eval expects mtp_k == action_dim - 1.")

    tokenizer_kwargs = {
        "text": build_generation_prompt(tokenizer, prompt),
        "return_tensors": "pt",
        "add_special_tokens": False,
    }
    if config.get("max_length") is not None:
        tokenizer_kwargs["max_length"] = max(int(config["max_length"]) - action_dim - mtp_k, 1)
        tokenizer_kwargs["truncation"] = True
    encoded = tokenizer(**tokenizer_kwargs)
    prompt_input_ids = encoded.input_ids.to(device)
    prompt_attention_mask = encoded.attention_mask.to(device)
    bin_token_ids = torch.tensor(action_codec.model_token_ids, device=device, dtype=torch.long)

    def propose(prefix_token_ids: list[int]) -> tuple[list[int], list[int], list[list[float]]]:
        remaining = action_dim - len(prefix_token_ids)
        if remaining <= 0:
            return [], [], []
        query_count = min(mtp_k, remaining - 1)
        tensors = _build_mtp_eval_tensors(
            prompt_input_ids,
            prompt_attention_mask,
            prefix_token_ids,
            query_count=query_count,
            device=device,
        )
        with torch.no_grad():
            output = model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                position_ids=tensors["position_ids"],
                action_query_mask=tensors["action_query_mask"],
                action_query_offsets=tensors["action_query_offsets"],
                action_query_source_positions=tensors["action_query_source_positions"],
                action_query_prev_token_ids=tensors["action_query_prev_token_ids"],
                mtp_bin=True,
            )
        source_pos = int(tensors["source_pos"])
        ntp_bin_logits = output.logits[0, source_pos].index_select(dim=-1, index=bin_token_ids)
        ntp_bin = int(ntp_bin_logits.argmax(dim=-1).item())
        bins = [ntp_bin]
        token_ids = [int(action_codec.model_token_ids[ntp_bin])]
        distributions = []
        if collect_probabilities:
            distributions.append(
                [float(value) for value in torch.softmax(ntp_bin_logits.float(), dim=-1).cpu().tolist()]
            )

        if query_count > 0:
            query_hidden = output.hidden_states[tensors["action_query_mask"]]
            previous_token_id = token_ids[0]
            embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is None:
                raise ValueError("mtp_bin eval requires a model output embedding/lm_head")
            sampler_dtype = _module_weight_dtype(decoder.sampler_head, query_hidden.dtype)
            output_dtype = _module_weight_dtype(output_embeddings, query_hidden.dtype)
            for query_idx in range(query_count):
                prev_ids = torch.tensor([previous_token_id], device=device, dtype=torch.long)
                prev_embed = embeddings(prev_ids).to(dtype=sampler_dtype)
                sampler_hidden = decoder.sampler_head(
                    query_hidden[query_idx : query_idx + 1].to(dtype=sampler_dtype),
                    prev_embed,
                )
                sampler_hidden = sampler_hidden.to(dtype=output_dtype)
                sampler_logits = output_embeddings(sampler_hidden)[0]
                sampler_bin_logits = sampler_logits.index_select(dim=-1, index=bin_token_ids)
                sampler_bin = int(sampler_bin_logits.argmax(dim=-1).item())
                bins.append(sampler_bin)
                previous_token_id = int(action_codec.model_token_ids[sampler_bin])
                token_ids.append(previous_token_id)
                if collect_probabilities:
                    distributions.append(
                        [
                            float(value)
                            for value in torch.softmax(sampler_bin_logits.float(), dim=-1).cpu().tolist()
                        ]
                    )
        return bins, token_ids, distributions

    proposal_bins, proposal_token_ids, proposal_distributions = propose([])
    if len(proposal_bins) < action_dim:
        raise RuntimeError(
            "mtp_bin decoding produced too few action bins: "
            f"got {len(proposal_bins)}, expected {action_dim}."
        )
    if not use_quadratic_decoding:
        selected_bins = proposal_bins[:action_dim]
        selected_token_ids = proposal_token_ids[:action_dim]
        selected_distributions = proposal_distributions[:action_dim] if collect_probabilities else []
        return (
            action_codec.display_text_for_bins(selected_bins),
            selected_token_ids,
            selected_distributions,
        )

    verified_bins = [proposal_bins[0]]
    verified_token_ids = [proposal_token_ids[0]]
    speculative_bins = proposal_bins[1:]
    speculative_token_ids = proposal_token_ids[1:]
    speculative_distributions = proposal_distributions[1:]
    final_distributions = proposal_distributions[:1] if collect_probabilities else []

    while len(verified_bins) < action_dim:
        verifier_bins, verifier_token_ids, verifier_distributions = propose(verified_token_ids)
        verifier_bin = verifier_bins[0]
        verifier_token_id = verifier_token_ids[0]
        if speculative_bins and speculative_bins[0] == verifier_bin:
            chosen_bin = speculative_bins.pop(0)
            chosen_token_id = speculative_token_ids.pop(0)
            chosen_distribution = speculative_distributions.pop(0) if speculative_distributions else []
        else:
            chosen_bin = verifier_bin
            chosen_token_id = verifier_token_id
            chosen_distribution = verifier_distributions[0] if verifier_distributions else []
            speculative_bins = verifier_bins[1:]
            speculative_token_ids = verifier_token_ids[1:]
            speculative_distributions = verifier_distributions[1:]
        verified_bins.append(chosen_bin)
        verified_token_ids.append(chosen_token_id)
        if collect_probabilities:
            final_distributions.append(chosen_distribution)

    generated_token_ids = action_codec.token_ids_for_bins(verified_bins)
    generated = action_codec.display_text_for_bins(verified_bins)
    return generated, generated_token_ids, final_distributions


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
        return generate_valid_continuous_actions_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            formatter=formatter,
            prompts=[prompt],
            config=config,
            action_context=action_context,
            action_shape=action_shape,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        )[0]

    if uses_mtp_bin(config):
        t0 = time.perf_counter()
        if uses_simple_mtp_bin(config):
            generated, generated_token_ids, distributions = generate_simple_mtp_bin_action(
                model,
                tokenizer,
                prompt,
                device,
                action_dim=action_dim,
                config=config,
                action_codec=action_context.action_codec,
                collect_probabilities=action_context.collect_bin_probabilities,
                action_generation_config=action_context.action_generation_config,
            )
        else:
            generated, generated_token_ids, distributions = generate_mtp_bin_action(
                model,
                tokenizer,
                prompt,
                device,
                action_dim=action_dim,
                config=config,
                action_codec=action_context.action_codec,
                collect_probabilities=action_context.collect_bin_probabilities,
                action_generation_config=action_context.action_generation_config,
            )
        action_time_seconds = time.perf_counter() - t0
        generated_probability_logs = []
        if action_context.collect_bin_probabilities:
            probability_log = format_action_bin_probability_log(
                distributions,
                config,
                action_context.action_codec,
            )
            generated_probability_logs.append(
                f"[Attempt 1]\n{probability_log}" if probability_log else "[Attempt 1]"
            )

        parsed_action, success = parse_action_for_mode(
            formatter,
            generated,
            generated_token_ids,
            config,
            action_dim=action_dim,
            action_codec=action_context.action_codec,
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
        parse_failures = 0
        if success and formatter.validate_action(parsed_action):
            action = np.clip(parsed_action.reshape(action_shape), low, high).astype(np.float32)
            parse_status = "success"
            fallback_count = 0
        else:
            action = np.clip(np.zeros(action_shape, dtype=np.float32), low, high).astype(np.float32)
            parse_status = "fallback"
            fallback_count = 1
            parse_failures = 1
        executed_action_text = format_action_for_mode(
            formatter,
            action.reshape(-1),
            config,
            action_context.action_codec,
        )
        return GeneratedActionResult(
            action=action,
            executed_action_text=executed_action_text,
            generated_attempts=[generated],
            generated_probability_logs=generated_probability_logs,
            attempt_count=1,
            parse_status=parse_status,
            parse_failures=parse_failures,
            fallback_count=fallback_count,
            action_time_seconds=action_time_seconds,
            generation_count=1,
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


def generate_valid_continuous_actions_batch(
    *,
    model,
    tokenizer,
    device: torch.device,
    formatter,
    prompts: list[str],
    config: dict,
    action_context: ActionRolloutContext,
    action_shape: tuple[int, ...],
    action_dim: int,
    action_low=None,
    action_high=None,
) -> list[GeneratedActionResult]:
    if not prompts:
        return []
    if not uses_continuous_actions(config):
        raise ValueError(
            "generate_valid_continuous_actions_batch requires a continuous action mode"
        )

    t0 = time.perf_counter()
    tokenizer_kwargs = {
        "text": [build_generation_prompt(tokenizer, prompt) for prompt in prompts],
        "return_tensors": "pt",
        "add_special_tokens": False,
        "padding": len(prompts) > 1,
    }
    if config.get("max_length") is not None:
        tokenizer_kwargs["max_length"] = int(config["max_length"])
        tokenizer_kwargs["truncation"] = True

    tokenizer_backend = get_tokenizer_backend(tokenizer)
    original_padding_side = getattr(tokenizer_backend, "padding_side", None)
    original_pad_token = getattr(tokenizer_backend, "pad_token", None)
    needs_temporary_pad_token = len(prompts) > 1 and original_pad_token is None
    temporary_pad_token = None
    if needs_temporary_pad_token:
        temporary_pad_token = getattr(tokenizer_backend, "eos_token", None)
        if temporary_pad_token is None:
            raise ValueError(
                "Batched continuous eval requires tokenizer.pad_token or eos_token"
            )
    if original_padding_side is not None:
        tokenizer_backend.padding_side = "left"
    if needs_temporary_pad_token:
        tokenizer_backend.pad_token = temporary_pad_token
    try:
        encoded = tokenizer(**tokenizer_kwargs)
    finally:
        if original_padding_side is not None:
            tokenizer_backend.padding_side = original_padding_side
        if needs_temporary_pad_token:
            tokenizer_backend.pad_token = None

    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    with torch.no_grad():
        predicted = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            continuous_action=True,
        )
    batch_action_time_seconds = time.perf_counter() - t0

    gaussian_mean = None
    gaussian_std = None
    student_t_mean = None
    student_t_scale = None
    if uses_gaussian_continuous_actions(config):
        mean = predicted.mean.float()
        if predicted.latent_mean is None:
            raise RuntimeError(
                "parallel_gaussian requires latent_mean from the continuous decoder"
            )
        latent_mean = predicted.latent_mean.float()
        gaussian_mean = mean
        gaussian_std = predicted.std.float()
        if action_context.action_generation_config["action_sampling"]:
            selected = torch.tanh(torch.normal(mean=latent_mean, std=gaussian_std))
        else:
            selected = mean
    elif uses_student_t_continuous_actions(config):
        mean = predicted.mean.float()
        student_t_mean = mean
        student_t_scale = predicted.scale.float()
        if action_context.action_generation_config["action_sampling"]:
            df = torch.as_tensor(
                resolve_student_t_df(config),
                dtype=mean.dtype,
                device=mean.device,
            )
            selected = torch.distributions.StudentT(
                df,
                loc=mean,
                scale=student_t_scale,
            ).sample()
        else:
            selected = mean
    else:
        selected = predicted

    selected = selected.detach().float().cpu().numpy().astype(np.float32)
    if selected.shape != (len(prompts), action_dim):
        raise ValueError(
            "Continuous action decoder returned unexpected shape: "
            f"got {selected.shape}, expected {(len(prompts), action_dim)}"
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
    action_time_seconds = batch_action_time_seconds / len(prompts)
    results = []
    for batch_idx, raw_action in enumerate(selected):
        action = np.clip(raw_action.reshape(action_shape), low, high).astype(np.float32)
        if formatter.validate_action(action.reshape(-1)):
            parse_status = "success"
            fallback_count = 0
        else:
            action = np.clip(
                np.zeros(action_shape, dtype=np.float32),
                low,
                high,
            ).astype(np.float32)
            parse_status = "fallback"
            fallback_count = 1
        executed_action_text = formatter.format_action(action.reshape(-1))
        results.append(
            GeneratedActionResult(
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
                gaussian_action_mean=(
                    [
                        float(value)
                        for value in gaussian_mean[batch_idx].detach().cpu().tolist()
                    ]
                    if gaussian_mean is not None
                    else None
                ),
                gaussian_action_std=(
                    [
                        float(value)
                        for value in gaussian_std[batch_idx].detach().cpu().tolist()
                    ]
                    if gaussian_std is not None
                    else None
                ),
                student_t_action_mean=(
                    [
                        float(value)
                        for value in student_t_mean[batch_idx].detach().cpu().tolist()
                    ]
                    if student_t_mean is not None
                    else None
                ),
                student_t_action_scale=(
                    [
                        float(value)
                        for value in student_t_scale[batch_idx].detach().cpu().tolist()
                    ]
                    if student_t_scale is not None
                    else None
                ),
            )
        )
    return results
