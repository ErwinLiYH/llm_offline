from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch

from utils.action_bins import uses_continuous_actions
from utils.eval_rollout import (
    build_action_rollout_context,
    generate_valid_action,
    generate_valid_continuous_actions_batch,
)
from utils.rollout.protocol import ActionRequest, ActionResponse


def _to_response(request: ActionRequest, result) -> ActionResponse:
    return ActionResponse(
        request_id=request.request_id,
        action=[float(value) for value in np.asarray(result.action).reshape(-1).tolist()],
        executed_action_text=result.executed_action_text,
        generated_attempts=list(result.generated_attempts),
        generated_probability_logs=list(result.generated_probability_logs),
        attempt_count=int(result.attempt_count),
        parse_status=str(result.parse_status),
        parse_failures=int(result.parse_failures),
        fallback_count=int(result.fallback_count),
        action_time_seconds=float(result.action_time_seconds),
        generation_count=int(result.generation_count),
        raw_continuous_action=result.raw_continuous_action,
        gaussian_action_mean=result.gaussian_action_mean,
        gaussian_action_std=result.gaussian_action_std,
        student_t_action_mean=result.student_t_action_mean,
        student_t_action_scale=result.student_t_action_scale,
    )


class RolloutPolicy:
    """Parent-process policy inference for worker action requests."""

    def __init__(
        self,
        *,
        config: dict,
        model,
        tokenizer,
        device: torch.device,
        formatter,
        collect_bin_probabilities: bool,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.formatter = formatter
        self.collect_bin_probabilities = bool(collect_bin_probabilities)
        self._contexts = {}

    def _context(self, action_dim: int):
        key = (int(action_dim), self.collect_bin_probabilities)
        if key not in self._contexts:
            self._contexts[key] = build_action_rollout_context(
                config=self.config,
                tokenizer=self.tokenizer,
                action_dim=int(action_dim),
                collect_bin_probabilities=self.collect_bin_probabilities,
            )
        return self._contexts[key]

    def respond(self, requests: list[ActionRequest]) -> list[ActionResponse]:
        if not requests:
            return []
        if uses_continuous_actions(self.config):
            return self._respond_continuous(requests)
        return [self._respond_single(request) for request in requests]

    def _respond_single(self, request: ActionRequest) -> ActionResponse:
        action_shape = tuple(int(value) for value in request.action_shape)
        action_low = (
            None
            if request.action_low is None
            else np.asarray(request.action_low, dtype=np.float32).reshape(action_shape)
        )
        action_high = (
            None
            if request.action_high is None
            else np.asarray(request.action_high, dtype=np.float32).reshape(action_shape)
        )
        result = generate_valid_action(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            formatter=self.formatter,
            prompt=request.prompt,
            config=self.config,
            action_context=self._context(request.action_dim),
            action_shape=action_shape,
            action_dim=int(request.action_dim),
            parse_retry_limit=int(self.config.get("parse_retry_limit", 3)),
            action_low=action_low,
            action_high=action_high,
        )
        return _to_response(request, result)

    def _respond_continuous(self, requests: list[ActionRequest]) -> list[ActionResponse]:
        grouped: dict[tuple, list[ActionRequest]] = defaultdict(list)
        for request in requests:
            key = (
                tuple(int(value) for value in request.action_shape),
                int(request.action_dim),
                tuple(request.action_low or ()),
                tuple(request.action_high or ()),
            )
            grouped[key].append(request)

        responses_by_id = {}
        for (action_shape, action_dim, action_low_values, action_high_values), group in grouped.items():
            action_low = (
                None
                if not action_low_values
                else np.asarray(action_low_values, dtype=np.float32).reshape(action_shape)
            )
            action_high = (
                None
                if not action_high_values
                else np.asarray(action_high_values, dtype=np.float32).reshape(action_shape)
            )
            results = generate_valid_continuous_actions_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                formatter=self.formatter,
                prompts=[request.prompt for request in group],
                config=self.config,
                action_context=self._context(action_dim),
                action_shape=action_shape,
                action_dim=int(action_dim),
                action_low=action_low,
                action_high=action_high,
            )
            for request, result in zip(group, results, strict=True):
                responses_by_id[request.request_id] = _to_response(request, result)

        return [responses_by_id[request.request_id] for request in requests]

