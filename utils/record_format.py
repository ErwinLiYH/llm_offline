"""Helpers for rendering prompt/action records as plain text."""

from __future__ import annotations


def format_prompt_action_text(prompt: str, action: str) -> str:
    return f"Prompt:\n{prompt}\n\nAction:\n{action}"


def format_eval_step_text(
    prompt: str,
    action: str,
    *,
    executed_action: str,
    parse_status: str,
    attempt_count: int,
    action_bin_probabilities: str | None = None,
    raw_continuous_action: list[float] | None = None,
    gaussian_action_mean: list[float] | None = None,
    gaussian_action_std: list[float] | None = None,
    student_t_action_mean: list[float] | None = None,
    student_t_action_scale: list[float] | None = None,
) -> str:
    base = format_prompt_action_text(prompt, action)
    text = (
        f"{base}\n\n"
        f"Executed Action:\n{executed_action}\n\n"
        f"Parse Status:\n{parse_status}\n\n"
        f"Attempt Count:\n{attempt_count}"
    )
    if action_bin_probabilities:
        text = f"{text}\n\nAction Bin Probabilities:\n{action_bin_probabilities}"
    if raw_continuous_action is not None:
        values = ", ".join(f"{float(value):.8f}" for value in raw_continuous_action)
        text = f"{text}\n\nRaw Continuous Action:\n[{values}]"
    if gaussian_action_mean is not None:
        values = ", ".join(f"{float(value):.8f}" for value in gaussian_action_mean)
        text = f"{text}\n\nGaussian Action Mean:\n[{values}]"
    if gaussian_action_std is not None:
        values = ", ".join(f"{float(value):.8f}" for value in gaussian_action_std)
        text = f"{text}\n\nGaussian Action Std:\n[{values}]"
    if student_t_action_mean is not None:
        values = ", ".join(f"{float(value):.8f}" for value in student_t_action_mean)
        text = f"{text}\n\nStudentT Action Mean:\n[{values}]"
    if student_t_action_scale is not None:
        values = ", ".join(f"{float(value):.8f}" for value in student_t_action_scale)
        text = f"{text}\n\nStudentT Action Scale:\n[{values}]"
    return text
