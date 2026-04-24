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
) -> str:
    base = format_prompt_action_text(prompt, action)
    return (
        f"{base}\n\n"
        f"Executed Action:\n{executed_action}\n\n"
        f"Parse Status:\n{parse_status}\n\n"
        f"Attempt Count:\n{attempt_count}"
    )
