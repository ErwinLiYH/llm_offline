"""Helpers for building model-native chat-template prompts."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase


def _ensure_chat_template(tokenizer: PreTrainedTokenizerBase):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer {tokenizer.name_or_path!r} does not define a chat_template; "
            "cannot use model-native chat formatting."
        )


def _apply_chat_template_no_thinking(tokenizer: PreTrainedTokenizerBase, messages: list[dict], **kwargs) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_generation_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    _ensure_chat_template(tokenizer)
    return _apply_chat_template_no_thinking(
        tokenizer,
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_training_conversation(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    assistant_text: str,
) -> str:
    _ensure_chat_template(tokenizer)
    return _apply_chat_template_no_thinking(
        tokenizer,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
