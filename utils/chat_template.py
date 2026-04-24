"""Helpers for building model-native chat-template prompts."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase


def _ensure_chat_template(tokenizer: PreTrainedTokenizerBase):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer {tokenizer.name_or_path!r} does not define a chat_template; "
            "cannot use model-native chat formatting."
        )


def build_generation_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    _ensure_chat_template(tokenizer)
    return tokenizer.apply_chat_template(
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
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
