import os
import json
import contextlib
import io

import yaml

with contextlib.redirect_stdout(io.StringIO()):
    from unsloth import FastLanguageModel

from utils.action_bins import (
    get_action_bin_token_ids,
    get_tokenizer_backend,
    register_action_tokens,
    uses_action_bins,
)


ACTION_BIN_TRAINABLE_TOKEN_MODULE = "embed_tokens"


def get_model_slug(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe slug.

    e.g. 'Qwen/Qwen3-0.6B' -> 'Qwen3-0.6B'
    """
    return model_name.split("/")[-1]


def _embedding_vocab_size(model) -> int | None:
    """Return the current input-embedding row count, or None if it cannot be read."""
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is None:
        return None
    return int(weight.shape[0])


def _ensure_pad_token(tokenizer):
    """Make causal-LM tokenizers usable for padded batches by reusing EOS as PAD."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _prepare_action_tokens_for_training(model, tokenizer, config: dict) -> list[int]:
    """Prepare a fresh training model for action-bin tokens.

    In `bin` / `gaussian_bin` mode this registers `<act_xx>` tokens on the
    tokenizer, resizes the model input embedding if the tokenizer grew, and
    returns the action token ids used by PEFT `trainable_token_indices`.
    This is the only path that is allowed to resize embeddings.
    """
    _ensure_pad_token(tokenizer)
    if not uses_action_bins(config):
        return []
    register_action_tokens(tokenizer, config)
    tok = get_tokenizer_backend(tokenizer)
    vocab_size = _embedding_vocab_size(model)
    if vocab_size is None or vocab_size < len(tok):
        model.resize_token_embeddings(len(tok))
    return get_action_bin_token_ids(tokenizer, config)


def _validate_action_tokens_for_checkpoint(model, tokenizer, config: dict):
    """Validate action-token structure after loading an adapter checkpoint.

    Checkpoint loading should restore the tokenizer and PEFT trainable-token
    adapter from disk. Therefore this function only checks that action tokens
    can be resolved and that the loaded model embedding is large enough for the
    tokenizer vocab. It intentionally does not resize embeddings; a mismatch
    means the checkpoint/tokenizer/base-model combination is inconsistent.
    """
    _ensure_pad_token(tokenizer)
    if not uses_action_bins(config):
        return
    register_action_tokens(tokenizer, config)
    tok = get_tokenizer_backend(tokenizer)
    vocab_size = _embedding_vocab_size(model)
    if vocab_size is None or vocab_size < len(tok):
        raise ValueError(
            "Tokenizer/model vocab size mismatch after loading action-bin checkpoint: "
            f"tokenizer={len(tok)}, model_embeddings={vocab_size}. The checkpoint tokenizer may be missing "
            "saved action tokens, or the adapter was saved with an incompatible base model."
        )
    get_action_bin_token_ids(tokenizer, config)


def _has_action_trainable_token_indices(adapter_config: dict) -> bool:
    """Return whether adapter_config declares saved trainable action-token rows."""
    token_indices = adapter_config.get("trainable_token_indices")
    if isinstance(token_indices, dict):
        indices = token_indices.get(ACTION_BIN_TRAINABLE_TOKEN_MODULE)
        return isinstance(indices, list) and len(indices) > 0
    return isinstance(token_indices, list) and len(token_indices) > 0


def _validate_action_bin_checkpoint(model_path: str, saved_config: dict):
    """Validate action-bin checkpoint metadata before loading model tensors.

    An action-bin adapter must declare `trainable_token_indices` in
    `adapter_config.json`; otherwise the `<act_xx>` embedding deltas were not
    saved and eval would usually produce near-uniform action-token probabilities.
    This metadata check is separate from `_validate_action_tokens_for_checkpoint`,
    which validates the loaded tokenizer/model embedding structure.
    """
    if not uses_action_bins(saved_config):
        return
    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg_path):
        return
    with open(adapter_cfg_path) as f:
        adapter_config = json.load(f)
    if _has_action_trainable_token_indices(adapter_config):
        return
    raise ValueError(
        "This checkpoint was trained with action_token_mode="
        f"{saved_config.get('action_token_mode')!r}, but adapter_config.json does not save action-token "
        "weights via trainable_token_indices. Its action-token embeddings cannot be restored, which makes "
        "action-bin probabilities uniform and parsing fail. Re-train with the current action-token setup, "
        "or evaluate a text-mode checkpoint."
    )


def load_model_and_tokenizer(config: dict):
    """Load base model + LoRA adapters and tokenizer from config using Unsloth."""
    model_name = config["model_name"]
    load_in_4bit = config.get("load_in_4bit", False)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config["max_length"],
        dtype=None,           # auto-detect: bf16 on Ampere+, fp16 otherwise
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    action_token_ids = _prepare_action_tokens_for_training(model, tokenizer, config)
    trainable_token_indices = (
        {ACTION_BIN_TRAINABLE_TOKEN_MODULE: action_token_ids}
        if uses_action_bins(config)
        else None
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM vs standard checkpointing
        random_state=42,
        trainable_token_indices=trainable_token_indices,
    )
    model.print_trainable_parameters()
    return model, tokenizer


def load_from_checkpoint(model_path: str, load_in_4bit: bool | None = None):
    """Load a model for evaluation using Unsloth.

    If model_path contains adapter_config.json, loads as a LoRA checkpoint
    (reads max_seq_length and optionally load_in_4bit from the saved config.yaml
    in the checkpoint directory). Otherwise loads as a plain base model with a
    default max_seq_length of 2048.
    """
    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    saved_config = {}

    if os.path.exists(adapter_cfg_path):
        saved_config_path = os.path.join(model_path, "config.yaml")
        if os.path.exists(saved_config_path):
            with open(saved_config_path) as f:
                saved_config = yaml.safe_load(f) or {}
            max_seq_length = saved_config.get("max_length", 2048)
        else:
            max_seq_length = 2048
    else:
        max_seq_length = 2048

    if load_in_4bit is None:
        load_in_4bit = saved_config.get("load_in_4bit", False)
    _validate_action_bin_checkpoint(model_path, saved_config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    if os.path.exists(adapter_cfg_path):
        _validate_action_tokens_for_checkpoint(model, tokenizer, saved_config)
    else:
        _prepare_action_tokens_for_training(model, tokenizer, saved_config)

    model.eval()
    FastLanguageModel.for_inference(model)
    return model, tokenizer
