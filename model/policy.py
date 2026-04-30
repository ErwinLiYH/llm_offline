import os
import json
import contextlib
import io

import yaml

with contextlib.redirect_stdout(io.StringIO()):
    from unsloth import FastLanguageModel

from utils.action_bins import get_tokenizer_backend, register_action_tokens, uses_action_bins


ACTION_BIN_MODULES_TO_SAVE = ("embed_tokens", "lm_head")


def get_model_slug(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe slug.

    e.g. 'Qwen/Qwen3-0.6B' -> 'Qwen3-0.6B'
    """
    return model_name.split("/")[-1]


def _embedding_vocab_size(model) -> int | None:
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is None:
        return None
    return int(weight.shape[0])


def _ensure_tokenizer_and_embedding_size(model, tokenizer, config: dict):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not uses_action_bins(config):
        return
    register_action_tokens(tokenizer, config)
    tok = get_tokenizer_backend(tokenizer)
    vocab_size = _embedding_vocab_size(model)
    if vocab_size is None or vocab_size != len(tok):
        model.resize_token_embeddings(len(tok))


def _has_required_action_modules(adapter_config: dict) -> bool:
    modules_to_save = adapter_config.get("modules_to_save") or []
    return all(
        any(module_name == required or module_name.endswith(f".{required}") for module_name in modules_to_save)
        for required in ACTION_BIN_MODULES_TO_SAVE
    )


def _validate_action_bin_checkpoint(model_path: str, saved_config: dict):
    if not uses_action_bins(saved_config):
        return
    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg_path):
        return
    with open(adapter_cfg_path) as f:
        adapter_config = json.load(f)
    if _has_required_action_modules(adapter_config):
        return
    raise ValueError(
        "This checkpoint was trained with action_token_mode="
        f"{saved_config.get('action_token_mode')!r}, but adapter_config.json does not save "
        f"{list(ACTION_BIN_MODULES_TO_SAVE)}. Its action-token embeddings/lm_head cannot be restored, "
        "which makes action-bin probabilities uniform and parsing fail. Re-train from a checkpoint "
        "created after this fix, or evaluate a text-mode checkpoint."
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
    _ensure_tokenizer_and_embedding_size(model, tokenizer, config)

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM vs standard checkpointing
        random_state=42,
        modules_to_save=list(ACTION_BIN_MODULES_TO_SAVE) if uses_action_bins(config) else None,
        ensure_weight_tying=uses_action_bins(config),
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
    _ensure_tokenizer_and_embedding_size(model, tokenizer, saved_config)

    model.eval()
    FastLanguageModel.for_inference(model)
    return model, tokenizer
