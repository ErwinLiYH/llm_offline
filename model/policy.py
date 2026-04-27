import os

import yaml
from unsloth import FastLanguageModel

from utils.action_bins import get_tokenizer_backend, register_action_tokens, uses_action_bins


def get_model_slug(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe slug.

    e.g. 'Qwen/Qwen3-0.6B' -> 'Qwen3-0.6B'
    """
    return model_name.split("/")[-1]


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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if uses_action_bins(config):
        register_action_tokens(tokenizer, config)
        model.resize_token_embeddings(len(get_tokenizer_backend(tokenizer)))

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM vs standard checkpointing
        random_state=42,
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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    FastLanguageModel.for_inference(model)
    return model, tokenizer
