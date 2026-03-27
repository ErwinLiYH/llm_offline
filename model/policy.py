import json
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_slug(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe slug.

    e.g. 'Qwen/Qwen3-0.6B' -> 'Qwen3-0.6B'
    """
    return model_name.split("/")[-1]


def load_model_and_tokenizer(config: dict):
    """Load base model + LoRA adapters and tokenizer from config."""
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def load_from_checkpoint(model_path: str):
    """Load a fine-tuned model from a saved checkpoint directory.

    The checkpoint directory must contain:
      - adapter_config.json  (PEFT adapter config with base_model_name_or_path)
      - adapter weights (adapter_model.safetensors or adapter_model.bin)
      - tokenizer files
    """
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model, tokenizer
