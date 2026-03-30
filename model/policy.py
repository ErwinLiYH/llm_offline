import os

import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel


def get_model_slug(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe slug.

    e.g. 'Qwen/Qwen3-0.6B' -> 'Qwen3-0.6B'
    """
    return model_name.split("/")[-1]


def load_model_and_tokenizer(config: dict):
    """Load base model + LoRA adapters and tokenizer from config using Unsloth."""
    model_name = config["model_name"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config["max_length"],
        dtype=None,           # auto-detect: bf16 on Ampere+, fp16 otherwise
        load_in_4bit=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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


def load_from_checkpoint(model_path: str):
    """Load a model for evaluation.

    If model_path contains adapter_config.json, loads as a PEFT LoRA checkpoint
    (base model + adapters). Otherwise loads the path directly as a base model,
    allowing evaluation of unmodified HuggingFace models.
    """
    import json
    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer
