import os
import contextlib
import io

import yaml
from transformers import AutoTokenizer

with contextlib.redirect_stdout(io.StringIO()):
    from unsloth import FastLanguageModel

from model.continuous_action import (
    ensure_continuous_action_decoder,
    load_continuous_action_decoder,
    resolve_continuous_policy_type,
    resolve_gaussian_log_std_bounds,
    unpatch_continuous_action_forward,
)
from model.mtp_bin import (
    ensure_mtp_bin_decoder,
    load_mtp_bin_decoder,
    resolve_mtp_k,
    unpatch_mtp_bin_forward,
    uses_mtp_bin,
)
from utils.action_bins import (
    action_bins_use_new_tokens,
    get_action_bin_codec,
    get_tokenizer_backend,
    uses_action_bins,
    uses_continuous_actions,
)


ACTION_BIN_LORA_MODULES = ("embed_tokens", "lm_head")


def _chat_template_targets(tokenizer):
    """Return outer and inner tokenizer-like objects that may need chat_template."""
    targets = [tokenizer]
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and not any(inner is item for item in targets):
        targets.append(inner)
    try:
        backend = get_tokenizer_backend(tokenizer)
    except AttributeError:
        backend = None
    if backend is not None and not any(backend is item for item in targets):
        targets.append(backend)
    return targets


def _existing_chat_template(tokenizer) -> str | None:
    for target in _chat_template_targets(tokenizer):
        template = getattr(target, "chat_template", None)
        if template:
            return template
    return None


def _restore_chat_template(tokenizer, *source_names: str | None):
    """Copy a missing chat_template from the original HF tokenizer when Unsloth drops it."""
    template = _existing_chat_template(tokenizer)
    source_name = None
    for candidate in source_names:
        if not candidate:
            continue
        source_name = candidate
        if template:
            break
        try:
            source_tokenizer = AutoTokenizer.from_pretrained(
                candidate,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            try:
                source_tokenizer = AutoTokenizer.from_pretrained(
                    candidate,
                    trust_remote_code=True,
                )
            except Exception:
                continue
        template = _existing_chat_template(source_tokenizer)
        if template:
            break
    if not template:
        return
    for target in _chat_template_targets(tokenizer):
        if not getattr(target, "chat_template", None):
            try:
                target.chat_template = template
            except Exception:
                pass
    if source_name and source_name != getattr(tokenizer, "name_or_path", None):
        print(f"[model] Restored tokenizer chat_template from {source_name}")


def get_model_slug(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe slug.

    e.g. 'Qwen/Qwen3-0.6B' -> 'Qwen3-0.6B'
    """
    return model_name.split("/")[-1]


def _embedding_vocab_size(embeddings) -> int | None:
    """Return an embedding/output-head row count, or None if it cannot be read."""
    if embeddings is None:
        return None
    weight = getattr(embeddings, "weight", None)
    if weight is None:
        return None
    return int(weight.shape[0])


def _ensure_pad_token(tokenizer):
    """Make causal-LM tokenizers usable for padded batches by reusing EOS as PAD."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _validate_model_vocab_size(model, tokenizer):
    """Ensure input/output token matrices cover the tokenizer vocabulary."""
    tok = get_tokenizer_backend(tokenizer)
    input_vocab_size = _embedding_vocab_size(model.get_input_embeddings())
    output_vocab_size = _embedding_vocab_size(model.get_output_embeddings())
    if input_vocab_size is not None and input_vocab_size < len(tok):
        raise ValueError(
            "Model input embedding does not cover tokenizer vocabulary: "
            f"tokenizer={len(tok)}, input_embeddings={input_vocab_size}."
        )
    if output_vocab_size is not None and output_vocab_size < len(tok):
        raise ValueError(
            "Model output head does not cover tokenizer vocabulary: "
            f"tokenizer={len(tok)}, output_embeddings={output_vocab_size}."
        )


def _prepare_action_tokens(model, tokenizer, config: dict, *, resize_embeddings: bool):
    """Register action-bin tokens and ensure the model token matrices fit them."""
    _ensure_pad_token(tokenizer)
    if not uses_action_bins(config):
        return []
    use_new_tokens = action_bins_use_new_tokens(config)
    codec = get_action_bin_codec(tokenizer, config, ensure_registered=use_new_tokens)
    tok = get_tokenizer_backend(tokenizer)
    input_vocab_size = _embedding_vocab_size(model.get_input_embeddings())
    output_vocab_size = _embedding_vocab_size(model.get_output_embeddings())
    needs_resize = (
        use_new_tokens
        and (
            input_vocab_size is None
            or input_vocab_size < len(tok)
            or (output_vocab_size is not None and output_vocab_size < len(tok))
        )
    )
    if needs_resize:
        if not resize_embeddings:
            raise ValueError(
                "Tokenizer/model vocab size mismatch after loading action-bin checkpoint: "
                f"tokenizer={len(tok)}, input_embeddings={input_vocab_size}, "
                f"output_embeddings={output_vocab_size}. Re-train with the current action-token setup."
            )
        model.resize_token_embeddings(len(tok))
    _validate_model_vocab_size(model, tokenizer)
    return list(codec.model_token_ids)


def _resolve_lora_target_modules(config: dict) -> list[str]:
    """Return LoRA target modules, force-adding action-token input/output layers."""
    raw_modules = config["lora_target_modules"]
    if isinstance(raw_modules, str):
        raw_modules = raw_modules.strip()
        if not raw_modules:
            raise ValueError("lora_target_modules must contain non-empty strings")
        modules = [raw_modules]
    else:
        modules = list(raw_modules)
    if uses_action_bins(config) and action_bins_use_new_tokens(config):
        modules.extend(ACTION_BIN_LORA_MODULES)

    resolved = []
    seen = set()
    for module in modules:
        if not isinstance(module, str) or not module.strip():
            raise ValueError(f"lora_target_modules must contain non-empty strings, got {module!r}")
        module = module.strip()
        if module not in seen:
            resolved.append(module)
            seen.add(module)
    return resolved


def _resolve_lora_layers_to_transform(config: dict):
    """Return optional layer indices for PEFT LoRA injection."""
    layers = config.get("lora_layers_to_transform")
    if layers is None:
        return None
    if isinstance(layers, int):
        return layers
    if not isinstance(layers, (list, tuple)):
        raise ValueError(
            "lora_layers_to_transform must be an int, a list of ints, or null/omitted"
        )

    resolved = []
    for layer in layers:
        if not isinstance(layer, int):
            raise ValueError(
                f"lora_layers_to_transform must contain ints, got {layer!r}"
            )
        resolved.append(layer)
    if not resolved:
        raise ValueError("lora_layers_to_transform must not be an empty list")
    return resolved


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
    _restore_chat_template(tokenizer, model_name)
    _prepare_action_tokens(model, tokenizer, config, resize_embeddings=True)
    target_modules = _resolve_lora_target_modules(config)
    layers_to_transform = _resolve_lora_layers_to_transform(config)
    layers_pattern = config.get("lora_layers_pattern")
    print(f"[model] LoRA target modules: {target_modules}")
    if layers_to_transform is not None:
        print(f"[model] LoRA layers to transform: {layers_to_transform}")
        if layers_pattern:
            print(f"[model] LoRA layers pattern: {layers_pattern}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM vs standard checkpointing
        random_state=42,
    )
    ensure_continuous_action_decoder(model, config)
    ensure_mtp_bin_decoder(model, config)
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
    _restore_chat_template(tokenizer, model_path, saved_config.get("model_name"))
    if os.path.exists(adapter_cfg_path):
        _prepare_action_tokens(model, tokenizer, saved_config, resize_embeddings=False)
    else:
        _prepare_action_tokens(model, tokenizer, saved_config, resize_embeddings=True)
    if uses_continuous_actions(saved_config):
        if "action_dim" not in saved_config:
            raise ValueError(
                "Checkpoint config.yaml uses a continuous action mode but does not contain action_dim."
            )
        gaussian_log_std_min = None
        gaussian_log_std_max = None
        if saved_config.get("action_token_mode") in {"parallel_gaussian", "parallel_t"}:
            gaussian_log_std_min, gaussian_log_std_max = resolve_gaussian_log_std_bounds(saved_config)
        load_continuous_action_decoder(
            model,
            model_path,
            expected_action_dim=int(saved_config["action_dim"]),
            expected_action_query_len=saved_config.get("action_query_len"),
            expected_action_head_num_blocks=saved_config.get("action_head_num_blocks"),
            expected_action_head_dropout=saved_config.get("action_head_dropout"),
            expected_policy_type=resolve_continuous_policy_type(saved_config),
            expected_gaussian_log_std_min=gaussian_log_std_min,
            expected_gaussian_log_std_max=gaussian_log_std_max,
        )
    if uses_mtp_bin(saved_config):
        if "action_dim" not in saved_config:
            raise ValueError(
                "Checkpoint config.yaml uses mtp_bin but does not contain action_dim."
            )
        load_mtp_bin_decoder(
            model,
            model_path,
            expected_action_dim=int(saved_config["action_dim"]),
            expected_mtp_k=resolve_mtp_k(
                int(saved_config["action_dim"]),
                saved_config.get("mtp_k"),
            ),
        )

    model.eval()
    unpatch_continuous_action_forward(model)
    unpatch_mtp_bin_forward(model)
    FastLanguageModel.for_inference(model)
    ensure_continuous_action_decoder(model, saved_config)
    ensure_mtp_bin_decoder(model, saved_config)
    return model, tokenizer
