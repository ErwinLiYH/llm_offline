# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Offline RL: behavior cloning on D4RL PointMaze environments using a fine-tuned LLM (via LoRA). Observations and goals are serialized to text; the model predicts actions as plain-text floats. The full design spec is in `DESIGN.md` (Chinese).

**Stack:** PyTorch + HuggingFace Transformers + Unsloth (training acceleration) + PEFT (LoRA), Minari (D4RL data), Gymnasium

## Commands

```bash
# Training
python train.py --config config.yaml

# Evaluation
python evaluate.py --config eval.yaml
```

`config.yaml` — training configuration:
```yaml
env_family: pointmaze
train_mode: single        # single | all
variant: open             # used only when train_mode=single
model_name: Qwen/Qwen3-0.6B  # any HuggingFace causal LM
load_in_4bit: false       # true = enable Unsloth 4-bit quantized loading
prompt_template_count: 1 # number of prompt templates used to build each training split

# Training hyperparameters
learning_rate: 1e-4
num_epochs: 3
batch_size: 32
max_length: 512

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: ["q_proj", "v_proj"]

# Evaluation during training
parse_retry_limit: 3
```

`eval.yaml` — evaluation configuration:
```yaml
model_path: checkpoints/pointmaze/Qwen3-0.6B/single/open/<experiment_id>/final
load_in_4bit: false      # optional override; when omitted, checkpoint eval uses saved training config
env_family: pointmaze
variant: open             # variant name, or "all" to evaluate all variants
num_episodes: 20
parse_retry_limit: 3
```

## Architecture

```
train.py                             # Entry point; reads config.yaml
evaluate.py                          # Entry point; reads eval.yaml
data/registry.py                     # Routes env_family → dataset + formatter; exposes
                                     #   get_dataset(env_family) and get_formatter(env_family)
data/base_dataset.py                 # Abstract interface (load, format, tokenize)
data/<env_family>/variants.py        # Variant metadata dict for that family
data/<env_family>/dataset.py         # Load data, expand to N samples/timestep, tokenize
data/<env_family>/formatting.py      # format_obs, format_action, parse_action, validate_action
model/policy.py                      # Load base model (from config) + LoRA adapters
utils/prompt_loader.py               # Load shared prompt templates for an environment family
prompts/<env_family>/<idx>.txt  # shared prompt templates for that family; PointMaze currently defines 5
```

**Data flow:** dataset → episode-level train/val split using `train_data_ratio` (default 0.9, so train uses the first 90% of episodes and val uses the remaining 10%) → for each timestep: call `format_obs` + `format_action` from the family's `formatting.py`, fill the first `prompt_template_count` templates, tokenize with loss mask (labels=-100 on prompt, loss only on action target) → `prompt_template_count` samples per timestep.

**Extending to a new environment family:** add `prompts/<family>/`, add `data/<family>/` with `variants.py`, `dataset.py`, and `formatting.py`, register one line in `data/registry.py`. No changes to `train.py` or `evaluate.py`.

## Key Implementation Details

- **Per-family formatting:** each `data/<family>/formatting.py` implements `format_obs`, `format_action`, `parse_action`, `validate_action`. `dataset.py` and `evaluate.py` call the family's own formatter — no shared global formatting utility. Adding a new environment family requires implementing all four functions.
- **Action parsing:** `evaluate.py` calls `registry.get_formatter(env_family)` to obtain `parse_action`/`validate_action`. On failure or invalid output, retries up to `parse_retry_limit` times (from `eval.yaml`); falls back to zero vector and logs parse-failure and fallback counts as auxiliary metrics.
  - *PointMaze:* regex parse of `float, float`, validate each component in `[-1, 1]`, clip and return.
- **Prompt templates:** training uses the first `prompt_template_count` templates from shared family templates in `prompts/<env_family>/<idx>.txt`, and evaluation always uses template 0. PointMaze currently defines 5 shared templates (0–2 English, 3–4 Chinese), but the loader follows the actual number of `.txt` templates present.
- **Multi-variant joint training:** weighted sampling by variant sample count to prevent large variants from dominating
- **Base model and dataset config:** `model_name` in `config.yaml` specifies the HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`, `meta-llama/Llama-3.2-1B`). `load_in_4bit` controls whether Unsloth loads the base model in 4-bit mode for training or evaluation. `prompt_template_count` controls how many prompt templates are used when building each dataset split. Checkpoint paths embed the model name slug (e.g. `checkpoints/pointmaze/Qwen3-0.6B/single/open/<experiment_id>/final/`) so experiments with different base models don't overwrite each other.
- **Checkpoint layout:** `checkpoints/<env_family>/<model_slug>/<train_mode>/<variant>/<experiment_id>/ep{N}/` saved after each epoch, plus `final/` at training end. Each directory contains LoRA adapter weights, tokenizer, and `config.yaml` copy. `eval.yaml` defaults to pointing at `final/`.
- **Results layout:** `results/` mirrors `checkpoints/` structure; records episode return and success rate per variant

## Not in scope (do not implement)

- Return-conditioning
- Online RL components
- Multi-GPU distributed training
