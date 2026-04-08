# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

LLM Offline RL: behavior cloning on D4RL PointMaze environments using a fine-tuned LLM via LoRA. Observations and goals are serialized to text; the model predicts actions as plain-text floats.

Main design reference: `DESIGN.md` (Chinese).

Stack:
- PyTorch
- HuggingFace Transformers
- Unsloth
- PEFT (LoRA)
- Minari
- Gymnasium

## Commands

Training:
- `micromamba run -n llm_offline python train.py --config config.yaml`

Evaluation:
- `micromamba run -n llm_offline python evaluate.py --config eval.yaml`

Prefer `micromamba run -n llm_offline` for Python commands in this repo.

## Architecture

Key files:
- `train.py`: entry point; reads `config.yaml`
- `evaluate.py`: entry point; reads `eval.yaml`
- `data/registry.py`: routes `env_family` to dataset + formatter
- `data/base_dataset.py`: abstract dataset interface
- `data/<env_family>/variants.py`: variant metadata
- `data/<env_family>/dataset.py`: load data, expand to `prompt_template_count` samples per timestep, tokenize
- `data/<env_family>/formatting.py`: `format_obs`, `format_action`, `parse_action`, `validate_action`
- `model/policy.py`: load base model and LoRA adapters
- `utils/prompt_loader.py`: load prompt templates for a variant
- `prompts/<env_family>/<variant>.yaml`: prompt templates for that variant; current PointMaze files contain 5

Data flow:
- dataset
- episode-level 9:1 train/val split
- per timestep: `format_obs` + `format_action`
- fill the first `prompt_template_count` templates
- tokenize with prompt tokens masked out (`labels = -100`)
- `prompt_template_count` samples per timestep

To add a new environment family:
- add `prompts/<family>/`
- add `data/<family>/variants.py`
- add `data/<family>/dataset.py`
- add `data/<family>/formatting.py`
- register it in `data/registry.py`

## Implementation Notes

- Formatting is per environment family. There is no shared global formatting helper.
- `evaluate.py` uses `registry.get_formatter(env_family)` for `parse_action` and `validate_action`.
- On parse failure or invalid output, evaluation retries up to `parse_retry_limit`, then falls back to a zero vector and logs fallback metrics.
- PointMaze actions are parsed from `float, float`, validated in `[-1, 1]`, then clipped.
- Training uses the first `prompt_template_count` templates from each variant prompt file; evaluation always uses template 0. The current PointMaze prompt files contain 5 templates, but the loader uses however many are actually defined.
- Multi-variant joint training uses weighted sampling by variant sample count.
- `config.yaml` controls the base model via `model_name`, whether Unsloth uses 4-bit loading via `load_in_4bit`, and how many prompt templates are used for dataset construction via `prompt_template_count`.
- Checkpoints are stored under `checkpoints/<env_family>/<model_slug>/<train_mode>/<variant>/`.
- Results mirror checkpoint structure under `results/`.

## Out Of Scope

Do not implement:
- Return-conditioning
- Online RL components
- Multi-GPU distributed training

## Migrated Project Memory

Migrated Claude project memories are stored at:
- `~/.codex/memories/llm_offline/MEMORY.md`
- `~/.codex/memories/llm_offline/project_env.md`
- `~/.codex/memories/llm_offline/user_profile.md`

Useful environment note:
- This project uses a micromamba environment named `llm_offline`.
