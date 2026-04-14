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
- `utils/prompt_loader.py`: load shared prompt templates for an environment family
- `utils/variant_selection.py`: resolve `single | all | except` plus variant lists into concrete training/eval sets
- `prompts/<env_family>/<idx>.txt`: shared prompt templates for that family; PointMaze currently defines 5

Data flow:
- dataset
- episode-level train/val split using `train_data_ratio` (default 0.9, so train uses the first 90% of episodes and val uses the remaining 10%)
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
- PointMaze actions are parsed from compact integer hundredths like `35,-72`, interpreted as action*100, validated in `[-1, 1]`, then clipped.
- Training uses the first `prompt_template_count` templates from shared family prompt files; evaluation always uses template 0. PointMaze currently defines 5 templates, but the loader uses however many indexed `.txt` templates are actually present.
- Training config uses `train_mode: single | all | except` plus list-valued `variants`.
  - `single`: `variants` must contain exactly one variant
  - `all`: `variants` should be empty/omitted
  - `except`: `variants` is the exclusion list
- Epoch eval selection is independent from training selection via optional `eval_mode` and `eval_variants`.
  - If `eval_mode` is omitted, epoch eval follows the resolved training selection.
  - `eval_variants` also uses list semantics; under `except` it is an exclusion list.
- Multi-variant training, including `all` and `except`, uses weighted sampling by variant sample count.
- `config.yaml` controls the base model via `model_name`, whether Unsloth uses 4-bit loading via `load_in_4bit`, and how many prompt templates are used for dataset construction via `prompt_template_count`.
- Checkpoints are stored under `checkpoints/<env_family>/<model_slug>/<selection_tag>/<experiment_id>/`.
  - `selection_tag` is the single variant name, `all`, or `except-<excluded variants joined by +>`.
- Results mirror checkpoint structure under `results/`.
- `eval.yaml` uses the same list-based variant selection semantics as training via `eval_mode` + `variants`; legacy `variant: <name|all>` is still accepted for compatibility.
- `eval.yaml` can record one rollout per variant via `record_video`; default output format is `gif`, while `mp4` requires an ffmpeg backend. Headless MuJoCo recording should use `mujoco_gl: egl`.

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

## Codex Skill

This repo includes a project-specific Codex skill for adding new environment families and variants:
- `skills/llm-offline-env-support/`

To install it into your own Codex setup:

```bash
mkdir -p ~/.codex/skills
cp -R skills/llm-offline-env-support ~/.codex/skills/
```

After that, Codex can use the skill when working on environment-family or variant support in this repo.
