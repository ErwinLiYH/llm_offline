# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

LLM Offline RL: behavior cloning on D4RL PointMaze environments using a fine-tuned LLM via LoRA. Per-family formatters convert observations into prompt render variables; the model predicts actions either as compact integer hundredths such as `35,-72` or as discrete action bins displayed as `<act_03><act_48>`. In the default bin path, those display tokens map to reused low-frequency tokenizer IDs rather than newly added special tokens.

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
- Training progress is written to `progress/<uuid>.txt`; `train.py` prints the path once, deletes it on successful completion, and leaves it behind on failure.

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
- `data/<env_family>/dataset.py`: load data, expand to one sample per selected prompt template per timestep, tokenize
- `data/<env_family>/formatting.py`: `format_obs`, `format_action`, `parse_action`, `validate_action`
- `model/policy.py`: load base model and LoRA adapters
- `utils/action_bins.py`: action-bin display/model token codec, token selection, parsing helpers, and Gaussian bin loss
- `utils/prompt_loader.py`: load shared prompt templates for an environment family
- `utils/variant_selection.py`: resolve `single | all | except` plus variant lists into concrete training/eval sets
- `prompts/<env_family>/<prompt_name>.txt`: shared prompt templates for that family; the filename stem is the prompt name

Data flow:
- dataset
- episode-level train/val split using randomized sampling: first draw up to `episode_keep_num` episodes as a sampled pool (if the dataset has fewer episodes, use all), then split inside that pool with `floor(pool_size * train_data_ratio)` train episodes and the remainder as val episodes
- per timestep: `format_obs(obs, meta)` + optional sampled history via `format_history(...)` + action formatting. Text mode uses `format_action`; bin modes compute bin indices and keep separate model text and display text through the action-bin codec.
- fill the prompt templates named by `prompt_templete_index`; prompt names are `.txt` filenames without the extension
- wrap the rendered prompt using the tokenizer's native `chat_template` as a user turn; training also appends the action as the assistant turn
- tokenize with prompt-turn tokens masked out (`labels = -100`)
- when `action_token_mode: gaussian_bin`, action token positions use a Gaussian soft-label CE over action-bin tokens while non-action assistant tokens such as the chat-template end token use ordinary CE
- dataset cache filenames include `newtok<0|1>` and the action-token mapping hash so tokenized samples from different internal action-token schemas are not reused
- one sample per selected prompt template per timestep

To add a new environment family:
- add `prompts/<family>/`
- add `data/<family>/variants.py`
- add `data/<family>/dataset.py`
- add `data/<family>/formatting.py`
- register it in `data/registry.py`

## Implementation Notes

- Formatting is per environment family. There is no shared global formatting helper.
- Shared prompt templates render the environment/task text only; final training/eval token sequences are built through the model tokenizer's native `chat_template`, not by plain-text concatenation.
- Qwen3.5 models loaded through Unsloth may return a `Qwen3VLProcessor` instead of a plain tokenizer. The outer processor does not expose tokenizer mutation methods such as `add_special_tokens`; unwrap `processor.tokenizer` first. In this repo, use `utils.action_bins.get_tokenizer_backend(...)` before adding action tokens, resizing embeddings, selecting reused action token ids, or looking up action token ids.
- `evaluate.py` uses `registry.get_formatter(env_family)` for text-mode `parse_action` and all-mode `validate_action`; bin-mode parsing is centralized in `utils.action_bins.ActionBinCodec` and uses generated token ids.
- On parse failure or invalid output, evaluation retries up to `parse_retry_limit`, then falls back to a zero vector and logs fallback metrics.
- `format_obs(obs, meta)` returns a dict of prompt render variables. It must contain `obs_text`, and may add family-specific fields.
- PointMaze also implements `format_history(history_entries, meta)`, which renders optional history prompt blocks from sampled past transitions.
- PointMaze text-mode actions are parsed from compact integer hundredths like `35,-72`, interpreted as action*100, validated in `[-1, 1]`, then clipped.
- `action_token_mode` supports `text`, `bin`, and `gaussian_bin`. In bin modes, `new_token: false` (default) reuses stable low-frequency tokenizer IDs from the end of the base vocabulary for model training/generation while logs and jsonl display `<act_00>` ... according to `action_num_bins`; `new_token: true` preserves the older path that registers `<act_XX>` as additional special tokens.
- With `new_token: false`, do not resize embeddings for action bins and do not automatically add `embed_tokens` / `lm_head` to LoRA target modules. With `new_token: true`, the tokenizer must register the special action tokens, resize model embeddings if needed, and train the new input/output rows.
- `gaussian_bin` stores per-token `action_bin_labels` in the dataset and trains action token positions with Gaussian soft labels controlled by `action_soft_label_sigma`; optional `action_soft_label_radius` restricts this CE to center +/- n bins so out-of-window action tokens receive no gradient. Chat-template stop tokens still train with ordinary CE.
- PointMaze `format_obs` also emits dynamic `map_sensing_en` / `map_sensing_zh`, which describe the current cell, goal cell, and four-neighbor `wall/free` status using 1-based row/column indexing from the top-left corner.
- PointMaze history entries contain the past step's start position plus executed action. Positions are shown as both grid coordinates and continuous `x/y`.
- If `history_num > 0`, training samples history from the same episode using indices `t-1`, `t-1-history_stride`, ... and renders entries in chronological order. The first step in each episode has no history block.
- Standalone eval and training-time epoch eval maintain an online history buffer of actually executed actions, including fallback zero actions on parse failure.
- Training uses the templates named by `prompt_templete_index` from shared family prompt files. Prompt names are file stems under `prompts/<env_family>/`, so `prompts/pointmaze/0.txt` is selected as `"0"`. Evaluation uses the first template in filename order unless evaluation code is explicitly changed.
- Training config uses `train_mode: single | all | except` plus list-valued `train_varients`.
  - `single`: `train_varients` must contain exactly one variant
  - `all`: if `train_varients` is non-empty, train exactly those variants; if empty/omitted, use every available variant
  - `except`: `train_varients` is the exclusion list
- Epoch eval selection is independent from training selection via optional `eval_mode` and `eval_variants`.
  - If `eval_mode` is omitted, epoch eval follows the resolved training selection.
  - `eval_variants` also uses list semantics; under `except` it is an exclusion list.
- Multi-variant training, including `all` and `except`, uses weighted sampling by variant sample count. Optional `balance_variant_episode_count: true` first equalizes the sampled episode pool size across selected variants to the smallest per-variant target.
- `config.yaml` controls the base model via `model_name`, whether Unsloth uses 4-bit loading via `load_in_4bit`, which prompt templates are used for dataset construction via `prompt_templete_index`, action encoding via `action_token_mode` / `action_num_bins` / `new_token` / `action_bin_min` / `action_bin_max` / `action_soft_label_sigma` / `action_soft_label_radius`, offline episode sampling via `episode_keep_num` / `balance_variant_episode_count` / `sampling_seed`, history prompt settings via `history_num` / `history_stride`, eval step logging via `record_step_logs`, eval video recording via `record_video` / `record_all` / `video_episode_index` / `video_fps` / `video_format` / `mujoco_gl`, and the eval result root via `result_root`.
- Checkpoints are stored under `checkpoints/<env_family>/<model_slug>/<selection_tag>/<experiment_id>/`.
  - `selection_tag` is the single variant name, `all`, `all-<selected variants joined by +>`, or `except-<excluded variants joined by +>`.
- Training-time eval results live under `<result_root>/<model_slug>/train=<env_family>-<selection_tag>/exp=<experiment_id>/epoch_<n>/eval=<env_family>-<variant>/result.json`.
- Standalone eval results live under `<result_root>/<model_slug>/train=<env_family>-<selection_tag>/exp=<experiment_id>/standalone_<eval_uuid>/eval=<env_family>-<variant>/result.json`.
- `eval.yaml` uses the same list-based variant selection semantics as training via `eval_mode` + `variants`; legacy `variant: <name|all>` is still accepted for compatibility.
- `eval.yaml` has its own `history_num` / `history_stride` for standalone eval; training-time epoch eval reuses the training config's history settings.
- Eval step logs are written by default under each `eval=<...>/episode_<n>/steps/`, using the same `Prompt:` / `Action:` text layout as `inspect_jsonl_record.py` plus executed-action metadata. In bin modes they display action bins as `<act_XX>` even when the model internally generated reused tokenizer IDs; in `gaussian_bin` mode they also include per-dimension action-bin probability distributions when `record_step_logs: true`.
- Eval videos are stored next to the step-log directory inside each `episode_<n>/`; `video_episode_index` accepts an int or list, and `record_all: true` records every episode. Default output format is `gif`, while `mp4` requires an ffmpeg backend. Headless MuJoCo recording should use `mujoco_gl: egl`.

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

This repo also includes a project-specific Codex skill for recording recent repo changes into project documentation:
- `skills/project-changelog/`

To install it into your own Codex setup:

```bash
mkdir -p ~/.codex/skills
cp -R skills/llm-offline-env-support ~/.codex/skills/
cp -R skills/project-changelog ~/.codex/skills/
```

After that, Codex can use these skills when working on environment-family changes or project documentation updates in this repo.
