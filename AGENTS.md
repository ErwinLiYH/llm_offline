# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

LLM Offline RL: behavior cloning on D4RL PointMaze environments using a fine-tuned LLM via LoRA. Per-family formatters convert observations into prompt render variables; the model predicts actions as compact integer hundredths such as `35,-72`, as discrete action bins displayed as `<act_03><act_48>`, via `parallel_llm_bin` PHT-token parallel action-bin classification, via `parallel_l1` continuous regression, via `parallel_gaussian` diagonal-Gaussian policy BC, or via `parallel_t` Student-t policy BC with learned action queries. In the default bin path, display tokens map to reused low-frequency tokenizer IDs rather than newly added special tokens.

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
- DDP single-node multi-GPU: `micromamba run -n llm_offline torchrun --standalone --nproc_per_node=<num_gpus> train.py --config config.yaml --parallel_backend ddp`
- Training progress is written to per-epoch `progress/<uuid>.txt` files; `train.py` prints each epoch's path, prints the final progress line and deletes that epoch file on successful epoch completion, and leaves it behind on failure.
- At startup, `train.py` saves the resolved runtime config to `exp_configs/<experiment_id>/config.yaml` before model loading and dataset construction.

Evaluation:
- `micromamba run -n llm_offline python evaluate.py --config eval.yaml`

Official normalized scoring:
- `micromamba run -n llm_offline python score.py --config score.yaml`
- `score.py` reads all run settings from `score.yaml`; the only supported CLI override is `--config`.

Prefer `micromamba run -n llm_offline` for Python commands in this repo.

## Architecture

Key files:
- `train.py`: entry point; reads `config.yaml`
- `evaluate.py`: entry point; reads `eval.yaml`
- `score.py`: official-style PointMaze normalized score entry point; reads `score.yaml`
- `data/registry.py`: routes `env_family` to dataset + formatter
- `data/base_dataset.py`: abstract dataset interface
- `data/<env_family>/variants.py`: variant metadata
- `data/<env_family>/dataset.py`: load data, expand to one sample per selected prompt template per timestep, tokenize
- `data/<env_family>/formatting.py`: `format_obs`, `format_action`, `parse_action`, `validate_action`
- `model/policy.py`: load base model and LoRA adapters
- `model/continuous_action.py`: learned action-query decoder and deterministic/Gaussian/Student-t continuous action heads
- `utils/action_bins.py`: action-bin display/model token codec, token selection, parsing helpers, and Gaussian bin loss
- `utils/distributed.py`: single/DDP process context, rank0 helpers, barriers, loss reduction, DDP unwrap
- `utils/distributed_sampler.py`: DDP-compatible weighted sampler for multi-variant training
- `utils/eval_rollout.py`: shared prompt rendering, history sampling, model action generation, parse retry, fallback, and action-bin eval logging helpers
- `utils/experiment_config.py`: save per-experiment runtime config snapshots
- `utils/pointmaze_score.py`: PointMaze score env specs, official remote reference scores, local reference validation, env fingerprints, and normalized score helpers
- `utils/prompt_loader.py`: load shared prompt templates for an environment family
- `utils/variant_selection.py`: resolve `single | all | except` plus variant lists into concrete training/eval sets
- `prompts/<env_family>/<prompt_name>.txt`: shared prompt templates for that family; the filename stem is the prompt name

Data flow:
- dataset
- episode-level train/val split using randomized sampling: first draw up to `episode_keep_num` episodes as a sampled pool (if the dataset has fewer episodes, use all), then split inside that pool with `floor(pool_size * train_data_ratio)` train episodes and the remainder as val episodes
- per timestep: `format_obs(obs, meta)` + optional sampled history via `format_history(...)` + action formatting. Text mode uses `format_action`; bin modes compute bin indices and keep separate model text and display text through the action-bin codec; `parallel_llm_bin` appends PHT tokens instead of assistant action text; continuous modes store `action_values` and do not append assistant action text.
- fill the prompt templates named by `prompt_templete_index`; prompt names are `.txt` filenames without the extension
- wrap the rendered prompt using the tokenizer's native `chat_template` as a user turn; text/bin/gaussian_bin training also appends the action as the assistant turn, `parallel_llm_bin` tokenizes the generation prompt and appends `action_dim` PHT tokens, while continuous modes tokenize only the generation prompt
- tokenize with prompt-turn tokens masked out (`labels = -100`); `parallel_llm_bin` keeps all labels at `-100` and trains hard CE from PHT-position logits to ABT ids; continuous modes keep all labels at `-100` and train from `action_values` with mean L1 loss (`parallel_l1`), diagonal Gaussian NLL (`parallel_gaussian`), or Student-t NLL (`parallel_t`)
- when `action_token_mode: gaussian_bin`, action token positions use a Gaussian soft-label CE over action-bin tokens while non-action assistant tokens such as the chat-template end token use ordinary CE
- dataset cache filenames are compact sha256-prefix hashes over the full tokenization signature, including variant/data signature, tokenizer/max length, selected prompt names and template contents, prompt vars, relevant source-file hashes, history settings, `action_dim`, `newtok<0|1>`, and the action-token mapping hash so tokenized samples from different prompt schemas or internal action-token schemas are not reused
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
- `evaluate.py` uses `registry.get_formatter(env_family)` for text-mode `parse_action` and all-mode `validate_action`; bin-mode parsing is centralized in `utils.action_bins.ActionBinCodec` and uses generated/PHT-selected token ids; `parallel_llm_bin` and continuous modes skip `generate()` and run direct forward paths.
- `evaluate.py` and training-time eval remain fast rollout/success-rate style evaluation. Do not retrofit official normalized score into their result schema; use `score.py` instead.
- `score.py` is currently PointMaze-only and supports `mode: score | reference` in `score.yaml`. It intentionally takes run settings from YAML only; apart from `--config`, do not add CLI overrides unless the project policy changes.
- `score.py mode: score` writes one `result.json` per variant plus run-level `summary.json`; the merged runtime score config is saved as `score_config.yaml` in the run directory.
- `score.py mode: reference` is for local/custom PointMaze variants. It generates `local_references/pointmaze/<variant>.json` by default, using a seeded random policy for `ref_min_score` and Farama `WaypointController(..., maze_solver="QIteration")` without action noise for `ref_max_score`.
- Remote PointMaze scoring uses static Minari/D4RL reference scores from `utils.pointmaze_score.REMOTE_POINTMAZE_REFERENCE_SCORES`; scoring must not download Minari datasets just to read reference scores.
- Remote PointMaze score envs use Farama single-goal eval maps, force `continuing_task: true` and `reset_target: false`, and keep official horizons: open/umaze 300, medium 600, large 800. Dense variants reuse the matching map shape with dense env IDs.
- Local PointMaze score envs require explicit `local_eval_maps.<variant>.goal_cell` in `score.yaml`; the 0-based row/col cell must be free. The env fingerprint includes env ID, maze map, reward type, continuing/reset flags, horizon, and goal cell.
- Local `score` mode refuses to score if the reference JSON is missing or its `env_fingerprint` does not match the current score env spec.
- On parse failure or invalid output, evaluation retries up to `parse_retry_limit`, then falls back to a zero vector and logs fallback metrics.
- `format_obs(obs, meta)` returns a dict of prompt render variables. It must contain `obs_text`, and may add family-specific fields.
- PointMaze also implements `format_history(history_entries, meta)`, which renders optional history prompt blocks from sampled past transitions.
- PointMaze text-mode actions are parsed from compact integer hundredths like `35,-72`, interpreted as action*100, validated in `[-1, 1]`, then clipped.
- `action_token_mode` supports `text`, `bin`, `gaussian_bin`, `parallel_llm_bin`, `parallel_l1`, `parallel_gaussian`, and `parallel_t`. In bin modes, `new_token: false` (default) reuses stable low-frequency tokenizer IDs from the end of the base vocabulary for model training/generation while logs and jsonl display `<act_00>` ... according to `action_num_bins`; `parallel_llm_bin` additionally reserves one distinct PHT token and predicts ABT ids from PHT-position `lm_head` logits. `new_token: true` registers `<act_XX>` and, for `parallel_llm_bin`, `<pht>` as additional special tokens. In continuous modes, training stores prompt-only tokens plus `action_values` and appends learned action queries inside the model; `parallel_l1` regresses actions with L1 loss, `parallel_gaussian` outputs `mean/log_std` and trains with diagonal Gaussian NLL, and `parallel_t` reuses the same `mean/log_scale` head with Student-t NLL controlled by `student_t_df`. `parallel_t` can add `continuous_mean_l1_weight * L1(mean, action)` as an auxiliary mean-fitting term.
- With `new_token: false`, do not resize embeddings for action bins/PHT and do not automatically add `embed_tokens` / `lm_head` to LoRA target modules. With `new_token: true`, the tokenizer must register the special action tokens plus `<pht>` for `parallel_llm_bin`, resize model embeddings if needed, and train the new input/output rows.
- `gaussian_bin` stores per-token `action_bin_labels` in the dataset and trains action token positions with Gaussian soft labels controlled by `action_soft_label_sigma`; optional `action_soft_label_radius` restricts this CE to center +/- n bins so out-of-window action tokens receive no gradient. Chat-template stop tokens still train with ordinary CE.
- Parallel/continuous action dimension is resolved by `registry.get_action_dim(env_family, variants)` after variant selection and saved into checkpoint `config.yaml`; PointMaze returns `2`, and future env families should expose their own flat action dimension.
- PointMaze `format_obs` also emits dynamic `location_sensing_en` / `location_sensing_zh` and `wall_sensing_en` / `wall_sensing_zh`. Location sensing describes the current cell and goal cell using 1-based row/column indexing from the top-left corner. Wall sensing describes four-neighbor `wall/free` status. Coordinate-to-cell conversion first applies the PointMaze floor/map-center formula; if that raw cell is a wall, it snaps to the nearest free cell center so prompts do not report wall cells as positions. Directional `wall/free` sensing is conservative near cell boundaries: if a free neighbor would be entered while close to a boundary whose diagonal cell is a wall, that direction is reported as `wall`.
- PointMaze history entries contain the past step's start position plus executed action. Positions are shown as both grid coordinates and continuous `x/y`.
- If `history_num > 0`, training samples history from the same episode using indices `t-1`, `t-1-history_stride`, ... and renders entries in chronological order. The first step in each episode has no history block.
- Standalone eval and training-time eval maintain an online history buffer of actually executed actions, including fallback zero actions on parse failure.
- Training uses the templates named by `prompt_templete_index` from shared family prompt files. Prompt names are file stems under `prompts/<env_family>/`, so `prompts/pointmaze/0.txt` is selected as `"0"`. PointMaze action-bin prompt templates are named `bin_full_sensing`, `bin_loca_sensing`, `bin_wall_sensing`, and `bin_no_sensing`; they are shared by `bin`, `gaussian_bin`, and `parallel_llm_bin`. PointMaze continuous prompt templates are named `parallel_full_sensing`, `parallel_loca_sensing`, `parallel_wall_sensing`, and `parallel_no_sensing`; they are shared by `parallel_l1`, `parallel_gaussian`, and `parallel_t`. Training-time eval uses the first resolved training prompt. Standalone eval defaults to the first prompt recorded in the checkpoint config, and `eval.yaml` may override it with exactly one `prompt_templete_index`; if the override was not used for training, `evaluate.py` prints a strong warning and requires `Y` unless run with `-y/--yes`.
- Training config uses `train_mode: single | all | except` plus list-valued `train_varients`.
  - `single`: `train_varients` must contain exactly one variant
  - `all`: if `train_varients` is non-empty, train exactly those variants; if empty/omitted, use every available variant
  - `except`: `train_varients` is the exclusion list
- Training-time eval selection is independent from training selection via optional `eval_mode` and `eval_variants`.
  - If `eval_mode` is omitted, training-time eval follows the resolved training selection.
  - `eval_variants` also uses list semantics; under `except` it is an exclusion list.
- Multi-variant training, including `all` and `except`, uses weighted sampling by variant sample count. Optional `balance_variant_episode_count: true` first equalizes the sampled episode pool size across selected variants to the smallest per-variant target.
- `config.yaml` controls the base model via `model_name`, whether Unsloth uses 4-bit loading via `load_in_4bit`, parallel training via `parallel_backend` / `ddp_find_unused_parameters` / `distributed_timeout_seconds`, which prompt templates are used for dataset construction via `prompt_templete_index`, action encoding via `action_token_mode` / `action_num_bins` / `new_token` / `action_bin_min` / `action_bin_max` / `action_soft_label_sigma` / `action_soft_label_radius` / `gaussian_log_std_min` / `gaussian_log_std_max` / `student_t_df` / `continuous_mean_l1_weight`, rollout action sampling via `action_sampling` / `action_temperature` / `action_top_p` / `action_top_k`, offline episode sampling via `episode_keep_num` / `balance_variant_episode_count` / `sampling_seed`, history prompt settings via `history_num` / `history_stride`, training-time eval cadence via `eval_step_interval`, eval step logging via `record_step_logs`, eval video recording via `record_video` / `record_all` / `video_episode_index` / `video_fps` / `video_format` / `mujoco_gl`, and the eval result root via `result_root`. `action_dim` is normally auto-resolved for parallel/continuous modes and saved into checkpoints rather than manually authored in `config.yaml`.
- `parallel_backend: single` preserves the original single-GPU Unsloth path. `parallel_backend: ddp` must be launched with `torchrun`; `batch_size` is per-GPU micro-batch and global effective batch is `batch_size * gradient_accumulation_steps * world_size`.
- In DDP, checkpoint saving, validation, training-time rollout eval, step logs, and videos are rank0-only. With `dataset_cache_dir`, rank0 builds tokenized caches before other ranks read them after a barrier.
- Training startup config snapshots are rank0-only and saved to `exp_configs/<experiment_id>/config.yaml` after resolving experiment id, variants, `action_dim`, continuous action head settings, `world_size`, and `global_effective_batch_size`; the snapshot also includes `train_config_source`.
- When W&B is enabled, batch logs include `train/loss`, `train/learning_rate`, and mode-specific loss parts: `train/l1`; `train/nll` / `train/mae` / `train/std`; `train/tnll` / `train/scale` / `train/mean_l1_aux` / `train/mean_l1_weight` / `train/df`; `train/action_loss` for `parallel_llm_bin`; or `train/action_loss` / `train/stop_loss` for `gaussian_bin`. DDP reduces these metrics across ranks before rank0 logs them.
- Checkpoints are stored under `checkpoints/<env_family>/<model_slug>/<selection_tag>/<experiment_id>/`.
  - `selection_tag` is the single variant name, `all`, `all-<selected variants joined by +>`, or `except-<excluded variants joined by +>`.
- Epoch checkpoints use `ep<N>`, step checkpoints use `step<N>`, and final checkpoints use `final`.
- `eval_step_interval` enables optional training-time step eval by global train batch step; if a trigger falls inside a gradient accumulation window, saving/eval waits until that window's `optimizer.step()` completes and uses the actual completed batch step in `step<N>`.
- If `eval_step_interval: 0` in an interactive run, `train.py` prints train batches per epoch and total train batches after dataloader construction, then prompts for an optional interval; non-interactive runs keep it disabled.
- If step eval and epoch eval land on the same epoch-end weight state, skip the duplicate step eval and keep the epoch checkpoint/eval.
- Training-time eval results live under `<result_root>/<model_slug>/train=<env_family>-<selection_tag>/exp=<experiment_id>/epoch_<n>/eval=<env_family>-<variant>/result.json` or `.../step<N>/eval=<env_family>-<variant>/result.json`.
- Standalone eval results live under `<result_root>/<model_slug>/train=<env_family>-<selection_tag>/exp=<experiment_id>/standalone_<eval_uuid>/eval=<env_family>-<variant>/result.json`; the merged runtime eval config is also saved at `.../standalone_<eval_uuid>/eval_config.yaml`.
- `eval.yaml` uses the same list-based variant selection semantics as training via `eval_mode` + `variants`; legacy `variant: <name|all>` is still accepted for compatibility.
- `eval.yaml` has its own `history_num` / `history_stride` for standalone eval; training-time eval reuses the training config's history settings.
- `score.yaml` uses the same list-based variant selection semantics via `eval_mode` + `variants`, plus `mode`, `model_path`, `num_episodes`, `num_reference_episodes`, `assume_yes`, and local reference settings. `assume_yes: true` is the score-mode equivalent of standalone eval's `-y/--yes` prompt-warning confirmation.
- Eval step logs are written by default under each `eval=<...>/episode_<n>/steps/`, using the same `Prompt:` / `Action:` text layout as `inspect_jsonl_record.py` plus executed-action metadata. In bin modes they display action bins as `<act_XX>` even when the model internally generated reused tokenizer IDs; `bin`, `gaussian_bin`, and `parallel_llm_bin` include per-dimension action-bin probability distributions when `record_step_logs: true`; continuous modes log the raw continuous action before clipping, `parallel_gaussian` also logs policy mean/std, and `parallel_t` logs policy mean/scale.
- Eval videos are stored next to the step-log directory inside each `episode_<n>/`; `video_episode_index` accepts an int or list, and `record_all: true` records every episode. Default output format is `gif`, while `mp4` requires an ffmpeg backend. Headless MuJoCo recording should use `mujoco_gl: egl`.

## Out Of Scope

Do not implement:
- Return-conditioning
- Online RL components
- Multi-node distributed training beyond the current single-node `torchrun`/DDP path

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
