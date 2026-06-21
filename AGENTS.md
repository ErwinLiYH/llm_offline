# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

LLM Offline RL: behavior cloning on D4RL PointMaze and AntMaze environments using a fine-tuned LLM via LoRA. Per-family formatters convert observations into prompt render variables; the model predicts actions as compact integer hundredths such as `35,-72`, as discrete action bins displayed as `<act_03><act_48>`, via `mtp_bin` AQT-based multi-token action-bin prediction, via `simple_mtp_bin` one-forward query-per-action-dimension action-bin prediction, via `parallel_l1` continuous regression, via `parallel_gaussian` tanh-squashed Gaussian policy BC with state-independent log std, or via `parallel_t` Student-t policy BC with learned action queries. In the default bin path, display tokens map to reused low-frequency tokenizer IDs rather than newly added special tokens.

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
- Override run identity from the CLI, for example Slurm job IDs: `micromamba run -n llm_offline python train.py --config config.yaml --experiment_id <id>`
- Tokenize/cache only, then exit before optimizer/training setup: `micromamba run -n llm_offline python train.py --config config.yaml --tokenize-only`
- DDP single-node multi-GPU: `micromamba run -n llm_offline torchrun --standalone --nproc_per_node=<num_gpus> train.py --config config.yaml --parallel_backend ddp`
- `--experiment_id` overrides any `experiment_id` in the config before automatic ID generation, DDP broadcast, resource monitoring, and runtime config snapshot saving.
- Training progress is written to one run-scoped `progress/<experiment_id>.txt` file; `train.py` prints the file path once, updates it across epochs, prints the final progress line and deletes the file on successful training completion, and leaves it behind on failure. In partitioned training, shard-round loading is also reflected in this file as a `loading data shard round ...` status.
- Optional system resource monitoring is controlled by `resource_monitor_enabled` and writes the latest RAM/swap/GPU status to `sys_info/<experiment_id>.txt` every `resource_monitor_interval_seconds`; in DDP only rank0 samples the whole machine. The file is latest-only, not an append log.
- At startup, `train.py` saves the resolved runtime config to `exp_configs/<experiment_id>/config.yaml`, Git metadata to `git.yaml`, and the text dirty worktree patch to `dirty.patch` before model loading and dataset construction.

Evaluation:
- `micromamba run -n llm_offline python evaluate.py --config eval.yaml`
- DDP multi-GPU variant-parallel eval: `micromamba run -n llm_offline torchrun --standalone --nproc_per_node=<num_gpus> evaluate.py --config eval.yaml --parallel_backend ddp`

Official normalized scoring:
- `micromamba run -n llm_offline python score.py --config score.yaml`
- `score.py` reads all run settings from `score.yaml`; the only supported CLI override is `--config`.

Prefer `micromamba run -n llm_offline` for Python commands in this repo.

## Architecture

Key files:
- `train.py`: entry point; reads `config.yaml`
- `evaluate.py`: entry point; reads `eval.yaml`
- `score.py`: official-style PointMaze normalized score entry point; reads `score.yaml`
- `data/registry.py`: routes `env_family` to dataset, formatter, variants, and eval env specs
- `data/base_dataset.py`: abstract dataset interface
- `data/<env_family>/variants.py`: variant metadata
- `data/<env_family>/dataset.py`: load data, expand to one sample per selected prompt template per timestep, tokenize
- `data/<env_family>/formatting.py`: `format_obs`, `format_action`, `parse_action`, `validate_action`
- `model/policy.py`: load base model and LoRA adapters
- `model/continuous_action.py`: learned action-query decoder and deterministic/Gaussian/Student-t continuous action heads
- `utils/action_bins.py`: action-bin display/model token codec, token selection, parsing helpers, and Gaussian bin loss
- `utils/distributed.py`: single/DDP process context, rank0 helpers, barriers, loss reduction, DDP unwrap
- `utils/distributed_sampler.py`: DDP-compatible weighted sampler for multi-variant training
- `utils/eval_parallel.py`: eval episode-count validation and DDP variant-to-rank assignment
- `utils/eval_rollout.py`: shared prompt rendering, history sampling, model action generation, parse retry, fallback, and action-bin eval logging helpers
- `utils/experiment_config.py`: save per-experiment runtime config snapshots
- `utils/maze_sensing.py`: shared maze xy-to-cell conversion plus location and conservative four-neighbor wall sensing
- `utils/pointmaze_score.py`: PointMaze score env specs, official remote reference scores, local reference validation, env fingerprints, and normalized score helpers
- `utils/prompt_loader.py`: load shared prompt templates for an environment family
- `utils/variant_selection.py`: resolve `single | all | except` plus variant lists into concrete training/eval sets
- `prompts/<env_family>/<prompt_name>.txt`: shared prompt templates for that family; the filename stem is the prompt name
- `config.antmaze.yaml` / `eval.antmaze.yaml`: official D4RL AntMaze train/eval examples

Data flow:
- dataset
- episode-level train/val split using randomized sampling: first draw up to `episode_keep_num` episodes as a sampled pool (if the dataset has fewer episodes, use all), then split inside that pool with `floor(pool_size * train_data_ratio)` train episodes and the remainder as val episodes
- per timestep: `format_obs(obs, meta)` + optional sampled history via `format_history(...)` + action formatting. Text mode uses `format_action`; bin modes compute bin indices and keep separate model text and display text through the action-bin codec; `mtp_bin` and `simple_mtp_bin` store NTP/AQT prediction metadata instead of assistant action text; continuous modes store `action_values` and do not append assistant action text.
- fill the prompt templates named by `prompt_templete_index`; prompt names are `.txt` filenames without the extension
- wrap the rendered prompt using the tokenizer's native `chat_template` as a user turn; text/bin/gaussian_bin training also appends the action as the assistant turn, `mtp_bin` and `simple_mtp_bin` tokenize the generation prompt plus the action prefix and AQT metadata, while continuous modes tokenize only the generation prompt
- tokenize with prompt-turn tokens masked out (`labels = -100`); `mtp_bin` keeps all labels at `-100` and trains base CE, sampler CE, and LCM from NTP/AQT prediction positions to ABT ids; `simple_mtp_bin` also keeps labels at `-100`, trains NTP CE on the AR action-prefix path, trains sampler CE from one query per action dimension, and can add LCM to align query hidden states to matching NTP anchors; continuous modes keep all labels at `-100` and train from `action_values` with mean L1 loss (`parallel_l1`), tanh-squashed Gaussian NLL over inverse-tanh targets (`parallel_gaussian`), or Student-t NLL (`parallel_t`)
- when `action_token_mode: gaussian_bin`, action token positions use a Gaussian soft-label CE over action-bin tokens while non-action assistant tokens such as the chat-template end token use ordinary CE
- dataset cache filenames are compact sha256-prefix hashes over the tokenization signature for the current code version, including variant/data signature, tokenizer/max length, selected prompt names and template contents, prompt vars, history settings, `action_dim`, `newtok<0|1>`, and the action-token mapping hash so tokenized samples from different prompt schemas or internal action-token schemas are not reused. Source-file hashes are intentionally not included; delete old caches manually after code changes that affect tokenization semantics.
- one sample per selected prompt template per timestep

To add a new environment family:
- add `prompts/<family>/`
- add `data/<family>/variants.py`
- add `data/<family>/dataset.py`
- add `data/<family>/formatting.py`
- register it in `data/registry.py`

## Implementation Notes

- Observation/action formatting remains per environment family; common maze geometry and sensing live in `utils/maze_sensing.py`.
- AntMaze supports the six official Minari D4RL datasets: `umaze`, `umaze-diverse`, `medium-play`, `medium-diverse`, `large-play`, and `large-diverse`. Local/custom AntMaze maps are not implemented.
- AntMaze intentionally uses the Minari metadata's Gymnasium Robotics v4 env specs. The offline contract is a 27-dimensional proprioceptive `observation`, 2D `achieved_goal`, 2D `desired_goal`, and 8-dimensional torque action. Do not silently switch rollout to v5 defaults, which include contact-force observations and change the input shape.
- AntMaze keeps `0` for text actions. Its action-bin prompts are `bin_full_sensing`, `bin_loca_sensing`, `bin_wall_sensing`, and `bin_no_sensing`, shared by `bin` / `gaussian_bin` / `mtp_bin` / `simple_mtp_bin`; continuous prompts are `parallel_full_sensing`, `parallel_loca_sensing`, `parallel_wall_sensing`, and `parallel_no_sensing`, shared by `parallel_l1` / `parallel_gaussian` / `parallel_t`.
- AntMaze text actions use eight comma-separated integer hundredths in actuator order: back-right hip/ankle, front-left hip/ankle, front-right hip/ankle, back-left hip/ankle.
- AntMaze emits the same `location_sensing_*` / `wall_sensing_*` fields as PointMaze, using torso `achieved_goal` xy and `maze_size_scaling: 4.0`; history entries include torso xy, 1-based row/column, and the executed action.
- Official AntMaze evaluation maps can differ from the offline collection maps, including UMaze wall orientation. During rollout, `evaluate.py` lets the AntMaze formatter refresh prompt map, visual map, and scaling from the instantiated environment before rendering sensing fields.
- Shared prompt templates render the environment/task text only; final training/eval token sequences are built through the model tokenizer's native `chat_template`, not by plain-text concatenation.
- Qwen3.5 models loaded through Unsloth may return a `Qwen3VLProcessor` instead of a plain tokenizer. The outer processor does not expose tokenizer mutation methods such as `add_special_tokens`; unwrap `processor.tokenizer` first. In this repo, use `utils.action_bins.get_tokenizer_backend(...)` before adding action tokens, resizing embeddings, selecting reused action token ids, or looking up action token ids.
- `evaluate.py` uses `registry.get_formatter(env_family)` for text-mode `parse_action` and all-mode `validate_action`; bin-mode parsing is centralized in `utils.action_bins.ActionBinCodec` and uses generated/AQT-selected token ids; `mtp_bin`, `simple_mtp_bin`, and continuous modes skip `generate()` and run direct forward paths.
- `evaluate.py` and training-time eval remain fast rollout/success-rate style evaluation. Do not retrofit official normalized score into their result schema; use `score.py` instead.
- `score.py` is currently PointMaze-only and supports `mode: score | reference` in `score.yaml`. It intentionally takes run settings from YAML only; apart from `--config`, do not add CLI overrides unless the project policy changes.
- `score.py mode: score` writes one `result.json` per variant plus run-level `summary.json`; the merged runtime score config is saved as `score_config.yaml` in the run directory.
- `score.py mode: reference` is for local/custom PointMaze variants. It generates `local_references/pointmaze/<variant>.json` by default, using a seeded random policy for `ref_min_score` and Farama `WaypointController(..., maze_solver="QIteration")` without action noise for `ref_max_score`.
- `score.py mode: score` can record rollout videos with `record_video` / `record_all` / `video_episode_index` / `video_fps` / `video_format` / `mujoco_gl`; videos are saved under each `score=<...>/episode_<n>/` directory and result JSON records the video paths.
- Remote PointMaze scoring uses static Minari/D4RL reference scores from `utils.pointmaze_score.REMOTE_POINTMAZE_REFERENCE_SCORES`; scoring must not download Minari datasets just to read reference scores.
- Remote PointMaze score envs use Farama single-goal eval maps, force `continuing_task: true` and `reset_target: false`, and keep official horizons: open/umaze 300, medium 600, large 800. Dense variants reuse the matching map shape with dense env IDs.
- Local PointMaze score envs require explicit `local_eval_maps.<variant>.goal_cell` in `score.yaml`; the 0-based row/col cell must be free. The env fingerprint includes env ID, maze map, reward type, continuing/reset flags, horizon, and goal cell.
- Local `score` mode refuses to score if the reference JSON is missing or its `env_fingerprint` does not match the current score env spec.
- Local PointMaze offline data is generated by `local_varient_gen.py`. Its default behavior truncates each generated episode at first success; `--post-success-hold-steps > 0` records additional fixed-goal hold transitions using a PD hold controller. Enabling hold data for an existing local dataset should use `--overwrite` to avoid mixing old goal-arrival-only episodes with hold episodes.
- On parse failure or invalid output, evaluation retries up to `parse_retry_limit`, then falls back to a zero vector and logs fallback metrics.
- `format_obs(obs, meta)` returns a dict of prompt render variables. It must contain `obs_text`, and may add family-specific fields.
- PointMaze also implements `format_history(history_entries, meta)`, which renders optional history prompt blocks from sampled past transitions.
- PointMaze text-mode actions are parsed from compact integer hundredths like `35,-72`, interpreted as action*100, validated in `[-1, 1]`, then clipped.
- `action_token_mode` supports `text`, `bin`, `gaussian_bin`, `mtp_bin`, `simple_mtp_bin`, `parallel_l1`, `parallel_gaussian`, and `parallel_t`. In bin modes, `new_token: false` (default) reuses stable low-frequency tokenizer IDs from the end of the base vocabulary for model training/generation while logs and jsonl display `<act_00>` ... according to `action_num_bins`; `mtp_bin` uses trainable Action Query Token embeddings from `mtp_bin_decoder.pt`, plus a sampler head and LCM loss. `mtp_quadratic_decoding: false` disables eval-time verifier looping and trusts the first MTP NTP+AQT proposal directly. `simple_mtp_bin` uses the same decoder file but fixes one query per action dimension, trains NTP and query paths jointly, ignores `mtp_k` / `mtp_quadratic_decoding`, and eval executes the pure query output in one forward. In continuous modes, training stores prompt-only tokens plus `action_values` and appends learned action queries inside the model; `parallel_l1` regresses actions with L1 loss, `parallel_gaussian` outputs a latent Gaussian mean plus a state-independent `gaussian_log_std` parameter, squashes actions through `tanh`, and trains with tanh-squashed Gaussian NLL on `atanh(action)` targets; `parallel_t` outputs `mean/log_scale` and trains Student-t NLL controlled by `student_t_df`. `gaussian_log_std_init` initializes the learned Gaussian log std, while `gaussian_log_std_min/max` clamp the log std used for loss and rollout. `parallel_t` can add `continuous_mean_l1_weight * L1(mean, action)` as an auxiliary mean-fitting term. `action_head_dropout` applies only inside the continuous action MLP head and is disabled by `model.eval()` during rollout/eval. `action_head_weight_decay`, when configured, creates optimizer param groups so only continuous action MLP Linear weights receive AdamW weight decay; LLM/LoRA parameters, learned `action_queries`, Gaussian `gaussian_log_std`, bias, and LayerNorm parameters do not.
- With `new_token: false`, do not resize embeddings for action bins and do not automatically add `embed_tokens` / `lm_head` to LoRA target modules. With `new_token: true`, the tokenizer must register the special action tokens, resize model embeddings if needed, and train the new input/output rows. AQT embeddings are never tokenizer tokens; they are stored separately in `mtp_bin_decoder.pt`.
- `gaussian_bin` stores per-token `action_bin_labels` in the dataset and trains action token positions with Gaussian soft labels controlled by `action_soft_label_sigma`; optional `action_soft_label_radius` restricts this CE to center +/- n bins so out-of-window action tokens receive no gradient. Chat-template stop tokens still train with ordinary CE.
- Parallel/continuous action dimension is resolved by `registry.get_action_dim(env_family, variants)` after variant selection and saved into checkpoint `config.yaml`; PointMaze returns `2` and AntMaze returns `8`.
- PointMaze and AntMaze `format_obs` emit dynamic `location_sensing_en` / `location_sensing_zh` and `wall_sensing_en` / `wall_sensing_zh`. Location sensing describes the current cell and goal cell using 1-based row/column indexing from the top-left corner. Wall sensing describes four-neighbor `wall/free` status. Shared coordinate-to-cell conversion applies the floor/map-center formula with the family map scaling; if that raw cell is a wall, it snaps to the nearest free cell center so prompts do not report wall cells as positions. For a free movement neighbor, corner-risk sensing reports `wall` only when the position is within the side threshold, the current side cell is free, and the forward diagonal cell is a wall. This catches newly appearing entrance corners without treating a continuous corridor wall as blocking forward motion.
- PointMaze history entries contain the past step's start position plus executed action. Positions are shown as both grid coordinates and continuous `x/y`.
- If `history_num > 0`, training samples history from the same episode using indices `t-1`, `t-1-history_stride`, ... and renders entries in chronological order. The first step in each episode has no history block.
- Standalone eval and training-time eval maintain an online history buffer of actually executed actions, including fallback zero actions on parse failure.
- Rollout success is accumulated from `info["success"]` when available, because continuing maze tasks may reach the goal without setting `terminated=True`.
- Standalone eval and training-time eval reset each episode with deterministic seeds. Standalone `eval.yaml seed: S` and training `config.yaml eval_seed: S` make episode `i` use reset seed `S + i`; result JSON records `seed` and `episode_seeds`.
- `eval_parallel_episodes > 1` batches active episodes into one model forward for `parallel_l1`, `parallel_gaussian`, and `parallel_t`. Finished slots are immediately reused for later episode indices. Other action modes print a fallback notice and remain serial.
- When `eval_parallel_episodes > 1` is actually used by a continuous action mode, the batched rollout suppresses per-episode progress and video-path prints because episodes complete out of order. The caller still prints normal startup/result-path messages and a summary after each variant completes; non-continuous modes that fall back to serial rollout retain serial per-episode logging.
- With continuous action sampling enabled, batched eval is reproducible for a fixed seed, `eval_parallel_episodes`, world size, and variant assignment, but changing parallelism can change the random-number consumption order and trajectories.
- Training uses the templates named by `prompt_templete_index` from shared family prompt files. Prompt names are file stems under `prompts/<env_family>/`, so `prompts/pointmaze/0.txt` is selected as `"0"`. PointMaze and AntMaze action-bin prompt templates are named `bin_full_sensing`, `bin_loca_sensing`, `bin_wall_sensing`, and `bin_no_sensing`; they are shared by `bin`, `gaussian_bin`, `mtp_bin`, and `simple_mtp_bin`. Their continuous prompt templates are named `parallel_full_sensing`, `parallel_loca_sensing`, `parallel_wall_sensing`, and `parallel_no_sensing`; they are shared by `parallel_l1`, `parallel_gaussian`, and `parallel_t`. Training-time eval uses the first resolved training prompt. Standalone eval defaults to the first prompt recorded in the checkpoint config, and `eval.yaml` may override it with exactly one `prompt_templete_index`; if the override was not used for training, `evaluate.py` prints a strong warning and requires `Y` unless run with `-y/--yes`.
- Training config uses `train_mode: single | all | except` plus list-valued `train_varients`.
  - `single`: `train_varients` must contain exactly one variant
  - `all`: if `train_varients` is non-empty, train exactly those variants; if empty/omitted, use every available variant
  - `except`: `train_varients` is the exclusion list
- Training-time eval selection is independent from training selection via optional `eval_mode` and `eval_variants`.
  - If `eval_mode` is omitted, training-time eval follows the resolved training selection.
  - `eval_variants` also uses list semantics; under `except` it is an exclusion list.
- Multi-variant training, including `all` and `except`, uses weighted sampling by variant sample count. Optional `balance_variant_episode_count: true` first equalizes the sampled episode pool size across selected variants to the smallest per-variant target.
- `dataset_load_partitions > 1` enables low-memory tokenized-data training. `dataset_cache_dir` is required. Rank0 loads raw trajectories and plans train shards after the normal episode-level train/val split; train timesteps are deterministically split into shard segment plans, and segments may split an episode as `[start_t, end_t)` while still passing the full episode context to tokenization workers so history prompts remain correct. In DDP, `dataset_load_partitions` must be at least `world_size` and divisible by `world_size`; each round has `world_size` shards, rank `r` processes only its round-local shard, and the local DataLoader pads/replaces samples from that local shard to the round `target_batches`. The full val split is rank0-only and not partitioned. One epoch still means all train shard rounds are trained once, with round order shuffled per epoch.
- `train.py --tokenize-only` reuses the normal model/tokenizer and dataset construction path, prepares all selected train/val caches (including every configured train partition), prints sample/batch totals plus per-epoch and all-epoch train batch-step planning information for `eval_step_interval`, and exits before DDP wrapping, optimizer creation, W&B initialization, validation, rollout, or training. In DDP partitioned mode, rank0 plans shards and scatters only each rank's current-round payload so ranks can build shard caches in parallel; val cache remains rank0-only. The planning summary also reports approximate global samples per batch step as `batch_size * world_size` and reminds that `eval_step_interval` resets each epoch. It requires `dataset_cache_dir`. It still loads the model through the normal Unsloth path and may occupy a GPU; CPU-only tokenizer loading is not implemented.
- `config.yaml` controls the base model via `model_name`, whether Unsloth uses 4-bit loading via `load_in_4bit`, LoRA module and optional decoder-layer filtering via `lora_target_modules` / `lora_layers_to_transform`, parallel training via `parallel_backend` / `ddp_find_unused_parameters` / `distributed_timeout_seconds`, optional latest-only system monitoring via `resource_monitor_enabled` / `resource_monitor_interval_seconds`, which prompt templates are used for dataset construction via `prompt_templete_index`, tokenization worker count via `dataset_workers` (per rank), tokenized dataset sharding via `dataset_load_partitions`, train/val batch loading via `dataloader_config` (`num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, `non_blocking`), action encoding via `action_token_mode` / `action_num_bins` / `mtp_k` / `mtp_lcm_weight` / `mtp_quadratic_decoding` / `new_token` / `action_bin_min` / `action_bin_max` / `action_soft_label_sigma` / `action_soft_label_radius` / `gaussian_log_std_init` / `gaussian_log_std_min` / `gaussian_log_std_max` / `student_t_df` / `continuous_mean_l1_weight` / `action_head_dropout` / `action_head_weight_decay`, rollout parallelism via `eval_parallel_episodes` / `eval_distribute_variants`, isolated training-time rollout via `training_eval_rollout_isolated`, rollout action sampling via `action_sampling` / `action_temperature` / `action_top_p` / `action_top_k`, offline episode sampling via `episode_keep_num` / `balance_variant_episode_count` / `sampling_seed`, history prompt settings via `history_num` / `history_stride`, training-time eval cadence and fixed episode seeds via `eval_step_interval` / `step_eval_skip` / `eval_seed`, eval step logging via `record_step_logs`, eval video recording via `record_video` / `record_all` / `video_episode_index` / `video_fps` / `video_format` / `video_save_workers` / `video_save_max_pending` / `mujoco_gl`, and the eval result root via `result_root`. `simple_mtp_bin` uses `mtp_lcm_weight` but not `mtp_k` or `mtp_quadratic_decoding`. `action_dim` is normally auto-resolved for parallel/continuous/MTP modes and saved into checkpoints rather than manually authored in `config.yaml`.
- `dataloader_config.persistent_workers` and `prefetch_factor` require `num_workers > 0`. `non_blocking: true` is most useful together with `pin_memory: true`; DDP applies the same DataLoader settings independently on every rank.
- `parallel_backend: single` preserves the original single-GPU Unsloth path. `parallel_backend: ddp` must be launched with `torchrun`; `batch_size` is per-GPU micro-batch and global effective batch is `batch_size * gradient_accumulation_steps * world_size`.
- In DDP, checkpoint saving and validation remain rank0-only. With `eval_distribute_variants: true`, training-time rollout variants are assigned round-robin across ranks; the owning rank writes that variant's result, step logs, and videos, while rank0 gathers summaries and writes W&B metrics. With `training_eval_rollout_isolated: true`, each rank launches a single-process `evaluate.py` subprocess for its assigned variants using the just-saved checkpoint, with DDP environment variables removed and `CUDA_VISIBLE_DEVICES` narrowed to the parent rank's local GPU. If the configured isolated rollout fails and `eval_parallel_episodes > 1`, the parent retries once with `eval_parallel_episodes: 1`; if that serial fallback also fails, it warns and training continues. With `dataset_cache_dir`, rank0 builds tokenized caches before other ranks read them after a barrier.
- Training startup config snapshots are rank0-only and saved under `exp_configs/<experiment_id>/` after resolving experiment id, variants, `action_dim`, continuous action head settings, `world_size`, and `global_effective_batch_size`; the directory includes `config.yaml`, `git.yaml`, and `dirty.patch`. `git.yaml` records HEAD metadata, porcelain status, patch hash/size, and binary files skipped from the text patch. To restore the recorded source state, use `git checkout <head_commit>` and then `git apply exp_configs/<experiment_id>/dirty.patch`.
- When W&B is enabled, batch logs include `train/loss`, `train/learning_rate`, and mode-specific loss parts: `train/l1`; `train/nll` / `train/mae` / `train/std` for `parallel_gaussian` where `nll` is squashed-Gaussian NLL, `mae` is action-space mean-action MAE, and `std` is the learned state-independent policy std; `train/tnll` / `train/scale` / `train/mean_l1_aux` / `train/mean_l1_weight` / `train/df`; `train/base_loss` / `train/sampler_loss` / `train/lcm_loss` for `mtp_bin` and `simple_mtp_bin`; or `train/action_loss` / `train/stop_loss` for `gaussian_bin`. All action-bin modes (`bin`, `gaussian_bin`, `mtp_bin`, `simple_mtp_bin`) also log `train/bin_l1`, a greedy predicted-bin-center MAE in continuous action units; MTP modes additionally log `train/mtp_bin_l1` and `train/ntp_bin_l1` for query-path and NTP-path equivalent MAE. Continuous validation for `parallel_l1`, `parallel_gaussian`, and `parallel_t` logs `val/mae` when validation runs. DDP reduces train batch metrics across ranks before rank0 logs them.
- Checkpoints are stored under `checkpoints/<env_family>/<model_slug>/<selection_tag>/<experiment_id>/`.
  - `selection_tag` is the single variant name, `all`, `all-<selected variants joined by +>`, or `except-<excluded variants joined by +>`.
- Epoch checkpoints use `ep<N>`, step checkpoints use `step<N>`, and final checkpoints use `final`.
- `eval_step_interval` enables optional training-time step eval by epoch-local train batch step. The counter resets at the start of each epoch, so the first step eval trigger in every epoch is at epoch-local batch `eval_step_interval`; if a trigger falls inside a gradient accumulation window, saving/eval waits until that window's `optimizer.step()` completes and uses the actual completed global batch step in `step<N>`.
- `step_eval_skip` defaults to `1`; values greater than `1` count step eval triggers per epoch and only every nth trigger runs validation loss plus environment rollout. Other step triggers save checkpoint-only `step<N>` checkpoints.
- If `eval_step_interval: 0` in an interactive run, `train.py` prints train batches per epoch and total train batches after dataloader construction, then prompts for an optional interval; non-interactive runs keep it disabled.
- If step eval would run within `0.25 * eval_step_interval` train batches before or after an epoch eval, save a checkpoint-only `step<N>` and keep validation/rollout for the epoch eval.
- Training-time eval results live under `<result_root>/<model_slug>/train=<env_family>-<selection_tag>/exp=<experiment_id>/epoch_<n>/eval=<env_family>-<variant>/result.json` or `.../step<N>/eval=<env_family>-<variant>/result.json`. With isolated training eval enabled, the merged eval config is saved at the corresponding `epoch_<n>/eval_config.yaml` or `step<N>/eval_config.yaml`, and per-attempt child config/stdout/stderr files live under `isolated_eval/rank_<rank>/attempt_<n>.*`.
- Standalone eval results live under `<result_root>/<model_slug>/train=<env_family>-<selection_tag>/exp=<experiment_id>/standalone_<eval_uuid>/eval=<env_family>-<variant>/result.json`; the merged runtime eval config is also saved at `.../standalone_<eval_uuid>/eval_config.yaml`. `evaluate.py` defaults to `eval_output_mode: standalone`; `eval_output_mode: training` is for training-launched isolated eval subprocesses and requires `training_eval_context`.
- `eval.yaml` uses the same list-based variant selection semantics as training via `eval_mode` + `variants`; legacy `variant: <name|all>` is still accepted for compatibility.
- `eval.yaml` has its own `history_num` / `history_stride` for standalone eval; training-time eval reuses the training config's history settings.
- `score.yaml` uses the same list-based variant selection semantics via `eval_mode` + `variants`, plus `mode`, `model_path`, `num_episodes`, `num_reference_episodes`, `assume_yes`, local reference settings, and optional score video settings. `reference.yaml` is the dedicated local-reference generation example. `assume_yes: true` is the score-mode equivalent of standalone eval's `-y/--yes` prompt-warning confirmation.
- Eval step logs are written by default to one `eval=<...>/episode_<n>/steps.txt` file per episode, with separator lines and `Step <n>` headings, using the same `Prompt:` / `Action:` text layout as `inspect_jsonl_record.py` plus executed-action metadata. In bin modes they display action bins as `<act_XX>` even when the model internally generated reused tokenizer IDs; `bin`, `gaussian_bin`, `mtp_bin`, and `simple_mtp_bin` include per-dimension action-bin probability distributions when `record_step_logs: true`; continuous modes log the raw continuous action before clipping, `parallel_gaussian` also logs policy mean/std, and `parallel_t` logs policy mean/scale.
- Dataset cache `.jsonl` files are human-readable previews. Their `action` field displays target ABTs as `<act_XX>`; for `mtp_bin` and `simple_mtp_bin` they also include `action_query` with `<aqt_i>` display markers. The `.pkl` cache remains the source of truth for actual `input_ids` and AQT metadata.
- Eval videos are stored next to the step-log directory inside each `episode_<n>/`; `video_episode_index` accepts an int or list, and `record_all: true` records every episode. For AntMaze `evaluate.py` and training-time eval, `record_video: true` saves both the normal follow camera `rollout.<gif|mp4>` and a default global top-down `rollout_global.<gif|mp4>` without extra config; result JSON keeps `video_path` / `video_paths` for follow-camera files and adds `global_video_path` / `global_video_paths` / `all_video_paths`. Default output format is `gif`, while `mp4` requires an ffmpeg backend. Headless MuJoCo recording should use `mujoco_gl: egl`.
- Eval and score video encoding uses a bounded background thread pool controlled by `video_save_workers` and `video_save_max_pending`. `video_save_workers` is the number of concurrent encoding threads; `video_save_max_pending` counts running plus queued video tasks and must be at least the worker count. `video_save_workers: 0` restores synchronous saving. A busy worker pool alone does not block submission while pending capacity remains; once the pending limit is reached, the next submission waits for at least one task to finish. Each variant waits for all remaining videos before returning so encoding errors are not hidden.

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

This repo also includes a project-specific Codex skill for auditing Isambard storage, inode, Slurm, and project limits:
- `skills/isambard-quota-check/`

To install it into your own Codex setup:

```bash
mkdir -p ~/.codex/skills
cp -R skills/llm-offline-env-support ~/.codex/skills/
cp -R skills/project-changelog ~/.codex/skills/
cp -R skills/isambard-quota-check ~/.codex/skills/
```

After that, Codex can use these skills for environment-family changes, project documentation updates, and Isambard quota audits in this repo.
