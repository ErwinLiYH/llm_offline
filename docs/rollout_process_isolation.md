# Rollout Process Isolation

This document describes the current rollout path used by `evaluate.py`,
`score.py`, and training-time eval after the process-isolation refactor.

## Mental Model

Rollout now has two layers of parallelism:

1. The eval rank owns model loading, variant assignment, result writing, and
   policy inference.
2. The rollout supervisor owned by that rank starts isolated env worker
   subprocesses for episode execution.

Each worker subprocess owns exactly one environment at a time. The worker
resets/steps/renders the env, builds prompts, writes step logs and videos, and
sends action requests to the parent rank. The parent rank runs the model and
sends action responses back to the worker.

This means `rollout_worker_num` is per rank. With `torchrun --nproc_per_node=4`
and `rollout_worker_num: 4`, a standalone DDP eval can use up to 16 env worker
subprocesses if all four ranks have assigned variants.

## Configuration

Current rollout fields:

```yaml
rollout_worker_num: 1   # per eval rank; isolated env worker processes
rollout_worker_lifetime: slot
rollout_worker_retries: 1
rollout_worker_start_timeout_seconds: 120
rollout_action_timeout_seconds: 300
policy_batch_timeout_ms: 10
eval_distribute_variants: true
```

Field semantics:

- `rollout_worker_num`: number of isolated env worker subprocesses started by
  each eval/score rank for a variant. It must be at least 1. The supervisor caps
  it at `num_episodes` for the current variant.
- `rollout_worker_lifetime: slot`: start one long-lived worker per slot and run
  multiple episodes sequentially in that subprocess.
- `rollout_worker_lifetime: episode`: start a fresh worker subprocess per
  episode. This gives stronger isolation and higher startup overhead.
- `rollout_worker_retries`: retry count for an episode after worker failure.
  When retries are exhausted, the episode is recorded as failed instead of
  killing the parent process.
- `rollout_worker_start_timeout_seconds`: max time for a worker to report ready.
- `rollout_action_timeout_seconds`: max time a worker waits for an action
  response from the parent policy process.
- `policy_batch_timeout_ms`: short parent-side batching window for pending
  action requests. Continuous action modes can share one forward pass across
  requests collected during this window.
- `eval_distribute_variants`: DDP eval variant assignment switch. When true,
  variants are assigned round-robin across ranks. When false, rank 0 evaluates
  all variants.

`eval_parallel_episodes` is deprecated and no longer supported. If it appears
in a config, config parsing raises an error telling you to rename it to
`rollout_worker_num`.

## Standalone Evaluate

Command shape is unchanged:

```bash
micromamba run -n llm_offline python evaluate.py --config eval.yaml
micromamba run -n llm_offline torchrun --standalone --nproc_per_node=4 evaluate.py --config eval.yaml --parallel_backend ddp
```

In single-process eval there is one eval rank, so the maximum worker count is
`rollout_worker_num`.

In DDP eval, ranks still exist. They are responsible for model placement and
variant distribution, while rollout workers are child processes of each rank.
The worker subprocesses are not DDP ranks and do not load the model.

`evaluate.py` writes one `result.json` per variant under the existing
standalone result layout:

```text
<result_root>/<model_slug>/train=<env_family>-<selection_tag>/exp=<experiment_id>/standalone_<eval_uuid>/eval=<env_family>-<variant>/result.json
```

## Training-Time Eval

Training-time rollout is always isolated from the training process.

When epoch eval or step eval triggers:

1. `train.py` saves a checkpoint for the current model state.
2. Each training rank resolves the variants assigned to that rank.
3. The rank launches a single-process `evaluate.py` subprocess with
   `--parallel_backend single`, `eval_output_mode: training`, the saved
   checkpoint path, and only that rank's assigned variants.
4. The child `evaluate.py` loads the checkpoint, starts its own rollout worker
   subprocesses, writes result artifacts, and exits.
5. The training parent reads the result JSON files. If the child fails, training
   records warning/failure metrics and continues.

In DDP training, the child environment removes DDP variables such as `RANK`,
`WORLD_SIZE`, and `LOCAL_RANK`. `CUDA_VISIBLE_DEVICES` is narrowed to the
parent rank's local GPU so that child eval does not accidentally join the
training process group.

Per-attempt child files are saved next to the training eval result:

```text
.../epoch_<n>/isolated_eval/rank_<rank>/attempt_<n>.yaml
.../epoch_<n>/isolated_eval/rank_<rank>/attempt_<n>.stdout
.../epoch_<n>/isolated_eval/rank_<rank>/attempt_<n>.stderr
```

The same layout is used under `step<N>/` for step eval.

`training_eval_rollout_isolated` is retained only as a compatibility/deprecated
field. It is no longer a behavior switch.

## Score Mode

`score.py` uses the same worker supervisor and policy path as `evaluate.py`,
but it still has score-specific behavior:

- PointMaze only.
- Supports `mode: score | reference`.
- Uses official/local score env specs and reference scores.
- Produces normalized scores and a run-level `summary.json`.
- Does not use `eval_distribute_variants`; score mode runs the selected
  variants in the single score process.

## Logging

Worker subprocesses do not normally print per-step or per-episode progress to
the terminal. This avoids interleaved stdout from multiple workers.

Standalone `evaluate.py` prints parent-process messages:

- backend/output mode/model path
- resolved variants and rank assignments
- rollout workers per rank
- variant start messages
- variant summary and result path
- completed variant summary on the main rank

`score.py` similarly prints score mode, model path, variant start messages,
variant summary, result path, config path, and summary path.

During training-time eval, child `evaluate.py` stdout/stderr are redirected to
the isolated attempt files. The training command line shows only the parent
training process messages, isolated attempt start lines, and warnings if child
eval fails.

Detailed episode data is written to artifacts instead:

- `record_step_logs: true` writes `steps.txt` under each episode directory.
- Video files are encoded inside the worker subprocess and returned as paths.
- Worker failures are recorded in `result.json` under `worker_failures`.

## Result Fields

Evaluate and score result JSONs keep their previous main metrics and add rollout
diagnostics:

```json
{
  "rollout_isolation": "process",
  "rollout_worker_num": 4,
  "rollout_worker_lifetime": "slot",
  "rollout_workers_used": [12345, 12346],
  "worker_failures": [],
  "completed_episodes": 20,
  "failed_episodes": 0
}
```

`rollout_workers_used` contains worker PIDs observed by the supervisor. If a
worker crashes and is replaced, more PIDs can appear than
`rollout_worker_num`.

`worker_failures` contains structured failure records with variant, worker id,
PID, episode index, attempt, error, exit code, and traceback when available.

Each `episode_results` entry also records `worker_id`, `worker_pid`,
`worker_failed`, and `failure_error`.

## Migration Checklist

- Replace `eval_parallel_episodes` with `rollout_worker_num`.
- Treat `rollout_worker_num` as per-rank capacity, not global capacity.
- Remove `training_eval_rollout_isolated` from authored configs; training eval
  is isolated unconditionally.
- Keep `eval_distribute_variants: true` when using DDP eval and you want
  variants split across ranks.
- Reduce either `--nproc_per_node` or `rollout_worker_num` if the total number
  of env subprocesses is too high for the node.
- Use `record_step_logs` and result JSON diagnostics instead of expecting
  worker progress lines in stdout.

## Relevant Files

- `utils/rollout/protocol.py`: queue message payload dataclasses.
- `utils/rollout/worker_main.py`: child env worker process.
- `utils/rollout/supervisor.py`: parent-side worker pool and episode scheduler.
- `utils/rollout/policy.py`: parent-side model inference and action parsing.
- `utils/rollout/evaluate_runner.py`: evaluate result aggregation.
- `utils/rollout/score_runner.py`: score result aggregation.
- `utils/eval_parallel.py`: rollout config validation and DDP variant assignment.
- `evaluate.py`: standalone/training eval CLI entry.
- `score.py`: score/reference CLI entry.
- `train.py`: training-time isolated eval subprocess launcher.
