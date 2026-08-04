# CrossMaze baselines

This directory isolates conventional offline-learning baselines from the LLM
training pipeline. It has two dependency-separated backends that share
CrossMaze dataset selection, numeric observations, rollout evaluation, and
run artifacts.

Implemented algorithms:

- `mlp_bc`: d3rlpy deterministic continuous BC with an MLP encoder.
- `td3_bc`: d3rlpy TD3+BC.
- `iql`: d3rlpy IQL.
- `rebrac`: d3rlpy ReBRAC.
- `crl`: JAX/Flax Contrastive RL with the DDPG+BC actor objective.
- `hiql`: JAX/Flax Hierarchical Implicit Q-Learning.

BC, TD3+BC, IQL, and ReBRAC use the MIT-licensed
[d3rlpy project](https://github.com/takuseno/d3rlpy), pinned to 2.8.1. CRL and
HIQL adapt the state-based MIT reference implementations from OGBench 1.2.1;
the exact source commit, local adaptations, and upstream license are recorded
in [`gcrl/UPSTREAM.md`](gcrl/UPSTREAM.md). Resolved algorithm settings and
runtime package versions are recorded in every run.

## Environment

Create or update the d3rlpy environment for BC/IQL/TD3+BC/ReBRAC:

```bash
bash baselines/setup_env.sh
```

d3rlpy declares `gymnasium==1.0.0`, while this repository uses Gymnasium 1.2.3,
Gymnasium Robotics 1.4.2, and Minari 0.5.3. The setup script installs the tested
maze stack explicitly, then installs d3rlpy with `--no-deps`. The training entry
point verifies the core package versions before loading data.

CRL and HIQL use a separate JAX environment so the CUDA/JAX stack cannot alter
the pinned PyTorch baseline stack:

```bash
bash baselines/setup_gcrl_env.sh
```

This installs `jax[cuda12]==0.6.2`, Flax 0.10.7, and Optax 0.2.5 in
`llm_offline_gcrl`. The CUDA 12 wheel is intentional for machines whose NVIDIA
driver does not satisfy the newer CUDA 13 wheel requirement. The entry point
checks all core versions and the selected JAX device before training.

## Training

Configs use the same layered merge utility as the LLM entry points. Later files
override earlier files.

```bash
micromamba run -n llm_offline_baselines python baseline_train.py \
  --config baselines/configs/base.pointmaze.yaml baselines/configs/mlp_bc.yaml

micromamba run -n llm_offline_baselines python baseline_train.py \
  --config baselines/configs/base.pointmaze.yaml baselines/configs/td3_bc.yaml

micromamba run -n llm_offline_baselines python baseline_train.py \
  --config baselines/configs/base.antmaze.yaml baselines/configs/iql.yaml

micromamba run -n llm_offline_baselines python baseline_train.py \
  --config baselines/configs/base.antmaze.yaml baselines/configs/rebrac.yaml

micromamba run -n llm_offline_gcrl python baseline_train.py \
  --config baselines/configs/base.pointmaze.yaml baselines/configs/crl.yaml

micromamba run -n llm_offline_gcrl python baseline_train.py \
  --config baselines/configs/base.antmaze.yaml baselines/configs/hiql.yaml
```

`n_steps` is the number of minibatch parameter updates. `n_steps_per_epoch` only
groups updates for logging, checkpoints, and evaluation; it does not mean a
full pass through the offline dataset. The defaults perform 1,000,000 updates,
group 10,000 updates as one logical epoch, and run rollout evaluation every 10
epochs (100,000 updates), plus the final epoch.

All four d3rlpy algorithms support checkpoint-boundary continuation by layering
a `resume` block over a larger target `n_steps`. Both paths are required:

```yaml
resume:
  checkpoint: baseline_runs/<source>/checkpoints/step_1000000.d3
  trainer_state: baseline_runs/<source>/checkpoints/trainer_state_step_1000000.pt
```

The `.d3` file restores networks, target networks, optimizers, schedulers, and
fitted scalers. The trainer-state sidecar restores Python/NumPy/PyTorch RNG,
and the runner restores d3rlpy's absolute gradient-step counter so delayed
TD3+BC/ReBRAC actor updates retain their original schedule. The source and
target recipes must match apart from the larger total budget.

PointMaze base observations are `[observation, desired_goal]` (6 values).
AntMaze base observations are `[achieved_goal, observation, desired_goal]` (31
values) for the d3rlpy algorithms. CRL/HIQL use the versioned
`full_observation_v1` protocol: PointMaze state and goal are both based on the
4D `observation`, AntMaze state and goal are both based on the 29D
`[achieved_goal, observation]`, and every relabeled goal equals a complete
state row. Optional numeric static-map, dynamic-map, location-sensing, and
wall-sensing components
can be concatenated through independent `observation` switches:

```yaml
observation:
  include_map: true
  include_dynamic_map: true
  include_location_sensing: true
  include_wall_sensing: true
  wall_sensing_version: v3
  map_sensing_boundary_risk_threshold: 0.10
```

The defaults keep all four components disabled for backward compatibility.
Map matrices use `0=free`, `1=wall`, row-major flattening, and `-1` padding to
the family-wide maximum shape (PointMaze `15x15`, AntMaze `12x16`). Dynamic
maps use `0=open`, `1=wall`, `2=current`, `3=goal`, and `4=current+goal`; they
are independent of static maps and may be enabled together. Location
sensing is the 0-based numeric vector
`[position_row, position_col, goal_row, goal_col]`; wall sensing is
`[up, down, left, right]` with `0=free`, `1=wall`, and `2=risk`. With all four
enabled, the final d3rlpy dimensions are 464 for PointMaze and 423 for AntMaze;
the corresponding GCRL state and goal dimensions are 460 and 419. The
offline adapter recomputes sensing from each variant's recorded coordinates
and map, while rollout uses the live CrossMaze layout. The complete vector is
then handled by the same training-fitted `StandardObservationScaler` as the
legacy observation. CRL/HIQL use one observation normalizer for every
observation-like tensor. Their static map is stored once per variant, while a
dynamic map is stored as current/original-desired-goal cell indices and restored
only for sampled rows. A relabeled goal reuses the target state row byte for
byte, including its original C/G/S map. No prompt or sensing text enters these
baselines.

### Procedural seed-map data and rollout

Baselines accept the same aggregate seed-map corpus and selection sections as
the LLM pipeline. An enabled `seed_map_train` takes priority over
`train_mode`/`train_variants`; each selected map becomes a synthetic
`seed-map-<N>` variant so GCRL relabeling and CRL in-batch negatives remain
within one layout. The explicit map/trajectory selection is deterministic and
is written, together with the corpus manifest and content hash, to the run's
resolved config and `dataset_manifest.json`.

```yaml
seed_map_train:
  enabled: true
  dataset_path: local_datasets/pointmaze-seed-map-v1-random-size9-15-sparse/1-501-50
  seed_ranges: [[1, 101]]       # half-open: map seeds 1 through 100
  seed_count: 100
  trajectories_per_seed: 50
  selection_seed: 0
  split_unit: trajectory
```

`split_unit: seed` is the default and keeps whole maps out of either train or
validation. `split_unit: trajectory` splits the selected trajectories globally
and allows different trajectories from one map to occur on the two sides; it
does not promise a nonempty per-map validation subset. Legacy episode-keep and
variant-balancing controls do not override this explicit n-map/m-trajectory
selection.

An enabled `seed_map_eval` independently takes priority over
`eval_mode`/`eval_variants`. It can inherit the generator spec from a corpus or
generate fresh held-out maps directly from ranges; `episodes_per_seed` becomes
the per-map rollout count.

```yaml
seed_map_eval:
  enabled: true
  seed_ranges: [[1, 101], [1001, 1051]]
  seed_count: 150
  episodes_per_seed: 1
  selection_seed: 0
  reward_type: sparse
  seed_map_version: v1
  seed_map_size_mode: random
  seed_map_min_size: 9
  seed_map_max_size: 15
```

Procedural maps default to `random-start-goal`; existing hard-sample eval
settings are also supported, while `fix-start-goal` is rejected because a
generated map has no canonical pair. For baselines these settings live under
`evaluation.env_config`. Offline and online observations share one
resolved padded map slot. It remains `15x15` for PointMaze and expands to
`13x16` for an AntMaze random-size-9-to-13 selection, avoiding an offline/
rollout schema mismatch. Each rollout variant records `map_seed`, generator
spec, actual topology `map_hash`, reward type, and horizon as provenance. See
[`../docs/seed_map.md`](../docs/seed_map.md) for the corpus format, half-open
range rules, and data-generation commands.

### BC scaling encoder

`mlp_bc` also supports the paper-style scaling encoders used by Wang et al.,
*1000 Layer Networks for Self-Supervised RL*. They retain d3rlpy's
deterministic tanh action head and standard observation scaler, but replace its
vector encoder with an activated input projection followed by either a plain
or residual body. Each body unit is exactly `Dense -> LayerNorm -> Swish` with
LeCun-uniform weights and zero bias. A residual block contains four units and
adds its identity after the fourth Swish.

```yaml
network:
  architecture: residual_mlp  # or plain_mlp; legacy_mlp preserves old configs
  width: 256
  body_depth: 64              # residual_mlp must be divisible by 4
  block_size: 4
  activation: swish
  use_batch_norm: false
  use_layer_norm: true
  dropout_rate: null
```

`plain_mlp` and `residual_mlp` are intentionally BC-only. They require the
four settings shown above so a depth/width comparison cannot silently vary
normalization, activation, or dropout. Existing `hidden_units` configurations
continue to select `legacy_mlp`; if inherited by a layered scaling config,
they are ignored and written as `null` in the resolved config. Each BC run
also writes `model_metadata.json` with the instantiated trainable parameter
count, body depth, residual block count, and total Dense-layer count including
the action head.

CRL/HIQL load complete episodes because their goals are relabeled from future
or random achieved states. Relabeling never crosses a maze variant. Each
minibatch is also variant-homogeneous, so CRL's in-batch negatives cannot mix
incompatible maps. HIQL uses the single upstream-style `high_actor_targets`
full-state tensor, and all state/goal dimensions and schemas are identical.
Online GCRL evaluation forces `continuing_task=false` and `reset_target=false`.
At reset it performs a seeded preliminary reset, five seeded random actions,
captures a full observation after teleporting only qpos xy to the target, then
repeats the same reset for the real episode; the captured goal dynamic map has
`S` at the target.

Local variants honor top-level `reward_type: sparse | dense` and select the
corresponding reward-typed dataset directory. Remote Minari variants have fixed
reward types and reject incompatible overrides. Training over mixed reward
types is rejected unless `allow_mixed_reward_types: true` is explicit. BC does
not optimize rewards, while TD3+BC and IQL do. CRL and HIQL construct their
goal-conditioned learning signal from relabeled state-goal matches rather than
using the stored environment reward as their objective.

Each run is written under `baseline_runs/<experiment_id>/` with the resolved
config, dataset split manifest, native d3rlpy logs, periodic evaluation JSONL,
checkpoints, final `model.d3`, and `summary.json`.

JAX runs use the same directory layout but write `training.jsonl`, versioned
`normalizer.json`, `.msgpack` checkpoints plus required `.metadata.json`
sidecars, and final `model.msgpack`. Missing or legacy `compact_xy` metadata is
rejected rather than silently loaded. Rollout
evaluation runs at configured epoch boundaries and at the final epoch. The
existing standalone d3rlpy checkpoint-sweep scripts have not yet been extended
to load JAX checkpoints, and the JAX runner currently requires W&B logging to
remain disabled; these do not affect training-time/final rollout evaluation.

For a non-result-bearing end-to-end smoke check on the local PointMaze fixture:

```bash
micromamba run -n llm_offline_gcrl python baseline_train.py --config \
  baselines/configs/base.pointmaze.yaml baselines/configs/crl.yaml \
  baselines/tests/fixtures/smoke.gcrl.local.yaml --experiment_id smoke-crl

micromamba run -n llm_offline_gcrl python baseline_train.py --config \
  baselines/configs/base.pointmaze.yaml baselines/configs/hiql.yaml \
  baselines/tests/fixtures/smoke.gcrl.local.yaml --experiment_id smoke-hiql
```

Rollout output keeps aggregate and per-variant success/return/length metrics
plus one record per episode. Each episode record contains its reset seed,
actual sampled start/goal cells and continuous coordinates, sampling mode,
success, one-based first-success step (`null` on failure), return, length, and
final termination flags. PointMaze uses the CrossMaze default random
start/goal reset; AntMaze uses its registered fixed pair unless an explicit
supported eval mode is configured.
