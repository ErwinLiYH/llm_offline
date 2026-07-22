# CrossMaze baselines

This directory isolates conventional offline-learning baselines from the LLM
training pipeline. It has two dependency-separated backends that share
CrossMaze dataset selection, numeric observations, rollout evaluation, and
run artifacts.

Implemented algorithms:

- `mlp_bc`: d3rlpy deterministic continuous BC with an MLP encoder.
- `td3_bc`: d3rlpy TD3+BC.
- `iql`: d3rlpy IQL.
- `crl`: JAX/Flax Contrastive RL with the DDPG+BC actor objective.
- `hiql`: JAX/Flax Hierarchical Implicit Q-Learning.

BC, TD3+BC, and IQL use the MIT-licensed
[d3rlpy project](https://github.com/takuseno/d3rlpy), pinned to 2.8.1. CRL and
HIQL adapt the state-based MIT reference implementations from OGBench 1.2.1;
the exact source commit, local adaptations, and upstream license are recorded
in [`gcrl/UPSTREAM.md`](gcrl/UPSTREAM.md). Resolved algorithm settings and
runtime package versions are recorded in every run.

## Environment

Create or update the d3rlpy environment for BC/IQL/TD3+BC:

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

PointMaze base observations are `[observation, desired_goal]` (6 values).
AntMaze base observations are `[achieved_goal, observation, desired_goal]` (31
values) for the d3rlpy algorithms. CRL/HIQL keep state and goal separate:
PointMaze state is the 4D `observation`, AntMaze state is the 29D
`[achieved_goal, observation]`, and both use a 2D xy goal. Optional numeric
map, location-sensing, and wall-sensing components
can be concatenated through independent `observation` switches:

```yaml
observation:
  include_map: true
  include_location_sensing: true
  include_wall_sensing: true
  wall_sensing_version: v3
  map_sensing_boundary_risk_threshold: 0.10
```

The defaults keep all three components disabled for backward compatibility.
Map matrices use `0=free`, `1=wall`, row-major flattening, and `-1` padding to
the family-wide maximum shape (PointMaze `15x15`, AntMaze `12x16`). Location
sensing is the 0-based numeric vector
`[position_row, position_col, goal_row, goal_col]`; wall sensing is
`[up, down, left, right]` with `0=free`, `1=wall`, and `2=risk`. With all three
enabled, the final dimensions are 239 for PointMaze and 231 for AntMaze. The
offline adapter recomputes sensing from each variant's recorded coordinates
and map, while rollout uses the live CrossMaze layout. The complete vector is
then handled by the same training-fitted `StandardObservationScaler` as the
legacy observation. CRL/HIQL instead fit separate state and goal normalizers;
map/current-cell/wall features belong to state, while goal-cell features belong
to goal. No prompt or sensing text enters these baselines.

CRL/HIQL load complete episodes because their goals are relabeled from future
or random achieved states. Relabeling never crosses a maze variant. Each
minibatch is also variant-homogeneous, so CRL's in-batch negatives cannot mix
incompatible maps. HIQL uses separate compact goal targets and full state
targets internally, which permits the state and goal dimensions above to
differ.

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

JAX runs use the same directory layout but write `training.jsonl`,
`normalizer.json`, `.msgpack` checkpoints, and final `model.msgpack`. Rollout
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
