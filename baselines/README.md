# CrossMaze baselines

This directory isolates conventional offline-learning baselines from the LLM
training pipeline. Algorithm implementations and minibatch training loops come
from `d3rlpy==2.8.1`; repository code only adapts CrossMaze datasets,
observations, evaluation, and artifacts.

Implemented algorithms:

- `mlp_bc`: d3rlpy deterministic continuous BC with an MLP encoder.
- `td3_bc`: d3rlpy TD3+BC.
- `iql`: d3rlpy IQL.

No algorithm implementation is vendored or reimplemented. The dependency is
the MIT-licensed [d3rlpy project](https://github.com/takuseno/d3rlpy), pinned to
2.8.1; the algorithm YAML files expose d3rlpy's corresponding configuration
fields and record their resolved values in every run.

## Environment

Create or update the independent environment:

```bash
bash baselines/setup_env.sh
```

d3rlpy declares `gymnasium==1.0.0`, while this repository uses Gymnasium 1.2.3,
Gymnasium Robotics 1.4.2, and Minari 0.5.3. The setup script installs the tested
maze stack explicitly, then installs d3rlpy with `--no-deps`. The training entry
point verifies the core package versions before loading data.

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
```

`n_steps` is the number of minibatch parameter updates. `n_steps_per_epoch` only
groups updates for logging, checkpoints, and evaluation; it does not mean a
full pass through the offline dataset. The defaults perform 1,000,000 updates,
group 10,000 updates as one logical epoch, and run rollout evaluation every 10
epochs (100,000 updates), plus the final epoch.

PointMaze base observations are `[observation, desired_goal]` (6 values).
AntMaze base observations are `[achieved_goal, observation, desired_goal]` (31
values). Optional numeric map, location-sensing, and wall-sensing components
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
legacy observation. No prompt or sensing text enters these MLP baselines.

Local variants honor top-level `reward_type: sparse | dense` and select the
corresponding reward-typed dataset directory. Remote Minari variants have fixed
reward types and reject incompatible overrides. Training over mixed reward
types is rejected unless `allow_mixed_reward_types: true` is explicit. BC does
not optimize rewards, while TD3+BC and IQL do.

Each run is written under `baseline_runs/<experiment_id>/` with the resolved
config, dataset split manifest, native d3rlpy logs, periodic evaluation JSONL,
checkpoints, final `model.d3`, and `summary.json`.

Rollout output keeps aggregate and per-variant success/return/length metrics
plus one record per episode. Each episode record contains its reset seed,
actual sampled start/goal cells and continuous coordinates, sampling mode,
success, one-based first-success step (`null` on failure), return, length, and
final termination flags. PointMaze uses the CrossMaze default random
start/goal reset; AntMaze uses its registered fixed pair unless an explicit
supported eval mode is configured.
