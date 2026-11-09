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
- `qrl`: JAX/Flax Quasimetric RL with IQE/MRN and DDPG+BC/AWR variants.
- `gcbc`: JAX/Flax goal-conditioned behavioral cloning.

BC, TD3+BC, IQL, and ReBRAC use the MIT-licensed
[d3rlpy project](https://github.com/takuseno/d3rlpy), pinned to 2.8.1. CRL,
HIQL, QRL, and GCBC adapt the state-based MIT reference implementations from OGBench 1.2.1;
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

CRL, HIQL, QRL, and GCBC use a separate JAX environment so the CUDA/JAX stack cannot alter
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

micromamba run -n llm_offline_gcrl python baseline_train.py \
  --config baselines/configs/base.pointmaze.yaml baselines/configs/qrl.yaml

micromamba run -n llm_offline_gcrl python baseline_train.py \
  --config baselines/configs/base.antmaze.yaml baselines/configs/gcbc.yaml
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
generate fresh held-out maps directly from ranges. The per-map rollout count
continues to come from the ordinary `evaluation.num_episodes` setting.

```yaml
evaluation:
  num_episodes: 1

seed_map_eval:
  enabled: true
  seed_ranges: [[1, 101], [1001, 1051]]
  seed_count: 150
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

### CNN / Transformer 地图编码器

BC、TD3+BC、IQL、ReBRAC 和 CRL 支持 `network.architecture: cnn_map | transformer_map`。
BC 的算法标识仍为 `mlp_bc`，区分模型时应同时记录 `network.architecture`。
其他 JAX 算法暂不支持空间编码器，配置校验会直接报错。

```bash
# PyTorch：算法配置也可以换成 td3_bc.yaml、iql.yaml 或 rebrac.yaml。
micromamba run -n llm_offline_baselines python baseline_train.py \
  --config baselines/configs/base.pointmaze.yaml baselines/configs/mlp_bc.yaml baselines/configs/encoder.cnn.yaml

# JAX CRL：CNN 和 Transformer 都可使用。
micromamba run -n llm_offline_gcrl python baseline_train.py \
  --config baselines/configs/base.antmaze.yaml baselines/configs/crl.yaml baselines/configs/encoder.transformer.yaml
```

两个 encoder 配置都是末尾加载的覆盖文件，仅开启动态地图，不主动开启静态地图或 sensing。
从已有实验配置继承的 observation 开关仍然有效，做架构对照时应核对最终配置的信息一致性。
至少开启 `observation.include_map` 或 `include_dynamic_map`，否则拒绝启动。

观测在存储和 rollout 接口中仍是展平向量，网络依据 observation schema 恢复地图并拆分其他特征。
静态地图转换为 wall 通道，动态地图转换为 wall/current/goal 通道，并额外提供一个 valid 通道。
动态地图的代码 `4` 同时激活 current 和 goal，padding `-1` 不作为可通行格子。
地图类别及 padding 保持原值，仅非地图数值分量使用训练集统计量标准化。
标准化规则和地图布局随模型或 run artifacts 保存，评估不重新拟合。

CNN 默认使用三个 `3×3`、stride=1 的卷积层，通道为 `[32, 64, 64]`。
每层卷积后屏蔽无效区域，然后按空间顺序展平并投影为地图特征，不使用全局平均池化。
Transformer 默认每格一个 token，使用固定二维 sin/cos 位置编码、padding attention mask 和可学习的 CLS 汇聚 token。
Transformer 使用 pre-LayerNorm、GELU FFN 和零 dropout，从头训练，不使用语言模型预训练。

```yaml
network:
  architecture: transformer_map  # 或 cnn_map
  spatial:
    cnn_channels: [32, 64, 64]
    map_dim: 256
    vector_hidden_units: [128, 128]
    fusion_hidden_units: [256]
    transformer_dim: 128
    transformer_layers: 4
    transformer_heads: 4
    transformer_ffn_dim: 512
```

地图分支默认输出 256 维，状态、目标和已启用 sensing 的数值分支默认使用 `[128, 128]` MLP。
两个分支拼接后经过 `[256]` 融合 MLP，再送入原算法的动作头或价值网络。
`transformer_dim` 必须同时被 4 和 head 数整除。
`network.activation` 控制 CNN、数值分支和融合 MLP，`use_layer_norm` 控制数值分支和融合 MLP 的 LayerNorm。
Transformer 内部的 LayerNorm 始终开启，不受该开关影响。
空间编码器不支持 BatchNorm 或配置 dropout，PyTorch 保留 actor/critic/value 的 LayerNorm 独立覆盖设置。
PyTorch 的空间编码器使用 `network.spatial` 指定 MLP 维度，不使用 `network.hidden_units`。
CRL 的 `network.hidden_units` 仍用于空间融合之后的 actor 和双线性 critic MLP。

TD3+BC、IQL 和 ReBRAC 的 critic 在观测融合后拼接 action，再经过非线性层，保留状态与动作的交互能力。
IQL 的 value 网络也使用空间编码器，各网络默认不共享编码器参数。
CRL 的 actor 分别编码 state 和 goal，critic 的 state-action 与 goal 分支分别编码地图，两个 critic 成员也拥有独立空间编码器。
CRL 保持原有对比损失、goal relabeling 和 `full_observation_v1` 语义，不重绘 relabeled goal 的动态地图。

PointMaze、AntMaze 和 seed-map 使用解析后的固定 padded `map_shape`，离线数据和在线 rollout 必须保持同一布局。
CNN 的展平投影依赖该形状，不能把已保存模型直接用于更大的输入槽位。
Transformer 也按 checkpoint 的布局校验输入，不承诺自动跨输入尺寸加载。
`model_metadata.json` 记录网络配置和参数量；PyTorch 的计数范围是 policy，JAX 的计数范围是所有网络参数，比较前应核对 scope。

```bash
micromamba run -n llm_offline_baselines python -m unittest baselines.tests.test_spatial_encoders
JAX_PLATFORMS=cpu micromamba run -n llm_offline_gcrl python -m unittest baselines.tests.test_gcrl_spatial
```

测试覆盖两类环境的更新与保存加载、PyTorch checkpoint 续训、地图通道和数值分支梯度，以及基于小型 seed-map corpus 的完整训练和真实环境短 rollout。
这些是 CPU 功能测试，不代表正式训练收敛或成功率结论。
CRL 会多次编码 state/goal，Transformer 的显存开销明显高于 MLP；正式实验应先用小 batch 检查显存，再确定并记录训练预算。

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

CRL/HIQL/QRL/GCBC load complete episodes because their goals are relabeled from future
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
types is rejected unless `allow_mixed_reward_types: true` is explicit. BC and GCBC do
not optimize stored rewards, while TD3+BC, IQL, and ReBRAC do. CRL, HIQL, and QRL construct their
goal-conditioned learning targets from relabeled state-goal pairs rather than
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

micromamba run -n llm_offline_gcrl python baseline_train.py --config \
  baselines/configs/base.pointmaze.yaml baselines/configs/qrl.yaml \
  baselines/tests/fixtures/smoke.gcrl.local.yaml --experiment_id smoke-qrl

micromamba run -n llm_offline_gcrl python baseline_train.py --config \
  baselines/configs/base.pointmaze.yaml baselines/configs/gcbc.yaml \
  baselines/tests/fixtures/smoke.gcrl.local.yaml --experiment_id smoke-gcbc
```

Rollout output keeps aggregate and per-variant success/return/length metrics
plus one record per episode. Each episode record contains its reset seed,
actual sampled start/goal cells and continuous coordinates, sampling mode,
success, one-based first-success step (`null` on failure), return, length, and
final termination flags. PointMaze uses the CrossMaze default random
start/goal reset; AntMaze uses its registered fixed pair unless an explicit
supported eval mode is configured.
