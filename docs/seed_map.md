# Seed-map 程序化地图与聚合数据

`seed_map` 是 PointMaze/AntMaze 的可复现程序化地图模式。内部实现使用随机生成的生成树，但公开配置、数据格式和日志只使用 `seed_map` 术语。

## 地图与种子语义

- `map_seed` 与版本化的 size 配置共同唯一确定地图。
- seed range 都是半开区间 `[start, end)`；`0-1000` 表示 1000 张地图。
- `random` size 模式默认从最终地图边长 `5, 7, ..., 15` 中按 map seed 选择。这里的尺寸包含外围墙。
- `fixed` 模式接受奇数的最终 `rows/cols`，因此也支持矩形地图。
- PointMaze 与 AntMaze 使用同一张 0/1 地图；AntMaze 的 cell scaling 仍为 4。

生成数据时，`--hard-sample` 与原脚本语义不变：

- 未设置：环境 reset 随机选择起点和终点。
- 设置：从当前 seed map 的可达起终点 pair 中按原 hard-sample 逻辑采样。

## 生成聚合数据

数据生成使用包含 `minari[create]` 的 `d4rl_datagen` 环境；尚未创建时先运行
`micromamba env create -f dataGen_env.yaml`。

PointMaze 示例：

```bash
micromamba run -n d4rl_datagen python local_pointmaze_gen.py \
  --use-seed-map \
  --seed-map-start 0 \
  --seed-map-end 1000 \
  --seed-map-trajectories-per-seed 10 \
  --seed-map-size-mode random \
  --seed-map-min-size 5 \
  --seed-map-max-size 15 \
  --num-workers 16 \
  --seed 42
```

AntMaze 使用相同的 seed-map 参数，并继续接受原来的 `--policy-file`、`--mode`、`--diverse-cell-mode`、`--action-noise` 和 hard-sample 参数：

```bash
micromamba run -n d4rl_datagen python local_antmaze_gen.py \
  --use-seed-map \
  --seed-map-start 0 \
  --seed-map-end 1000 \
  --seed-map-trajectories-per-seed 10 \
  --seed-map-size-mode random \
  --num-workers 16 \
  --seed 42
```

仓库提供一对 hard-sample Slurm 脚本：

```bash
sbatch sbatch/dataGen.point.hard.seedmap.slurm
sbatch sbatch/dataGen.ant.hard.seedmap.slurm
```

两者默认生成 `[0, 100)`、每个 map seed 50 条成功轨迹，并从最难的 200 个 reachable start/goal pair 中均匀随机采样（`HARD_SAMPLE_ALPHA=0.0`）。PointMaze/AntMaze 分别默认 `HARD_SAMPLE_MAX_PATH_LEN=40/25`：先在完整 reachable pair space 上计算并排序原 difficulty，跳过超过 family 限制的 pair，再继续向较低排名扫描直至补足 200 个 eligible pair；设为 `0` 可关闭该保险。`path_len` 是 `len(path)-1`，即网格移动次数。PointMaze 默认最终尺寸为 `9/11/13/15`，AntMaze 为 `9/11/13`。六个 DataGen Slurm 脚本都把 `DATASET_ROOT` 默认设为 HPC 环境变量 `${PROJECTDIR}/local_datasets`；脚本内部的 `PROJECT_ROOT` 仅用于仓库代码路径。seed-map resolver 会继续在该 base root 下加入 family/version/size/reward namespace，避免 PointMaze 和 AntMaze 的同名 `0-100-50` corpus 冲突。脚本默认续写已有 corpus；设置 `OVERWRITE=1` 才会重建。可通过 `SEED_MAP_START`、`SEED_MAP_END`、`TRAJECTORIES_PER_SEED`、`SEED_MAP_MIN_SIZE`、`SEED_MAP_MAX_SIZE`、`MAX_EPISODE_STEPS`、`HARD_SAMPLE_TOP_N`、`HARD_SAMPLE_ALPHA`、`HARD_SAMPLE_MAX_PATH_LEN`、`NUM_WORKERS`、`SEED`、`REWARD_TYPE`、`DATASET_ROOT` 和 `TEMPORARY_DATASET_ROOT` 等环境变量覆盖。

不通过 Slurm 时可运行同名 `.sh`：

```bash
bash sbatch/dataGen.point.hard.seedmap.sh
bash sbatch/dataGen.ant.hard.seedmap.sh
```

例如，生成本实验使用的 `[1, 501)`、每图 50 条轨迹 corpus：

```bash
SEED_MAP_START=1 SEED_MAP_END=501 TRAJECTORIES_PER_SEED=50 \
SEED_MAP_MIN_SIZE=9 SEED_MAP_MAX_SIZE=15 SEED=42 \
MAX_EPISODE_STEPS=1350 HARD_SAMPLE_TOP_N=0 HARD_SAMPLE_MAX_PATH_LEN=40 \
bash sbatch/dataGen.point.hard.seedmap.sh

SEED_MAP_START=1 SEED_MAP_END=501 TRAJECTORIES_PER_SEED=50 \
SEED_MAP_MIN_SIZE=9 SEED_MAP_MAX_SIZE=13 SEED=42 \
HARD_SAMPLE_TOP_N=200 HARD_SAMPLE_MAX_PATH_LEN=25 \
bash sbatch/dataGen.ant.hard.seedmap.sh
```

范围是半开区间；上例分别生成 seed 1–500。PointMaze 的
`HARD_SAMPLE_TOP_N=0` 表示保留 explicit reachable-pair reset 与成功过滤，但从
所有 `path_len <= HARD_SAMPLE_MAX_PATH_LEN` 的 reachable pair 均匀抽样，而不是
只使用 hardest top-N；同时把 `HARD_SAMPLE_MAX_PATH_LEN=0` 才会恢复完整 pair space。
该值属于 aggregate corpus 的 immutable `collection_config`；旧的 incomplete
PointMaze corpus 若没有此字段，必须使用新输出路径或 `OVERWRITE=1` 重建，不能把
两种 pair 分布续写进同一 corpus。

本地 shell 入口直接运行 `python local_*_gen.py`，要求用户提前激活好生成环境，不调用 Conda 或 Slurm。生成参数默认值与对应 `.slurm` 一致，唯一差异是 `DATASET_ROOT` 默认留空、不传 `--dataset-root`，由 Python 生成器使用自身的 derived default path；显式设置 `DATASET_ROOT=/path` 时仍会覆盖。

输出是一个逻辑数据集，而不是每张地图一个目录：

```text
.../0-1000-10/
  manifest.json
  index.jsonl
  data/main_data.hdf5
```

默认父目录还包含环境 family、seed-map 版本、size 配置和 reward type。可以用 `--dataset-root` 指定最终数据 base root；程序会在其下保留 `<family>-seed-map-<version>-<size>-<reward>/<start>-<end>-<trajectories>` 两级派生路径。也可以直接给出叶名称匹配的精确 `0-1000-10` 目录，或给出名称匹配的 family/spec namespace 目录。旧 `--seed-map-dataset-root` 作为兼容别名保留。`--temporary-dataset-root` 同时用于中间 Minari shard 和 map 级合并 workspace，省略时读取 `TMPDIR`，若环境变量也不存在则使用平台临时目录；短别名是 `--temp-dataset-root`。每张 map 的 shard 先在这个临时目录完成合并，再把一个完整的 map 数据批次追加进最终聚合 corpus；发布结束后删除该 map 的 shard 和 merge workspace。当前 seed-map collector 每张 map 生成一个 Minari shard，但 map 级合并接口允许以后把同一张 map 拆成多个 shard 而不改变最终发布语义。重复运行会按 map seed 恢复缺失数据；`--overwrite` 会重建精确目标数据集。

## 训练选择

在原训练配置上增加 `seed_map_train` 即可。原来的 `train_mode` 和 `train_varients` 可以保留；启用并正确解析 seed-map section 后，它们不会参与训练数据选择。

同一 section 也可直接用于 `baseline_train.py`（baseline 的旧字段名是
`train_variants`）。baseline 会把每张选中地图表示为独立的
`seed-map-<N>` synthetic variant，以保证 CRL/HIQL 的 goal relabeling 和
CRL batch negatives 不跨地图；d3rlpy baseline 复用同一 corpus loader。

```yaml
seed_map_train:
  enabled: true
  dataset_path: local_datasets/pointmaze-seed-map-v1-random-size5-15-sparse/0-1000-10

  # 可组合多个不重叠的半开区间。
  seed_ranges:
    - [0, 500]
    - [700, 900]

  # 从以上范围无放回、确定性地抽地图。
  seed_count: 100

  # 每张选中地图从 corpus 的 10 条中抽 4 条。
  trajectories_per_seed: 4
  selection_seed: 42

  # 默认 seed，保证 train/val 不共享地图。
  # trajectory 会按轨迹切分，允许同一地图进入两边。
  split_unit: seed
```

`train_data_ratio` 在 `split_unit: seed` 下按地图数切 train/val，在 `trajectory` 下按轨迹数切分。`episode_keep_num`、`episode_keep_per_varient` 和 `balance_variant_episode_count` 不再控制 seed-map 的 n/m 抽样；n/m 只由 section 明确决定。

tokenized cache 签名包含 corpus manifest/content hash、地图/轨迹选择 hash、每张地图的 prompt metadata 和原有 tokenizer/prompt/action/sensing 配置。

## 评估选择

`seed_map_eval` 独立于训练数据源，并优先于固定 `eval_mode` / `eval_variants`（standalone eval 中优先于 `variant` / `variants`）。

评估 corpus 内地图：

```yaml
seed_map_eval:
  enabled: true
  dataset_path: local_datasets/pointmaze-seed-map-v1-random-size5-15-sparse/0-1000-10
  seed_ranges:
    - [900, 1000]
  seed_count: 20
  episodes_per_seed: 5
  selection_seed: 7
```

评估未生成离线数据的新地图时不需要 `dataset_path`，但必须给出 range 和生成器配置：

```yaml
seed_map_eval:
  enabled: true
  seed_ranges:
    - [1000, 2000]
  seed_count: 100
  episodes_per_seed: 5
  selection_seed: 7
  reward_type: sparse

  seed_map_version: v1
  seed_map_size_mode: random
  seed_map_min_size: 5
  seed_map_max_size: 15
```

每个 map seed 作为一个 synthetic eval variant，可继续使用现有的多 GPU variant 分配和每个 variant 内的 rollout workers。

`baseline_train.py` 也使用相同的 synthetic variant 和 fresh-map 构造路径；
`episodes_per_seed` 会覆盖 baseline 的 `evaluation.num_episodes`。baseline
resolved config 会根据 train/eval generator spec 固化 `observation.map_shape`，
因此 AntMaze 13×13 seed map 与旧 12×16 registered-map slot 会统一成 13×16，
离线 dmap 与 rollout dmap 的维度保持一致。baseline 的逐 variant rollout
结果还会记录 map seed、generator spec、实际 topology hash、reward type 和
horizon，便于审计 fresh-map 构造是否与协议一致。

程序化地图没有注册的 canonical fixed pair，因此 PointMaze 和 AntMaze 的默认起终点模式都是 `random-start-goal`。也可显式配置原有 `hard-sample` 选项；`fix-start-goal` 会报错：

```yaml
eval_start_goal_mode: hard-sample
eval_hard_sample_top_n: 100
eval_hard_sample_alpha: 1.0
```

## 兼容性

- `seed_map_train` 不存在或 `enabled: false`：训练完全使用原 variant 解析与数据加载路径。
- `seed_map_eval` 不存在或 `enabled: false`：评估完全使用原 variant 路径。
- section 启用但字段错误、范围越界、m 超过 corpus 容量、manifest 不完整或 family 不匹配时直接报错，不会静默回退到 variants。
- 只启用 `seed_map_train` 而未启用 `seed_map_eval` 时，training-time eval 继续使用 legacy 固定 variant 选择；若希望在程序化地图上评估，应明确配置 `seed_map_eval`。
