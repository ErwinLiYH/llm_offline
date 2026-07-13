# CrossMaze Path and Map Difficulty

本文档定义 `crossmaze.eval_position` 使用的路径难度、地图难度和
hard-sample 排序规则。这里的 difficulty 是普通 PointMaze/AntMaze rollout
eval 使用的几何启发式指标，不是 D4RL official normalized score，也不等价于
某个策略的真实成功率。

当前规则版本为 `v2`。版本、常量和实现的唯一代码源位于
`crossmaze/eval_position.py`。它只用于普通 eval 的 start-goal 描述、地图
难度和 eval hard-sample。

本指标与 topology `static_difficulty` 是并列但独立的两类指标：

- `utils/maze_metrics.py` 的 topology `static_difficulty` 用于地图生成、筛选和
  静态拓扑检查，其公式和文档不受 eval difficulty v2 影响；
- 本文档的 path/map difficulty 用于评估。

这里的 hard-sample 仅指 `eval_start_goal_mode: hard-sample`。本地 AntMaze
数据生成器的 `--hard-sample` 仍使用原有 legacy pair difficulty，不调用本文
的 v2 公式；它和 topology 地图生成逻辑也都保持不变。

## 1. 图和路径定义

maze map 中值不为 `1` 的 cell 是 free cell。每个 free cell 是一个图节点，
上下左右相邻的 free cells 之间存在无向边，每条边长度为一个 grid step。

对候选集合中每个有序 start/goal pair `(s, g)`：

- `d = dist(s, g)`：图上的最短路径长度；
- `m = |s.row-g.row| + |s.col-g.col|`：Manhattan distance；
- `P = (p_0=s, ..., p_d=g)`：当前 BFS 邻居顺序选出的 canonical shortest
  path；
- `D`：候选 pair space 中最大的 `d`。普通注册地图 eval 使用全部 free
  cells，因此此时 `D` 就是地图 graph diameter。

实现会以每个候选 cell 为根各执行一次 BFS，并保留 distance 和 canonical
parent。hard-sample 不会为不同 pair 重复执行新的 BFS。

## 2. 路径难度 v2

路径难度由绝对长度、干扰岔路和绕路三项加权求和：

```text
path_difficulty =
    0.4 * length_score
    + 0.3 * branch_score
    + 0.3 * detour_score
```

三项指标均非负且小于等于 `1`，所以最终 difficulty 也位于 `[0, 1]`。
采用加法而不是乘法，避免某一项为零时抹掉其他难度来源。

### 2.1 绝对长度难度

```text
length_scale = 20 grid steps
length_score = d / (d + length_scale)
```

`length_scale` 是半饱和距离：当 `d=20` 时，`length_score=0.5`。它是固定、
版本化的 cell-distance 标尺，不从当前 variant 列表或 registry 动态计算。
因此新增地图不会改变旧地图已有 pair 的分数。

内部 record 继续保留兼容诊断字段 `path_score=d/D`，但 v2 difficulty 不再
使用它；真正参与公式的是固定尺度的 `length_score`。

### 2.2 干扰岔路难度

branch metric 只把会导致路径严格变长的出口视为干扰。另一条等长最短路
不是干扰路线。

在 canonical path 的每个动作位置 `p_i`（goal 本身不再需要决策）检查合法
邻居：

1. 中间节点忽略刚走过的 `p_(i-1)`，回头不算岔路；
2. 设当前剩余最短距离为 `r=d-i`；
3. 若邻居 `n` 满足 `dist(n,g)=r-1`，它仍位于某条最短路上，不算干扰；
4. 其他非回头邻居会增加完成路径的长度，计为干扰出口。

令 `b_i` 为位置 `p_i` 的干扰出口数量。四邻接网格中，起点最多有三个
干扰出口，中间节点排除来路且至少保留一个最短路出口后最多有两个：

```text
local_branch_score(i) =
    b_i / 3,  i = 0
    b_i / 2,  0 < i < d

branch_score = sum(local_branch_score(i), i=0..d-1) / d
```

同时保留两个便于审计的原始计数：

- `distractor_point_count`：至少存在一个干扰出口的路径位置数量；
- `distractor_exit_count`：所有路径位置的干扰出口总数。

branch score 使用沿路径的密度而不是仅使用总数，避免与绝对路径长度重复
计分。绝对长度项负责长时间控制，branch 项负责单位路径上的错误选择密度。

### 2.3 绕路难度

```text
detour_score = 1 - m / d
```

没有墙体迫使绕行时 `d=m`，得分为 `0`；实际最短路相对 Manhattan distance
越长，得分越接近 `1`。

实现继续保存兼容字段：

```text
away_steps = canonical path 中使 Manhattan distance 增大的步数
away_frac = away_steps / d
```

在四邻接网格最短路径上有：

```text
detour_score = 2 * away_frac
```

v2 difficulty 使用归一化更完整的 `detour_score`，不再直接使用
`away_frac`。

## 3. 地图难度

先计算 pair space 中全部可达有序 start/goal paths 的 v2 path difficulty，
从低到高排序。设可达有序 pair 总数为 `N`：

```text
K = max(1, ceil(0.10 * N))
map_difficulty = mean(the K highest path difficulties)
```

也就是说，地图难度是最难前 10% 路径的平均难度。使用平均值而不是单一
最大值，可以降低某一个异常 pair 或 canonical-path tie break 对地图分数的
影响，同时仍聚焦地图的困难区域。

普通 CrossMaze 注册地图通过 `get_map_difficulty_config(...)` 使用全部 free
cells 构造 pair space，因此结果是真正的全地图评估难度。

地图难度元数据包括：

- `map_difficulty`；
- `map_difficulty_top_fraction = 0.10`；
- `map_difficulty_path_count = K`；
- `map_reachable_pair_count = N`；
- `map_diameter = D`。

## 4. Hard-sample 与 difficulty 的关系

普通 eval 的 `hard-sample` 直接使用上述 v2 `path_difficulty`：

1. 构造候选集合中的全部可达有序 pair；
2. 用 v2 公式计算每个 pair 的 difficulty；
3. 按 difficulty 从低到高排序；
4. eval 通过 `eval_hard_sample_top_percent` 或 `eval_hard_sample_top_n` 保留
   hard pool；
5. 在保留的 pool 内重新计算 rank-linear 权重：

```text
rank_score = rank / max(pool_size - 1, 1)
sample_weight = 1 + hard_sample_alpha * rank_score
```

6. eval 使用 run seed 从 hard pool 无放回抽取最多 100 个 pair，再按稳定
   permutation cycle 分配给 episode。

map difficulty 只用于描述地图，不参与 hard pool 的二次排序；hard pool 本身
已经由同一套 path difficulty 规则排序。

## 5. 结果保存

`fix-start-goal` 和 `hard-sample` 中，每个实际 eval episode 在 variant
`result.json` 的 `episode_results` 中保存：

```json
{
  "start_cell": [1, 1],
  "goal_cell": [13, 11],
  "start_goal_difficulty": 0.54,
  "start_goal_difficulty_components": {
    "version": "v2",
    "path_len": 46,
    "manhattan_distance": 14,
    "length_scale": 20.0,
    "length_score": 0.697,
    "distractor_point_count": 9,
    "distractor_exit_count": 12,
    "branch_score": 0.174,
    "away_steps": 16,
    "away_frac": 0.348,
    "detour_score": 0.696,
    "weights": {"length": 0.4, "branch": 0.3, "detour": 0.3}
  }
}
```

variant 级 `result.json` 同时保存三项 episode 均值、difficulty config 和地图
难度元数据。hard-sample 模式还会在同一目录写：

```text
eval_position_pool.json
```

该文件保存 seed 最终选中的最多 100 个 pair 及完整分解。全部中间 pair 默认
只保留在进程内缓存中，不写入结果目录。

`random-start-goal` 不使用预选 pair，因此 episode 级 start-goal components
保持为 `null`；variant 级 map difficulty 仍然按注册地图的全部 free-cell pair
space 计算并写入 `result.json`。

## 6. 可复现性和修改规则

- `PATH_DIFFICULTY_VERSION`、length scale、三项权重和 map top fraction 都是
  版本化常量；
- 修改任一公式或常量时必须提升 difficulty version；
- 同一版本中不能根据当前 eval variant 列表动态改变标尺；
- canonical shortest path 受固定 BFS 邻居顺序控制；若改变邻居顺序，branch
  metrics 可能改变，也应提升 version；
- difficulty 是环境几何启发式指标。若未来使用 rollout 成功率拟合权重，应
  建立新的、明确依赖基准策略的 metric，不能静默覆盖当前几何 v2 定义。
