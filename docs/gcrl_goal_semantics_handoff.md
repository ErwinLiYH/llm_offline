# GCRL full-state goal 语义修复交接

> 实施状态（2026-07-31）：本文描述的迁移已落地为唯一协议
> `full_observation_v1`。当前实现包含同一 observation normalizer、在线双 reset goal capture 和 checkpoint metadata sidecar。下文“当前实现”的 compact-goal
> 描述保留为迁移前问题记录，不再代表仓库现状。

## 0. 给新 session 的任务结论

本次任务不是给现有 compact goal 增加特征，而是迁移 CRL/HIQL 的输入协议：

> 对任意被采样为 goal 的状态索引 `j`，goal 必须是该索引对应的完整 GCRL state observation `s_j`。`state` 与 `goal` 使用完全相同的向量空间、特征顺序、维度和预处理。离线训练和在线 rollout 必须遵守同一个协议。

也就是说，目标状态不是 `s_j` 的 xy 投影，也不是“xy 加 goal cell”；它就是另一条完整 state observation。地图仍然属于 state observation，因此也必须同时出现在 `s` 和 `g` 中。HIQL 的 `goal_rep([s, g])` 应收到两个同构的完整 state observation；CRL 的 `psi(g)` 和 actor 也应收到完整 goal-state observation。

不要把 `use_oracle_rep=True` 当成本任务的修法。oracle representation 是只给位置坐标的另一套 ablation/协议，恰好接近当前 compact-goal 设计；本任务明确要恢复普通 state-based、非-oracle 的 full-observation goal 语义。

## 1. 参考来源与应对齐的口径

本仓库 GCRL 后端声明基于 OGBench 1.2.1、commit `1d4140997f60c52c6fb0702ec100dc988b18c548`，见 [`baselines/gcrl/UPSTREAM.md`](../baselines/gcrl/UPSTREAM.md)。实现时以这个固定 commit 为主要代码口径，并用论文公式核对语义：

- [OGBench `GCDataset` / `HGCDataset`](https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/impls/utils/datasets.py)
- [OGBench HIQL agent](https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/impls/agents/hiql.py)
- [OGBench CRL agent](https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/impls/agents/crl.py)
- [OGBench Maze rollout goal construction](https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/ogbench/locomaze/maze.py)
- [HIQL paper](https://arxiv.org/abs/2307.11949)
- [CRL paper](https://arxiv.org/abs/2206.07568)

这里的“对齐原文”指 goal 的语义和网络公式对齐，不是把 CrossMaze 的 state 特征删成 OGBench 环境的原始维度。我们仍保留本仓库选定的本体状态、地图、位置感知和墙体感知，只是要求目标侧也是同一套完整 state 特征。

## 2. 三方对比

| 项目 | 论文/算法语义 | OGBench 固定 commit | 本仓库当前实现 |
|---|---|---|---|
| goal 的含义 | CRL 使用 goal state `s_g`；HIQL 设 `G = S`，goal 是状态 | 从 observation 数组按 goal index 取完整 observation | 从单独的 `arrays.goals` 取 xy/goal-cell 投影 |
| state/goal 空间 | 同构，`g in S` | `ex_goals = ex_observations` | 异构，`state_dim != goal_dim` |
| CRL critic | `f(s,a,g) = phi(s,a)^T psi(g)` | `psi` 输入完整 observation；batch 内其他完整 observations 是 negatives | `psi` 输入 2D xy，开 location sensing 时为 4D |
| CRL actor | `pi(a | s,g)` | actor 输入完整 `s` 和完整 `g` | actor 输入完整 `s` 和 compact `g` |
| HIQL representation | 理论写 `phi(g)`；state-based 实验实现为 state-dependent `phi([s,g])` | `goal_rep(concat([observation, goal]))`，两侧同构 | 公式结构还在，但输入是“高维完整 `s` + 极低维 compact `g`” |
| HIQL relabel tensors | value goal、low goal、high goal、high target 都是状态 | 均调用 `get_observations(sampled_idxs)` | goals 来自 `arrays.goals[...]`；high target 被拆成 full state 与 compact goal 两份 |
| rollout goal | goal observation | 非-oracle 时把 agent 放到 goal xy 后调用 `get_ob()`，把结果放入 `info['goal']` | wrapper 直接从环境 `desired_goal` 取 xy，并追加 goal cell |
| oracle 模式 | 位置表示是一项可选 ablation | `use_oracle_rep=True` 时返回 xy oracle rep | 当前 compact goal 在语义上更接近这一分支，但并没有完整复刻其数据/配置协议 |

当前 full-sensing 维度能直观看出不对称：

| family | 当前 state | 当前 goal | 修复后 state | 修复后 goal |
|---|---:|---:|---:|---:|
| PointMaze | 235 = base 4 + map 225 + cell 2 + wall 4 | 4 = xy 2 + goal cell 2 | 235 | 235 |
| AntMaze | 227 = base 29 + map 192 + cell 2 + wall 4 | 4 = xy 2 + goal cell 2 | 227 | 227 |

无 sensing 时也必须同构：PointMaze 为 `4/4`，AntMaze 为 `29/29`，而不是当前的 `4/2`、`29/2`。

## 3. 原始实现的关键证据

### 3.1 数据 relabeling

OGBench `GCDataset.sample()` 的核心语义是：

```python
batch['value_goals'] = self.get_observations(value_goal_idxs)
batch['actor_goals'] = self.get_observations(actor_goal_idxs)
```

而 `get_observations(idxs)` 返回 `dataset['observations'][idxs]`。因此 goal 和 `batch['observations']` 来自同一个数组、同一个 observation space。

OGBench `HGCDataset.sample()` 同样使用：

```python
batch['value_goals'] = self.get_observations(value_goal_idxs)
batch['low_actor_goals'] = self.get_observations(low_goal_idxs)
batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
batch['high_actor_targets'] = self.get_observations(high_target_idxs)
```

这里没有“高层 target state + 高层 target xy”两套 tensor；`high_actor_targets` 本身就是完整 observation。

### 3.2 HIQL 网络与 loss

OGBench 创建网络时显式执行 `ex_goals = ex_observations`。state-based 配置下：

```text
goal_rep: phi([s, g]) -> normalized rep_dim vector
value:    V(s, phi([s, g]))
high:     pi_h(z | s, g)
low:      pi_l(a | s, z)
```

对应训练关系为：

```text
low advantage  = V(s_next, w) - V(s, w)
high advantage = V(s_target, g) - V(s, g)
high target z  = phi([s, s_target])
```

其中 `w`、`g`、`s_target` 都是完整 state observation。论文把 goal space 假设为 state space；论文实验/OGBench 代码又进一步使用 state-dependent representation `phi([s,g])`。所以“`goal_rep` 需要 `s`”时，应传入本仓库定义的完整 state vector；拼接另一侧 `g` 时，`g` 也必须是同构的完整 state vector。

### 3.3 CRL 网络与 loss

CRL 的双线性 critic 是：

```text
f(s, a, g) = phi(s, a)^T psi(g)
```

OGBench 对一个 batch 构造 `B x B` logits：对角线是 `(s_i, a_i, g_i)` 正样本，非对角线使用其他样本的完整 goal observations 作为负样本，然后计算 sigmoid binary cross-entropy。创建网络时同样使用 `ex_goals = ex_observations`，actor 接收 `(s,g)`。

因此，本次迁移会同时影响 HIQL 和 CRL；不能只修 HIQL 的 `goal_rep`。

### 3.4 rollout goal observation

OGBench 普通 state-based maze reset 的顺序是：先 reset 并随机动作若干步使 locomotion state 稳定，设定 goal，调用 `set_xy(goal_xy)`，随后用 `get_ob()` 取得 `goal_ob`；再执行真正的 reset 并放到起点，把刚才的 `goal_ob` 放入 `info['goal']`。`use_oracle_rep=True` 才改为只返回 xy oracle representation。

这也回答了 AntMaze 的疑问：原实现不是把一只随意姿态的 ant 粗暴传送后继续运行；它先生成一份稳定的 locomotion 状态，再只改 xy 并“拍下” goal observation，随后重新 reset 到真正起点。goal observation 的非 xy 维度确实带有一个代表性的姿态/速度，它们对 xy 成功判定而言可能是 nuisance，但这是普通非-oracle 协议本身的一部分。

## 4. 当前仓库的真实偏离

偏离是本仓库主动做过的一次协议改造，并非网络里某条 `Identity()` 路径导致：

1. [`baselines/data/observation.py`](../baselines/data/observation.py)
   - `goal_conditioned_observation_dims()` 明确返回不同的 state/goal 维度。
   - docstring 明写 desired goal 不复制进 state。
   - `vectorize_goal_conditioned_observation()` 构造 `state_features=[state,...]`，而 `goal_features=[goal_xy, goal_cell]`。
   - `GoalConditionedObservationWrapper` 在线也产出这个异构 Dict。
2. [`baselines/gcrl/data.py`](../baselines/gcrl/data.py)
   - `GCRLEpisode` / `_VariantArrays` 分别保存 `states` 与 `goals`。
   - `_convert_episode()` 用 achieved xy 生成每行 compact goal。
   - sampler 的 `value_goals`、CRL `actor_goals`、HIQL 三种 goals 都来自 `arrays.goals[...]`。
   - HIQL 的 high target 被拆为 `high_actor_target_states` 和 `high_actor_target_goals`。
   - normalizer 分别拟合 state 与 compact-goal statistics。
3. [`baselines/gcrl/agents.py`](../baselines/gcrl/agents.py)
   - 大部分公式骨架与 OGBench 一致，但维度由异构示例输入初始化。
   - HIQL high target representation 当前拼的是 `[observations, high_actor_target_goals]`，不是 `[observations, high_actor_targets]`。
4. [`baselines/gcrl/runner.py`](../baselines/gcrl/runner.py) 与 [`baselines/evaluation.py`](../baselines/evaluation.py)
   - agent 初始化、normalizer 和 rollout 都接受当前 compact-goal schema。
5. 文档中的 compact-goal 描述也要迁移：[`AGENTS.md`](../AGENTS.md)、[`DESIGN.md`](../DESIGN.md)、[`baselines/README.md`](../baselines/README.md)、[`CHANGELOG.md`](../CHANGELOG.md) 和 [`baselines/gcrl/UPSTREAM.md`](../baselines/gcrl/UPSTREAM.md)。

## 5. 目标数据协议

建议把协议写成一个可测试的不变量，而不是靠维度碰巧相等：

```text
state_schema == goal_schema
goal(j) == full_state_observation(j)
state_dim == goal_dim
```

本仓库的完整 state observation 顺序应为：

```text
PointMaze: observation(4)
           + optional padded map(225)
           + optional current position_cell(2)
           + optional current wall_sensing(4)

AntMaze:   achieved_goal(2) + proprioception(27)
           + optional padded map(192)
           + optional current position_cell(2)
           + optional current wall_sensing(4)
```

对 goal index `j`，goal 的 `position_cell` 和 `wall_sensing` 都从 `s_j` 的 achieved position 计算；它不是 episode 原始 `desired_goal` 的 cell。地图是同一 variant 的静态地图，也要放到 goal 的同一槽位。

离线 relabel 示例：

```python
value_goals = restore_full_state(arrays, arrays.states[value_goal_indices])
actor_goals = restore_full_state(arrays, arrays.states[actor_goal_indices])
```

HIQL 应恢复上游命名/结构：

```python
low_actor_goals = full_state(low_goal_indices)
high_actor_goals = full_state(high_goal_indices)
high_actor_targets = full_state(high_target_indices)
```

并在 agent 中使用：

```python
V(high_actor_targets, high_actor_goals)
goal_rep(concat([observations, high_actor_targets]))
```

## 6. 建议实施顺序

### 6.1 先统一 vectorization/schema API

- 让 GCRL 的 state vectorization 只表达“一个 state observation”，不再同时返回 compact goal。
- `goal_conditioned_observation_dims/schema` 应表达两侧同构，或者改成单一 `gcrl_state_*` API 并由 wrapper 复用两次。
- 尽量从类型/数据结构上消除“goal 有独立 schema”的可能，避免以后再次漂移。

### 6.2 离线数据只从 states relabel

- 最干净的做法是删除 `GCRLEpisode.goals` 与 `_VariantArrays.goals`；所有 relabeled goals 都按索引从 `states` 取。
- 当前为了大数据集节省内存，把 variant-static map 从逐行 state 中抽出，只在 batch 采样时由 `_restore_map_features()` 插回。这个优化可以保留，但每一种 goal tensor 和 `high_actor_targets` 都必须走相同的 restore 路径。
- 保持现有 episode boundary、within-variant relabeling、variant-homogeneous minibatch 不变。这些是 CrossMaze 多地图场景需要的本地正确改动，与 goal 语义修复不冲突。

### 6.3 合并 HIQL high target

- 用单一 `high_actor_targets` 取代当前 `high_actor_target_states` + `high_actor_target_goals`。
- 修改 normalizer key 分类、agent loss、测试和所有日志/fixture。
- 不要把“同一个 tensor 同时存两份”作为最终设计；若为了分步迁移暂留 alias，完成前应删除。

### 6.4 统一归一化语义

OGBench goal 与 observation 属于同一个 observation space。本仓库增加了 observation normalization，因此修复后应优先使用同一组 state mean/std 同时归一化 state 与 goal，而不是从不同采样分布拟合两套统计量。

如果保留 `goal_mean/goal_std` 字段用于 artifact 兼容，它们必须与 `state_mean/state_std` 完全相等，并通过测试锁住；更干净的方案是 artifact 版本升级后只保存一套 observation statistics。不能继续从 compact `arrays.goals` 拟合 goal statistics。

### 6.5 构造在线 full goal observation

当前 evaluation 环境只天然给出 `desired_goal` xy，不能仅靠 observation wrapper 把 Ant 的 27 维非 xy 状态凭空还原。需要在 reset 时构造一条代表性的完整 goal observation，并确保不改变真正 rollout 的初态。

建议新 session 先检查 `crossmaze.make()` 产生的 wrapper 链和 Gymnasium Robotics v4 reset/state API，再选择最小、可测试的实现层。原则如下：

1. 完成真实 episode reset，取得稳定且合法的 simulator state；或严格复现 OGBench 的“预 reset + 若干随机动作稳定”过程。
2. 保存完整 MuJoCo `qpos/qvel` 及任何会被操作影响的 wrapper/env 状态。
3. 仅把 torso/point xy 替换为目标 xy，执行 MuJoCo forward，读取标准 dict observation。
4. 用和离线 state 完全相同的 vectorizer，为这条 goal observation 追加同一地图、goal 位置对应的 `position_cell` 和 `wall_sensing`。
5. 恢复 simulator 和 wrapper 状态，再返回实际起点 observation。

AntMaze 必须维持本仓库 v4 contract：`achieved_goal(2) + observation(27) = 29`，不要因访问底层环境而意外切到 v5 contact-force observation。也不要把 goal 的 27 个非 xy 槽简单清零；那既不是数据中的真实 future state，也不是 OGBench 的代表性 goal observation。

PointMaze 本仓库的 base state 是 position+velocity 4D，而 OGBench Point state 细节不必机械复制；关键是在线 goal 也要生成合法的 4D Point state，并与离线 future-state 行采用同一 schema。

如果实现 OGBench 风格预稳定会额外消耗环境 RNG，应确保同一 eval seed 仍可重现，且 goal-observation 构造不会改变实际起点/目标抽样。必要时保存/恢复 RNG state，或使用隔离的 goal-observation 构造环境。

### 6.6 迁移 artifact/checkpoint 协议

- 在 resolved config、dataset manifest、checkpoint metadata 中写入明确版本，例如 `gcrl_goal_semantics: full_observation_v1`。
- 旧 compact-goal checkpoint 的参数 shape 与新网络不同，必须明确拒绝加载，不能静默兼容。
- 旧实验结果仍可作为历史结果，但文档需标注它们运行于 `compact_xy` 语义，不能与修复后结果当作同一实现直接比较。
- 当前任务目标是切换默认/唯一实现到 full-state goal；不要未经确认额外维护一个 compact-goal 配置分支。如果后续需要 xy oracle ablation，应作为显式、单独命名的实验轴实现。

### 6.7 更新说明文档

- `baselines/gcrl/UPSTREAM.md` 中 “Local changes split CrossMaze states from compact goals” 必须删除/改写。
- `AGENTS.md` 与 `baselines/README.md` 当前明确写着 CRL/HIQL 使用 2D xy goal，必须同步修改。
- `DESIGN.md`、`CHANGELOG.md` 和相关配置注释中搜索 `compact goal`、`goal_xy`、`goal cell`、`state_dim`/`goal_dim`，逐项核对。

## 7. 必须通过的验收测试

至少覆盖以下不变量：

1. PointMaze 和 AntMaze 在 no-sensing/full-sensing 下都满足 `state_dim == goal_dim`，并核对具体维度。
2. 给定固定 goal index，`value_goals`、CRL `actor_goals`、HIQL `low_actor_goals`/`high_actor_goals` 都逐元素等于 restore 后的 `states[index]`。
3. HIQL `high_actor_targets` 逐元素等于 restore 后的 target state 行。
4. current-goal 分支中，归一化后的 goal 与 observation 仍逐元素相等。
5. map 槽在所有 state/goal/high-target tensors 中都存在、位置一致、值一致；仍只在 variant level 紧凑存储。
6. goal 的 cell/wall slots 来自被 relabel 的 target state 位置，而不是原 episode 的 `desired_goal`。
7. rollout wrapper 返回相同 shape/schema 的 `state` 与 `goal`；goal 的 achieved xy 等于本 episode desired target xy（允许环境既有噪声/容差）。
8. 构造 rollout goal observation 前后，真实 episode 的起点 simulator state、目标选择和 RNG 可重现性不被破坏。
9. AntMaze goal base 恰为 29D v4 contract、数值有限，不含 v5 contact-force 维度；goal observation 的姿态合法。
10. CRL 和 HIQL 都至少能完成一次 agent 初始化、一次 update 和一次 predict，所有网络输入 shape 与公式一致。
11. 旧 `compact_xy` manifest/checkpoint 被明确拒绝。
12. 现有 within-variant goal relabeling 和 one-variant-per-batch 行为不回退，CRL negatives 不跨地图。

重点测试文件预计包括：

- [`baselines/tests/test_observation.py`](../baselines/tests/test_observation.py)
- [`baselines/tests/test_gcrl_data.py`](../baselines/tests/test_gcrl_data.py)
- [`baselines/tests/test_gcrl_agents.py`](../baselines/tests/test_gcrl_agents.py)
- [`baselines/tests/test_evaluation.py`](../baselines/tests/test_evaluation.py)

目标测试命令（按实际模块名调整）为：

```bash
JAX_PLATFORMS=cpu micromamba run -n llm_offline_gcrl python -m unittest \
  baselines.tests.test_observation \
  baselines.tests.test_gcrl_data \
  baselines.tests.test_gcrl_agents \
  baselines.tests.test_evaluation
```

完成单元测试后，分别做 PointMaze 与 AntMaze 的最小 CRL/HIQL smoke run，至少覆盖一个 update 和一个 reset/rollout action。不要只根据网络能跑通就宣告完成；offline/online goal schema 的一致性才是这次修复的核心验收项。

## 8. 明确不做的事情

- 不把地图从 `phi([s,g])`、CRL `psi(g)` 或 actor 输入中删掉；地图是本仓库 state observation 的一部分。
- 不用缩放、加权 goal 特征、改 bottleneck、额外 regularization 等 trick 掩盖协议问题。先恢复原始语义，再单独做算法 ablation。
- 不把 `use_oracle_rep=True` 当成 full-state 修复。
- 不跨 variant/map 采样 random goals 或 CRL negatives。
- 不静默复用旧 compact-goal checkpoint。

## 9. 新 session 开工时可直接采用的任务描述

> 阅读 `docs/gcrl_goal_semantics_handoff.md`、仓库 `AGENTS.md` 和 `baselines/gcrl/UPSTREAM.md`。把 CRL/HIQL 从当前 compact xy goal 协议迁移到 OGBench 普通 state-based、非-oracle 的 full-state goal 协议：离线任何 relabeled goal 都必须是目标索引对应的完整 GCRL state observation；在线 rollout goal 也必须构造成同构完整 observation；HIQL 恢复单一 full `high_actor_targets`；state/goal 使用同一 schema 和 normalization。保留 within-variant relabeling、variant-homogeneous batch 和静态 map 紧凑存储。实现代码、迁移 artifact 版本、更新文档并完成 Point/Ant 的数据、agent 和 rollout 测试。不要实现 oracle/compact-goal 兼容分支，除非先得到用户确认。
