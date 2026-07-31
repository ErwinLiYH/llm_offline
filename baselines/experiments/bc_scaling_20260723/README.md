# CrossMaze BC scaling 实验

本目录执行数值 goal-conditioned deterministic BC 的 width/depth scaling
实验。训练图与测试图沿用既有多环境协议：PointMaze 为 16 个训练图和 6 个
`test-layoutV2-*` 测试图；AntMaze 为 16 个训练图和 4 个 `test-layout-*`
测试图。测试图不进入离线数据、episode sampling 或 observation scaler。

本次执行只保持已确认的“训练地图集 / 测试地图集”语义，不额外引入
development/held-out split；每个 checkpoint 都在固定测试地图记录结果，但不会
依据测试结果回改已经启动的训练配方。每个设置完成三个 seed 后，测试图按下述
冻结规则参与 checkpoint 选择，因此 selected-checkpoint 结果不解释为独立
held-out 推断。

所有正式 run 使用：每个变体最多 300 条 episode、按变体平衡、90/10
episode train/validation split、map/location/wall 全开、`v3` wall sensing、
固定 `sampling_seed=0`。每 100k updates 保存并在每张训练/测试图上 rollout
30 个固定 reset seed（`0..29`）。训练过程仍会看到测试图结果，这是与此前
多环境 baseline 保持一致的既有 train/test 语义；不会基于测试结果改写已启动
run 的配置。为使完整 30 episode 评测可执行，单一地图的 30 个独立环境会各自
reset/step，但同步把当前 observation 合并成一次 `policy.predict`；这不改变每条
episode 的 seed、动作校验、环境动力学或逐 episode JSON 记录。

每个训练 epoch 还会写入 `training_diagnostics.jsonl`：采样的反向传播前
global gradient norm、parameter global norm、固定训练 observation probe 上的
动作 mean/std/min/max、吞吐、累计 processed examples、peak RSS 与 peak GPU
allocated memory。probe 只由训练 episode 构成；因此它不读取测试地图，也不改变
batch sampler 或 BC objective。

新启动的 run 在每个 `.d3` checkpoint 完成评测后，同时保存
`trainer_state_step_<step>.pt`，其中包含 Python、NumPy、PyTorch CPU/CUDA RNG
状态及 epoch/step。它为以后 1M→2M 的精确 minibatch 续训保留必要状态；旧
checkpoint 没有该 sidecar，不能声称能够 bitwise-equivalent 地续训。

续训配置把目标总步数写在 budget 文件中，并同时指定模型与 trainer-state
sidecar。例如从 1M 延长到 2M 时，另建一个不提交的本地 override：

```yaml
resume:
  checkpoint: baseline_runs/<source>/checkpoints/step_1000000.d3
  trainer_state: baseline_runs/<source>/checkpoints/trainer_state_step_1000000.pt
```

然后在原始算法/协议/architecture/seed 配置后叠加该 override 与
`budget/u2m.yaml`。runner 会验证训练数据、observation、network、optimizer、
evaluation 和 seed 配方与源 run 完全一致，恢复模型、optimizer、scaler 和 RNG，
并让新 checkpoint 从 `step_1100000` 继续编号。新 summary 合并源 run 在 resume
step 之前的 diagnostics/evaluation 曲线，同时记录每个 step 的真实 artifact
路径。

2M 条件延长使用
`bc-scaling-extension-<family>-<config>-s<seed>-u2m-<date>` 命名。一旦审计器
发现某个 family/config 的任一 2M run，就视为该配置的三 seed 延长矩阵已经触发：
`--strict` 会要求三个 seed 都完成，并审计 100k--2M 的全部 checkpoint、逐环境
30 episode、诊断记录和跨源 run artifact 路径。未触发的配置不会被误报为缺失。

PointMaze 的条件 1M run 同样使用首次发现即激活三 seed 矩阵的规则，命名为
`bc-scaling-budget-pointmaze-<config>-s<seed>-u1m-<date>`。预算延长建议只在
共同预算末端计算：PointMaze 500k 核心 width/depth，或 AntMaze 1M 核心
width/depth。末三个 checkpoint 的 test 跨 seed 均值首尾至少增加 2 pp，或者
train 至少增加 3 pp 且 test 下降不超过 1 pp，并且至少 2/3 seed 同方向时，
才建议延长；检测到持续过拟合时不延长。触发点必须与相邻配置成组延长，优先
选择较小邻点；处于最小边界时选择较大邻点。旧 AntMaze 500k 前缀、
legacy/batch 对照和已延长曲线不会触发下一预算。

PointMaze 条件结构点 `W5120/W6144/W7168/W8192/D512/D1024` 的 500k 正式 run
也遵守同一完整性约束：任一 seed 的正式目录一旦出现，`--strict` 就要求
该结构补齐全部三个 seed。AntMaze 的条件结构点属于 1M 共同预算，因此由
上述 budget discovery 自动激活，而不是进入旧 500k 表。

`experiment_log.md` 与 `report.md` 是中文运行产物，受 Git ignore 保护；
原始 checkpoint、summary、逐 episode rollout 位于 `baseline_runs/`。

## 网络

`plain_mlp` 与 `residual_mlp` 使用论文 *1000 Layer Networks for
Self-Supervised RL* 的统一单元：`Dense -> LayerNorm -> Swish`，Dense 使用
LeCun-uniform 权重和零 bias。二者均先经过一个相同的 activated input
projection；残差版本每四个 body Dense 在第四个 Swish 后加 identity。
`body_depth` 只统计 body 的 Dense 数，不包括 input projection 或 d3rlpy 的
deterministic action head。`legacy.yaml` 是历史 `[1024, 1024, 1024]` ReLU
锚点，不能混入新配方的单因素 scaling 曲线。

正式 width 序列从 `W256/W512/W1024/W2048/W3072/W4096` 开始。`W2048`
之后按固定 `+1024` 增加，不再直接翻倍；W5120 相对 W4096 提升超过
2 pp，而 W6144 相对 W5120 首次回落。最近两次增量尚未同时低于 2 pp，
因此按冻结的两次低增益早停规则再加入 `W7168`。W7168 正式结果虽使协议门
允许继续到 `W8192`，但 W5120 之后的 W6144/W7168 未形成持续提升；2026-07-30
按实验负责人决定在 W7168 截止 PointMaze width scaling。W8192 的 `3e-5`
range 在 49k 左右停止，保留为不完整诊断，不启动正式三 seed，也不作为性能
点。审计同时保留原协议门的 `eligible_for_50k_range` 证据，并将执行状态标为
`stopped_by_user_decision`。PointMaze residual depth 主序列为
`D4/D16/D64/D128/D256`；D128 避免从 D64
直接四倍跳到 D256，且须独立完成学习率 range。稳定后再逐级
`D512/D1024`。PointMaze 共同预算为 500k，AntMaze 共同预算为 1M；Ant 的
1M run 在本目录协议后额外叠加 `budget/u1m.yaml`。

Point W3072 的论文起始学习率 `3e-4` 在 seed 0/1 正常、seed 2 首个 25k
诊断中出现两维 action std 都为 0 的确定性坍缩。该组三 seed 因而降级为
pilot，不允许把 seed 0/1 的 `3e-4` 与 seed 2 的较低学习率混成正式曲线。
坍缩 seed 依次执行 `1e-4 → 3e-5 → 1e-5` 的完整 50k range；首个稳定值
冻结后，全部三个 seed 从头统一重跑 500k。审计 JSON 的
`width_recipe_audits.pointmaze.w3072-d4` 保留 pilot 坍缩、range 和三 seed
共享正式学习率证据；多个日期目录按获批学习率解析，旧 pilot 不必删除。
W4096 在 W3072 配方冻结前不得启动，其学习率将从 W3072 的稳定值开始独立
range，避免把已观察到的宽网络不稳定继续传播。

W5120 在 W4096 正式结果出现前使用固定门槛：W4096 相对 W3072 selected test
至少提高 2 pp、至少 2/3 seed 同向、无塌缩或持续过拟合，并且 W4096 最后三个
共同预算 checkpoint 的 train/test 首尾变化绝对值分别不超过 3/2 pp。满足后
只允许 W5120 进入 50k range；若 W4096 末段仍在提高，则先成组延长
W3072/W4096。审计 JSON 的 `next_width_range_gates` 和报告对应表记录该判断。
W6144 由 W4096/W5120 的共同 500k 结果判断，且必须先通过最近三档的
axis 早停规则，再从 W5120 已冻结的学习率开始独立 50k range。W6144
正式结果出现后，后续 `+1024` 扩展原由两次连续低增益规则主导：单次回落不
停止，W7168 使用 `W4096 → W5120 → W6144` 判断，并从 W6144 冻结的
学习率开始独立 range。该规则在 W7168 后本来允许 W8192 进入 range；实际
执行按上述人工 plateau 判断在 W7168 截止，并明确披露为计划覆盖而非把它
改写成原早停规则的结论。

所有核心网格完成后还应用一条 axis 早停：按三个 seed 的共享 selected
checkpoint 比较最近三个相邻设置；若最近两次 width 或 depth 扩展的 test
宏平均成功率增量都严格小于 2 pp（下降也计入），该 family 的该轴停止，不再
启动下一条件扩展。核心网格和已启动设置不追溯取消；因此 D512 使用
`D64 → D128 → D256`，W5120 使用 `W2048 → W3072 → W4096` 判断，
W6144 使用 `W3072 → W4096 → W5120`，W7168 使用
`W4096 → W5120 → W6144`，W8192 使用
`W5120 → W6144 → W7168` 判断。
机器可读结果位于 `axis_early_stop_gates`。

plain D4 在正式 schema 下的 CPU 实例化参数量为：

| Family | W2048 | W3072 | W4096 | W5120 | W6144 | W7168 | W8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Point | 17,301,506 | 38,535,170 | 68,157,442 | 106,168,322 | 152,567,810 | 207,355,906 | 270,532,610 |
| Ant | 17,297,416 | 38,529,032 | 68,149,256 | 106,158,088 | 152,555,528 | 207,341,576 | 270,516,232 |

两类 W2048 均与真实 run metadata 完全匹配；更宽配置的参数量不能替代 GPU
显存、吞吐和 wall-clock 实测。

已经完成的 batch=512/1024 与 matched-example 对照继续保留在审计 JSON 和
补充结果中，但不属于当前 width/depth 的 strict 正式矩阵，也不会触发新的
训练或预算延长。

## 最佳 checkpoint

同一设置的三个 seed 必须选择同一个 step。审计器先在三 seed 均值曲线上识别
持续过拟合：局部 test 峰值后至少两个 checkpoint 连续下降、累计下降至少
2 percentage points，后续未恢复到峰值 1 percentage point 以内，endpoint
train 不低于峰值 train，且至少 2/3 seed 的 endpoint test 低于峰值。检测到
后，下降阶段不进入候选。

其余候选按
`joint_success = 0.5 * train_success_macro + 0.5 * test_success_macro`
选择；并列时依次优先 test 更高、step 更早。报告同时保留 selected checkpoint
和 max-budget endpoint，禁止为每个 seed 分别挑峰值。

## 分层运行示例

先做计划规定的 1k-update 全地图 smoke：

```bash
micromamba run -n llm_offline_baselines python baseline_train.py --config \
  baselines/configs/base.pointmaze.yaml \
  baselines/configs/mlp_bc.yaml \
  baselines/experiments/bc_scaling_20260723/protocol.pointmaze.yaml \
  baselines/experiments/bc_scaling_20260723/optimizer.base.yaml \
  baselines/experiments/bc_scaling_20260723/architecture/residual_d4_w256.yaml \
  baselines/experiments/bc_scaling_20260723/smoke.yaml \
  --experiment_id bc-scaling-preflight-pointmaze-r4-w256-s0-20260723
```

PointMaze 正式 `D64`、seed 1 示例：

```bash
micromamba run -n llm_offline_baselines python baseline_train.py --config \
  baselines/configs/base.pointmaze.yaml \
  baselines/configs/mlp_bc.yaml \
  baselines/experiments/bc_scaling_20260723/protocol.pointmaze.yaml \
  baselines/experiments/bc_scaling_20260723/optimizer.base.yaml \
  baselines/experiments/bc_scaling_20260723/architecture/residual_d64_w256.yaml \
  baselines/experiments/bc_scaling_20260723/rescue/d64_lr1e4.yaml \
  baselines/experiments/bc_scaling_20260723/seed/s1.yaml \
  --experiment_id bc-scaling-formal-pointmaze-r64-w256-s1-500k-20260726
```

每个环境 family 的深度轴都独立以 `3e-4` 做 range 起点。若某深度发生 action
collapse，则只扫描 learning rate；首个经完整 range 验证为稳定的值冻结为该
family/depth 的正式配置，并在报告和审计中与该 depth 一同记录。PointMaze
找到的 rescue 学习率不会自动用于 AntMaze。它用于比较“每个深度可正常训练时的
性能”，而不是强制所有深度共用学习率的纯优化消融。

当前已冻结的 PointMaze 深度配方是：D64 使用 `1e-4`，D256 使用 `1e-5`；
D128 从 `3e-4` 开始独立执行完整 range，并冻结首个稳定学习率。
D256 的 `3e-4`、`1e-4` 和 `3e-5` 均发生边界常数动作塌缩；`1e-5` 完成了
50k range，loss、validation MSE、action distribution、gradient、五个全地图
checkpoint/evaluation 和 trainer-state sidecar 均正常。AntMaze 对这两个深度
仍各自从 `3e-4` 开始，不继承 PointMaze 的 rescue 值。

AntMaze D256 的最低候选 `1e-5` 在 50k 出现与整体坍缩不同的慢启动：loss、
gradient 和 validation 条件正常，最后三个 epoch 已有至少 7/8 动作维持续达到
std `0.05`，但最后一维仍未达到原 range 标准。该 50k run 仍如实保留为
`unstable`，不会放宽或覆盖。只有
`assess_slow_start_extension.py` 确认原失败原因仅为这一项、且至少 7/8 维活跃
时，才允许从 50k 的 `.d3` 与 trainer-state sidecar 精确续训至 100k。
100k continuation 仍由 `assess_range.py --expected-n-steps 100000` 使用完全相同
的 loss、全动作维 std、gradient、validation、checkpoint 和每图 30 episode
阈值判定；通过后才可把 `1e-5` 冻结为 Ant D256 三 seed 正式配方。这个延长是
查看 50k 慢启动轨迹后加入的 post-hoc rescue，报告中必须与原 50k 失败一起展示。
若该 continuation 到 100k 仍只有 7/8 维活跃，则不再继续延长同一条已经精确
饱和的轨迹；下一项是独立从头运行 `lr=3e-6` 的完整 50k range。这个值是根据
`3e-5` 全维坍缩、`1e-5` 恢复 7/8 维后，沿原约三倍下降间隔追加的 post-hoc
候选，不是论文默认超参数。它仍须通过未放宽的完整 range gate，失败则 D256
记为没有稳定配方，不再继续无界降低学习率。

`D512/D1024` 的平台条件在 Point D128/D256 正式结果完成前固定：检查共同预算最后
三个 checkpoint，要求没有持续过拟合，且首尾 train success 变化绝对值不超过
3 pp、test success 不超过 2 pp。当前深度还必须相对前一深度 selected test
提高至少 3 pp，并有至少 2/3 seed 同向。满足这些条件只允许下一深度进入 50k
range；正式三 seed 仍须由 `assess_range.py` 验证其 family-specific 学习率、
动作分布、显存、RSS、吞吐与 wall-clock。审计 JSON 的
`next_depth_range_gates` 和 Markdown 对应表会自动记录每项门槛。

Point 239D observation / 2D continuous action 使用正式 layered config 的
CPU policy 实例化参数量为：
D128 `8,549,378`、D256 `17,036,290`、D512 `34,010,114`、
D1024 `67,957,762`。D128 已实际构建并验证为 128 个 body Dense、32 个
residual blocks、包含 input projection/action head 共 130 个 Dense；D256 与真实 range
metadata 完全一致；D512/D1024 的参数量只用于 range 前容量检查，不能替代它们
各自的 GPU 显存、吞吐和 wall-clock 实测。
Ant 231D observation / 8D continuous action 下，D16/D64/D128/D256/D512/D1024
分别为 `1,122,824` / `4,305,416` / `8,548,872` / `17,035,784` / `34,009,608` /
`67,957,256` 参数；D16 与真实 Ant run metadata 完全一致。family 间不复用
参数量。

新 depth/rescue 的 50k range 完成后必须运行：

```bash
micromamba run -n llm_offline_baselines python \
  baselines/experiments/bc_scaling_20260723/assess_range.py \
  baseline_runs/<range-experiment-id>
```

W5120 的门槛在 W3072/W4096 共同 1M 预算完整时优先使用这组结果，不再回退到
延长前的 500k 选择；其 50k range 从 W4096 已获批的学习率开始，只有 range
不稳定时才沿 `1e-4 → 3e-5 → 1e-5` 继续降低。

该审计要求 10 个完整 5k epoch、10% 以上训练 loss 降幅、末三个 epoch 每个动作
维度 std 至少 0.05、至少 80% epoch 的采样梯度非零、validation MSE 不恶化超过
5%，以及五次全地图评测的每图 30 episodes 完整。每图 episode seed 必须恰为
`0..29`，逐 episode 成功数必须与汇总一致；五个 checkpoint 的模型文件和
trainer-state sidecar、最终模型也必须齐全。任何条件失败都返回非零状态，不能
进入正式三 seed。确实观测到训练 update 后失败才记为 `unstable`，并按预注册
学习率顺序继续；若没有观测到任何 update，则记为 `invalid` 基础设施失败，
只能在相同学习率重跑，不能作为降低学习率的证据。`stable` / `unstable` /
`invalid` 判定、阈值和实测指标都会保存到 run 内的
`range_assessment.json`。该文件还
汇总实例化参数量、峰值 GPU allocated memory、峰值进程 RSS、末 epoch callback
时的累计 wall-clock 与据此计算的实际 updates/s，作为加入下一深度前资源预算
检查的直接证据。该 wall-clock 包含此前 checkpoint rollout，但不包含末 epoch
callback 之后执行的最终 rollout。

PointMaze `W3072` 正式 run 通过替换 architecture 文件启动，例如 seed 0：

```bash
micromamba run -n llm_offline_baselines python baseline_train.py --config \
  baselines/configs/base.pointmaze.yaml \
  baselines/configs/mlp_bc.yaml \
  baselines/experiments/bc_scaling_20260723/protocol.pointmaze.yaml \
  baselines/experiments/bc_scaling_20260723/optimizer.base.yaml \
  baselines/experiments/bc_scaling_20260723/architecture/plain_d4_w3072.yaml \
  baselines/experiments/bc_scaling_20260723/seed/s0.yaml \
  --experiment_id bc-scaling-formal-pointmaze-w3072-d4-s0-500k-<date>
```

AntMaze 1M run 必须叠加共同预算文件，例如 `W2048`、seed 0：

```bash
micromamba run -n llm_offline_baselines python baseline_train.py --config \
  baselines/configs/base.antmaze.yaml \
  baselines/configs/mlp_bc.yaml \
  baselines/experiments/bc_scaling_20260723/protocol.antmaze.yaml \
  baselines/experiments/bc_scaling_20260723/optimizer.base.yaml \
  baselines/experiments/bc_scaling_20260723/architecture/plain_d4_w2048.yaml \
  baselines/experiments/bc_scaling_20260723/budget/u1m.yaml \
  baselines/experiments/bc_scaling_20260723/seed/s0.yaml \
  --experiment_id bc-scaling-budget-antmaze-w2048-d4-s0-u1m-<date>
```

正式 run 完成后：

```bash
micromamba run -n llm_offline_baselines python \
  baselines/experiments/bc_scaling_20260723/audit_results.py --strict
micromamba run -n llm_offline_baselines python \
  baselines/experiments/bc_scaling_20260723/render_report.py
```
