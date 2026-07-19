# Baseline experiments

这个目录保存 conventional offline-RL baseline 的**可复现实验配置和运行脚本**。生成的
`baseline_runs/`、本地数据和 `reports/` 均被 `.gitignore` 忽略，不应作为实验配置
commit 的一部分。

所有训练都由 `baseline_train.py` 执行，并按从左到右的 YAML 覆盖顺序组合配置，例如：

```bash
micromamba run -n llm_offline_baselines python baseline_train.py --config \
  baselines/configs/base.antmaze.yaml \
  baselines/configs/td3_bc.yaml \
  baselines/experiments/paper_obs_v1/common.yaml \
  baselines/experiments/paper_obs_v1/antmaze.yaml \
  baselines/experiments/paper_obs_v1/td3_antmaze.yaml
```

## 实验脉络

1. **初始 16-layout 筛选**

   - `pointmaze16.sweep.yaml`、`antmaze16.sweep.yaml`：两个环境族的 16 个训练布局
     和完整 eval 布局；100k update、每变体 500 条 episode 的初筛协议。
   - `mlp_bc.{a,b,c}.yaml`、`iql.{a,b,c}.yaml`、`td3_bc.{a,b,c}.yaml`：第一轮算法
     候选超参数。
   - `run_short_sweeps.sh`：运行这轮 100k 筛选；`run_final_baselines.sh` 用选出的
     MLP-BC C、IQL C、TD3+BC A 跑 500k full-eval。选择依据写在脚本注释中：MLP 按
     validation action MSE，IQL 综合 rollout/offline 指标，TD3 选唯一未发散候选。
   - `*16.full_eval.yaml` 与 `final-500k.yaml` 只用于这一轮 finalist 的完整评估和
     500k 预算。

2. **`sparse_v2/`：扩大超参数筛选**

   初筛不足以判断数据量、网络大小、学习率和 actor-critic 专有参数的影响，因此这一轮
   在同一 16-layout 协议上分别改变：

   - `data*.yaml`：每变体保留 episode 数；`net*.yaml`：MLP 宽度/深度；
     `*.lr*.yaml`：学习率；`batch512.*.yaml`：batch size。
   - `iql.*.yaml`：reward、expectile、temperature；`td3.*.yaml`：TD3+BC 的 alpha。
   - `run_sparse_v2_sweeps.sh`：两个环境族的 100k screen；
     `run_antmaze_parallel.sh`：AntMaze 的并行版本；
     `run_finalists_parallel.sh`：把筛出的候选拉长到 250k/500k。

   这是探索性 sweep；它用于缩小候选范围，不能把其中事后最优 checkpoint 当成正式
   paper protocol。

3. **`paper_obs_v1/`：冻结的论文候选协议与预算曲线**

   这一轮固定 numeric `map + location + wall` observation（231D）、每变体 300 条
   episode、训练 seed 0、eval seed 20260716、每变体 100 个 rollout episode。它把
   筛选阶段的选择固定为可审计的 500k 主协议，并记录逐 episode 的 eval 结果。

   - `common.yaml`：上述共享预算、observation 和 eval 设置。
   - `pointmaze.yaml`、`antmaze.yaml`：各环境族的训练/eval variant 与 start-goal
     规则；`mlp_bc.yaml`、`iql.yaml`、`td3_pointmaze.yaml`、`td3_antmaze.yaml`：
     最终算法参数。
   - `run_all.sh`：六个 500k formal run；AntMaze MLP 会在其余任务释放数据内存后
     单独运行，避免 host RAM 峰值。
   - `run_checkpoint_rollouts.sh`、`evaluate_checkpoints.py`、
     `aggregate_checkpoint_rollouts.py`：补跑 100k 间隔 checkpoint，并审计完整曲线。
   - `one_million.yaml` / `two_million.yaml` 与 `run_antmaze_{1m,2m}.sh`：AntMaze
     从随机初始化独立训练到 1M/2M；相应 `*_checkpoint_rollouts.sh` 与
     `aggregate_antmaze_{1m,2m}.py` 审计预算延长曲线。2M 结果显示：MLP 在约 1M
     平台，TD3 有有限后续收益，IQL rollout 上升但 value 明显发散。
   - `audit_results.py`：500k formal result 的逐 episode 一致性审计。

4. **`umaze_only_v1/`：方向分布外控制实验**

   formal AntMaze 的 UMaze 在默认反向固定 eval pair 上均未成功，因此该控制实验只用
   UMaze 训练，保持 paper observation 和算法参数不变。

   - `umaze.yaml`、`run_all.sh`：单布局训练。
   - `evaluate_same_direction.py`：在离线数据覆盖的同向 pair `(3, 1) -> (1, 1)`
     上重评估。
   - `audit_results.py`：同时核对单布局和 16-layout 的方向对照。

   该对照用于区分“模型不会走迷宫”和“默认 eval 方向不在训练数据分布内”；它不是替换
   formal 16-layout 结果的主协议。

## 共享 observation 配置

`observation/map_location_wall.yaml` 是 map、当前位置/目标位置和四邻域 wall sensing 的
独立覆盖配置。`paper_obs_v1/common.yaml` 固化了同一组字段，避免正式 protocol 依赖
外部叠加文件。

## 运行约定

- 所有 `run_*.sh` 会跳过已完成的 run，并拒绝或跳过不完整目录，避免无意覆盖。
- 运行结果、checkpoint、逐 episode JSON 和报告只保存在忽略目录中；要复核具体数字，
  查看对应 run 的 `summary.json`、`checkpoint_rollouts/` 和本地 report。
- 新增实验时，优先增加一个小的覆盖 YAML；只有形成独立研究问题或固定 protocol 时，
  再增加新的子目录和运行脚本。
