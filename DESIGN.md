## 任务描述：LLM Offline RL 初步实验代码

### 项目目标
用预训练 LLM 在 D4RL 离线数据上做 behavior cloning（BC），以纯文本格式输入 obs、输出 action，验证 LLM 处理低维连续控制任务的能力，以及多任务联合训练带来的泛化能力。

---

### 技术栈
- **基座模型**：`Qwen/Qwen3-0.6B`（HuggingFace 加载，LoRA finetune）
- **数据集**：D4RL PointMaze 系列（`minari` 库加载）
- **训练框架**：PyTorch + HuggingFace Transformers + PEFT（LoRA）+ Unsloth（训练加速）

**依赖版本约束：**
- 当前环境验证为 `unsloth==2026.4.8` + `transformers==5.2.0`。早期 Unsloth 2026.3.x 曾与 transformers 5.x 的 `generate()` / cache 行为不兼容，需要临时降级到 4.56.1；该问题在当前环境中已解决，不再默认要求降级 transformers。

---

### PointMaze 变种完整列表

定义在 `data/pointmaze/variants.py` 的 `POINTMAZE_VARIANTS` 字典中，包含所有 8 个变种的 `dataset_id`、`env_id`、`prompt_vars`。

`maze_map` 和 `reward_type` 现在收在每个变种的 `prompt_vars` 中，供共享 prompt 渲染使用。

8 个变种及其信息：

```python
POINTMAZE_VARIANTS = {
    "open": {
        "dataset_id": "D4RL/pointmaze/open-v2",
        "env_id": "PointMaze_Open-v3",
        "maze_map": [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        "reward_type": "sparse",
    },
    "open-dense": {
        "dataset_id": "D4RL/pointmaze/open-dense-v2",
        "env_id": "PointMaze_OpenDense-v3",
        "maze_map": [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        "reward_type": "dense",
    },
    "umaze": {
        "dataset_id": "D4RL/pointmaze/umaze-v2",
        "env_id": "PointMaze_UMaze-v3",
        "maze_map": [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        "reward_type": "sparse",
    },
    "umaze-dense": {
        "dataset_id": "D4RL/pointmaze/umaze-dense-v2",
        "env_id": "PointMaze_UMazeDense-v3",
        "maze_map": [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        "reward_type": "dense",
    },
    "medium": {
        "dataset_id": "D4RL/pointmaze/medium-v2",
        "env_id": "PointMaze_Medium-v3",
        "maze_map": [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "reward_type": "sparse",
    },
    "medium-dense": {
        "dataset_id": "D4RL/pointmaze/medium-dense-v2",
        "env_id": "PointMaze_MediumDense-v3",
        "maze_map": [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "reward_type": "dense",
    },
    "large": {
        "dataset_id": "D4RL/pointmaze/large-v2",
        "env_id": "PointMaze_Large-v3",
        "maze_map": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "reward_type": "sparse",
    },
    "large-dense": {
        "dataset_id": "D4RL/pointmaze/large-dense-v2",
        "env_id": "PointMaze_LargeDense-v3",
        "maze_map": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "reward_type": "dense",
    },
}
```

---

### Prompt 设计

#### 核心原则（通用，适用于所有环境族）

- 共享风格模板按环境族存放在 `prompts/<env_family>/<prompt_name>.txt`，文件名 stem 就是 prompt 名
- 每个 variant 只在 `data/<env_family>/variants.py` 中维护自己的 `prompt_vars`，提供环境名、迷宫拓扑、迷宫可视化、结构说明等差异化信息
- 训练时使用 `prompt_templete_index` 指定的共享模板名，因此每个 timestep 产生“所选模板数”条训练样本
- 评估时固定使用模板 `0`，保证可复现
- 模板里可以引用 `prompt_vars` 中定义的任意字段以及运行时注入的动态字段；PointMaze 当前动态字段至少包括 `obs_text`、`map_sensing_en/zh`、`history_block_en/zh`
- 共享模板当前按“静态 Env Description 在前、动态 Current Status 在后”的结构组织，以提高前缀 cache 命中率
- 渲染出的共享模板文本只负责环境/任务语义；最终输入序列会再通过 tokenizer 自带的 `chat_template` 包装成 `user` / `assistant` 对话格式

#### 模板文件格式

共享模板是纯文本文件，例如：

```text
# prompts/<env_family>/0.txt
Environment: {env_name}
Maze:
{maze_visual}
## Current Status
Current observation:
{obs_text}
Action:
```

#### PointMaze 当前实现

- 当前 `prompts/pointmaze/` 下定义了 5 个共享模板：`0`–`2` 英文、`3`–`4` 中文
- `POINTMAZE_VARIANTS` 中的每个变种通过 `prompt_vars` 提供共享模板需要的静态字段，如 `env_name`、`maze_map`、`maze_shape`、`maze_visual`、`structure_desc_en`、`structure_desc_zh`
- PointMaze prompt 当前不再使用 reward 描述；`prompt 0` 也不再输出 raw matrix，只保留 visual maze
- target 文本仍由 `data/pointmaze/formatting.py` 定义，动作格式为紧凑的百分位整数，如 `35,-72`
- `format_obs(obs, meta)` 负责生成 `obs_text` 与动态 `map_sensing_en/zh`
- `format_history(history_entries, meta)` 负责生成可选历史块 `history_block_en/zh`
- 当历史块存在时，历史条目按时间从早到晚排列：第一条是最早采样到的历史 step，最后一条是当前 step 之前最近的采样历史 step

### 数据处理

- 每个 timestep 的 `(obs, [goal,] action)` 元组展开为多条训练样本，每条对应 `prompt_templete_index` 中指定的一个共享模板
- obs、goal 的序列化方式（精度、格式）由各环境族的 `formatting.py` 中的 `format_obs` 函数定义，结果填入模板占位符
- 如启用历史 prompt，训练数据会在同一 episode 内按 `t-1`、`t-1-history_stride`、... 回溯采样过去 transition，最多取 `history_num` 条，再通过 `format_history(...)` 注入 prompt
- action 的目标文本由动作编码模式决定：`text` 使用 `formatting.py` 中的 `format_action` 生成 `35,-72`；`bin` / `gaussian_bin` 使用共享特殊 token `<act_XX>` 表示离散动作 bin
- 训练 tokenization 不再直接编码 `prompt + action_text`；而是将渲染后的 prompt 作为 `user` 消息、`action_text` 作为 `assistant` 消息，通过模型原生 `chat_template` 构造最终 sequence
- `gaussian_bin` 会额外在 dataset 中记录 `action_bin_labels`，动作 token 位置使用高斯 soft-label CE；若设置 `action_soft_label_radius`，则每个动作位置只在中心 bin 及左右 n 个相邻 bin 上做 softmax，窗口外 action token 不产生梯度。chat-template 结束 token 等非动作 assistant token 仍使用普通 CE
- train/val 划分在 **episode 级别**进行：先按 `episode_keep_num` 随机无放回抽样一个 episode pool（如果真实 episode 数更少则使用全部），再在该 pool 内按 `floor(pool_size * train_data_ratio)` 划分 train，剩余 episodes 作为 val，避免同一 episode 同时出现在 train 和 val 中
- 每个 episode 的第一个 timestep 没有历史；评估 rollout 中也同样如此，只有一步实际动作执行完成后才会写入在线 history buffer

---

### 训练模式

通过 `config.yaml` 控制，该文件包含所有训练相关配置：

```yaml
# 环境与任务
env_family: pointmaze
train_mode: single       # single | all | except
train_varients: [open]   # single: 恰好一个；all: 指定子集或留空表示全部；except: 排除列表
history_num: 0           # 采样多少条历史 transition 注入 prompt；0 = 关闭历史
history_stride: 1        # 每隔多少步采样一条历史

# 基座模型
model_name: Qwen/Qwen3-0.6B   # 任意 HuggingFace causal LM
prompt_templete_index: ["0"]  # 使用的 prompt 文件名（不含 .txt）

# 训练超参数
learning_rate: 1e-4
num_epochs: 3
batch_size: 32
max_length: 512

# LoRA 参数
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: ["q_proj", "v_proj"]

# 评估辅助
parse_retry_limit: 3     # action 解析失败时的最大重试次数
action_token_mode: text  # text | bin | gaussian_bin
action_num_bins: 10      # bin 模式下的共享动作 token 数
action_bin_min: -1.0
action_bin_max: 1.0
action_soft_label_sigma: 1.0  # gaussian_bin 的高斯宽度，单位是 bin index
action_soft_label_radius: 2   # gaussian_bin 的局部训练窗口，中心 bin 左右各 n 个

# Debug（注释掉为正常训练）
# max_data_num: 100      # 每个 dataset split 最多使用多少条样本；注释掉 = 全量数据
episode_keep_num: 5000  # 参与 train/val 划分的最大 episode 数；真实 episode 更少时使用全部，只作用于未命中 cache 的 offline dataset 构建
balance_variant_episode_count: false  # 多 variant 时是否把 sampled episode pool 对齐到最小 variant
sampling_seed: 0         # 控制 episode 随机抽样的可复现性
```

对 PointMaze 环境族，共训练以下 9 个模型：
- 单变种模型 × 8（每个变种独立训练）
- 全变种联合模型 × 1

联合训练时各变种数据按样本数加权采样，避免大变种压制小变种。

---

### 路径约定

项目中所有持久化数据遵循统一的路径规范，便于不同实验结果的对比与管理。

---

#### 1. Checkpoint 路径

训练完成后，LoRA adapter 权重、tokenizer、`config.yaml` 副本保存在：

```
checkpoints/
└── <env_family>/
    └── <model_slug>/              # HuggingFace model ID 的最后一段，如 Qwen3-0.6B
        └── <selection_tag>/       # 训练选择标签：单变种名、all、或 except-<excluded variants>
            └── <experiment_id>/   # 自动生成或手动指定的实验 ID
                ├── ep1/           # epoch 1 结束时保存的中间 checkpoint
                ├── ep2/
                ├── ep3/
                └── final/         # 训练全部结束后保存（与最后一个 epN 内容相同）
                    ├── adapter_config.json
                    ├── adapter_model.safetensors
                    ├── tokenizer.json
                    ├── tokenizer_config.json
                    └── config.yaml
```

**示例：**

| 训练场景 | 路径 |
|----------|------|
| Qwen3-0.6B 单独训练 open 变种（epoch 2 中间） | `checkpoints/pointmaze/Qwen3-0.6B/open/<experiment_id>/ep2/` |
| Qwen3-0.6B 单独训练 open 变种（训练完成） | `checkpoints/pointmaze/Qwen3-0.6B/open/<experiment_id>/final/` |
| Qwen3-0.6B 联合训练所有变种 | `checkpoints/pointmaze/Qwen3-0.6B/all/<experiment_id>/final/` |
| Qwen3-0.6B 使用 except 模式排除 `large` 和 `large-dense` | `checkpoints/pointmaze/Qwen3-0.6B/except-large+large-dense/<experiment_id>/final/` |
| Llama-3.2-1B 单独训练 umaze 变种 | `checkpoints/pointmaze/Llama-3.2-1B/umaze/<experiment_id>/final/` |

`model_slug` 由 `model/policy.py` 的 `get_model_slug()` 生成（取 `/` 后的部分），不同基座模型的实验不会相互覆盖。`selection_tag` 已经包含训练选择语义，因此路径里不再单独重复 `train_mode`。

训练 batch 进度不再通过终端 carriage return 渲染，而是写入单行快照文件 `progress/<uuid>.txt`；`train.py` 启动后打印该路径，正常结束时删除文件，异常退出时保留最后一次进度。

---

#### 2. 数据集缓存路径

Tokenize 后的数据集缓存在 `dataset_cache_dir`（由 `config.yaml` 配置，默认 `dataset_cache/`）：

```
dataset_cache/
├── <env_family>-<variant>-train-prompts-<prompt_names>-hist<H>-stride<S>-split<train_pct>-action-<mode>-bins<B>-range<min>to<max>.pkl
├── <env_family>-<variant>-train-prompts-<prompt_names>-hist<H>-stride<S>-split<train_pct>-action-<mode>-bins<B>-range<min>to<max>.jsonl
├── <env_family>-<variant>-val-prompts-<prompt_names>-hist<H>-stride<S>-split<train_pct>-action-<mode>-bins<B>-range<min>to<max>.pkl
└── <env_family>-<variant>-val-prompts-<prompt_names>-hist<H>-stride<S>-split<train_pct>-action-<mode>-bins<B>-range<min>to<max>.jsonl
```

**示例：** `dataset_cache/pointmaze-open-train-prompts-0+3-hist4-stride1-split95-action-text-bins10-range-1to1.pkl`

- `.pkl` 用于快速加载（下次训练直接跳过 tokenize，节省约 10 分钟）
- `.jsonl` 每行是 `{"prompt": "...", "action": "35,-72"}` 或 `{"prompt": "...", "action": "<act_03><act_48>"}`，供人工抽检数据质量
- 若 `config.yaml` 中未设置 `dataset_cache_dir`（注释掉），则不缓存，每次重新 tokenize
- `max_data_num` 截断发生在 cache 读取之后的内存中，cache 文件始终保存完整数据
- 不同 `prompt_templete_index`、`history_num/history_stride` 和 action 编码配置会写入不同 cache 文件名，避免不同 prompt、历史或动作 tokenization 配置误复用同一份 tokenized 数据
- `episode_keep_num` / `balance_variant_episode_count` / `sampling_seed` 只在未命中 cache 时参与 episode 抽样；命中现有 cache 时会直接复用 `.pkl`，并打印这些设置本次未生效

---

#### 3. 评估结果路径

评估结果统一保存在 `result_root`（默认 `results/`）下，路径编码了"用哪个模型"、"训练背景"、"是训练期还是独立评估"、"在哪个变种上评估"四层信息：

```
<result_root>/
└── <model_slug>/
    └── train=<env_family>-<selection_tag>/
        └── exp=<experiment_id>/
            ├── epoch_<n>/
            │   └── eval=<env_family>-<variant>/
            │       ├── result.json
            │       └── episode_<n>/
            │           ├── rollout.gif|mp4
            │           └── steps/
            │               └── step_<n>.txt
            └── standalone_<eval_uuid>/
                └── eval=<env_family>-<variant>/
                    ├── result.json
                    └── episode_<n>/
                        ├── rollout.gif|mp4
                        └── steps/
                            └── step_<n>.txt
```

**路径字段说明：**

| 字段 | 含义 | 示例 |
|------|------|------|
| `model_slug` | 基座模型名 | `Qwen3-0.6B` |
| `selection_tag` | 训练选择标签 | `open`、`all`、`except-large+large-dense` |
| `result_root` | 结果根目录配置项 | `results`、`resultsV2` |
| `variant` | 当前评估变种名 | `open`、`umaze`、`medium` |
| `epoch_<n>` / `standalone_<eval_uuid>` | 区分训练期中间评估与独立评估运行 | `epoch_2`、`standalone_ab12cd34` |

**示例路径：**

| 场景 | 路径 |
|------|------|
| 训练 open 变种 epoch 2 中间评估 open | `results/Qwen3-0.6B/train=pointmaze-open/exp=<experiment_id>/epoch_2/eval=pointmaze-open/result.json` |
| except 模式排除 `large` 和 `large-dense` 后训练，并在 epoch 1 评估 medium | `results/Qwen3-0.6B/train=pointmaze-except-large+large-dense/exp=<experiment_id>/epoch_1/eval=pointmaze-medium/result.json` |
| standalone 评估 open | `results/Qwen3-0.6B/train=pointmaze-open/exp=<experiment_id>/standalone_<eval_uuid>/eval=pointmaze-open/result.json` |
| 评估未微调的原始基座模型 | `results/Qwen3-0.6B/train=pretrained/standalone_<eval_uuid>/eval=pointmaze-open/result.json` |

`evaluate.py` 和 `train.py` 使用同一套基础路径语义，均以单个 `variant` 作为 `eval=<...>` 目录粒度。训练期评估通过 `epoch_<n>` 区分不同轮次，standalone `evaluate.py` 通过 `standalone_<eval_uuid>` 区分不同次独立运行。每个 `episode_<n>` 目录同时保存 rollout 视频和逐步文本日志，其中 `steps/step_<n>.txt` 记录渲染后的 prompt、模型原始输出、最终执行动作、parse 状态和尝试次数；`gaussian_bin` 且 `record_step_logs=true` 时还会记录每个动作维度上所有 bin token 的生成概率。

**结果文件字段：**

```json
{
  "variant": "open",
  "num_episodes": 20,
  "mean_return": 0.85,
  "std_return": 0.12,
  "success_rate": 0.80,
  "total_parse_failures": 2,
  "total_fallbacks": 0,
  "mean_action_time_ms": 241.3,
  "train_loss": 0.4637,   // 仅 result_epN.json 有
  "val_loss": 0.4702      // 仅 result_epN.json 有
}
```

---

### 评估

```bash
python evaluate.py --config eval.yaml
```

通过 `eval.yaml` 控制所有评估配置：

```yaml
model_path: checkpoints/pointmaze/Qwen3-0.6B/open/<experiment_id>/final
env_family: pointmaze
eval_mode: single       # single | all | except
variants: [open]         # single: 恰好一个；all: 指定子集或留空表示全部；except: 排除列表
num_episodes: 20
parse_retry_limit: 3
env_kwargs:
  continuing_task: false # false = 每 episode 一个目标，到达即结束
  # max_episode_steps: 300  # 可选，覆盖环境默认值
```

`model_path` 可填 checkpoint 路径或 HuggingFace model ID（如 `Qwen/Qwen3-0.6B`），后者用于评估未微调的基座模型。评估时固定使用模板 0（第一个英文模板）。

---

### 代码结构

```
project/
├── config.yaml
├── prompts/
│   └── <env_family>/            # 每个环境族一个子目录
│       └── <prompt_name>.txt    # 共享 prompt 模板，文件名 stem 是 prompt 名
├── data/
│   ├── <env_family>/            # 每个环境族一个子目录
│   │   ├── variants.py          # 该族所有变种的元信息字典
│   │   ├── dataset.py           # 数据加载、tokenize（每 timestep 展开5条）
│   │   └── formatting.py        # obs 序列化/附加 prompt 变量、action 目标文本生成、action 解析
│   ├── base_dataset.py          # 抽象基类，定义通用接口（load、format、tokenize）
│   └── registry.py              # 环境族注册表，按 env_family 路由 dataset 和 formatter
├── model/
│   └── policy.py                # 模型加载（从 config 读取 model_name）、LoRA 设置
├── train.py                     # 训练入口，读取 config 决定训练模式
├── evaluate.py                  # Rollout 评估
├── checkpoints/
├── results/
└── utils/
    └── prompt_loader.py         # 加载指定环境族的共享模板，返回列表
```

**`data/<env_family>/formatting.py` 接口规范**（每个环境族必须实现）：

```python
def format_obs(obs, meta) -> dict:
    """返回 prompt 渲染变量字典；必须包含 obs_text，其他字段可自定义"""

def format_action(action) -> str:
    """将 action 向量序列化为训练 target 文本"""

def parse_action(text) -> tuple[np.ndarray, bool]:
    """从模型输出文本中解析 action 向量；返回 (action, success)"""

def validate_action(action) -> bool:
    """校验解析出的 action 是否在合法范围内"""
```

`registry.py` 同时暴露 `get_dataset(env_family)` 和 `get_formatter(env_family)`，让 `train.py` 和 `evaluate.py` 只需传入 `env_family` 即可自动路由。新增环境族时在 `data/` 下新建子文件夹（含 `formatting.py`）并在 `registry.py` 注册一行即可，`train.py` 和 `evaluate.py` 无需改动。

---

### 关键实现细节

1. **Loss masking**：labels 中 `user` turn 与 assistant 前缀部分设为 `-100`，只训练 assistant 动作文本及其结束标记
2. **每 timestep 按所选 prompt 展开**：`dataset.py` 构造数据时对每个 timestep 遍历 `prompt_templete_index` 指定的共享模板名，生成对应数量的独立样本
3. **Action parsing（环境族绑定）**：evaluate.py 通过 `registry.get_formatter(env_family)` 获取该族的 `parse_action` 和 `validate_action`。若解析失败或校验不通过，最多重新让模型生成 `parse_retry_limit` 次（来自 `eval.yaml`）。若达到上限仍失败，fallback 到零向量。全程记录 parse 失败次数和 fallback 次数作为辅助指标。
   - *PointMaze 实现*：正则解析紧凑的百分位整数格式 `35,-72`，除以 100 后校验各分量在 `[-1, 1]` 内，clip 后返回
   - 其他环境族在各自 `formatting.py` 中实现对应逻辑，格式和校验规则完全自定义
4. **Obs/Action 序列化（环境族绑定）**：`dataset.py` 调用同族 `formatting.py` 的 `format_obs` 和 `format_action`，不依赖任何全局 formatting 工具
   - *PointMaze 实现*：`format_obs(obs, meta)` 接收环境观测对象（当前为 dict），返回 `obs_text` 以及动态 `map_sensing_en` / `map_sensing_zh`
   - `map_sensing` 会直接给出当前位置格子、目标格子，以及上下左右相邻格子的 `wall/free` 状态；行列从左上角开始按 1-based 计数
5. **Episode 级别 train/val 划分**：先按 `episode_keep_num` 随机抽样 episode pool（真实 episode 更少时使用全部），再在 pool 内按 `floor(pool_size * train_data_ratio)` 划分 train，剩余作为 val，防止数据泄露
6. **多变种混合采样**：联合训练时按各变种样本数加权，保证各变种均匀覆盖
7. **新环境族扩展**：在 `prompts/` 新建目录、`data/` 下新建子文件夹（含 `variants.py`、`dataset.py`、`formatting.py`）、`registry.py` 注册一行，`train.py` 和 `evaluate.py` 无需改动

---

### 暂不需要实现
- Return-conditioning
- Online RL 组件
- 多 GPU 分布式训练
