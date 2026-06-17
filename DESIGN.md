## 任务描述：LLM Offline RL 初步实验代码

### 项目目标
用预训练 LLM 在 D4RL 离线数据上做 behavior cloning（BC），以纯文本格式输入 obs、输出 action，验证 LLM 处理低维连续控制任务的能力，以及多任务联合训练带来的泛化能力。

---

### 技术栈
- **基座模型**：`Qwen/Qwen3-0.6B`（HuggingFace 加载，LoRA finetune）
- **数据集**：D4RL PointMaze 与 AntMaze 系列（`minari` 库加载）
- **训练框架**：PyTorch + HuggingFace Transformers + PEFT（LoRA）+ Unsloth（训练加速）

**依赖版本约束：**
- 当前环境验证为 `unsloth==2026.4.8` + `transformers==5.2.0`。早期 Unsloth 2026.3.x 曾与 transformers 5.x 的 `generate()` / cache 行为不兼容，需要临时降级到 4.56.1；该问题在当前环境中已解决，不再默认要求降级 transformers。

---

### PointMaze 变种完整列表

定义在 `data/pointmaze/variants.py` 的 `POINTMAZE_VARIANTS` 字典中。当前 registry 包含 8 个 remote D4RL 变种、`local-layout-01..13` 和 `test-layout-01..03`；remote 使用 `dataset_id` / `env_id`，local/test 使用 `dataset_path` / `env_paras`，全部通过 `prompt_vars` 提供 prompt metadata。

`maze_map` 和 `reward_type` 现在收在每个变种的 `prompt_vars` 中，供共享 prompt 渲染使用。

以下代码块列出 8 个 remote D4RL 基础变种；local/test 地图以 `_LOCAL_LAYOUT_*` / `_TEST_LAYOUT_*` 常量和 `_build_local_variant()` 注册：

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
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
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
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "reward_type": "dense",
    },
}
```

---

### AntMaze 官方 D4RL 变种

`data/antmaze/variants.py` 注册 Minari 当前提供的 6 个官方数据集：

| variant | Minari dataset | rollout env |
|---|---|---|
| `umaze` | `D4RL/antmaze/umaze-v1` | `AntMaze_UMaze-v4` |
| `umaze-diverse` | `D4RL/antmaze/umaze-diverse-v1` | `AntMaze_UMaze-v4` |
| `medium-play` | `D4RL/antmaze/medium-play-v1` | `AntMaze_Medium-v4` |
| `medium-diverse` | `D4RL/antmaze/medium-diverse-v1` | `AntMaze_Medium_Diverse_GR-v4` |
| `large-play` | `D4RL/antmaze/large-play-v1` | `AntMaze_Large-v4` |
| `large-diverse` | `D4RL/antmaze/large-diverse-v1` | `AntMaze_Large_Diverse_GR-v4` |

AntMaze 必须保持 Minari metadata 中的 v4 数据契约：`observation` 为 27 维本体状态，另有 2 维 `achieved_goal` / `desired_goal`，action 为 8 维关节 torque。不能直接替换成 v5 默认环境，因为 v5 默认包含 contact-force observation，会改变输入维度。每个 variant 的 `env_kwargs` 保存官方 eval map、稀疏奖励、horizon 对应的 continuing/reset 语义；训练期或 standalone eval 可用配置中的 `env_kwargs` 覆盖 `continuing_task`。官方 eval map 与离线采集 map 不保证相同，例如 UMaze 的中间墙朝向不同，因此 rollout 创建环境后会从 `env.unwrapped.maze` 刷新 prompt 使用的 map、visual map 和 `maze_size_scaling`。

当前只支持上述官方地图，不包含本地自定义 AntMaze 数据生成或地图注册。`score.py` 仍是 PointMaze-only；AntMaze 当前支持训练和普通 return/success-rate rollout eval。

---

### Prompt 设计

#### 核心原则（通用，适用于所有环境族）

- 共享风格模板按环境族存放在 `prompts/<env_family>/<prompt_name>.txt`，文件名 stem 就是 prompt 名
- 每个 variant 只在 `data/<env_family>/variants.py` 中维护自己的 `prompt_vars`，提供环境名、迷宫拓扑、迷宫可视化、结构说明等差异化信息
- 训练时使用 `prompt_templete_index` 指定的共享模板名，因此每个 timestep 产生“所选模板数”条训练样本
- 训练期评估默认使用训练 prompt 列表中的第一个模板；standalone eval 默认使用 checkpoint config 中记录的第一个训练 prompt。`eval.yaml` 可用单个 `prompt_templete_index` 覆盖 standalone eval prompt；若覆盖值不在 checkpoint 训练 prompt 列表中，`evaluate.py` 会强警告并要求输入 `Y`，或通过 `-y/--yes` 自动确认
- 模板里可以引用 `prompt_vars` 中定义的任意字段以及运行时注入的动态字段；PointMaze 与 AntMaze 当前都提供 `obs_text`、`location_sensing_en/zh`、`wall_sensing_en/zh` 和 history block
- PointMaze action-bin prompt 使用 `bin_full_sensing`、`bin_loca_sensing`、`bin_wall_sensing`、`bin_no_sensing`，由 `bin`、`gaussian_bin` 和 `mtp_bin` 共享
- PointMaze 连续动作 prompt 使用去模式化命名：`parallel_full_sensing`、`parallel_loca_sensing`、`parallel_wall_sensing`、`parallel_no_sensing`，由 `parallel_l1`、`parallel_gaussian` 和 `parallel_t` 共享；当前 `config.yaml` 默认使用 `parallel_full_sensing` + `parallel_l1`
- AntMaze 保留 `0` 作为 text prompt；action-bin 使用 `bin_full_sensing`、`bin_loca_sensing`、`bin_wall_sensing`、`bin_no_sensing`，continuous 使用 `parallel_full_sensing`、`parallel_loca_sensing`、`parallel_wall_sensing`、`parallel_no_sensing`
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

- 当前 `prompts/pointmaze/` 下定义了 legacy `0`–`4` 文本模板、`bin_*_sensing` action-bin 模板和 `parallel_*_sensing` 连续动作模板
- `POINTMAZE_VARIANTS` 中的每个变种通过 `prompt_vars` 提供共享模板需要的静态字段，如 `env_name`、`maze_map`、`maze_shape`、`maze_visual`、`structure_desc_en`、`structure_desc_zh`
- PointMaze prompt 当前不再使用 reward 描述；`prompt 0` 也不再输出 raw matrix，只保留 visual maze
- text 模式 target 文本由 `data/pointmaze/formatting.py` 定义，动作格式为紧凑的百分位整数，如 `35,-72`；bin / gaussian_bin / mtp_bin 由共享 action-bin codec 负责离散化、model token 映射和 display text
- `format_obs(obs, meta)` 负责生成 `obs_text` 与动态 `location_sensing_en/zh`、`wall_sensing_en/zh`
- `format_history(history_entries, meta)` 负责生成可选历史块 `history_block_en/zh`
- 当历史块存在时，历史条目按时间从早到晚排列：第一条是最早采样到的历史 step，最后一条是当前 step 之前最近的采样历史 step

#### AntMaze 当前实现

- `format_obs` 展开 torso xy/goal、torso height、四元数、8 个 joint angle、torso linear/angular velocity 和 8 个 joint velocity
- `format_obs` 使用 `achieved_goal` 作为 torso xy，以 `maze_size_scaling=4.0` 映射到 1-based 行列，并与 PointMaze 复用 `utils/maze_sensing.py` 的 location/wall sensing
- bin 和 parallel prompt 均提供 full、location-only、wall-only、no-sensing 四种变体，命名与 PointMaze 一致
- text action 是 8 个逗号分隔的整数百分位，actuator 顺序为 back-right hip/ankle、front-left hip/ankle、front-right hip/ankle、back-left hip/ankle
- history 仅保留过去 torso xy、对应格子和实际执行动作，避免把 27 维本体状态重复塞入 prompt
- `data/antmaze/dataset.py` 复用参数化后的 goal-maze episode cache/tokenization 管线，因此支持 episode split、partition cache、history、全部 action token mode 和多进程 tokenization

### 数据处理

- 每个 timestep 的 `(obs, [goal,] action)` 元组展开为多条训练样本，每条对应 `prompt_templete_index` 中指定的一个共享模板
- obs、goal 的序列化方式（精度、格式）由各环境族的 `formatting.py` 中的 `format_obs` 函数定义，结果填入模板占位符
- 如启用历史 prompt，训练数据会在同一 episode 内按 `t-1`、`t-1-history_stride`、... 回溯采样过去 transition，最多取 `history_num` 条，再通过 `format_history(...)` 注入 prompt
- action 的目标文本由动作编码模式决定：`text` 使用 `formatting.py` 中的 `format_action` 生成 `35,-72`；`bin` / `gaussian_bin` / `mtp_bin` 使用离散 action bin。默认 `new_token: false` 时，模型内部复用 tokenizer 词表末尾筛选出的稳定低频 token ID；jsonl、step log 和 history prompt 中的人类可读显示仍统一为 `<act_XX>`。`mtp_bin` 的 AQT 不进入 tokenizer，而是由 `mtp_bin_decoder.pt` 保存可训练 embedding 和 sampler head
- 训练 tokenization 不再直接编码 `prompt + action_text`；text/bin/gaussian_bin 将渲染后的 prompt 作为 `user` 消息、`action_text` 作为 `assistant` 消息，通过模型原生 `chat_template` 构造最终 sequence；`mtp_bin` 构造 generation prompt、action prefix token 和 AQT metadata
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
gradient_accumulation_steps: 1
max_length: 512
dataloader_config:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  non_blocking: true
resource_monitor_enabled: false
resource_monitor_interval_seconds: 1.0

# LoRA 参数
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: ["q_proj", "v_proj"]
lora_layers_to_transform: null  # 可选；如 [3, 7, 11, 15, 19, 23] 只在这些 decoder layer 注入 LoRA

# 评估辅助
parse_retry_limit: 3     # action 解析失败时的最大重试次数
eval_step_interval: 0    # 0 = dataloader 构建后交互式提示；非交互运行保持关闭
step_eval_skip: 1        # 1 = 每次 step eval 都完整评估；n = 每个 epoch 内每第 n 次触发才跑 val/rollout，其余只保存 checkpoint
action_sampling: false   # 生成式/bin 模式 true = 采样 / false = greedy；parallel_gaussian/parallel_t true = 策略采样 / false = mean action
action_temperature: 1.0
action_top_p: 1.0
action_top_k: 0          # 0 = 不启用 top-k 截断
action_token_mode: text  # text | bin | gaussian_bin | mtp_bin | parallel_l1 | parallel_gaussian | parallel_t
action_num_bins: 10      # action-bin 模式下的共享动作 token 数
mtp_k: null              # mtp_bin only；null = action_dim - 1
mtp_lcm_weight: 1.0      # mtp_bin latent consistency matching weight
mtp_quadratic_decoding: true  # mtp_bin eval；false = 直接信任第一次 MTP NTP+AQT proposal
new_token: false         # false = 内部复用低频 token ID；true = 新增 <act_XX> special tokens
action_bin_min: -1.0
action_bin_max: 1.0
action_soft_label_sigma: 1.0  # gaussian_bin 的高斯宽度，单位是 bin index
action_soft_label_radius: 2   # gaussian_bin 的局部训练窗口，中心 bin 左右各 n 个
gaussian_log_std_init: -1.0   # parallel_gaussian 的 state-independent log std 初始值
gaussian_log_std_min: -5.0    # parallel_gaussian 的 log std、parallel_t 的 log scale 下界
gaussian_log_std_max: 1.0     # parallel_gaussian 的 log std、parallel_t 的 log scale 上界
student_t_df: 3.0             # parallel_t 的 Student-t 自由度
continuous_mean_l1_weight: 0.1 # parallel_t 的 mean L1 辅助项权重，0 = 关闭
action_head_dropout: 0.0      # continuous action MLP head 内部 dropout，仅训练模式生效
action_head_weight_decay: 0.0 # 仅 continuous action MLP Linear weights 的 AdamW weight decay

# Debug（注释掉为正常训练）
# max_data_num: 100      # 每个 dataset split 最多使用多少条样本；注释掉 = 全量数据
dataset_load_partitions: 1  # >1 时只分区 tokenize/load train tokenized 数据；需要 dataset_cache_dir
episode_keep_num: 5000  # 参与 train/val 划分的最大 episode 数；真实 episode 更少时使用全部，cache 命中后仍会重新生效
balance_variant_episode_count: false  # 多 variant 时是否把 sampled episode pool 对齐到最小 variant
sampling_seed: 0         # 控制 episode 随机抽样的可复现性
eval_seed: 1             # 训练期 eval 的 episode reset seeds 为 eval_seed, eval_seed+1, ...
```

PointMaze 训练集合不再固定为“8 个单变种 + 1 个联合模型”，而是由 `train_mode` 和 `train_varients` 从当前 registry 中选择。联合训练时各变种数据按样本数加权采样，避免大变种压制小变种。

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
                ├── step10002/     # 可选：训练 batch step 触发的中间 checkpoint
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
| Qwen3-0.6B 单独训练 open 变种（batch step 10002 中间） | `checkpoints/pointmaze/Qwen3-0.6B/open/<experiment_id>/step10002/` |
| Qwen3-0.6B 单独训练 open 变种（训练完成） | `checkpoints/pointmaze/Qwen3-0.6B/open/<experiment_id>/final/` |
| Qwen3-0.6B 联合训练所有变种 | `checkpoints/pointmaze/Qwen3-0.6B/all/<experiment_id>/final/` |
| Qwen3-0.6B 使用 except 模式排除 `large` 和 `large-dense` | `checkpoints/pointmaze/Qwen3-0.6B/except-large+large-dense/<experiment_id>/final/` |
| Llama-3.2-1B 单独训练 umaze 变种 | `checkpoints/pointmaze/Llama-3.2-1B/umaze/<experiment_id>/final/` |

`model_slug` 由 `model/policy.py` 的 `get_model_slug()` 生成（取 `/` 后的部分），不同基座模型的实验不会相互覆盖。`selection_tag` 已经包含训练选择语义，因此路径里不再单独重复 `train_mode`。

训练 batch 进度不再通过终端 carriage return 渲染，而是写入 run 级别的单行快照文件 `progress/<experiment_id>.txt`；`train.py` 启动训练 loop 时打印一次路径，跨 epoch 持续覆盖更新，训练成功完成最终 checkpoint/barrier 后打印最终进度行并删除该文件，异常退出时保留最后一次进度。`dataset_load_partitions > 1` 时，每个 train shard 加载前会把同一 progress 文件刷新为 `loading data partition i/N` 状态。

可选的系统资源监控由 `resource_monitor_enabled` 控制，默认关闭；启用后 rank0 启动轻量后台线程，每 `resource_monitor_interval_seconds` 秒覆盖写 `sys_info/<experiment_id>.txt`，记录当前 RAM/swap 和所有 GPU 的显存、利用率、温度、功耗。RAM/swap 来自 `/proc/meminfo`，GPU 来自 `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`；GPU 查询失败只在文件中记录 `gpu_error`，不影响训练。DDP 下只由 rank0 采样整机状态，避免多 rank 重复写同一份机器信息；该文件是 latest-only，不保存历史序列，也不写入 W&B。

---

#### 2. 实验配置快照路径

训练启动时，`train.py` 会在模型加载、dataset 构建和正式训练之前保存一次完整运行配置快照：

```
exp_configs/
└── <experiment_id>/
    ├── config.yaml
    ├── git.yaml
    └── dirty.patch
```

`config.yaml` 在原始配置基础上包含运行时解析字段，例如 `experiment_id`、`train_config_source`、`resolved_train_variants`、`train_selection_tag`、`resolved_eval_variants`、`action_dim`、continuous action head 参数、`world_size` 和 `global_effective_batch_size`。`git.yaml` 记录当前 repo root、branch、HEAD commit/subject/date、`git status --porcelain`、dirty 状态、patch 大小和 sha256，以及未写入 patch 的二进制文件。`dirty.patch` 是当前工作区相对 HEAD 的文本 patch，包含 tracked 文本改动和 untracked 文本文件；ignored 文件和二进制文件不写入 patch。恢复实验源码状态的固定流程是先 `git checkout <head_commit>`，再 `git apply exp_configs/<experiment_id>/dirty.patch`。DDP 下只有 rank0 写入，其他 rank 在 barrier 后继续。

---

#### 3. 数据集缓存路径

Tokenize 后的数据集缓存在 `dataset_cache_dir`（由 `config.yaml` 配置，默认 `dataset_cache/`）：

```
dataset_cache/
├── <cache_signature_hash>.pkl
└── <cache_signature_hash>.jsonl
```

**示例：** `dataset_cache/9f1a2b3c4d5e6f708192a0b1c2d3e4f5.pkl`

- `.pkl` 用于快速加载（下次训练直接跳过 tokenize，节省约 10 分钟）
- `.pkl` 按 `episode_idx -> tokenized samples` 保存；非分区模式仍使用兼容旧 hash 的共享 train/val cache，分区模式则使用 train shard cache 和单独的完整 val cache
- `.jsonl` 每行包含 `episode_idx`、`timestep`、`prompt` 和 `action`，供人工抽检数据质量
- 若 `config.yaml` 中未设置 `dataset_cache_dir`（注释掉），则不缓存，每次重新 tokenize
- `dataset_load_partitions > 1` 时必须设置 `dataset_cache_dir`。原始轨迹仍按现有逻辑一次性加载并完成 episode 级 train/val selection；随后每个 variant 的 train episode 会按 `sampling_seed` 确定性打乱并切成固定 shard，每次只 tokenize/load 一个 train tokenized shard，训练完释放后再加载下一个 shard。val split 不分区，启动时构建一次完整 val loader 并在训练期间复用。一个 epoch 仍表示跑完所有 train shard，epoch 间只打乱 shard 访问顺序，shard cache 可稳定复用。
- `episode_keep_num`、`train_data_ratio`、`sampling_seed` 和 `balance_variant_episode_count` 不写入 cache 文件名；cache 命中后会重新按当前配置选择 episode 并切分 train/val
- 如果现有 cache 不覆盖当前 sampled episodes，则忽略旧 cache，重新 tokenize 当前 sampled pool 并覆盖同一个 variant 级 cache
- `max_data_num` 截断发生在最终 dataset 组装之后，只影响本次训练返回的数据，不影响 cache 内容和 cache 命中判断
- cache 文件名是 32 位 sha256 前缀；hash payload 包含 variant/data signature、tokenizer/max length、`prompt_templete_index` 解析后的 prompt 名称、prompt 模板内容、variant prompt vars、`history_num/history_stride` 和 action 编码配置，避免不同 tokenization 或 prompt 配置误复用同一份 tokenized 数据。源码文件 hash 不进入 payload；若代码改动影响 tokenization 语义，需要手动删除旧 cache。
- 分区模式下，train shard cache hash payload 额外包含 split、`dataset_partition_count` 和 `dataset_partition_index`；完整 val cache 只包含 `split: val`，不包含 partition count/index；`dataset_load_partitions: 1` 保持旧 hash payload 兼容
- action-bin 模式下，cache signature payload 和 metadata 额外记录 `new_token`、`mtp_k` 与 `action_token_schema_hash`。该 hash 由 `new_token`、真实 ABT token ids 和 display tokens 计算得到；若 cache metadata 与当前 signature 不一致，加载阶段直接报错，避免把旧 action-token 映射下的 tokenized samples 用到新训练里
- `.jsonl` 中的 `action` 永远使用 display text，例如 `<act_24><act_37>`；`mtp_bin` 还会记录 `action_query`，用于查看 AQT 显示标记。`.pkl` 中保存的 `input_ids` 和 AQT metadata 才是模型实际训练使用的数据。`new_token: false` 时 display text 与真实 token ids 不是同一组文本 token

---

#### 4. 评估结果路径

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
            │           ├── rollout_global.gif|mp4  # AntMaze only
            │           └── steps.txt
            ├── step<n>/
            │   └── eval=<env_family>-<variant>/
            │       ├── result.json
            │       └── episode_<n>/
            │           ├── rollout.gif|mp4
            │           ├── rollout_global.gif|mp4  # AntMaze only
            │           └── steps.txt
            └── standalone_<eval_uuid>/
                └── eval=<env_family>-<variant>/
                    ├── result.json
                    └── episode_<n>/
                        ├── rollout.gif|mp4
                        ├── rollout_global.gif|mp4  # AntMaze only
                        └── steps.txt
```

**路径字段说明：**

| 字段 | 含义 | 示例 |
|------|------|------|
| `model_slug` | 基座模型名 | `Qwen3-0.6B` |
| `selection_tag` | 训练选择标签 | `open`、`all`、`except-large+large-dense` |
| `result_root` | 结果根目录配置项 | `results`、`resultsV2` |
| `variant` | 当前评估变种名 | `open`、`umaze`、`medium` |
| `epoch_<n>` / `step<n>` / `standalone_<eval_uuid>` | 区分训练期中间评估与独立评估运行 | `epoch_2`、`step10002`、`standalone_ab12cd34` |

**示例路径：**

| 场景 | 路径 |
|------|------|
| 训练 open 变种 epoch 2 中间评估 open | `results/Qwen3-0.6B/train=pointmaze-open/exp=<experiment_id>/epoch_2/eval=pointmaze-open/result.json` |
| 训练 open 变种 batch step 10002 中间评估 open | `results/Qwen3-0.6B/train=pointmaze-open/exp=<experiment_id>/step10002/eval=pointmaze-open/result.json` |
| except 模式排除 `large` 和 `large-dense` 后训练，并在 epoch 1 评估 medium | `results/Qwen3-0.6B/train=pointmaze-except-large+large-dense/exp=<experiment_id>/epoch_1/eval=pointmaze-medium/result.json` |
| standalone 评估 open | `results/Qwen3-0.6B/train=pointmaze-open/exp=<experiment_id>/standalone_<eval_uuid>/eval=pointmaze-open/result.json` |
| 评估未微调的原始基座模型 | `results/Qwen3-0.6B/train=pretrained/standalone_<eval_uuid>/eval=pointmaze-open/result.json` |
| AntMaze eval 全局视角录像 | `results/Qwen3-0.6B/train=antmaze-large-play/exp=<experiment_id>/standalone_<eval_uuid>/eval=antmaze-large-play/episode_0/rollout_global.mp4` |
| official normalized score open | `score_results/score_<score_id>/score=pointmaze-open/result.json` |
| score 录像 episode 0 | `score_results/score_<score_id>/score=pointmaze-open/episode_0/rollout.gif` |
| local reference 分数生成 | `score_results/reference_<score_id>/score=pointmaze-local-layout-07/result.json`，并写入 `local_references/pointmaze/local-layout-07.json` |

`evaluate.py` 和 `train.py` 使用同一套基础路径语义，均以单个 `variant` 作为 `eval=<...>` 目录粒度。训练期评估通过 `epoch_<n>` 或 `step<n>` 区分不同轮次；`step<n>` 使用实际完成梯度更新后的全局 train batch step 作为唯一目录名，但 step eval 的触发计数在每个 epoch 开始时重置，所以每个 epoch 的第一次触发都在 epoch-local batch `eval_step_interval`。如果配置的触发点落在梯度累积窗口内，会延后到该窗口的 `optimizer.step()` 完成后保存 checkpoint；是否计算 val loss 和运行环境 rollout 再由 `step_eval_skip` 与 epoch-near 规则决定。`step_eval_skip` 默认为 1；大于 1 时每个 epoch 内的 step eval 触发计数从 1 开始，只有可被该值整除的触发执行完整 validation+rollout，其余只保存 `step<N>` checkpoint。如果实际 step eval 位置落在 epoch eval 前后 `0.25 * eval_step_interval` 的 train batch 窗口内，则该 step 也只保存 checkpoint，不跑 val loss/rollout，epoch eval 仍照常执行。分区训练也保持这个 epoch-local 触发与梯度累积时机：触发点在 train shard 中间时立即执行 step eval，不等待 shard 边界。training-time eval 每次调用都使用同一组 episode reset seeds：第 `i` 个 episode 使用 `eval_seed + i`，默认即 `1..eval_num_episodes`，保证不同 step/epoch eval 间的环境初始条件可比。`eval_parallel_episodes > 1` 时，`parallel_l1` / `parallel_gaussian` / `parallel_t` 会维护多个活跃环境槽位，并把同一步 prompts padding 后合并成一次模型 forward；完成的槽位立即装入后续 episode。其他 action mode 暂时回退串行。连续策略启用 `action_sampling` 时，相同 seed、episode 并行度、world size 和 variant 分配可复现；改变并行度会改变随机数消费顺序，因此轨迹可能变化。`eval_step_interval: 0` 且交互式运行时，`train.py` 会在 dataloader 构建完成后打印 batch 数并允许临时输入 interval；非交互运行保持关闭。standalone `evaluate.py` 通过 `standalone_<eval_uuid>` 区分不同次独立运行，并把合并后的实际 eval 配置保存到该目录下的 `eval_config.yaml`；standalone eval 同样使用 `seed + i` 作为 episode reset seeds。每个 `episode_<n>` 目录同时保存 rollout 视频和逐步文本日志，其中 AntMaze 在 `record_video: true` 时默认额外保存全局俯视角 `rollout_global.<gif|mp4>`，原有 `rollout.<gif|mp4>` 继续表示 MuJoCo 默认跟随视角。`result.json` 保留 `video_path` / `video_paths` 指向跟随视角，并为 AntMaze 全局视角增加 `global_video_path` / `global_video_paths` / `all_video_paths`。`steps.txt` 汇总该 episode 的所有 step，并用分割线和 `Step <n>` 标题分段记录渲染后的 prompt、模型原始输出、最终执行动作、parse 状态和尝试次数；bin 模式日志统一把动作显示为 `<act_XX>`，即使 `new_token: false` 时模型内部实际生成的是复用 token ID；`bin`、`gaussian_bin` 和 `mtp_bin` 且 `record_step_logs=true` 时还会记录每个动作维度上所有 bin token 的概率与对应 token id。

`score.py` 使用独立路径语义，不嵌入训练期/standalone eval 的 `eval=<...>` 目录。每次运行写入 `<result_root>/<mode>_<score_id>/`，其中每个变种写 `score=<env_family>-<variant>/result.json`，run 根目录写 `summary.json` 和实际使用的 `score_config.yaml`。当 `record_video: true` 时，score rollout 视频保存在对应 `score=<...>/episode_<n>/rollout.<gif|mp4>` 下，并在 variant `result.json` 中记录 `video_path` / `video_paths` / `episode_artifact_dirs`。

**结果文件字段：**

```json
{
  "variant": "open",
  "num_episodes": 20,
  "seed": 1,
  "episode_seeds": [1, 2, 3],
  "mean_return": 0.85,
  "std_return": 0.12,
  "success_rate": 0.80,
  "total_parse_failures": 2,
  "total_fallbacks": 0,
  "mean_action_time_ms": 241.3,
  "train_loss": 0.4637,   // 训练期评估有
  "val_loss": 0.4702,     // 训练期评估有
  "eval_type": "step",    // "epoch" 或 "step"
  "eval_tag": "step10002",
  "checkpoint_path": "checkpoints/pointmaze/Qwen3-0.6B/open/<experiment_id>/step10002",
  "video_path": ".../episode_0/rollout.mp4",
  "global_video_path": ".../episode_0/rollout_global.mp4",
  "all_video_paths": [".../episode_0/rollout.mp4", ".../episode_0/rollout_global.mp4"]
}
```

---

### 评估

```bash
python evaluate.py --config eval.yaml
# 多 GPU：每个 rank 处理不同 variants
torchrun --standalone --nproc_per_node=<num_gpus> evaluate.py --config eval.yaml --parallel_backend ddp
```

通过 `eval.yaml` 控制所有评估配置：

```yaml
model_path: checkpoints/pointmaze/Qwen3-0.6B/open/<experiment_id>/final
parallel_backend: single # single | ddp
env_family: pointmaze
eval_mode: single       # single | all | except
variants: [open]         # single: 恰好一个；all: 指定子集或留空表示全部；except: 排除列表
num_episodes: 20
eval_parallel_episodes: 4   # continuous action modes 在每个 rank 内合批的活跃 episode 数
eval_distribute_variants: true  # DDP 下把 variants 轮转分配给不同 ranks
seed: 1                  # episode reset seeds 为 seed, seed+1, ...
parse_retry_limit: 3
# prompt_templete_index: bin_full_sensing  # 可选；standalone eval 只允许一个 prompt 覆盖值
env_kwargs:
  continuing_task: false # false = 每 episode 一个目标，到达即结束
  # max_episode_steps: 300  # 可选，覆盖环境默认值
```

`model_path` 可填 checkpoint 路径或 HuggingFace model ID（如 `Qwen/Qwen3-0.6B`），后者用于评估未微调的基座模型。checkpoint 评估默认使用 checkpoint `config.yaml` 中记录的第一个训练 prompt；`eval.yaml` 可用单个 `prompt_templete_index` 覆盖，覆盖值若不在训练 prompt 列表中需要强确认。

当 continuous action mode 实际使用 `eval_parallel_episodes > 1` 时，不等长 episode 可能乱序完成。合批 rollout 因此关闭逐 episode 进度和逐视频路径输出；调用端仍保留启动信息、结果路径以及每个 variant 完成后的成功率汇总。非 continuous action mode 回退串行后继续使用原有逐 episode 日志。

`evaluate.py` 和 `score.py` 的视频编码默认通过 `video_save_workers: 1` 放到后台线程执行，rollout 在提交后继续。`video_save_workers` 是并发编码线程数；`video_save_max_pending` 统计正在编码与在线程池中排队的视频任务总数，并且不能小于 worker 数。仅线程全部忙碌并不会立即阻塞，只要 pending 数仍低于上限就可以继续排队；达到上限后，下一次提交会等待至少一个任务完成，以免 `record_all` 把全部 frame 长时间保留在内存中。AntMaze eval 每个被录制的 episode 会提交跟随视角和全局视角两个视频任务，两者都计入 pending 上限；`score.py` 仍只保存普通 `rollout.<gif|mp4>`。每个 variant 返回和写最终结果前仍会等待其全部视频完成并传播编码错误。设 `video_save_workers: 0` 可恢复同步保存。

---

### Local PointMaze data generation

本地 PointMaze offline 数据由 `local_varient_gen.py` 生成，数据写入 `local_datasets/`，训练数据读取逻辑只消费最终 Minari/HDF5 数据，不在训练阶段重新生成轨迹。生成脚本复用 Farama 官方 `WaypointController` / `QIteration`，但 Minari step callback 在本仓库实现，用于控制 episode 边界并记录 `qpos`、`qvel`、`goal`。

默认生成逻辑保持 D4RL/Minari PointMaze 风格：`continuing_task=True`、`reset_target=True`，每次 first success 时把该 step 标记为 episode truncation。为补充“到达后保持”数据，可以使用：

```bash
micromamba run -n d4rl_datagen python local_varient_gen.py \
  --variants local-layout-07 \
  --num-workers 4 \
  --target-episodes 1000 \
  --post-success-hold-steps 100 \
  --post-success-hold-noise-std 0.0 \
  --overwrite \
  --seed 42
```

启用 `--post-success-hold-steps > 0` 时，采集环境临时使用 `reset_target=False`，first success 后继续在固定目标上记录 N 个 transition；hold 阶段默认执行确定性 PD action `10 * (desired_goal - achieved_goal) - velocity`，再 clip 到动作范围。已有 local dataset 启用 hold 数据时应使用 `--overwrite` 覆盖重生成，避免把 goal-arrival-only episode 与 hold episode 混在同一个数据集中。

---

### Official normalized score

`evaluate.py` 和训练期 eval 保持快速 rollout / success-rate 风格评估，不承担 official normalized score。PointMaze official-style 分数通过独立入口 `score.py` 计算：

```bash
python score.py --config score.yaml
```

`score.py` 的运行参数统一来自 `score.yaml`；命令行只保留 `--config` 用来选择配置文件，不使用 `--mode`、`--variants`、`--num-episodes` 等覆盖项。强 prompt 警告的自动确认由 `assume_yes: true` 控制。

```yaml
model_path: checkpoints/pointmaze/Qwen3-0.6B/<selection>/<experiment_id>/ep1
result_root: score_results
env_family: pointmaze
mode: score              # score | reference
eval_mode: single        # single | all | except
variants: [open]
num_episodes: 100
num_reference_episodes: 100
seed: 123
parse_retry_limit: 3
assume_yes: false
prompt_templete_index: null
history_num: 0
history_stride: 1
action_sampling: false
action_temperature: 1.0
action_top_p: 1.0
action_top_k: 0
local_reference_root: local_references/pointmaze
record_video: false
record_all: false
video_episode_index: 0
video_fps: 30
video_format: gif
mujoco_gl: egl
local_eval_maps:
  local-layout-07:
    goal_cell: [7, 6]
```

`mode: score` 会加载 checkpoint 并对选中的 PointMaze variant 做 official-style rollout，输出字段包括 `mean_return`、`std_return`、`episode_returns`、`normalized_score`、`std_normalized_score`、`ref_min_score`、`ref_max_score`、`reference_source`、`score_env_spec`、`prompt_template_name`、parse failure / fallback 统计，以及可选录像路径字段 `video_path`、`video_paths`、`video_episode_indices`、`episode_artifact_dirs`。

normalized score 使用 D4RL 0-100 量纲：

```text
normalized_score = 100 * (mean_return - ref_min_score) / (ref_max_score - ref_min_score)
std_normalized_score = 100 * std_return / abs(ref_max_score - ref_min_score)
```

remote D4RL/Minari PointMaze 变种使用 Farama single-goal eval map，强制 `continuing_task: true`、`reset_target: false`，并保留 official horizon：open/umaze 为 300，medium 为 600，large 为 800；dense 变种复用对应 map 形状与 dense env ID。remote reference score 是 `utils/pointmaze_score.py` 中的静态 Minari metadata 表，scoring 不下载数据集读取 reference。

local/custom PointMaze 必须先运行 `mode: reference` 生成本地 reference：

```yaml
mode: reference
eval_mode: single
variants: [local-layout-07]
num_reference_episodes: 100
local_eval_maps:
  local-layout-07:
    goal_cell: [7, 6]   # 0-based row/col，必须是 free cell
```

reference 生成会用 seeded random policy 估计 `ref_min_score`，用 Farama `WaypointController(..., maze_solver="QIteration")` 且无动作噪声估计 `ref_max_score`。默认写入 `local_references/pointmaze/<variant>.json`，文件包含 reference 分数、seed、episode count、horizon、goal cell、reward type、env fingerprint 和 method metadata。后续 local `mode: score` 会校验 reference 文件存在且 env fingerprint 与当前 `score.yaml` 一致，不匹配时拒绝打分。

仓库提供 `reference.yaml` 作为本地 reference 生成示例，默认列出 `local-layout-01` 到 `local-layout-09` 的固定 goal cell；`score.yaml` 则作为模型评分示例，包含 checkpoint、variant、local reference root 和可选录像配置。虽然训练/eval registry 还包含 `local-layout-10..13` 和 `test-layout-01..03`，这些新增地图尚未在示例 YAML 中配置 local score goal cell；对其运行 official-style local score 前需要补充 goal cell 并生成 fingerprint 匹配的 reference。两者都通过 `python score.py --config <yaml>` 运行。

---

### 代码结构

```
project/
├── config.yaml
├── eval.yaml
├── score.yaml
├── reference.yaml
├── prompts/
│   └── <env_family>/            # 每个环境族一个子目录
│       └── <prompt_name>.txt    # 共享 prompt 模板，文件名 stem 是 prompt 名
├── data/
│   ├── <env_family>/            # 每个环境族一个子目录
│   │   ├── variants.py          # 该族所有变种的元信息字典
│   │   ├── dataset.py           # 数据加载、tokenize（每 timestep 展开5条）
│   │   └── formatting.py        # obs 序列化/附加 prompt 变量、text action 生成与解析、action 校验
│   ├── base_dataset.py          # 抽象基类，定义通用接口（load、format、tokenize）
│   └── registry.py              # 环境族注册表，路由 dataset、formatter、variants 和 eval env spec
├── model/
│   └── policy.py                # 模型加载（从 config 读取 model_name）、LoRA 设置
├── train.py                     # 训练入口，读取 config 决定训练模式
├── evaluate.py                  # Rollout 评估
├── score.py                     # PointMaze official-style normalized score / local reference 入口
├── local_varient_gen.py          # local PointMaze Minari 数据生成入口
├── checkpoints/
├── results/
├── score_results/
├── local_references/
└── utils/
    ├── action_bins.py           # action-bin display/model token codec、token selection、gaussian bin loss
    ├── eval_parallel.py         # eval episode 并行参数与 DDP variant 分配
    ├── eval_rollout.py          # eval/score 共享 prompt、history、动作生成、parse retry、fallback 逻辑
    ├── pointmaze_score.py       # PointMaze score env、reference、fingerprint 和 normalized score 逻辑
    └── prompt_loader.py         # 加载指定环境族的共享模板，返回列表
```

**`data/<env_family>/formatting.py` 接口规范**（每个环境族必须实现）：

```python
def format_obs(obs, meta) -> dict:
    """返回 prompt 渲染变量字典；必须包含 obs_text，其他字段可自定义"""

def format_action(action) -> str:
    """text 模式：将 action 向量序列化为训练 target 文本"""

def parse_action(text) -> tuple[np.ndarray, bool]:
    """text 模式：从模型输出文本中解析 action 向量；返回 (action, success)"""

def validate_action(action) -> bool:
    """校验解析出的 action 是否在合法范围内"""
```

`registry.py` 暴露 dataset、formatter、variant metadata 和 eval env spec 查询，让 `train.py` 和 `evaluate.py` 只需传入 `env_family` 即可自动路由。新增环境族时在 `data/` 下新建子文件夹（含 `formatting.py`）并在 `registry.py` 注册一行即可，`train.py` 和 `evaluate.py` 无需改动。

仅预构建当前训练选择对应的 train/val tokenized cache，不进入训练：

```bash
micromamba run -n llm_offline python train.py --config config.yaml --tokenize-only
```

该模式要求配置 `dataset_cache_dir`。`dataset_load_partitions: 1` 时构建或加载完整 train/val cache；`dataset_load_partitions > 1` 时依次准备完整 val cache 和所有 train partition cache。完成后打印 train/val sample 与 batch 汇总、每个 epoch 和全部 epochs 的 train batch steps，并按 `batch_size * world_size` 给出每个 batch step 的近似 global sample 数；摘要同时提醒 `eval_step_interval` 按 epoch-local batch step 计数并在每个 epoch 重置。随后在 DDP 包装、optimizer、W&B、validation、rollout 和训练循环之前退出。

当前 `--tokenize-only` 仍复用正常的 `load_model_and_tokenizer()`，因此会加载 Unsloth 模型并可能占用 GPU；它只保证不进入训练，不提供 CPU-only tokenizer 路径。生成的 cache signature 不包含 DDP rank/world size，之后可直接由单卡或 DDP 训练复用。

---

### 关键实现细节

1. **LoRA 层选择**：`lora_target_modules` 控制按模块名匹配哪些 Linear 层挂 LoRA；可选 `lora_layers_to_transform` 进一步限制 decoder layer index。未配置或设为 `null` 时，所有匹配 `lora_target_modules` 的层都会训练；例如 Qwen3-0.6B 配置 `lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]` 且 `lora_layers_to_transform: [3, 7, 11, 15, 19, 23]` 时，只会训练这 6 层 self-attention 的 q/k/v/o LoRA，用于和 Qwen3.5-0.8B 当前 6 层 full-attention LoRA 覆盖做更干净的对照
2. **Loss masking**：labels 中 `user` turn 与 assistant 前缀部分设为 `-100`，text/bin/gaussian_bin 只训练 assistant 动作文本及其结束标记。bin 模式另有 `action_bin_labels` mask：action token 位置记录 bin index，其余位置为 `-1`；`gaussian_bin` 用它选择 action 位置做 soft-label CE，`bin` 模式保留该 mask 但仍走普通 causal LM loss。`mtp_bin` 的 labels 全部为 `-100`，`action_bin_labels` 在 NTP/AQT 预测位置记录目标 bin，并使用 base CE、sampler CE 和 LCM
3. **每 timestep 按所选 prompt 展开**：`dataset.py` 构造数据时对每个 timestep 遍历 `prompt_templete_index` 指定的共享模板名，生成对应数量的独立样本
4. **Action generation / parsing**：默认 rollout 使用 greedy decoding。若 `action_sampling: true`，text 模式使用普通采样并继续依赖 parse retry / fallback 兜底；bin / gaussian_bin 模式只允许 action-bin token 参与生成，并固定生成 `action_dim` 个 token，避免 EOS 或普通 token 导致动作维度缺失。`mtp_bin` 不调用 `generate()`，而是使用 generation prompt、AQT embedding、sampler head 和 Quadratic Decoding 路径得到固定长度 action bins；若 `mtp_quadratic_decoding: false`，eval 只执行第一次 MTP forward，直接信任 NTP + AQT/sampler proposal，不再 verifier。当前 `mtp_bin` eval 要求 `action_sampling: false`。`parallel_l1`、`parallel_gaussian` 和 `parallel_t` 模式也不调用 `generate()`，而是由 decoder 在 generation prompt embedding 后追加可训练 `action_queries`，使用自定义 attention mask 保持 prompt 内 causal、action query 可看完整非 padding prompt 且 action query 之间双向可见，再读取最后的 query hidden states，经 OFT-style MLPResNet action head 并行预测当前步动作；`action_head_dropout` 只加在该 MLP head 的 hidden path 中，训练模式生效，eval/rollout 下由 `model.eval()` 自动关闭。`parallel_l1` 输出确定性连续动作；`parallel_gaussian` 输出 latent mean，并使用一个 shape 为 `[action_dim]` 的 state-independent `gaussian_log_std` 参数定义 tanh-squashed Gaussian 策略，`action_sampling: true` 时执行 `tanh(Normal(latent_mean, std))` 采样，`false` 时执行 `tanh(latent_mean)`；`parallel_t` 输出 Student-t 策略的 `mean/log_scale`，`action_sampling: true` 时从对应策略采样、`false` 时执行 mean，并在进入环境前按 action bounds clip。text 模式通过 `registry.get_formatter(env_family)` 获取该族的 `parse_action` 解析 decoded 文本；bin 模式不进入环境族 formatter，而是通过共享 `ActionBinCodec` 从 generated/AQT-selected token ids 反查 action bin，再映射回连续动作，因此 `new_token: false` 不依赖低频 token 的 decoded 文本。生成式模式都会调用环境族 `validate_action` 做合法性校验。若解析失败或校验不通过，最多重新让模型生成 `parse_retry_limit` 次（来自 `eval.yaml`）；`mtp_bin` 输出受 ABT 限制，单次 direct forward 失败时直接 fallback。若达到上限仍失败，fallback 到零向量。全程记录 parse 失败次数和 fallback 次数作为辅助指标。
   - *PointMaze text 模式实现*：正则解析紧凑的百分位整数格式 `35,-72`，除以 100 后校验各分量在 `[-1, 1]` 内，clip 后返回；bin 模式由共享 action-bin codec 从 generated/AQT-selected token ids 解析
   - *AntMaze text 模式实现*：解析 8 个逗号分隔的整数百分位并映射为 8 维 torque；bin/continuous 模式共享现有通用实现
   - 其他环境族在各自 `formatting.py` 中实现 text action 解析和校验；bin action 的 token-id 解析保持共享
5. **Obs/Action 序列化**：`dataset.py` 调用同族 `formatting.py` 的 `format_obs`；text 模式调用同族 `format_action`，bin 模式调用共享 action-bin codec 生成 model text 和 display text；`mtp_bin` tokenize generation prompt 和 action prefix，并为 AQT 位置保存 `action_query_*` metadata；连续模式只 tokenize generation prompt，存储连续 `action_values`，不拼 assistant action 文本；`parallel_l1` 用 L1 BC；`parallel_gaussian` 先将目标动作 clamp 到 `(-1, 1)` 内并做 `atanh(action)`，再用 latent-space Gaussian NLL 加 tanh change-of-variables correction 训练 bounded likelihood；`parallel_t` 用 Student-t NLL BC，并可通过 `continuous_mean_l1_weight` 额外加入 `alpha * L1(mean, action)`。若配置 `action_head_weight_decay`，训练 optimizer 只对 continuous action MLP head 内的 Linear weight 使用该 AdamW weight decay；LLM/LoRA、`action_queries`、Gaussian `gaussian_log_std`、bias 和 LayerNorm 参数不受影响
   - *PointMaze 实现*：`format_obs(obs, meta)` 接收环境观测对象（当前为 dict），返回 `obs_text`、动态 `location_sensing_en/zh` 和动态 `wall_sensing_en/zh`
   - *AntMaze 实现*：严格校验 27 维 v4 本体 observation，并结合 `achieved_goal` / `desired_goal` 渲染 torso、姿态、关节和速度状态；训练使用离线数据地图，rollout 使用实例化 eval env 的真实地图
   - `location_sensing` 会直接给出当前位置格子和目标格子；`wall_sensing` 会给出上下左右相邻格子的 `wall/free` 状态；行列从左上角开始按 1-based 计数。坐标由 `utils/maze_sensing.py` 按 `floor + map_center + maze_size_scaling` 公式换算；如果原始结果落在墙格，则吸附到最近的 free cell 中心，避免贴墙/边界数值误差让 prompt 报告墙内位置。若移动方向邻格本身 free，只有在当前位置落入某一侧 threshold、当前同侧格为 free、而前方同侧对角格为 wall 时，才把该方向报告为 `wall`。这样可以预警路口进入窄道时新出现的墙角，但连续直道侧墙不会导致沿直道方向持续误报 `wall`。
6. **Episode 级别 train/val 划分**：先按 `episode_keep_num` 随机抽样 episode pool（真实 episode 更少时使用全部），再在 pool 内按 `floor(pool_size * train_data_ratio)` 划分 train，剩余作为 val，防止数据泄露
7. **多变种混合采样**：联合训练时按各变种样本数加权，保证各变种均匀覆盖；DDP 下通过分布式 weighted sampler 保持同一语义
8. **DataLoader 与设备搬运**：`dataloader_config` 统一控制 train/val loader 的 `num_workers`、`pin_memory`、`persistent_workers` 和 `prefetch_factor`，以及 batch tensor 搬到训练设备时的 `non_blocking`。`persistent_workers` / `prefetch_factor` 仅在 `num_workers > 0` 时合法；`pin_memory: true` 配合 `non_blocking: true` 可让 CUDA H2D copy 具备异步重叠条件。DDP 下每个 rank 独立创建相同数量的 DataLoader workers
9. **DDP 并行训练与评估**：默认 `parallel_backend: single` 保留单卡 Unsloth 路径；`parallel_backend: ddp` 通过 `torchrun` 单机多进程启动，使用 NCCL 同步梯度。DDP 下 `batch_size` 是每 GPU micro-batch，全局有效 batch 为 `batch_size * gradient_accumulation_steps * world_size`。checkpoint 和 validation 仍只由 rank0 执行；`eval_distribute_variants: true` 时训练期和 standalone rollout 把 variants 轮转分配到各 rank，由所属 rank 写对应 result、step logs 和视频，rank0 聚合结果并写 W&B
10. **系统资源监控**：`resource_monitor_enabled: true` 时，仅 rank0 启动后台线程每 `resource_monitor_interval_seconds` 秒覆盖写 `sys_info/<experiment_id>.txt`。该 latest-only 文件记录 RAM/swap 和全部 GPU 状态；RAM/swap 读取 `/proc/meminfo`，GPU 使用结构化 `nvidia-smi --query-gpu`，查询失败只写入 `gpu_error`，不阻断训练。
11. **W&B 指标**：启用 `wandb_enabled` 时，batch 级日志除 `train/loss` / `train/learning_rate` 外，还记录动作模式相关 loss parts：L1 模式记录 `train/l1`，Gaussian 模式记录 `train/nll`、`train/mae`、`train/std`，其中 `nll` 是 squashed-Gaussian NLL，`mae` 是 `tanh(latent_mean)` 与目标动作的 action-space MAE，`std` 是 state-independent policy std 的均值；Student-t 模式记录 `train/tnll`、`train/mae`、`train/scale`、`train/mean_l1_aux`、`train/mean_l1_weight`、`train/df`，`mtp_bin` 记录 `train/base_loss`、`train/sampler_loss`、`train/lcm_loss`，bin soft-label 模式记录 `train/action_loss`、`train/stop_loss`。所有 action-bin 模式（`bin`、`gaussian_bin`、`mtp_bin`）还记录 `train/bin_l1`，即 greedy 预测 bin center 与目标 bin center 在连续动作单位上的 MAE。DDP 下这些指标先跨 rank 平均，再由 rank0 写入。
12. **Official normalized score**：`score.py` 复用 `utils/eval_rollout.py` 中的 prompt 渲染、history、模型动作生成、parse retry 和 fallback 逻辑，但结果 schema 与路径独立于 `evaluate.py`。remote PointMaze reference 使用静态 Minari metadata；local reference 使用显式 goal cell、固定 score env fingerprint 和本地 JSON 校验，避免在 goal/horizon/reward type 变动后误复用旧 reference
13. **新环境族扩展**：在 `prompts/` 新建目录、`data/` 下新建子文件夹（含 `variants.py`、`dataset.py`、`formatting.py`）、`registry.py` 注册一行，`train.py` 和 `evaluate.py` 无需改动

---

### 暂不需要实现
- Return-conditioning
- Online RL 组件
- 多节点分布式训练（当前仅支持单节点 `torchrun`/DDP）
