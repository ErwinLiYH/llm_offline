## 任务描述：LLM Offline RL 初步实验代码

### 项目目标
用预训练 LLM 在 D4RL 离线数据上做 behavior cloning（BC），以纯文本格式输入 obs、输出 action，验证 LLM 处理低维连续控制任务的能力，以及多任务联合训练带来的泛化能力。

---

### 技术栈
- **基座模型**：`Qwen/Qwen3-0.6B`（HuggingFace 加载，LoRA finetune）
- **数据集**：D4RL PointMaze 系列（`minari` 库加载）
- **训练框架**：PyTorch + HuggingFace Transformers + PEFT（LoRA）+ Unsloth（训练加速）

**依赖版本约束：**
- `transformers==4.56.1`：Unsloth 2026.3.x 与 transformers 5.x 不兼容（`generate()` 路径中 `DynamicCache` 与 Unsloth fast inference 返回格式冲突，导致 Qwen3 RoPE shape mismatch）。如需升级 transformers，请先验证 `model.generate()` 在 Unsloth 模型上是否正常工作，或切换 `generate_action` 为注释中的 manual greedy fallback。

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

- 共享风格模板按环境族存放在 `prompts/<env_family>/<idx>.txt`，索引从 `0` 连续编号
- 每个 variant 只在 `data/<env_family>/variants.py` 中维护自己的 `prompt_vars`，提供环境名、reward 描述、迷宫矩阵/可视化、结构说明等差异化信息
- 训练时使用前 `prompt_template_count` 个共享模板，因此每个 timestep 产生 `prompt_template_count` 条训练样本
- 评估时固定使用模板 `0`，保证可复现
- 模板里可以引用 `prompt_vars` 中定义的任意字段以及运行时注入的 `obs_text`；variant 可以提供额外变量，但不能缺少模板实际引用的变量

#### 模板文件格式

共享模板是纯文本文件，例如：

```text
# prompts/<env_family>/0.txt
Environment: {env_name}
Reward structure: {reward_desc_en}
Maze:
{maze_visual}
Current observation:
{obs_text}
Action:
```

#### PointMaze 当前实现

- 当前 `prompts/pointmaze/` 下定义了 5 个共享模板：0–2 英文、3–4 中文
- `POINTMAZE_VARIANTS` 中的每个变种通过 `prompt_vars` 提供：`env_name`、`reward_desc_en`、`reward_desc_zh`、`maze_shape`、`maze_raw_matrix`、`maze_visual`、`structure_desc_en`、`structure_desc_zh`
- target 文本仍由 `data/pointmaze/formatting.py` 定义，动作格式为 `0.35, -0.72`

### 数据处理

- 每个 timestep 的 `(obs, [goal,] action)` 元组展开为 `prompt_template_count` 条训练样本（对应前 `prompt_template_count` 个共享模板）
- obs、goal 的序列化方式（精度、格式）由各环境族的 `formatting.py` 中的 `format_obs` 函数定义，结果填入模板占位符
- action 的目标文本由 `formatting.py` 中的 `format_action` 函数生成
- train/val 划分在 **episode 级别**进行（`train_data_ratio`，默认 `0.9`，即前 90% episode 用于 train，剩余 10% 用于 val），再展开 timestep，避免同一 episode 数据同时出现在 train 和 val 中

---

### 训练模式

通过 `config.yaml` 控制，该文件包含所有训练相关配置：

```yaml
# 环境与任务
env_family: pointmaze
train_mode: single       # single | all
variant: open            # train_mode=single 时指定变种短名

# 基座模型
model_name: Qwen/Qwen3-0.6B   # 任意 HuggingFace causal LM

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

# Debug（注释掉为正常训练）
# max_data_num: 100      # 每个 dataset split 最多使用多少条样本；注释掉 = 全量数据
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
    └── <model_slug>/          # HuggingFace model ID 的最后一段，如 Qwen3-0.6B
        └── <train_mode>/      # single | all
            └── <variant>/     # 变种名（single 模式）或 "all"（all 模式）
                └── <experiment_id>/  # 自动生成或手动指定的实验 ID
                    ├── ep1/       # epoch 1 结束时保存的中间 checkpoint
                    ├── ep2/
                    ├── ep3/
                    └── final/     # 训练全部结束后保存（与最后一个 epN 内容相同）
                        ├── adapter_config.json
                        ├── adapter_model.safetensors
                        ├── tokenizer.json
                        ├── tokenizer_config.json
                        └── config.yaml
```

**示例：**

| 训练场景 | 路径 |
|----------|------|
| Qwen3-0.6B 单独训练 open 变种（epoch 2 中间） | `checkpoints/pointmaze/Qwen3-0.6B/single/open/<experiment_id>/ep2/` |
| Qwen3-0.6B 单独训练 open 变种（训练完成） | `checkpoints/pointmaze/Qwen3-0.6B/single/open/<experiment_id>/final/` |
| Qwen3-0.6B 联合训练所有变种 | `checkpoints/pointmaze/Qwen3-0.6B/all/all/<experiment_id>/final/` |
| Llama-3.2-1B 单独训练 umaze 变种 | `checkpoints/pointmaze/Llama-3.2-1B/single/umaze/<experiment_id>/final/` |

`model_slug` 由 `model/policy.py` 的 `get_model_slug()` 生成（取 `/` 后的部分），不同基座模型的实验不会相互覆盖。

---

#### 2. 数据集缓存路径

Tokenize 后的数据集缓存在 `dataset_cache_dir`（由 `config.yaml` 配置，默认 `dataset_cache/`）：

```
dataset_cache/
├── <env_family>-<variant>-train-prompts<N>-split<train_pct>-<val_pct>.pkl    # 二进制缓存，训练集 token 数据
├── <env_family>-<variant>-train-prompts<N>-split<train_pct>-<val_pct>.jsonl  # 可读文本副本（prompt + action 原文）
├── <env_family>-<variant>-val-prompts<N>-split<train_pct>-<val_pct>.pkl
└── <env_family>-<variant>-val-prompts<N>-split<train_pct>-<val_pct>.jsonl
```

**示例：** `dataset_cache/pointmaze-open-train-prompts1-split<train_pct>-<val_pct>.pkl`

- `.pkl` 用于快速加载（下次训练直接跳过 tokenize，节省约 10 分钟）
- `.jsonl` 每行是 `{"prompt": "...", "action": "0.35, -0.72"}`，供人工抽检数据质量
- 若 `config.yaml` 中未设置 `dataset_cache_dir`（注释掉），则不缓存，每次重新 tokenize
- `max_data_num` 截断发生在 cache 读取之后的内存中，cache 文件始终保存完整数据

---

#### 3. 评估结果路径

评估结果统一保存在 `results/` 下，路径编码了"用哪个模型"、"训练背景"、"在哪个变种上评估"三层信息：

```
results/
└── <model_slug>/
    └── train=<env_family>-<train_variant>-<train_mode>/
        └── exp=<experiment_id>/
            └── eval=<env_family>-<eval_variant>/
                ├── results.json          # evaluate.py 完整评估结果
                ├── result_ep1.json       # 训练 epoch 1 结束后的中间评估
                ├── result_ep2.json
                └── ...
```

**路径字段说明：**

| 字段 | 含义 | 示例 |
|------|------|------|
| `model_slug` | 基座模型名 | `Qwen3-0.6B` |
| `train_variant` | 训练时使用的变种（联合训练为 `all`） | `open`、`all` |
| `train_mode` | 训练模式 | `single`、`all` |
| `eval_variant` | 评估时测试的变种 | `open`、`umaze` |

**示例路径：**

| 场景 | 路径 |
|------|------|
| 训练 open 变种后评估 open | `results/Qwen3-0.6B/train=pointmaze-open-single/eval=pointmaze-open/results.json` |
| 训练 open 变种 epoch 2 中间评估 | `results/Qwen3-0.6B/train=pointmaze-open-single/eval=pointmaze-open/result_ep2.json` |
| 联合训练后评估 umaze | `results/Qwen3-0.6B/train=pointmaze-all-all/eval=pointmaze-umaze/results.json` |
| 评估未微调的原始基座模型 | `results/Qwen3-0.6B/train=pretrained/eval=pointmaze-open/results.json` |

`evaluate.py` 和 `train.py` 使用同一套路径生成逻辑，确保训练中间评估（`result_epN.json`）与最终评估（`results.json`）落在同一目录下，方便对比。

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
model_path: checkpoints/pointmaze/Qwen3-0.6B/single/open/<experiment_id>/final
env_family: pointmaze
variant: open            # 变种短名，或 "all" 表示评估全部变种
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
│       └── <variant_name>.yaml
├── data/
│   ├── <env_family>/            # 每个环境族一个子目录
│   │   ├── variants.py          # 该族所有变种的元信息字典
│   │   ├── dataset.py           # 数据加载、tokenize（每 timestep 展开5条）
│   │   └── formatting.py        # obs/goal 序列化、action 目标文本生成、action 解析
│   ├── base_dataset.py          # 抽象基类，定义通用接口（load、format、tokenize）
│   └── registry.py              # 环境族注册表，按 env_family 路由 dataset 和 formatter
├── model/
│   └── policy.py                # 模型加载（从 config 读取 model_name）、LoRA 设置
├── train.py                     # 训练入口，读取 config 决定训练模式
├── evaluate.py                  # Rollout 评估
├── checkpoints/
├── results/
└── utils/
    └── prompt_loader.py         # 加载指定变种的全部模板，返回列表
```

**`data/<env_family>/formatting.py` 接口规范**（每个环境族必须实现）：

```python
def format_obs(obs, goal) -> str:
    """将 obs 和 goal 序列化为填入 prompt 的文本"""

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

1. **Loss masking**：labels 中 prompt 部分设为 `-100`
2. **每 timestep 展开 `prompt_template_count` 条**：`dataset.py` 构造数据时对每个 timestep 遍历前 `prompt_template_count` 个共享模板，生成对应数量的独立样本
3. **Action parsing（环境族绑定）**：evaluate.py 通过 `registry.get_formatter(env_family)` 获取该族的 `parse_action` 和 `validate_action`。若解析失败或校验不通过，最多重新让模型生成 `parse_retry_limit` 次（来自 `eval.yaml`）。若达到上限仍失败，fallback 到零向量。全程记录 parse 失败次数和 fallback 次数作为辅助指标。
   - *PointMaze 实现*：正则解析 `float, float`，校验各分量在 `[-1, 1]` 内，clip 后返回
   - 其他环境族在各自 `formatting.py` 中实现对应逻辑，格式和校验规则完全自定义
4. **Obs/Action 序列化（环境族绑定）**：`dataset.py` 调用同族 `formatting.py` 的 `format_obs` 和 `format_action`，不依赖任何全局 formatting 工具
5. **Episode 级别 train/val 划分**：先按 episode `train_data_ratio`（默认 `0.9`）划分，train 用前 90%，val 用剩余 10%，再展开 timestep，防止数据泄露
6. **多变种混合采样**：联合训练时按各变种样本数加权，保证各变种均匀覆盖
7. **新环境族扩展**：在 `prompts/` 新建目录、`data/` 下新建子文件夹（含 `variants.py`、`dataset.py`、`formatting.py`）、`registry.py` 注册一行，`train.py` 和 `evaluate.py` 无需改动

---

### 暂不需要实现
- Return-conditioning
- Online RL 组件
- 多 GPU 分布式训练