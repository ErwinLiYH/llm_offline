## 任务描述：LLM Offline RL 初步实验代码

### 项目目标
用预训练 LLM 在 D4RL 离线数据上做 behavior cloning（BC），以纯文本格式输入 obs、输出 action，验证 LLM 处理低维连续控制任务的能力，以及多任务联合训练带来的泛化能力。

---

### 技术栈
- **基座模型**：`Qwen/Qwen3-0.6B`（HuggingFace 加载，LoRA finetune）
- **数据集**：D4RL PointMaze 系列（`minari` 库加载）
- **训练框架**：PyTorch + HuggingFace Transformers + PEFT（LoRA）

---

### PointMaze 变种完整列表

定义在 `data/pointmaze/variants.py` 的 `POINTMAZE_VARIANTS` 字典中，包含所有 8 个变种的 `dataset_id`、`env_id`、`maze_map`、`reward_type`。

maze_map 含义：二维整数矩阵，`1` 表示墙壁（不可通行），`0` 表示可通行空地。

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

- 每个变种对应 **5 个语义完全等价、语言风格不同** 的 prompt 模板，保存在 `prompts/<env_family>/<variant_name>.yaml`
- 语言分配：**模板 1、2、3 为英文**（正式学术、简洁指令、对话描述三种风格），**模板 4、5 为中文**（正式说明、简洁指令两种风格）
- 每个 (obs, action) 对针对所有 5 个模板各生成一条训练样本，即每个 timestep 产生 **5 条**训练数据
- 评估时固定使用模板 0（第一个英文模板），保证可复现

#### 模板文件格式

```yaml
# prompts/<env_family>/<variant_name>.yaml
templates:
  - |
    [英文模板1：正式学术风格]
    ...包含：环境通用说明、变种结构描述、obs 含义与具体值、goal 含义与具体值、action 格式要求与取值范围...
    Action:
  - |
    [英文模板2：简洁指令风格]
    ...
    Action:
  - |
    [英文模板3：对话描述风格]
    ...
    Action:
  - |
    [中文模板4：正式说明风格]
    ...
    动作：
  - |
    [中文模板5：简洁指令风格]
    ...
    动作：
```

#### 每个模板必须包含的语义内容（通用要求）

每个环境族在设计模板时必须涵盖以下语义，具体措辞和数值格式由该环境族自行定义：

1. **任务描述**：这是什么类型的控制任务
2. **变种/场景描述**：该变种的结构特征（地图、布局、难度等）
3. **Observation 语义**：obs 各维度的物理含义，以及当前 obs 的具体数值
4. **Goal 语义**（如有）：goal 各维度含义，以及当前 goal 的具体数值
5. **Action 输出格式要求**：格式规范、取值范围、示例。要明确：只输出动作本身，不输出任何多余内容（无单位、无括号、无说明文字、无换行）

> **PointMaze 具体实现**（其他环境族另行定义）：
> - obs：4 维 `[x, y, vx, vy]`；goal：2 维 `[gx, gy]`；action：2 维力 `[ax, ay]`，范围 `[-1, 1]`
> - 变种描述包含 maze_map 原始矩阵及结构文字说明
> - Action 示例：`0.35, -0.72`
> - Target 格式：`{ax:.2f}, {ay:.2f}`

#### Target 格式（通用规则）

Target 文本格式由各环境族的 `data/<env_family>/formatting.py` 定义（见"代码结构"）。训练时 prompt 部分 labels 设为 `-100`，只对 target 部分计算 loss。

---

### 数据处理

- 每个 timestep 的 (obs, [goal,] action) 元组展开为 **5 条**训练样本（对应 5 个模板）
- obs、goal 的序列化方式（精度、格式）由各环境族的 `formatting.py` 中的 `format_obs` 函数定义，结果填入模板占位符
- action 的目标文本由 `formatting.py` 中的 `format_action` 函数生成
- train/val 划分在 **episode 级别**进行（9:1），再展开 timestep，避免同一 episode 数据同时出现在 train 和 val 中

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
```

对 PointMaze 环境族，共训练以下 9 个模型：
- 单变种模型 × 8（每个变种独立训练）
- 全变种联合模型 × 1

联合训练时各变种数据按样本数加权采样，避免大变种压制小变种。

---

### 模型保存结构

```
checkpoints/
└── pointmaze/
    ├── single/
    │   ├── open/
    │   │   ├── checkpoint-500/
    │   │   └── final/
    │   ├── open-dense/
    │   │   └── final/
    │   ├── umaze/
    │   │   └── final/
    │   ├── umaze-dense/
    │   │   └── final/
    │   ├── medium/
    │   │   └── final/
    │   ├── medium-dense/
    │   │   └── final/
    │   ├── large/
    │   │   └── final/
    │   └── large-dense/
    │       └── final/
    └── all/
        ├── checkpoint-500/
        └── final/
```

每个 `final/` 下保存：LoRA adapter 权重、tokenizer、训练所用 `config.yaml` 副本。

---

### 评估

```bash
python evaluate.py --config eval.yaml
```

通过 `eval.yaml` 控制所有评估配置：

```yaml
model_path: checkpoints/pointmaze/Qwen3-0.6B/single/open/final
env_family: pointmaze
variant: open            # 变种短名，或 "all" 表示评估全部变种
num_episodes: 20
parse_retry_limit: 3
```

评估时固定使用模板 0。结果保存到 `results/`，目录结构与 `checkpoints/` 对应，记录每个变种的 episode return 和 success rate。

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
2. **每 timestep 展开 5 条**：`dataset.py` 构造数据时对每个 timestep 遍历全部 5 个模板，生成 5 条独立样本
3. **Action parsing（环境族绑定）**：evaluate.py 通过 `registry.get_formatter(env_family)` 获取该族的 `parse_action` 和 `validate_action`。若解析失败或校验不通过，最多重新让模型生成 `parse_retry_limit` 次（来自 `eval.yaml`）。若达到上限仍失败，fallback 到零向量。全程记录 parse 失败次数和 fallback 次数作为辅助指标。
   - *PointMaze 实现*：正则解析 `float, float`，校验各分量在 `[-1, 1]` 内，clip 后返回
   - 其他环境族在各自 `formatting.py` 中实现对应逻辑，格式和校验规则完全自定义
4. **Obs/Action 序列化（环境族绑定）**：`dataset.py` 调用同族 `formatting.py` 的 `format_obs` 和 `format_action`，不依赖任何全局 formatting 工具
5. **Episode 级别 train/val 划分**：先按 episode 9:1 划分，再展开 timestep，防止数据泄露
6. **多变种混合采样**：联合训练时按各变种样本数加权，保证各变种均匀覆盖
7. **新环境族扩展**：在 `prompts/` 新建目录、`data/` 下新建子文件夹（含 `variants.py`、`dataset.py`、`formatting.py`）、`registry.py` 注册一行，`train.py` 和 `evaluate.py` 无需改动

---

### 暂不需要实现
- Return-conditioning
- Online RL 组件
- 多 GPU 分布式训练