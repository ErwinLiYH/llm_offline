---
name: Changelog 2026-03-31
description: Session changes — Unsloth integration, evaluate.py fixes, checkpoint path fixes
type: project
---

## model/policy.py

**训练路径改用 Unsloth：**
- `load_model_and_tokenizer`：`AutoModelForCausalLM + LoraConfig + get_peft_model` → `FastLanguageModel.from_pretrained + FastLanguageModel.get_peft_model`
- 启用 `use_gradient_checkpointing="unsloth"`，dtype=None（自动检测，RTX 5090 → bf16）

**推理路径保持标准 PEFT：**
- `load_from_checkpoint`：改回 `AutoModelForCausalLM + PeftModel`，绕过 Unsloth 对 Qwen3 `generate` 的错误 patch（`RuntimeError: shape mismatch in RoPE cos/sin`）
- 新增支持直接传 HuggingFace model ID：无 `adapter_config.json` 时当 base model 加载，用于评估未微调的原模型

## train.py

- `_save_checkpoint`：保存后 patch `adapter_config.json`，将 `base_model_name_or_path` 从 `unsloth/Qwen3-0.6B` 改回原始 `config["model_name"]`，防止离线加载失败

## evaluate.py

- 修复成功率恒为 0 的 bug：原来依赖 `terminated=True`，但环境默认 `continuing_task=True` 时 terminated 永远不触发
- `gym.make` 改为透传 `env_kwargs` 字典，支持任意环境参数
- `generate_action`：传入 `attention_mask` 消除 HuggingFace 警告
- 新增 `mean_action_time_ms` 到结果输出
- `get_results_dir` 重写：路径格式 `results/<model_slug>/train=<context>/eval=<family>-<variant>/`，base model 时 train context 为 `pretrained`

## eval.yaml

- 新增 `env_kwargs` 块（`continuing_task: false`）

---

## train.py（续）

**新增 epoch 结束后环境评估：**
- 新增 `get_eval_results_dir(config, variant)`：生成与 evaluate.py 对齐的结果路径
- 新增 `_run_epoch_eval(...)`：每 epoch 结束后对指定 variant 列表跑 n 个 episode，复用 evaluate.py 的 `evaluate_variant`，结果追加 `train_loss`/`val_loss`，保存为 `result_ep{epoch}.json`
- `_run_training` 新增 `tokenizer` 和 `eval_variants` 参数，epoch 末尾触发评估（`eval_num_episodes=0` 时跳过）
- `train_single_variant`：默认 `eval_variants=[variant]`；`train_all_variants`：默认 `eval_variants=[]`

## evaluate.py（续）

- `get_results_dir` 的 train_tag 格式从 `{env_family}-{slug}-{train_mode}` 改为 `{env_family}-{variant}-{train_mode}`，消除路径中 slug 重复（slug 已在父目录）

## config.yaml

- 新增 `eval_num_episodes: 5`、`eval_variants: []`、`eval_env_kwargs: {continuing_task: false}`

---

## evaluate.py（2026-03-31 续）

**新增每 episode 步数统计：**
- `evaluate_variant`：新增 `episode_steps` 列表，每步 `ep_steps += 1`，episode 结束后 append
- 结果中新增 `mean_episode_steps` 和 `std_episode_steps` 字段
- episode 进度打印中增加 `steps=` 字段

## train.py（2026-03-31 续）

- `_run_epoch_eval` 打印中增加 `mean_steps=` 字段

---

## data/pointmaze/dataset.py（2026-03-31 续）

**新增 `max_data_num` 参数（debug 用）：**
- `PointMazeDataset.__init__` 新增 `max_data_num: int | None = None` 参数
- `load()` 结束时（包含 cache 命中路径）若设置了 `max_data_num`，截断 `self._samples[:max_data_num]`
- cache 文件始终保存完整数据，截断仅发生在内存中，不影响 cache

## train.py（2026-03-31 续2）

- `train_single_variant` 和 `train_all_variants` 创建 dataset 时透传 `config.get("max_data_num")`

## config.yaml（2026-03-31 续）

- 新增注释掉的 `# max_data_num: 100`，取消注释即可限制训练样本数用于快速 debug

---

## 环境依赖（2026-03-31 续）

**Transformers 降级至 4.56.1：**
- Unsloth 2026.3.17 与 transformers 5.3.0 不兼容：`unsloth_fast_generate` 强制注入 `cache_implementation="dynamic"`，但 Unsloth fast inference 路径返回旧式 `list[(K, V)]` 而非 `DynamicCache`，导致 Transformers 5.x 在 decode 步骤重算全序列 `position_ids`，引发 `RuntimeError: shape mismatch in Qwen3Attention_fast_forward_inference (Qn *= cos)`
- 降级到 `transformers==4.56.1` 解决（TRL 要求 >=4.56.1，Unsloth 支持该版本）

## evaluate.py（2026-03-31 续2）

**`generate_action` 保留 manual greedy 备用方案：**
- 主路径恢复使用 `model.generate()`（transformers 降级后正常工作）
- 函数体末尾注释保留手动 greedy loop 实现（调 `model.forward(use_cache=False)`），作为 fallback：若 Unsloth/transformers 版本再次出现 generate 兼容问题，取消注释即可绕开整个 `model.generate()` 路径

## train.py（2026-03-31 续3）

**`_run_epoch_eval` eval/train 模式切换：**
- eval 前调 `model.eval()` + `FastLanguageModel.for_inference(model)`（关闭 gradient checkpointing，启用推理模式）
- eval 后调 `model.train()` + `FastLanguageModel.for_training(model)`（恢复 gradient checkpointing）
- `from unsloth import FastLanguageModel` 移至文件顶部 import

---

## model/policy.py（2026-03-31 续）

**`load_from_checkpoint` 改用 Unsloth 加载：**
- 原实现：`AutoModelForCausalLM + PeftModel.from_pretrained`（标准 PEFT 路径）
- 新实现：`FastLanguageModel.from_pretrained(model_name=model_path, ...)`（Unsloth 自动识别 `adapter_config.json` 并加载 base model + adapter）
- LoRA checkpoint 时从目录内 `config.yaml` 读取 `max_length` 作为 `max_seq_length`，不存在则默认 2048；base model 时直接默认 2048
- 移除不再使用的 `AutoModelForCausalLM`、`AutoTokenizer`、`PeftModel` import

---

## train.py（2026-04-01）

**每个 epoch 结束后保存中间 checkpoint：**
- `get_checkpoint_dir` 新增 `epoch: int | None = None` 参数：传入整数时路径末尾为 `ep{epoch}`，不传时仍为 `final`
- `_run_training` 新增 `variant` 参数（默认 `"all"`），在 val loss 打印后立即调 `_save_checkpoint` 保存当前 epoch 的 checkpoint
- 路径示例：`checkpoints/pointmaze/Qwen3-0.6B/single/open/ep1`、`ep2`、`ep3`，训练结束后仍保存 `final`
- `train_single_variant` 和 `train_all_variants` 调用时分别传入对应 `variant` / `"all"`

---

## model/policy.py（2026-04-07）

**新增 Unsloth 4-bit 量化开关：**
- `load_model_and_tokenizer` 新增读取 `config["load_in_4bit"]`，并透传到 `FastLanguageModel.from_pretrained(..., load_in_4bit=...)`
- `load_from_checkpoint` 新增 `load_in_4bit: bool | None = None` 参数；若未显式传入，则从 checkpoint 内保存的 `config.yaml` 读取 `load_in_4bit`，默认 `false`

## evaluate.py（2026-04-07）

- `main()` 调用 `load_from_checkpoint(...)` 时新增透传 `config.get("load_in_4bit")`
- 评估配置现在可以显式控制是否以 Unsloth 4-bit 模式加载模型

## config.yaml（2026-04-07）

- 新增 `load_in_4bit: false` 配置项，用于控制训练时是否启用 Unsloth 4-bit 量化加载

## eval.yaml（2026-04-07）

- 新增 `load_in_4bit: false` 配置项，用于控制评估时是否启用 4-bit 量化加载；若评估的是 LoRA checkpoint，也可不显式设置，回退到 checkpoint 保存的训练配置

---

## data/pointmaze/dataset.py（2026-04-07）

**新增 `prompt_template_count` 数据集构建参数：**
- `PointMazeDataset.__init__` 新增 `prompt_template_count: int = 1`
- 训练构建数据集时不再固定使用全部 5 个模板，而是只使用前 `prompt_template_count` 个模板
- 新增参数校验：`prompt_template_count` 必须在 `1..模板文件实际数量` 范围内
- 缓存命名改为 `pointmaze-<variant>-<split>-prompts<N>.{pkl,jsonl}`，不同 prompt 数只读取各自匹配的缓存；若不存在匹配缓存则重新构建

## train.py（2026-04-07）

- `train_single_variant` 和 `train_all_variants` 创建 dataset 时新增透传 `prompt_template_count`
- `train_all_variants` 创建 dataset 时补传 `cache_dir`，联合训练现在也会使用数据缓存

## config.yaml（2026-04-07 续）

- 新增 `prompt_template_count: 1`，默认每个 timestep 仅使用 1 个 prompt 模板构建训练样本

---

## Prompt System（2026-04-08）

**PointMaze prompt 系统改为共享模板 + variant 元数据渲染：**
- `prompts/pointmaze/` 从按 variant 的 YAML 改为按 family 共享的 `0.txt` 到 `4.txt`
- `utils/prompt_loader.py` 改为加载 `prompts/<env_family>/<idx>.txt`，要求索引从 0 连续
- 新增 `render_template(...)`，模板缺失变量时直接报错；允许 variant 提供额外未使用变量
- `data/pointmaze/variants.py` 为每个 variant 新增 `prompt_vars`，集中定义环境名、reward 描述、迷宫矩阵/可视化、结构说明等渲染变量
- 训练和评估都改为使用共享 family 模板；评估仍固定使用模板 0
- 共享模板重构后，旧缓存应手动删除，随后按现有 `prompts<N>` 规则重新生成

---

## train.py / evaluate.py（2026-04-08）

**checkpoint 和 results 路径新增实验 ID 层：**
- 训练配置新增可选 `experiment_id`；若未提供，训练启动时自动生成 8 位短 UUID
- checkpoint 路径改为 `checkpoints/<env_family>/<model_slug>/<train_mode>/<variant>/<experiment_id>/epN|final/`
- 训练时保存到 checkpoint 目录的 `config.yaml` 会包含最终使用的 `experiment_id`
- 训练期中评估与 `evaluate.py` 的结果路径同步新增 `exp=<experiment_id>` 一层，避免同参数重复实验互相覆盖

---

## data/pointmaze/dataset.py（2026-04-08）

**episode 切分边界现在由单个比例控制：**
- 使用 `train_data_ratio` 配置项控制 train 使用多少比例的 episodes
- `val` 自动使用剩余 episodes；`train_data_ratio: 0.9` 就是原来的 9:1
- dataset cache 文件名中的 split 标记会体现这个比例，例如 `split90`

---

## data/pointmaze/dataset.py / config.yaml（2026-04-08）

**episode 切分改为单个比例配置项：**
- 新增 `train_data_ratio` 配置项，默认值为 `0.9`
- train 使用前 `train_data_ratio` 比例的 episodes，val 自动使用剩余 episodes
- dataset cache 文件名中的 split 标记改为根据当前配置动态生成，如 `split90`
- 非法配置会直接报错：要求 `0 < train_data_ratio < 1`

---

## evaluate.py / eval.yaml（2026-04-09）

**评估支持导出 rollout 视频：**
- 新增 `record_video`、`video_episode_index`、`video_fps`、`video_format` 配置项
- 录制开启时会自动把环境切到 `rgb_array` 渲染，并把指定 episode 保存到 `results/...` 目录
- 默认推荐 `gif` 输出；`mp4` 需要可用的 ffmpeg backend

- 录像评估新增 `mujoco_gl` 配置项；headless MuJoCo 默认推荐 `egl`
