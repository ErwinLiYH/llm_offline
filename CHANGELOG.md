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

---

## train.py / config.yaml（2026-04-10）

**训练模式与训练期 eval 选择重构：**
- `train_mode` 新增 `except`，可使用“除若干变种外”的其余变种联合训练
- `config.yaml` 的训练选择从单值 `variant` 改为列表 `variants`
- 训练期环境评估新增独立的 `eval_mode` 和 `eval_variants`，不再强绑定训练模式
- 新增 `utils/variant_selection.py` 统一解析 `single | all | except` 的训练/评估变种集合与路径 tag
- `single` 要求 `variants` 恰好一个；`all` 要求列表为空；`except` 使用列表作为排除集合
- checkpoint / 训练期 eval 结果路径中的选择层改为 `selection_tag`，`except` 形如 `except-open+large`
- `single`、`all`、`except` 统一复用同一套 dataset / dataloader 构建流程，多变种训练继续按样本量加权采样

- 最终路径约定进一步收敛为 `selection_tag` 结构：checkpoint 使用 `checkpoints/<env_family>/<model_slug>/<selection_tag>/<experiment_id>/...`，results 使用 `train=<env_family>-<selection_tag>`

- `utils/train_variant_selection.py` 更名为 `utils/variant_selection.py`，并扩展为 train / standalone eval 共用
- `evaluate.py` 新增 `eval_mode` + 列表 `variants` 支持，语义与训练侧一致；保留旧 `variant: <name|all>` 兼容读取

- PointMaze 动作文本格式改为紧凑的百分位整数：由 `0.35, -0.72` 改为 `35,-72`；同步更新 formatter、decoder、shared prompts 和相关文档

---

## PointMaze Prompt / Eval（2026-04-18）

**PointMaze prompt 感知信息改为动态 `map_sensing`：**
- 共享 prompt 模板不再写静态 `Grid-coordinate mapping` 公式说明，改为直接注入动态 `map_sensing_en` / `map_sensing_zh`
- `map_sensing` 现在直接给出当前位置格子、目标格子，以及上下左右相邻格子的 `wall/free` 状态
- 行列编号统一为从左上角开始的 `1-based`
- 坐标到格子的换算逻辑改为与 PointMaze 环境源码一致的 `floor + map_center + maze_size_scaling` 公式，避免不同迷宫宽高奇偶下的偏差

**`format_obs` 接口重构：**
- formatter 接口从返回单个字符串改为返回 prompt 渲染变量字典，且必须包含 `obs_text`
- PointMaze 的 `format_obs` 现在统一为 `format_obs(obs, meta) -> dict`，由 formatter 自己解析环境观测结构并补充 `map_sensing`
- 训练数据构造和 standalone eval 统一复用 `format_obs(...)` 的返回值渲染 prompt
- `format_obs` 内部显式将 `observation` 和 `desired_goal` 转成 `np.float32`

**standalone eval 录像配置增强：**
- `evaluate.py` 运行 standalone eval 时，结果目录改为 `.../eval=<tag>#<eval_uuid>/results.json`
- `train.py` 的训练期 eval 路径保持不变
- `eval.yaml` 的 `video_episode_index` 现在支持单个整数或列表
- 新增 `record_all`；为 `true` 时忽略 `video_episode_index` 并录制全部 episode
- 开启录像时会自动将 `env_kwargs.render_mode` 覆盖为 `rgb_array`

---

## data/pointmaze/formatting.py / data/pointmaze/dataset.py / train.py / evaluate.py（2026-04-22）

**PointMaze 主流程新增历史轨迹 prompt：**
- 新增 `history_num` / `history_stride` 配置项；`config.yaml` 和 `eval.yaml` 各自独立配置，`0` 表示关闭历史
- `data/pointmaze/formatting.py` 新增 `format_history(...)`，统一渲染中英文历史块；历史只包含每条过去 transition 的起始位置和动作
- `PointMazeDataset` 现在会从 offline episode 中按 `t-1, t-1-stride, ...` 回溯采样历史，并将历史配置编码进 dataset cache 文件名，避免误复用旧缓存
- `evaluate_variant(...)` 现在在 rollout 中维护在线 history buffer；每个 episode 的首步显式使用空历史，之后每步把实际执行动作写入历史，fallback 零动作也会被记录
- 训练期 epoch eval 复用训练配置中的历史参数，standalone `evaluate.py` 使用 `eval.yaml` 中的历史参数

## prompts/pointmaze/*.txt（2026-04-22）

**共享 PointMaze prompts 接入历史块并重排静态/动态区域：**
- 全部模板新增 `{history_block_en}` / `{history_block_zh}` 占位符；历史为空时渲染为空字符串，不额外输出标题
- 模板整体重排为“静态描述在前、动态状态在后”，以提高前缀缓存命中率
- `prompt 0` 进一步简化：移除 raw matrix 与 reward 描述，只保留 visual maze，并采用更结构化的 `Env Description` / `Current Status` 组织方式
- 历史排序语义说明移到静态区域：第一条是最早采样到的历史 step，最后一条是当前 step 之前最近的采样历史 step

## inspect_jsonl_record.py（2026-04-22）

**新增 JSONL 样本检查脚本：**
- 新增 `inspect_jsonl_record.py`
- 用法：`python inspect_jsonl_record.py <jsonl_path> <record_index>`
- 按固定格式打印单条记录：
  - `Prompt:\n...`
  - `Action:\n...`
- 用于快速核查 dataset cache 导出的 `.jsonl` 样本内容

## plan_probe.py（2026-04-22）

**新增规划能力探针脚本：**
- 新增 `plan_probe.py`，用于把路径规划能力从低层控制中拆出来单独测试
- 支持两种模式：一次性输出整条路径的 `path` 模式，以及逐步决策的 `step` 模式
- 支持本地 checkpoint 与 OpenAI 兼容云端接口两种后端；云端模式在 `responses.create` 不可用时自动回退到 `chat.completions.create`
- 新增可选的直接输出/允许思考开关，以及 `step` 模式的历史轨迹注入参数，用于验证记忆对规划成功率的影响
- 训练期环境评估新增独立的 `eval_mode` 和 `eval_variants`，不再强绑定训练模式
- 新增 `utils/variant_selection.py` 统一解析 `single | all | except` 的训练/评估变种集合与路径 tag
- `single` 要求 `variants` 恰好一个；`all` 要求列表为空；`except` 使用列表作为排除集合
- checkpoint / 训练期 eval 结果路径中的选择层改为 `selection_tag`，`except` 形如 `except-open+large`
- `single`、`all`、`except` 统一复用同一套 dataset / dataloader 构建流程，多变种训练继续按样本量加权采样
- 最终路径约定进一步收敛为 `selection_tag` 结构：checkpoint 使用 `checkpoints/<env_family>/<model_slug>/<selection_tag>/<experiment_id>/...`，results 使用 `train=<env_family>-<selection_tag>`
- `utils/train_variant_selection.py` 更名为 `utils/variant_selection.py`，并扩展为 train / standalone eval 共用
- `evaluate.py` 新增 `eval_mode` + 列表 `variants` 支持，语义与训练侧一致；保留旧 `variant: <name|all>` 兼容读取
- PointMaze 动作文本格式改为紧凑的百分位整数：由 `0.35, -0.72` 改为 `35,-72`；同步更新 formatter、decoder、shared prompts 和相关文档

## evaluate.py / train.py / config.yaml / eval.yaml（2026-04-24）

**eval 结果目录重排，并新增可配置结果根目录：**
- 新增 `result_root` 配置项；`config.yaml` 和 `eval.yaml` 都可独立指定 eval 结果根目录，默认仍为 `results`
- 训练期 eval 结果现在落在 `.../exp=<experiment_id>/epoch_<n>/eval=<env_family>-<variant>/result.json`
- standalone `evaluate.py` 结果现在落在 `.../exp=<experiment_id>/standalone_<eval_uuid>/eval=<env_family>-<variant>/result.json`
- standalone eval 的目录粒度改为单个 `variant`，不再使用 `eval=<selection_tag>#<eval_uuid>` 作为最终结果目录

**eval 新增逐步对话日志与按 episode 归档的视频：**
- `evaluate.py` 和训练期 epoch eval 现在都会默认保存逐步文本日志，可通过 `record_step_logs` 开关关闭
- 每个 `episode_<n>` 目录下新增 `steps/step_<n>.txt`，记录渲染后的 prompt、模型原始输出、最终执行动作、parse 状态和尝试次数
- rollout 视频现在与 `steps/` 同级保存在对应 `episode_<n>` 目录下，不再直接平铺到 eval 根目录
- `inspect_jsonl_record.py` 的 `Prompt:` / `Action:` 文本格式被抽成共享 helper，eval step log 复用同一基础格式

**训练期 eval 接入独立 eval 的视频配置：**
- `config.yaml` 新增训练期 eval 可用的 `record_video`、`record_all`、`video_episode_index`、`video_fps`、`video_format`、`mujoco_gl`
- `train.py` 的 epoch eval 现在会透传这些配置给 `evaluate_variant(...)`，训练过程中也能按配置录制视频

## data/pointmaze/dataset.py / evaluate.py / utils/chat_template.py（2026-04-24）

**训练与评估现在严格使用模型自带的 chat template：**
- 新增 `utils/chat_template.py`，统一通过 tokenizer 自带的 `chat_template` 构造最终对话序列
- 训练样本不再直接编码 `prompt + action_text`；而是把渲染后的环境 prompt 作为 `user` 消息，把动作文本作为 `assistant` 消息
- eval 推理也改为先把环境 prompt 包装成带 `add_generation_prompt=True` 的 chat-template 输入，再调用 `model.generate(...)`
- 如果 tokenizer 未定义 `chat_template`，现在会显式报错，而不是静默回退到纯文本拼接

**dataset cache 行为切换到 chat template 版本：**
- PointMaze dataset 默认按 chat-template 方式构造 tokenized cache；旧的 plain-text cache 需要手动清理后重建

## train.py / data/pointmaze/dataset.py / config.yaml（2026-04-24）

**offline dataset 新增按 episode 子采样与跨 variant 训练 quota 平衡：**
- `config.yaml` 新增 `episode_keep_ratio`、`balance_variant_episode_count`、`sampling_seed`
- PointMaze offline dataset 不再按前缀切分 episode；现在先随机无放回抽样 train episodes，再从剩余 episodes 中按 `train_data_ratio` 反推 quota 抽样 val
- 极小 `episode_keep_ratio` 仍至少保留 1 个 train episode；若剩余 episodes 不足 val quota，会自动降级为“使用全部剩余 episodes”并打印原因
- 多 variant 训练时，`train.py` 会先统计各 variant 的原始 episode 数和初始 train quota；若 `balance_variant_episode_count=true`，则统一裁到最小 quota，并打印全局平衡摘要
- 每个 variant 的 dataset 构建会打印原始 episode/step 数、初始 train quota、是否被平衡裁小、最终 train/val episode 数和 step 数

**dataset cache 行为保持不变：**
- cache 文件名不包含这三个新配置，命中现有 `.pkl` 时仍直接读取旧 cache
- 命中 cache 时会明确打印 `episode_keep_ratio` / `balance_variant_episode_count` / `sampling_seed` 本次未生效
- `max_data_num` 仍只在 cache 读取或新构建完成后做样本级内存截断，不替代 episode 级抽样

## prompts / tokenizer compatibility / project-changelog skill（2026-04-25）

**PointMaze history 与 prompt 0 输出格式调整：**
- `format_history(...)` 现在在有历史时自行渲染完整 `## History` section；`history_num=0` 或首步无历史时仍渲染为空，不在 prompt 中残留 history 标题
- history 条目改为显式描述“前 n 步的起始点 / 所在格 / 动作”，不再使用 `1. 2. 3.` 序号列表
- offline dataset 和 rollout eval 都会为 history entry 传入真实 `steps_ago`，eval 侧按在线 history buffer 动态计算相对步数
- `prompts/pointmaze/0.txt` 将 `{history_block_en}` 放在 `## Env Description` 和 `## Current Status` 之间，并收紧动作格式示例，强调 `"35,-72"` / `"-5,100"` 这类紧凑文本输出

**Qwen3.5 / Qwen3VLProcessor 兼容：**
- `Qwen/Qwen3.5-4B` 通过 Unsloth 加载时第二返回值可能是 `Qwen3VLProcessor`，不是普通 tokenizer；外层 processor 没有 `name_or_path`
- tokenizer / processor 调用统一使用显式 `text=...`，避免 `Qwen3VLProcessor.__call__` 将 prompt 位置参数误解释为 `images`
- chat template 统一尝试传 `enable_thinking=False`，让 eval generation prompt 停在 closed empty thinking block 后，和训练目标中动作出现的位置对齐
- `PointMazeDataset` 新增 `tokenizer_name_or_path`，`train.py` 构建 dataset 时传入 `config["model_name"]`，dataset worker reload 不再依赖外层 processor 的 `name_or_path`

---

## action token modes / gaussian bin loss（2026-04-26）

**训练动作输出新增三种模式：**
- `config.yaml` 新增 `action_token_mode: text | bin | gaussian_bin`，其中 `text` 保持原有 `35,-72` 紧凑整数动作格式
- `bin` / `gaussian_bin` 使用共享特殊 token `<act_00>` ... `<act_N>` 表示各维动作 bin；当前配置切到 `gaussian_bin`、`action_num_bins: 50`、`action_soft_label_sigma: 1.0`
- `model/policy.py` 在 bin 模式下会把动作 token 注册为 tokenizer 的 `additional_special_tokens`，并在 LoRA 注入前 resize token embedding

**PointMaze dataset 与 loss 接入 action bin：**
- `PointMazeDataset` 根据 action mode 生成 assistant 动作文本；bin 模式输出纯动作 token，如 `<act_03><act_48>`
- dataset 样本新增 `action_bin_labels` mask，动作 token 位置记录真实 bin index，非动作位置为 `-1`
- `gaussian_bin` 训练时动作 token 位置使用 Gaussian soft-label CE；chat template 产生的结束 token 仍使用普通 CE
- dataset cache 文件名新增 action mode、bin 数和 bin 范围，避免误复用 text/bin 不同编码方式的 tokenized cache

**评估与 step log 支持 bin 模式：**
- `evaluate.py` 的 bin / gaussian_bin 模式会保留 special tokens 解码，解析 `<act_XX>` 后映射回连续动作；解析失败仍走原有 retry/fallback 流程
- standalone eval 的 action 编码配置固定来自 checkpoint 内 `config.yaml`，`eval.yaml` 不允许覆盖 action 相关配置
- `record_step_logs: true` 且 `action_token_mode: gaussian_bin` 时，step log 会额外记录每个动作维度上所有 action bin 的生成概率分布

---

## utils/variant_selection.py / config docs（2026-04-27）

**`all` 模式支持显式 variant 子集：**
- `train_mode: all` / `eval_mode: all` 下，如果 `variants` / `eval_variants` 非空，则只选择列表中指定的 variants
- `all` 模式下如果 variant 列表为空或省略，仍选择所有可用 variants
- 子集 all 的 `selection_tag` 改为 `all-<selected variants joined by +>`，避免实际只跑子集时路径仍显示为 `all`
- 同步更新 `config.yaml`、`eval.yaml`、`AGENTS.md` 和 `DESIGN.md` 中关于 `all` 模式列表语义的说明

---

## prompt selection by filename（2026-04-27）

**训练 prompt 选择从数量切换为文件名列表：**
- `config.yaml` 将 `prompt_template_count` 替换为 `prompt_templete_index: ["0"]`，列表元素是 `prompts/<env_family>/` 下 `.txt` 文件名去掉扩展名后的 prompt 名
- `utils/prompt_loader.py` 不再要求 prompt 文件名是连续数字；现在按 filename stem 建立 prompt name 到模板文本的映射
- `PointMazeDataset` 按 `prompt_templete_index` 的顺序选择模板，每个 timestep 仍按所选模板数展开为多条训练样本
- dataset cache 文件名中的 prompt 部分从 `prompts<N>` 改为 `prompts-<prompt_names>`，避免不同 prompt 组合误复用同一份 tokenized cache
- 保留旧 `prompt_template_count` 作为未配置 `prompt_templete_index` 时的兼容 fallback；`train.py` 也兼容正确拼写别名 `prompt_template_index`

---

## config.yaml organization（2026-04-27）

**训练配置字段与分组整理：**
- `config.yaml` 中训练 variant 列表从 `variants` 改名为 `train_varients`，与 `eval_variants` 对称；`train.py` 优先读取新字段，并保留旧 `variants` 作为兼容 fallback
- `config.yaml` 重新整理为 `General settings`、`Train-related settings`、`Eval-related settings` 三个大板块，并在训练板块内细分 variant、prompt、data、history、action、optimization、LoRA 等小节
- 重排时保留现有配置值、注释掉的 `experiment_id`、注释掉的 `max_data_num` 和旧 LoRA target_modules 示例

---

## episode sampling split semantics（2026-04-27）

**`episode_keep_ratio` 改为先抽 pool 再划分 train/val：**
- PointMaze dataset 现在先按 `floor(total_episodes * episode_keep_ratio)` 随机无放回抽取 episode pool（至少 1 条）
- 在抽取出的 pool 内再按 `floor(pool_size * train_data_ratio)` 划分 train episodes，剩余 episodes 全部作为 val
- `episode_keep_ratio: 1` 现在会使用全部 episodes 作为 pool，并按 `train_data_ratio` 正常产生 val，不再出现“train 预留全部 episode 后 val 剩余 0”的 fallback
- 多 variant balance 现在对齐的是 sampled episode pool 大小；相同 `train_data_ratio` 下各 variant 的 train split 规模随之对齐

---

## gaussian bin local soft-label window（2026-04-27）

**`gaussian_bin` 支持只训练中心附近的 action bins：**
- 新增 `action_soft_label_radius` 配置项；例如 `2` 表示每个动作位置只训练真实 bin、左侧 2 个 bin、右侧 2 个 bin
- radius 模式下 loss 只 gather 窗口内 action token logits，并只在该窗口内做 softmax；窗口外 action token 不参与该位置的 loss，也不会收到该位置的梯度
- 未设置 `action_soft_label_radius` 时保持原行为：在全部 action bins 上计算 Gaussian soft-label CE

---

## PointMaze local dataset + official generation（2026-04-28）

**新增 local / remote variant 数据源区分：**
- `data/pointmaze/variants.py`：remote variant 补充 `varient_type: "remote"`；新增 local variant 元数据，使用 `dataset_path`、`env_paras` 和本地 `maze_map`
- `data/pointmaze/dataset.py`：remote 继续通过 `minari.load_dataset(..., download=True)` 读取；local 通过本地 Minari dataset 的 `data/` 目录读取
- local dataset cache 文件名加入 `localsteps<total_steps>` 签名，避免本地数据扩展后误读旧 tokenized cache
- `evaluate.py`：local variant 评估时用 `env_paras` 创建环境，并保留 `env_kwargs` 覆盖能力

**新增 9 个正式本地 PointMaze layout：**
- 新增 `local-layout-01` 到 `local-layout-09`
- 地图尺寸覆盖 8 到 14 行列范围
- 每行地图右侧增加 `#` / `.` 可视化注释，便于直接在代码里检查墙体和通路
- 移除临时 `custom-open-01` / `custom-open-02` variant

**官方 PointMaze 数据生成接入：**
- 新增 `local_varient_gen.py`，核心生成逻辑 import Farama 官方 `minari-dataset-generation-scripts` 的 PointMaze 代码
- 官方仓库作为 git submodule 放在 `third_party/minari-dataset-generation-scripts`
- 保留项目侧薄封装：variant 选择、本地输出路径、并行 shard、合并、覆盖已有 dataset、读取已有进度
- 数据生成目标从 `--target-steps` 改为 `--target-episodes`，按成功 episode 数收集数据
- 动作噪声和 controller 逻辑沿用官方脚本方式

**独立数据生成环境：**
- `environment.yaml` 改名为 `dataGen_env.yaml`
- data generation 使用独立 `d4rl_datagen` 环境，避免污染训练环境 `llm_offline`

---

## episode_keep_num sampling（2026-04-28）

**`episode_keep_ratio` 改为 `episode_keep_num`：**
- `config.yaml` 改为 `episode_keep_num: 5000`
- `train.py` 和 `data/pointmaze/dataset.py` 不再按比例保留 episode，而是按具体 episode 数上限抽样
- 如果真实 episode 数少于 `episode_keep_num`，自动使用全部 episode
- 不配置或配置为 `null` 时使用全部 episode
- `train.py` 对旧字段 `episode_keep_ratio` 做显式报错，避免配置被静默忽略
- `AGENTS.md` 和 `DESIGN.md` 已同步更新说明

---

## file-based training progress（2026-04-28）

**训练进度条改为文件快照：**
- 新增 `utils/file_progress.py`，提供可复用 `FileProgress`，用 `seek()` / `write()` / `truncate()` / `flush()` 覆盖写入单行进度快照
- `train.py` 删除原有 stdout carriage-return progress helper，train / val batch 进度统一调用 `FileProgress.update(...)`
- 每次训练创建 `progress/<uuid>.txt`，启动后只打印一次进度文件路径；epoch summary、warning、checkpoint 和 eval 输出继续正常 print
- 正常训练结束并保存 final checkpoint 后删除进度文件，异常退出时保留最后一次进度快照
- `.gitignore` 新增 `progress/`

---

## dataset build/cache refactor（2026-04-30）

**PointMaze dataset 构建接口统一为 batch 路径：**
- `train.py` 统一通过 `dataset_cls.build_batch(...)` 构造所有 selected variants 的 train/val dataset
- `data/base_dataset.py` 明确 `DatasetBuildRequest`、样本 schema、`collate_fn` padding 规则和 `build_batch` 契约
- `PointMazeDataset` 的直接构造只表示已加载样本容器；offline load/tokenize/cache 逻辑统一集中在 `build_batch`

**dataset tokenization 从线程并行改为进程并行：**
- PointMaze tokenization 改为 `ProcessPoolExecutor`，避免 Python 线程在 tokenizer-heavy 工作上的 GIL/资源争用
- 多 variant、train/val cache miss 共用一个进程池；worker 初始化时只加载一次 tokenizer
- 并行粒度改为 episode payload，worker 在单个 episode 内处理所有 timestep 和 prompt templates
- worker 通过 `job_id` 区分不同 variant 的 prompt、history、action 配置；同一批 pending jobs 要求 tokenizer/action-bin schema 一致

**新增 `MultiWorkerFileProgress`：**
- `utils/file_progress.py` 新增 `MultiWorkerFileProgress`，支持一个总进度文件叠加多个 worker 子进度
- dataset tokenization 现在只打印一个 joint progress path：`Tokenizing pointmaze datasets`
- worker 子进度会显示当前处理的 variant、cache job、episode 和样本/step 信息

**PointMaze cache 改为 episode 级、不分 split：**
- tokenized cache 保存为 `episode_idx -> tokenized samples`，保留 episode 边界
- cache 文件名不再包含 `train/val`、`train_data_ratio`、`episode_keep_num`、`sampling_seed`、`max_data_num`
- cache 命中后重新按当前 `episode_keep_num`、`train_data_ratio`、`sampling_seed` 和 balance 配置选择 episode 并切分 train/val
- 如果 cache 不覆盖当前 sampled episodes，则重新 tokenize 当前 sampled pool 并覆盖同一个 variant 级 cache
- `max_data_num` 只截断最终返回 dataset，不影响 cache 内容和 cache 命中判断

---

## action-token checkpoint / local dataset / tooling（2026-04-30）

**action-bin special token checkpoint 修复：**
- `bin` / `gaussian_bin` 模式下 LoRA 保存加入 `embed_tokens` 和 `lm_head`
- 加载 action-bin checkpoint 时检查旧 checkpoint 是否缺少 action token 相关权重，并给出明确错误
- tokenizer 注册 action tokens 后会同步 resize embedding，避免 eval 时 action token 概率均匀、持续解析失败

**本地 PointMaze HDF5 fallback：**
- local dataset 读取先尝试 Minari reader；如果当前 Minari reader 无法识别已有 `main_data.hdf5` 布局，则 fallback 直接读取 HDF5 episode
- local cache 签名继续使用数据步数，避免本地数据变化后误复用旧 cache

**Unsloth import banner 静音：**
- `train.py` / `model/policy.py` 对 Unsloth import 的 stdout 做 redirect
- 只屏蔽 import 阶段 banner，不屏蔽训练/评估过程中的正常日志

**project-changelog skill 规则更新：**
- `skills/project-changelog/SKILL.md` 明确：记录 changelog 前必须先给用户看总结，并等用户同意后才能写入
