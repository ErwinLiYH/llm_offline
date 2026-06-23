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

---

## action-bin precision and tokenization memory fixes（2026-05-01）

**action-bin loss 精度修复：**
- `gaussian_action_loss()` 中 action-bin logits 在 `log_softmax` 前转为 float32，避免 bf16 下 `log(50)` 显示/计算成 `3.90625`
- radius/window 模式下 gather 出来的 action logits 也转为 float32
- stop-token CE 改为用 float32 logits 计算，和 HuggingFace `ForCausalLMLoss` 的 upcast 行为对齐
- 训练进度中的 `action_loss` 从 4 位改为 6 位，便于观察细微变化
- eval 记录 action-bin 概率时，softmax 前也将 bin logits 转为 float32

**tokenization worker 生命周期保护：**
- PointMaze tokenization worker 初始化时设置 Linux `PR_SET_PDEATHSIG=SIGTERM`
- 父进程 `train.py` 被 kill 或异常退出时，worker 会收到 SIGTERM，避免残留 `PPID=1` 孤儿进程继续占内存
- 增加父进程已在初始化竞态中死亡时的直接退出保护

**human-readable jsonl cache 限制：**
- `.jsonl` inspect cache 只保留每个 tokenization job 前 3 个 episode 的 `prompt/action` 文本记录
- `.pkl` 训练 cache 不变，仍保存完整 tokenized samples
- worker 返回结构调整为 `(episode_idx, text_record_or_none, token_sample)`，把 episode 归档信息和可选文本记录分离

**tokenization 阶段内存生命周期优化：**
- 创建 tokenization job 或 cache hit 后，及时释放 `selection["episodes"]` 中的完整原始 episode 引用
- 去掉 `futures = list(executor.map(...))`，改为边接收 worker 结果边归档
- 每接收一个 episode 结果后清掉对应 payload，降低原始轨迹数组保留时间
- 每 finalize 一个 job 后清理 `results_by_job`、`job.episode_payloads` 和临时 `episode_samples`，并触发 `gc.collect()` 降低峰值内存

---

## local eval horizon / training optimizer updates（2026-05-04）

**本地 PointMaze layout 新增有限 eval horizon：**
- `local-layout-*` 的 `max_episode_steps` 不再使用临时的超大值 `1000000`
- 本地 layout 现在按地图面积估算有限步数上限，避免失败 rollout 在 eval 时长时间不结束

**训练循环支持 gradient accumulation：**
- 新增 `gradient_accumulation_steps` 配置项，默认可退化为每 batch 一次 optimizer step
- 训练时按 accumulation boundary 或 epoch 最后一个 batch 执行 `optimizer.step()`
- 训练进度输出新增 `opt_step`、`accum` 和当前 learning rate，便于确认实际参数更新次数

---

## action-bin token strategy refactor（2026-05-04）

**action-bin 新增 token 路径修正：**
- `new_token: true` 时保留新增 `<act_XX>` special token 的路径，并自动把 `embed_tokens` / `lm_head` 加入 LoRA target modules
- checkpoint 保存不再对 action-bin 模式传 `save_embedding_layers=False`，避免新增 action token 的输入/输出权重没有被保存
- 移除旧的 `trainable_token_indices` 检查与相关绕路逻辑，统一通过当前 action-token 准备流程校验 tokenizer 与模型 vocab size

**新增 OpenVLA 风格复用低频 token 路径：**
- 新增 `new_token` 配置项；`bin` / `gaussian_bin` 下默认 `false`
- `new_token: false` 时不新增 tokenizer token、不 resize embedding，也不自动训练 `embed_tokens` / `lm_head`
- action bin 内部复用 tokenizer 词表末尾筛选出的稳定低频 token id；筛选时跳过 special ids，并校验 decode/tokenize roundtrip 稳定
- 新增 action-bin codec，统一维护 `bin_idx -> model token id`、`model token id -> bin_idx` 和人类可读 `<act_XX>` display token 映射

**dataset / eval 的显示层映射：**
- PointMaze tokenization 使用真实 model token id 训练，但 `.jsonl`、history prompt 和 eval step log 仍显示 `<act_XX>`
- dataset cache 文件名与 metadata 加入 `new_token` 和 action-token mapping hash，避免不同 action-token schema 误复用同一份 cache
- eval 的 bin action 解析改为优先从 generated token ids 反查 action bin，不再依赖低频 token 的 decoded 文本
- `gaussian_bin` 概率日志继续按 `<act_XX>` 展示，并附带实际 model token id 便于排查

**gaussian-bin loss 改为 full-vocab 竞争：**
- action token 位置的 Gaussian soft-label loss 改为基于 full vocabulary `log_softmax`
- action bin 不再只在 action-token 子集内部竞争，而是同时和全词表 token 竞争
- action/stop loss 中相关 logits 统一转为 float32 后计算，降低低精度训练下的数值误差

---

## OpenVLA reference notes and diagnostics scripts（2026-05-04）

**OpenVLA action tokenization 参考文档：**
- 新增 `docs/openvla_action_tokenization.md`
- 记录 OpenVLA 如何归一化动作、分 bin、复用词表末尾 token、用标准 causal LM loss 训练，以及 inference 时如何从 token id 解码动作
- 文档同时说明本项目 `new_token: false` 与 `new_token: true` 两条路径和 OpenVLA 原实现的差异

**辅助诊断脚本：**
- 新增 `scripts/added_token_lora_smoke.py`，用小模型和 PEFT 官方 notebook 数据集 smoke test 新增 special token 的 LoRA 训练、保存和加载流程
- 新增 `scripts/set_oom_score_tree.py`，递归遍历给定主进程的子进程树并设置 `oom_score_adj`，便于降低长训练任务被 OOM killer 优先选中的概率

---

## action format/parse boundary cleanup（2026-05-04）

**bin 动作解析职责集中到 ActionBinCodec：**
- 环境族 `formatting.py` 只负责 text-mode action 的 `format_action` / `parse_action` 以及所有模式共享的 `validate_action`
- bin / gaussian_bin 不再要求环境族实现 `format_action_bin_tokens` 或 `parse_action_bin_tokens`
- `ActionBinCodec` 新增 action -> bin indices、action -> display text、generated token ids -> continuous action 的封装方法
- `evaluate.py` 中 text 模式继续调用环境 formatter 解析 decoded 文本；bin 模式改为通过 codec 从 generated token ids 解码动作，并仅复用环境 formatter 做最终 action 校验
- eval 执行动作与 fallback 动作的日志显示也改为通过 codec 生成 `<act_XX>` display text，避免 display/parse 逻辑散落到环境 formatter 中

---

## training step eval（2026-05-06）

**训练期新增按 step 触发的环境评估：**
- 新增 `eval_step_interval` 配置项，默认 `0` 表示关闭；开启后按全局 train batch step 触发 step eval
- `eval_step_interval: 0` 且交互式运行时，dataloader 构建完成后会打印每个 epoch 的 batch 数和全训练 batch 数，并允许临时输入 step eval interval；非交互运行保持关闭
- 如果 step eval 触发点落在 gradient accumulation 窗口内，会等到当前窗口 `optimizer.step()` 完成后再保存 checkpoint 和运行 eval
- 如果 step eval 和 epoch eval 撞在同一个 epoch 末尾权重点，只保留 epoch checkpoint/eval，跳过重复的 step eval
- step eval 会先完整跑一次 `val_loader` 得到当前 `val_loss`，再保存 checkpoint，并复用现有 `eval_num_episodes`、`eval_variants`、日志和视频配置运行 rollout eval

**checkpoint / result 路径新增 `step<N>`：**
- epoch checkpoint/result 继续使用 `ep<N>` 和 `epoch_<N>`
- step checkpoint 写入 `checkpoints/<env_family>/<model_slug>/<selection_tag>/<experiment_id>/step<N>/`
- step eval result 写入 `<result_root>/<model_slug>/train=<env_family>-<selection_tag>/exp=<experiment_id>/step<N>/eval=<env_family>-<variant>/result.json`
- `N` 使用实际完成梯度更新后的全局 batch step，例如计划在 10000 触发但 10002 才完成更新时，目录为 `step10002`

**训练代码结构整理：**
- 训练期 rollout eval 抽为纯 `_run_eval(...)`
- `_run_training` 中 step/epoch 分支显式先 `_save_checkpoint(...)`，再按条件调用 `_run_eval(...)`，避免保存和评估混在一个组合函数里
- 训练进度中的 `opt_step` 改为当前 epoch 内的 optimizer step 计数，避免显示跨 epoch 累计值
- `DESIGN.md` 和 `AGENTS.md` 同步更新 step eval 的路径与训练期 eval 语义

---

## eval sampling and prompt alignment（2026-05-15）

**动作生成支持可控采样：**
- 新增 `action_sampling`、`action_temperature`、`action_top_p`、`action_top_k` 评估配置项；默认关闭时保持原 greedy decoding
- text 模式开启采样后使用 HuggingFace `generate()` 的普通采样路径，并继续复用现有 parse retry / fallback 逻辑
- bin / gaussian_bin 模式开启采样后只允许 action-bin token 参与生成，并固定生成 `action_dim` 个 token，避免 EOS 或普通 token 导致动作维度缺失
- 训练期 eval 透传同一套采样配置，使 `config.yaml` 和 standalone `eval.yaml` 的 rollout 行为一致

**bin 概率日志扩展：**
- `record_step_logs: true` 时，`bin` 和 `gaussian_bin` 都会记录每个动作维度上所有 action bin 的概率、display token、model token id 和连续动作中心值
- 采样开启时，日志概率反映 temperature / top-p / top-k 以及 action-token mask 后的采样分布；未采样时记录 action-bin 子集上的 softmax 分布

**修复 eval prompt 默认使用 `0.txt` 的问题：**
- 训练开始时会将 prompt 配置标准化为 `prompt_templete_index` 列表，并随 checkpoint `config.yaml` 保存，确保训练所用 prompt 可复现
- 训练期 eval 改为使用训练 prompt 列表的第一个模板，不再按文件名排序取 `0.txt`
- standalone eval 默认读取 checkpoint 中保存的训练 prompt 列表，并使用其中第一个模板
- `eval.yaml` 可用单个 `prompt_templete_index` 覆盖 standalone eval prompt；若覆盖值不在 checkpoint 训练 prompt 列表中，会打印强警告并要求输入大写 `Y`
- `evaluate.py` 新增 `-y/--yes`，用于自动确认这类强警告

**standalone eval 配置归档：**
- standalone eval 会把合并后的实际评估配置写入本次结果目录的 `eval_config.yaml`
- 归档配置包含原始 eval config 路径、`standalone_eval_id`、结果目录、resolved eval variants、checkpoint action config 和实际使用的 prompt 名

**prompt loader 清理：**
- 删除不再使用的 `load_templates()`；训练、训练期 eval、standalone eval 和 dataset 构建都改为通过 prompt 名称加载模板

---

## torchrun / DDP training support（2026-05-15）

**保留单卡默认路径并新增 DDP backend：**
- `train.py` 新增 `--parallel_backend single|ddp`，默认 `single`，现有 `python train.py --config config.yaml` 行为不变
- `config.yaml` 新增 `parallel_backend`、`ddp_find_unused_parameters`、`distributed_timeout_seconds`，DDP 通过 `torchrun` 启动并使用 NCCL process group
- 新增 `utils/distributed.py` 封装 rank/world/local_rank、barrier、rank0 打印、object broadcast、loss all-reduce 和 DDP wrapper unwrap

**DDP 数据加载与训练语义：**
- 单变种训练使用 `DistributedSampler`，多变种训练新增 `DistributedWeightedSampler`，保留原有按变种样本数加权采样语义
- DDP 下 `batch_size` 表示每 GPU micro-batch，全局有效 batch 为 `batch_size * gradient_accumulation_steps * world_size`
- gradient accumulation 期间使用 DDP `no_sync()`，只在 accumulation boundary 同步梯度
- 有 `dataset_cache_dir` 时 rank0 先构建 tokenized cache，其他 rank 在 barrier 后读取，避免多进程同时写同一 cache

**rank0-only 保存与评估：**
- checkpoint、validation、训练期 rollout eval、step logs 和视频只由 rank0 执行，其他 rank 在 barrier 等待
- DDP 保存时 unwrap 底层 Unsloth/PEFT 模型，保持 checkpoint 目录仍可被现有 `evaluate.py` 读取
- 项目内文档同步记录单节点 4 GPU GH200/H100 的 `torchrun --standalone --nproc_per_node=4` 用法，以及保持或放大全局 batch 的配置建议

---

## official normalized score entry（2026-05-20）

**新增独立 `score.py`：**
- `evaluate.py` 和训练期 eval 继续保持原有快速 success-rate 风格评估语义，不把 official normalized score 混入现有结果 schema
- 新增 `score.py` / `score.yaml`，专门用于 PointMaze official-style normalized score
- `mode: score` 运行训练 checkpoint，在每个 variant 下写 `result.json`，并在 run 根目录写 `summary.json`
- `mode: reference` 只用于 local/custom PointMaze，生成本地归一化参考分数 JSON，供后续 `score` 模式校验和使用

**配置入口统一到 YAML：**
- `score.py` 命令行只保留 `--config`，不再接受 `--mode`、`--variants`、`--num-episodes`、`--model-path` 等运行参数覆盖
- 自动确认强 prompt 警告从 `-y/--yes` 改为配置项 `assume_yes: true|false`
- 每次运行会把实际使用的 score 配置保存到结果目录的 `score_config.yaml`

**PointMaze official-style env 与 reference：**
- remote D4RL/Minari variants 使用 Farama PointMaze single-goal eval maps，强制 `continuing_task: true`、`reset_target: false`
- remote variants 使用静态 Minari metadata reference score 表，不在 scoring 时下载数据集读取 reference
- open/umaze 保持 official horizon 300，medium 600，large 800；dense variants 复用对应 map shape 与 dense env id
- local variants 要在 `local_eval_maps.<variant>.goal_cell` 显式指定 0-based goal cell；该 cell 必须是 free cell
- local reference 文件包含 `ref_min_score`、`ref_max_score`、seed、episode count、horizon、goal cell、reward type、env fingerprint 和 method metadata
- local `score` 模式会拒绝缺失 reference 或 env fingerprint 不匹配的 variant，避免用错参考分数

**共享 rollout 工具：**
- 新增 `utils/eval_rollout.py`，集中模型动作生成、prompt 渲染、history 采样、parse retry、fallback、action-bin token 解码与概率日志格式化
- `evaluate.py` 改为复用共享 rollout helper，但输出字段和评估语义保持不变
- 新增 `utils/pointmaze_score.py`，集中 normalized score 公式、remote reference lookup、official score env spec、本地 reference 校验和 env fingerprint 逻辑

**验证：**
- 新增 `tests/test_score_utils.py`，覆盖 normalized score 公式、remote reference lookup、official env horizon/kwargs、本地 goal-cell 校验、reference 缺失和 fingerprint mismatch
- 已通过 `python -m py_compile score.py evaluate.py utils/eval_rollout.py utils/pointmaze_score.py tests/test_score_utils.py`
- 已通过 `python -m unittest discover -s tests -p 'test_score_utils.py'`
- 已用 `/tmp` 输出路径完成 local reference 1 episode smoke 和 remote `open` 1 episode score smoke

---

## PointMaze map_sensing / dataset cache（2026-05-21）

**Large 地图元数据修正：**
- `data/pointmaze/variants.py` 中的 `_LARGE` 从旧的 `11x12` layout 修正为当前 Gymnasium-Robotics `PointMaze_Large-v3` / `PointMaze_LargeDense-v3` 实际使用的 `9x12` layout
- 已核对 `open`、`umaze`、`medium`、`large` 及 dense 对应变种的 prompt map、注册环境 map、score official base map 三者一致

**`map_sensing` 坐标转格子更稳健：**
- `data/pointmaze/formatting.py` 的坐标转换仍优先使用 PointMaze 原始 `floor + map_center + maze_size_scaling` 公式
- 如果原始 cell 落在墙格，则吸附到最近的 free cell center，避免贴墙或边界数值误差导致 prompt 报告墙内位置
- 四邻 `wall/free` 判断保持二值输出；如果邻格本身 free，但当前位置距离对应边界小于默认 `0.10 * maze_size_scaling` 且对角格为墙，则保守报告该方向为 `wall`

**dataset cache 文件名改为纯 hash：**
- `data/pointmaze/dataset.py` 不再生成包含 variant/tokenizer/prompt/action 配置的长文件名，cache 文件统一为 `<cache_signature_hash>.pkl` 和 `<cache_signature_hash>.jsonl`
- `cache_signature_hash` 由完整 tokenization signature payload 计算，payload 包含 variant/data signature、tokenizer/max length、prompt names/templates、prompt vars、history 配置、action 编码配置、action-token schema hash，以及 dataset/formatter/action-bin/chat-template/prompt-loader 源文件 hash
- cache metadata 只保留 `cache_format`、`cache_signature_hash`、`cache_signature_payload`、`total_episodes`、`episode_indices`；不保留旧长文件名或旧 metadata 兼容路径

---

## PointMaze sensing split（2026-05-22）

**动态 prompt 感知拆分：**
- `format_obs` 不再返回旧的混合 sensing 字段，改为返回 `location_sensing_en/zh` 和 `wall_sensing_en/zh`
- location sensing 只描述当前格子、目标格子和 1-based 行列计数规则
- wall sensing 只描述上下左右相邻格子的 `wall/free` 状态
- 共享 prompt 模板同步改为两个独立段落注入，`bin_no_sensing.txt` 保持无 sensing

---

## parallel_l1 continuous action regression（2026-05-22）

**新增连续动作模式：**
- `action_token_mode` 新增合法值 `parallel_l1`，用于 action head + L1 regression 的当前步并行动作预测
- `train.py` 在 variant selection 后调用 registry 解析 `action_dim`，写入运行时 config 并随 checkpoint `config.yaml` 保存；PointMaze 当前返回 `2`
- `data/registry.py` / `BaseOfflineDataset` 新增 `get_action_dim(env_family, variants)` 接口，后续 AntMaze 可在对应 dataset 中返回 `8`
- `config.yaml`、`DESIGN.md` 和 `AGENTS.md` 同步标注 `parallel_l1` 模式及其训练/eval 数据流

**ContinuousActionDecoder：**
- 新增 `model/continuous_action.py`，包含 `ContinuousActionDecoder(action_dim, hidden_size)`
- decoder 持有 `action_dim` 个 learned action query，追加到 prompt embedding 后，经原始 causal LM 输出最后一层 hidden states，再由 4-layer ReLU MLP action head 输出 `[batch, action_dim]`
- 新增 4D attention mask：prompt token 保持 causal；action query 可看全部非 padding prompt token；action query 之间双向可见
- `attach_continuous_action_decoder()` 将 decoder 注册到模型上，并 patch `model.forward(..., continuous_action=True)`；普通 forward 路径保持原样

**dataset / training 路径：**
- `parallel_l1` 下 PointMaze dataset 只 tokenize generation prompt，不拼 assistant action 文本，`labels` 全部为 `-100`
- 样本新增 `action_values: float32[action_dim]`，collate 时动态 stack，并拒绝混合有/无 `action_values` 的 batch
- dataset 构建时校验每条 action shape 等于配置中的 `action_dim`
- `_compute_batch_loss()` 在 `parallel_l1` 下调用 continuous forward，loss 使用 mean L1，并在训练进度中显示 `l1=...`
- `text` / `bin` / `gaussian_bin` 的原有 action token、CE 和 gaussian soft-label loss 路径保持不变

**checkpoint / eval / score：**
- `parallel_l1` checkpoint 额外保存 `continuous_action_decoder.pt`，其中包含 `action_dim`、`hidden_size` 和 decoder `state_dict`
- `load_from_checkpoint()` 读取 checkpoint `config.yaml` 的 `action_dim` 后加载 sidecar decoder，并在缺失 decoder 文件或维度不匹配时直接报错
- standalone eval 和 `score.py` 会用真实环境 `env.action_space.shape` 校验 checkpoint `action_dim`
- `parallel_l1` rollout 不调用 `model.generate()`，直接 forward 得到连续动作，clip 到环境 action bounds 后执行
- eval step log 新增 `Raw Continuous Action`，记录未 clip 的连续动作数组，便于高维动作排查

**验证：**
- 新增 `tests/test_continuous_action.py`，覆盖 `parallel_l1` mode helper、attention mask、`action_dim=2/8` forward shape、decoder 保存加载
- 已通过 `python -m py_compile train.py evaluate.py score.py model/policy.py utils/eval_rollout.py data/base_dataset.py data/registry.py data/pointmaze/dataset.py model/continuous_action.py`
- 已通过 `python -m unittest discover -s tests -p "test_continuous_action.py"`
- 已用真实 Qwen3-0.6B PointMaze smoke 跑通 1 epoch 训练、checkpoint 保存/加载和 1 episode standalone eval；该 smoke 的 trainable params 为 `4,742,145 / 857,728,065`，trainable% `0.5529`

---

## parallel_l1 OFT-style action slots（2026-05-23）

**ContinuousActionDecoder 改为 zero-placeholder action slots：**
- `parallel_l1` 不再使用可学习 `action_queries` 参数，训练/eval 都在 prompt 后显式追加 `action_dim` 个 placeholder token
- placeholder 位置通过 `action_query_mask` 标记，进入 transformer 前将对应 input embedding 强制清零；placeholder token id 只作为占位，不提供语义
- 新增 action-slot 4D attention mask：prompt 内保持 causal，action slots 可看完整非 padding prompt，且 action slots 之间双向可见
- continuous forward 直接读取 action placeholder 自身的最后层 hidden states，不走 token logits，也不再使用 causal-shift hidden 位置

**OFT-style action head：**
- action head 从逐维 `hidden -> 1` MLP 改为 MLPResNet，将 `[batch, action_dim, hidden]` flatten 为 `[batch, action_dim * hidden]` 后联合输出 `[batch, action_dim]`
- MLP 结构为 `LayerNorm -> Linear -> ReLU -> 2x residual MLP block -> LayerNorm -> Linear -> Tanh`
- 最后一层使用 PyTorch 默认初始化；输出经 `Tanh` 限制到 `[-1, 1]`
- loss 仍为 `F.l1_loss(predicted_actions, action_values, reduction="mean")`

**dataset / eval / cache 语义：**
- PointMaze `parallel_l1` tokenization 预留 `action_dim` 个长度并追加 placeholders，样本新增 `action_query_mask`
- collate 支持 `action_query_mask` 并用 `False` padding；训练和 eval continuous forward 都显式传入该 mask
- standalone eval / training eval 继承 checkpoint `max_length`，避免追加 placeholders 后超过训练时序列长度
- 移除 decoder/cache 的人为版本 tag；dataset cache tag 只区分同一代码版本下会改变 tokenized samples 的配置、数据、模板、tokenizer/action schema 等差异，不负责跨代码版本语义兼容

---

## W&B training tracking（2026-05-23）

**训练指标追踪：**
- 新增 `utils/wandb_logging.py`，封装 W&B optional import、run 初始化、metric 轴定义、日志写入和 DDP 下 env step 计数
- `train.py` 接入 W&B 日志，记录 `train/loss`、`train/epoch_loss`、`val/loss` 和 `eval/<variant>/success_rate`
- batch 级日志新增动作模式相关 loss parts：`train/l1`、`train/nll` / `train/mae` / `train/std`、`train/tnll` / `train/scale` / `train/mean_l1_aux` / `train/mean_l1_weight` / `train/df`、`train/action_loss` / `train/stop_loss`
- W&B 图表统一使用 `train/env_steps` 作为 x 轴，环境步数按已消费训练样本数除以 prompt 数折算
- DDP 下只有 rank0 初始化和写入 W&B；所有 rank 在启用时参与全局 batch sample 计数

**配置：**
- `config.yaml` 新增并启用 `wandb_enabled: true`
- 新增 `wandb_project`、`wandb_entity`、`wandb_mode`、`wandb_log_interval` 配置
- 当前实现只记录指标和配置，不上传 checkpoint、视频、step logs 或 result artifacts

---

## parallel_l1 learned query restore（2026-05-23）

**ContinuousActionDecoder 恢复内部可训练 query：**
- `parallel_l1` 从 zero-placeholder action slots 改回 decoder 内部 `action_queries: nn.Parameter[action_dim, hidden_size]`
- forward 只接收 prompt `input_ids` / `attention_mask`，先查 prompt embedding，再在末尾拼接可训练 action query embedding
- attention mask 保持 prompt 内 causal、action query 可看完整非 padding prompt、action query 之间双向可见
- continuous forward 读取最后 `action_dim` 个 hidden states，经当前 MLPResNet + `Tanh` action head 输出连续动作

**dataset / training / eval：**
- PointMaze `parallel_l1` 重新只 tokenize generation prompt，不再预留长度或追加 placeholder token
- 样本不再生成 `action_query_mask`；训练和 eval continuous forward 不再传该参数
- loss 仍为 `F.l1_loss(predicted_actions, action_values, reduction="mean")`
- 新 checkpoint 的 `continuous_action_decoder.pt` 包含 `action_queries`；zero-placeholder 版本 sidecar 会因 state dict 不兼容而加载失败

---

## parallel_gaussian continuous policy（2026-05-25）

**新增高斯连续动作模式：**
- `action_token_mode` 新增 `parallel_gaussian`，复用 `parallel_l1` 的 prompt-only dataset、learned action queries 和 4D attention mask
- `ContinuousActionDecoder` 新增 `policy_type`：`deterministic` 对应 `parallel_l1`，`gaussian` 对应 `parallel_gaussian`
- Gaussian head 输出 `mean/log_std/std`，其中 `mean` 经 `tanh` 限制到 `[-1, 1]`，`log_std` 由 `gaussian_log_std_min/max` clamp

**training / checkpoint / eval：**
- `_compute_batch_loss()` 在 `parallel_gaussian` 下使用 diagonal Gaussian NLL 做 BC，并显示 `nll`、`mae` 和平均 `std`
- `continuous_action_decoder.pt` 新增 `policy_type` 和 Gaussian log-std bounds；旧 `parallel_l1` sidecar 缺省视为 `deterministic`
- rollout 中 `parallel_gaussian` 沿用 `action_sampling`：`true` 从策略分布采样，`false` 执行 mean；执行前按环境 action bounds clip
- eval step log 在 Gaussian 模式额外记录 policy mean/std

**配置 / 文档 / 测试：**
- `config.yaml`、`DESIGN.md`、`AGENTS.md`、`eval.yaml`、`score.yaml` 标注 `parallel_gaussian` 和 log-std 配置
- `tests/test_continuous_action.py` 覆盖 Gaussian mode helper、decoder 输出形状/边界、sidecar 保存加载和 policy type mismatch

---

## continuous prompt rename（2026-05-25）

**PointMaze continuous prompt 命名去 L1 化：**
- 将 `prompts/pointmaze/parallel_l1_full_sensing.txt` 重命名为 `parallel_full_sensing.txt`
- 将 `parallel_l1_loca_sensing.txt`、`parallel_l1_no_sensing.txt`、`parallel_l1_wall_sensing.txt` 分别重命名为 `parallel_loca_sensing.txt`、`parallel_no_sensing.txt`、`parallel_wall_sensing.txt`
- `config.yaml` 默认 `prompt_templete_index` 更新为 `["parallel_full_sensing"]`
- 这次只调整 prompt 文件名；`action_token_mode: parallel_l1` 仍是确定性 L1 连续动作模式名，`parallel_gaussian` 继续复用同一组 continuous prompts

---

## experiment config snapshots（2026-05-25）

**训练启动时保存完整运行配置：**
- `train.py` 在解析完 experiment id、variant selection、action_dim、continuous action head 参数、world size 和 global effective batch size 后，立即保存运行时 config 快照
- 快照路径为 `exp_configs/<experiment_id>/config.yaml`，并额外记录 `train_config_source`
- 保存发生在模型加载、dataset 构建和正式训练之前；DDP 下只有 rank0 写入，其他 rank 通过 barrier 等待
- 新增 `utils/experiment_config.py` 和 `tests/test_experiment_config.py` 覆盖快照路径、内容和参数校验

---

## parallel_t Student-t continuous policy（2026-05-25）

**新增 t 分布连续动作模式：**
- `action_token_mode` 新增 `parallel_t`，复用 continuous prompt、prompt-only dataset、learned action queries 和 4D attention mask
- `ContinuousActionDecoder` 新增 `policy_type: student_t`，动作头输出 `mean/log_scale/scale`
- `student_t_df` 控制 Student-t 自由度，默认 `3.0`；`gaussian_log_std_min/max` 在该模式下复用为 log scale clamp
- 新增 `continuous_mean_l1_weight`，可在 `parallel_t` loss 中加入 `alpha * L1(mean, action)` 辅助均值拟合

**training / checkpoint / eval：**
- `_compute_batch_loss()` 在 `parallel_t` 下使用 Student-t NLL 做 BC，并显示 `tnll`、`mae`、`aux_l1`、平均 `scale` 和 `df`
- `continuous_action_decoder.pt` 保存并校验 `policy_type: student_t`，避免和 `parallel_gaussian` checkpoint 混用
- rollout 中 `action_sampling: true` 从 Student-t 策略采样，`false` 执行 mean；eval step log 额外记录 Student-t mean/scale

**配置 / 文档 / 测试：**
- `config.yaml` 默认示例切到 `parallel_t` 并新增 `student_t_df: 3.0`
- `DESIGN.md`、`AGENTS.md`、`eval.yaml`、`score.yaml` 标注 `parallel_t`
- `tests/test_continuous_action.py` 覆盖 mode helper、df 解析、Student-t NLL、decoder 输出形状和 sidecar 保存加载

---

## parallel_llm_bin PHT action-bin policy（2026-05-25）

**新增并行动作 bin 模式：**
- `action_token_mode` 新增 `parallel_llm_bin`，输入为 generation prompt 后追加 `action_dim` 个共享 PHT
- `new_token: false` 时 action-bin codec 额外保留一个不同于 ABT 的低频 tokenizer id 作为 PHT；`new_token: true` 时注册 `<pht>`
- PHT 之间使用双向 attention，每个 PHT hidden state 直接经原模型 `lm_head` 预测对应维度 ABT

**training / eval：**
- PointMaze dataset 为 `parallel_llm_bin` 保留 PHT 长度，`action_bin_labels` 在 PHT 位置记录目标 bin，`labels` 全部为 `-100`
- `_compute_batch_loss()` 在 PHT 位置使用 hard full-vocab CE，不做 causal shift，也不训练 chat stop token
- rollout/score 不调用 `generate()`，一次 direct forward 得到全部动作维度；`action_sampling` 在 PHT 的 ABT logits 上做采样或 greedy
- 所有 action-bin 模式新增日志指标 `bin_l1` / `train/bin_l1`，按 greedy 预测 bin center 与目标 bin center 的连续动作 MAE 计算，只用于记录，不参与训练 loss

**配置 / 文档 / 测试：**
- `config.yaml`、`DESIGN.md`、`AGENTS.md`、`eval.yaml`、`score.yaml` 标注 `parallel_llm_bin`
- `config.yaml` 默认示例切到 `action_token_mode: parallel_llm_bin`，并将训练 prompt 切到 `bin_full_sensing`
- `tests/test_continuous_action.py` 扩展覆盖 token schema、PHT attention mask、loss 对齐、PointMaze tokenization 和 direct-forward eval

---

## dimension-specific PHT for parallel_llm_bin（2026-05-26）

**逐动作维度 PHT：**
- 新增 `parallel_llm_bin_pht_mode: shared | dimension`；缺省 `shared` 兼容旧配置，当前 `config.yaml` 示例切到 `dimension`
- `dimension` 模式下 `new_token: false` 为 ABT 之外额外保留 `action_dim` 个低频 tokenizer id；`new_token: true` 注册 `<pht_0>...<pht_{D-1}>`
- `ActionBinCodec.placeholder_token_ids(action_dim)` 统一返回 shared 重复 PHT 或 dimension 逐维 PHT，dataset/eval 直接追加该列表

**缓存 / eval / 文档：**
- dataset cache signature 和 `action_token_schema_hash` 纳入 PHT mode 与 PHT token ids，避免 shared/dimension tokenized samples 混用
- human-readable dataset cache `.jsonl` 为 `parallel_llm_bin` 额外记录 `place holder` PHT 显示文本
- standalone eval 从 checkpoint config 读取 `parallel_llm_bin_pht_mode`；旧 checkpoint 未记录时按 `shared` 加载
- `DESIGN.md`、`AGENTS.md` 和测试覆盖 shared/dimension 两种 PHT token schema、PointMaze tokenization 与 direct-forward eval

---

## mtp_bin AQT action-bin policy（2026-05-27）

**从 PHT 模式切到 MTP/AQT 实现：**
- 移除 `parallel_llm_bin` / `parallel_llm_bin_pht_mode` 路径，新增 `action_token_mode: mtp_bin`
- 新增 AQT（Action Query Token）作为模型外的可训练 query embedding；AQT 不进入 tokenizer，也不叫 mask/PHT
- `mtp_k` 控制 AQT 数量，默认 `action_dim - 1`；当前实现要求 `mtp_k == action_dim - 1`
- AQT embedding、sampler head 等参数保存到独立 sidecar `mtp_bin_decoder.pt`

**training / loss：**
- PointMaze `mtp_bin` 样本 tokenize generation prompt、动作前缀 token 和 AQT metadata；`labels` 全部为 `-100`
- `action_bin_labels` 在 NTP/AQT 预测位置记录目标 ABT bin，训练时使用 base CE、sampler CE 和 LCM
- `mtp_lcm_weight` 控制 latent consistency matching 权重，默认 `1.0`
- sampler head 使用 AQT hidden state 与前一个动作 token embedding 预测对应未来 ABT
- 训练日志和 W&B 记录 `base_loss`、`sampler_loss`、`lcm_loss`，所有 action-bin 模式继续记录 `bin_l1`

**eval / decoding：**
- `mtp_bin` eval 不调用 `generate()`，直接走 forward + AQT sampler 路径
- 新增 `mtp_quadratic_decoding`；为 `false` 时关闭 verifier loop，直接信任第一次 NTP + AQT proposal
- eval / score 从 checkpoint config 继承 `mtp_k` 和 `mtp_quadratic_decoding`，`eval.yaml` / `score.yaml` 可覆盖 decoding 设置
- step log 继续用 `<act_XX>` 展示 ABT，并可记录各动作维度 bin 概率

**缓存 / 文档 / 测试：**
- dataset cache signature 纳入 `mtp_k`、action-token schema 和 `mtp_bin` 相关源码 hash
- 普通 `bin` / `gaussian_bin` 的 cache signature 不再受 `model/mtp_bin.py` 源码变化影响
- human-readable dataset cache `.jsonl` 为 `mtp_bin` 额外记录 `<aqt_i>` action query 显示字段
- `DESIGN.md`、`AGENTS.md`、`eval.yaml`、`score.yaml` 更新为 `mtp_bin` / AQT 命名
- 测试覆盖 MTP attention mask、loss 对齐、AQT tokenization、sidecar 保存加载、direct-forward eval 和 bf16 base + fp32 sampler head dtype 兼容

---

## experiment Git snapshots（2026-05-27）

**训练启动时记录源码状态：**
- `save_experiment_config_snapshot()` 现在在 `exp_configs/<experiment_id>/` 同时写入 `config.yaml`、`git.yaml` 和 `dirty.patch`
- `git.yaml` 记录 repo root、branch、HEAD commit/subject/date、`git status --porcelain`、dirty 状态、patch 大小和 sha256
- `dirty.patch` 记录当前工作区相对 HEAD 的文本 patch，覆盖 tracked 文本改动和 untracked 文本文件；ignored 文件不记录，二进制文件跳过并写入 `skipped_files`
- Git 不可用或当前目录不是 Git repo 时不中断训练，写入 `available: false` metadata 和空 patch
- `train.py` 在 rank0 打印 config、git metadata 和 dirty patch 三个快照路径

---

## training eval W&B metrics（2026-05-28）

**训练时 eval 记录平均步数：**
- `train.py` 的 `_run_eval()` 在记录 `eval/<variant>/success_rate` 时，同步向 W&B 写入 `eval/<variant>/mean_episode_steps`
- 新指标直接复用 `evaluate_variant()` 产出的 `result["mean_episode_steps"]`，与 `result.json` 和控制台 `mean_steps=` 保持一致
- 对训练过程中的 step eval 和 epoch eval 生效，并覆盖所有 `action_token_mode`；standalone `evaluate.py` 仍不初始化 W&B logging

---

## tokenized dataset partition loading（2026-05-28）

**低内存 tokenized 数据分区训练：**
- 新增 `dataset_load_partitions`，默认 `1` 保持原有一次性 tokenized dataset 加载；`>1` 时要求配置 `dataset_cache_dir`
- 原始轨迹仍按现有逻辑一次性加载并完成 episode-level train/val selection，随后每个 variant 的 train episode 按 `sampling_seed` 确定性打乱并切成固定 shard
- 分区模式下每次只 tokenize/load 一个 train tokenized shard，写入独立 cache 后释放；val split 不分区，启动时构建一次完整 val loader 并在训练、validation 和 step eval 中复用
- train shard cache signature 额外包含 split、partition count 和 partition index；完整 val cache 只记录 `split: val`，不记录 partition count/index；`dataset_load_partitions: 1` 保持旧 cache hash payload 兼容
- 分区训练的 step eval 恢复原有全局 batch 语义：触发点在梯度累积窗口内时等到 `optimizer.step()`，触发点在 train shard 中间时立即执行，只跳过与 epoch end 重合的重复 step eval
- 训练循环新增分区预热和分区训练路径，W&B/global batch/optimizer step 继续全局累计

---

## dataset cache signature simplification（2026-05-28）

**cache hash 不再记录源码版本：**
- PointMaze dataset cache signature 移除 dataset/formatter/action-bin/chat-template/prompt-loader/mtp_bin 等源码文件 hash
- cache hash 现在只区分同一代码版本下的配置、数据、tokenizer、prompt、history 和 action schema 差异
- 代码改动影响 tokenization 语义时，旧 cache 由用户手动删除

---

## epoch-local step eval cadence（2026-06-01）

**step eval 触发节奏改为每个 epoch 重置：**
- `eval_step_interval` 的触发计数从全局 batch step 改为 epoch-local batch step；每个 epoch 的第一次 step eval 触发点都是 epoch-local batch `eval_step_interval`
- step checkpoint/result 目录仍使用实际完成梯度更新后的全局 batch step `step<N>`，避免不同 epoch 的 step eval 输出互相覆盖
- 如果触发点落在 gradient accumulation 窗口内，仍会等到当前窗口 `optimizer.step()` 完成后再保存 checkpoint 和运行 eval
- 如果实际 step eval 位置落在 epoch eval 前后 `0.25 * eval_step_interval` 的 train batch 窗口内，自动跳过该 step eval，只保留 epoch checkpoint/eval

---

## continuous action MLP regularization（2026-06-02）

**只作用于连续动作 MLP head 的正则项：**
- 新增 `action_head_dropout`，默认 `0.0`；在 `parallel_l1` / `parallel_gaussian` / `parallel_t` 的 `MLPResNetActionHead` hidden path 中启用 `nn.Dropout`，仅训练模式生效，eval/rollout 下由 `model.eval()` 自动关闭
- 新增 `action_head_weight_decay`；显式配置时训练 optimizer 会拆分 param groups，只给 continuous action head 内 `ndim >= 2` 的 Linear weight 设置 AdamW weight decay
- LLM/LoRA 参数、continuous `action_queries`、bias 和 LayerNorm 参数不使用该 weight decay，保持只正则 MLP action head 的范围
- `continuous_action_decoder.pt` 记录 `action_head_dropout`，checkpoint 加载和 standalone eval 会从 checkpoint config 继承并校验该结构参数；旧 sidecar 缺省按 `0.0` 兼容
- 当前 `config.yaml` 示例为 `parallel_l1` 增加 `action_head_dropout: 0.05` 和 `action_head_weight_decay: 0.0001`

---

## LoRA decoder-layer filtering（2026-06-02）

**可按 decoder layer index 限制 LoRA 注入范围：**
- `model/policy.py` 新增读取可选 `lora_layers_to_transform`，并透传给 Unsloth / PEFT 的 `get_peft_model(..., layers_to_transform=...)`
- 未配置、注释掉或配置为 `null` 时保持旧行为：所有匹配 `lora_target_modules` 的模块都会挂 LoRA
- `config.yaml` 可通过取消注释 `lora_layers_to_transform` 示例来限制 Qwen3-0.6B 的 LoRA 层覆盖；配合 `lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]` 时，只训练指定 decoder layers 的 attention q/k/v/o
- dry-run 验证 Qwen3-0.6B 在该配置下命中 24 个 LoRA module，trainable params 为 `983,040`，用于和 Qwen3.5-0.8B 当前 6 层 full-attention LoRA 覆盖做更干净的对照
- `DESIGN.md` 和 `AGENTS.md` 同步记录该配置项及“注释掉即回到全匹配层训练”的语义

---

## parallel_gaussian squashed policy（2026-06-04）

**bounded Gaussian BC：**
- `parallel_gaussian` 从直接在 bounded action 上建模 `Normal(tanh(mean_raw), state_dependent_std)` 改为 latent-space tanh-squashed Gaussian：action head 输出 `latent_mean`，执行均值为 `tanh(latent_mean)`
- Gaussian `log_std` 改为 PPO-style state-independent `nn.Parameter[action_dim]`，由 `gaussian_log_std_init` 初始化，并在 forward/loss/eval 中按 `gaussian_log_std_min/max` clamp 后取 `exp`
- `_compute_batch_loss()` 在 `parallel_gaussian` 下先 clamp target action 到 `(-1, 1)`，再用 `atanh(action)` 作为 latent target；loss 使用 Gaussian NLL 加 tanh change-of-variables correction
- `train/nll` / `train/loss` 表示 squashed-Gaussian NLL，`train/mae` 仍为 `tanh(latent_mean)` 与 target action 的 action-space MAE，`train/std` 是 state-independent policy std 的均值

**checkpoint / eval / docs：**
- `continuous_action_decoder.pt` 保存 `gaussian_log_std_init` 和实际学到的 `gaussian_log_std` 参数；旧 state-dependent Gaussian sidecar 会给出明确不兼容提示
- rollout 中 `action_sampling: true` 改为执行 `tanh(Normal(latent_mean, std))`，`false` 执行 `tanh(latent_mean)`
- standalone eval 从 checkpoint config 继承并规范化 `gaussian_log_std_init`
- `DESIGN.md` 和 `AGENTS.md` 同步更新 `parallel_gaussian` 的 squashed Gaussian、state-independent std、loss 和日志语义

---

## score rollout videos and configs（2026-06-04）

**score.py 支持录制 rollout：**
- `score.py mode: score` 新增 `record_video`、`record_all`、`video_episode_index`、`video_fps`、`video_format` 和 `mujoco_gl`
- 录制时 score env 会以 `render_mode="rgb_array"` 创建，并把视频保存到 `score_results/score_<id>/score=pointmaze-<variant>/episode_<n>/rollout.<gif|mp4>`
- 每个 variant 的 `result.json` 新增 `video_path`、`video_paths`、`video_episode_indices`、`episode_artifact_dirs` 和 `episode_artifacts_dir`
- `utils/pointmaze_score.make_pointmaze_score_env()` 支持传入 `render_mode`，供 score 录像复用同一套 official/local score env spec

**reference / score 配置拆分：**
- 新增 `reference.yaml`，专门用于 `mode: reference`，一次性生成所有 local layout 的 reference score
- `score.yaml` 保持为模型 scoring 配置，包含 checkpoint、variant、seed、local reference root、local goal cell 和可选录像配置
- `score.py --config <yaml>` 仍是唯一 CLI 覆盖入口，运行参数继续由 YAML 承载

---

## local PointMaze post-success hold data（2026-06-04）

**本地数据生成器支持到达后保持段：**
- `local_varient_gen.py` 新增 `--post-success-hold-steps` 和 `--post-success-hold-noise-std`
- 默认 `--post-success-hold-steps 0` 保持官方式行为：first success 那一步截断 episode
- 启用 hold 后，采集环境临时使用 `reset_target=False`，first success 后继续记录固定目标下的 N 个 hold transition
- hold 阶段默认使用确定性 PD action：`10 * (desired_goal - achieved_goal) - velocity`，再 clip 到动作空间
- 启用 hold 且目标 local dataset 已存在时要求 `--overwrite`，避免把 goal-arrival-only episode 和 hold episode 混在一个数据集中
- 本地 callback 改为直接继承 Minari `StepDataCallback`，只从官方脚本复用 `WaypointController` / `QIteration`，不再 import 官方 `create_pointmaze_dataset.py` 的 check 依赖

**文档：**
- `docs/official_pointmaze_generation.md` 增加 post-success hold 生成示例，并说明启用 hold 时应覆盖重生成旧数据

---

## fixed eval episode seeds（2026-06-04）

**训练期和 standalone eval 使用固定 episode seeds：**
- `evaluate_variant()` 每个 episode 现在显式使用 `seed + ep_idx` 调用 `env.reset(seed=...)`
- 每个 episode 开始时同步重置 Python、NumPy、Torch 和 CUDA RNG，并 seed `env.action_space`，让 `action_sampling: true` 的 rollout 也更可复现
- training-time eval 从 `config.yaml` 的 `eval_seed` 读取 base seed，默认 `1`
- standalone `evaluate.py` 从 `eval.yaml` 的 `seed` 读取 base seed，默认 `1`
- eval `result.json` 新增 `seed` 和 `episode_seeds`，用于确认不同 step/epoch eval 使用同一组 episode 初始条件

---

## dataset cache parallel writes（2026-06-04）

**tokenized cache 文件写入改为线程池并行：**
- `PointMazeDataset.build_batch()` 在 tokenization jobs 完成后创建 `ThreadPoolExecutor`，并把每个 cache job 的 `.pkl` 数据缓存和 `.jsonl` 可读预览作为独立写入任务提交
- 写入线程数使用当前可见逻辑 CPU 数的一半：`max(1, (os.cpu_count() or 2) // 2)`
- cache 写入开始时打印 `[dataset] Writing dataset caches with <N> threads`，便于确认实际写入并行度
- 主流程仍会等待所有 write futures 完成；任一写入异常会在等待时抛出，避免静默生成损坏或缺失的 cache 文件
- `_finalize_tokenization_job()` 保留无 executor 时的串行写入 fallback，方便单独调用和测试

---

## step eval checkpoint-only cadence（2026-06-06）

**降低高频 step eval 的 validation / rollout 开销：**
- 新增 `step_eval_skip` 配置，默认 `1` 保持每次 step eval trigger 都执行完整 validation 和环境 rollout；设为 `n > 1` 时，每个 epoch 内仅第 `n`、`2n`、`3n` 次 trigger 执行完整评估
- 未命中完整评估频率的 trigger 仍在梯度累积窗口完成后保存 `step<N>` checkpoint，但跳过 val loss、rollout eval 和对应 W&B eval 指标
- step eval trigger 计数在每个 epoch 开始时重置，并同时应用于普通训练和 `dataset_load_partitions > 1` 的分区训练路径
- `step_eval_skip` 在训练启动时规范化并校验为大于等于 `1` 的整数；训练配置摘要和 step 日志会记录实际值、epoch 内 trigger 序号及 checkpoint-only 原因

**epoch eval 邻近规则：**
- 位于 epoch eval 前后 `0.25 * eval_step_interval` 窗口内的 step trigger 不再完全跳过，而是保存 checkpoint-only `step<N>`
- 邻近 epoch eval 时仍由 epoch checkpoint/eval 执行 validation 和 rollout，避免重复评估，同时保留中间训练状态

**文档 / 配置：**
- `config.yaml` 增加 `step_eval_skip` 示例
- `DESIGN.md` 和 `AGENTS.md` 同步记录 trigger 计数、checkpoint-only 行为及 epoch eval 邻近规则

---

## DataLoader runtime configuration and tokenize-only（2026-06-09）

**DataLoader 可配置化：**
- `config.yaml` 新增 `dataloader_config`，统一配置 train/val DataLoader 的 `num_workers`、`pin_memory`、`persistent_workers` 和 `prefetch_factor`
- 新增 `non_blocking` 控制训练 batch tensor 的设备搬运；推荐与 `pin_memory: true` 配合，为 CUDA H2D copy 与计算重叠提供条件
- 启动时规范化并打印最终 DataLoader 配置；未知字段会报错，且 `persistent_workers` / `prefetch_factor` 在 `num_workers: 0` 时禁止使用
- 单卡与 DDP、普通与 `dataset_load_partitions > 1` 的训练路径均复用同一套 DataLoader 配置；DDP 下配置按 rank 独立生效

**仅构建 tokenized cache：**
- `train.py` 新增 `--tokenize-only`，复用正常模型/tokenizer、variant selection、episode split、prompt/action schema 和 dataset cache signature
- 非 partition 模式构建或加载所选 variants 的完整 train/val cache；partition 模式先准备完整 val cache，再依次遍历并准备全部 train partition cache
- 完成后打印 partition 数、train/val sample 数和 batch 数，并在 DDP wrapping、optimizer、W&B、validation、rollout 和训练循环之前退出
- 该模式要求配置 `dataset_cache_dir`；生成的 `.pkl` cache 可由后续单卡或 DDP 训练直接复用，DDP rank/GPU 数不参与 tokenization cache signature
- 当前实现仍调用正常 `load_model_and_tokenizer()`，因此可能占用 GPU；本次不包含 CPU-only tokenizer 加载路径

**文档：**
- `AGENTS.md` 增加 `--tokenize-only` 命令、执行边界、缓存要求和 DDP 复用说明
- `DESIGN.md` 增加 tokenize-only 工作流及 partition/non-partition 行为说明

---

## Parallel rollout evaluation（2026-06-10）

- `parallel_l1`、`parallel_gaussian` 和 `parallel_t` eval 新增 `eval_parallel_episodes`，在每个 rank 内维护多个活跃 episode，并将 prompts 合批执行一次 continuous action forward。
- standalone `evaluate.py` 新增 `--parallel_backend single|ddp`；`eval_distribute_variants: true` 时 variants 按 rank 轮转分配。
- DDP 训练期 step/epoch rollout 不再局限于 rank0；checkpoint 和 validation 仍由 rank0 完成，`val_loss` 广播后各 rank 执行所属 variants，rank0 聚合结果和 W&B 指标。
- text/bin/gaussian_bin/mtp_bin 暂不做 episode 合批，在 `eval_parallel_episodes > 1` 时明确回退串行。
- 连续策略采样在固定 seed、episode 并行度、world size 和 variant 分配下可复现；改变并行配置可能改变采样轨迹。

---

## Eval logging and background video encoding（2026-06-11）

- continuous action mode 实际启用 episode 合批时，不再逐 episode 打印完成顺序或视频路径，避免不等长 episode 的乱序日志造成误解；调用端仍保留启动/结果路径信息和每个 variant 完成后的成功率汇总。非 continuous mode 回退串行时保留原有逐 episode 日志。
- eval 和 score 视频编码新增有界后台线程池：`video_save_workers` 控制并发编码线程数，`video_save_max_pending` 限制“正在编码 + 排队”的视频任务总数且不能小于 worker 数。worker 全忙但 pending 未满时仍可继续排队；pending 达到上限后下一次提交阻塞到任一任务完成。variant 返回前统一等待剩余任务并传播保存错误，`video_save_workers: 0` 可恢复同步保存。

---

## Tokenize-only planning, PointMaze variants, and episode logs（2026-06-15）

**`--tokenize-only` 输出 step eval 规划信息：**
- cache 准备完成后除 train/val sample 与 batch 汇总外，还会打印每个 epoch 的 train batch steps、跨全部 epochs 的总 train batch steps，以及 `eval_step_interval` 每个 epoch 重置的语义
- 同时按 `batch_size * world_size` 打印每个 batch step 近似处理的 global samples，便于在正式训练前规划 step eval interval
- 普通 dataset 加载和 `dataset_load_partitions > 1` 的分区 cache 预热路径使用同一套摘要

**扩展 PointMaze local/test variants：**
- 新增 `local-layout-10` 到 `local-layout-13`，继续使用本地 Minari dataset、动态 local horizon 和中英文 maze structure prompt metadata
- 新增 `test-layout-01` 到 `test-layout-03`，用于与训练 local layouts 分开的泛化测试地图
- `_build_local_variant()` 支持显式 `variant_name` 和 `env_name`，使非 `local-layout-XX` 命名也能生成对应 `local_datasets/pointmaze-<variant>-v0` 路径
- `reference.yaml` / `score.yaml` 的 local goal-cell 示例仍只覆盖 `local-layout-01..09`；对新增 layout 做 official-style local score 前需要另行补充 goal cell 并生成匹配 fingerprint 的 reference

**每个 episode 合并 step logs：**
- eval step log 从 `episode_<n>/steps/step_<n>.txt` 改为每个 episode 一个 `episode_<n>/steps.txt`
- 第一个 step 覆盖初始化文件，后续 step 按顺序追加；每段使用分割线和 `Step 0001` 形式的标题，同时保留原有 prompt、模型输出、执行动作、parse 状态、概率分布和 continuous policy metadata
- 串行 rollout 与 continuous batched rollout 继续复用同一个写入函数，显著减少长 rollout 产生的小文件和 inode 数量
- 新增测试验证单文件输出、step 顺序、分割线和旧 `steps/` 目录不再创建

---

## Official AntMaze support and shared maze sensing（2026-06-15）

**官方 D4RL/Minari AntMaze 环境族：**
- 新增 `antmaze` 环境族，支持 `umaze`、`umaze-diverse`、`medium-play`、`medium-diverse`、`large-play` 和 `large-diverse` 六个官方数据集；当前不包含本地自定义 AntMaze 地图
- 保持 Minari Gymnasium Robotics v4 数据契约：27 维本体 `observation`、2 维 `achieved_goal` / `desired_goal` 和 8 维 torque action，避免 v5 contact-force observation 改变输入维度
- 新增 AntMaze formatter、共享 prompt、registry 路由以及 `config.antmaze.yaml` / `eval.antmaze.yaml` 示例；text action 使用 8 个逗号分隔的整数百分位，bin、MTP 和 continuous action mode 复用现有通用路径
- AntMaze 的 action-bin 与 continuous prompt 各提供 full、location-only、wall-only、no-sensing 四种模板，命名与 PointMaze 的 `bin_*_sensing` / `parallel_*_sensing` 保持一致；示例训练配置默认使用 `parallel_full_sensing`
- PointMaze dataset/tokenization 管线改为按 `ENV_FAMILY`、variant metadata、formatter 和 `ACTION_DIM` 参数化，AntMaze 通过子类复用 episode split、history、partition cache、多进程 tokenization 和全部 action encoding mode
- 普通 rollout success 优先读取 `info["success"]`，兼容 continuing maze task 到达目标但不设置 `terminated=True` 的行为

**共享 location / wall sensing：**
- 新增 `utils/maze_sensing.py`，PointMaze 与 AntMaze 共用连续 xy 到迷宫格、墙格吸附、1-based location sensing 和四方向 wall sensing
- AntMaze 使用 torso `achieved_goal` 作为当前位置，并按官方 `maze_size_scaling=4.0` 计算当前格、目标格和历史格子；PointMaze 继续使用 point-mass observation 和对应地图缩放
- AntMaze 官方 eval map 可能与离线采集地图不同，包括 UMaze 中间墙朝向；rollout 创建环境后会从 `env.unwrapped.maze` 刷新 prompt map、visual map 和缩放参数
- 墙角风险逻辑调整为：移动方向邻格为 free 时，只有当前位置落入某一侧 threshold、当前同侧格为 free、而前方同侧对角格为 wall，才把该方向报告为 `wall`
- 新逻辑会预警十字路口进入窄道时新出现的墙角，但连续直道侧墙不会让贴边前进持续误报 `wall`；正前方邻格为墙时仍直接报告 `wall`
- PointMaze 与 AntMaze cache format 升级到 v2，并将 cache format 纳入 tokenization signature，避免复用包含旧 wall sensing 文本的缓存

**验证：**
- 新增六个官方 AntMaze train/eval 地图坐标转换、prompt sensing、history、8 维 action、bin/continuous tokenization 和 eval prompt-map 刷新测试
- 新增四个移动方向、左右两侧共 8 种新墙角/连续墙对照测试，并保留正前方真实墙阻塞测试

---

## AntMaze eval global videos（2026-06-16）

**AntMaze 录制默认增加全局俯视角：**
- `evaluate.py` 在 `env_family: antmaze` 且 `record_video: true` 时，除原有跟随视角 `rollout.<ext>` 外，默认额外保存 `rollout_global.<ext>`，无需新增配置项
- 全局视角使用 MuJoCo offscreen free camera，按 AntMaze 实例的 `maze.map_width`、`maze.map_length` 和 `maze_size_scaling` 自动计算俯视距离，覆盖完整地图；每帧录制后恢复原相机状态，避免影响原跟随视角
- 串行 rollout 和 `parallel_l1` / `parallel_gaussian` / `parallel_t` 的 batched continuous rollout 均支持双路视频帧采集与保存
- AntMaze standalone eval 与训练期 eval 的 episode 目录现在同时包含 `rollout.<ext>` 和 `rollout_global.<ext>`；PointMaze 与 `score.py` 视频路径保持不变
- eval `result.json` 保留原有 `video_path` / `video_paths` 指向跟随视角，并新增 `global_video_path` / `global_video_paths` / `all_video_paths` 记录全局视角和合并视频路径

**验证：**
- 已用 AntMaze `rgb_array` smoke test 验证同一步能同时捕获跟随视角和全局俯视帧
- 已用同步 GIF 写盘 smoke test 验证同一 episode 下会生成 `rollout.gif` 与 `rollout_global.gif`

---

## Training progress file lifecycle（2026-06-17）

**训练进度文件改为 run 级别：**
- train loop 的进度文件从每个 epoch 一个 `progress/<uuid>.txt` 改为每次训练一个 `progress/<experiment_id>.txt`，文件名使用最终解析后的 `experiment_id`；未在配置中显式设置时继续使用自动生成并广播后的 experiment id
- 普通训练和 `dataset_load_partitions > 1` 的分区训练都复用同一个 progress 文件跨 epoch 更新，启动时只打印一次路径
- 成功完成最终 checkpoint 和 barrier 后才打印最后一条 progress 并删除文件；训练失败时保留该文件，便于查看卡住或失败前的最后状态

**分区加载状态可见化：**
- 分区训练在每个 train shard round 加载前会强制刷新 progress 文件为 `loading data shard round i/N` 状态，并附带 `loading data ...` 说明
- `FileProgress.update(...)` 新增 `force` 参数，用于绕过刷新间隔立即写入 epoch-start 和 loading 状态；默认调用语义保持不变
- `AGENTS.md` 同步更新训练 progress 文件路径、成功清理和分区 loading 状态说明；`DESIGN.md` 同步修正为当前 run-scoped progress 语义

---

## Train system resource monitor（2026-06-17）

**latest-only 系统状态文件：**
- 新增 `resource_monitor_enabled` 和 `resource_monitor_interval_seconds` 训练配置，默认关闭；启用后 rank0 每秒覆盖写 `sys_info/<experiment_id>.txt`
- 监控覆盖普通训练和 `--tokenize-only`，从 experiment id 解析后开始，到正常结束时写入 `status: stopped`
- 输出为当前 RAM/swap 和所有 GPU 的 latest 状态，不追加历史、不写入 W&B；DDP 下只由 rank0 采样整机状态

**轻量采样实现：**
- 新增 `utils/resource_monitor.py`，RAM/swap 直接读取 `/proc/meminfo`，GPU 使用结构化 `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`
- 写文件使用临时文件加 `os.replace()` 原子覆盖，避免读取到半行状态
- GPU 查询失败或不可用时只在文件中记录 `gpu_error`，不影响训练主流程
- `config.yaml` / `config.antmaze.yaml`、`AGENTS.md`、`DESIGN.md` 和 `.gitignore` 同步更新；新增单测覆盖解析、渲染、后台写入和停止状态

---

## Slurm experiment id override（2026-06-17）

**训练启动参数：**
- `train.py` 新增 `--experiment_id` CLI 参数，可覆盖配置文件中的 `experiment_id`；传入空字符串会直接报错
- 覆盖发生在 experiment id 自动生成、DDP 广播、资源监控和运行配置快照之前，因此 checkpoint、result、W&B run name、`progress/<experiment_id>.txt`、`sys_info/<experiment_id>.txt` 和 `exp_configs/<experiment_id>/` 都使用 CLI 传入值

**Isambard sbatch 脚本：**
- 所有调用 `train.py` 的训练与 tokenize-only Slurm 脚本都会传入 `--experiment_id "${SLURM_JOB_ID}"`，使 Slurm job id 成为实验 ID，便于将 scheduler 日志、checkpoint、result 和 runtime snapshot 对齐
- 新增 `sbatch/train.isb.ant.1.slurm`，用于通过 DDP 运行 `configs/train/config.isb.ant.1.yaml`

---

## simple_mtp_bin action mode（2026-06-19）

**简化版 MTP action-bin 解码：**
- 新增 `action_token_mode: simple_mtp_bin`，作为独立 action-bin 模式；动作 query 数固定等于 `action_dim`，不需要配置 `action_query_len`、`mtp_k` 或 `mtp_quadratic_decoding`
- 训练样本构造为 `prompt + A0..A(D-2) + Q0..Q(D-1)`：NTP 路径继续训练 `Pn->A0, A0->A1, ...`，MTP 路径训练 `Q0->A0, Q1->A1, ...`
- query 侧可看完整 prompt 和所有 query，但不能看 teacher-forced action prefix，避免泄漏真实动作；simple 模式的 sampler head 使用 learned `simple_prev_embedding`，不喂真实 action token
- loss 保持 `base_loss + sampler_loss + mtp_lcm_weight * lcm_loss`，其中 simple 模式下 `base_loss` 只覆盖 NTP 位置，`sampler_loss` 覆盖 query 输出，LCM 将 `Qi` 对齐到预测同一 `Ai` 的 NTP anchor hidden
- 训练和 W&B 指标新增 `mtp_bin_l1` 与 `ntp_bin_l1`，分别记录纯 MTP query 输出和纯 NTP 路径的等效连续 MAE；原有 `bin_l1` 保留为合并 action-bin 预测的等效 MAE

**Eval 与 checkpoint 行为：**
- `simple_mtp_bin` eval 不调用 `generate()`，也不做 quadratic verifier loop；每步一次 forward，直接执行 `Q0..Q(D-1)` 的纯 MTP greedy bin 输出
- 第一版仅支持 `action_sampling: false`；开启 sampling 会直接报错
- `mtp_bin_decoder.pt` payload 新增 `mode` 字段，加载时校验 `mtp_bin` / `simple_mtp_bin`，避免两种 decoder 混用；旧 `mtp_bin` checkpoint 仍按默认 mode 兼容加载

**AntMaze Isambard 配置：**
- 新增 `configs/train/config.isb.ant.4.yaml`，使用 `simple_mtp_bin`、`bin_full_sensing`、`action_num_bins: 50`、`new_token: false` 和 `mtp_lcm_weight: 0.1`
- 新增 `sbatch/train.isb.ant.4.slurm`，沿用现有 4-GPU DDP Isambard 训练脚本风格，Slurm job/output 前缀为 `ant4`

---

## Training resume state（2026-06-19）

**训练恢复入口：**
- `train.py` 新增 `resume_from_checkpoint` 配置项和 `--resume_from_checkpoint` CLI 覆盖；值为 `null`、空值或注释掉时保持普通新训练流程
- 所有 AntMaze Isambard 训练脚本 `sbatch/train.isb.ant*.slurm` 支持可选脚本参数 `--resume <checkpoint_dir>` / `--resume_from_checkpoint <checkpoint_dir>`；不传参数时保持普通训练
- 非空路径触发 resume 时，从 checkpoint 目录加载 LoRA adapter、tokenizer、continuous/MTP sidecar decoder，并读取 `trainer_state.pt` 恢复训练状态；旧 checkpoint 若没有 `trainer_state.pt` 会直接报错
- resume 输出仍使用当前 run 的新 `experiment_id`，适合 Slurm 重新提交；checkpoint state 记录来源 checkpoint 路径和来源 `experiment_id`

**additional-epoch 语义：**
- resume 时 `num_epochs` 表示“额外训练多少个完整 epoch”，不是总 epoch 数
- 从 `epK` resume 且 `num_epochs: N` 时训练 `K+1 ... K+N`
- 从 epoch `K` 中间的 `step<N>` resume 时先补完 epoch `K`，再训练 `K+1 ... K+N`
- 从中间 `step<N>` resume 且 `num_epochs: 0` 时只补完当前 epoch

**`trainer_state.pt` 内容：**
- 每个 step、epoch 和 final checkpoint 额外保存 `trainer_state.pt`
- 保存 optimizer state、原始 LR scheduler horizon（type、base LR、warmup/decay steps、min LR ratio、total planned updates、optimizer step）、loop state（epoch、epoch-local/global batch step、step-eval trigger、partition 位置）和 compatibility metadata
- resume 后继续使用源 checkpoint 保存的原始 LR 计划，不按新配置中的 additional `num_epochs` 重新计算；超过原始 horizon 时 linear/cosine 保持在 `min_lr_ratio`
- resume 时校验 train variants、world size、batch size、gradient accumulation、action mode、action dim、partition stats、每 epoch batch 数和 optimizer param group 签名，不一致直接报错

**训练 loop 行为：**
- 普通 dataloader resume 会跳过当前 epoch 中已完成的 batch；分区训练 resume 会恢复 epoch partition order、active partition 和 partition-local batch 位置
- step checkpoint 仍只在 optimizer step 后保存，因此不恢复半个 gradient accumulation 窗口
- 触发 resume 后日志打印 `Resuming training from ...` 或 `Resuming partitioned training from ...`，包含 epoch、已完成 batch、optimizer step 和 global batch step

**文档与验证：**
- `DESIGN.md` 新增 resume 配置、命令示例、additional-epoch 规则、`trainer_state.pt` 字段说明和兼容性限制
- 新增 `tests/test_resume_training_state.py`，覆盖 epoch target 计算、step checkpoint 的 `num_epochs: 0` 行为、optimizer-boundary 校验、compatibility failure 和 LR horizon 继续使用
- 已通过 `mamba run -n llm_offline python -m pytest tests/test_resume_training_state.py tests/test_lr_scheduler.py`

---

## AntMaze observation prompt cleanup（2026-06-19）

**结构化 AntMaze 观测文本：**
- `data/antmaze/formatting.py` 的 `obs_text` 改为更接近 PointMaze 的固定分组结构：`Position`、`Goal`、`Torso`、`Velocity`、`Joints`、`JointVel`
- 关节角和关节速度从逐个 `name=value` 展开改为紧凑数组 `q=[...]` / `dq=[...]`，减少 prompt 长度
- AntMaze `obs_text` 数值统一保留两位小数，降低无意义精度和 token 开销

**同步 prompt 描述：**
- 所有 `prompts/antmaze/*.txt` 的 observation 说明改为 `Observation semantics:` 小节，明确 `x/y`、`gx/gy`、`z/quat`、`linear/angular`、`q/dq` 的含义
- 当前状态标题统一为 `Current observation:`，去掉额外的 `(Ant state summary)` 描述

---

## Parallel validation MAE（2026-06-20）

**连续动作 validation 指标：**
- `train.py` 的 validation 现在会为 `parallel_l1`、`parallel_gaussian` 和 `parallel_t` 聚合额外的 `mae` 指标
- `parallel_l1` 的 `val_mae` 使用 validation L1 loss；`parallel_gaussian` 使用 action-space mean-action MAE；`parallel_t` 使用 Student-t mean-action MAE

**日志与结果输出：**
- step/epoch validation 控制台输出新增 `val_mae`，W&B 新增 `val/mae`
- 训练期 rollout eval 的 `result.json` 保留原有 `val_loss` 字段，并在有 continuous validation MAE 时新增 `val_metrics: {"mae": ...}` 和顶层 `val_mae`

---

## DDP partition shard loading（2026-06-20）

- `dataset_workers` 语义明确为每 rank 的 tokenization worker 数量。
- `dataset_load_partitions > 1` 在 DDP 下要求 `>= world_size` 且能被 `world_size` 整除。
- 分区训练改为 rank0 规划全局 train shard，按 DDP round scatter 当前 rank 需要的 episode segment payload；val split 保持 rank0-only 且不分区。
- train shard cache 改为 flat shard samples + segment metadata，cache signature 包含 partition index/count、segment plan 和 `partition_plan_hash`；PointMaze/AntMaze cache format bump 到 v3。
- 每个 DDP round 使用本地 shard padding/replacement sampler 对齐到共同 `target_batches`，resume compat metadata 记录 plan hash 与 round stats。

---

## Isolated training eval rollout（2026-06-21）

**训练配置：**
- 新增 `training_eval_rollout_isolated`，默认 `false`；关闭时训练期 rollout 仍在训练进程内执行，保持原有行为
- 默认示例配置 `config.yaml` 与 `config.antmaze.yaml` 同步加入该配置项和注释

**`evaluate.py` 输出模式：**
- 新增 `eval_output_mode: standalone | training`，默认 `standalone`；普通 `evaluate.py --config eval.yaml` 继续写 `standalone_<eval_uuid>`
- 新增内部 `training_eval_context`，用于隔离训练期 rollout；`training` 模式直接写 checkpoint 推导出的训练 run 目录下的 `epoch_<n>` 或 `step<n>`
- `training` 模式的 `result.json` 补齐训练期 eval 字段，包括 `train_loss`、`val_loss`、`val_metrics`、`val_mae`、`experiment_id`、`eval_type`、`eval_tag`、`epoch`、`batch_step`、`epoch_step`、`optimizer_step`、`scheduled_step`、`scheduled_epoch_step`、`checkpoint_path`、`eval_rank`、`eval_world_size` 和 `eval_distribute_variants`
- 隔离 eval 的合并运行配置保存到对应 `epoch_<n>/eval_config.yaml` 或 `step<n>/eval_config.yaml`

**隔离子进程行为：**
- 开启 `training_eval_rollout_isolated: true` 后，每个训练 rank 为自己分配到的 variants 启动一个单进程 `evaluate.py` 子进程
- 子进程配置固定使用 `parallel_backend: single`、`model_path: <just-saved checkpoint>`、本 rank 的 variants、`eval_output_mode: training` 和 `wandb_enabled: false`
- 子进程环境会移除 `RANK`、`WORLD_SIZE`、`LOCAL_RANK` 等 DDP 变量；DDP 训练时把 `CUDA_VISIBLE_DEVICES` 限制到父 rank 的 `local_rank` 对应 GPU，避免子进程误入 DDP
- 每次尝试的临时 config、stdout 和 stderr 写入 `epoch_<n>/isolated_eval/rank_<rank>/attempt_<n>.*` 或 `step<n>/isolated_eval/rank_<rank>/attempt_<n>.*`
- 子进程第一次按配置运行；如果失败且 `eval_parallel_episodes > 1`，父进程固定再尝试一次 `eval_parallel_episodes: 1` 的 serial fallback；fallback 也失败时只 warning 并继续训练
- rank0 对失败 variant 记录 W&B `eval/<variant>/rollout_failed=1` 和 `eval/<variant>/isolated_attempts`，不写假的 success rate

**修复与验证：**
- 修复隔离 eval 子进程 config 携带无关 continuous action 字段的问题：`parallel_l1` 不再传 `gaussian_log_std_*`、`student_t_df` 或 `continuous_mean_l1_weight`，避免 checkpoint action config 校验失败
- 修复 AntMaze 录视频时 `eval_parallel_episodes > 1` 可能触发 native MuJoCo/EGL abort 的问题：isolated training eval 首次仍按配置并行运行，失败后自动降级到 `eval_parallel_episodes: 1`；该 fallback 策略对所有环境族生效，降级仍失败则跳过 rollout 并继续训练
- Isambard AntMaze 训练配置的 `eval_parallel_episodes` 保持为 `5`，开启 isolated eval 后由失败 fallback 兜底
- 新增单元测试覆盖 standalone/training 输出路径、training context result 字段、隔离 eval 成功读取结果、失败后 serial fallback、fallback 全部失败 warning、W&B failure flag、DDP variant 分配和 mode-specific action config keys
- 已通过 `python -m py_compile evaluate.py train.py tests/test_eval_parallel.py tests/test_resume_training_state.py`
- 已通过 `git diff --check`
- 已通过 `mamba run -n llm_offline python -m pytest tests/test_eval_parallel.py tests/test_resume_training_state.py`，结果 `24 passed`
- 已用 1-GPU Slurm smoke job 验证 `training_eval_rollout_isolated: true` 的 PointMaze `parallel_l1` 短训练，产出正确 `result.json`、`eval_config.yaml` 和 `isolated_eval/rank_0/attempt_1.yaml`

---

## AntMaze local layouts and generators（2026-06-22）

**AntMaze local/custom variant 注册：**
- `data/antmaze/variants.py` 新增 `local-layout-01` 到 `local-layout-09` 以及 `test-layout-01` 到 `test-layout-04`，用于本地 AntMaze 地图、数据生成和训练/eval 接入
- local variant 使用 `varient_type: local`、`dataset_path: local_datasets/antmaze-<variant>-v0`，并同时保存 `collection_env_paras` 和带固定 `r/g` 标记的 `env_paras`
- 新增 `_maze_from_strings(...)`、`_build_local_variant(...)`、`_mark_cells(...)`、`get_antmaze_variant_type(...)` 和 `resolve_local_dataset_path(...)`，统一处理字符串地图、评测起终点、local/remote variant 类型和本地数据路径
- 9 个 local layout 分为 large-like 和 hard 两组；4 个 test layout 用作 held-out 地图，其中 `test-layout-03/04` 重新生成后避免 `2x2`、`2x3`、`3x2` 连续开放块

**AntMaze local dataset 加载：**
- `data/antmaze/dataset.py` 支持 local AntMaze 数据集，不再只读取远程 Minari D4RL dataset id
- local 数据优先通过 `MinariDataset(<dataset_path>/data)` 读取；遇到 Minari storage metadata 不完整但存在 `main_data.hdf5` 时，复用 PointMaze local HDF5 episode fallback loader
- local cache signature 新增基于 total steps 的 `localsteps<N>` 标记，避免同名 local dataset 追加或覆盖后误复用旧 tokenized cache
- local 数据缺失时直接报错并提示先运行 `local_antmaze_gen.py`

**官方风格 AntMaze 本地数据生成：**
- 新增 `local_antmaze_gen.py`，使用 Farama 官方 `minari-dataset-generation-scripts/scripts/D4RL/antmaze/controller.py` 的 `WaypointController` 和默认 `GoalReachAnt_model.zip` SAC policy 生成 local AntMaze Minari 数据
- 支持 `--variants`、`--target-episodes`、`--num-workers`、`--overwrite`、`--seed`、`--max-episode-steps`、`--policy-file`、`--maze-solver QIteration|DFS`、`--action-noise`、`--truncate-on-success`、`--min-success-rate` 和 `--max-episode-attempts`
- 多 worker 先生成临时 Minari shard，再 merge 到 `local_datasets/antmaze-<variant>-v0`；完成后清理临时 dataset id
- `StepDataCallback` 记录 `success`、`qpos`、`qvel` 和 `goal`，并可选择在首次 success 时截断 episode
- `--min-success-rate` 开启后先生成目标 episode 数；若保存集成功率不足，则继续补采，补采失败 episode 直接丢弃，补采成功 episode 随机替换保存集里一条失败 episode，最终保存数量保持 `--target-episodes`
- `generation_summary.json` 同时记录最终保存数据集成功率 `success_rate` / `saved_success_rate` 和包含丢弃失败轨迹在内的真实采样成功率 `true_success_rate`

**Maze topology metrics and inspection：**
- 新增 `utils/maze_metrics.py`，提供二维网格迷宫的通用拓扑指标：连通性、直径、最短路长度、路径转弯、路径岔路、死胡同、junction 数、cycle rank、割点、桥边、走廊长度和 `static_difficulty`
- `static_difficulty` 是静态拓扑启发式分数，综合相对路径长度、转弯率、路径岔路、割点/桥边、死胡同、墙密度和环路比例；不是 D4RL 官方 normalized score 或真实 rollout 成功率
- 新增 `inspect_antmaze_layouts.py`，可对注册的 AntMaze variants 输出表格或 JSON 指标，默认使用 eval map 中的 `r/g` 标记作为起终点

**Design-centric AntMaze layout generation：**
- 新增 `generate_antmaze_layouts.py`，按 `large-like` / `hard` profile 随机生成候选地图，并按拓扑指标筛选最佳 layout
- 生成器使用全图 spanning-corridor 骨架，再添加 loops 和 pockets，避免早期随机扩张造成“局部通路 + 大片墙体”的地图形态
- 候选选择加入空间覆盖惩罚，压制空行/空列和大面积实心墙块；hard profile 额外强惩罚开放块，避免 `2x2`、`2x3`、`3x2` 房间状连续 free 区域，同时保留长直走廊
- `--mode suite` 生成 9 个 local + 4 个 test layout；`--mode candidates` 可按 profile 生成候选；支持输出 JSON 和可复制到 `variants.py` 的 Python 片段
- 当前 seed 42 生成记录保存为 `generated_antmaze_layouts_seed42.json` 和 `generated_antmaze_layouts_seed42.py`

**验证：**
- 已通过 `python -m py_compile data/antmaze/variants.py data/antmaze/dataset.py generate_antmaze_layouts.py inspect_antmaze_layouts.py utils/maze_metrics.py local_antmaze_gen.py`
- 已通过 `python -m json.tool generated_antmaze_layouts_seed42.json`
- 已通过 `python inspect_antmaze_layouts.py --variants medium-play large-play local-layout-01 ... local-layout-09 test-layout-01 ... test-layout-04` 检查所有新增 AntMaze layout 指标和起终点合法性
- 已额外验证 `test-layout-03` 和 `test-layout-04` 的 `open_2x2=0`、`open_2x3=0`、`open_3x2=0`

---

## local_antmaze_gen.py（2026-06-22）

- `--truncate-on-success` 改为 `--mode {diverse,play}`；两种模式都采用官方 AntMaze fixed-horizon 语义，到达目标只记录 `info["success"]`，不再截断 episode
- `diverse` 模式支持在本地 collection map 上确定性选择代表性 free cells 并标为 `c` 的采样方式；`play` 模式保留所有 free cells 作为 reset/goal 候选
- `generation_summary.json` 新增 `mode` 和 `collection_combined_cells`，用于记录本次本地数据生成的采样模式和 diverse 候选 cell
- `data/antmaze/variants.py` 的 local/test dataset style 文案改为中性的 reset/goal trajectories，避免与 `--mode play` 冲突

## sbatch/dataGen.ant.slurm（2026-06-22）

- 新增 `MODE` 环境变量，默认 `diverse`，提交时传给 `local_antmaze_gen.py --mode`
- 去掉 `--truncate-on-success`，并将任务时间恢复为 `48:00:00`，适配 fixed-horizon AntMaze 数据生成

## local_antmaze_gen.py（2026-06-23）

- 新增 `--diverse-cell-mode {all-free,representative-c}`，默认 `all-free`
- `--mode diverse` 默认不再在 collection map 上标 `c`，reset/goal 改为从所有 free cells 中随机采样
- 旧的代表性 `c` cells 行为保留为显式选项：`--mode diverse --diverse-cell-mode representative-c`
- `generation_summary.json` 新增 `diverse_cell_mode`，并继续记录 `collection_combined_cells`；默认 all-free 时该列表为空

## sbatch/dataGen.ant.slurm（2026-06-23）

- 新增 `DIVERSE_CELL_MODE` 环境变量，默认 `all-free`，提交给 `local_antmaze_gen.py --diverse-cell-mode`

## docs（2026-06-23）

- 新增 `docs/official_maze_dataset_semantics.md`，记录官方 PointMaze 与 AntMaze 数据语义差异、AntMaze `play` / `diverse` 的 reset/goal 采样区别，以及当前本地 AntMaze 生成与官方数据的对齐和差异

## AntMaze training data preprocessing（2026-06-23）

- `config.antmaze.yaml` 新增 `antmaze_data_config`，默认 `filter_success: false`、`truncate: false`、`truncate_holding: 0`，仅在 `env_family: antmaze` 的训练数据构建中生效
- AntMaze raw episodes 加载后、`episode_keep_num` 抽样和 train/val split 前，可按原始 `infos.success` 过滤失败 episode，并可在第一次 success 或保守翻车事件后截断轨迹
- 翻车检测使用 action 后状态的 torso z 与 quaternion body-up 方向，避免用 `z > 1.0` 误删正常跳跃；截断会同步切分 observations/actions/rewards/terminations/truncations/infos，并在缩短且存在 truncations 时标记最后一步 truncation
- AntMaze tokenized cache signature 和 shard cache signature 纳入 normalized `antmaze_data_config`，AntMaze cache format bump 到 v4；PointMaze 默认 cache signature 不包含该配置
- AntMaze local HDF5 fallback loader 现在读取 `infos`、`rewards`、`terminations` 和 `truncations`，供预处理和后续数据统计使用
