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
