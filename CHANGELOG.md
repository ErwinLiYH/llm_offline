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
