---
name: llm-offline-migration-packager
description: Use this skill when the user asks to package, export, or prepare the llm_offline project for migration to another server, especially when they want a lightweight archive that excludes tokenized dataset_cache, includes local Hugging Face Qwen model cache and Minari PointMaze data, exports a conda/micromamba environment file, and writes a concise Chinese migration guide with only 解压, 环境安装, and 基本检查 sections.
---

# LLM Offline Migration Packager

## Defaults

Use this skill for `llm_offline` server migration packaging.

Default artifacts:
- `llm_offline_project_<YYYYMMDD>.tar.gz`
- `hf_cache_qwen_models_<YYYYMMDD>.tar.gz`
- `minari_pointmaze_<YYYYMMDD>.tar.gz`
- `llm_offline_env.conda.yaml`
- `llm_offline_env.explicit.txt`
- `llm_offline_pip_requirements.txt`
- `MIGRATION_GUIDE_zh.md`
- `SHA256SUMS`
- outer bundle: `llm_offline_migration_artifacts_<YYYYMMDD>.tar.gz`

Default policy:
- Do not package `dataset_cache`; it can be regenerated locally from Minari raw data.
- Also exclude `progress`, `__pycache__`, `unsloth_compiled_cache`, `results`, and `resultsV2` from the project archive.
- Include `local_datasets` and `checkpoints` in the project archive.
- Package these Hugging Face model caches by default:
  - `Qwen/Qwen3.5-0.8B`
  - `Qwen/Qwen3-0.6B`
  - `Qwen/Qwen2.5-0.5B`
- Package Minari raw PointMaze data from `~/.minari/datasets/D4RL/pointmaze`.
- Write the migration guide in concise Chinese.
- Keep exactly one path convention in the guide: extract under the target server user's `$HOME`.
- Keep guide structure to three sections only:
  - `解压`
  - `环境安装`
  - `基本检查`
- Include the first step for extracting the outer `llm_offline_migration_artifacts_<YYYYMMDD>.tar.gz` bundle.
- Do not include transfer instructions such as `rsync`/`scp`.
- Do not include `sha256sum` instructions in the guide, even though the script still creates `SHA256SUMS`.

## Workflow

1. Review current cache state if needed:

```bash
micromamba run -n llm_offline hf cache ls --filter type=model --sort name
du -sh ~/.minari/datasets/D4RL/pointmaze /home/worker/llm_offline 2>/dev/null
```

2. Run a dry run:

```bash
python3 skills/llm-offline-migration-packager/scripts/build_migration_artifacts.py --dry-run
```

3. Build artifacts:

```bash
python3 skills/llm-offline-migration-packager/scripts/build_migration_artifacts.py
```

4. Report the output directory and outer bundle path to the user.

## Script

Use `scripts/build_migration_artifacts.py` for the deterministic packaging flow. Read or patch the script only when the user asks to change defaults, such as adding/removing models, including `dataset_cache`, changing the environment name, or changing the output path.

Common options:

```bash
python3 skills/llm-offline-migration-packager/scripts/build_migration_artifacts.py \
  --project-dir /home/worker/llm_offline \
  --output-dir /home/worker/llm_offline_migration_artifacts \
  --env-name llm_offline
```

If the user explicitly wants to include tokenized cache despite the default preference:

```bash
python3 skills/llm-offline-migration-packager/scripts/build_migration_artifacts.py --include-dataset-cache
```
