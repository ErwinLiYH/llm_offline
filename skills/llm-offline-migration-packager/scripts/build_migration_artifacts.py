#!/usr/bin/env python3
"""Build llm_offline migration artifacts with project-specific defaults."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys


DEFAULT_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B",
]


def run(cmd: list[str], *, dry_run: bool, stdout_path: Path | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    if stdout_path is not None:
        printable += f" > {shlex.quote(str(stdout_path))}"
    print(f"$ {printable}")
    if dry_run:
        return
    if stdout_path is None:
        subprocess.run(cmd, check=True)
        return
    with stdout_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f)


def run_capture(cmd: list[str], *, dry_run: bool) -> str:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"$ {printable}")
    if dry_run:
        return ""
    return subprocess.check_output(cmd, text=True)


def write_text(path: Path, text: str, *, dry_run: bool) -> None:
    print(f"write {path}")
    if dry_run:
        return
    path.write_text(text, encoding="utf-8")


def model_cache_dir(model_id: str, hf_home: Path) -> Path:
    return hf_home / "hub" / ("models--" + model_id.replace("/", "--"))


def require_existing(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required migration inputs:\n" + "\n".join(missing))


def relative_to_base(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError as exc:
        raise ValueError(f"{path} must be under {base} for the default migration layout") from exc


def export_environment(conda_cmd: str, env_name: str, output_dir: Path, *, dry_run: bool) -> None:
    env_yaml = output_dir / "llm_offline_env.conda.yaml"
    explicit = output_dir / "llm_offline_env.explicit.txt"
    pip_req = output_dir / "llm_offline_pip_requirements.txt"

    env_text = run_capture([conda_cmd, "env", "export", "-n", env_name, "--no-build"], dry_run=dry_run)
    if not dry_run:
        env_text = "\n".join(line for line in env_text.splitlines() if not line.startswith("prefix:")) + "\n"
    write_text(env_yaml, env_text, dry_run=dry_run)

    run([conda_cmd, "env", "export", "-n", env_name, "--explicit"], dry_run=dry_run, stdout_path=explicit)
    run([conda_cmd, "run", "-n", env_name, "python", "-m", "pip", "freeze"], dry_run=dry_run, stdout_path=pip_req)


def migration_guide(date_tag: str) -> str:
    return f"""# LLM Offline 迁移说明

如果收到的是整个迁移产物压缩包，先在目标服务器解压：

```bash
tar -xzf llm_offline_migration_artifacts_{date_tag}.tar.gz -C "$HOME"
```

解压后迁移产物目录为：

```bash
~/llm_offline_migration_artifacts
```

包含：

```text
llm_offline_project_{date_tag}.tar.gz      项目代码、配置、local_datasets、checkpoints；不包含 dataset_cache/results/progress
hf_cache_qwen_models_{date_tag}.tar.gz     Hugging Face 模型缓存：Qwen3.5-0.8B、Qwen3-0.6B、Qwen2.5-0.5B
minari_pointmaze_{date_tag}.tar.gz         Minari D4RL PointMaze 原始数据
llm_offline_env.conda.yaml               conda/micromamba 环境文件
```

本次不迁移 `dataset_cache`。目标服务器第一次训练会基于本地 Minari 原始数据重新生成它，不需要重新下载远程数据。

## 解压

在目标服务器上，将三个压缩包解压到当前用户的 `$HOME`：

```bash
cd ~/llm_offline_migration_artifacts

tar -xzf llm_offline_project_{date_tag}.tar.gz -C "$HOME"
tar -xzf hf_cache_qwen_models_{date_tag}.tar.gz -C "$HOME"
tar -xzf minari_pointmaze_{date_tag}.tar.gz -C "$HOME"
```

解压后应得到：

```text
~/llm_offline
~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B
~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B
~/.minari/datasets/D4RL/pointmaze
```

离线运行时可设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 环境安装

推荐使用 micromamba：

```bash
micromamba env create -f ~/llm_offline_migration_artifacts/llm_offline_env.conda.yaml
micromamba activate llm_offline
```

也可以使用 conda：

```bash
conda env create -f ~/llm_offline_migration_artifacts/llm_offline_env.conda.yaml
conda activate llm_offline
```

如果 `llm_offline` 环境已经存在，先检查再决定是否删除或换环境名：

```bash
micromamba env list
```

## 基本检查

进入项目：

```bash
cd ~/llm_offline
```

检查 Python、CUDA 和关键依赖：

```bash
micromamba run -n llm_offline python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
micromamba run -n llm_offline python -c "import transformers, minari, gymnasium_robotics; print('imports ok')"
```

检查模型缓存：

```bash
micromamba run -n llm_offline hf cache ls --filter type=model --sort name
```

检查 Minari 数据：

```bash
micromamba run -n llm_offline python -c "import minari; d=minari.load_dataset('D4RL/pointmaze/open-dense-v2', download=False); print(d.total_steps)"
```

启动训练：

```bash
micromamba run -n llm_offline python train.py --config config.yaml
```

多卡 DDP：

```bash
micromamba run -n llm_offline torchrun --standalone --nproc_per_node=<num_gpus> train.py --config config.yaml --parallel_backend ddp
```

Standalone eval：

```bash
micromamba run -n llm_offline python evaluate.py --config eval.yaml
```

官方 score：

```bash
micromamba run -n llm_offline python score.py --config score.yaml
```
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256(output_dir: Path, files: list[Path], *, dry_run: bool) -> None:
    lines = [f"{sha256_file(path)}  {path.name}" for path in files]
    write_text(output_dir / "SHA256SUMS", "\n".join(lines) + "\n", dry_run=dry_run)


def build(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    home = Path.home().resolve()
    hf_home = Path(os.environ.get("HF_HOME", home / ".cache" / "huggingface")).expanduser().resolve()
    minari_dir = Path(args.minari_pointmaze_dir).expanduser().resolve()
    date_tag = args.date_tag or dt.datetime.now().strftime("%Y%m%d")

    model_dirs = [model_cache_dir(model, hf_home) for model in args.models]
    required = [project_dir, minari_dir, *model_dirs]
    if not args.dry_run:
        require_existing(required)
        output_dir.mkdir(parents=True, exist_ok=True)

    project_archive = output_dir / f"llm_offline_project_{date_tag}.tar.gz"
    hf_archive = output_dir / f"hf_cache_qwen_models_{date_tag}.tar.gz"
    minari_archive = output_dir / f"minari_pointmaze_{date_tag}.tar.gz"
    guide_path = output_dir / "MIGRATION_GUIDE_zh.md"
    outer_archive = output_dir.parent / f"{output_dir.name}_{date_tag}.tar.gz"
    gzip_program = f"gzip -{args.compression_level}"

    excludes: list[str] = []
    if not args.include_dataset_cache:
        excludes.append(f"{project_dir.name}/dataset_cache")
    excludes.extend(
        [
            f"{project_dir.name}/progress",
            f"{project_dir.name}/__pycache__",
            f"{project_dir.name}/*/__pycache__",
            f"{project_dir.name}/*/*/__pycache__",
            f"{project_dir.name}/unsloth_compiled_cache",
            f"{project_dir.name}/results",
            f"{project_dir.name}/resultsV2",
        ]
    )
    project_cmd = ["tar", "-C", str(project_dir.parent), *[f"--exclude={x}" for x in excludes], "-I", gzip_program, "-cf", str(project_archive), project_dir.name]
    run(project_cmd, dry_run=args.dry_run)

    hf_sources = [relative_to_base(path, home) for path in model_dirs]
    run(["tar", "-C", str(home), "-I", gzip_program, "-cf", str(hf_archive), *hf_sources], dry_run=args.dry_run)

    minari_source = relative_to_base(minari_dir, home)
    run(["tar", "-C", str(home), "-I", gzip_program, "-cf", str(minari_archive), minari_source], dry_run=args.dry_run)

    export_environment(args.conda_command, args.env_name, output_dir, dry_run=args.dry_run)
    write_text(guide_path, migration_guide(date_tag), dry_run=args.dry_run)

    checksum_files = [
        project_archive,
        hf_archive,
        minari_archive,
        output_dir / "llm_offline_env.conda.yaml",
        output_dir / "llm_offline_env.explicit.txt",
        output_dir / "llm_offline_pip_requirements.txt",
        guide_path,
    ]
    if not args.dry_run:
        write_sha256(output_dir, checksum_files, dry_run=False)

    if not args.skip_outer_bundle:
        run(["tar", "-C", str(output_dir.parent), "-I", gzip_program, "-cf", str(outer_archive), output_dir.name], dry_run=args.dry_run)

    print(f"artifacts_dir={output_dir}")
    if not args.skip_outer_bundle:
        print(f"outer_bundle={outer_archive}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=str(Path.cwd()), help="Path to llm_offline project root.")
    parser.add_argument("--output-dir", default=str(Path.cwd().parent / "llm_offline_migration_artifacts"))
    parser.add_argument("--date-tag", default=None, help="Archive date tag, default YYYYMMDD.")
    parser.add_argument("--env-name", default="llm_offline")
    parser.add_argument("--conda-command", default="micromamba")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--minari-pointmaze-dir", default="~/.minari/datasets/D4RL/pointmaze")
    parser.add_argument("--compression-level", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--include-dataset-cache", action="store_true")
    parser.add_argument("--skip-outer-bundle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
