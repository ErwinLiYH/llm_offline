#!/bin/bash

#SBATCH --job-name=llm_offline_eval
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/eval-%j.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/eval-%j.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=03:00:00

set -euo pipefail

PROJECT_ROOT=/home/u6mx/yl1118.u6mx/llm_offline
ENV_BIN=/home/u6mx/yl1118.u6mx/miniforge3/envs/llm_offline/bin

CONFIG_PATHS=()
MODEL_PATH=""
SEED=""
PARALLEL_BACKEND=""
NPROC_PER_NODE=1
ASSUME_YES=false
EXTRA_EVAL_ARGS=()

usage() {
    cat <<EOF
Usage:
  sbatch $0 --config <eval.yaml> [options]
  sbatch $0 --config <base.yaml> <override.yaml> [options]
  sbatch $0 --config <eval.yaml> --model_path <checkpoint_dir>
  sbatch $0 --config <eval.yaml> --model_path '<checkpoint_prefix>*'
  sbatch $0 --config <eval.yaml> --seed 42
  sbatch $0 --config <eval.yaml> --yes
  sbatch --gpus=4 --cpus-per-task=128 sbatch/evaluate.isb.slurm --config <eval.yaml> --parallel_backend ddp --nproc-per-node 4
  sbatch $0 --config <eval.yaml> -- --future-evaluate-arg value

Options:
  --config <path> [path ...]         Evaluation config(s) passed to evaluate.py.
  --model_path, --model-path <path>  Override model_path from the merged eval config.
                                     A single trailing '*' wildcard is supported; quote it in shells.
  --seed <integer>                   Override seed from the merged eval config.
  --parallel_backend, --parallel-backend <single|ddp>
                                     Override config parallel_backend. Default: use eval config.
  --nproc-per-node <n>               torchrun processes for ddp. Default: ${NPROC_PER_NODE}.
  -y, --yes                          Automatically confirm strong eval warnings.
  -h, --help                         Show this help.

Slurm resources are fixed by the #SBATCH lines in this file. Override them with
sbatch options before the script name, for example:
  sbatch --time=04:00:00 --gpus=1 sbatch/evaluate.isb.slurm --config eval.yaml

Arguments after "--" are appended directly to evaluate.py.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            shift
            if [[ $# -lt 1 || "$1" == -* ]]; then
                echo "ERROR: --config requires at least one path" >&2
                usage >&2
                exit 2
            fi
            CONFIG_PATHS=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                CONFIG_PATHS+=("$1")
                shift
            done
            ;;
        --model_path|--model-path)
            shift
            if [[ $# -lt 1 || "$1" == -* ]]; then
                echo "ERROR: --model_path requires a checkpoint path or model id" >&2
                usage >&2
                exit 2
            fi
            MODEL_PATH_VALUES=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                MODEL_PATH_VALUES+=("$1")
                shift
            done
            if [[ ${#MODEL_PATH_VALUES[@]} -ne 1 ]]; then
                echo "ERROR: --model_path expects exactly one value; quote trailing wildcards or use a more specific pattern" >&2
                exit 2
            fi
            MODEL_PATH="${MODEL_PATH_VALUES[0]}"
            ;;
        --seed)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --seed requires an integer" >&2
                usage >&2
                exit 2
            fi
            SEED="$2"
            shift 2
            ;;
        --parallel_backend|--parallel-backend)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: $1 requires single or ddp" >&2
                usage >&2
                exit 2
            fi
            PARALLEL_BACKEND="$2"
            shift 2
            ;;
        --nproc-per-node)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --nproc-per-node requires an integer" >&2
                usage >&2
                exit 2
            fi
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        -y|--yes)
            ASSUME_YES=true
            shift
            ;;
        --)
            shift
            EXTRA_EVAL_ARGS+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ${#CONFIG_PATHS[@]} -eq 0 ]]; then
    echo "ERROR: --config is required" >&2
    usage >&2
    exit 2
fi
for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "ERROR: config file not found: ${CONFIG_PATH}" >&2
        exit 2
    fi
done
if [[ -n "${PARALLEL_BACKEND}" && "${PARALLEL_BACKEND}" != "single" && "${PARALLEL_BACKEND}" != "ddp" ]]; then
    echo "ERROR: --parallel_backend must be single or ddp, got: ${PARALLEL_BACKEND}" >&2
    exit 2
fi
if [[ -n "${SEED}" ]] && ! [[ "${SEED}" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: --seed must be an integer, got: ${SEED}" >&2
    exit 2
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    echo "ERROR: --nproc-per-node must be a positive integer, got: ${NPROC_PER_NODE}" >&2
    exit 2
fi

# Do not inherit login-session TMPDIR (/local/user/<uid> -> /run/user, deleted by logind ~20s into the job)
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${PROJECT_ROOT}"

if [[ -n "${PARALLEL_BACKEND}" ]]; then
    EFFECTIVE_PARALLEL_BACKEND="${PARALLEL_BACKEND}"
else
    EFFECTIVE_PARALLEL_BACKEND="$("${ENV_BIN}/python" -c 'import sys; from utils.config_loader import load_merged_config; config = load_merged_config(sys.argv[1:]); print(config.get("parallel_backend") or "single")' "${CONFIG_PATHS[@]}")"
fi
if [[ "${EFFECTIVE_PARALLEL_BACKEND}" != "single" && "${EFFECTIVE_PARALLEL_BACKEND}" != "ddp" ]]; then
    echo "ERROR: effective parallel_backend must be single or ddp, got: ${EFFECTIVE_PARALLEL_BACKEND}" >&2
    exit 2
fi

echo "Job ${SLURM_JOB_ID:-manual} started at $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Configs: ${CONFIG_PATHS[*]}"
if [[ -n "${MODEL_PATH}" ]]; then
    echo "Model path override: ${MODEL_PATH}"
else
    echo "Model path override: config default"
fi
if [[ -n "${SEED}" ]]; then
    echo "Seed override: ${SEED}"
else
    echo "Seed override: config default"
fi
if [[ -n "${PARALLEL_BACKEND}" ]]; then
    echo "Parallel backend override: ${PARALLEL_BACKEND}"
else
    echo "Parallel backend override: config default"
fi
echo "Effective parallel backend: ${EFFECTIVE_PARALLEL_BACKEND}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "Assume yes: ${ASSUME_YES}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi

EVAL_ARGS=(--config "${CONFIG_PATHS[@]}")
if [[ -n "${MODEL_PATH}" ]]; then
    EVAL_ARGS+=(--model_path "${MODEL_PATH}")
fi
if [[ -n "${SEED}" ]]; then
    EVAL_ARGS+=(--seed "${SEED}")
fi
if [[ -n "${PARALLEL_BACKEND}" ]]; then
    EVAL_ARGS+=(--parallel_backend "${PARALLEL_BACKEND}")
fi
if [[ "${ASSUME_YES}" == true ]]; then
    EVAL_ARGS+=(--yes)
fi
EVAL_ARGS+=("${EXTRA_EVAL_ARGS[@]}")

if [[ "${EFFECTIVE_PARALLEL_BACKEND}" == "ddp" ]]; then
    srun --ntasks=1 "${ENV_BIN}/torchrun" \
        --standalone \
        --nproc_per_node="${NPROC_PER_NODE}" \
        evaluate.py \
        "${EVAL_ARGS[@]}"
else
    srun --ntasks=1 "${ENV_BIN}/python" evaluate.py "${EVAL_ARGS[@]}"
fi

echo "Job ${SLURM_JOB_ID:-manual} finished at $(date --iso-8601=seconds)"
