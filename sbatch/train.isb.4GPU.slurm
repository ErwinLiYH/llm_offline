#!/bin/bash

#SBATCH --job-name=llm_offline_train
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/train-%j.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/train-%j.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=440G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT=/home/u6mx/yl1118.u6mx/llm_offline
ENV_BIN=/home/u6mx/yl1118.u6mx/miniforge3/envs/llm_offline/bin
NPROC_PER_NODE=4

CONFIG_PATHS=()
RESUME_FROM_CHECKPOINT=""
EXPERIMENT_ID="${SLURM_JOB_ID:-}"
EXTRA_TRAIN_ARGS=()

usage() {
    cat <<EOF
Usage:
  sbatch $0 --config <config.yaml> [options]
  sbatch $0 --config <base.yaml> <override.yaml> [options]
  sbatch $0 --config <config.yaml> --resume <checkpoint_dir>
  sbatch $0 --config <config.yaml> -- --tokenize-only

Options:
  --config <path> [path ...]          Training config(s) passed to train.py.
  --resume, --resume_from_checkpoint  Resume from a checkpoint directory containing trainer_state.pt.
  --experiment_id <id>                Override experiment id. Defaults to the Slurm job id.
  -h, --help                          Show this help.

Slurm resources are fixed by the #SBATCH lines in this file. Override them with
sbatch options before the script name, for example:
  sbatch --time=12:00:00 sbatch/train.isb.4GPU.slurm --config configs/train/example.yaml

Arguments after "--" are appended directly to train.py.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            shift
            if [[ $# -lt 1 || "$1" == --* ]]; then
                echo "ERROR: --config requires at least one path" >&2
                usage >&2
                exit 2
            fi
            CONFIG_PATHS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                CONFIG_PATHS+=("$1")
                shift
            done
            ;;
        --resume|--resume_from_checkpoint)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: $1 requires a checkpoint directory" >&2
                usage >&2
                exit 2
            fi
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        --experiment_id)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --experiment_id requires a value" >&2
                usage >&2
                exit 2
            fi
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_TRAIN_ARGS+=("$@")
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
if [[ -z "${EXPERIMENT_ID}" ]]; then
    echo "ERROR: EXPERIMENT_ID is empty and SLURM_JOB_ID is unavailable" >&2
    exit 2
fi

# Do not inherit login-session TMPDIR (/local/user/<uid> -> /run/user, deleted by logind ~20s into the job)
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${PROJECT_ROOT}"

echo "Job ${SLURM_JOB_ID:-manual} started at $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Configs: ${CONFIG_PATHS[*]}"
echo "Experiment id: ${EXPERIMENT_ID}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
else
    echo "Resume checkpoint: none"
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

TRAIN_ARGS=(
    --config "${CONFIG_PATHS[@]}"
    --experiment_id "${EXPERIMENT_ID}"
    --parallel_backend ddp
)
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
TRAIN_ARGS+=("${EXTRA_TRAIN_ARGS[@]}")

srun --ntasks=1 "${ENV_BIN}/torchrun" \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train.py \
    "${TRAIN_ARGS[@]}"

echo "Job ${SLURM_JOB_ID:-manual} finished at $(date --iso-8601=seconds)"
