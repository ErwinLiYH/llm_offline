#!/bin/bash

#SBATCH --job-name=llm_offline_isb_ant0.0.3
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/train-isb-ant0.0.3-%j.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/train-isb-ant0.0.3-%j.err
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
CONFIG_PATHS=(
    configs/train/config.isb.ant.0.0.0.yaml
    configs/train/config.isb.ant.0.0.3.yaml
)
RESUME_FROM_CHECKPOINT=""

usage() {
    cat <<EOF
Usage:
  sbatch $0
  sbatch $0 --resume <checkpoint_dir>

Options:
  --resume, --resume_from_checkpoint  Resume from a checkpoint directory containing trainer_state.pt.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume|--resume_from_checkpoint)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: $1 requires a checkpoint directory" >&2
                usage >&2
                exit 2
            fi
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
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

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Job ${SLURM_JOB_ID} started at $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Configs: ${CONFIG_PATHS[*]}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
else
    echo "Resume checkpoint: none"
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

TRAIN_ARGS=(
    --config "${CONFIG_PATHS[@]}"
    --experiment_id "${SLURM_JOB_ID}"
    --parallel_backend ddp
)
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

srun --ntasks=1 "${ENV_BIN}/torchrun" \
    --standalone \
    --nproc_per_node=4 \
    train.py \
    "${TRAIN_ARGS[@]}"

echo "Job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"
