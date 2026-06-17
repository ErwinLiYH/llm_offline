#!/bin/bash

#SBATCH --job-name=llm_offline_isb0.1
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/train-isb0.1-%j.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/train-isb0.1-%j.err
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
CONFIG_PATH=configs/train/config.isb.0.1.yaml

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

for index in $(seq -f "%02g" 1 9); do
    dataset="${PROJECT_ROOT}/local_datasets/pointmaze-local-layout-${index}-v0/data"
    if [[ ! -d "${dataset}" ]]; then
        echo "Missing local dataset: ${dataset}" >&2
        echo "Finish the pointmaze_datagen job before submitting this training job." >&2
        exit 1
    fi
done

echo "Job ${SLURM_JOB_ID} started at $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun --ntasks=1 "${ENV_BIN}/torchrun" \
    --standalone \
    --nproc_per_node=4 \
    train.py \
    --config "${CONFIG_PATH}" \
    --experiment_id "${SLURM_JOB_ID}" \
    --parallel_backend ddp

echo "Job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"
