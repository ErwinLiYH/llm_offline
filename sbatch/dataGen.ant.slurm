#!/bin/bash

#SBATCH --job-name=antmaze_datagen
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-ant-%A_%a.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-ant-%A_%a.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --array=0-8%4

set -euo pipefail

PROJECT_ROOT=/home/u6mx/yl1118.u6mx/llm_offline
CONDA_BIN=/home/u6mx/yl1118.u6mx/miniforge3/bin/conda
ENV_NAME=d4rl_datagen

TARGET_EPISODES=${TARGET_EPISODES:-2000}
MIN_SUCCESS_RATE=${MIN_SUCCESS_RATE:-0.8}
NUM_WORKERS=${NUM_WORKERS:-16}
SEED=${SEED:-42}
ACTION_NOISE=${ACTION_NOISE:-0.2}
MAX_EPISODE_ATTEMPTS=${MAX_EPISODE_ATTEMPTS:-}

VARIANTS=(
    local-layout-01
    local-layout-02
    local-layout-03
    local-layout-04
    local-layout-05
    local-layout-06
    local-layout-07
    local-layout-08
    local-layout-09
)

VARIANT=${VARIANTS[${SLURM_ARRAY_TASK_ID}]}

EXTRA_ARGS=()
if [[ -n "${MAX_EPISODE_ATTEMPTS}" ]]; then
    EXTRA_ARGS+=(--max-episode-attempts "${MAX_EPISODE_ATTEMPTS}")
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"

echo "Job ${SLURM_JOB_ID} started at $(date --iso-8601=seconds)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Mode: Diverse-like local AntMaze generation"
echo "Variant: ${VARIANT}"
echo "Target episodes per variant: ${TARGET_EPISODES}"
echo "Minimum saved success rate: ${MIN_SUCCESS_RATE}"
echo "Workers per variant: ${NUM_WORKERS}"
echo "Seed: ${SEED}"
echo "Action noise: ${ACTION_NOISE}"
if [[ -n "${MAX_EPISODE_ATTEMPTS}" ]]; then
    echo "Max episode attempts per variant: ${MAX_EPISODE_ATTEMPTS}"
else
    echo "Max episode attempts per variant: local_antmaze_gen.py default"
fi

"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    python -u local_antmaze_gen.py \
    --variants "${VARIANT}" \
    --num-workers "${NUM_WORKERS}" \
    --target-episodes "${TARGET_EPISODES}" \
    --min-success-rate "${MIN_SUCCESS_RATE}" \
    --seed "${SEED}" \
    --action-noise "${ACTION_NOISE}" \
    --maze-solver QIteration \
    --overwrite \
    "${EXTRA_ARGS[@]}"

echo "Job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"
