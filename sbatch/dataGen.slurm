#!/bin/bash

#SBATCH --job-name=pointmaze_datagen
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-%A_%a.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-%A_%a.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --time=2:00:00
#SBATCH --array=0-11%4

set -euo pipefail

PROJECT_ROOT=/home/u6mx/yl1118.u6mx/llm_offline
CONDA_BIN=/home/u6mx/yl1118.u6mx/miniforge3/bin/conda
ENV_NAME=d4rl_datagen

TARGET_EPISODES=${TARGET_EPISODES:-5000}
NUM_WORKERS=${NUM_WORKERS:-10}
SEED=${SEED:-42}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}
POST_SUCCESS_HOLD_STEPS=${POST_SUCCESS_HOLD_STEPS:-100}
POST_SUCCESS_HOLD_NOISE_STD=${POST_SUCCESS_HOLD_NOISE_STD:-}
OVERWRITE=${OVERWRITE:-1}

VARIANTS=(
    local-layoutV2-01
    local-layoutV2-02
    local-layoutV2-03
    local-layoutV2-04
    local-layoutV2-05
    local-layoutV2-06
    local-layoutV2-07
    local-layoutV2-08
    local-layoutV2-09
    local-layoutV2-10
    local-layoutV2-11
    local-layoutV2-12
)

VARIANT=${VARIANTS[${SLURM_ARRAY_TASK_ID}]}

EXTRA_ARGS=()
if [[ -n "${MAX_EPISODE_STEPS}" ]]; then
    EXTRA_ARGS+=(--max-episode-steps "${MAX_EPISODE_STEPS}")
fi
if [[ -n "${POST_SUCCESS_HOLD_NOISE_STD}" ]]; then
    EXTRA_ARGS+=(--post-success-hold-noise-std "${POST_SUCCESS_HOLD_NOISE_STD}")
fi
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" || "${OVERWRITE}" == "yes" ]]; then
    EXTRA_ARGS+=(--overwrite)
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
echo "Mode: local PointMaze generation"
echo "Variant: ${VARIANT}"
echo "Target episodes per variant: ${TARGET_EPISODES}"
echo "Workers per variant: ${NUM_WORKERS}"
echo "Seed: ${SEED}"
if [[ -n "${MAX_EPISODE_STEPS}" ]]; then
    echo "Max episode steps: ${MAX_EPISODE_STEPS}"
else
    echo "Max episode steps: local_pointmaze_gen.py default"
fi
echo "Post-success hold steps: ${POST_SUCCESS_HOLD_STEPS}"
if [[ -n "${POST_SUCCESS_HOLD_NOISE_STD}" ]]; then
    echo "Post-success hold noise std: ${POST_SUCCESS_HOLD_NOISE_STD}"
else
    echo "Post-success hold noise std: local_pointmaze_gen.py default"
fi
echo "Overwrite existing dataset: ${OVERWRITE}"

"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    python -u local_pointmaze_gen.py \
    --variants "${VARIANT}" \
    --num-workers "${NUM_WORKERS}" \
    --target-episodes "${TARGET_EPISODES}" \
    --post-success-hold-steps "${POST_SUCCESS_HOLD_STEPS}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"

echo "Job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"
