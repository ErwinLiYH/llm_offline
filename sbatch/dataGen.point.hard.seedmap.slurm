#!/bin/bash

#SBATCH --job-name=pointmaze_hard_seedmap
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-point-hard-seedmap-%j.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-point-hard-seedmap-%j.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT=/home/u6mx/yl1118.u6mx/llm_offline
CONDA_BIN=/home/u6mx/yl1118.u6mx/miniforge3/bin/conda
ENV_NAME=d4rl_datagen

SEED_MAP_START=${SEED_MAP_START:-0}
SEED_MAP_END=${SEED_MAP_END:-100}
TRAJECTORIES_PER_SEED=${TRAJECTORIES_PER_SEED:-50}
SEED_MAP_VERSION=${SEED_MAP_VERSION:-v1}
SEED_MAP_MIN_SIZE=${SEED_MAP_MIN_SIZE:-9}
SEED_MAP_MAX_SIZE=${SEED_MAP_MAX_SIZE:-15}
DATASET_ROOT=${DATASET_ROOT:-${SEED_MAP_DATASET_ROOT:-${PROJECTDIR}/local_datasets}}
TEMPORARY_DATASET_ROOT=${TEMPORARY_DATASET_ROOT:-}

NUM_WORKERS=${NUM_WORKERS:-10}
SEED=${SEED:-42}
REWARD_TYPE=${REWARD_TYPE:-sparse}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}
POST_SUCCESS_HOLD_STEPS=${POST_SUCCESS_HOLD_STEPS:-0}
POST_SUCCESS_HOLD_NOISE_STD=${POST_SUCCESS_HOLD_NOISE_STD:-0.0}
HARD_RETRY=${HARD_RETRY:-5}
HARD_SAMPLE_ALPHA=${HARD_SAMPLE_ALPHA:-0.0}
HARD_SAMPLE_TOP_N=${HARD_SAMPLE_TOP_N:-200}
OVERWRITE=${OVERWRITE:-0}

EXTRA_ARGS=()
if [[ -n "${DATASET_ROOT}" ]]; then
    EXTRA_ARGS+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${TEMPORARY_DATASET_ROOT}" ]]; then
    EXTRA_ARGS+=(--temporary-dataset-root "${TEMPORARY_DATASET_ROOT}")
fi
if [[ -n "${MAX_EPISODE_STEPS}" ]]; then
    EXTRA_ARGS+=(--max-episode-steps "${MAX_EPISODE_STEPS}")
fi
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" || "${OVERWRITE}" == "yes" ]]; then
    EXTRA_ARGS+=(--overwrite)
fi

# Do not inherit login-session TMPDIR (/local/user/<uid> -> /run/user, deleted by logind ~20s into the job)
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"

echo "Job ${SLURM_JOB_ID} started at $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Mode: hard-sample procedural seed-map PointMaze generation"
echo "Map seed range: [${SEED_MAP_START}, ${SEED_MAP_END})"
echo "Trajectories per map seed: ${TRAJECTORIES_PER_SEED}"
echo "Seed-map version: ${SEED_MAP_VERSION}"
echo "Final map sizes: odd values in [${SEED_MAP_MIN_SIZE}, ${SEED_MAP_MAX_SIZE}]"
echo "Final dataset root override: ${DATASET_ROOT:-default derived path}"
echo "Temporary dataset root: ${TEMPORARY_DATASET_ROOT:-TMPDIR (${TMPDIR:-platform default})}"
echo "Workers: ${NUM_WORKERS}"
echo "Collection seed: ${SEED}"
echo "Reward type: ${REWARD_TYPE}"
if [[ -n "${MAX_EPISODE_STEPS}" ]]; then
    echo "Max episode steps: ${MAX_EPISODE_STEPS}"
else
    echo "Max episode steps: local_pointmaze_gen.py default"
fi
echo "Post-success hold steps: ${POST_SUCCESS_HOLD_STEPS}"
echo "Post-success hold noise std: ${POST_SUCCESS_HOLD_NOISE_STD}"
echo "Hard retry: ${HARD_RETRY}"
echo "Hard sample alpha: ${HARD_SAMPLE_ALPHA}"
echo "Hard sample top N: ${HARD_SAMPLE_TOP_N}"
echo "Overwrite existing corpus: ${OVERWRITE}"

"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    python -u local_pointmaze_gen.py \
    --use-seed-map \
    --seed-map-start "${SEED_MAP_START}" \
    --seed-map-end "${SEED_MAP_END}" \
    --seed-map-trajectories-per-seed "${TRAJECTORIES_PER_SEED}" \
    --seed-map-version "${SEED_MAP_VERSION}" \
    --seed-map-size-mode random \
    --seed-map-min-size "${SEED_MAP_MIN_SIZE}" \
    --seed-map-max-size "${SEED_MAP_MAX_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --reward-type "${REWARD_TYPE}" \
    --post-success-hold-steps "${POST_SUCCESS_HOLD_STEPS}" \
    --post-success-hold-noise-std "${POST_SUCCESS_HOLD_NOISE_STD}" \
    --seed "${SEED}" \
    --hard-sample \
    --hard-retry "${HARD_RETRY}" \
    --hard-sample-alpha "${HARD_SAMPLE_ALPHA}" \
    --hard-sample-top-n "${HARD_SAMPLE_TOP_N}" \
    "${EXTRA_ARGS[@]}"

echo "Job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"
