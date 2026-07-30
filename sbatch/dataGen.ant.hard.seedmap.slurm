#!/bin/bash

#SBATCH --job-name=antmaze_hard_seedmap
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-ant-hard-seedmap-%j.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-ant-hard-seedmap-%j.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
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
SEED_MAP_MAX_SIZE=${SEED_MAP_MAX_SIZE:-13}
DATASET_ROOT=${DATASET_ROOT:-${SEED_MAP_DATASET_ROOT:-${PROJECT_ROOT}/local_datasets}}
TEMPORARY_DATASET_ROOT=${TEMPORARY_DATASET_ROOT:-}

NUM_WORKERS=${NUM_WORKERS:-10}
SEED=${SEED:-42}
REWARD_TYPE=${REWARD_TYPE:-sparse}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}
POLICY_FILE=${POLICY_FILE:-}
ACTION_NOISE=${ACTION_NOISE:-0.2}
MODE=${MODE:-diverse}
DIVERSE_CELL_MODE=${DIVERSE_CELL_MODE:-all-free}
MIN_SUCCESS_RATE=${MIN_SUCCESS_RATE:-1.0}
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
if [[ -n "${POLICY_FILE}" ]]; then
    EXTRA_ARGS+=(--policy-file "${POLICY_FILE}")
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
echo "Mode: ${MODE} hard-sample procedural seed-map AntMaze generation"
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
    echo "Max episode steps: local_antmaze_gen.py default"
fi
echo "Policy file override: ${POLICY_FILE:-local_antmaze_gen.py default}"
echo "Action noise: ${ACTION_NOISE}"
echo "Diverse cell mode: ${DIVERSE_CELL_MODE}"
echo "Minimum saved success rate: ${MIN_SUCCESS_RATE}"
echo "Hard retry: ${HARD_RETRY}"
echo "Hard sample alpha: ${HARD_SAMPLE_ALPHA}"
echo "Hard sample top N: ${HARD_SAMPLE_TOP_N}"
echo "Overwrite existing corpus: ${OVERWRITE}"

"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    python -u local_antmaze_gen.py \
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
    --min-success-rate "${MIN_SUCCESS_RATE}" \
    --seed "${SEED}" \
    --action-noise "${ACTION_NOISE}" \
    --mode "${MODE}" \
    --diverse-cell-mode "${DIVERSE_CELL_MODE}" \
    --maze-solver QIteration \
    --hard-sample \
    --hard-retry "${HARD_RETRY}" \
    --hard-sample-alpha "${HARD_SAMPLE_ALPHA}" \
    --hard-sample-top-n "${HARD_SAMPLE_TOP_N}" \
    "${EXTRA_ARGS[@]}"

echo "Job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"
