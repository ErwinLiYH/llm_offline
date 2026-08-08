#!/bin/bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_ROOT}"

SEED_MAP_START=${SEED_MAP_START:-0}
SEED_MAP_END=${SEED_MAP_END:-100}
TRAJECTORIES_PER_SEED=${TRAJECTORIES_PER_SEED:-50}
SEED_MAP_VERSION=${SEED_MAP_VERSION:-v1}
SEED_MAP_MIN_SIZE=${SEED_MAP_MIN_SIZE:-9}
SEED_MAP_MAX_SIZE=${SEED_MAP_MAX_SIZE:-13}
DATASET_ROOT=${DATASET_ROOT:-${SEED_MAP_DATASET_ROOT:-}}
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
HARD_SAMPLE_ALPHA=${HARD_SAMPLE_ALPHA:-1.0}
HARD_SAMPLE_TOP_N=${HARD_SAMPLE_TOP_N:-400}
HARD_SAMPLE_MAX_PATH_LEN=${HARD_SAMPLE_MAX_PATH_LEN:-25}
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

python local_antmaze_gen.py \
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
    --hard-sample-max-path-len "${HARD_SAMPLE_MAX_PATH_LEN}" \
    "${EXTRA_ARGS[@]}"
