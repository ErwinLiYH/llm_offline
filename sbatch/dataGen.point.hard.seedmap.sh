#!/bin/bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_ROOT}"

SEED_MAP_START=${SEED_MAP_START:-0}
SEED_MAP_END=${SEED_MAP_END:-100}
TRAJECTORIES_PER_SEED=${TRAJECTORIES_PER_SEED:-50}
SEED_MAP_VERSION=${SEED_MAP_VERSION:-v1}
SEED_MAP_MIN_SIZE=${SEED_MAP_MIN_SIZE:-9}
SEED_MAP_MAX_SIZE=${SEED_MAP_MAX_SIZE:-15}
DATASET_ROOT=${DATASET_ROOT:-${SEED_MAP_DATASET_ROOT:-}}
TEMPORARY_DATASET_ROOT=${TEMPORARY_DATASET_ROOT:-}
NUM_WORKERS=${NUM_WORKERS:-10}
SEED=${SEED:-42}
REWARD_TYPE=${REWARD_TYPE:-sparse}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}
POST_SUCCESS_HOLD_STEPS=${POST_SUCCESS_HOLD_STEPS:-100}
POST_SUCCESS_HOLD_NOISE_STD=${POST_SUCCESS_HOLD_NOISE_STD:-}
HARD_RETRY=${HARD_RETRY:-5}
HARD_SAMPLE_ALPHA=${HARD_SAMPLE_ALPHA:-1.0}
HARD_SAMPLE_TOP_N=${HARD_SAMPLE_TOP_N:-400}
HARD_SAMPLE_MAX_PATH_LEN=${HARD_SAMPLE_MAX_PATH_LEN:-40}
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
if [[ -n "${POST_SUCCESS_HOLD_NOISE_STD}" ]]; then
    EXTRA_ARGS+=(--post-success-hold-noise-std "${POST_SUCCESS_HOLD_NOISE_STD}")
fi
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" || "${OVERWRITE}" == "yes" ]]; then
    EXTRA_ARGS+=(--overwrite)
fi

python local_pointmaze_gen.py \
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
    --seed "${SEED}" \
    --hard-sample \
    --hard-retry "${HARD_RETRY}" \
    --hard-sample-alpha "${HARD_SAMPLE_ALPHA}" \
    --hard-sample-top-n "${HARD_SAMPLE_TOP_N}" \
    --hard-sample-max-path-len "${HARD_SAMPLE_MAX_PATH_LEN}" \
    "${EXTRA_ARGS[@]}"
