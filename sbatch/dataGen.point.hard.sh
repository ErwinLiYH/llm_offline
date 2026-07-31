#!/bin/bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_ROOT}"

TARGET_EPISODES=${TARGET_EPISODES:-5000}
NUM_WORKERS=${NUM_WORKERS:-10}
SEED=${SEED:-42}
REWARD_TYPE=${REWARD_TYPE:-sparse}
DATASET_ROOT=${DATASET_ROOT:-}
TEMPORARY_DATASET_ROOT=${TEMPORARY_DATASET_ROOT:-}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}
POST_SUCCESS_HOLD_STEPS=${POST_SUCCESS_HOLD_STEPS:-100}
POST_SUCCESS_HOLD_NOISE_STD=${POST_SUCCESS_HOLD_NOISE_STD:-}
HARD_RETRY=${HARD_RETRY:-5}
HARD_SAMPLE_ALPHA=${HARD_SAMPLE_ALPHA:-1.0}
HARD_SAMPLE_TOP_N=${HARD_SAMPLE_TOP_N:-400}
OVERWRITE=${OVERWRITE:-1}
VARIANTS=${VARIANTS:-"local-layoutV2-01 local-layoutV2-02 local-layoutV2-03 local-layoutV2-04 local-layoutV2-05 local-layoutV2-06 local-layoutV2-07 local-layoutV2-08 local-layoutV2-09 local-layoutV2-10 local-layoutV2-11 local-layoutV2-12"}

read -r -a VARIANT_ARGS <<< "${VARIANTS}"
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
    --variants "${VARIANT_ARGS[@]}" \
    --num-workers "${NUM_WORKERS}" \
    --target-episodes "${TARGET_EPISODES}" \
    --reward-type "${REWARD_TYPE}" \
    --post-success-hold-steps "${POST_SUCCESS_HOLD_STEPS}" \
    --seed "${SEED}" \
    --hard-sample \
    --hard-retry "${HARD_RETRY}" \
    --hard-sample-alpha "${HARD_SAMPLE_ALPHA}" \
    --hard-sample-top-n "${HARD_SAMPLE_TOP_N}" \
    "${EXTRA_ARGS[@]}"
