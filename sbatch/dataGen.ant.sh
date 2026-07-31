#!/bin/bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_ROOT}"

TARGET_EPISODES=${TARGET_EPISODES:-2000}
MIN_SUCCESS_RATE=${MIN_SUCCESS_RATE:-0.8}
NUM_WORKERS=${NUM_WORKERS:-16}
SEED=${SEED:-42}
REWARD_TYPE=${REWARD_TYPE:-sparse}
DATASET_ROOT=${DATASET_ROOT:-}
TEMPORARY_DATASET_ROOT=${TEMPORARY_DATASET_ROOT:-}
ACTION_NOISE=${ACTION_NOISE:-0.2}
MODE=${MODE:-diverse}
DIVERSE_CELL_MODE=${DIVERSE_CELL_MODE:-all-free}
MAX_EPISODE_ATTEMPTS=${MAX_EPISODE_ATTEMPTS:-}
VARIANTS=${VARIANTS:-"local-layout-01 local-layout-02 local-layout-03 local-layout-04 local-layout-05 local-layout-06 local-layout-07 local-layout-08 local-layout-09"}

read -r -a VARIANT_ARGS <<< "${VARIANTS}"
EXTRA_ARGS=()
if [[ -n "${DATASET_ROOT}" ]]; then
    EXTRA_ARGS+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${TEMPORARY_DATASET_ROOT}" ]]; then
    EXTRA_ARGS+=(--temporary-dataset-root "${TEMPORARY_DATASET_ROOT}")
fi
if [[ -n "${MAX_EPISODE_ATTEMPTS}" ]]; then
    EXTRA_ARGS+=(--max-episode-attempts "${MAX_EPISODE_ATTEMPTS}")
fi

python local_antmaze_gen.py \
    --variants "${VARIANT_ARGS[@]}" \
    --num-workers "${NUM_WORKERS}" \
    --target-episodes "${TARGET_EPISODES}" \
    --reward-type "${REWARD_TYPE}" \
    --min-success-rate "${MIN_SUCCESS_RATE}" \
    --seed "${SEED}" \
    --action-noise "${ACTION_NOISE}" \
    --mode "${MODE}" \
    --diverse-cell-mode "${DIVERSE_CELL_MODE}" \
    --maze-solver QIteration \
    --overwrite \
    "${EXTRA_ARGS[@]}"
