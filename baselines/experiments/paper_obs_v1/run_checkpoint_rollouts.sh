#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

SEED=20260716

run_one() {
  local family="$1"
  local algorithm="$2"
  local experiment_id="paperobs1-${family}-${algorithm}-e300-500k-r100-s${SEED}"
  printf '[checkpoint rollout] launch %s\n' "${experiment_id}"
  micromamba run -n llm_offline_baselines python \
    baselines/experiments/paper_obs_v1/evaluate_checkpoints.py \
    --experiment-id "${experiment_id}"
}

pids=()
for family in pointmaze antmaze; do
  for algorithm in mlp_bc iql td3_bc; do
    run_one "${family}" "${algorithm}" & pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
