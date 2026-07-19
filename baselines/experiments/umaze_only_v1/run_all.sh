#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PAPER_DIR="baselines/experiments/paper_obs_v1"
EXP_DIR="baselines/experiments/umaze_only_v1"
SEED=20260716

run_one() {
  local algorithm="$1"
  local algorithm_config="$2"
  local experiment_id="umazeonly1-antmaze-${algorithm}-e300-500k-r100-s${SEED}"
  local run_dir="baseline_runs/${experiment_id}"

  if [[ -f "${run_dir}/summary.json" ]]; then
    printf '[umaze only v1] skip completed %s\n' "${experiment_id}"
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    printf '[umaze only v1] refuse incomplete existing run %s\n' "${experiment_id}" >&2
    return 1
  fi

  printf '[umaze only v1] start %s\n' "${experiment_id}"
  micromamba run -n llm_offline_baselines python baseline_train.py \
    --config \
    baselines/configs/base.antmaze.yaml \
    "baselines/configs/${algorithm}.yaml" \
    "${PAPER_DIR}/common.yaml" \
    "${EXP_DIR}/umaze.yaml" \
    "${PAPER_DIR}/${algorithm_config}" \
    --experiment_id "${experiment_id}"
  local status=$?
  printf '[umaze only v1] finish %s status=%d\n' "${experiment_id}" "${status}"
  return "${status}"
}

pids=()
run_one mlp_bc mlp_bc.yaml & pids+=("$!")
run_one iql iql.yaml & pids+=("$!")
run_one td3_bc td3_antmaze.yaml & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
