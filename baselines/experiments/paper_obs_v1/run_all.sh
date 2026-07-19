#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CFG_DIR="baselines/experiments/paper_obs_v1"
SEED=20260716

run_one() {
  local family="$1"
  local algorithm="$2"
  local algorithm_config="$3"
  local experiment_id="paperobs1-${family}-${algorithm}-e300-500k-r100-s${SEED}"
  local run_dir="baseline_runs/${experiment_id}"

  if [[ -f "${run_dir}/summary.json" ]]; then
    printf '[paper obs v1] skip completed %s\n' "${experiment_id}"
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    printf '[paper obs v1] refuse incomplete existing run %s\n' "${experiment_id}" >&2
    return 1
  fi

  printf '[paper obs v1] start %s\n' "${experiment_id}"
  micromamba run -n llm_offline_baselines python baseline_train.py \
    --config \
    "baselines/configs/base.${family}.yaml" \
    "baselines/configs/${algorithm}.yaml" \
    "${CFG_DIR}/common.yaml" \
    "${CFG_DIR}/${family}.yaml" \
    "${CFG_DIR}/${algorithm_config}" \
    --experiment_id "${experiment_id}"
  local status=$?
  printf '[paper obs v1] finish %s status=%d\n' "${experiment_id}" "${status}"
  return "${status}"
}

pids=()
run_one pointmaze mlp_bc mlp_bc.yaml & pids+=("$!")
run_one pointmaze iql iql.yaml & pids+=("$!")
run_one pointmaze td3_bc td3_pointmaze.yaml & pids+=("$!")
run_one antmaze iql iql.yaml & pids+=("$!")
run_one antmaze td3_bc td3_antmaze.yaml & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

# Loading all three AntMaze datasets together exceeds this machine's host-memory
# budget. Run MLP-BC after the first five jobs release their dataset copies.
run_one antmaze mlp_bc mlp_bc.yaml || status=1
exit "${status}"
