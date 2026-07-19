#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CFG_DIR="baselines/experiments/paper_obs_v1"
SEED=20260716

run_one() {
  local algorithm="$1"
  local algorithm_config="$2"
  local experiment_id="paperobs1-antmaze-${algorithm}-e300-1m-r100-s${SEED}"
  local run_dir="baseline_runs/${experiment_id}"

  if [[ -f "${run_dir}/summary.json" ]]; then
    printf '[paper obs v1 antmaze 1m] skip completed %s\n' "${experiment_id}"
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    printf '[paper obs v1 antmaze 1m] refuse incomplete existing run %s\n' \
      "${experiment_id}" >&2
    return 1
  fi

  printf '[paper obs v1 antmaze 1m] start %s\n' "${experiment_id}"
  micromamba run -n llm_offline_baselines python baseline_train.py \
    --config \
    baselines/configs/base.antmaze.yaml \
    "baselines/configs/${algorithm}.yaml" \
    "${CFG_DIR}/common.yaml" \
    "${CFG_DIR}/antmaze.yaml" \
    "${CFG_DIR}/${algorithm_config}" \
    "${CFG_DIR}/one_million.yaml" \
    --experiment_id "${experiment_id}"
  local status=$?
  printf '[paper obs v1 antmaze 1m] finish %s status=%d\n' \
    "${experiment_id}" "${status}"
  return "${status}"
}

# Keep at most two copies of the 4.5M-transition AntMaze training buffer in
# memory. The original formal run used the same scheduling constraint.
pids=()
run_one iql iql.yaml & pids+=("$!")
run_one td3_bc td3_antmaze.yaml & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

# MLP-BC is faster and runs after both actor-critic jobs release their buffers.
run_one mlp_bc mlp_bc.yaml || status=1
exit "${status}"
