#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CFG_DIR="baselines/experiments/paper_obs_v1"
SEED=20260716

run_one() {
  local algorithm="$1"
  local old_id="paperobs1-antmaze-${algorithm}-e300-500k-r100-s${SEED}"
  local new_id="paperobs1-antmaze-${algorithm}-e300-1m-r100-s${SEED}"
  local old_dir="baseline_runs/${old_id}"
  local new_dir="baseline_runs/${new_id}"
  local steps=(600000 700000 800000 900000)

  if [[ ! -f "${new_dir}/summary.json" ]]; then
    printf '[paper obs v1 antmaze 1m rollout] missing completed run %s\n' \
      "${new_id}" >&2
    return 1
  fi

  local step
  for step in 100000 200000 300000 400000 500000; do
    if ! cmp -s \
      "${old_dir}/checkpoints/step_${step}.d3" \
      "${new_dir}/checkpoints/step_${step}.d3"; then
      printf '[paper obs v1 antmaze 1m rollout] %s differs at %d; evaluate full trajectory\n' \
        "${algorithm}" "${step}"
      steps=(100000 200000 300000 400000 500000 600000 700000 800000 900000)
      break
    fi
  done

  printf '[paper obs v1 antmaze 1m rollout] launch %s steps=%s\n' \
    "${new_id}" "${steps[*]}"
  micromamba run -n llm_offline_baselines python \
    "${CFG_DIR}/evaluate_checkpoints.py" \
    --experiment-id "${new_id}" \
    --steps "${steps[@]}"
}

pids=()
for algorithm in mlp_bc iql td3_bc; do
  run_one "${algorithm}" & pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
