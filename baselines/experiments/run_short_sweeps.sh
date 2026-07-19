#!/usr/bin/env bash
set -u

if [[ $# -gt 0 ]]; then
  wait_pid="$1"
  printf '[sweep queue] waiting for pid %s\n' "${wait_pid}"
  while [[ -d "/proc/${wait_pid}" ]]; do
    sleep 30
  done
fi

migrate_completed_run() {
  local old_dir="$1"
  local new_dir="$2"
  if [[ -f "${old_dir}/summary.json" && ! -e "${new_dir}" ]]; then
    printf '[sweep queue] migrate completed run %s -> %s\n' "${old_dir}" "${new_dir}"
    mv "${old_dir}" "${new_dir}"
  fi
}

# A previous queue invocation accidentally omitted the algorithm tag. Preserve
# its completed artifacts under the corrected reproducible experiment ids.
migrate_completed_run \
  baseline_runs/sweep-pointmaze--b-100k-s20260716 \
  baseline_runs/sweep-pointmaze-mlpbc-b-100k-s20260716
migrate_completed_run \
  baseline_runs/sweep-pointmaze--c-100k-s20260716 \
  baseline_runs/sweep-pointmaze-mlpbc-c-100k-s20260716
migrate_completed_run \
  baseline_runs/sweep-pointmaze--a-100k-s20260716 \
  baseline_runs/sweep-pointmaze-iql-a-100k-s20260716

run_sweep() {
  local family="$1"
  local algorithm="$2"
  local candidate="$3"
  local algorithm_tag="${algorithm//_/}"
  local experiment_id="sweep-${family}-${algorithm_tag}-${candidate}-100k-s20260716"
  local run_dir="baseline_runs/${experiment_id}"

  if [[ -f "${run_dir}/summary.json" ]]; then
    printf '[sweep queue] skip completed %s\n' "${experiment_id}"
    return 0
  fi
  if [[ -d "${run_dir}" ]]; then
    printf '[sweep queue] skip incomplete existing directory %s\n' "${experiment_id}"
    return 0
  fi

  printf '[sweep queue] start %s\n' "${experiment_id}"
  micromamba run -n llm_offline_baselines python baseline_train.py \
    --config \
    "baselines/configs/base.${family}.yaml" \
    "baselines/configs/${algorithm}.yaml" \
    "baselines/experiments/${family}16.sweep.yaml" \
    "baselines/experiments/${algorithm}.${candidate}.yaml" \
    --experiment_id "${experiment_id}"
  local status=$?
  printf '[sweep queue] finish %s status=%d\n' "${experiment_id}" "${status}"
  return "${status}"
}

for candidate in b c; do
  run_sweep pointmaze mlp_bc "${candidate}"
done
for candidate in a b c; do
  run_sweep antmaze mlp_bc "${candidate}"
done

for family in pointmaze antmaze; do
  for candidate in a b c; do
    run_sweep "${family}" iql "${candidate}"
  done
done

for family in pointmaze antmaze; do
  for candidate in a b c; do
    run_sweep "${family}" td3_bc "${candidate}"
  done
done
