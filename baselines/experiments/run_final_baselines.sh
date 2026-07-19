#!/usr/bin/env bash
set -u

run_final() {
  local family="$1"
  local algorithm="$2"
  local candidate="$3"
  local algorithm_tag="${algorithm//_/}"
  local experiment_id="final-${family}-${algorithm_tag}-${candidate}-500k-s20260716"
  local run_dir="baseline_runs/${experiment_id}"

  if [[ -f "${run_dir}/summary.json" ]]; then
    printf '[final queue] skip completed %s\n' "${experiment_id}"
    return 0
  fi
  if [[ -d "${run_dir}" ]]; then
    printf '[final queue] skip incomplete existing directory %s\n' "${experiment_id}"
    return 0
  fi

  printf '[final queue] start %s\n' "${experiment_id}"
  micromamba run -n llm_offline_baselines python baseline_train.py \
    --config \
    "baselines/configs/base.${family}.yaml" \
    "baselines/configs/${algorithm}.yaml" \
    "baselines/experiments/${family}16.sweep.yaml" \
    "baselines/experiments/${algorithm}.${candidate}.yaml" \
    baselines/experiments/final-500k.yaml \
    "baselines/experiments/${family}16.full_eval.yaml" \
    --experiment_id "${experiment_id}"
  local status=$?
  printf '[final queue] finish %s status=%d\n' "${experiment_id}" "${status}"
  return "${status}"
}

# Short-screen selections:
# - MLP-BC C: lowest validation action MSE in both families.
# - IQL C: tied-best PointMaze rollout and best AntMaze offline metrics.
# - TD3+BC A: only non-divergent critic/TD-error configuration.
for family in pointmaze antmaze; do
  run_final "${family}" mlp_bc c
  run_final "${family}" iql c
  run_final "${family}" td3_bc a
done
