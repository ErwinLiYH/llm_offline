#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CFG_DIR="baselines/experiments/sparse_v2"
SEED=20260716

run_finalist() {
  local family="$1"
  local algorithm="$2"
  local candidate="$3"
  local updates="$4"
  shift 4

  local algorithm_tag="${algorithm//_/}"
  local experiment_id="finalv2-${family}-${algorithm_tag}-${candidate}-${updates}-s${SEED}"
  local run_dir="baseline_runs/${experiment_id}"

  if [[ -f "${run_dir}/summary.json" ]]; then
    printf '[sparse-v2 finalist] skip completed %s\n' "${experiment_id}"
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    printf '[sparse-v2 finalist] skip active/incomplete %s\n' "${experiment_id}"
    return 0
  fi

  printf '[sparse-v2 finalist] start %s\n' "${experiment_id}"
  micromamba run -n llm_offline_baselines python baseline_train.py \
    --config \
    "baselines/configs/base.${family}.yaml" \
    "baselines/configs/${algorithm}.yaml" \
    "baselines/experiments/${family}16.sweep.yaml" \
    "$@" \
    "baselines/experiments/${family}16.full_eval.yaml" \
    --experiment_id "${experiment_id}"
  local status=$?
  printf '[sparse-v2 finalist] finish %s status=%d\n' "${experiment_id}" "${status}"
  return "${status}"
}

run_point_mlp_lr5() {
  run_finalist pointmaze mlp_bc lr5e4 500k \
    "${CFG_DIR}/batch512.mlp.yaml" \
    "${CFG_DIR}/data500.yaml" \
    "${CFG_DIR}/net512x3.yaml" \
    "${CFG_DIR}/mlp.lr5e-4.yaml" \
    "${CFG_DIR}/long500.yaml"
}

run_point_mlp_n1024() {
  run_finalist pointmaze mlp_bc n1024 500k \
    "${CFG_DIR}/batch512.mlp.yaml" \
    "${CFG_DIR}/data500.yaml" \
    "${CFG_DIR}/net1024x3.yaml" \
    "${CFG_DIR}/mlp.lr3e-4.yaml" \
    "${CFG_DIR}/long500.yaml"
}

run_point_td3_n512() {
  run_finalist pointmaze td3_bc n512 500k \
    "${CFG_DIR}/batch512.actor_critic.yaml" \
    "${CFG_DIR}/td3.fixed.yaml" \
    "${CFG_DIR}/data500.yaml" \
    "${CFG_DIR}/net512x3.yaml" \
    "${CFG_DIR}/actor_critic.lr3e-4.yaml" \
    "${CFG_DIR}/long500.yaml"
}

run_point_td3_n1024() {
  run_finalist pointmaze td3_bc n1024 500k \
    "${CFG_DIR}/batch512.actor_critic.yaml" \
    "${CFG_DIR}/td3.fixed.yaml" \
    "${CFG_DIR}/data500.yaml" \
    "${CFG_DIR}/net1024x3.yaml" \
    "${CFG_DIR}/actor_critic.lr3e-4.yaml" \
    "${CFG_DIR}/long500.yaml"
}

run_ant_mlp_then_iql() {
  run_finalist antmaze mlp_bc n1024 500k \
    "${CFG_DIR}/batch512.mlp.yaml" \
    "${CFG_DIR}/data500.yaml" \
    "${CFG_DIR}/net1024x3.yaml" \
    "${CFG_DIR}/mlp.lr3e-4.yaml" \
    "${CFG_DIR}/long500.yaml"
  run_finalist antmaze iql d1000 250k \
    "${CFG_DIR}/batch512.actor_critic.yaml" \
    "${CFG_DIR}/iql.fixed.yaml" \
    "${CFG_DIR}/data1000.yaml" \
    "${CFG_DIR}/net512x3.yaml" \
    "${CFG_DIR}/actor_critic.lr3e-4.yaml" \
    "${CFG_DIR}/long250.yaml"
}

run_ant_td3_n1024() {
  run_finalist antmaze td3_bc n1024 500k \
    "${CFG_DIR}/batch512.actor_critic.yaml" \
    "${CFG_DIR}/td3.fixed.yaml" \
    "${CFG_DIR}/data500.yaml" \
    "${CFG_DIR}/net1024x3.yaml" \
    "${CFG_DIR}/actor_critic.lr3e-4.yaml" \
    "${CFG_DIR}/long500.yaml"
}

pids=()
run_point_mlp_lr5 & pids+=("$!")
run_point_mlp_n1024 & pids+=("$!")
run_point_td3_n512 & pids+=("$!")
run_point_td3_n1024 & pids+=("$!")
run_ant_mlp_then_iql & pids+=("$!")
run_ant_td3_n1024 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
