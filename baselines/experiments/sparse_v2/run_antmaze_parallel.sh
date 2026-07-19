#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CFG_DIR="baselines/experiments/sparse_v2"
FAMILY_CONFIG="baselines/experiments/antmaze16.sweep.yaml"
SEED=20260716

run_screen() {
  local algorithm="$1"
  local candidate="$2"
  shift 2

  local algorithm_tag="${algorithm//_/}"
  local experiment_id="sparsev2-antmaze-${algorithm_tag}-${candidate}-100k-s${SEED}"
  local run_dir="baseline_runs/${experiment_id}"

  if [[ -f "${run_dir}/summary.json" ]]; then
    printf '[sparse-v2 ant parallel] skip completed %s\n' "${experiment_id}"
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    printf '[sparse-v2 ant parallel] skip active/incomplete %s\n' "${experiment_id}"
    return 0
  fi

  printf '[sparse-v2 ant parallel] start %s\n' "${experiment_id}"
  micromamba run -n llm_offline_baselines python baseline_train.py \
    --config \
    baselines/configs/base.antmaze.yaml \
    "baselines/configs/${algorithm}.yaml" \
    "${FAMILY_CONFIG}" \
    "$@" \
    --experiment_id "${experiment_id}"
  local status=$?
  printf '[sparse-v2 ant parallel] finish %s status=%d\n' \
    "${experiment_id}" "${status}"
  return 0
}

run_mlp() {
  local batch="${CFG_DIR}/batch512.mlp.yaml"
  local n512="${CFG_DIR}/net512x3.yaml"
  local lr3="${CFG_DIR}/mlp.lr3e-4.yaml"

  run_screen mlp_bc d200 "${batch}" "${CFG_DIR}/data200.yaml" "${n512}" "${lr3}"
  run_screen mlp_bc d1000 "${batch}" "${CFG_DIR}/data1000.yaml" "${n512}" "${lr3}"
  run_screen mlp_bc lr1e4 "${batch}" "${CFG_DIR}/data500.yaml" "${n512}" "${CFG_DIR}/mlp.lr1e-4.yaml"
  run_screen mlp_bc lr5e4 "${batch}" "${CFG_DIR}/data500.yaml" "${n512}" "${CFG_DIR}/mlp.lr5e-4.yaml"
  run_screen mlp_bc n256 "${batch}" "${CFG_DIR}/data500.yaml" "${CFG_DIR}/net256x2.yaml" "${lr3}"
  run_screen mlp_bc n1024 "${batch}" "${CFG_DIR}/data500.yaml" "${CFG_DIR}/net1024x3.yaml" "${lr3}"
}

run_iql() {
  local batch="${CFG_DIR}/batch512.actor_critic.yaml"
  local fixed="${CFG_DIR}/iql.fixed.yaml"
  local d500="${CFG_DIR}/data500.yaml"
  local n512="${CFG_DIR}/net512x3.yaml"
  local lr3="${CFG_DIR}/actor_critic.lr3e-4.yaml"

  run_screen iql d200 "${batch}" "${fixed}" "${CFG_DIR}/data200.yaml" "${n512}" "${lr3}"
  run_screen iql d1000 "${batch}" "${fixed}" "${CFG_DIR}/data1000.yaml" "${n512}" "${lr3}"
  run_screen iql lr1e4 "${batch}" "${fixed}" "${d500}" "${n512}" "${CFG_DIR}/actor_critic.lr1e-4.yaml"
  run_screen iql lr5e4 "${batch}" "${fixed}" "${d500}" "${n512}" "${CFG_DIR}/actor_critic.lr5e-4.yaml"
  run_screen iql n256 "${batch}" "${fixed}" "${d500}" "${CFG_DIR}/net256x2.yaml" "${lr3}"
  run_screen iql n1024 "${batch}" "${fixed}" "${d500}" "${CFG_DIR}/net1024x3.yaml" "${lr3}"
  run_screen iql raw "${batch}" "${fixed}" "${d500}" "${n512}" "${lr3}" "${CFG_DIR}/iql.raw_reward.yaml"
  run_screen iql exp07 "${batch}" "${fixed}" "${d500}" "${n512}" "${lr3}" "${CFG_DIR}/iql.expectile07.yaml"
  run_screen iql temp10 "${batch}" "${fixed}" "${d500}" "${n512}" "${lr3}" "${CFG_DIR}/iql.temp10.yaml"
}

run_td3() {
  local batch="${CFG_DIR}/batch512.actor_critic.yaml"
  local fixed="${CFG_DIR}/td3.fixed.yaml"
  local d500="${CFG_DIR}/data500.yaml"
  local n256="${CFG_DIR}/net256x2.yaml"
  local lr3="${CFG_DIR}/actor_critic.lr3e-4.yaml"

  run_screen td3_bc base "${batch}" "${fixed}" "${d500}" "${n256}" "${lr3}"
  run_screen td3_bc d200 "${batch}" "${fixed}" "${CFG_DIR}/data200.yaml" "${n256}" "${lr3}"
  run_screen td3_bc d1000 "${batch}" "${fixed}" "${CFG_DIR}/data1000.yaml" "${n256}" "${lr3}"
  run_screen td3_bc lr1e4 "${batch}" "${fixed}" "${d500}" "${n256}" "${CFG_DIR}/actor_critic.lr1e-4.yaml"
  run_screen td3_bc lr5e4 "${batch}" "${fixed}" "${d500}" "${n256}" "${CFG_DIR}/actor_critic.lr5e-4.yaml"
  run_screen td3_bc n512 "${batch}" "${fixed}" "${d500}" "${CFG_DIR}/net512x3.yaml" "${lr3}"
  run_screen td3_bc n1024 "${batch}" "${fixed}" "${d500}" "${CFG_DIR}/net1024x3.yaml" "${lr3}"
  run_screen td3_bc alpha1 "${batch}" "${fixed}" "${d500}" "${n256}" "${lr3}" "${CFG_DIR}/td3.alpha1.yaml"
  run_screen td3_bc alpha5 "${batch}" "${fixed}" "${d500}" "${n256}" "${lr3}" "${CFG_DIR}/td3.alpha5.yaml"
}

run_mlp &
mlp_pid=$!
run_iql &
iql_pid=$!
run_td3 &
td3_pid=$!

status=0
wait "${mlp_pid}" || status=1
wait "${iql_pid}" || status=1
wait "${td3_pid}" || status=1
exit "${status}"
