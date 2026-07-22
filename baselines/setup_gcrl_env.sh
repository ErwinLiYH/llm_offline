#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="llm_offline_gcrl"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  micromamba env update -n "${ENV_NAME}" -f "${REPO_ROOT}/baselines/environment.gcrl.yaml" -y
else
  micromamba env create -f "${REPO_ROOT}/baselines/environment.gcrl.yaml" -y
fi

micromamba run -n "${ENV_NAME}" python -m pip install --no-deps -e "${REPO_ROOT}"
micromamba run -n "${ENV_NAME}" python -c \
  "from baselines.gcrl.runner import runtime_versions; print(runtime_versions())"
