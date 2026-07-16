#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="llm_offline_baselines"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  micromamba env update -n "${ENV_NAME}" -f "${REPO_ROOT}/baselines/environment.yaml" -y
else
  micromamba env create -f "${REPO_ROOT}/baselines/environment.yaml" -y
fi

# d3rlpy 2.8.1 pins gymnasium 1.0.0. CrossMaze uses the repository's tested
# Gymnasium Robotics stack, so install d3rlpy itself without dependency changes.
micromamba run -n "${ENV_NAME}" python -m pip install --no-deps "d3rlpy==2.8.1"
micromamba run -n "${ENV_NAME}" python -m pip install --no-deps -e "${REPO_ROOT}"

micromamba run -n "${ENV_NAME}" python -c \
  "from baselines.runner import runtime_versions; print(runtime_versions())"
