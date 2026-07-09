#!/bin/bash

#SBATCH --job-name=remote_command
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/remote-command-%j.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/remote-command-%j.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=440G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT=/home/u6mx/yl1118.u6mx/llm_offline
ENV_BIN=/home/u6mx/yl1118.u6mx/miniforge3/bin
REMOTE_COMMAND_SERVER=/home/u6mx/yl1118.u6mx/miniforge3/bin/remote-command-server
REMOTE_COMMAND_PORT=${REMOTE_COMMAND_PORT:-8000}
STATE_DIRECTORY=${SLURM_TMPDIR:-/tmp}/remote-command-${USER}-${SLURM_JOB_ID}

export PATH="${ENV_BIN}:${PATH}"
# Do not inherit login-session TMPDIR (/local/user/<uid> -> /run/user, deleted by logind ~20s into the job)
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export REMOTE_COMMAND_TOKEN=200113

mkdir -p "${STATE_DIRECTORY}"

echo "Job ${SLURM_JOB_ID} started at $(date --iso-8601=seconds)"
echo "Compute hostname: $(hostname --fqdn 2>/dev/null || hostname)"
echo "RemoteCommand port: ${REMOTE_COMMAND_PORT}"
echo "Default working directory: ${PROJECT_ROOT}"
echo "Candidate IPv4 addresses:"
IP_ADDRESSES=$(ip -o -4 addr show scope global 2>/dev/null | awk '{print "  " $2 ": " $4}' || true)
if [[ -n "${IP_ADDRESSES}" ]]; then
    printf '%s\n' "${IP_ADDRESSES}"
else
    echo "  ip(8) address enumeration unavailable."
fi
echo
SERVICE_IP=$(ip -o -4 addr show dev nmn0 2>/dev/null | awk 'NR == 1 {sub(/\/.*/, "", $4); print $4}' || true)
if [[ -n "${SERVICE_IP}" ]]; then
    echo "Recommended service URL: http://${SERVICE_IP}:${REMOTE_COMMAND_PORT}"
else
    echo "Service URL: http://<reachable-compute-node-ip>:${REMOTE_COMMAND_PORT}"
fi
echo "From the login node, configure the client with:"
echo "  export REMOTE_COMMAND_URL=http://${SERVICE_IP:-<reachable-compute-node-ip>}:${REMOTE_COMMAND_PORT}"
echo "  export REMOTE_COMMAND_TOKEN=<the configured token>"
echo "Then test with:"
echo "  remote-command --command 'hostname && pwd && nvidia-smi -L'"
echo "Stop the server and release the allocation with: scancel ${SLURM_JOB_ID}"
echo

srun --ntasks=1 "${REMOTE_COMMAND_SERVER}" \
    --host 0.0.0.0 \
    --port "${REMOTE_COMMAND_PORT}" \
    --working-directory "${PROJECT_ROOT}" \
    --state-directory "${STATE_DIRECTORY}" \
    --shell /bin/bash
