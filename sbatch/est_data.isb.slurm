#!/bin/bash

#SBATCH --job-name=llm_offline_est_data
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/est-data-%j.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/est-data-%j.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT=/home/u6mx/yl1118.u6mx/llm_offline
ENV_BIN=/home/u6mx/yl1118.u6mx/miniforge3/envs/llm_offline/bin

CONFIG_PATHS=()
WORLD_SIZE=4
SAMPLE_EPISODES_PER_VARIANT=4
SAMPLE_SEED=""
PRINT_JSON=false
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage:
  sbatch $0 --config <config.yaml> [options]
  sbatch $0 --config <base.yaml> <override.yaml> [options]
  sbatch $0 --config <config.yaml> --world_size 4 --sample-episodes-per-variant 8
  sbatch $0 --config <config.yaml> -- --json

Options:
  --config <path> [path ...]         Training config(s) passed to estimate_dataset.py.
  --world_size, --world-size <n>     DDP world-size estimate for batch math only. Default: ${WORLD_SIZE}.
  --sample-episodes-per-variant <n>  Complete episodes tokenized per variant. Default: ${SAMPLE_EPISODES_PER_VARIANT}.
  --sample-seed <seed>               Override estimator episode sampling seed.
  --json                             Print final estimate as JSON.
  -h, --help                         Show this help.

Slurm resources are fixed by the #SBATCH lines in this file. Override them with
sbatch options before the script name, for example:
  sbatch --mem=220G sbatch/est_data.isb.slurm --config configs/train/base.yaml configs/train/override.yaml

Arguments after "--" are appended directly to estimate_dataset.py.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --config requires at least one path" >&2
                usage >&2
                exit 2
            fi
            shift
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --world_size|--world-size|--sample-episodes-per-variant|--sample-seed|--json|--|-h|--help)
                        break
                        ;;
                    --*)
                        break
                        ;;
                    *)
                        CONFIG_PATHS+=("$1")
                        shift
                        ;;
                esac
            done
            if [[ ${#CONFIG_PATHS[@]} -eq 0 ]]; then
                echo "ERROR: --config requires at least one path" >&2
                usage >&2
                exit 2
            fi
            ;;
        --world_size|--world-size)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: $1 requires an integer" >&2
                usage >&2
                exit 2
            fi
            WORLD_SIZE="$2"
            shift 2
            ;;
        --sample-episodes-per-variant)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --sample-episodes-per-variant requires an integer" >&2
                usage >&2
                exit 2
            fi
            SAMPLE_EPISODES_PER_VARIANT="$2"
            shift 2
            ;;
        --sample-seed)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --sample-seed requires an integer" >&2
                usage >&2
                exit 2
            fi
            SAMPLE_SEED="$2"
            shift 2
            ;;
        --json)
            PRINT_JSON=true
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ${#CONFIG_PATHS[@]} -eq 0 ]]; then
    echo "ERROR: --config is required" >&2
    usage >&2
    exit 2
fi
for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "ERROR: config file not found: ${CONFIG_PATH}" >&2
        exit 2
    fi
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

cd "${PROJECT_ROOT}"

echo "Job ${SLURM_JOB_ID:-manual} started at $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Configs: ${CONFIG_PATHS[*]}"
echo "World size estimate: ${WORLD_SIZE}"
echo "Sample episodes per variant: ${SAMPLE_EPISODES_PER_VARIANT}"
if [[ -n "${SAMPLE_SEED}" ]]; then
    echo "Sample seed: ${SAMPLE_SEED}"
else
    echo "Sample seed: config default"
fi

ESTIMATE_ARGS=(
    --config "${CONFIG_PATHS[@]}"
    --world_size "${WORLD_SIZE}"
    --sample-episodes-per-variant "${SAMPLE_EPISODES_PER_VARIANT}"
)
if [[ -n "${SAMPLE_SEED}" ]]; then
    ESTIMATE_ARGS+=(--sample-seed "${SAMPLE_SEED}")
fi
if [[ "${PRINT_JSON}" == true ]]; then
    ESTIMATE_ARGS+=(--json)
fi
ESTIMATE_ARGS+=("${EXTRA_ARGS[@]}")

srun --ntasks=1 "${ENV_BIN}/python" estimate_dataset.py "${ESTIMATE_ARGS[@]}"

echo "Job ${SLURM_JOB_ID:-manual} finished at $(date --iso-8601=seconds)"
