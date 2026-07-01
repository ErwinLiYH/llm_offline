#!/bin/bash

#SBATCH --job-name=antmaze_hard_datagen
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-ant-hard-%A_%a.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-ant-hard-%A_%a.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --array=0-8%4

set -euo pipefail

PROJECT_ROOT=/home/u6mx/yl1118.u6mx/llm_offline
CONDA_BIN=/home/u6mx/yl1118.u6mx/miniforge3/bin/conda
ENV_NAME=d4rl_datagen

TARGET_EPISODES=${TARGET_EPISODES:-2000}
MIN_SUCCESS_RATE=${MIN_SUCCESS_RATE:-1.0}
NUM_WORKERS=${NUM_WORKERS:-16}
SEED=${SEED:-42}
ACTION_NOISE=${ACTION_NOISE:-0.2}
MODE=${MODE:-diverse}
DIVERSE_CELL_MODE=${DIVERSE_CELL_MODE:-all-free}
HARD_RETRY=${HARD_RETRY:-5}
HARD_SAMPLE_ALPHA=${HARD_SAMPLE_ALPHA:-1.0}
HARD_SAMPLE_TOP_N=${HARD_SAMPLE_TOP_N:-400}

VARIANTS=(
    local-layout-01
    local-layout-02
    local-layout-03
    local-layout-04
    local-layout-05
    local-layout-06
    local-layout-07
    local-layout-08
    local-layout-09
)

VARIANT=${VARIANTS[${SLURM_ARRAY_TASK_ID}]}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}"

echo "Job ${SLURM_JOB_ID} started at $(date --iso-8601=seconds)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Mode: ${MODE} hard-sample local AntMaze generation"
echo "Variant: ${VARIANT}"
echo "Target successful episodes per variant: ${TARGET_EPISODES}"
echo "Minimum saved success rate: ${MIN_SUCCESS_RATE}"
echo "Workers per variant: ${NUM_WORKERS}"
echo "Seed: ${SEED}"
echo "Action noise: ${ACTION_NOISE}"
echo "Diverse cell mode: ${DIVERSE_CELL_MODE}"
echo "Hard retry: ${HARD_RETRY}"
echo "Hard sample alpha: ${HARD_SAMPLE_ALPHA}"
echo "Hard sample top N: ${HARD_SAMPLE_TOP_N}"
echo "Max episode attempts per variant: ignored by hard-sample mode"

"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    python -u local_antmaze_gen.py \
    --variants "${VARIANT}" \
    --num-workers "${NUM_WORKERS}" \
    --target-episodes "${TARGET_EPISODES}" \
    --min-success-rate "${MIN_SUCCESS_RATE}" \
    --seed "${SEED}" \
    --action-noise "${ACTION_NOISE}" \
    --mode "${MODE}" \
    --diverse-cell-mode "${DIVERSE_CELL_MODE}" \
    --maze-solver QIteration \
    --hard-sample \
    --hard-retry "${HARD_RETRY}" \
    --hard-sample-alpha "${HARD_SAMPLE_ALPHA}" \
    --hard-sample-top-n "${HARD_SAMPLE_TOP_N}" \
    --overwrite

echo "Job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"
