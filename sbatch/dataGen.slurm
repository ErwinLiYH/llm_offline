#!/bin/bash

#SBATCH --job-name=pointmaze_datagen
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-%j.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-%j.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=36G
#SBATCH --time=5:00:00

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID} started at $(date --iso-8601=seconds)"
echo "Node: $(hostname)"

/home/u6mx/yl1118.u6mx/miniforge3/bin/conda run --no-capture-output -n d4rl_datagen \
    python -u local_varient_gen.py \
    --variants local-layout-10 local-layout-11 local-layout-12 local-layout-13 test-layout-01 test-layout-02 test-layout-03 \
    --num-workers 10 \
    --target-episodes 5000 \
    --post-success-hold-steps 100 \
    --overwrite \
    --seed 42

echo "Job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"
