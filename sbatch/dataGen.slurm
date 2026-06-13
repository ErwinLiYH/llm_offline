#!/bin/bash

#SBATCH --job-name=pointmaze_datagen
#SBATCH --output=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-%j.out
#SBATCH --error=/home/u6mx/yl1118.u6mx/llm_offline/sbatch/datagen-%j.err
#SBATCH --chdir=/home/u6mx/yl1118.u6mx/llm_offline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115G
#SBATCH --time=20:00:00

set -euo pipefail

export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID} started at $(date --iso-8601=seconds)"
echo "Node: $(hostname)"

/home/u6mx/yl1118.u6mx/miniforge3/bin/conda run --no-capture-output -n d4rl_datagen \
    python -u local_varient_gen.py \
    --variants local-layout-01 local-layout-02 local-layout-03 local-layout-04 local-layout-05 local-layout-06 local-layout-07 local-layout-08 local-layout-09 \
    --num-workers 20 \
    --target-episodes 5000 \
    --post-success-hold-steps 100 \
    --overwrite \
    --seed 42

echo "Job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"
