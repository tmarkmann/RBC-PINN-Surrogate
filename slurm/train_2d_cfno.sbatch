#!/bin/bash

#SBATCH --job-name=2d_cfno
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/homes/%u/RBC-PINN-Surrogate/logs/slurm/%x_%A_%a.out
#SBATCH --mail-type=FAIL

set -euo pipefail
mkdir -p "/homes/$USER/RBC-PINN-Surrogate/logs/slurm"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcpinn

cd ~/RBC-PINN-Surrogate
srun uv run python scripts/train_2d_cfno.py seed=$SLURM_ARRAY_TASK_ID experiment/cfno=full