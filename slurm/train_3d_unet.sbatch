#!/bin/bash

#SBATCH --job-name=3d_unet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/homes/%u/RBC-PINN-Surrogate/logs/slurm/%x_%A_%a.out
#SBATCH --mail-type=FAIL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcpinn

cd ~/RBC-PINN-Surrogate
srun uv run python scripts/train_3d_unet.py seed=$SLURM_ARRAY_TASK_ID options=gpucluster