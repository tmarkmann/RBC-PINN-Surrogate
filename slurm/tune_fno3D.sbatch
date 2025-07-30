#!/bin/bash

#SBATCH --job-name=fno3d_tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/homes/%u/RBC-PINN-Surrogate/logs/slurm/%x_%j.out
#SBATCH --mail-type=FAIL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcpinn

cd ~/RBC-PINN-Surrogate
srun uv run python scripts/tune_fno_3D.py