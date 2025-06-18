#!/bin/bash

#SBATCH --job-name=fno
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/homes/%u/RBC-PINN-SURROGATE/logs/slurm/%x_%j.out
#SBATCH --mail-type=FAIL

cd ~/RBC-Control-SARL
srun uv run python scripts/train_fno.py