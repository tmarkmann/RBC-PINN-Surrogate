#!/bin/bash

#SBATCH --job-name=dataset
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/homes/%u/RBC-PINN-Surrogate/logs/slurm/%x_%j.out
#SBATCH --mail-type=FAIL

cd ~/RBC-PINN-Surrogate
srun uv run python scripts/create_dataset.py --ra 3000000
