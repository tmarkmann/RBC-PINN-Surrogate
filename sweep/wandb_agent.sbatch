#!/bin/bash

#SBATCH --job-name=sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/homes/%u/RBC-PINN-Surrogate/logs/slurm/%x_%j.out
#SBATCH --mail-type=FAIL


# source conda env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcuv

cd ~/RBC-PINN-Surrogate

# start job
srun uv run wandb agent $1 --project RayleighBenard-FNO --entity sail-project