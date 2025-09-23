#!/bin/bash
#SBATCH --job-name=3d_lran
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/homes/%u/RBC-PINN-Surrogate/logs/slurm/%x_%A_%a.out
#SBATCH --mail-type=FAIL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcpinn

cd ~/RBC-PINN-Surrogate

# one seed per array task (0..4). No Hydra multirun here.
srun uv run python scripts/train_3d_lran.py seed=${SLURM_ARRAY_TASK_ID}