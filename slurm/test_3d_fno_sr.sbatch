#!/bin/bash
#SBATCH --job-name=3d_fno_sr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/homes/%u/RBC-PINN-Surrogate/logs/slurm/%x_%A_%a.out
#SBATCH --mail-type=FAIL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcpinn

cd ~/RBC-PINN-Surrogate
srun uv run python scripts/test_3d.py experiment/test=fno_sr sr=8 seed=${SLURM_ARRAY_TASK_ID}
