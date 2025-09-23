#!/bin/bash
#SBATCH --job-name=3d_ae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/homes/%u/RBC-PINN-Surrogate/logs/slurm/%x_%A_%a.out
#SBATCH --mail-type=FAIL

set -euo pipefail
mkdir -p "/homes/$USER/RBC-PINN-Surrogate/logs/slurm"

# Define the list of RA values
RA_VALUES=(2500)

# Select RA based on array index
RA=${RA_VALUES[$SLURM_ARRAY_TASK_ID]}
echo "Running with data.ra=${RA}"

# Activate conda env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcpinn

cd ~/RBC-PINN-Surrogate

# Run training with this RA
srun uv run python scripts/train_3d_ae.py -m seed=0,1,2,3,4 data.ra=${RA}