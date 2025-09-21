#!/bin/bash
#SBATCH --job-name=ae_ra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-6%4                # 7 jobs total, max 4 at once
#SBATCH --output=/homes/%u/RBC-PINN-Surrogate/logs/slurm/%x_%A_%a.out
#SBATCH --mail-type=FAIL

set -euo pipefail
mkdir -p "/homes/$USER/RBC-PINN-Surrogate/logs/slurm"

# Define the list of RA values
RA_VALUES=(10000 30000 100000 300000 1000000 3000000 10000000)

# Select RA based on array index
RA=${RA_VALUES[$SLURM_ARRAY_TASK_ID]}
echo "Running with data.ra=${RA}"

# Activate conda env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcpinn

cd ~/RBC-PINN-Surrogate

# Run training with this RA
srun uv run python scripts/train_ae_2D.py -m seed=0,1,2,3,4 data.data_dir=/vol/cfd/rbc/2D data.ra=${RA}