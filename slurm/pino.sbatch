#!/bin/bash

#SBATCH --job-name=pino
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --time=1-00:00:00
#SBATCH --output=/homes/%u/RBC-PINN-Surrogate/logs/slurm/%x_%A_%a.out
#SBATCH --mail-type=FAIL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcuv

cd ~/RBC-PINN-Surrogate

# Dispatch command based on array index
case $SLURM_ARRAY_TASK_ID in
  0) CMD="" ;;
  1) CMD="algo.do_finetuning=True" ;;
  2) CMD="model.pino_weight=0" ;;
esac

srun uv run python scripts/train_pino.py $CMD