#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:large:1
#SBATCH --mem=60G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/fl_%j.out
#SBATCH --error=logs/fl_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA


python -m app.main \
    --fname configs/train/vitb16/fl-internal.yaml \
    --devices cuda:0

