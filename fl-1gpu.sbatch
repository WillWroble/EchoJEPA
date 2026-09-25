#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:large:1
#SBATCH --mem=120G
#SBATCH --partition=bch-gpu
#SBATCH --exclude=gpu-b07-0,gpu-b07-1,gpu-b07-2,gpu-b07-3,gpu-b08-0,gpu-b09-0,gpu-b11-0,gpu-b11-1,gpu-b11-2,gpu-b11-3
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/fl_baseline_vitl_%j.out
#SBATCH --error=logs/fl_baseline_vitl_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA
python -m app.main \
    --fname configs/train/vitl16/fl-baseline-internal.yaml \
    --devices cuda:0
