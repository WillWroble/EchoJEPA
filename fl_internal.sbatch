#!/bin/bash
#SBATCH --job-name=fl-internal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:large:1
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/fl_internal_%j.out
#SBATCH --error=logs/fl_internal_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA

python -m app.vjepa.train_federated \
    --fname configs/train/vitb16/fl-internal.yaml \
    --device cuda:0
