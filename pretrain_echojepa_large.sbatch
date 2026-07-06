#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:xlarge:4
#SBATCH --mem=300G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=echojepa_vitl
#SBATCH --output=logs/pretrain_vitl_%j.out
#SBATCH --error=logs/pretrain_vitl_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA
python -m app.main \
    --fname configs/train/vitl16/cooldown-mimic-bch.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3
