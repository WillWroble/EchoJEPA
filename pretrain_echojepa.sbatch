#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:large:8
##SBATCH --partition=bch-compute-pe
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=echojepa
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA

python -m app.main \
    --fname configs/train/vitb16/pretrain-bch-platon.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
