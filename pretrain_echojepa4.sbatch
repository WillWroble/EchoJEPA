#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:large:4
#SBATCH --partition=bch-compute-pe
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=echojepa4
#SBATCH --output=logs/pretrain4_%j.out
#SBATCH --error=logs/pretrain4_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA

python -m app.main \
    --fname configs/train/vitb16/pretrain-bch-224px-16f.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3
