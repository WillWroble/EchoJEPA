#!/bin/bash
#SBATCH --job-name=fl-mode2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:large:2
#SBATCH --mem=240G
#SBATCH --partition=bch-gpu
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/fl_mode2_%j.out
#SBATCH --error=logs/fl_mode2_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA

python app/fl/server_federated.py --num_rounds 720 --min_clients 2 --port 8080 &
sleep 10

CUDA_VISIBLE_DEVICES=0 python -m app.vjepa.train_federated --fname configs/train/vitl16/fl-mode2-internal.yaml --device cuda:0 &
CUDA_VISIBLE_DEVICES=1 python -m app.vjepa.train_federated --fname configs/train/vitl16/fl-mode2-external.yaml --device cuda:0 &

wait
