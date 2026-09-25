#!/bin/bash
#SBATCH --job-name=fl-mode1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:xlarge:2
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/fl_mode1_%j.out
#SBATCH --error=logs/fl_mode1_%j.err
#SBATCH --partition=bch-gpu-xlarge
#SBATCH --exclude=gpu-20-1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA

python app/fl/server_federated.py --num_rounds 800 --min_clients 2 --port 8080 &
sleep 10

CUDA_VISIBLE_DEVICES=0 python -m app.vjepa.train_federated --fname configs/train/vitl16/fl-mode1-internal.yaml --device cuda:0 &
CUDA_VISIBLE_DEVICES=1 python -m app.vjepa.train_federated --fname configs/train/vitl16/fl-mode1-external.yaml --device cuda:0 &

wait
