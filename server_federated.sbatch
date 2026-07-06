#!/bin/bash
#SBATCH --job-name=fl_server
#SBATCH --output=logs/fl_server_%j.out
#SBATCH --error=logs/fl_server_%j.err
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA

python -u app/fl/server_federated.py --num_rounds 72000 --min_clients 2 --port 8081
