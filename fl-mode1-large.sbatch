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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA
export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:$PYTHONPATH

# register the local superlink connection (idempotent)
mkdir -p ~/.flwr
grep -q "superlink.local-deployment" ~/.flwr/config.toml 2>/dev/null || cat >> ~/.flwr/config.toml << 'EOF'

[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
EOF

flower-superlink --insecure &
sleep 10

CUDA_VISIBLE_DEVICES=0 flower-supernode --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 \
    --node-config 'config-path="/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/configs/train/vitl16/fl-mode1-internal.yaml"' &

CUDA_VISIBLE_DEVICES=1 flower-supernode --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 \
    --node-config 'config-path="/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/configs/train/vitl16/fl-mode1-external.yaml"' &

sleep 10
flwr run flower local-deployment --stream

