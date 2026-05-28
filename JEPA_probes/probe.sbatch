#!/bin/bash
#SBATCH --job-name=jepa_probe
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes/logs/probe_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes/logs/probe_%j.err
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes

mkdir -p logs

python -u probe.py \
    --checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitb_platon/epoch_day_1.pt \
    --output_dir results/epoch_day_1 \
    --train_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/jepa_probe_platon_train.txt \
    --val_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/jepa_probe_platon_val.txt \
    --data_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k \
    --videos_per_study 48 \
    --epochs 20 \
    --lr 1e-4 \
    --batch_size 32
