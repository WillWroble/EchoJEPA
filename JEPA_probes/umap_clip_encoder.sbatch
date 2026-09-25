#!/bin/bash
#SBATCH --job-name=umap_v3
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes/logs/umap_v3_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes/logs/umap_v3_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=bch-gpu-pe
#SBATCH --exclude=gpu-b11-2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:$PYTHONPATH
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes
mkdir -p logs
#--encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitl_mimic_bch_cooldown_24fps/latest.pt \
python -u umap_clip_encoder.py \
    --encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/results/v7/best.pt \
    --v3_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/results/v8/best.pt \
    --data_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k \
    --manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_val.txt \
    --output_dir results/clip_encoder/v7/best \
    --n_studies 50
