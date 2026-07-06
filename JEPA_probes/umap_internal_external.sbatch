#!/bin/bash
#SBATCH --job-name=jepa_umap_ie
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes/logs/umap_ie_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes/logs/umap_ie_%j.err
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes

mkdir -p logs

python -u umap_internal_external.py \
    --checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/fl/mode2_v2/internal/epoch_day_6.pt \
    --internal_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_train.txt \
    --internal_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k \
    --external_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/study_outside.txt \
    --external_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Outside \
    --output_dir results/fl/mode2_v2/internal/epoch_day_6 \
    --n_samples 10000
