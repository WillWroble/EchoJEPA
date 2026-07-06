#!/bin/bash
#SBATCH --job-name=jepa_umap_date
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes/logs/umap_date_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes/logs/umap_date_%j.err
#SBATCH --mem=128G
#SBATCH --gres=gpu:large:1
#SBATCH --time=1-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes

mkdir -p logs

python -u umap_study_age.py \
    --checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/fl/mode2_v2/internal/epoch_day_6.pt \
    --output_dir results/fl/mode2_v2/internal/epoch_day_6 \
    --data_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k \
    --n_samples 50000
