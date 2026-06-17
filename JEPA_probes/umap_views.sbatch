#!/bin/bash
#SBATCH --job-name=jepa_umap
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes/logs/umap_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes/logs/umap_%j.err
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes

mkdir -p logs

python -u umap_views.py \
    --checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitl_platon_mimic/epoch_day_5.pt \
    --output_dir results/epoch_day_5_mimic_1024d \
    --view_labels /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/view_labels.csv \
    --data_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k
