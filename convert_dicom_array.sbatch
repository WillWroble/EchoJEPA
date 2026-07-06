#!/bin/bash
#SBATCH --array=1-3138%64
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=dcm2mp4
#SBATCH --output=logs/dcm2mp4_%A_%a.out
#SBATCH --error=logs/dcm2mp4_%A_%a.err
#SBATCH --partition=bch-compute

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA

export DCM_INPUT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" new_mimic_study_dirs.txt)
export DCM_OUTPUT=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/mimic_iv
python -u data/convert_dicom.py
