#!/bin/bash
# Per-plant SHAP for U.S. (uses new 100m data + new item-split best model).
#
#SBATCH -J shap_us
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH -o /net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs/shap_us.out
#SBATCH -e /net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs/shap_us.err

set -e
echo "Job: $SLURM_JOB_ID | Node: $SLURM_JOB_NODELIST | Start: $(date)"

PY=/net/fs01/home/rzhuang/miniforge3/bin/python
$PY -u /net/fs06/d3/rzhuang/TROPOMI/code/7_analysis/us/7_feature_importance_per_power_plant_plot_no_interference_no_stats.py

echo "End: $(date)"
