#!/bin/bash
# Permutation feature importance for the latest item-split MLP models.
# Runs both regions (us, world) sequentially in one GPU job.
#
#SBATCH -J perm_imp
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -t 6:00:00
#SBATCH -o /net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs/feature_importance.out
#SBATCH -e /net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs/feature_importance.err

set -e
echo "Job: $SLURM_JOB_ID | Node: $SLURM_JOB_NODELIST | Start: $(date)"

PY=/net/fs01/home/rzhuang/miniforge3/bin/python
SCRIPT=/net/fs06/d3/rzhuang/TROPOMI/code/7_analysis/feature_importance_permutation.py

echo ""; echo "=== US (item, hourly) ==="
$PY -u $SCRIPT --region us

echo ""; echo "=== WORLD (item, annual) ==="
$PY -u $SCRIPT --region world

echo ""; echo "End: $(date)"
