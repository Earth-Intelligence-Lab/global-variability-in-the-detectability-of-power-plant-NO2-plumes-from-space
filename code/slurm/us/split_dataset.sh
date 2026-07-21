#!/bin/bash
#
# B5: Assign train/val/test split to the merged NetCDF dataset.
#
# Submit:
#     sbatch slurm/us/split_dataset.sh

#SBATCH -J split_ds
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

echo "Job running on: $SLURM_JOB_NODELIST"
echo "Start: $(date)"

cd /net/fs06/d3/rzhuang/TROPOMI/code_reorg/7_regression_dataset
/net/fs01/home/rzhuang/miniforge3/bin/python -u split_dataset.py --region us --strategy power_plant

echo "End: $(date)"
