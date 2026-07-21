#!/bin/bash
#
# B3: Extract ERA5 2D patches on the same WGS84 grid as TROPOMI patches.
#
# Submit:
#     sbatch slurm/us/extract_era5_fields.sh

#SBATCH -J era5_patch
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0

echo "Job running on: $SLURM_JOB_NODELIST"
echo "Start: $(date)"

cd /net/fs06/d3/rzhuang/TROPOMI/code_reorg/7_regression_dataset
/net/fs01/home/rzhuang/miniforge3/bin/python -u extract_era5_fields.py --region us --method linear --workers $SLURM_CPUS_PER_TASK

echo "End: $(date)"
