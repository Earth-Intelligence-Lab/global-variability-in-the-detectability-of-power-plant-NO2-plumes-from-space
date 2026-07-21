#!/bin/bash
#
# B2: Extract TROPOMI 2D patches (7 channels, 100x100 WGS84 grid) for all
# plume-positive snapshots in the regression dataset.
#
# Submit:
#     sbatch slurm/us/extract_tropomi_fields.sh
#
# Output: <regression>/fields_cache/*_patches.nc

#SBATCH -J tropomi_patch
#SBATCH -n 1
#SBATCH -p edr
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

echo "Job running on: $SLURM_JOB_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

cd /net/fs06/d3/rzhuang/TROPOMI/code_reorg/7_regression_dataset
/net/fs01/home/rzhuang/miniforge3/bin/python -u extract_tropomi_fields.py --region us --workers $SLURM_CPUS_PER_TASK

echo "End: $(date)"
