#!/bin/bash
#
# Submit the AWS ERA5 expanded download to the cluster.
#
# Submit:
#     sbatch slurm/us/download_era5_aws.sh
#
# Output: /net/fs06/d3/rzhuang/TROPOMI_US/data/era5/expanded/<short>_<year>.nc

#SBATCH -J era5_dl
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0      # all memory on the node

echo "Job running on: $SLURM_JOB_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

mkdir -p /net/fs06/d3/rzhuang/TROPOMI_US/data/era5/expanded
mkdir -p /tmp/era5_aws_raw_$SLURM_JOB_ID

cd /net/fs06/d3/rzhuang/TROPOMI/code_reorg/1_data_prep/us
# 48 parallel (var, year) downloads — 12 vars × 6 years = 72 tasks total
python -u download_era5_aws.py \
    --workers 48 \
    --raw-dir /tmp/era5_aws_raw_$SLURM_JOB_ID

rm -rf /tmp/era5_aws_raw_$SLURM_JOB_ID

echo "End: $(date)"
