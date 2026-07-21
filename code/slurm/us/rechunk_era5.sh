#!/bin/bash
#
# One-shot: rewrite all yearly ERA5 files in <era5>/expanded/ to (1, lat, lon)
# chunk layout. Skips files already in that layout.
#
# Submit:
#     sbatch slurm/us/rechunk_era5.sh

#SBATCH -J rechunk
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0

echo "Job running on: $SLURM_JOB_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

cd /net/fs06/d3/rzhuang/TROPOMI/code_reorg/1_data_prep/us
/net/fs01/home/rzhuang/miniforge3/bin/python -u download_era5_aws.py \
    --rechunk-only --workers $SLURM_CPUS_PER_TASK

echo "End: $(date)"
