#!/bin/bash
#
# B4: Assemble the final regression dataset NetCDF from B1+B2+B3 outputs.
#
# Submit:
#     sbatch slurm/us/build_dataset.sh

#SBATCH -J build_ds
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0

echo "Job running on: $SLURM_JOB_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

cd /net/fs06/d3/rzhuang/TROPOMI/code_reorg/7_regression_dataset
/net/fs01/home/rzhuang/miniforge3/bin/python -u build_dataset.py --region us --workers $SLURM_CPUS_PER_TASK

echo "End: $(date)"
