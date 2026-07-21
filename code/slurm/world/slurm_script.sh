#!/bin/bash
#SBATCH -J labeling
#SBATCH -n 1
#SBATCH -p hdr200
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=48

export PYTHONUNBUFFERED=1

echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Cores per node:     $SLURM_TASKS_PER_NODE"
echo "Tasks per node:      $SLURM_NTASKS_PER_NODE"

python -u /net/fs06/d3/rzhuang/TROPOMI_world/code/1_filter_out_duplicate_power_plant.py