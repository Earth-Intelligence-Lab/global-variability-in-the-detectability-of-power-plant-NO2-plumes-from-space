#!/bin/bash
#
# filename: slurm_script
#
# Example SLURM script to run a job on the svante cluster.
# The lines beginning #SBATCH set various queuing parameters.
#
# Set name of submitted job

#SBATCH -J labeling
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=48

echo 'Your job is running on node(s):'
echo $SLURM_JOB_NODELIST
echo 'Cores per node:'
echo $SLURM_TASKS_PER_NODE

python -u /net/fs06/d3/rzhuang/TROPOMI_US/code/7_feature_importance_per_power_plant_plot_no_interference_no_stats.py