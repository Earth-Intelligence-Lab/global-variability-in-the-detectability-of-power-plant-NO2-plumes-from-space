#!/bin/bash
# MLP training on the U.S. dataset, power_plant split, no_stats feature set.
# data_type=annual variant.
#
#SBATCH -J train_us_pp_annual
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -t 4-00:00:00
#SBATCH -o /net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs/train_us_pp_annual.out
#SBATCH -e /net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs/train_us_pp_annual.err

set -e
mkdir -p /net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "Cores:     $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"

PY=/net/fs01/home/rzhuang/miniforge3/bin/python
cd /net/fs06/d3/rzhuang/TROPOMI/code/6_training

$PY -u train_mlp.py \
  --region us --split_mode power_plant --feature_set no_stats --data_type annual --interference \
  --data_csv            /net/fs06/d3/rzhuang/TROPOMI/pipeline_100m_run/Run_100m_20260414/updated_tropomi_hourly_emissions_full_variables_augmented_localtz.csv \
  --plant_csv           /net/fs06/d3/rzhuang/TROPOMI/data/us/facility_emissions_by_plant_comprehensive.csv \
  --annual_emissions_csv /net/fs06/d3/rzhuang/TROPOMI/data/us/annual-emissions-facility-aggregation-2019-2024.csv \
  --cities_csv          /net/fs06/d3/rzhuang/TROPOMI/data/world/worldcities.csv \
  --datasets            all_data_filtered_yearly top100_filtered_yearly top50_filtered_yearly top20_filtered_yearly \
  --output_dir          /net/fs06/d3/rzhuang/TROPOMI/data/us/Run_100m_20260414/training_no_stats_pp_annual

echo "End:       $(date)"
