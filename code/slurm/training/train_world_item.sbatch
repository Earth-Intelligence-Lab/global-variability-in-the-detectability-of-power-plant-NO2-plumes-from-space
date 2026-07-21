#!/bin/bash
# MLP training on the global dataset, item split, no_stats feature set.
# Uses the world full-variables CSV with the 100 m re-labelled plume_label
# merged in (Run_100m_20260428).
#
#SBATCH -J train_world_item
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -t 4-00:00:00
#SBATCH -o /net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs/train_world_item.out
#SBATCH -e /net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs/train_world_item.err

set -e
mkdir -p /net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "Cores:     $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"

PY=/net/fs01/home/rzhuang/miniforge3/bin/python
cd /net/fs06/d3/rzhuang/TROPOMI/code/6_training

$PY -u train_mlp.py \
  --region world --split_mode item --feature_set no_stats --data_type annual --interference \
  --data_csv   /net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m/Run_100m_20260428/updated_tropomi_emissions_full_variables_with_fuel_100mlabel.csv \
  --plant_csv  /net/fs06/d3/rzhuang/TROPOMI/data/world/power_plant_location/power_plants_with_combined_nearby_stats.csv \
  --cities_csv /net/fs06/d3/rzhuang/TROPOMI/data/world/worldcities.csv \
  --datasets   all_data_filtered top100_filtered top50_filtered top20_filtered \
  --output_dir /net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m/Run_100m_20260428/training_no_stats_item

echo "End:       $(date)"
