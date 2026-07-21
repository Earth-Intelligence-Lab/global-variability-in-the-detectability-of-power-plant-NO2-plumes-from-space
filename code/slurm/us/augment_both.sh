#!/bin/bash
#
# Re-run the augment step (NOx via plant LOCAL time + nearby + wind_speed)
# for BOTH the 10m and 100m pipeline outputs. Skips 5a/5b/5c since those
# already exist. Writes *_augmented_localtz.csv next to each.
#
# Submit:
#     sbatch slurm/us/augment_both.sh

#SBATCH -J augment2
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0

set -e
echo "Job running on: $SLURM_JOB_NODELIST"
echo "Start: $(date)"

PY=/net/fs01/home/rzhuang/miniforge3/bin/python
SCRIPT=/net/fs06/d3/rzhuang/TROPOMI/code/5_tables/augment_final_table.py

echo ""; echo "=== Augmenting 10m pipeline output ==="
$PY -u $SCRIPT \
    --input  /net/fs06/d3/rzhuang/TROPOMI/pipeline_10m_run/updated_tropomi_hourly_emissions_full_variables.csv \
    --output /net/fs06/d3/rzhuang/TROPOMI/pipeline_10m_run/updated_tropomi_hourly_emissions_full_variables_augmented_localtz.csv

echo ""; echo "=== Augmenting 100m pipeline output ==="
$PY -u $SCRIPT \
    --input  /net/fs06/d3/rzhuang/TROPOMI/pipeline_100m_run/Run_100m_20260414/updated_tropomi_hourly_emissions_full_variables.csv \
    --output /net/fs06/d3/rzhuang/TROPOMI/pipeline_100m_run/Run_100m_20260414/updated_tropomi_hourly_emissions_full_variables_augmented_localtz.csv

echo ""; echo "End: $(date)"
ls -la /net/fs06/d3/rzhuang/TROPOMI/pipeline_10m_run/ /net/fs06/d3/rzhuang/TROPOMI/pipeline_100m_run/Run_100m_20260414/
