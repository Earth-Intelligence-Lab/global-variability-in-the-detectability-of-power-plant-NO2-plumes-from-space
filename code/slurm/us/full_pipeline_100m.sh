#!/bin/bash
#
# B1 → B2 → B3 → B4 → B5 for the 100m-wind labelling pipeline
# Reads:  pipeline_100m_run/Run_100m_20260414/updated_..._augmented_localtz.csv
# Writes: data/us/regression_100m/{master, fields_cache, era5_fields.nc, nox_regression_dataset.nc}

#SBATCH -J pipe_100m
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0

set -e
echo "Job: $SLURM_JOB_NODELIST | Cores: $SLURM_CPUS_PER_TASK | Start: $(date)"

PY=/net/fs01/home/rzhuang/miniforge3/bin/python
CODE=/net/fs06/d3/rzhuang/TROPOMI/code/7_regression_dataset
INPUT=/net/fs06/d3/rzhuang/TROPOMI/pipeline_100m_run/Run_100m_20260414/updated_tropomi_hourly_emissions_full_variables_augmented_localtz.csv
OUT=/net/fs06/d3/rzhuang/TROPOMI/data/us/regression_100m
mkdir -p $OUT

echo ""; echo "=== B1: filter samples ==="
$PY -u $CODE/filter_samples.py --region us --input $INPUT --out-dir $OUT

echo ""; echo "=== B2: TROPOMI patches ==="
$PY -u $CODE/extract_tropomi_fields.py --region us \
    --input $OUT/regression_samples_master.csv \
    --out-dir $OUT/fields_cache --workers $SLURM_CPUS_PER_TASK

echo ""; echo "=== B3: ERA5 patches ==="
$PY -u $CODE/extract_era5_fields.py --region us \
    --input $OUT/regression_samples_master.csv \
    --out $OUT/era5_fields.nc --method linear --workers $SLURM_CPUS_PER_TASK

echo ""; echo "=== B4: build_dataset ==="
$PY -u $CODE/build_dataset.py --region us \
    --scalar-csv $OUT/regression_samples_master.csv \
    --cache-dir $OUT/fields_cache \
    --era5-fields $OUT/era5_fields.nc \
    --output $OUT/nox_regression_dataset.nc \
    --workers $SLURM_CPUS_PER_TASK

echo ""; echo "=== B5: split ==="
$PY -u $CODE/split_dataset.py --region us \
    --dataset $OUT/nox_regression_dataset.nc --strategy power_plant

echo ""; echo "End: $(date)"
ls -la $OUT/
