#!/bin/bash
#
# B1 → B2 → B3 → B4 → B5 for the 100m-wind STRICT (plant_mask=50) labelling.
# Cell: factor=3.0σ, area_min=100 km², plant_mask=50 km, close_distance=0,
#       city scale 12 / city max 120 km.
# Reads:  pipeline_100m_run/Run_100m_strict_3.0_100_plant50/updated_..._augmented_localtz.csv
# Writes: data/us/regression_100m_strict_plant50/{master, fields_cache, era5_fields.nc, nox_regression_dataset.nc}

#SBATCH -J pipe_strict50
#SBATCH -p edr
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH -t 12:00:00
#SBATCH -o /net/fs06/d3/rzhuang/TROPOMI/code/slurm/us/full_pipeline_100m_strict_plant50.out
#SBATCH -e /net/fs06/d3/rzhuang/TROPOMI/code/slurm/us/full_pipeline_100m_strict_plant50.err

set -e
echo "Job: $SLURM_JOB_NODELIST | Cores: $SLURM_CPUS_PER_TASK | Start: $(date)"

PY=/net/fs01/home/rzhuang/miniforge3/bin/python
CODE=/net/fs06/d3/rzhuang/TROPOMI/code/7_regression_dataset
INPUT=/net/fs06/d3/rzhuang/TROPOMI/pipeline_100m_run/Run_100m_strict_3.0_100_plant50/updated_tropomi_hourly_emissions_full_variables_augmented_localtz.csv
OUT=/net/fs06/d3/rzhuang/TROPOMI/data/us/regression_100m_strict_plant50
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
