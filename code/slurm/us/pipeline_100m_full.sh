#!/bin/bash
#
# Full chained pipeline for the 100m-wind re-labelling run:
#
#   (labelling already done → Run_100m_20260414/valid_tropomi_emissions_with_qa.csv)
#     ↓
#   stage 5a (generate_table_tropomi)   → valid_tropomi_emissions_with_qa_with_all_vars.csv
#     ↓
#   stage 5b (generate_table_era5)      → processed_valid_tropomi_emissions_with_qa_updated.csv
#     ↓
#   stage 5c (merge_tables)             → updated_tropomi_hourly_emissions_full_variables.csv
#     ↓
#   augment (NOx + nearby + wind_speed) → updated_tropomi_hourly_emissions_full_variables_augmented.csv
#
# Submit:
#     sbatch slurm/us/pipeline_100m_full.sh

#SBATCH -J pipe_100m
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0

set -e
echo "Job running on: $SLURM_JOB_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

PY=/net/fs01/home/rzhuang/miniforge3/bin/python
CODE=/net/fs06/d3/rzhuang/TROPOMI/code
RUN_NAME=Run_100m_20260414
OUT=/net/fs06/d3/rzhuang/TROPOMI/pipeline_100m_run/$RUN_NAME
LABEL_CSV=/net/fs06/d3/rzhuang/TROPOMI/pipeline_test_labelling_100m/$RUN_NAME/valid_tropomi_emissions_with_qa.csv

mkdir -p $OUT
# Symlink the labelling output so merge_tables (which expects a run_dir) can find it
ln -sfn $LABEL_CSV $OUT/valid_tropomi_emissions_with_qa.csv
# Also symlink into data_dir/$RUN_NAME/ so merge_tables --run-id resolves to OUT
ln -sfn $OUT /net/fs06/d3/rzhuang/TROPOMI/data/us/$RUN_NAME

echo ""; echo "=== Stage 5a: TROPOMI variable extraction ==="
$PY -u $CODE/5_tables/generate_table_tropomi.py --region us \
    --input  $OUT/valid_tropomi_emissions_with_qa.csv \
    --output $OUT/valid_tropomi_emissions_with_qa_with_all_vars.csv

echo ""; echo "=== Stage 5b: ERA5 nearest-neighbor (old 3 vars: t2m, tisr, tcwv) ==="
cd $CODE/5_tables
$PY -u generate_table_era5.py --region us \
    --input  $OUT/valid_tropomi_emissions_with_qa_with_all_vars.csv \
    --output $OUT/processed_valid_tropomi_emissions_with_qa_updated.csv

echo ""; echo "=== Stage 5c: merge plume labels ==="
$PY -u merge_tables.py --region us \
    --run-id    $RUN_NAME \
    --data-type hourly \
    --reference-csv $OUT/processed_valid_tropomi_emissions_with_qa_updated.csv

echo ""; echo "=== Augment: hourly NOx + nearby stats + wind_speed ==="
$PY -u augment_final_table.py \
    --input  $OUT/updated_tropomi_hourly_emissions_full_variables.csv \
    --output $OUT/updated_tropomi_hourly_emissions_full_variables_augmented.csv

echo ""; echo "End: $(date)"
ls -la $OUT/
