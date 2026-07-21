#!/bin/bash
#
# Pipeline regeneration test (stages 5a → 5b → 5c, US, hourly).
#
# Reads:  /net/fs06/d3/rzhuang/TROPOMI/pipeline_test_20260407/valid_tropomi_emissions_with_qa.csv
#         (symlinked from Run_20250623_203825)
# Writes: /net/fs06/d3/rzhuang/TROPOMI/pipeline_test_20260407/
#           valid_tropomi_emissions_with_qa_with_all_vars.csv         (5a)
#           processed_valid_tropomi_emissions_with_qa_updated.csv     (5b)
#           updated_tropomi_hourly_emissions_full_variables.csv       (5c)
#
# Submit:
#     sbatch slurm/us/regen_pipeline_test.sh

#SBATCH -J regen_pipe
#SBATCH -n 1
#SBATCH -p edr
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

set -e

echo "Job running on: $SLURM_JOB_NODELIST"
echo "Start: $(date)"

OUT=/net/fs06/d3/rzhuang/TROPOMI/pipeline_test_20260407
RUN_ID=pipeline_test_20260407
INPUT=$OUT/valid_tropomi_emissions_with_qa.csv

SCRIPTS=/net/fs06/d3/rzhuang/TROPOMI/code_reorg/5_tables

echo "=== Stage 5a: TROPOMI variable extraction ==="
# file_path column contains '../data/TROPOMI_2019-2024/...' relative paths;
# they resolve correctly when cwd is a subdir of /net/fs06/d3/rzhuang/TROPOMI_US/
cd /net/fs06/d3/rzhuang/TROPOMI_US/code
/net/fs01/home/rzhuang/miniforge3/bin/python -u $SCRIPTS/generate_table_tropomi.py --region us \
    --input  $INPUT \
    --output $OUT/valid_tropomi_emissions_with_qa_with_all_vars.csv

cd $SCRIPTS

echo "=== Stage 5b: ERA5 nearest-neighbor ==="
/net/fs01/home/rzhuang/miniforge3/bin/python -u generate_table_era5.py --region us \
    --input  $OUT/valid_tropomi_emissions_with_qa_with_all_vars.csv \
    --output $OUT/processed_valid_tropomi_emissions_with_qa_updated.csv

echo "=== Stage 5c: merge plume labels ==="
/net/fs01/home/rzhuang/miniforge3/bin/python -u merge_tables.py --region us \
    --run-id      $RUN_ID \
    --data-type   hourly \
    --reference-csv $OUT/processed_valid_tropomi_emissions_with_qa_updated.csv

echo "End: $(date)"
ls -la $OUT
