#!/bin/bash
#
# Re-run World stage-2 labelling using ERA5 100m wind (instead of TROPOMI-
# embedded 10m operational-forecast wind) as the plume-direction driver.
# Uses the SAME load_era5_100m_wind() integration as the US 100m labelling.
#
# Output: /data/world/pipeline_test_labelling_100m/Run_100m_<date>/
#           valid_tropomi_emissions_with_qa.csv
#
# Submit:
#     sbatch slurm/world/labelling_100m.sh

#SBATCH -J w_lbl100
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0

set -e
echo "Job: $SLURM_JOB_NODELIST | Cores: $SLURM_CPUS_PER_TASK | Start: $(date)"

export PYTHONPATH=/net/fs06/d3/rzhuang/TROPOMI/code/shared:$PYTHONPATH

OUT_BASE=/net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m
mkdir -p $OUT_BASE

/net/fs01/home/rzhuang/miniforge3/bin/python -u \
    /net/fs06/d3/rzhuang/TROPOMI/code/2_snapshots/world/2_find_snapshots_labelling_100m.py \
    --run_id 100m_$(date +%Y%m%d) \
    --base_output_dir $OUT_BASE \
    --num_cpus $SLURM_CPUS_PER_TASK

echo "End: $(date)"
ls -la $OUT_BASE/
