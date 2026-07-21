#!/bin/bash
#
# Re-run stage-2 labelling using ERA5 100m wind (instead of TROPOMI-embedded
# 10m operational-forecast wind) as the plume-direction driver.
#
# Output: /net/fs06/d3/rzhuang/TROPOMI/pipeline_test_labelling_100m/Run_100m_<date>/
#           valid_tropomi_emissions_with_qa.csv
#
# Submit:
#     sbatch slurm/us/labelling_100m.sh

#SBATCH -J label_100m
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 16:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0

set -e
echo "Job running on: $SLURM_JOB_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

# plotting.py lives in code/shared — add to PYTHONPATH
export PYTHONPATH=/net/fs06/d3/rzhuang/TROPOMI/code/shared:$PYTHONPATH

OUT_BASE=/net/fs06/d3/rzhuang/TROPOMI/pipeline_test_labelling_100m
mkdir -p $OUT_BASE

/net/fs01/home/rzhuang/miniforge3/bin/python -u \
    /net/fs06/d3/rzhuang/TROPOMI/code/2_snapshots/us/2_find_snapshots_labelling_100m.py \
    --run_id 100m_$(date +%Y%m%d) \
    --base_output_dir $OUT_BASE \
    --num_cpus $SLURM_CPUS_PER_TASK

echo "End: $(date)"
ls -la $OUT_BASE/
