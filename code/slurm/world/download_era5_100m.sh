#!/bin/bash
#
# Download world ERA5 100m wind (u100, v100) from AWS for 2024.
# (2018 is already covered by data/world/era5/wind_uv_100.nc.)
# Output: /data/world/era5/expanded/{u100,v100}_2024.nc
#
# Submit:
#     sbatch slurm/world/download_era5_100m.sh

#SBATCH -J w_era5dl
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0

set -e
echo "Job: $SLURM_JOB_NODELIST | Start: $(date)"

PY=/net/fs01/home/rzhuang/miniforge3/bin/python
SCRIPT=/net/fs06/d3/rzhuang/TROPOMI/code/1_data_prep/us/download_era5_aws.py

$PY -u $SCRIPT --region world --years 2024 \
    --var u100 --var v100 --workers 4

echo "End: $(date)"
ls -lh /net/fs06/d3/rzhuang/TROPOMI/data/world/era5/expanded/
