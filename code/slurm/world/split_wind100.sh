#!/bin/bash
#
# Split data/world/era5/wind_uv_100.nc into u100_2018.nc + v100_2018.nc
# under data/world/era5/expanded/ to match the US AWS-format yearly layout.
# Also normalizes lon 0..360 → -180..180 so the same load_era5_100m_wind()
# function in the labelling script works for both US and World.

#SBATCH -J w_split
#SBATCH -n 1
#SBATCH -p hdr
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=0

set -e
echo "Job: $SLURM_JOB_NODELIST | Start: $(date)"
/net/fs01/home/rzhuang/miniforge3/bin/python -u \
    /net/fs06/d3/rzhuang/TROPOMI/code/1_data_prep/world/split_wind_uv_100.py
echo "End: $(date)"
ls -lh /net/fs06/d3/rzhuang/TROPOMI/data/world/era5/expanded/
