"""
Unified snapshot finder for TROPOMI pipeline (US + World).

Usage:
    python find_snapshots.py --region us
    python find_snapshots.py --region world

Replaces the duplicated us/2_find_snapshots.py and world/2_find_snapshots.py.
The only differences between the original two scripts were file paths and
column names — both are now driven by config/{us,world}.py.

Behavior is intended to be byte-equivalent to the originals when run with the
default config.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

# Make `config` and `shared` importable when run as a script from this dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # code/

from config import get_config, RegionConfig  # noqa: E402
from shared.data_utils import (  # noqa: E402
    open_tropomi_basics, build_balltree, plant_valid,
    nearest_pixel_index, get_local_utc_time, get_valid_no2_mask,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tropomi_processing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger()


def process_file(file_path: str, emissions_df: pd.DataFrame,
                 cfg: RegionConfig, process_id: int = 0):
    """Process a single TROPOMI file. Returns list of valid plant records."""
    t0 = time.time()
    file_name = os.path.basename(file_path)
    logger.info(f"[Process {process_id}] Processing file: {file_name}")

    data = open_tropomi_basics(file_path)
    lats, lons = data['lats'], data['lons']
    no2 = data['no2']
    wind_u, wind_v = data['wind_u'], data['wind_v']
    utc_time_data = data['utc_time']

    logger.info(f"[Process {process_id}] UTC time shape: {utc_time_data.shape}")

    min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
    min_lon, max_lon = np.nanmin(lons), np.nanmax(lons)

    tree = build_balltree(lats, lons)
    valid_no2 = get_valid_no2_mask(no2)

    valid_plants = []
    plant_count = 0
    file_valid_count = 0

    for _, emission in emissions_df.iterrows():
        plant_lat = emission[cfg.lat_col]
        plant_lon = emission[cfg.lon_col]

        plant_count += 1
        if plant_count % 1000 == 0:
            logger.info(
                f"[Process {process_id}] Checked {plant_count} plants for file: {file_name}"
            )

        if not plant_valid(
            plant_lat, plant_lon, min_lat, max_lat, min_lon, max_lon,
            tree, valid_no2,
            radius_km=cfg.radius_km, valid_ratio=cfg.valid_ratio,
        ):
            continue

        file_valid_count += 1
        grid_idx = nearest_pixel_index(tree, plant_lat, plant_lon, lats.shape)
        local_wind_u = wind_u[grid_idx]
        local_wind_v = wind_v[grid_idx]
        local_utc_time = get_local_utc_time(utc_time_data, grid_idx)

        valid_plants.append({
            'location': emission[cfg.loc_id_col],
            'latitude': plant_lat,
            'longitude': plant_lon,
            'utc_time': local_utc_time,
            'wind_u': local_wind_u,
            'wind_v': local_wind_v,
            'file_path': file_path,
        })

    elapsed = time.time() - t0
    logger.info(
        f"[Process {process_id}] Completed file: {file_name} in {elapsed:.2f}s "
        f"({file_valid_count}/{plant_count} valid)"
    )
    return valid_plants


def _worker(args):
    file_path, emissions_df, cfg, process_id = args
    return process_file(file_path, emissions_df, cfg, process_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=["us", "world"])
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: cfg.snapshots_csv)")
    parser.add_argument("--num-processes", type=int, default=None,
                        help="Override config.num_processes")
    args = parser.parse_args()

    cfg = get_config(args.region)
    output_path = args.output or cfg.snapshots_csv
    num_processes = args.num_processes or cfg.num_processes

    t0 = time.time()
    logger.info("=" * 80)
    logger.info(f"Starting TROPOMI processing (region={cfg.name})")
    logger.info(f"CPU count: {mp.cpu_count()}")
    logger.info("=" * 80)

    logger.info(f"Loading emissions from {cfg.emissions_csv}")
    emissions = pd.read_csv(cfg.emissions_csv)
    emissions = cfg.data_filter(emissions)
    logger.info(f"Loaded {len(emissions)} emission sources after filter")

    files = sorted([
        os.path.join(cfg.tropomi_dir, f)
        for f in os.listdir(cfg.tropomi_dir) if f.endswith('.nc')
    ])
    logger.info(f"Found {len(files)} TROPOMI files in {cfg.tropomi_dir}")

    valid_tropomi_list = []
    process_args = [
        (f, emissions, cfg, i % num_processes) for i, f in enumerate(files)
    ]

    logger.info(f"Using {num_processes} workers")
    with mp.Pool(processes=num_processes) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_worker, process_args, chunksize=1)
        ):
            valid_tropomi_list.extend(result)
            logger.info(
                f"Completed {i+1}/{len(files)} files. "
                f"Total valid plants so far: {len(valid_tropomi_list)}"
            )

    df = pd.DataFrame(valid_tropomi_list)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("=" * 80)
    logger.info(f"Total time: {(time.time()-t0):.2f}s")
    logger.info(f"Total valid points: {len(df)} → {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
