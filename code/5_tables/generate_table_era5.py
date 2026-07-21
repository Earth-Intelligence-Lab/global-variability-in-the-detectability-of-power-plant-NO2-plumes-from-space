"""
Unified ERA5 nearest-neighbor interpolation (step 5b).

Replaces us/5_generate_whole_table_era5.py and world/5_generate_whole_table_era5.py.
The two original scripts were 99% identical (only file paths and the ERA5 file
list differed). Both are now driven by config/{us,world}.py.

Usage:
    python generate_table_era5.py --region us
    python generate_table_era5.py --region world

Behavior is intended to be byte-equivalent to the originals.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from config import get_config  # noqa: E402
from shared.data_utils import (  # noqa: E402
    wrap_lons_to_grid, era5_get_grid_bounds, interp_era5_var,
)

N_JOBS = 48


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=["us", "world"])
    parser.add_argument("--input", default=None,
                        help="Input CSV (default: cfg.tropomi_table_csv)")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: cfg.era5_table_csv)")
    args = parser.parse_args()

    cfg = get_config(args.region)
    in_csv = args.input or cfg.tropomi_table_csv
    out_csv = args.output or cfg.era5_table_csv

    print("Loading plume metadata…")
    df = pd.read_csv(in_csv)
    print(f"  Plume records: {df.shape}")

    df["iso"] = (
        pd.to_datetime(df["utc_time"], utc=True)
          .dt.round("h")
          .dt.tz_localize(None)
    )
    times = df["iso"].values
    raw_lats = df["latitude"].values
    raw_lons = df["longitude"].values

    # Use first ERA5 file to determine grid bounds for clamping/wrapping
    first_path = os.path.join(cfg.era5_dir, cfg.era5_files[0])
    lon0, lon1, lat0, lat1 = era5_get_grid_bounds(first_path)
    lats = np.clip(raw_lats, lat0, lat1)
    lons = wrap_lons_to_grid(raw_lons, lon0, lon1)

    unique_times = np.unique(times)
    print(f"  Unique times to query: {len(unique_times)}")

    def interp_and_store(da: xr.DataArray, col_name: str, time_dim: str):
        print(f"    Interpolating '{col_name}' ({da.ndim}D) over {len(times)} pts…")
        arr = interp_era5_var(da, time_dim, times, lats, lons)
        df[col_name] = arr
        nan_count = np.isnan(arr).sum()
        print(f"    → '{col_name}' NaNs: {nan_count:,} "
              f"({nan_count/len(arr)*100:.1f}%)")

    t0 = time.time()
    for fname in cfg.era5_files:
        print(f"\n=== START FILE: {fname} ===")
        path = os.path.join(cfg.era5_dir, fname)
        is_nc = fname.endswith(".nc")

        if is_nc:
            with xr.open_dataset(
                path, engine="netcdf4",
                decode_timedelta=False,
                chunks={"latitude": 256, "longitude": 256},
            ) as ds:
                time_dim = "valid_time" if "valid_time" in ds.dims else "time"
                level_dim = "pressure_level" if "pressure_level" in ds.dims else "isobaricInhPa"
                print(f"  Found dims → time: {time_dim!r}, level: {level_dim!r}")
                print(f"  Variables: {list(ds.data_vars)}")

                file_times = ds[time_dim].values
                avail = np.intersect1d(unique_times, file_times)
                print(f"  File times: {len(file_times)}, overlap: {len(avail)}")
                if len(avail) == 0:
                    print("   → skip (no time overlap)")
                    continue

                idx = np.nonzero(np.isin(file_times, avail))[0]
                ds = ds.isel({time_dim: idx})

                chunk_dict = {time_dim: 1, "latitude": 256, "longitude": 256}
                if level_dim in ds.dims:
                    chunk_dict[level_dim] = 1
                ds = ds.chunk(chunk_dict)

                tasks = []
                for var in ds.data_vars:
                    levs = ds[level_dim].values if level_dim in ds[var].dims else [None]
                    for lvl in levs:
                        tasks.append((var, lvl))
                print(f"  Built {len(tasks)} tasks")

                def run(var, lvl):
                    col = var if lvl is None else f"{var}_{int(lvl)}"
                    da = ds[var]
                    if lvl is not None:
                        da = da.sel({level_dim: lvl})
                    interp_and_store(da, col, time_dim)

                Parallel(N_JOBS, backend="threading")(
                    delayed(run)(var, lvl) for var, lvl in tasks
                )

        else:
            # GRIB branch (open one level at a time)
            with xr.open_dataset(
                path, engine="cfgrib",
                decode_timedelta=False,
                chunks={"latitude": 256, "longitude": 256},
            ) as meta:
                time_dim = "valid_time" if "valid_time" in meta.dims else "time"
                level_dim = "isobaricInhPa"

                file_times = meta[time_dim].values
                avail = np.intersect1d(unique_times, file_times)
                idx = np.nonzero(np.isin(file_times, avail))[0]

                for var in meta.data_vars:
                    levels = meta[level_dim].values if level_dim in meta[var].dims else [None]
                    for lvl in levels:
                        col = var if lvl is None else f"{var}_{int(lvl)}"
                        print(f"    → processing '{col}'")

                        kwargs = dict(
                            engine="cfgrib",
                            decode_timedelta=False,
                            chunks={time_dim: 1, "latitude": 256, "longitude": 256},
                            backend_kwargs={
                                "filter_by_keys": {
                                    "shortName": var,
                                    "typeOfLevel": level_dim,
                                    "level": int(lvl),
                                }
                            } if lvl is not None else {},
                        )

                        with xr.open_dataset(path, **kwargs) as ds_lvl:
                            if not ds_lvl.data_vars:
                                continue
                            real_var = var if var in ds_lvl.data_vars else list(ds_lvl.data_vars)[0]
                            da = ds_lvl[real_var]
                            if time_dim in da.dims:
                                da = da.isel({time_dim: idx})
                            interp_and_store(da, col, time_dim)

        gc.collect()
        print(f"=== END {fname} (elapsed {(time.time()-t0)/60:.1f} min) ===")

    print("\nWriting output CSV…")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.drop(columns=["iso"]).to_csv(out_csv, index=False)
    print(f"Finished in {(time.time()-t0)/60:.1f} min → {out_csv}")


if __name__ == "__main__":
    main()
