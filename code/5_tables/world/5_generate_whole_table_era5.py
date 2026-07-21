#!/usr/bin/env python3
"""
Vectorised + multi‑threaded interpolation of every ERA5 field to every
TROPOMI‑plume point (nearest‑neighbour) without huge intermediate arrays.
For GRIB files we open **one pressure level at a time**; for NetCDF we keep
the old all‑vars‑at‑once path.  Now with extra prints!
"""

import os, time, gc
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

# ───────────────────────── config ─────────────────────────
DATA_DIR   = "../data"
META_CSV   = os.path.join(DATA_DIR, "Run_1/valid_tropomi_emissions_with_qa_with_all_vars.csv")
OUT_CSV    = os.path.join(DATA_DIR, "Run_1/processed_valid_tropomi_emissions_with_qa_with_all_vars.csv")
ERA5_DIR   = os.path.join(DATA_DIR, "era5_compact")
N_JOBS     = 48
ERA5_FILES = [
    "TOA_incident_solar_radiation.nc",
    "Total_column_water_vapour.nc",
    "temperature.nc"
]

# ───────────────────── plume metadata ─────────────────────
print("Loading plume metadata…")
df = pd.read_csv(META_CSV)
print(f"  Plume records: {df.shape}")

df["iso"] = (
    pd.to_datetime(df["utc_time"], utc=True)
      .dt.round("h")
      .dt.tz_localize(None)
)
times      = df["iso"].values   

raw_lats   = df["latitude"].values
raw_lons   = df["longitude"].values

# peek at your first file’s lon/lat bounds
with xr.open_dataset(os.path.join(ERA5_DIR, ERA5_FILES[0]),
                     engine="netcdf4") as tmp:
    lon0, lon1 = tmp.longitude.min().item(), tmp.longitude.max().item()
    lat0, lat1 = tmp.latitude .min().item(), tmp.latitude .max().item()
# clamp into valid ERA5 lat-range
lats = np.clip(raw_lats, lat0, lat1)

# wrap longitudes into whichever interval your file uses
if lon0 >= 0:
    # 0…360 system:
    # 1) bring negatives into [0,360)
    wrapped = np.where(raw_lons < 0, raw_lons + 360, raw_lons)

    # 2) any points already in [lon0, lon1] stay where they are
    lons = np.clip(wrapped, lon0, lon1)

    # 3) for wrapped > lon1, snap to whichever boundary (0 or lon1) is nearer on the globe
    mask = wrapped > lon1
    if mask.any():
        # distance (circular) to 0°
        d0 = np.minimum(wrapped[mask], 360 - wrapped[mask])
        # distance (circular) to lon1
        diff = np.abs(wrapped[mask] - lon1)
        d1   = np.minimum(diff, 360 - diff)
        # choose boundary
        lons[mask] = np.where(d0 < d1, lon0, lon1)

elif lon1 <= 180:
    # –180…+180 system:
    # wrap into [–180, +180)
    lons = ((raw_lons + 180) % 360) - 180

else:
    # fallback if grid uses a nonstandard range
    lons = raw_lons

unique_times = np.unique(times)
print(f"  Unique times to query: {len(unique_times)}")

# ───────────── helper: interpolate + store in df ──────────
def interp_and_store(da: xr.DataArray, col_name: str, time_dim: str):
    print(f"    Interpolating '{col_name}' ({da.ndim}D) over {len(times)} points…")
    out = da.interp(
        **{time_dim: ("points", times)},
        latitude  = ("points", lats),
        longitude = ("points", lons),
        method="nearest",
    )
    arr = out.compute().values
    print(f"    → result shape {arr.shape}, writing df['{col_name}']")
    df[col_name] = arr
    # report NaNs count for this variable
    nan_count = np.isnan(arr).sum()
    print(f"    → '{col_name}' NaNs: {nan_count:,} ({nan_count/len(arr)*100:.1f}%)")
    
    # Debug: show sample NaN locations for blh
    if col_name == "blh" and nan_count:
        # show where the NaNs occur
        mask = np.isnan(arr)
        sample_idx = np.flatnonzero(mask)[:10]
        print("    → sample NaN rows for 'blh':")
        for idx in sample_idx:
            print(f"       i={idx:<7} time={times[idx]} lat={lats[idx]:7.3f} lon={lons[idx]:8.3f}")


# ───────────────────────── main loop ─────────────────────
t0 = time.time()
for fname in ERA5_FILES:
    print(f"\n=== START FILE: {fname} ===")
    path  = os.path.join(ERA5_DIR, fname)
    is_nc = fname.endswith(".nc")
    print(f"Opening {'NetCDF4' if is_nc else 'cfgrib'} dataset…")

    if is_nc:
        # ---------- NetCDF branch ----------
        with xr.open_dataset(
            path,
            engine="netcdf4",
            decode_timedelta=False,
            chunks={"latitude":256, "longitude":256},
        ) as ds:
            time_dim  = "valid_time" if "valid_time" in ds.dims else "time"
            level_dim = "pressure_level" if "pressure_level" in ds.dims else "isobaricInhPa"
            print(f"  Found dims → time: {time_dim!r}, level: {level_dim!r}")
            print(f"  Variables: {list(ds.data_vars)}")

            file_times = ds[time_dim].values
            avail      = np.intersect1d(unique_times, file_times)
            print(f"  File times: {len(file_times)}, overlap: {len(avail)}")
            if len(avail) == 0:
                print("   → skip (no time overlap)")
                continue

            idx = np.nonzero(np.isin(file_times, avail))[0]
            print(f"  Selecting {len(idx)} time slices by index")
            ds = ds.isel({time_dim: idx})

            chunk_dict = {time_dim:1, "latitude":256, "longitude":256}
            if level_dim in ds.dims:
                chunk_dict[level_dim] = 1
            print(f"  Rechunking with {chunk_dict}")
            ds = ds.chunk(chunk_dict)

            # Build tasks
            tasks = []
            for var in ds.data_vars:
                levs = ds[level_dim].values if level_dim in ds[var].dims else [None]
                for lvl in levs:
                    tasks.append((var, lvl))
            print(f"  Built {len(tasks)} tasks")

            # Run
            def run(var, lvl):
                col = var if lvl is None else f"{var}_{int(lvl)}"
                da  = ds[var]
                if lvl is not None:
                    print(f"    → selecting level {lvl} for '{var}'")
                    da = da.sel({level_dim: lvl})
                interp_and_store(da, col, time_dim)

            print("  Launching Parallel interpolation…")
            Parallel(N_JOBS, backend="threading")(
                delayed(run)(var, lvl) for var,lvl in tasks
            )

    else:
        # ---------- GRIB branch (open one level at a time) ----------
        with xr.open_dataset(
            path,
            engine="cfgrib",
            decode_timedelta=False,
            chunks={"latitude":256, "longitude":256},
        ) as meta:
            time_dim  = "valid_time" if "valid_time" in meta.dims else "time"
            level_dim = "isobaricInhPa"

            file_times = meta[time_dim].values
            avail      = np.intersect1d(unique_times, file_times)
            idx        = np.nonzero(np.isin(file_times, avail))[0]

            for var in meta.data_vars:
                levels = meta[level_dim].values if level_dim in meta[var].dims else [None]
                for lvl in levels:
                    col = var if lvl is None else f"{var}_{int(lvl)}"
                    print(f"    → processing '{col}'")

                    # now use 'level' as the GRIB key, not the dim-name:
                    kwargs = dict(
                        engine="cfgrib",
                        decode_timedelta=False,
                        chunks={time_dim:1, "latitude":256, "longitude":256},
                        backend_kwargs={
                            "filter_by_keys": {
                                "shortName":   var,
                                "typeOfLevel": level_dim,
                                "level":       int(lvl),
                            }
                        } if lvl is not None else {}
                    )

                    with xr.open_dataset(path, **kwargs) as ds_lvl:
                        print(f"      ds_vars = {list(ds_lvl.data_vars)}")
                        if not ds_lvl.data_vars:
                            print("      empty slice, skip")
                            continue

                        real_var = var if var in ds_lvl.data_vars else list(ds_lvl.data_vars)[0]
                        da       = ds_lvl[real_var]

                        if time_dim in da.dims:
                            print(f"      slicing time dim by {len(idx)} indices")
                            da = da.isel({time_dim: idx})
                        else:
                            print("      scalar-time slice, no indexing")

                        interp_and_store(da, col, time_dim)

    gc.collect()
    print(f"=== END {fname} (elapsed {(time.time()-t0)/60:.1f} min) ===")

print("\nWriting output CSV…")
df.drop(columns=["iso"], axis=1).to_csv(OUT_CSV, index=False)
print(f"Finished in {(time.time()-t0)/60:.1f} min → {OUT_CSV}")