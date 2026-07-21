"""
Join hourly NOx emission from the bulk CAMPD hourly file onto a snapshot CSV.

Replaces the deprecated 3_request_hourly_emission.py which called the EPA API
one row at a time.  The bulk file (sorted_hourly_emission.nc, ~28 GB) covers
all US facilities from 1995–2024 at hourly resolution and is indexed by
(FacilityID, Date, Hour).

This script converts each snapshot's UTC time to the plant's Local Standard
Time (LST, fixed UTC offset year-round, no DST shift) — matching the CAMPD
reporting convention — and then joins NOx Mass (lbs) via a dictionary lookup.

Usage:
    python join_hourly_emission.py --input  <snapshot_csv>
                                   --output <output_csv>

    # Example: augment the pipeline test output
    python join_hourly_emission.py \
        --input  /net/fs06/d3/rzhuang/TROPOMI/data/us/Run_20250623_203825/updated_tropomi_hourly_emissions_full_variables.csv \
        --output /tmp/augmented.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import netCDF4 as nc
import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder
import pytz

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), os.pardir))

DEFAULT_EMISSION_NC = os.path.join(
    os.path.dirname(_HERE), os.pardir, os.pardir,
    "data", "us", "sorted_hourly_emission.nc",
)

tf = TimezoneFinder()


def load_emission_lookup(nc_path: str, year_min: int = 2019,
                         year_max: int = 2024) -> dict:
    """Build a (FacilityID, date_ns, hour) → NOxMasslbs lookup dict.

    Only loads rows within [year_min, year_max] to save memory.
    """
    t0 = time.time()
    ds = nc.Dataset(nc_path)

    dates = ds.variables["Date"][:]          # int64 nanoseconds since epoch
    hours = ds.variables["Hour"][:]          # int16
    fids  = ds.variables["FacilityID"][:]    # int32
    nox   = ds.variables["NOxMasslbs"][:]    # float64, masked

    ds.close()
    print(f"  loaded {len(dates):,} rows in {time.time()-t0:.1f}s")

    # Filter to year range
    epoch = np.datetime64("1970-01-01", "ns")
    date_min = (np.datetime64(f"{year_min}-01-01") - epoch).astype(np.int64)
    date_max = (np.datetime64(f"{year_max+1}-01-01") - epoch).astype(np.int64)
    mask = (dates >= date_min) & (dates < date_max)
    dates = dates[mask]
    hours = hours[mask]
    fids  = fids[mask]
    nox   = nox[mask]
    print(f"  filtered to {year_min}–{year_max}: {len(dates):,} rows")

    # Build dict.  nox is a masked array — fill masked with NaN.
    nox_filled = np.ma.filled(nox, np.nan)
    lookup = {}
    for i in range(len(dates)):
        key = (int(fids[i]), int(dates[i]), int(hours[i]))
        lookup[key] = float(nox_filled[i])
    print(f"  lookup dict: {len(lookup):,} entries ({time.time()-t0:.1f}s)")
    return lookup


import datetime as _dt
_LST_TZ_CACHE: dict = {}


def utc_to_local_date_hour(utc_dt, lon, lat):
    """Convert a UTC datetime to (lst_date_ns, lst_hour) using a fixed LST
    offset (no DST shift), matching the EPA CAMPD bulk-file convention."""
    tzname = tf.timezone_at(lng=float(lon), lat=float(lat))
    if not tzname:
        return None, None
    lst_tz = _LST_TZ_CACHE.get(tzname)
    if lst_tz is None:
        lst_off = pytz.timezone(tzname).utcoffset(_dt.datetime(2024, 1, 15))
        lst_tz = pytz.FixedOffset(int(lst_off.total_seconds() / 60))
        _LST_TZ_CACHE[tzname] = lst_tz
    local_dt = utc_dt.astimezone(lst_tz)
    # Date as nanoseconds-since-epoch (midnight of that day)
    local_date = pd.Timestamp(local_dt.date()).value  # int64 ns
    return local_date, local_dt.hour


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input CSV with utc_time, location, latitude, longitude")
    parser.add_argument("--output", required=True, help="Output CSV with NOx Mass (lbs) added")
    parser.add_argument("--emission-nc", default=DEFAULT_EMISSION_NC,
                        help=f"Bulk hourly emission NetCDF (default: {DEFAULT_EMISSION_NC})")
    parser.add_argument("--year-min", type=int, default=2019)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--col-name", default="NOx Mass (lbs)",
                        help="Output column name for the joined emission")
    args = parser.parse_args()

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Emission NC: {args.emission_nc}")

    # 1) Load emission lookup
    print("\nLoading emission lookup…")
    lookup = load_emission_lookup(args.emission_nc, args.year_min, args.year_max)

    # 2) Load snapshot CSV
    print("\nLoading snapshot CSV…")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} rows, {len(df.columns)} cols")

    # 3) Join
    print("\nJoining…")
    t0 = time.time()
    utc_times = pd.to_datetime(df["utc_time"], utc=True, format="ISO8601")
    results = np.full(len(df), np.nan, dtype=np.float64)

    for i in range(len(df)):
        utc_dt = utc_times.iloc[i].to_pydatetime()
        lon = df["longitude"].iloc[i]
        lat = df["latitude"].iloc[i]
        local_date_ns, local_hour = utc_to_local_date_hour(utc_dt, lon, lat)
        if local_date_ns is None:
            continue
        key = (int(df["location"].iloc[i]), local_date_ns, local_hour)
        results[i] = lookup.get(key, np.nan)

    df[args.col_name] = results
    matched = int(np.isfinite(results).sum())
    print(f"  matched: {matched:,} / {len(df):,} ({matched/len(df)*100:.1f}%)")
    print(f"  elapsed: {time.time()-t0:.1f}s")

    # 4) Write
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✓ wrote {args.output}")


if __name__ == "__main__":
    main()
