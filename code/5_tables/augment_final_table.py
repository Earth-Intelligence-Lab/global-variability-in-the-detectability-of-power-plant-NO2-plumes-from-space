"""
Post-processing step that adds the 17 columns missing from the 5c merge output:

    - NOx Mass (lbs)               <- sorted_hourly_emission.nc  (hourly CAMPD)
    - wind_speed                   <- np.hypot(wind_u, wind_v)
    - 15 × nearby_* columns        <- power_plants_with_yearly_nearby_stats.csv

Input:  <5c output>/updated_tropomi_hourly_emissions_full_variables.csv  (39 cols)
Output: same path (or --output) with 56 cols

For the NOx join we convert each snapshot's UTC time to the plant's local time
via TimezoneFinder, matching the CAMPD reporting convention (Date + Hour are
recorded in local standard time).

Usage:
    python augment_final_table.py --input  .../updated_tropomi_hourly_emissions_full_variables.csv
                                  --output .../augmented.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import netCDF4 as nc
from timezonefinder import TimezoneFinder
import pytz

NEARBY_COLS = [
    'nearby_plants_count_20km', 'total_emission_20km', 'percentage_emission_20km',
    'nearby_plants_count_50km', 'total_emission_50km', 'percentage_emission_50km',
    'nearby_plants_count_100km', 'total_emission_100km', 'percentage_emission_100km',
    'nearby_cities_count_20km', 'nearby_cities_pop_20km',
    'nearby_cities_count_50km', 'nearby_cities_pop_50km',
    'nearby_cities_count_100km', 'nearby_cities_pop_100km',
]


def load_hourly_nox_lookup(nc_path: str, year_min: int = 2019, year_max: int = 2024) -> dict:
    """Build (FacilityID, local_date_ns, hour) → NOxMasslbs dict from the bulk nc."""
    t0 = time.time()
    ds = nc.Dataset(nc_path)
    dates = ds.variables['Date'][:]
    hours = ds.variables['Hour'][:]
    fids = ds.variables['FacilityID'][:]
    nox = ds.variables['NOxMasslbs'][:]
    ds.close()

    epoch = np.datetime64('1970-01-01', 'ns')
    d_min = (np.datetime64(f'{year_min}-01-01') - epoch).astype(np.int64)
    d_max = (np.datetime64(f'{year_max + 1}-01-01') - epoch).astype(np.int64)
    mask = (dates >= d_min) & (dates < d_max)
    dates = dates[mask]
    hours = hours[mask]
    fids = fids[mask]
    nox = np.ma.filled(nox[mask], np.nan)

    lookup = {}
    for i in range(len(dates)):
        lookup[(int(fids[i]), int(dates[i]), int(hours[i]))] = float(nox[i])
    print(f'  loaded {len(lookup):,} emission records ({time.time()-t0:.1f}s)', flush=True)
    return lookup


def join_nox(df: pd.DataFrame, lookup: dict) -> pd.Series:
    """Return NOx Mass (lbs) series aligned to df, joining on the plant's
    LOCAL STANDARD date+hour at the time of the satellite overpass.

    EPA CAMPD reports hourly emissions in Local Standard Time (LST), with a
    fixed UTC offset year-round (no DST shift). Verified empirically: on
    spring-forward 2022-03-13 every active facility has 24 records including
    hour 2 (which doesn't exist in wall-clock time). Therefore we convert
    each TROPOMI overpass UTC time to LST using a fixed winter offset, NOT
    the DST-aware IANA conversion (which previously shifted summer queries
    by +1 hour, corrupting ~67% of the dataset's NOx labels).
    """
    import datetime as _dt
    tf = TimezoneFinder()
    utc = pd.to_datetime(df['utc_time'], utc=True, format='ISO8601')
    fids = df['location'].astype(np.int64).values
    lons = df['longitude'].astype(np.float64).values
    lats = df['latitude'].astype(np.float64).values
    out = np.full(len(df), np.nan)
    lst_tz_cache = {}
    for i in range(len(df)):
        tzname = tf.timezone_at(lng=float(lons[i]), lat=float(lats[i]))
        if not tzname:
            continue
        lst_tz = lst_tz_cache.get(tzname)
        if lst_tz is None:
            lst_off = pytz.timezone(tzname).utcoffset(_dt.datetime(2024, 1, 15))
            lst_tz = pytz.FixedOffset(int(lst_off.total_seconds() / 60))
            lst_tz_cache[tzname] = lst_tz
        local_dt = utc.iat[i].astimezone(lst_tz)
        local_date_ns = pd.Timestamp(local_dt.date()).value
        out[i] = lookup.get((int(fids[i]), int(local_date_ns), int(local_dt.hour)), np.nan)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hourly-nc',
                        default='/net/fs06/d3/rzhuang/TROPOMI/data/us/sorted_hourly_emission.nc')
    parser.add_argument('--nearby-csv',
                        default='/net/fs06/d3/rzhuang/TROPOMI/data/us/power_plants_with_yearly_nearby_stats.csv')
    parser.add_argument('--year-min', type=int, default=2019)
    parser.add_argument('--year-max', type=int, default=2024)
    args = parser.parse_args()

    print(f'Input: {args.input}')
    print(f'Output: {args.output}', flush=True)

    df = pd.read_csv(args.input)
    print(f'  loaded {len(df):,} × {len(df.columns)}', flush=True)

    print('\n[1/3] wind_speed = hypot(wind_u, wind_v)')
    df['wind_speed'] = np.hypot(df['wind_u'], df['wind_v'])

    print('\n[2/3] Joining yearly nearby stats')
    nearby = pd.read_csv(args.nearby_csv)[['Facility ID', 'Year'] + NEARBY_COLS]
    df['__year'] = pd.to_datetime(df['utc_time'], utc=True, format='ISO8601').dt.year
    df = df.merge(nearby, left_on=['location', '__year'],
                  right_on=['Facility ID', 'Year'], how='left')
    df = df.drop(columns=['Facility ID', 'Year', '__year'])
    n_nan = df['nearby_plants_count_20km'].isna().sum()
    print(f'  {n_nan:,} rows have no nearby-stats match', flush=True)

    print('\n[3/3] Joining hourly NOx Mass (lbs) via local-time key')
    lookup = load_hourly_nox_lookup(args.hourly_nc, args.year_min, args.year_max)
    df['NOx Mass (lbs)'] = join_nox(df, lookup)
    n_nox_nan = df['NOx Mass (lbs)'].isna().sum()
    print(f'  {n_nox_nan:,} rows have no hourly NOx match', flush=True)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'\n✓ wrote {len(df):,} × {len(df.columns)} → {args.output}')


if __name__ == '__main__':
    main()
