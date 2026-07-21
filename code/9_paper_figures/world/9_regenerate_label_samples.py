"""Paper-revision sample regeneration. Single script that handles both the
GLOBAL and U.S. datasets.

Three modes follow the paper:

  - `tune`           : 200 obs total = 100 GLOBAL stratified across 6
                       continents x 5 emission quantiles (~3-4 per cell)
                       + 100 U.S. stratified across 5 emission quantiles
                       (~20 per bin). Paper Sect. 3.2.1.
  - `val_random`     : 400 U.S. obs stratified across 5 emission quantiles
                       (~80 per bin). Paper Sect. 3.2.7 / Fig 5c (U.S.-side
                       validation).
  - `val_stratified` : 400 GLOBAL obs = 80 per emission bin x 5 bins, 80 obs
                       per bin distributed evenly across 6 continents (~13
                       per cell). Paper Sect. 3.2.7 / Fig 5d.

The `val_stratified` global validation drops Antarctica (no power plants
above the CoCO2 catalog's emission threshold; the catalog's 13,141 plants
include zero Antarctic entries). Rows whose ISO3 country code does not map
to one of the six continents are filtered out before sampling.

Labeling parameters match the production runs:
- GLOBAL rows -> TROPOMIConfig in `4_sampling/world/4_sample_snapshots_label.py`
- U.S.  rows -> plotting_parameters in `4_sampling/us/4_sample_snapshots_label.py`

Usage:
    python 9_regenerate_label_samples.py --mode val_stratified
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm

# `process_zoomed_data` lives in shared/plotting.py.
sys.path.insert(0, '/net/fs06/d3/rzhuang/TROPOMI/code/shared')
from plotting import process_zoomed_data  # noqa: E402

plt.rcdefaults()


# ─── Paths (paper revision; both sides use 100 m wind labelling) ────────────
# U.S. (Run_100m_20260414): 100 m ERA5 wind, local-tz-corrected hourly NOx.
US_HOURLY_CSV    = '/net/fs06/d3/rzhuang/TROPOMI/pipeline_100m_run/Run_100m_20260414/updated_tropomi_hourly_emissions_full_variables_augmented_localtz.csv'
# By-year US plants CSV (Facility ID x Year): each (id, year) row carries
# `nearby_plants_count_{20,50,100}km`, `total_emission_{...}km`,
# `percentage_emission_{...}km`, `nearby_cities_count_{20,50,100,200}km`,
# `nearby_cities_pop_{20,50,100,200}km` for that year. The labeling pipeline
# filters this frame per-row to the snapshot's year before plotting, so each
# annotated panel reflects the correct year's interference statistics.
#
# Note: the original `power_plants_with_yearly_nearby_stats.csv` only goes up
# to 100 km city radii; this `_200km` variant has the 200 km columns added on
# top via a one-shot BallTree query against worldcities.csv (pop >= 200,000).
US_FACILITY_CSV  = '/net/fs06/d3/rzhuang/TROPOMI/data/us/power_plants_with_yearly_nearby_stats_200km.csv'
US_ANNUAL_CSV    = '/net/fs06/d3/rzhuang/TROPOMI/data/us/annual-emissions-facility-aggregation-2019-2024.csv'

# Global (2026-04-28 100 m re-labelling, stage-2 output): ERA5 100 m wind
# replaces the 10 m TROPOMI-embedded wind that Run_3 used. file_path is
# already absolute in this CSV.
WORLD_EMIT_CSV   = '/net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m/Run_100m_20260428/valid_tropomi_emissions_with_qa.csv'
WORLD_LOC_CSV    = '/net/fs06/d3/rzhuang/TROPOMI/data/world/power_plant_location/power_plants_with_combined_nearby_stats.csv'
WORLD_CITIES_CSV = '/net/fs06/d3/rzhuang/TROPOMI/data/world/worldcities.csv'

# ─── ISO3 -> continent mapping (verbatim from
#     4_sampling/world/4_sample_snapshots_label_continent_emission.py) ──────
ISO3_TO_CONTINENT = {
    # Africa
    'DZA': 'Africa', 'AGO': 'Africa', 'BEN': 'Africa', 'BWA': 'Africa', 'BFA': 'Africa',
    'BDI': 'Africa', 'CMR': 'Africa', 'CPV': 'Africa', 'CAF': 'Africa', 'TCD': 'Africa',
    'COM': 'Africa', 'COG': 'Africa', 'COD': 'Africa', 'CIV': 'Africa', 'DJI': 'Africa',
    'EGY': 'Africa', 'GNQ': 'Africa', 'ERI': 'Africa', 'ETH': 'Africa', 'GAB': 'Africa',
    'GMB': 'Africa', 'GHA': 'Africa', 'GIN': 'Africa', 'GNB': 'Africa', 'KEN': 'Africa',
    'LSO': 'Africa', 'LBR': 'Africa', 'LBY': 'Africa', 'MDG': 'Africa', 'MWI': 'Africa',
    'MLI': 'Africa', 'MRT': 'Africa', 'MUS': 'Africa', 'MAR': 'Africa', 'MOZ': 'Africa',
    'NAM': 'Africa', 'NER': 'Africa', 'NGA': 'Africa', 'RWA': 'Africa', 'STP': 'Africa',
    'SEN': 'Africa', 'SYC': 'Africa', 'SLE': 'Africa', 'SOM': 'Africa', 'ZAF': 'Africa',
    'SSD': 'Africa', 'SDN': 'Africa', 'SWZ': 'Africa', 'TZA': 'Africa', 'TGO': 'Africa',
    'TUN': 'Africa', 'UGA': 'Africa', 'ZMB': 'Africa', 'ZWE': 'Africa',
    # Asia
    'AFG': 'Asia', 'ARM': 'Asia', 'AZE': 'Asia', 'BHR': 'Asia', 'BGD': 'Asia', 'BTN': 'Asia',
    'BRN': 'Asia', 'KHM': 'Asia', 'CHN': 'Asia', 'CYP': 'Asia', 'GEO': 'Asia', 'IND': 'Asia',
    'IDN': 'Asia', 'IRN': 'Asia', 'IRQ': 'Asia', 'ISR': 'Asia', 'JPN': 'Asia', 'JOR': 'Asia',
    'KAZ': 'Asia', 'KWT': 'Asia', 'KGZ': 'Asia', 'LAO': 'Asia', 'LBN': 'Asia', 'MYS': 'Asia',
    'MDV': 'Asia', 'MNG': 'Asia', 'MMR': 'Asia', 'NPL': 'Asia', 'PRK': 'Asia', 'OMN': 'Asia',
    'PAK': 'Asia', 'PSE': 'Asia', 'PHL': 'Asia', 'QAT': 'Asia', 'SAU': 'Asia', 'SGP': 'Asia',
    'KOR': 'Asia', 'LKA': 'Asia', 'SYR': 'Asia', 'TWN': 'Asia', 'TJK': 'Asia', 'THA': 'Asia',
    'TLS': 'Asia', 'TUR': 'Asia', 'TKM': 'Asia', 'ARE': 'Asia', 'UZB': 'Asia', 'VNM': 'Asia',
    'YEM': 'Asia', 'HKG': 'Asia', 'MAC': 'Asia',
    # Europe
    'ALB': 'Europe', 'AND': 'Europe', 'AUT': 'Europe', 'BLR': 'Europe', 'BEL': 'Europe',
    'BIH': 'Europe', 'BGR': 'Europe', 'HRV': 'Europe', 'CZE': 'Europe', 'DNK': 'Europe',
    'EST': 'Europe', 'FIN': 'Europe', 'FRA': 'Europe', 'DEU': 'Europe', 'GRC': 'Europe',
    'HUN': 'Europe', 'ISL': 'Europe', 'IRL': 'Europe', 'ITA': 'Europe', 'XKX': 'Europe',
    'LVA': 'Europe', 'LIE': 'Europe', 'LTU': 'Europe', 'LUX': 'Europe', 'MKD': 'Europe',
    'MLT': 'Europe', 'MDA': 'Europe', 'MCO': 'Europe', 'MNE': 'Europe', 'NLD': 'Europe',
    'NOR': 'Europe', 'POL': 'Europe', 'PRT': 'Europe', 'ROU': 'Europe', 'RUS': 'Europe',
    'SMR': 'Europe', 'SRB': 'Europe', 'SVK': 'Europe', 'SVN': 'Europe', 'ESP': 'Europe',
    'SWE': 'Europe', 'CHE': 'Europe', 'UKR': 'Europe', 'GBR': 'Europe', 'VAT': 'Europe',
    'GIB': 'Europe', 'FRO': 'Europe', 'GGY': 'Europe', 'IMN': 'Europe', 'JEY': 'Europe',
    # North America
    'ATG': 'North America', 'BHS': 'North America', 'BRB': 'North America', 'BLZ': 'North America',
    'CAN': 'North America', 'CRI': 'North America', 'CUB': 'North America', 'DMA': 'North America',
    'DOM': 'North America', 'SLV': 'North America', 'GRD': 'North America', 'GTM': 'North America',
    'HTI': 'North America', 'HND': 'North America', 'JAM': 'North America', 'MEX': 'North America',
    'NIC': 'North America', 'PAN': 'North America', 'KNA': 'North America', 'LCA': 'North America',
    'VCT': 'North America', 'TTO': 'North America', 'USA': 'North America', 'BMU': 'North America',
    'GRL': 'North America', 'SPM': 'North America',
    # South America
    'ARG': 'South America', 'BOL': 'South America', 'BRA': 'South America', 'CHL': 'South America',
    'COL': 'South America', 'ECU': 'South America', 'GUF': 'South America', 'GUY': 'South America',
    'PRY': 'South America', 'PER': 'South America', 'SUR': 'South America', 'URY': 'South America',
    'VEN': 'South America',
    # Oceania
    'AUS': 'Oceania', 'FJI': 'Oceania', 'KIR': 'Oceania', 'MHL': 'Oceania', 'FSM': 'Oceania',
    'NRU': 'Oceania', 'NZL': 'Oceania', 'PLW': 'Oceania', 'PNG': 'Oceania', 'WSM': 'Oceania',
    'SLB': 'Oceania', 'TON': 'Oceania', 'TUV': 'Oceania', 'VUT': 'Oceania', 'NCL': 'Oceania',
    'PYF': 'Oceania', 'GUM': 'Oceania', 'ASM': 'Oceania', 'COK': 'Oceania', 'NIU': 'Oceania',
    'NFK': 'Oceania', 'MNP': 'Oceania', 'TKL': 'Oceania', 'WLF': 'Oceania',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', default='tune',
                   choices=['tune', 'val_random', 'val_stratified'])
    p.add_argument('--n', type=int, default=None,
                   help='Override sample size (default: tune=100, val_*=400).')
    p.add_argument('--seed', type=int, default=42,
                   help='Sampling RNG seed.')
    p.add_argument('--n_emission_bins', type=int, default=5)
    return p.parse_args()


def attach_continent(df, country_col='country'):
    df = df.copy()
    df['continent'] = df[country_col].map(ISO3_TO_CONTINENT).fillna('Unknown')
    unknown = df[df['continent'] == 'Unknown'][country_col].unique()
    if len(unknown):
        print(f"Unmapped ISO3 codes ({len(unknown)}): "
              f"{sorted(unknown)[:15]}{'...' if len(unknown) > 15 else ''}")
    return df


def _draw_balanced(group_df, n, group_col, rng_seed):
    """Draw `n` rows from `group_df`, distributing roughly evenly across
    distinct values of `group_col` (~ floor(n / k) per group, then top up
    randomly from leftover rows to hit exactly `n`)."""
    if len(group_df) <= n:
        return group_df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

    groups = list(group_df.groupby(group_col))
    k = len(groups)
    base = n // k
    parts = []
    for _, g in groups:
        parts.append(g.sample(n=min(base, len(g)), random_state=rng_seed))
    out = pd.concat(parts) if parts else group_df.iloc[:0]

    if len(out) < n:
        leftover = group_df.drop(out.index, errors='ignore')
        if len(leftover) > 0:
            top_up = leftover.sample(n=min(n - len(out), len(leftover)),
                                     random_state=rng_seed)
            out = pd.concat([out, top_up])
    return out


def sample_paper(df, mode, n=None, n_bins=5, rng_seed=42,
                 emission_col='annual_nox_emission'):
    """Draw the paper-method sample from the full world dataset."""
    if n is None:
        n = 100 if mode == 'tune' else 400

    if mode == 'val_random':
        return df.sample(n=min(n, len(df)), random_state=rng_seed)

    sub = df.dropna(subset=[emission_col]).copy()
    sub = sub[pd.to_numeric(sub[emission_col], errors='coerce') > 0]
    if sub.empty:
        raise ValueError(f"No rows with positive '{emission_col}'.")

    sub['_bin'] = pd.qcut(pd.to_numeric(sub[emission_col], errors='coerce'),
                          q=n_bins, labels=False, duplicates='drop')

    if mode == 'val_stratified':
        # 400 obs = 80 per emission bin x 5 bins; within each bin, balance
        # across continents (paper Sect. 3.2.7: "80 obs per emission bin").
        per_bin = n // sub['_bin'].nunique()
        parts = []
        for _, bin_df in sub.groupby('_bin'):
            parts.append(_draw_balanced(bin_df, per_bin, 'continent', rng_seed))
        out = pd.concat(parts).drop(columns='_bin')
    else:  # tune
        # 100 obs across (continent x emission) cells; ~3-4 per cell when
        # 6 continents x 5 bins = 30 cells. Equivalent to balanced draw on
        # the joint key.
        sub['_cell'] = sub['continent'].astype(str) + '|' + sub['_bin'].astype(str)
        out = _draw_balanced(sub, n, '_cell', rng_seed).drop(columns=['_bin', '_cell'])

    if len(out) > n:
        out = out.sample(n=n, random_state=rng_seed)
    return out


def load_cities():
    min_pop, lat_c, lon_c, name_c, pop_c = 200000, 'latitude', 'longitude', 'name', 'population'
    try:
        cities = pd.read_csv(WORLD_CITIES_CSV)
        cities[lat_c] = pd.to_numeric(cities[lat_c], errors='coerce')
        cities[lon_c] = pd.to_numeric(cities[lon_c], errors='coerce')
        cities = cities.dropna(subset=[lat_c, lon_c])
        if pop_c in cities.columns:
            cities[pop_c] = pd.to_numeric(cities[pop_c], errors='coerce')
            cities = cities.dropna(subset=[pop_c])
            cities = cities[cities[pop_c] >= min_pop].copy()
        print(f"Cities: {len(cities)} (pop >= {min_pop})")
        return cities
    except Exception as e:
        print(f"Cities load error: {e}")
        return pd.DataFrame(columns=[name_c, lat_c, lon_c])


def load_locations():
    print(f"Loading locations: {WORLD_LOC_CSV}")
    df = pd.read_csv(WORLD_LOC_CSV)
    rename = {'Facility ID': 'ID', 'Facility Name': 'facility_name',
              'NOx Mass (short tons)': 'nox_emis_ty',
              'Latitude': 'latitude', 'Longitude': 'longitude',
              'lat': 'latitude', 'lon': 'longitude'}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if 'latitude'  in df.columns: df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
    if 'longitude' in df.columns: df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    if 'nox_emis_ty' in df.columns: df['nox_emis_ty'] = pd.to_numeric(df['nox_emis_ty'], errors='coerce')
    if 'ID' not in df.columns:
        raise ValueError("'ID' missing in locations_df")
    print(f"Locations: {df.shape}")
    return df


def process_and_save_row(args):
    row_tuple, row_index, locs, cities, params, outdir = args
    row = pd.Series(row_tuple, index=row_index)
    fig = None
    location_id = row.get('location', 'UnknownLocation')
    iso_code    = row.get('country', row.get('ISO3', 'UnknownISO'))
    time        = row.get('utc_time', 'UnknownTime')
    save_path   = os.path.join(outdir, f"sampled_location_{location_id}_{iso_code}_{time}.png")
    try:
        fig = process_zoomed_data(
            row=row,
            global_locations_df=(locs.copy()   if locs   is not None else pd.DataFrame()),
            cities_df=         (cities.copy() if cities is not None else pd.DataFrame()),
            **params)
        if fig:
            fig.savefig(save_path, dpi=params.get('plot_dpi', 200))
            plt.close(fig)
            return save_path
        return None
    except Exception as e:
        print(f"Error for {location_id} ({iso_code}): {type(e).__name__} - {e}")
        if fig:
            plt.close(fig)
        return None


def load_us_locations():
    """U.S. plants with the columns `process_zoomed_data` expects. The
    `power_plants_with_yearly_nearby_stats.csv` is one row per (Facility ID,
    Year), so we keep the `Year` column for downstream year-filtering before
    each plot call."""
    print(f"Loading U.S. locations: {US_FACILITY_CSV}")
    df = pd.read_csv(US_FACILITY_CSV)
    rename = {
        'Facility ID':           'ID',
        'Facility_ID':           'ID',
        'Facility Name':         'facility_name',
        'Facility_Name':         'facility_name',
        'NOx Mass (short tons)': 'nox_emis_ty',
        'Total_NOx_Mass':        'nox_emis_ty',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if 'Latitude'    in df.columns: df['latitude']    = pd.to_numeric(df['Latitude'],    errors='coerce')
    if 'Longitude'   in df.columns: df['longitude']   = pd.to_numeric(df['Longitude'],   errors='coerce')
    if 'nox_emis_ty' in df.columns: df['nox_emis_ty'] = pd.to_numeric(df['nox_emis_ty'], errors='coerce')
    if 'Year'        in df.columns: df['Year']        = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    if 'ID' not in df.columns:
        raise ValueError(f"'ID' missing in US locations_df; columns: {df.columns.tolist()[:10]}")
    print(f"U.S. locations: {df.shape}  (years: {sorted(df['Year'].dropna().unique().tolist())})")
    return df


def sample_us_tune(df, n=100, n_bins=5, rng_seed=42,
                   emission_col='annual_nox_emission'):
    """100 U.S. obs stratified across 5 emission quantiles (~20 per bin)."""
    sub = df.dropna(subset=[emission_col]).copy()
    sub = sub[pd.to_numeric(sub[emission_col], errors='coerce') > 0]
    if sub.empty:
        raise ValueError(f"No U.S. rows with positive '{emission_col}'.")
    sub['_bin'] = pd.qcut(pd.to_numeric(sub[emission_col], errors='coerce'),
                          q=n_bins, labels=False, duplicates='drop')
    out = _draw_balanced(sub, n, '_bin', rng_seed).drop(columns='_bin')
    if len(out) > n:
        out = out.sample(n=n, random_state=rng_seed)
    return out


# ─── Production labeling parameters (paper Sect. 3.1.2 - 3.1.6) ─────────────
# One shared dict for both U.S. and global rows.
PROD_PARAMS = {
    # Step 1: Masking interference zones
    'interf_max_distance_km':   150,        # 150 km search radius
    'interf_city_pop_thresh':   200000,     # city population threshold
    'interf_plant_emis_thresh': 1.0,        # plant emission ratio >= 1.0 -> interfering
    'city_base_radius':         0.0,        # city radius = scale*log10(pop), [10, 90] km
    'city_pop_scale':           9.0,
    'city_radius_min':          10.0,
    'city_radius_max':          90.0,
    'plant_base_radius':        20.0,       # plant radius fixed at 20 km
    'plant_emission_scale':     0.0,
    'plant_radius_min':         20.0,
    'plant_radius_max':         20.0,
    'close_distance_km_mask':   20,         # enforce full 20 km plant mask
    'max_angle_diff_mask':      0,
    # Step 2: Downwind plume zone
    'max_distance_km':          20.0,       # max search distance
    'close_distance_km':        5.0,        # close-range distance
    'max_angle_diff':           25.0,       # +/- 25 deg wind cone
    'flagged_area':             25.0,       # min plume area km^2
    # Step 4: Background
    'background_mode':          'directional',
    'upwind_angle_tolerance':   60,         # +/- 60 deg upwind sector
    'background_dist_min_km':   10,
    'background_dist_max_km':   100,
    # Step 5: Detection thresholds
    'threshold_factor':         2.0,        # 2-sigma statistical significance
    'threshold_abs_min':        5e-6,       # absolute minimum mol/m^2
    'threshold_radius_km':      50.0,       # characterized at 50 km
    # Misc
    'zoom_radius_km':           100,        # 100 km analysis window
    'sigma':                    10,
    'plot_dpi':                 200,
    'plot_interference_zones':  True,
}


def load_global_snapshots():
    print(f"Loading global snapshots: {WORLD_EMIT_CSV}")
    df = pd.read_csv(WORLD_EMIT_CSV, low_memory=False)
    print(f"Global snapshots: {df.shape}")
    df['wind_u'] = pd.to_numeric(df['wind_u'], errors='coerce')
    df['wind_v'] = pd.to_numeric(df['wind_v'], errors='coerce')
    if 'annual_nox_emission' not in df.columns and 'nox_emis_ty' in df.columns:
        df['annual_nox_emission'] = pd.to_numeric(df['nox_emis_ty'], errors='coerce')
    df = attach_continent(df, country_col='country')
    # The CoCO2 catalog has zero Antarctic plants (research-station diesel
    # gensets are below its emission threshold); this filter mainly protects
    # against any unmapped territory codes forming an "Unknown" pseudo-stratum.
    n_before = len(df)
    df = df[df['continent'] != 'Unknown'].copy()
    if n_before != len(df):
        print(f"Dropped {n_before - len(df):,} rows with Unknown continent "
              f"({len(df):,} remain).")
    return df


def load_us_snapshots():
    print(f"Loading U.S. snapshots: {US_HOURLY_CSV}")
    df = pd.read_csv(US_HOURLY_CSV, low_memory=False)
    print(f"U.S. snapshots: {df.shape}")
    df['wind_u'] = pd.to_numeric(df['wind_u'], errors='coerce')
    df['wind_v'] = pd.to_numeric(df['wind_v'], errors='coerce')
    if 'annual_nox_emission' not in df.columns and 'NOx Mass (short tons)' in df.columns:
        df['annual_nox_emission'] = pd.to_numeric(df['NOx Mass (short tons)'], errors='coerce')
    return df


if __name__ == "__main__":
    args = parse_args()

    out_dir_base = '/net/fs06/d3/rzhuang/TROPOMI/data/world/paper_figures/labeling_samples'
    OUTPUT_FIGURE_DIR = os.path.join(out_dir_base, args.mode)
    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

    cities = load_cities()
    locations    = None  # global locations (loaded only if a global sample is needed)
    us_locations = None  # U.S. locations  (loaded only if a U.S. sample is needed)

    parts = []  # list of (sampled_df, dataset_label) to concat for labeling

    if args.mode == 'tune':
        # 100 global + 100 U.S.
        df = load_global_snapshots()
        sampled_global = sample_paper(df, mode='tune',
                                      n=args.n if args.n else 100,
                                      n_bins=args.n_emission_bins,
                                      rng_seed=args.seed)
        sampled_global['_dataset'] = 'global'
        locations = load_locations()
        parts.append(sampled_global)

        us_df = load_us_snapshots()
        sampled_us = sample_us_tune(us_df,
                                    n=args.n if args.n else 100,
                                    n_bins=args.n_emission_bins,
                                    rng_seed=args.seed)
        sampled_us['_dataset'] = 'us'
        us_locations = load_us_locations()
        parts.append(sampled_us)

    elif args.mode == 'val_random':
        # 400 U.S. obs stratified across 5 emission quantiles (Fig 5c, U.S.-side).
        us_df = load_us_snapshots()
        sampled_us = sample_us_tune(us_df,
                                    n=args.n if args.n else 400,
                                    n_bins=args.n_emission_bins,
                                    rng_seed=args.seed)
        sampled_us['_dataset'] = 'us'
        us_locations = load_us_locations()
        parts.append(sampled_us)

    elif args.mode == 'val_stratified':
        # 400 global obs, 80 per emission bin x 5 bins, balanced across 6 continents.
        df = load_global_snapshots()
        sampled_global = sample_paper(df, mode='val_stratified',
                                      n=args.n if args.n else 400,
                                      n_bins=args.n_emission_bins,
                                      rng_seed=args.seed)
        sampled_global['_dataset'] = 'global'
        locations = load_locations()
        parts.append(sampled_global)

    sampled_df = pd.concat(parts, ignore_index=True, sort=False)
    sampled_path = os.path.join(OUTPUT_FIGURE_DIR, 'sampled_emission_snapshots.csv')
    sampled_df.to_csv(sampled_path, index=False)
    print(f"\nTotal sampled: {len(sampled_df)} rows -> {sampled_path}")
    print('Per-dataset counts:')
    print(sampled_df['_dataset'].value_counts())

    # Pre-split US locations by year so we can pass the right slice to each
    # snapshot. The by-year CSV has one row per (Facility ID, Year); the
    # plotting helper looks up `nearby_*` stats by ID, so we hand it the year
    # that matches the snapshot's utc_time.
    us_locations_by_year = {}
    if us_locations is not None and 'Year' in us_locations.columns:
        for y, g in us_locations.dropna(subset=['Year']).groupby('Year'):
            us_locations_by_year[int(y)] = g.reset_index(drop=True)

    def _us_locs_for_year(yr):
        if yr in us_locations_by_year:
            return us_locations_by_year[yr]
        # Fall back to the closest year if the snapshot's year is outside the
        # CSV's coverage (shouldn't happen for 2019-2024 production data).
        if not us_locations_by_year:
            return us_locations
        nearest = min(us_locations_by_year, key=lambda y: abs(y - yr))
        return us_locations_by_year[nearest]

    # ── Build per-row label/plot tasks ─────────────────────────────────────
    args_list = []
    for _, row in sampled_df.iterrows():
        if row.get('_dataset') == 'us':
            yr = pd.to_datetime(row.get('utc_time'), utc=True, errors='coerce')
            yr = int(yr.year) if pd.notna(yr) else 2022
            loc_df_for_row = _us_locs_for_year(yr)
        else:
            loc_df_for_row = locations
        args_list.append(
            (tuple(row), sampled_df.columns, loc_df_for_row, cities,
             PROD_PARAMS, OUTPUT_FIGURE_DIR)
        )

    max_workers = os.cpu_count()
    print(f"Generating {len(args_list)} plots in parallel with up to {max_workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(tqdm(pool.map(process_and_save_row, args_list), total=len(args_list)))

    ok   = [r for r in results if r is not None]
    fail = len(results) - len(ok)
    print(f"\nDone. {len(ok)} plots saved, {fail} failed.")
    print(f"Output: {OUTPUT_FIGURE_DIR}")
