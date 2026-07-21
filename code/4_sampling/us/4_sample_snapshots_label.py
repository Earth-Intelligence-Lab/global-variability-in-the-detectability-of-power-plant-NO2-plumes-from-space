import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from pyproj import Geod
from scipy.ndimage import gaussian_filter
import contextily as ctx
from haversine import haversine, Unit
from sklearn.neighbors import BallTree
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from plotting import process_zoomed_data
import concurrent.futures
from tqdm import tqdm
import netCDF4 as nc
from math import radians, log10, sqrt

plt.rcdefaults()

# --- Configuration ---
LOCATION_DATA_PATH = '/net/fs06/d3/rzhuang/TROPOMI_US/data/top-CAMPD-sources-2019-2024.csv'
SNAPSHOT_DATA_PATH = '/net/fs06/d3/rzhuang/TROPOMI_US/data/Run_20250623_203825/updated_tropomi_hourly_emissions_full_variables.csv'
CITY_DATA_PATH = '/net/fs06/d3/rzhuang/TROPOMI_world/data/worldcities.csv'
ANNUAL_EMISSIONS_PATH = '/net/fs06/d3/rzhuang/TROPOMI_US/data/annual-emissions-facility-aggregation-2019-2024.csv'
OUTPUT_FIGURE_DIR = '/net/fs06/d3/rzhuang/TROPOMI_US/code/figure_parallel'

os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

# --- City Data Loading ---
min_city_population = 50000
CITY_POPULATION_COL = 'population'
CITY_LAT_COL = 'latitude'
CITY_LON_COL = 'longitude'
CITY_NAME_COL = 'name'

try:
    world_cities_df = pd.read_csv(CITY_DATA_PATH)
    print(f"Successfully loaded {len(world_cities_df)} total entries from {CITY_DATA_PATH}")
    if CITY_LAT_COL not in world_cities_df.columns or CITY_LON_COL not in world_cities_df.columns:
        raise KeyError(f"'{CITY_LAT_COL}' or '{CITY_LON_COL}' column not found.")
    world_cities_df[CITY_LAT_COL] = pd.to_numeric(world_cities_df[CITY_LAT_COL], errors='coerce')
    world_cities_df[CITY_LON_COL] = pd.to_numeric(world_cities_df[CITY_LON_COL], errors='coerce')
    initial_count = len(world_cities_df)
    world_cities_df.dropna(subset=[CITY_LAT_COL, CITY_LON_COL], inplace=True)
    if len(world_cities_df) < initial_count:
        print(f"Dropped {initial_count - len(world_cities_df)} rows with invalid coordinates.")
    if CITY_POPULATION_COL in world_cities_df.columns:
        world_cities_df[CITY_POPULATION_COL] = pd.to_numeric(world_cities_df[CITY_POPULATION_COL], errors='coerce')
        world_cities_df.dropna(subset=[CITY_POPULATION_COL], inplace=True)
        initial_count_before_pop_filter = len(world_cities_df)
        world_cities_df = world_cities_df[world_cities_df[CITY_POPULATION_COL] >= min_city_population].copy()
        print(f"Filtered cities by population >= {min_city_population}. Kept {len(world_cities_df)} out of {initial_count_before_pop_filter} cities.")
    else:
        print(f"Warning: Population column '{CITY_POPULATION_COL}' not found. Skipping population filter.")
    if CITY_NAME_COL not in world_cities_df.columns:
        print(f"Warning: City name column '{CITY_NAME_COL}' not found.")
    if world_cities_df.empty:
        print("Warning: No cities remaining after filtering.")
    else:
        print("\nFirst 5 rows of filtered city data:\n", world_cities_df.head())
except FileNotFoundError:
    print(f"Error: City data file not found at {CITY_DATA_PATH}. City locations cannot be processed.")
    world_cities_df = pd.DataFrame(columns=[CITY_NAME_COL, CITY_LAT_COL, CITY_LON_COL])
except KeyError as e:
    print(f"Error: Column {e} not found in {CITY_DATA_PATH}. Please check column name settings.")
    world_cities_df = pd.DataFrame(columns=[CITY_NAME_COL, CITY_LAT_COL, CITY_LON_COL])
except Exception as e:
    print(f"An unexpected error occurred while loading or processing city data: {e}.")
    world_cities_df = pd.DataFrame(columns=[CITY_NAME_COL, CITY_LAT_COL, CITY_LON_COL])

# --- Load Location Data ---
print(f"Loading location data from: {LOCATION_DATA_PATH}")
try:
    locations_df = pd.read_csv(LOCATION_DATA_PATH)

    rename = {
        'Facility ID': 'ID',
        'Facility Name': 'facility_name',
        'NOx Mass (short tons)': 'nox_emis_ty',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'State': 'State',
        'country': 'country',
        'ISO3': 'ISO3',
    }
    locations_df = locations_df.rename(columns={k: v for k, v in rename.items() if k in locations_df.columns})

    # Provide lowercase aliases expected by new visualization
    if 'Latitude' in locations_df.columns:
        locations_df['latitude'] = pd.to_numeric(locations_df['Latitude'], errors='coerce')
    if 'Longitude' in locations_df.columns:
        locations_df['longitude'] = pd.to_numeric(locations_df['Longitude'], errors='coerce')
    if 'nox_emis_ty' in locations_df.columns:
        locations_df['nox_emis_ty'] = pd.to_numeric(locations_df['nox_emis_ty'], errors='coerce')

    # Basic checks
    if 'ID' not in locations_df.columns:
        raise ValueError("Required column 'ID' missing in locations_df")
    if 'latitude' not in locations_df.columns or 'longitude' not in locations_df.columns:
        raise ValueError("Required columns 'latitude'/'longitude' missing in locations_df")

    print(f"Location data loaded. Shape: {locations_df.shape}")
except Exception as e:
    print(f"Error loading location data: {e}")
    raise SystemExit(1)

# --- Load Snapshot Data ---
print(f"Loading snapshot data from: {SNAPSHOT_DATA_PATH}")
try:
    emission_detections_df = pd.read_csv(SNAPSHOT_DATA_PATH, low_memory=False)
    print(f"Snapshot data loaded. Shape: {emission_detections_df.shape}")
    if 'location' not in emission_detections_df.columns:
        raise ValueError("Required column 'location' not found")
    emission_detections_df['wind_u'] = pd.to_numeric(emission_detections_df['wind_u'], errors='coerce')
    emission_detections_df['wind_v'] = pd.to_numeric(emission_detections_df['wind_v'], errors='coerce')
    # helpful aliases that new viz uses
    if 'latitude' not in emission_detections_df.columns and 'Latitude' in emission_detections_df.columns:
        emission_detections_df['latitude'] = pd.to_numeric(emission_detections_df['Latitude'], errors='coerce')
    if 'longitude' not in emission_detections_df.columns and 'Longitude' in emission_detections_df.columns:
        emission_detections_df['longitude'] = pd.to_numeric(emission_detections_df['Longitude'], errors='coerce')
except Exception as e:
    print(f"Error loading or processing data: {e}")
    raise SystemExit(1)

# Ensure emission column exists in snapshots (for title/stats)
if 'annual_nox_emission' not in emission_detections_df.columns:
    if 'NOx Mass (short tons)' in emission_detections_df.columns:
        emission_detections_df['annual_nox_emission'] = pd.to_numeric(
            emission_detections_df['NOx Mass (short tons)'], errors='coerce'
        )

# --- Optional: Load annual emissions (kept to minimize changes) ---
try:
    emissions_df_annual = pd.read_csv(ANNUAL_EMISSIONS_PATH)
    print(f"Annual emissions loaded: {emissions_df_annual.shape}")
except Exception as e:
    print(f"Error loading annual emissions: {e}")
    emissions_df_annual = pd.DataFrame()

# --- Sampling Function (unchanged behavior; emission-only stratification) ---
def sample_emission_snapshots(df: pd.DataFrame,
                              n_samples: int = 200,
                              emission_col: str = 'annual_nox_emission',
                              n_emission_bins: int = 5,
                              random_state: int = 42) -> pd.DataFrame:
    print(f"Starting sampling process for {n_samples} snapshots...")
    print(f"Original DataFrame shape: {df.shape}")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if emission_col not in df.columns:
        for cand in ['annual_nox_ty', 'annual_NOx_tpy', 'annual_nox_tons',
                     'NOx Mass (short tons)', 'annual_nox_ton']:
            if cand in df.columns:
                df = df.copy()
                df[emission_col] = pd.to_numeric(df[cand], errors='coerce')
                break
    if emission_col not in df.columns:
        raise ValueError(f"Emission column '{emission_col}' not found.")
    df_processed = df.copy()
    before = len(df_processed)
    df_processed.dropna(subset=[emission_col], inplace=True)
    if len(df_processed) < before:
        print(f"Dropped {before - len(df_processed)} rows with missing '{emission_col}'.")
    if df_processed.empty:
        print("Error: DataFrame empty after dropping NaNs.")
        return pd.DataFrame(columns=df.columns)
    if n_samples > len(df_processed):
        print(f"Warning: n_samples > available data. Sampling all {len(df_processed)}.")
        n_samples = len(df_processed)
    if n_samples <= 0:
        print("Warning: Adjusted n_samples is 0.")
        return pd.DataFrame(columns=df.columns)
    try:
        df_processed['emission_bin'] = pd.qcut(
            pd.to_numeric(df_processed[emission_col], errors='coerce'),
            q=n_emission_bins, labels=False, duplicates='drop'
        )
    except ValueError as e:
        print(f"Warning: qcut failed ({e}). Falling back to rank bins.")
        ranks = df_processed[emission_col].rank(method='first')
        df_processed['emission_bin'] = pd.qcut(
            ranks, q=min(n_emission_bins, max(1, int(ranks.nunique()))),
            labels=False, duplicates='drop'
        )
    try:
        _, sampled_df = train_test_split(
            df_processed, test_size=n_samples,
            stratify=df_processed['emission_bin'],
            random_state=random_state
        )
        print(f"Successfully sampled {len(sampled_df)} snapshots (emission-stratified).")
    except ValueError as e:
        print(f"Stratified sampling failed ({e}). Using simple random sampling.")
        sampled_df = df_processed.sample(n=n_samples, random_state=random_state)
    return sampled_df.drop(columns=['emission_bin'], errors='ignore')

# --- Sample ---
sampled_df = sample_emission_snapshots(
    emission_detections_df,
    n_samples=500,
    emission_col='annual_nox_emission',
    n_emission_bins=5,
    random_state=345
)
print(f"Sampled {len(sampled_df)} rows.")
sampled_df.to_csv(os.path.join(OUTPUT_FIGURE_DIR, 'sampled_emission_snapshots.csv'), index=False)

# === PARALLEL PROCESSING SECTION ===

# --- Define the Wrapper Function for a Single Row ---
# --- Parallel wrapper (minimal changes: now passes global_locations_df & cities_df) ---
def process_and_save_row(args):
    row_tuple, row_index, locations_df_global, world_cities_df_global, plot_params, output_dir = args
    row = pd.Series(row_tuple, index=row_index)
    fig = None
    location_id = row.get('location', 'UnknownLocation')
    iso_code = row.get('country', row.get('ISO3', 'UnknownISO'))
    time = row.get('utc_time', 'UnknownTime')
    save_path = os.path.join(output_dir, f"sampled_location_{location_id}_{iso_code}_{time}.png")
    try:
        loc_df_copy  = locations_df_global.copy()    if locations_df_global  is not None else pd.DataFrame()
        city_df_copy = world_cities_df_global.copy() if world_cities_df_global is not None else pd.DataFrame()
        fig = process_zoomed_data(
            row=row,
            global_locations_df=loc_df_copy,
            cities_df=city_df_copy,
            **plot_params
        )
        if fig:
            fig.savefig(save_path, dpi=plot_params.get('plot_dpi', 200))
            plt.close(fig)
            return save_path
        else:
            print(f"Skipped saving plot for {location_id} ({iso_code}): process_zoomed_data returned None.")
            return None
    except Exception as e:
        print(f"Error processing row for {location_id} ({iso_code}): {type(e).__name__} - {e}")
        if fig:
            plt.close(fig)
        return None

# --- Main execution ---
if __name__ == "__main__":
    # Locations subset (message text kept)
    if 'ID' in locations_df.columns:
        locations_df_subset = locations_df.copy()  # dataset is small; keep all
        print(f"\nUsing subset of locations_df (first 6000 rows) for interference checks: Shape={locations_df_subset.shape}")
    else:
        print("\nWarning: 'ID' column not found in locations_df. Using full locations_df for interference checks, which might be slow.")
        locations_df_subset = locations_df.copy()

    plotting_parameters = {
        'zoom_radius_km': 100,
        'threshold_factor': 2.5,
        'threshold_abs_min': 5e-6,
        'max_distance_km': 30,
        'close_distance_km': 3,
        'max_angle_diff': 35,
        'max_angle_diff_mask': 40,
        'close_distance_km_mask': 5,
        'flagged_area': 50.0,
        'sigma': 10,
        'plot_dpi': 200,
        'interf_max_distance_km': 150,
        'interf_city_pop_thresh': 200000,
        'interf_plant_emis_thresh': 1,
        'threshold_radius_km': 25,
        'background_mode': 'directional',
        'upwind_angle_tolerance': 60,
        'background_dist_min_km': 10,
        'background_dist_max_km': 75,
        'plot_interference_zones': True
    }

    args_list = [
        (
            tuple(row),
            sampled_df.columns,
            locations_df_subset,
            world_cities_df,
            plotting_parameters,
            OUTPUT_FIGURE_DIR
        )
        for _, row in sampled_df.iterrows()
    ]

    max_workers = os.cpu_count()
    print(f"\nStarting parallel plot generation using up to {max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_and_save_row, args_list), total=len(args_list)))

    successful_plots = [res for res in results if res is not None]
    failed_count = len(results) - len(successful_plots)

    print(f"\n--- Parallel Processing Complete ---")
    print(f"Successfully generated {len(successful_plots)} plots.")
    if failed_count > 0:
        print(f"Failed to generate plots for {failed_count} locations (see messages above).")
    print("\nScript finished.")