import pandas as pd
import numpy as np
import netCDF4 as nc
from sklearn.neighbors import BallTree
# Assuming plot_pipeline.py contains your functions
from plotting import process_zoomed_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Need pyplot for closing figures

# --- Parallel Processing Imports ---
import concurrent.futures
import os
from tqdm import tqdm # Optional: for progress bar

# --- Configuration ---
LOCATION_DATA_PATH = '/net/fs06/d3/rzhuang/TROPOMI_world/data/power_plant_location/power_plants_with_combined_nearby_stats.csv'
SNAPSHOT_DATA_PATH = '/net/fs06/d3/rzhuang/TROPOMI_world/data/valid_tropomi_combined_all_vars_dropna.csv'
CITY_DATA_PATH = '/net/fs06/d3/rzhuang/TROPOMI_world/data/worldcities.csv'
OUTPUT_FIGURE_DIR = '/net/fs06/d3/rzhuang/TROPOMI_world/code/figure_parallel_US' # <<<--- Specify directory for output figures

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

# --- City Data Loading (Keep your existing robust loading code) ---
min_city_population = 50000
CITY_POPULATION_COL = 'population'
CITY_LAT_COL = 'latitude'
CITY_LON_COL = 'longitude'
CITY_NAME_COL = 'name' # Or 'city_ascii'

try:
    world_cities_df = pd.read_csv(CITY_DATA_PATH)
    print(f"Successfully loaded {len(world_cities_df)} total entries from {CITY_DATA_PATH}")
    # Apply cleaning and filtering as in your original code...
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
    else: print(f"Warning: Population column '{CITY_POPULATION_COL}' not found. Skipping population filter.")
    if CITY_NAME_COL not in world_cities_df.columns: print(f"Warning: City name column '{CITY_NAME_COL}' not found.")
    if world_cities_df.empty: print("Warning: No cities remaining after filtering.")
    else: print("\nFirst 5 rows of filtered city data:\n", world_cities_df.head())

except FileNotFoundError:
    print(f"Error: City data file not found at {CITY_DATA_PATH}. City locations cannot be processed.")
    world_cities_df = pd.DataFrame(columns=[CITY_NAME_COL, CITY_LAT_COL, CITY_LON_COL])
except KeyError as e:
    print(f"Error: Column {e} not found in {CITY_DATA_PATH}. Please check column name settings.")
    world_cities_df = pd.DataFrame(columns=[CITY_NAME_COL, CITY_LAT_COL, CITY_LON_COL])
except Exception as e:
    print(f"An unexpected error occurred while loading or processing city data: {e}.")
    world_cities_df = pd.DataFrame(columns=[CITY_NAME_COL, CITY_LAT_COL, CITY_LON_COL])

# --- Load Location and Snapshot Data (Keep your existing loading & merging code) ---
print(f"Loading location data from: {LOCATION_DATA_PATH}")
try:
    locations_df = pd.read_csv(LOCATION_DATA_PATH)
    print(f"Location data loaded. Shape: {locations_df.shape}")
    if 'ID' not in locations_df.columns or 'nox_emis_ty' not in locations_df.columns:
        raise ValueError("Required columns 'ID' or 'nox_emis_ty' not found")
except Exception as e: print(f"Error loading location data: {e}"); exit()

print(f"Loading snapshot data from: {SNAPSHOT_DATA_PATH}")
try:
    emission_detections_df = pd.read_csv(SNAPSHOT_DATA_PATH, low_memory=False)
    print(f"Snapshot data loaded. Shape: {emission_detections_df.shape}")
    if 'location' not in emission_detections_df.columns:
        raise ValueError("Required column 'location' not found")

    # Convert wind columns
    emission_detections_df['wind_u'] = pd.to_numeric(emission_detections_df['wind_u'], errors='coerce')
    emission_detections_df['wind_v'] = pd.to_numeric(emission_detections_df['wind_v'], errors='coerce')
    print("Wind column conversion complete.")

except Exception as e: print(f"Error loading or processing data: {e}"); exit()


# --- Sampling Function (Keep as is) ---
def stratified_sample(
    df: pd.DataFrame,
    n_samples: int,
    country_col: str = 'country',
    emission_col: str = 'annual_nox_emission',
    n_emission_bins: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """ Stratified sampling of USA snapshots by emission bins. """

    # --- Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    for col in (country_col, emission_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if n_samples <= 0:
        print("Warning: n_samples <= 0.")
        return pd.DataFrame(columns=df.columns)

    # --- Filter & clean ---
    df_proc = df.copy()
    # 1) drop any rows missing the key cols
    df_proc.dropna(subset=[country_col, emission_col], inplace=True)
    # 2) keep only USA Top100
    
    # load the full power_plants DF
    power_plants = pd.read_csv('/net/fs06/d3/rzhuang/TROPOMI_world/data/power_plant_location/power_plants_with_combined_nearby_stats.csv')

    # filter: no other plants within 20 km, but at least one city within 20 km
    mask = (
        (power_plants['nearby_plants_count_20km'] == 0) &
        (power_plants['nearby_cities_count_20km'] == 0) &
        (power_plants['ISO3'] == 'USA')
    )

    # apply filter and take the first 100 (or, if you have a ranking metric, sort before .head)
    Top100US = power_plants.loc[mask].sort_values(by='nox_emis_ty', ascending=False).head(100)
    
    df = df[df['location'].isin(Top100US['ID'])]

    if df_proc.empty:
        print("Error: no data after filtering for USA and dropping NaNs.")
        return pd.DataFrame(columns=df.columns)

    available = len(df_proc)
    if n_samples > available:
        print(f"Warning: requested {n_samples} > available {available}, sampling all.")
        n_samples = available

    # --- Bin into strata ---
    try:
        df_proc['emission_bin'] = pd.qcut(
            df_proc[emission_col],
            q=n_emission_bins,
            labels=False,
            duplicates='drop'
        )
    except ValueError:
        print("Warning: qcut failed, falling back to dense rank.")
        df_proc['emission_bin'] = (
            df_proc[emission_col]
            .rank(method='dense')
            .astype(int)
        )

    # --- Stratified sample ---
    try:
        if available == n_samples:
            sampled = df_proc.copy()
        else:
            _, sampled = train_test_split(
                df_proc,
                test_size=n_samples,
                stratify=df_proc['emission_bin'],
                random_state=random_state
            )
        print(f"Stratified sampled {len(sampled)} rows.")
    except ValueError as e:
        print(f"Warning: stratified split failed ({e}), falling back to random sample.")
        sampled = df_proc.sample(n=n_samples, random_state=random_state)
        print(f"Randomly sampled {len(sampled)} rows.")

    # --- Cleanup & return ---
    return sampled.drop(columns=['emission_bin'], errors='ignore')

# --- Call the sampling function ---
try:
    sampled_df = stratified_sample(
        emission_detections_df,
        n_samples=100, # Or however many you need
        country_col='country',
        emission_col='annual_nox_emission',
        n_emission_bins=5,
        random_state=345
    )
    if not sampled_df.empty:
        print("\n--- Sample Verification ---")
        print(f"Number of samples obtained: {len(sampled_df)}")
        # Add verification prints if desired...
        print(sampled_df.head())
        sampled_df.to_csv(os.path.join(OUTPUT_FIGURE_DIR, 'sampled_emission_snapshots.csv'), index=False)
    else:
        print("Sampling resulted in an empty DataFrame. Exiting.")
        exit()

except (ValueError, TypeError) as e:
    print(f"\nAn error occurred during sampling: {e}")
    exit()
    
# sampled_df = pd.read_csv("/net/fs06/d3/rzhuang/TROPOMI_world/data/sampled_emission_snapshots.csv")

# === PARALLEL PROCESSING SECTION ===

# --- Define the Wrapper Function for a Single Row ---
def process_and_save_row(args):
    """
    Processes a single row from the sampled DataFrame and saves the plot.
    Designed to be used with pool.map or executor.map.

    Args:
        args (tuple): A tuple containing:
            - row_tuple (tuple): A tuple representing the row data (faster than Series for multiprocessing).
            - row_index (pd.Index): The index of the row (used for accessing Series data if needed).
            - locations_df_global (pd.DataFrame): The global locations data.
            - world_cities_df_global (pd.DataFrame): The global cities data.
            - plot_params (dict): Dictionary containing plotting parameters.
            - output_dir (str): Directory to save the figures.

    Returns:
        str or None: The path to the saved figure if successful, None otherwise.
    """
    row_tuple, row_index, locations_df_global, world_cities_df_global, plot_params, output_dir = args
    # Reconstruct Series from tuple and index (more robust for complex data types)
    row = pd.Series(row_tuple, index=row_index)
    fig = None # Initialize fig to None
    location_id = row.get('location', 'UnknownLocation')
    iso_code = row.get('country', 'UnknownISO')
    time = row.get('utc_time', 'UnknownTime')
    save_path = os.path.join(output_dir, f"sampled_location_{location_id}_{iso_code}_{time}.png")

    try:
        # print(f"Processing: {location_id} ({iso_code})") # Too verbose for parallel

        # Ensure necessary dataframes are valid before passing
        loc_df_copy = locations_df_global.copy() if locations_df_global is not None else pd.DataFrame()
        city_df_copy = world_cities_df_global.copy() if world_cities_df_global is not None else pd.DataFrame()

        # Call the main processing function
        fig = process_zoomed_data(
            row=row,
            global_locations_df=loc_df_copy, # Pass the COPIES
            cities_df=city_df_copy,
            **plot_params # Unpack plotting parameters
        )

        # Save the figure if generated successfully
        if fig:
            fig.savefig(save_path, dpi=plot_params.get('plot_dpi', 200)) # Use plot_dpi from params
            plt.close(fig) # IMPORTANT: Close the figure to release memory
            return save_path # Return path on success
        else:
            # process_zoomed_data might return None if it encounters errors or skips plotting
            print(f"Skipped saving plot for {location_id} ({iso_code}): process_zoomed_data returned None.")
            return None

    except Exception as e:
        print(f"Error processing row for {location_id} ({iso_code}): {type(e).__name__} - {e}")
        import traceback
        # traceback.print_exc() # Optionally print full traceback for debugging
        if fig:
            plt.close(fig) # Ensure figure is closed even if saving fails
        return None # Return None on failure

# --- Main Execution Block ---
if __name__ == "__main__":
    # Use only the first ~6000 rows of locations_df for interference checking
    # (as per the original loop's slicing) - DO THIS ONCE before preparing args
    # Make sure 'ID' column exists for the slicing/lookup logic in process_zoomed_data
    if 'ID' in locations_df.columns:
        locations_df_subset = locations_df.iloc[:6000].copy()
        print(f"\nUsing subset of locations_df (first 6000 rows) for interference checks: Shape={locations_df_subset.shape}")
    else:
        print("\nWarning: 'ID' column not found in locations_df. Using full locations_df for interference checks, which might be slow.")
        locations_df_subset = locations_df.copy() # Use full df if ID is missing

    # --- Define Plotting Parameters (Consistent for all processes) ---
    plotting_parameters = {
        'zoom_radius_km': 100,
        'threshold_factor': 2.5,
        'threshold': 5e-6,
        'max_distance_km': 30,
        'close_distance_km': 3,
        'max_angle_diff': 35,
        'max_angle_diff_mask': 40,
        'close_distance_km_mask': 5,
        'flagged_area': 50.0,
        'sigma': 10,
        'plot_dpi': 200, # Consistent DPI
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

    # --- Prepare Arguments for Parallel Processing ---
    # Convert rows to tuples for potentially faster pickling/transfer
    # Pass the relevant dataframes and parameters needed by the worker function
    args_list = [
        (
            tuple(row),               # Row data as a tuple
            sampled_df.columns,       # Row index/columns to reconstruct Series
            locations_df_subset,      # The subsetted locations dataframe (read-only for workers)
            world_cities_df,          # The cities dataframe (read-only for workers)
            plotting_parameters,      # Dictionary of plot params (read-only)
            OUTPUT_FIGURE_DIR         # Output directory string (read-only)
        )
        for _, row in sampled_df.iterrows() # Iterate through the sampled dataframe
    ]

    # --- Determine Number of Workers ---
    # Start with number of CPU cores, but maybe use fewer if memory constrained
    max_workers = os.cpu_count()
    print(f"\nStarting parallel plot generation using up to {max_workers} workers...")

    # --- Execute in Parallel ---
    results = []
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to wrap the executor.map call for a progress bar
        # The map function applies process_and_save_row to each item in args_list
        results = list(tqdm(executor.map(process_and_save_row, args_list), total=len(args_list)))

    # --- Report Results ---
    successful_plots = [res for res in results if res is not None]
    failed_count = len(results) - len(successful_plots)

    print(f"\n--- Parallel Processing Complete ---")
    print(f"Successfully generated {len(successful_plots)} plots.")
    if failed_count > 0:
        print(f"Failed to generate plots for {failed_count} locations (see error messages above).")
    # print("Generated plots:", successful_plots) # Optionally print list of saved files

    print("\nScript finished.")