import pandas as pd
import numpy as np
from scipy.spatial import cKDTree # For efficient spatial queries
from math import radians, cos, sin, asin, sqrt # For Haversine
from joblib import Parallel, delayed
# from multiprocessing import cpu_count # cpu_count is not used directly here
import time
import io # To load sample plant data if needed
import os # To load city data

# --- 1. Haversine Distance Function ---
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees). More robust than the library
    version for pair-wise calculation inside the loop.
    """
    # Check for NaN inputs
    if any(pd.isna(x) for x in [lon1, lat1, lon2, lat2]):
        # print(f"DEBUG: Haversine received NaN: lon1={lon1}, lat1={lat1}, lon2={lon2}, lat2={lat2}") # Optional: Uncomment for very detailed debug
        return np.inf # Return infinity for invalid coordinates to exclude them

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers.
    return c * r

# --- 2. City Data Loading Function (Copied from previous example) ---
def load_simplemaps_cities(filepath):
    """Loads city data from SimpleMaps CSV, cleans, and prepares for use."""
    print(f"Attempting to load SimpleMaps city data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"DEBUG: City file not found: {filepath}") # DEBUG
        return None
    cities_df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)

    cities_df['latitude'] = pd.to_numeric(cities_df['latitude'], errors='coerce')
    cities_df['longitude'] = pd.to_numeric(cities_df['longitude'], errors='coerce')
    cities_df['population'] = pd.to_numeric(cities_df['population'], errors='coerce')
    lat_nan_count = cities_df['latitude'].isna().sum()
    lon_nan_count = cities_df['longitude'].isna().sum()
    
    if lat_nan_count > 0 or lon_nan_count > 0:
        print(f"DEBUG: Coerced NaNs in coords - latitude: {lat_nan_count}, longitude: {lon_nan_count}") #DEBUG
        rows_before_coord_drop = len(cities_df)
        
        cities_df.dropna(subset=['latitude', 'longitude'], inplace=True) # Need coords
        rows_after_coord_drop = len(cities_df)
        print(f"DEBUG: Dropped {rows_before_coord_drop - rows_after_coord_drop} cities due to NaN coordinates.") # DEBUG

    # --- DEBUG: Check population values again after potential drops ---
    if 'population' in cities_df.columns:
        print(f"DEBUG: Population stats after coord drop (before filtering): \n{cities_df['population'].describe()}")
    # --- END DEBUG ---
    print(f"Processed city data: {len(cities_df):,} cities with valid coordinates.")
    return cities_df

# --- 3. Combined Worker Function ---
def calculate_combined_nearby_stats(
    reference_plant_row,
    all_plants_df,
    city_tree, # Pass the k-d tree object
    cities_filtered_df, # Pass the filtered city DataFrame
    plant_lat_col, plant_lon_col, emission_col,
    city_lat_col, city_lon_col, city_pop_col, # City column names
    plant_radii_km, # Radii for plant-plant analysis
    city_radii_km   # Radii for plant-city analysis
    ):
    """
    Calculates nearby plant AND nearby city statistics for a single reference plant.

    Args:
        reference_plant_row (pd.Series): Row of the reference plant.
        all_plants_df (pd.DataFrame): DataFrame of all power plants.
        city_tree (scipy.spatial.cKDTree or None): Pre-built k-d tree for cities.
        cities_filtered_df (pd.DataFrame or None): Filtered DataFrame of cities.
        plant_lat_col, plant_lon_col, emission_col: Plant column names.
        city_lat_col, city_lon_col, city_pop_col: City column names.
        plant_radii_km (list): Radii for plant-plant calculations.
        city_radii_km (list): Radii for plant-city calculations.

    Returns:
        pd.Series: Combined statistics for the reference plant.
    """
    results = {}
    earth_radius_km = 6371 # For tree query conversion
    ref_index = reference_plant_row.name # Get index early for debug messages

    # --- Initialize results with NaN ---
    for radius in plant_radii_km:
        results[f'nearby_plants_count_{radius}km'] = np.nan
        results[f'total_emission_{radius}km'] = np.nan
        results[f'percentage_emission_{radius}km'] = np.nan
    for radius in city_radii_km:
        results[f'nearby_cities_count_{radius}km'] = np.nan
        results[f'nearby_cities_pop_{radius}km'] = np.nan

    # Check for valid coordinates in the reference plant row
    ref_lat = reference_plant_row[plant_lat_col]
    ref_lon = reference_plant_row[plant_lon_col]
    if pd.isna(ref_lat) or pd.isna(ref_lon):
        # print(f"DEBUG (Worker {ref_index}): Skipping plant due to NaN coordinates ({ref_lat}, {ref_lon})", flush=True) # Optional Debug
        return pd.Series(results) # Return NaNs if ref coords invalid
    ref_emission = reference_plant_row[emission_col]

    # --- Part A: Nearby Plant Calculations ---
    # (Assuming this part works, no major debug changes needed here unless plant stats are also wrong)
    valid_target_plants = all_plants_df.dropna(subset=[plant_lat_col, plant_lon_col])
    target_plant_coords = list(zip(valid_target_plants[plant_lat_col], valid_target_plants[plant_lon_col]))
    plant_distances = pd.Series(
        [haversine(ref_lon, ref_lat, target_lon, target_lat)
         for target_lat, target_lon in target_plant_coords],
        index=valid_target_plants.index
    )
    plant_distances = plant_distances.reindex(all_plants_df.index, fill_value=np.inf)

    for radius in plant_radii_km:
        within_radius_mask = plant_distances <= radius
        nearby_plants_df = all_plants_df[within_radius_mask]
        count_nearby = nearby_plants_df[nearby_plants_df.index != ref_index].shape[0]
        total_emissions_nearby = pd.to_numeric(nearby_plants_df[emission_col], errors='coerce').sum()

        if pd.isna(ref_emission): percentage = np.nan
        elif total_emissions_nearby > 0: percentage = (ref_emission / total_emissions_nearby) * 100
        elif total_emissions_nearby == 0 and ref_emission == 0: percentage = 100.0
        elif total_emissions_nearby == 0 and ref_emission > 0: percentage = np.inf
        else: percentage = np.nan

        results[f'nearby_plants_count_{radius}km'] = count_nearby
        results[f'total_emission_{radius}km'] = total_emissions_nearby
        results[f'percentage_emission_{radius}km'] = percentage

    # --- Part B: Nearby City Calculations (Using k-d tree) ---

    # === DEBUG CHECK: Will city calculation block be skipped? ===
    city_calc_possible = city_tree is not None and cities_filtered_df is not None and not cities_filtered_df.empty
    if not city_calc_possible:
        print(f"DEBUG (Worker {ref_index}): Skipping city calculation. Reason: "
              f"city_tree is None: {city_tree is None}, "
              f"cities_filtered_df is None: {cities_filtered_df is None}, "
              f"cities_filtered_df is empty: {cities_filtered_df.empty if cities_filtered_df is not None else 'N/A'}", flush=True)
    # === END DEBUG CHECK ===

    if city_calc_possible:
        # print(f"DEBUG (Worker {ref_index}): Entering city calculation block.", flush=True) # Optional Debug
        plant_lat_rad = radians(ref_lat)
        plant_lon_rad = radians(ref_lon)
        max_city_radius_rad = max(city_radii_km) / earth_radius_km

        try:
            potential_indices = city_tree.query_ball_point([plant_lat_rad, plant_lon_rad], r=max_city_radius_rad)
            # print(f"DEBUG (Worker {ref_index}): Found {len(potential_indices)} potential cities within {max(city_radii_km)}km radius.", flush=True) # Optional Debug

            if len(potential_indices) > 0:
                nearby_potential_cities = cities_filtered_df.iloc[potential_indices]

                city_distances = [
                    haversine(ref_lon, ref_lat, city_lon, city_lat)
                    for city_lon, city_lat in zip(nearby_potential_cities[city_lon_col], nearby_potential_cities[city_lat_col])
                ]
                nearby_potential_cities_with_dist = nearby_potential_cities.copy()
                nearby_potential_cities_with_dist['distance_km'] = city_distances

                for r_km in city_radii_km:
                    cities_within_radius = nearby_potential_cities_with_dist[
                        nearby_potential_cities_with_dist['distance_km'] <= r_km
                    ]
                    count = len(cities_within_radius)
                    total_pop = pd.to_numeric(cities_within_radius[city_pop_col], errors='coerce').fillna(0).sum()
                    results[f'nearby_cities_count_{r_km}km'] = count
                    results[f'nearby_cities_pop_{r_km}km'] = int(total_pop)
            else:
                # --- Explicitly set to 0 if k-d tree finds nothing nearby ---
                print(f"DEBUG (Worker {ref_index}): No potential cities found by k-d tree within {max(city_radii_km)}km. Setting counts/pop to 0.", flush=True) # Optional Debug
                for r_km in city_radii_km:
                    results[f'nearby_cities_count_{r_km}km'] = 0
                    results[f'nearby_cities_pop_{r_km}km'] = 0

        except KeyError as e:
             print(f"DEBUG (Worker {ref_index}): KeyError during city calculation! Missing column: {e}. Check column names passed vs. filtered DataFrame.", flush=True)
             # Results remain NaN
        except Exception as e:
            print(f"Warning/DEBUG: Error during city calculation for plant index {ref_index}: {e}. Type: {type(e)}", flush=True) # Added Type
            # Results for cities will remain NaN

    # --- Return combined results ---
    # print(f"DEBUG (Worker {ref_index}): Returning results. City keys sample: {results.get(f'nearby_cities_count_{city_radii_km[0]}km')}", flush=True) # Optional Debug
    return pd.Series(results)

# --- 4. Main Script Logic ---

# --- Configuration ---
power_plant_file = '/net/fs06/d3/rzhuang/TROPOMI_US/data/top-CAMPD-sources-2019-2024.csv'
city_file = '/net/fs06/d3/rzhuang/TROPOMI_US/data/worldcities.csv' # Path to your city file

# Column names
plant_lat_col = 'Latitude'
plant_lon_col = 'Longitude'
emission_col = 'CO2 Mass (short tons)' # Adjust as needed
city_lat_col = 'latitude' # After renaming in load_simplemaps_cities
city_lon_col = 'longitude' # After renaming in load_simplemaps_cities
city_pop_col = 'population'

# Radii
plant_radii_km = [20, 50, 100]
city_radii_km = [20, 50, 100, 200]
min_city_population = 50000 # Filter cities by this population <<-- Check if this is too high?

# Parallelism
n_jobs = 48 # Adjust as needed

# --- Load Power Plant Data ---
print(f"Loading power plants from: {power_plant_file}")
try:
    locations_df = pd.read_csv(power_plant_file)
except FileNotFoundError:
    print(f"Error: Power plant file not found: {power_plant_file}")
    exit()
except Exception as e:
    print(f"Error reading power plant file: {e}")
    exit()

# --- Validate Power Plant Data ---
required_plant_cols = [plant_lat_col, plant_lon_col, emission_col]
missing_plant_cols = [col for col in required_plant_cols if col not in locations_df.columns]
if missing_plant_cols:
    print(f"Error: Missing required power plant columns: {missing_plant_cols}")
    exit()

# Convert and clean power plant data
try:
    locations_df[plant_lat_col] = pd.to_numeric(locations_df[plant_lat_col], errors='coerce')
    locations_df[plant_lon_col] = pd.to_numeric(locations_df[plant_lon_col], errors='coerce')
    locations_df[emission_col] = pd.to_numeric(locations_df[emission_col], errors='coerce')
except Exception as e:
    print(f"Error during power plant numeric conversion: {e}")
    exit()

initial_rows = len(locations_df)
locations_df = locations_df.dropna(subset=[plant_lat_col, plant_lon_col]) # Need coords for analysis
rows_dropped = initial_rows - len(locations_df)
if rows_dropped > 0:
    print(f"Dropped {rows_dropped} power plants due to missing coordinates.")

locations_df = locations_df.reset_index(drop=True)

if locations_df.empty:
    print("Error: Power plant DataFrame is empty after cleaning.")
    exit()

print(f"Processing {len(locations_df)} power plants with valid coordinates.")


# --- Load and Prepare City Data ---
print("\n--- Loading and Preparing City Data ---") # Section Header
city_dataframe_sm = load_simplemaps_cities(city_file)
cities_filtered_df = None
city_tree = None

if city_dataframe_sm is not None:
    # Filter cities
    if city_pop_col in city_dataframe_sm.columns:
        print(f"DEBUG: Filtering cities by population >= {min_city_population:,} and valid coordinates.") # DEBUG
        print(f"DEBUG: Population column dtype before filter: {city_dataframe_sm[city_pop_col].dtype}") #DEBUG
        pop_nan_before_filter = city_dataframe_sm[city_pop_col].isna().sum()
        print(f"DEBUG: Population NaNs before filter: {pop_nan_before_filter}") # DEBUG

        # --- The Filter ---
        cities_filtered_df = city_dataframe_sm[
            (city_dataframe_sm[city_pop_col].fillna(-1) >= min_city_population) & # Handle NaN during comparison explicitly
            city_dataframe_sm[city_lat_col].notna() &
            city_dataframe_sm[city_lon_col].notna()
        ].copy()
        # --- End Filter ---

        print(f"Filtered cities: Found {len(cities_filtered_df)} cities meeting criteria.") # Modified print

        if not cities_filtered_df.empty:
            print(f"DEBUG: Filtered city columns: {cities_filtered_df.columns.tolist()}") # DEBUG
            print(f"DEBUG: Filtered city population stats:\n{cities_filtered_df[city_pop_col].describe()}") # DEBUG
            # Build k-d tree
            print("DEBUG: Building k-d tree...") # DEBUG
            cities_filtered_df['lat_rad'] = np.radians(cities_filtered_df[city_lat_col])
            cities_filtered_df['lon_rad'] = np.radians(cities_filtered_df[city_lon_col])
            city_coords_rad = cities_filtered_df[['lat_rad', 'lon_rad']].values
            try:
                 city_tree = cKDTree(city_coords_rad)
                 print("DEBUG: Built k-d tree for filtered cities successfully.") # DEBUG
            except Exception as e:
                 print(f"Error building k-d tree: {e}. City analysis will be skipped.")
                 print("DEBUG: k-d tree build FAILED.") # DEBUG
                 city_tree = None
                 cities_filtered_df = None # Can't use df without tree
        else:
            print("DEBUG: No cities met the filtering criteria. City analysis will be skipped.") # DEBUG
    else:
        print(f"Error/DEBUG: City population column '{city_pop_col}' not found AFTER loading/renaming. City analysis will be skipped.") # DEBUG
else:
    print("DEBUG: City data loading failed (returned None). City analysis will be skipped.") # DEBUG


# --- DEBUG: Final check before parallel execution ---
print("\n--- DEBUG: State Before Parallel Execution ---")
if cities_filtered_df is not None:
    print(f"cities_filtered_df is NOT None. Is empty? {cities_filtered_df.empty}. Shape: {cities_filtered_df.shape}")
    print(f"Columns available to worker: {cities_filtered_df.columns.tolist()}")
    print("Sample data passed to worker:")
    print(cities_filtered_df.head())
else:
    print("cities_filtered_df IS None.")
print(f"city_tree is None? {city_tree is None}")
print("------------------------------------------\n")
# --- END DEBUG ---


# --- Parallel Calculation Setup ---
print(f"Using {n_jobs} cores for parallel processing.")

# Prepare arguments for the parallel function calls
tasks = [
    delayed(calculate_combined_nearby_stats)(
        locations_df.loc[idx], # Pass the row (Series)
        locations_df,          # Pass the entire plant DataFrame
        city_tree,             # Pass the city k-d tree (or None)
        cities_filtered_df,    # Pass the filtered city DataFrame (or None)
        plant_lat_col, plant_lon_col, emission_col,
        city_lat_col, city_lon_col, city_pop_col,
        plant_radii_km,
        city_radii_km
    ) for idx in locations_df.index
]

# --- Execute in Parallel ---
start_time = time.time()
print("Starting parallel calculation...")

results_list = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(tasks)

end_time = time.time()
print(f"Parallel calculation finished in {end_time - start_time:.2f} seconds.")

# --- Combine results ---
if results_list:
    try:
        stats_results_df = pd.DataFrame(results_list, index=locations_df.index)
        # --- DEBUG: Check results from parallel processing ---
        print("\n--- DEBUG: Sample of Raw Results from Workers ---")
        print(stats_results_df.head())
        print("\n--- DEBUG: Describe Raw Results ---")
        print(stats_results_df.describe(include='all')) # Use include='all' to see NaNs
        print("--- END DEBUG ---")
        # --- END DEBUG ---

        cols_to_drop = [col for col in stats_results_df.columns if col in locations_df.columns]
        locations_df = locations_df.drop(columns=cols_to_drop)
        locations_df = pd.concat([locations_df, stats_results_df], axis=1)

        print("Calculations complete.")
    except Exception as e:
        print(f"Error combining results: {e}")
else:
    print("Parallel execution did not return results (results_list is empty or None).")


# --- Display Results (Example) ---
print("\nDataFrame with added statistics:")
plant_stat_cols = [f'{stat}_{r}km' for r in plant_radii_km for stat in ['nearby_plants_count', 'total_emission', 'percentage_emission']]
city_stat_cols = [f'{stat}_{r}km' for r in city_radii_km for stat in ['nearby_cities_count', 'nearby_cities_pop']]
columns_to_show = [plant_lat_col, plant_lon_col, emission_col] + plant_stat_cols + city_stat_cols

columns_to_show = [col for col in columns_to_show if col in locations_df.columns]

if columns_to_show: # Only print if there are columns to show
    print(locations_df[columns_to_show].head())
    print("\n--- DEBUG: Final DataFrame Description (City Stats Focus) ---")
    # Ensure city stat cols actually exist before describing
    city_stat_cols_exist = [col for col in city_stat_cols if col in locations_df.columns]
    if city_stat_cols_exist:
        print(locations_df[city_stat_cols_exist].describe())
    else:
        print("No city statistic columns were generated in the final DataFrame.")
    print("--- END DEBUG ---")
else:
    print("Warning: No columns selected for display (perhaps calculation failed entirely).")


# --- Optional: Save Results ---
output_filename = '../data/power_plants_with_combined_nearby_stats_parallel_debug.csv' # Changed name
try:
    locations_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")