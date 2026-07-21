import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from multiprocessing import Pool, cpu_count, Lock, Array
import time
import os

# --- Configuration ---
LOCATION_DATA_PATH = '/net/fs06/d3/rzhuang/TROPOMI_world/data/power_plant_location/coco2_ps_catalogue_v2.0.csv' # Original path provided
# For testing, you might want to use a local path if the network path is slow/unavailable
# Example: LOCATION_DATA_PATH = 'coco2_ps_catalogue_v2.0.csv' # Make sure this file exists locally

lat_col = 'latitude'        # Adjust if your latitude column name is different
lon_col = 'longitude'       # Adjust if your longitude column name is different
country_col = 'ISO3'        # Column name for country (ISO3 country code)
emission_cols = [
    'co2_emis_ty', 'nox_emis_ty', 'sox_emis_ty',
    'co_emis_ty', 'ch4_emis_ty'
]
distance_threshold_km = 3.0
earth_radius_km = 6371.0  # Approximate radius of Earth

# --- Data Loading ---
print(f"Loading data from: {LOCATION_DATA_PATH}")
try:
    # Check if the file exists before attempting to read
    if not os.path.exists(LOCATION_DATA_PATH):
         raise FileNotFoundError(f"Error: File not found at {LOCATION_DATA_PATH}")
    locations_df = pd.read_csv(LOCATION_DATA_PATH)
    print(f"Successfully loaded {len(locations_df)} records.")
except FileNotFoundError as e:
    print(e)
    exit() # Exit if the file cannot be loaded
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- Preprocessing & Setup ---
start_time = time.time()

# 1. Handle potential missing coordinate/emission data (optional, depends on desired behavior)
# Option A: Drop rows with any NaN in relevant columns
# locations_df.dropna(subset=[lat_col, lon_col] + emission_cols, inplace=True)
# locations_df.reset_index(drop=True, inplace=True) # Reset index after dropping

# Option B: Fill NaN in emissions with a value that won't match others (e.g., -1 if emissions are non-negative)
# for col in emission_cols:
#    locations_df[col].fillna(-1, inplace=True)
# locations_df.dropna(subset=[lat_col, lon_col], inplace=True) # Still need valid coordinates
# locations_df.reset_index(drop=True, inplace=True)

# Ensure emission columns are numeric (important for comparison)
for col in emission_cols:
    locations_df[col] = pd.to_numeric(locations_df[col], errors='coerce') # Coerce errors to NaN

# Drop rows where essential columns became NaN after coercion or were initially NaN
print(f"Original rows: {len(locations_df)}")

# Count original power plants by country
if country_col in locations_df.columns:
    original_country_counts = locations_df[country_col].value_counts().to_dict()
else:
    print(f"Warning: '{country_col}' column not found in dataset.")
    original_country_counts = {}

locations_df.dropna(subset=[lat_col, lon_col] + emission_cols, inplace=True)
locations_df.reset_index(drop=True, inplace=True) # Reset index is crucial after dropping
print(f"Rows after dropping NaN in key columns: {len(locations_df)}")

if locations_df.empty:
    print("DataFrame is empty after cleaning. No processing needed.")
    exit()

# 2. Convert coordinates to radians for BallTree (Haversine distance)
locations_df['rad_lat'] = np.radians(locations_df[lat_col])
locations_df['rad_lon'] = np.radians(locations_df[lon_col])

# 3. Create BallTree for efficient spatial queries
print("Building BallTree...")
tree = BallTree(locations_df[['rad_lat', 'rad_lon']].values, metric='haversine')
print("BallTree built.")

# 4. Calculate radius in radians
distance_threshold_rad = distance_threshold_km / earth_radius_km

# 5. Prepare data for comparison (extract emission values as numpy array for speed)
emission_data = locations_df[emission_cols].values

# --- Parallel Processing Setup ---

# Shared array to mark rows for removal (initialized to 0, meaning "keep")
# Use 'i' for integer type. Size is the number of rows.
shared_remove_flags = Array('i', len(locations_df), lock=True) # Using lock=True for atomic access

# Shared lock for synchronizing access to shared_remove_flags
# Although Array with lock=True provides some atomicity, complex checks might still need explicit locking
lock = Lock()

# --- Worker Function ---
def process_indices(indices_chunk):
    """
    Processes a chunk of indices to find duplicates.
    Marks duplicates (except the one with the lowest index) in the shared_remove_flags array.
    """
    global tree, distance_threshold_rad, emission_data, shared_remove_flags, lock, locations_df

    rows_to_remove_in_chunk = set() # Keep track locally before writing to shared array

    for i in indices_chunk:
        # Read the flag first without lock to potentially skip work quickly
        # Reading is generally safe without lock if we only care about 0 vs non-zero
        if shared_remove_flags[i] != 0:
            continue # Already marked for removal by another process

        # Find neighbors within the distance threshold
        # query_radius returns distances and indices
        # We need indices relative to the original dataframe used to build the tree
        try:
            indices_within_radius = tree.query_radius(
                locations_df[['rad_lat', 'rad_lon']].iloc[i:i+1].values, # Query for point i
                r=distance_threshold_rad
            )[0] # Result is a list containing one array; get the array
        except Exception as e:
             print(f"Error during tree query for index {i}: {e}")
             continue # Skip this point if query fails

        # Identify potential duplicates: nearby points with exact emission match
        current_emissions = emission_data[i]
        duplicate_indices = [i] # Start with the current point itself

        for j in indices_within_radius:
            # Skip self-comparison and check if j is already marked for removal
            if i == j or shared_remove_flags[j] != 0:
                continue

            # Compare emissions (use np.array_equal for efficient exact match)
            if np.array_equal(current_emissions, emission_data[j]):
                 # Check distance again explicitly if needed (query_radius should be accurate)
                 # dist = tree.dist(locations_df[['rad_lat', 'rad_lon']].iloc[i:i+1].values,
                 #                 locations_df[['rad_lat', 'rad_lon']].iloc[j:j+1].values)[0][0] * earth_radius_km
                 # if dist <= distance_threshold_km: # Redundant check usually
                 duplicate_indices.append(j)


        # If duplicates were found (more than just the point itself)
        if len(duplicate_indices) > 1:
            # Decide which one to keep: the one with the minimum original index
            min_index_in_group = min(duplicate_indices)

            # Mark all others in the group for removal
            with lock: # Ensure atomic update of shared flags
                 for idx in duplicate_indices:
                     if idx != min_index_in_group:
                         # Check again inside the lock to prevent race conditions
                         # where another process might have processed this idx
                         # between the outer check and acquiring the lock.
                         if shared_remove_flags[idx] == 0:
                            shared_remove_flags[idx] = 1 # Mark for removal
    # Return value isn't strictly necessary here as we modify the shared array
    return True


# --- Execute in Parallel ---
num_processes = min(cpu_count(), 32)
print(f"Starting parallel processing with {num_processes} processes...")

# Divide indices into chunks for workers
chunk_size = max(1, len(locations_df) // num_processes)
index_ranges = [
    range(i, min(i + chunk_size, len(locations_df)))
    for i in range(0, len(locations_df), chunk_size)
]

# Create a pool of worker processes
with Pool(processes=num_processes) as pool:
    # Run the processing function on the chunks
    results = pool.map(process_indices, index_ranges)

print("Parallel processing finished.")

# --- Final Filtering ---
# Convert the shared array (which is a ctypes array wrapper) to a numpy array or list
remove_flags_list = list(shared_remove_flags)

# Indices to keep are where the flag is 0
indices_to_keep = [i for i, flag in enumerate(remove_flags_list) if flag == 0]

# Create the filtered DataFrame
filtered_df = locations_df.iloc[indices_to_keep].copy()

# Clean up temporary columns
filtered_df.drop(columns=['rad_lat', 'rad_lon'], inplace=True)

end_time = time.time()

# --- Output Results ---
print("\n--- Results ---")
print(f"Original number of power plants: {len(locations_df)}")
print(f"Number of power plants after filtering: {len(filtered_df)}")
print(f"Number of duplicate plants removed: {len(locations_df) - len(filtered_df)}")
print(f"Processing time: {end_time - start_time:.2f} seconds")

# Print statistics by country
if country_col in locations_df.columns and country_col in filtered_df.columns:
    print("\n--- Power Plants Filtered by Country ---")
    filtered_country_counts = filtered_df[country_col].value_counts().to_dict()
    
    # Calculate removed counts per country
    all_countries = set(original_country_counts.keys()) | set(filtered_country_counts.keys())
    removed_by_country = []
    
    for country in all_countries:
        original_count = original_country_counts.get(country, 0)
        filtered_count = filtered_country_counts.get(country, 0)
        removed_count = original_count - filtered_count
        if removed_count > 0:  # Only include countries where plants were removed
            removed_by_country.append((country, original_count, filtered_count, removed_count))
    
    # Sort by number of removed plants (descending)
    removed_by_country.sort(key=lambda x: x[3], reverse=True)
    
    print(f"{'Country':<15} {'Original':<12} {'After Filter':<15} {'Removed':<10}")
    print("-" * 55)
    for country, orig, filtered, removed in removed_by_country:
        print(f"{country:<15} {orig:<12} {filtered:<15} {removed:<10}")
    
    total_removed = sum([x[3] for x in removed_by_country])
    print("-" * 55)
    print(f"{'TOTAL':<15} {len(locations_df):<12} {len(filtered_df):<15} {total_removed:<10}")

# Optional: Save the filtered data
filtered_df.to_csv('filtered_power_plants.csv', index=False)
# print("Filtered data saved to 'filtered_power_plants.csv'")

# Display first few rows of filtered data (optional)
# print("\nFirst 5 rows of filtered data:")
# print(filtered_df.head())