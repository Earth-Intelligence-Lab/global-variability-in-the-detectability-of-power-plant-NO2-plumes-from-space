import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from haversine import haversine, Unit
from math import radians, log10, sqrt
from tqdm import tqdm # A progress bar for long loops

# --- Constants ---
EARTH_RADIUS_KM = 6371

# --- Helper Function to prepare data for searching ---
def _process_source_dataframe(df_raw, lat_col, lon_col, id_col=None, value_col=None):
    """
    Standardises a dataframe of sources (plants or cities) and builds a BallTree
    for efficient spatial lookups.
    """
    if df_raw is None or df_raw.empty:
        return None, None

    df = df_raw.copy()
    # Define standard column names
    std_lat, std_lon, std_id, std_value = 'latitude', 'longitude', 'ID', 'value'

    col_map = {lat_col: std_lat, lon_col: std_lon}
    if id_col: col_map[id_col] = std_id
    if value_col: col_map[value_col] = std_value

    # Keep only the necessary columns and rename them
    df = df[list(col_map.keys())].rename(columns=col_map)

    # Convert to numeric types, coercing errors to NaN
    for c in [std_lat, std_lon, std_value]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing essential data
    df.dropna(subset=[std_lat, std_lon], inplace=True)
    
    # --- inside _process_source_dataframe (after dropna) ---
    df = df.reset_index(drop=True)  # stable indices for BallTree/self-skip
    
    if df.empty:
        return df, None

    # Create radian versions of coordinates for the BallTree
    df['lat_rad'] = np.radians(df[std_lat])
    df['lon_rad'] = np.radians(df[std_lon])

    # Build the BallTree for fast distance queries
    tree = BallTree(df[['lat_rad', 'lon_rad']].values, metric='haversine')

    return df, tree

# --- MODIFIED Main Analysis Function ---
def analyze_geometric_interference_detailed(plants_df_raw, cities_df_raw):
    """
    Counts how many power plants fall into the interference zone of another
    plant and how many fall into the interference zone of a city.

    Returns:
        - A set of IDs for plants interfered with by other plants.
        - A set of IDs for plants interfered with by cities.
    """

    print("Step 1: Preparing dataframes and search trees...")

    # --- Configuration for Interference Zones ---
    PLANT_MAX_SEARCH_KM = 150.0
    PLANT_RADIUS_SCALE = 0
    PLANT_RADIUS_BASE_KM = 20.0
    PLANT_RADIUS_MAX_KM = 50.0

    CITY_MAX_SEARCH_KM = 150.0
    CITY_POP_THRESHOLD = 200000
    CITY_RADIUS_SCALE = 9.0
    CITY_RADIUS_BASE_KM = 10.0
    CITY_RADIUS_MAX_KM = 90.0

    # Prepare the source dataframes
    target_plants_df = plants_df_raw.copy().rename(columns={'ID': 'ID'})

    source_plants_df, plant_tree = _process_source_dataframe(
        plants_df_raw, 'latitude', 'longitude', id_col='ID', value_col='nox_emis_ty'
    )
    source_cities_df, city_tree = _process_source_dataframe(
        cities_df_raw, 'latitude', 'longitude', id_col='name', value_col='population'
    )

    # --- MODIFIED --- Using two separate sets to track interference sources
    plant_interfered_ids = set()
    city_interfered_ids = set()

    print("Step 2: Checking each power plant for interference from all sources...")
    for _, target_plant in tqdm(target_plants_df.iterrows(), total=len(target_plants_df)):
        target_id = target_plant['ID']
        target_lat = target_plant['latitude']
        target_lon = target_plant['longitude']
        target_emissions = target_plant.get('nox_emis_ty', 0)

        if pd.isna(target_lat) or pd.isna(target_lon) or pd.isna(target_emissions):
            continue

        target_coords_rad = np.array([[radians(target_lat), radians(target_lon)]])

        # A. Find nearby power plants and check for interference
        # This loop completes for each target plant.
        if plant_tree:
            max_rad = PLANT_MAX_SEARCH_KM / EARTH_RADIUS_KM
            nearby_indices = plant_tree.query_radius(target_coords_rad, r=max_rad)[0]

            for idx in nearby_indices:
                source_plant = source_plants_df.iloc[idx]
                if source_plant['ID'] == target_id:
                    continue

                # --- inside plant loop (A. plants) ---
                emissions = source_plant.get('value', 0)
                # require STRICTLY higher emissions (match previous run)
                if pd.isna(emissions) or emissions <= target_emissions:
                    continue

                radius = PLANT_RADIUS_BASE_KM + (sqrt(max(0, emissions)) * PLANT_RADIUS_SCALE)
                interference_radius_km = min(radius, PLANT_RADIUS_MAX_KM)

                distance_km = haversine(
                    (target_lat, target_lon),
                    (source_plant['latitude'], source_plant['longitude']),
                    unit=Unit.KILOMETERS
                )

                if distance_km < interference_radius_km:
                    plant_interfered_ids.add(target_id)
                    break # Found a plant interference, no need to check other plants

        # B. Find nearby cities and check for interference
        # This loop also completes for each target plant, regardless of plant interference.
        if city_tree:
            max_rad = CITY_MAX_SEARCH_KM / EARTH_RADIUS_KM
            nearby_indices = city_tree.query_radius(target_coords_rad, r=max_rad)[0]

            for idx in nearby_indices:
                source_city = source_cities_df.iloc[idx]
                population = source_city.get('value', 0)

                if pd.isna(population) or population < CITY_POP_THRESHOLD:
                    continue

                radius = CITY_RADIUS_BASE_KM + (log10(max(1, population)) * CITY_RADIUS_SCALE)
                interference_radius_km = min(radius, CITY_RADIUS_MAX_KM)

                distance_km = haversine(
                    (target_lat, target_lon),
                    (source_city['latitude'], source_city['longitude']),
                    unit=Unit.KILOMETERS
                )

                if distance_km < interference_radius_km:
                    city_interfered_ids.add(target_id)
                    break # Found a city interference, no need to check other cities

    return plant_interfered_ids, city_interfered_ids


# ─── Training-specific interference functions ────────────────────────────────
# Simplified versions used by the MLP training scripts.

def identify_interference_us_by_year(plants_df, annual_emissions_df, cities_df,
                                      plant_subset_ids=None,
                                      plant_radius_km=20.0, city_pop_threshold=200000,
                                      city_radius_scale=9.0, city_radius_base_km=10.0,
                                      city_radius_max_km=90.0):
    """Identify interfered US plants per year.

    Returns dict: year -> set of Facility IDs.
    """
    print("Identifying plants in interference zones by year...")

    if plant_subset_ids is not None:
        plants_df = plants_df[plants_df['Facility_ID'].isin(plant_subset_ids)].copy()
        annual_emissions_df = annual_emissions_df[
            annual_emissions_df['Facility ID'].isin(plant_subset_ids)].copy()

    plants_m = plants_df.copy()
    if 'Facility_ID' in plants_m.columns:
        plants_m.rename(columns={'Facility_ID': 'Facility ID'}, inplace=True)
    for c in ['Latitude', 'Longitude', 'State', 'Facility_Name', 'Primary_Fuel_Type']:
        plants_m.drop(columns=[c], errors='ignore', inplace=True)

    full = pd.merge(annual_emissions_df, plants_m, on='Facility ID', how='left')
    full.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    cities_f = cities_df[cities_df['population'] >= city_pop_threshold].copy()
    cities_f['lat_rad'] = np.radians(cities_f['latitude'])
    cities_f['lon_rad'] = np.radians(cities_f['longitude'])
    city_tree = BallTree(cities_f[['lat_rad', 'lon_rad']].values, metric='haversine') \
        if len(cities_f) > 0 else None

    year_interfered = {}
    for year in [int(y) for y in sorted(full['Year'].unique())]:
        pyr = full[full['Year'] == year].copy()
        if pyr.empty:
            year_interfered[year] = set()
            continue

        pyr['lat_rad'] = np.radians(pyr['Latitude'])
        pyr['lon_rad'] = np.radians(pyr['Longitude'])
        ptree = BallTree(pyr[['lat_rad', 'lon_rad']].values, metric='haversine')

        interfered = set()
        for _, tgt in pyr.iterrows():
            tid, tlat, tlon = tgt['Facility ID'], tgt['Latitude'], tgt['Longitude']
            temis = tgt.get('NOx Mass (short tons)', 0)
            if pd.isna(tlat) or pd.isna(tlon):
                continue
            crd = np.array([[radians(tlat), radians(tlon)]])

            for pi in ptree.query_radius(crd, r=plant_radius_km / EARTH_RADIUS_KM)[0]:
                src = pyr.iloc[pi]
                if src['Facility ID'] == tid:
                    continue
                se = src.get('NOx Mass (short tons)', 0)
                if pd.notna(se) and se > temis:
                    if haversine((tlat, tlon), (src['Latitude'], src['Longitude'])) < plant_radius_km:
                        interfered.add(int(tid))
                        break

            if city_tree and tid not in interfered:
                for ci in city_tree.query_radius(crd, r=city_radius_max_km / EARTH_RADIUS_KM)[0]:
                    city = cities_f.iloc[ci]
                    ir = min(city_radius_base_km + log10(max(1, city['population'])) * city_radius_scale,
                             city_radius_max_km)
                    if haversine((tlat, tlon), (city['latitude'], city['longitude'])) < ir:
                        interfered.add(int(tid))
                        break

        year_interfered[year] = interfered
        print(f"Year {year}: {len(interfered)} interfered / {len(pyr)} plants")
    return year_interfered


def identify_interference_world(plants_df, cities_df, plant_subset_ids=None,
                                 plant_radius_km=20.0, city_pop_threshold=200000,
                                 city_radius_scale=9.0, city_radius_base_km=10.0,
                                 city_radius_max_km=90.0):
    """Identify interfered World plants (static, no per-year).

    Returns set of plant IDs.
    """
    print("Identifying plants in interference zones (global)...")

    if plant_subset_ids is not None:
        plants_df = plants_df[plants_df['ID'].isin(plant_subset_ids)].copy()

    if 'nox_emis_ty' in plants_df.columns:
        plants_df['annual_nox_emission'] = plants_df['nox_emis_ty']
    elif 'annual_nox_emission' not in plants_df.columns:
        plants_df['annual_nox_emission'] = -plants_df.index

    plants_df['lat_rad'] = np.radians(plants_df['latitude'])
    plants_df['lon_rad'] = np.radians(plants_df['longitude'])
    ptree = BallTree(plants_df[['lat_rad', 'lon_rad']].values, metric='haversine')

    cities_f = cities_df[cities_df['population'] >= city_pop_threshold].copy()
    cities_f['lat_rad'] = np.radians(cities_f['latitude'])
    cities_f['lon_rad'] = np.radians(cities_f['longitude'])
    city_tree = BallTree(cities_f[['lat_rad', 'lon_rad']].values, metric='haversine') \
        if len(cities_f) > 0 else None

    interfered = set()
    for idx, tgt in tqdm(plants_df.iterrows(), total=len(plants_df), desc="Checking interference"):
        tid, tlat, tlon = tgt['ID'], tgt['latitude'], tgt['longitude']
        temis = tgt.get('annual_nox_emission', 0)
        if pd.isna(tlat) or pd.isna(tlon):
            continue
        crd = np.array([[radians(tlat), radians(tlon)]])

        for pi in ptree.query_radius(crd, r=plant_radius_km / EARTH_RADIUS_KM)[0]:
            if pi == idx:
                continue
            src = plants_df.iloc[pi]
            se = src.get('annual_nox_emission', 0)
            if pd.notna(se) and se > temis:
                if haversine((tlat, tlon), (src['latitude'], src['longitude'])) < plant_radius_km:
                    interfered.add(tid)
                    break

        if city_tree and tid not in interfered:
            for ci in city_tree.query_radius(crd, r=city_radius_max_km / EARTH_RADIUS_KM)[0]:
                city = cities_f.iloc[ci]
                ir = min(city_radius_base_km + log10(max(1, city['population'])) * city_radius_scale,
                         city_radius_max_km)
                if haversine((tlat, tlon), (city['latitude'], city['longitude'])) < ir:
                    interfered.add(tid)
                    break

    print(f"Found {len(interfered)} interfered / {len(plants_df)} plants")
    return interfered


def filter_data_by_year_interference(tropomi_df, year_interfered_dict):
    """Exclude observations from plants interfered in ANY year (strict, matches paper).

    A plant is dropped from ALL years if it was flagged as interfered in any single
    year — i.e. the kept set is the intersection of per-year non-interfered plants
    ("never within interference zones across six years"; see Sec. 3.3 of the paper).
    """
    ever_interfered = set().union(*[set(v) for v in year_interfered_dict.values()]) \
        if year_interfered_dict else set()
    filtered = []
    for year in sorted(tropomi_df['year'].unique()):
        yd = tropomi_df[tropomi_df['year'] == year]
        yf = yd[~yd['location'].isin(ever_interfered)]
        filtered.append(yf)
        print(f"Year {year}: {len(yd)} -> {len(yf)} "
              f"({yd['location'].nunique() - yf['location'].nunique()} plants removed)")
    print(f"Strict filter: {len(ever_interfered)} plants ever-interfered → "
          f"dropped from all years")
    return pd.concat(filtered, ignore_index=True) if filtered else pd.DataFrame()


def extract_year_from_datetime(df):
    """Extract year column from TROPOMI data, trying multiple column names."""
    if 'year' in df.columns:
        return df
    if 'utc_time' in df.columns:
        try:
            df['year'] = pd.to_datetime(df['utc_time'], utc=True).dt.year
        except Exception:
            try:
                df['year'] = pd.to_datetime(df['utc_time']).dt.year
            except Exception:
                df['year'] = df['utc_time'].astype(str).str[:4].astype(int)
        return df
    for col in ['datetime', 'date', 'time', 'measurement_time', 'time_utc']:
        if col in df.columns:
            try:
                df['year'] = pd.to_datetime(df[col]).dt.year
                return df
            except Exception:
                continue
    print("WARNING: Could not extract year. Defaulting to 2024.")
    df['year'] = 2024
    return df


if __name__ == '__main__':
    # --- Load Your Data ---
    # Update these paths to your actual file locations
    plants_file = '/net/fs06/d3/rzhuang/TROPOMI_world/data/power_plant_location/power_plants_with_combined_nearby_stats.csv'
    cities_file = '/net/fs06/d3/rzhuang/TROPOMI_world/data/worldcities.csv'

    try:
        print(f"Loading power plants data from: {plants_file}")
        # --- when loading plants ---
        all_plants = (pd.read_csv(plants_file)
                    .assign(nox_emis_ty=pd.to_numeric(lambda x: x['nox_emis_ty'], errors='coerce'))
                    .dropna(subset=['latitude','longitude','nox_emis_ty'])
                    .sort_values('nox_emis_ty', ascending=False)
                    .head(6000)
                    .reset_index(drop=True))
        print(f"Loaded {len(all_plants)} power plant records.")

        print(f"Loading cities data from: {cities_file}")
        all_cities = pd.read_csv(cities_file)
        print(f"Loaded {len(all_cities)} city records.")
        print("-" * 30)

        # --- MODIFIED --- Run the detailed analysis
        plant_interfered_set, city_interfered_set = analyze_geometric_interference_detailed(all_plants, all_cities)

        # --- MODIFIED --- Calculate and display detailed results
        plant_interference_count = len(plant_interfered_set)
        city_interference_count = len(city_interfered_set)
        
        # A plant can be in both zones, so we use a union to find the total unique count
        total_unique_interfered_count = len(plant_interfered_set.union(city_interfered_set))
        
        # A plant can be in both zones, so we use an intersection to find the overlap
        both_interference_count = len(plant_interfered_set.intersection(city_interfered_set))

        print("\n" + "=" * 40)
        print("          Analysis Complete")
        print("=" * 40)
        print("\n--- Interference Zone Breakdown ---")
        print(f"Plants covered by a power plant's interference zone: {plant_interference_count}")
        print(f"Plants covered by a city's interference zone:      {city_interference_count}")
        print(f"Plants covered by BOTH types of zones:              {both_interference_count}")
        print("-" * 35)
        print(f"Total unique power plants in any interference zone: {total_unique_interfered_count}\n")


    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a data file.")
        print(f"Please check the path: {e.filename}")