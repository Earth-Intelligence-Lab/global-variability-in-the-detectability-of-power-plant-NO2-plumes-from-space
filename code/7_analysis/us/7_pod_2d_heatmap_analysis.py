#!/usr/bin/env python3
# ───────────────────────────────────────────────────────────────
#  2D Feature Analysis: Probability of Detection Heatmaps
#  Analyzing: (NOx emission, surface albedo) and (NOx emission, wind speed)
# ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict, Any, Optional
import seaborn as sns
from pathlib import Path
from math import radians, log10, sin, cos, asin, sqrt
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

# ───────────────────────────────────────────────────────────────
#  Paths (EDIT THESE)
# ───────────────────────────────────────────────────────────────
OBS_FILES = {
    "Global": Path("/net/fs06/d3/rzhuang/TROPOMI_world/data/Run_3/updated_tropomi_emissions_full_variables_with_fuel.csv"),
    "US": Path("/net/fs06/d3/rzhuang/TROPOMI_US/data/Run_20250623_203825/updated_tropomi_hourly_emissions_full_variables.csv"),
}

PATHS = {
    "US_PLANTS": Path('/net/fs06/d3/rzhuang/TROPOMI_US/data/facility_emissions_by_plant_comprehensive.csv'),
    "US_ANNUAL_EMIS": Path('/net/fs06/d3/rzhuang/TROPOMI_US/data/annual-emissions-facility-aggregation-2019-2024.csv'),
    "GLOBAL_PLANTS": Path('/net/fs06/d3/rzhuang/TROPOMI_world/data/power_plant_location/power_plants_with_combined_nearby_stats.csv'),
    "CITIES": Path('/net/fs06/d3/rzhuang/TROPOMI_world/data/worldcities.csv'),
}

# ───────────────────────────────────────────────────────────────
#  Constants
# ───────────────────────────────────────────────────────────────
EARTH_RADIUS_KM = 6371.0
PLANT_RADIUS_BASE_KM = 20.0
CITY_POP_THRESHOLD = 200000
CITY_RADIUS_BASE_KM = 10.0
CITY_RADIUS_SCALE = 9.0
CITY_RADIUS_MAX_KM = 90.0

# Binning parameters
N_BINS_X = 10  # Number of bins for x-axis
N_BINS_Y = 10  # Number of bins for y-axis
MIN_COUNT = 5  # Minimum samples per bin to show
LBS_TO_KG = 0.45359237

# Styling
plt.style.use('default')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'Nimbus Roman',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'text.color': '#2E3440',
})

TARGET_CANDIDATES = ['plume_label', 'label', 'detected', 'is_detected', 'target', 'y']

# ───────────────────────────────────────────────────────────────
#  Helper Functions
# ───────────────────────────────────────────────────────────────
def haversine(p1, p2, radius_km: float = EARTH_RADIUS_KM) -> float:
    lat1, lon1 = p1
    lat2, lon2 = p2
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1r, lat2r = radians(lat1), radians(lat2)
    a = sin(dlat/2)**2 + cos(lat1r) * cos(lat2r) * sin(dlon/2)**2
    return 2 * radius_km * asin(sqrt(a))

def extract_year_from_utc_time(df: pd.DataFrame) -> pd.DataFrame:
    if 'year' in df.columns:
        print(f"Found 'year' column. Years: {sorted(df['year'].dropna().unique())}")
        return df
    if 'utc_time' not in df.columns:
        print("ERROR: 'utc_time' column not found! Setting year=2024.")
        df['year'] = 2024
        return df
    s = df['utc_time']
    try:
        year = pd.to_datetime(s, utc=True, errors='coerce', format='mixed').dt.year
    except TypeError:
        year = pd.to_datetime(s, utc=True, errors='coerce').dt.year
    if year.isna().any():
        guess = pd.to_numeric(s.astype(str).str[:4], errors='coerce')
        year = year.fillna(guess)
    df['year'] = year.astype('Int64')
    years_in_data = sorted([int(x) for x in df['year'].dropna().unique()])
    print(f"Successfully extracted years: {years_in_data}")
    return df

def identify_plants_in_interference_zones_by_year(plants_df: pd.DataFrame,
                                                  annual_emissions_df: pd.DataFrame,
                                                  cities_df: pd.DataFrame,
                                                  plant_subset_ids: Optional[List[Any]] = None) -> Dict[int, set]:
    print("Identifying plants in interference zones by year...")
    if plant_subset_ids is not None:
        plants_df = plants_df[plants_df['Facility_ID'].isin(plant_subset_ids)].copy()
        annual_emissions_df = annual_emissions_df[annual_emissions_df['Facility ID'].isin(plant_subset_ids)].copy()
    plants_df_for_merge = plants_df.copy()
    if 'Facility_ID' in plants_df_for_merge.columns:
        plants_df_for_merge.rename(columns={'Facility_ID': 'Facility ID'}, inplace=True)
    columns_to_drop = ['Latitude', 'Longitude', 'State', 'Facility_Name', 'Primary_Fuel_Type']
    plants_df_for_merge = plants_df_for_merge.drop(columns=columns_to_drop, errors='ignore')
    full_plant_data = pd.merge(annual_emissions_df, plants_df_for_merge, on='Facility ID', how='left')
    full_plant_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    cities_filtered = cities_df[cities_df['population'] >= CITY_POP_THRESHOLD].copy()
    cities_filtered['lat_rad'] = np.radians(cities_filtered['latitude'])
    cities_filtered['lon_rad'] = np.radians(cities_filtered['longitude'])
    if len(cities_filtered) > 0:
        city_tree = BallTree(cities_filtered[['lat_rad', 'lon_rad']].values, metric='haversine')
    else:
        city_tree = None
    years = sorted(full_plant_data['Year'].unique())
    year_interfered_plants = {}
    for year in years:
        plants_this_year = full_plant_data[full_plant_data['Year'] == year].copy()
        if plants_this_year.empty:
            year_interfered_plants[year] = set()
            continue
        plants_this_year['lat_rad'] = np.radians(plants_this_year['Latitude'])
        plants_this_year['lon_rad'] = np.radians(plants_this_year['Longitude'])
        plant_tree = BallTree(plants_this_year[['lat_rad', 'lon_rad']].values, metric='haversine')
        interfered_plants = set()
        for idx, target_plant in plants_this_year.iterrows():
            target_id = target_plant['Facility ID']
            target_lat = target_plant['Latitude']
            target_lon = target_plant['Longitude']
            target_emissions = target_plant.get('NOx Mass (short tons)', 0)
            if pd.isna(target_lat) or pd.isna(target_lon):
                continue
            target_coords_rad = np.array([[radians(target_lat), radians(target_lon)]])
            search_radius = PLANT_RADIUS_BASE_KM / EARTH_RADIUS_KM
            nearby_plant_indices = plant_tree.query_radius(target_coords_rad, r=search_radius)[0]
            for plant_idx in nearby_plant_indices:
                source_plant = plants_this_year.iloc[plant_idx]
                if source_plant['Facility ID'] == target_id:
                    continue
                source_emissions = source_plant.get('NOx Mass (short tons)', 0)
                if pd.notna(source_emissions) and source_emissions > target_emissions:
                    distance_km = haversine((target_lat, target_lon), (source_plant['Latitude'], source_plant['Longitude']))
                    if distance_km < PLANT_RADIUS_BASE_KM:
                        interfered_plants.add(str(target_id).strip())
                        break
            if city_tree and target_id not in interfered_plants:
                search_radius = CITY_RADIUS_MAX_KM / EARTH_RADIUS_KM
                nearby_city_indices = city_tree.query_radius(target_coords_rad, r=search_radius)[0]
                for city_idx in nearby_city_indices:
                    source_city = cities_filtered.iloc[city_idx]
                    population = source_city['population']
                    radius = CITY_RADIUS_BASE_KM + (log10(max(1, population)) * CITY_RADIUS_SCALE)
                    interference_radius_km = min(radius, CITY_RADIUS_MAX_KM)
                    distance_km = haversine((target_lat, target_lon), (source_city['latitude'], source_city['longitude']))
                    if distance_km < interference_radius_km:
                        interfered_plants.add(str(target_id).strip())
                        break
        year_interfered_plants[year] = interfered_plants
    return year_interfered_plants

def filter_data_by_year_interference(tropomi_df: pd.DataFrame, 
                                     year_interfered_dict: Dict[int, set]) -> pd.DataFrame:
    filtered_dfs = []
    years_in_data = sorted(tropomi_df['year'].unique())
    for year in years_in_data:
        year_data = tropomi_df[tropomi_df['year'] == year].copy()
        if year in year_interfered_dict:
            interfered_ids = {str(x).strip() for x in year_interfered_dict[year]}
            loc = year_data['location'].astype(str).str.strip()
            year_data_filtered = year_data[~loc.isin(interfered_ids)]
            filtered_dfs.append(year_data_filtered)
        else:
            filtered_dfs.append(year_data)
    if filtered_dfs:
        return pd.concat(filtered_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def identify_plants_in_interference_zones_world(plants_df, cities_df, plant_subset_ids=None):
    print("Identifying plants in interference zones (global)...")
    if plant_subset_ids is not None:
        plants_df = plants_df[plants_df['ID'].isin(plant_subset_ids)].copy()
    if 'nox_emis_ty' in plants_df.columns:
        plants_df['annual_nox_emission'] = plants_df['nox_emis_ty']
    else:
        plants_df['annual_nox_emission'] = -plants_df.index
    plants_df['lat_rad'] = np.radians(plants_df['latitude'])
    plants_df['lon_rad'] = np.radians(plants_df['longitude'])
    plant_tree = BallTree(plants_df[['lat_rad', 'lon_rad']].values, metric='haversine')
    cities_filtered = cities_df[cities_df['population'] >= CITY_POP_THRESHOLD].copy()
    cities_filtered['lat_rad'] = np.radians(cities_filtered['latitude'])
    cities_filtered['lon_rad'] = np.radians(cities_filtered['longitude'])
    if len(cities_filtered) > 0:
        city_tree = BallTree(cities_filtered[['lat_rad', 'lon_rad']].values, metric='haversine')
    else:
        city_tree = None
    interfered_plants = set()
    for idx, target_plant in tqdm(plants_df.iterrows(), total=len(plants_df), desc="Checking interference"):
        target_id = target_plant['ID']
        target_lat = target_plant['latitude']
        target_lon = target_plant['longitude']
        target_emissions = target_plant.get('annual_nox_emission', 0)
        if pd.isna(target_lat) or pd.isna(target_lon):
            continue
        target_coords_rad = np.array([[radians(target_lat), radians(target_lon)]])
        search_radius = PLANT_RADIUS_BASE_KM / EARTH_RADIUS_KM
        nearby_plant_indices = plant_tree.query_radius(target_coords_rad, r=search_radius)[0]
        for plant_idx in nearby_plant_indices:
            if plant_idx == idx:
                continue
            source_plant = plants_df.iloc[plant_idx]
            source_emissions = source_plant.get('annual_nox_emission', 0)
            if pd.notna(source_emissions) and source_emissions > target_emissions:
                distance_km = haversine((target_lat, target_lon), (source_plant['latitude'], source_plant['longitude']))
                if distance_km < PLANT_RADIUS_BASE_KM:
                    interfered_plants.add(str(target_id).strip())
                    break
        if city_tree and target_id not in interfered_plants:
            search_radius = CITY_RADIUS_MAX_KM / EARTH_RADIUS_KM
            nearby_city_indices = city_tree.query_radius(target_coords_rad, r=search_radius)[0]
            for city_idx in nearby_city_indices:
                source_city = cities_filtered.iloc[city_idx]
                population = source_city['population']
                radius = CITY_RADIUS_BASE_KM + (log10(max(1, population)) * CITY_RADIUS_SCALE)
                interference_radius_km = min(radius, CITY_RADIUS_MAX_KM)
                distance_km = haversine((target_lat, target_lon), (source_city['latitude'], source_city['longitude']))
                if distance_km < interference_radius_km:
                    interfered_plants.add(str(target_id).strip())
                    break
    return interfered_plants

def _to_str_set(x):
    return set(pd.Series(list(x)).dropna().astype(str).str.strip())

def _ensure_obs_id_column_global(df: pd.DataFrame, valid_ids: set) -> str:
    if 'location' in df.columns:
        return 'location'
    if 'plant_id' in df.columns:
        df['location'] = df['plant_id']
        return 'location'
    if 'ID' in df.columns:
        df['location'] = df['ID'] 
        return 'location'
    raise KeyError("No valid plant ID column found")

def filter_data_by_interference_global(tropomi_df: pd.DataFrame, interfered_ids: set, valid_ids: set) -> pd.DataFrame:
    id_col = _ensure_obs_id_column_global(tropomi_df, valid_ids)
    before = len(tropomi_df)
    bad = tropomi_df[id_col].astype(str).str.strip().isin(_to_str_set(interfered_ids))
    out = tropomi_df[~bad].copy()
    return out

def find_target_col(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"None of target columns {TARGET_CANDIDATES} found.")

def resolve_feature_col(df: pd.DataFrame, feature_candidates: List[str]) -> str:
    """Try to find a matching column from a list of candidates"""
    for cand in feature_candidates:
        if cand in df.columns:
            return cand
    raise KeyError(f"Could not find any of {feature_candidates} in data columns.")

def compute_2d_pod_heatmap(df: pd.DataFrame, 
                          x_col: str, 
                          y_col: str, 
                          target_col: str,
                          n_bins_x: int = N_BINS_X,
                          n_bins_y: int = N_BINS_Y,
                          min_count: int = MIN_COUNT,
                          log_x: bool = False,
                          log_y: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D probability of detection heatmap
    
    Returns:
        pod_grid: 2D array of POD values
        count_grid: 2D array of sample counts
        x_edges: Bin edges for x-axis
        y_edges: Bin edges for y-axis
    """
    # Extract and clean data
    x_vals = pd.to_numeric(df[x_col], errors='coerce')
    y_vals = pd.to_numeric(df[y_col], errors='coerce')
    target = pd.to_numeric(df[target_col], errors='coerce')
    
    # Remove NaNs and ensure positive values for log scale
    valid = x_vals.notna() & y_vals.notna() & target.notna()
    if log_x:
        valid = valid & (x_vals > 0)
    if log_y:
        valid = valid & (y_vals > 0)
    
    x_vals = x_vals[valid].values
    y_vals = y_vals[valid].values
    target = target[valid].values
    
    if len(x_vals) == 0:
        return np.zeros((n_bins_y, n_bins_x)), np.zeros((n_bins_y, n_bins_x)), np.array([]), np.array([])
    
    # Apply log transform if needed
    x_vals_for_binning = np.log10(x_vals) if log_x else x_vals
    y_vals_for_binning = np.log10(y_vals) if log_y else y_vals
    
    # Create bins using quantiles to ensure balanced distribution
    try:
        x_edges_transformed = np.percentile(x_vals_for_binning, np.linspace(0, 100, n_bins_x + 1))
        y_edges_transformed = np.percentile(y_vals_for_binning, np.linspace(0, 100, n_bins_y + 1))
        
        # Remove duplicate edges
        x_edges_transformed = np.unique(x_edges_transformed)
        y_edges_transformed = np.unique(y_edges_transformed)
    except:
        x_edges_transformed = np.linspace(x_vals_for_binning.min(), x_vals_for_binning.max(), n_bins_x + 1)
        y_edges_transformed = np.linspace(y_vals_for_binning.min(), y_vals_for_binning.max(), n_bins_y + 1)
    
    # Convert back to original scale for display
    x_edges = 10**x_edges_transformed if log_x else x_edges_transformed
    y_edges = 10**y_edges_transformed if log_y else y_edges_transformed
    
    # Digitize data into bins
    x_bins = np.digitize(x_vals, x_edges) - 1
    y_bins = np.digitize(y_vals, y_edges) - 1
    
    # Clip to valid range
    x_bins = np.clip(x_bins, 0, len(x_edges) - 2)
    y_bins = np.clip(y_bins, 0, len(y_edges) - 2)
    
    # Initialize grids
    pod_grid = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan)
    count_grid = np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    
    # Compute POD for each bin
    for i in range(len(y_edges) - 1):
        for j in range(len(x_edges) - 1):
            mask = (x_bins == j) & (y_bins == i)
            count = mask.sum()
            count_grid[i, j] = count
            
            if count >= min_count:
                pod = target[mask].mean()
                pod_grid[i, j] = pod
    
    return pod_grid, count_grid, x_edges, y_edges

def plot_2d_heatmap(ax, pod_grid, count_grid, x_edges, y_edges, 
                   title: str, xlabel: str, ylabel: str, 
                   vmin: float = 0.0, vmax: float = 1.0,
                   log_x: bool = False, log_y: bool = False):
    """Plot 2D POD heatmap with proper formatting"""
    
    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_edges, y_edges)
    
    # Plot heatmap
    im = ax.pcolormesh(X, Y, pod_grid, cmap='RdYlGn', vmin=vmin, vmax=vmax, 
                       shading='flat', edgecolors='face', linewidth=0.5, alpha=0.9)
    
    # Set log scale if needed
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probability of Detection', fontsize=11, fontweight='600')
    cbar.ax.tick_params(labelsize=9)
    
    # Overlay sample counts as text (optional, only for larger counts)
    for i in range(len(y_edges) - 1):
        for j in range(len(x_edges) - 1):
            count = count_grid[i, j]
            if count >= MIN_COUNT and not np.isnan(pod_grid[i, j]):
                x_center = np.sqrt(x_edges[j] * x_edges[j + 1]) if log_x else (x_edges[j] + x_edges[j + 1]) / 2
                y_center = np.sqrt(y_edges[i] * y_edges[i + 1]) if log_y else (y_edges[i] + y_edges[i + 1]) / 2
                
                # Only show count if reasonable
                if count >= 20:
                    ax.text(x_center, y_center, f'{int(count)}', 
                           ha='center', va='center', fontsize=7, 
                           color='black', alpha=0.6, fontweight='500')
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='600')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='600')
    ax.set_title(title, fontsize=14, fontweight='700', pad=10)
    ax.tick_params(labelsize=10)
    ax.grid(False)
    
    return im

# ───────────────────────────────────────────────────────────────
#  Main Analysis
# ───────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("2D POD HEATMAP ANALYSIS")
    print("="*60)
    
    # Load observation datasets
    obs_datasets = {}
    for name, p in OBS_FILES.items():
        if p and p.exists():
            try:
                obs_datasets[name] = pd.read_csv(p).dropna()
                print(f"✓ Loaded {name} data: {len(obs_datasets[name])} rows")
            except Exception:
                try:
                    obs_datasets[name] = pd.read_parquet(p)
                    print(f"✓ Loaded {name} data: {len(obs_datasets[name])} rows")
                except Exception as e:
                    raise RuntimeError(f"[{name}] Could not read {p}: {e}")
        else:
            raise FileNotFoundError(f"[{name}] Missing observation file: {p}")
    
    df_us_raw = obs_datasets['US'].copy()
    df_gl_raw = obs_datasets['Global'].copy()
    
    # Load auxiliary filtering data
    us_plants = pd.read_csv(PATHS["US_PLANTS"]) if PATHS["US_PLANTS"].exists() else None
    us_annual = pd.read_csv(PATHS["US_ANNUAL_EMIS"]) if PATHS["US_ANNUAL_EMIS"].exists() else None
    gl_plants = pd.read_csv(PATHS["GLOBAL_PLANTS"]) if PATHS["GLOBAL_PLANTS"].exists() else None
    cities = pd.read_csv(PATHS["CITIES"]) if PATHS["CITIES"].exists() else None
    
    if cities is None:
        raise FileNotFoundError(f"[CITIES] Missing cities CSV at {PATHS['CITIES']}")
    
    # Top-500 US plants by total NOx
    us_annual['Facility ID'] = us_annual['Facility ID'].astype(str).str.strip()
    us_annual['NOx Mass (short tons)'] = pd.to_numeric(
        us_annual['NOx Mass (short tons)'], errors='coerce'
    ).fillna(0.0)
    
    top500_ids = (
        us_annual.groupby('Facility ID')['NOx Mass (short tons)']
        .sum()
        .sort_values(ascending=False)
        .head(500)
        .index.tolist()
    )
    print(f"✓ Selected top 500 US plants by NOx emission")
    
    # Filter US observations
    df_us_raw = df_us_raw[df_us_raw['location'].astype(str).str.strip().isin(set(top500_ids))].copy()
    
    # Apply interference filtering
    gl_ids = df_gl_raw['location'].astype(str).str.strip().unique().tolist()
    
    # US filtering
    print("\n[US] Filtering interference...")
    df_us = extract_year_from_utc_time(df_us_raw.copy())
    if (us_plants is not None) and (us_annual is not None):
        year_interfered = identify_plants_in_interference_zones_by_year(
            us_plants, us_annual, cities, plant_subset_ids=top500_ids
        )
        df_us = filter_data_by_year_interference(df_us, year_interfered)
    print(f"✓ US data after filtering: {len(df_us)} rows")
    
    # Global filtering
    print("\n[Global] Filtering interference...")
    if gl_plants is not None:
        interfered_global_ids = identify_plants_in_interference_zones_world(
            gl_plants, cities, plant_subset_ids=gl_ids
        )
        valid_global_ids = _to_str_set(gl_ids)
        df_gl = filter_data_by_interference_global(df_gl_raw.copy(), interfered_global_ids, valid_global_ids)
    else:
        df_gl = df_gl_raw.copy()
    print(f"✓ Global data after filtering: {len(df_gl)} rows")
    
    # Find target columns
    tgt_us = find_target_col(df_us)
    tgt_gl = find_target_col(df_gl)
    print(f"\n✓ Target columns: US='{tgt_us}', Global='{tgt_gl}'")
    
    # Ensure binary targets
    for df, tgt in [(df_us, tgt_us), (df_gl, tgt_gl)]:
        uniq = pd.Series(df[tgt]).dropna().unique().tolist()
        if set(uniq) - {0, 1}:
            df[tgt] = (pd.to_numeric(df[tgt], errors='coerce') > 0).astype(int)
    
    # Resolve feature columns
    print("\n" + "="*60)
    print("RESOLVING FEATURE COLUMNS")
    print("="*60)
    
    # NOx emission columns
    us_emission_col = resolve_feature_col(df_us, ['NOx Mass (lbs)', 'hourly_emission_rate', 'annual_nox_emission'])
    gl_emission_col = resolve_feature_col(df_gl, ['annual_nox_emission', 'nox_emis_ty'])
    print(f"✓ Emission: US='{us_emission_col}', Global='{gl_emission_col}'")
    
    # Determine emission types and units
    us_is_hourly = 'hourly' in us_emission_col.lower()
    gl_is_hourly = 'hourly' in gl_emission_col.lower()
    
    # Convert US emission from lbs to kg/h if needed
    if 'lbs' in us_emission_col.lower():
        print(f"  Converting US emission from lbs to kg/h...")
        df_us['emission_kgh'] = pd.to_numeric(df_us[us_emission_col], errors='coerce') * LBS_TO_KG
        us_emission_col = 'emission_kgh'
    
    # Set emission labels based on units
    us_emission_label = 'NOx Emission (kg/h)' if us_is_hourly else 'NOx Emission (tons/year)'
    gl_emission_label = 'NOx Emission (kg/h)' if gl_is_hourly else 'NOx Emission (tons/year)'
    print(f"✓ Emission units: US='{us_emission_label}', Global='{gl_emission_label}'")
    
    # Surface albedo columns
    us_albedo_col = resolve_feature_col(df_us, ['surface_albedo', 'surface_albedo_nitrogendioxide_window'])
    gl_albedo_col = resolve_feature_col(df_gl, ['surface_albedo', 'surface_albedo_nitrogendioxide_window'])
    print(f"✓ Albedo: US='{us_albedo_col}', Global='{gl_albedo_col}'")
    
    # Wind speed columns
    us_wind_col = resolve_feature_col(df_us, ['wind_speed', 'wind_speed_10m', '10m_wind_speed'])
    gl_wind_col = resolve_feature_col(df_gl, ['wind_speed', 'wind_speed_10m', '10m_wind_speed'])
    print(f"✓ Wind: US='{us_wind_col}', Global='{gl_wind_col}'")
    
    # ═══════════════════════════════════════════════════════════
    #  CREATE 2x2 GRID PLOT
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CREATING 2D HEATMAPS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor='white')
    fig.suptitle('Probability of Detection Analysis: 2D Feature Interactions', 
                fontsize=24, fontweight='bold', y=0.98)
    
    # Row 1: NOx Emission vs Surface Albedo
    print("\n[1/4] Computing: US - NOx Emission vs Surface Albedo")
    pod_grid, count_grid, x_edges, y_edges = compute_2d_pod_heatmap(
        df_us, us_emission_col, us_albedo_col, tgt_us, log_x=True
    )
    plot_2d_heatmap(axes[0, 0], pod_grid, count_grid, x_edges, y_edges,
                   title='U.S. - NOx Emission vs Surface Albedo',
                   xlabel=us_emission_label,
                   ylabel='Surface Albedo',
                   log_x=True)
    
    print("[2/4] Computing: Global - NOx Emission vs Surface Albedo")
    pod_grid, count_grid, x_edges, y_edges = compute_2d_pod_heatmap(
        df_gl, gl_emission_col, gl_albedo_col, tgt_gl, log_x=True
    )
    plot_2d_heatmap(axes[0, 1], pod_grid, count_grid, x_edges, y_edges,
                   title='Global - NOx Emission vs Surface Albedo',
                   xlabel=gl_emission_label,
                   ylabel='Surface Albedo',
                   log_x=True)
    
    # Row 2: NOx Emission vs Wind Speed
    print("[3/4] Computing: US - NOx Emission vs Wind Speed")
    pod_grid, count_grid, x_edges, y_edges = compute_2d_pod_heatmap(
        df_us, us_emission_col, us_wind_col, tgt_us, log_x=True
    )
    plot_2d_heatmap(axes[1, 0], pod_grid, count_grid, x_edges, y_edges,
                   title='U.S. - NOx Emission vs Wind Speed',
                   xlabel=us_emission_label,
                   ylabel='Wind Speed (m/s)',
                   log_x=True)
    
    print("[4/4] Computing: Global - NOx Emission vs Wind Speed")
    pod_grid, count_grid, x_edges, y_edges = compute_2d_pod_heatmap(
        df_gl, gl_emission_col, gl_wind_col, tgt_gl, log_x=True
    )
    plot_2d_heatmap(axes[1, 1], pod_grid, count_grid, x_edges, y_edges,
                   title='Global - NOx Emission vs Wind Speed',
                   xlabel=gl_emission_label,
                   ylabel='Wind Speed (m/s)',
                   log_x=True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    
    # Save figure
    output_pdf = 'us_global_pod_2d_heatmap_analysis.pdf'
    output_png = 'us_global_pod_2d_heatmap_analysis.png'
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_pdf}")
    print(f"✓ Saved: {output_png}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
