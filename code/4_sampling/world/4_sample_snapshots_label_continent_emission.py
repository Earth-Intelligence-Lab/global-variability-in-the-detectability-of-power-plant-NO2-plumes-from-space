import pandas as pd
import numpy as np
import netCDF4 as nc
from sklearn.neighbors import BallTree
from plotting import process_zoomed_data
import matplotlib.pyplot as plt
import concurrent.futures
import os
from tqdm import tqdm
import seaborn as sns

# config.py - Centralized configuration for TROPOMI plume detection
class TROPOMIConfig:
    """Centralized configuration for TROPOMI NO2 plume detection and visualization."""
    
    # --- Plume Detection Parameters ---
    PLUME_DETECTION = {
        'zoom_radius_km': 100,                # Radius for zooming around the plant
        'threshold_factor': 2.0,              # Factor for anomaly threshold calculation
        'threshold_abs_min': 5e-6,            # Minimum absolute threshold for NO2 anomalies
        'max_distance_km': 20.0,                 # Maximum distance for plume detection
        'close_distance_km': 5.0,               # Close distance for relaxed angle criteria
        'max_angle_diff': 25.0,                 # Maximum angle difference for plume cone
        'flagged_area': 25.0,                 # Minimum area (km²) for significant plume
        'stat_radius': 50.0,                  # Radius for NO2 statistics calculation
        'threshold_radius_km': 50.0,          # Radius for local threshold calculation
    }
    
    # --- Background Estimation Parameters ---
    BACKGROUND = {
        'mode': 'directional',                # 'directional' or 'gaussian'
        'upwind_angle_tolerance': 60,         # Angle tolerance for upwind sector
        'dist_min_km': 10,                    # Minimum distance for background
        'dist_max_km': 100,                    # Maximum distance for background
        'gaussian_sigma': 10,                 # Sigma for Gaussian filter
    }
    
    PLANT_MASK = {
        'max_angle_diff_mask': 0,            # Maximum angle for interference mask
        'close_distance_km_mask': 20,          # Close distance for interference mask
    }
    
    # --- Interference Source Parameters ---
    INTERFERENCE = {
        'max_distance_km': 150,               # Maximum distance to consider interference
        
        # City interference parameters
        'city': {
            'base_radius': 0.0,               # Base radius for cities
            'pop_scale': 9.0,                   # Population scaling factor
            'radius_min': 10.0,               # Minimum city interference radius
            'radius_max': 90.0,               # Maximum city interference radius
            'min_population': 200000,         # Minimum population threshold
        },
        
        # Plant interference parameters
        'plant': {
            'base_radius': 0.0,               # Base radius for plants
            'emission_scale': 0.0,            # Emission scaling factor
            'radius_min': 0.0,               # Minimum plant interference radius
            'radius_max': 0.0,               # Maximum plant interference radius
            'min_emission_threshold': 1.0,    # Minimum emission threshold (relative)
            'use_emission_scaling': True,     # Whether to scale by emissions
        }
    }
    
    # --- Visualization Parameters ---
    VISUALIZATION = {
        'plot_dpi': 200,                      # DPI for saved figures
        'plot_interference_zones': True,      # Whether to plot interference zones
        'nearby_plant_radius_km': 200,        # Radius for plotting nearby plants
        'basemap_zoom': 'auto',               # Basemap zoom level
        'colormap_no2': 'viridis',            # Colormap for NO2 concentration
        'colormap_anomaly': 'coolwarm',       # Colormap for anomalies
    }
    
    # --- Data Processing Parameters ---
    PROCESSING = {
        'min_city_population': 50000,         # Minimum city population for loading
        'locations_subset_size': 6000,        # Number of locations for interference
    }
    
    # --- Sampling Parameters ---
    SAMPLING = {
        'n_samples': 400,                     # Total number of samples to process
        'n_emission_bins': 5,                 # Number of emission bins for stratification
        'random_state': 345,                  # Random state for reproducibility
        'country_col': 'country',             # Column name for country
        'emission_col': 'annual_nox_emission', # Column name for emissions
    }
    
    @classmethod
    def get_plume_params(cls):
        """Get parameters for plume detection function."""
        return {
            'zoom_radius_km': cls.PLUME_DETECTION['zoom_radius_km'],
            'threshold_factor': cls.PLUME_DETECTION['threshold_factor'],
            'threshold_abs_min': cls.PLUME_DETECTION['threshold_abs_min'],
            'max_distance_km': cls.PLUME_DETECTION['max_distance_km'],
            'close_distance_km': cls.PLUME_DETECTION['close_distance_km'],
            'max_angle_diff': cls.PLUME_DETECTION['max_angle_diff'],
            'flagged_area': cls.PLUME_DETECTION['flagged_area'],
            'threshold_radius_km': cls.PLUME_DETECTION['threshold_radius_km'],
            
            'max_angle_diff_mask': cls.PLANT_MASK['max_angle_diff_mask'],
            'close_distance_km_mask': cls.PLANT_MASK['close_distance_km_mask'],
            
            'background_mode': cls.BACKGROUND['mode'],
            'upwind_angle_tolerance': cls.BACKGROUND['upwind_angle_tolerance'],
            'background_dist_min_km': cls.BACKGROUND['dist_min_km'],
            'background_dist_max_km': cls.BACKGROUND['dist_max_km'],
            
            'interf_max_distance_km': cls.INTERFERENCE['max_distance_km'],
            'interf_city_pop_thresh': cls.INTERFERENCE['city']['min_population'],
            'interf_plant_emis_thresh': cls.INTERFERENCE['plant']['min_emission_threshold'],
            
            'city_base_radius': cls.INTERFERENCE['city']['base_radius'],
            'city_pop_scale': cls.INTERFERENCE['city']['pop_scale'],
            'city_radius_min': cls.INTERFERENCE['city']['radius_min'],
            'city_radius_max': cls.INTERFERENCE['city']['radius_max'],
            
            'plant_base_radius': cls.INTERFERENCE['plant']['base_radius'],
            'plant_emission_scale': cls.INTERFERENCE['plant']['emission_scale'],
            'plant_radius_min': cls.INTERFERENCE['plant']['radius_min'],
            'plant_radius_max': cls.INTERFERENCE['plant']['radius_max'],
            
            'sigma': cls.BACKGROUND['gaussian_sigma'],
            'stat_radius': cls.PLUME_DETECTION['stat_radius'],
        }
    
    @classmethod
    def get_interference_params(cls):
        """Get parameters for interference calculation."""
        return {
            'max_distance_km': cls.INTERFERENCE['max_distance_km'],
            'city_base_radius': cls.INTERFERENCE['city']['base_radius'],
            'city_pop_scale': cls.INTERFERENCE['city']['pop_scale'],
            'city_radius_min': cls.INTERFERENCE['city']['radius_min'],
            'city_radius_max': cls.INTERFERENCE['city']['radius_max'],
            'min_city_population_threshold': cls.INTERFERENCE['city']['min_population'],
            'plant_base_radius': cls.INTERFERENCE['plant']['base_radius'],
            'plant_emission_scale': cls.INTERFERENCE['plant']['emission_scale'],
            'plant_radius_min': cls.INTERFERENCE['plant']['radius_min'],
            'plant_radius_max': cls.INTERFERENCE['plant']['radius_max'],
            'min_plant_emission_threshold': cls.INTERFERENCE['plant']['min_emission_threshold'],
            'use_plant_emission_scaling': cls.INTERFERENCE['plant']['use_emission_scaling'],
        }
    
    @classmethod
    def get_plotting_params(cls):
        """Get all parameters needed for process_zoomed_data."""
        params = cls.get_plume_params()
        params.update({
            'plot_dpi': cls.VISUALIZATION['plot_dpi'],
            'plot_interference_zones': cls.VISUALIZATION['plot_interference_zones'],
        })
        # Remove 'stat_radius' as it's not used in process_zoomed_data
        params.pop('stat_radius', None)
        return params
    
    @classmethod
    def update_params(cls, **kwargs):
        """Update configuration parameters dynamically."""
        for key, value in kwargs.items():
            if key == 'plume_detection' and isinstance(value, dict):
                cls.PLUME_DETECTION.update(value)
            elif key == 'background' and isinstance(value, dict):
                cls.BACKGROUND.update(value)
            elif key == 'interference' and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key in cls.INTERFERENCE:
                        if isinstance(cls.INTERFERENCE[sub_key], dict):
                            cls.INTERFERENCE[sub_key].update(sub_value)
                        else:
                            cls.INTERFERENCE[sub_key] = sub_value
            elif key == 'visualization' and isinstance(value, dict):
                cls.VISUALIZATION.update(value)
            elif key == 'processing' and isinstance(value, dict):
                cls.PROCESSING.update(value)
            elif key == 'sampling' and isinstance(value, dict):
                cls.SAMPLING.update(value)
            else:
                # Try to update individual parameters by searching all categories
                updated = False
                for category in [cls.PLUME_DETECTION, cls.BACKGROUND, cls.VISUALIZATION, 
                               cls.PROCESSING, cls.SAMPLING]:
                    if key in category:
                        category[key] = value
                        updated = True
                        break
                if not updated:
                    print(f"Warning: Parameter '{key}' not found in configuration.")
    
    @classmethod
    def print_config(cls):
        """Print current configuration in a readable format."""
        import json
        config = {
            'PLUME_DETECTION': cls.PLUME_DETECTION,
            'BACKGROUND': cls.BACKGROUND,
            'INTERFERENCE': cls.INTERFERENCE,
            'VISUALIZATION': cls.VISUALIZATION,
            'PROCESSING': cls.PROCESSING,
            'SAMPLING': cls.SAMPLING,
        }
        print(json.dumps(config, indent=2))

# --- File Paths ---
LOCATION_DATA_PATH = '/net/fs06/d3/rzhuang/TROPOMI_world/data/power_plant_location/power_plants_with_combined_nearby_stats.csv'
SNAPSHOT_DATA_PATH = '/net/fs06/d3/rzhuang/TROPOMI_world/data/Run_3/updated_tropomi_emissions_full_variables_with_fuel.csv'
CITY_DATA_PATH = '/net/fs06/d3/rzhuang/TROPOMI_world/data/worldcities.csv'
OUTPUT_FIGURE_DIR = '/net/fs06/d3/rzhuang/TROPOMI_world/code/figure_snapshots'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

# Print current configuration
print("Current Configuration:")
TROPOMIConfig.print_config()

# --- City Data Loading ---
min_city_population = TROPOMIConfig.PROCESSING['min_city_population']
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

# --- Load Location and Snapshot Data ---
print(f"Loading location data from: {LOCATION_DATA_PATH}")
try:
    locations_df = pd.read_csv(LOCATION_DATA_PATH)
    print(f"Location data loaded. Shape: {locations_df.shape}")
    if 'ID' not in locations_df.columns or 'nox_emis_ty' not in locations_df.columns:
        raise ValueError("Required columns 'ID' or 'nox_emis_ty' not found")
except Exception as e:
    print(f"Error loading location data: {e}")
    exit()

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
    
except Exception as e:
    print(f"Error loading or processing data: {e}")
    exit()

# --- Stratified Sampling Function ---
def sample_emission_snapshots_stratified(df_path: str = None, config: type = TROPOMIConfig) -> pd.DataFrame:
    """
    Sample emission snapshots stratified by emissions and continents.
    """
    # ISO3 to continent mapping
    iso3_to_continent = {
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
        'NFK': 'Oceania', 'MNP': 'Oceania', 'TKL': 'Oceania', 'WLF': 'Oceania'
    }
    
    # Load the dataset
    if df_path is None:
        df_path = '/net/fs06/d3/rzhuang/TROPOMI_world/data/Run_3/updated_tropomi_emissions_full_variables_with_fuel.csv'
    
    print(f"Loading dataset from: {df_path}")
    try:
        df = pd.read_csv(df_path, low_memory=False)
        print(f"Dataset loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()
    
    # Check for required columns
    required_cols = ['annual_nox_emission', 'ISO3']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Try alternative column names
        if 'country' in df.columns and 'ISO3' not in df.columns:
            df['ISO3'] = df['country']
        if 'nox_emis_ty' in df.columns and 'annual_nox_emission' not in df.columns:
            df['annual_nox_emission'] = df['nox_emis_ty']
        
        # Check again
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found: {missing_cols}")
    
    # Add continent column
    df['continent'] = df['ISO3'].map(iso3_to_continent)
    unknown_iso3 = df[df['continent'].isna()]['ISO3'].unique()
    if len(unknown_iso3) > 0:
        print(f"Warning: Unknown ISO3 codes found: {unknown_iso3[:10]}...")
        df.loc[df['continent'].isna(), 'continent'] = 'Unknown'
    
    # Print continent distribution
    print(f"\nContinent distribution in full dataset:")
    print(df['continent'].value_counts())
    
    # Filter out invalid emissions
    df = df[df['annual_nox_emission'] > 0].copy()
    print(f"\nDataset after filtering positive emissions: {len(df)} rows")
    
    # Create emission bins
    try:
        df['emission_bin'] = pd.qcut(df['annual_nox_emission'], 
                                     q=config.SAMPLING['n_emission_bins'], 
                                     labels=[f'E{i+1}' for i in range(config.SAMPLING['n_emission_bins'])],
                                     duplicates='drop')
    except Exception as e:
        print(f"Error creating emission bins with qcut: {e}")
        # Fallback to fixed bins
        df['emission_bin'] = pd.cut(df['annual_nox_emission'], 
                                   bins=config.SAMPLING['n_emission_bins'], 
                                   labels=[f'E{i+1}' for i in range(config.SAMPLING['n_emission_bins'])],
                                   include_lowest=True)
    
    # Get unique combinations of continent and emission bin
    strata = df.groupby(['continent', 'emission_bin']).size().reset_index(name='count')
    strata = strata[strata['count'] > 0]
    n_strata = len(strata)
    
    print(f"\nFound {n_strata} unique strata (continent x emission bin combinations)")
    
    # Calculate samples per stratum
    n_total = config.SAMPLING['n_samples']
    base_per_stratum = max(1, n_total // n_strata)
    remainder = n_total - (base_per_stratum * n_strata)
    
    # Sample from each stratum
    sampled_dfs = []
    np.random.seed(config.SAMPLING['random_state'])
    
    for idx, (_, stratum) in enumerate(strata.iterrows()):
        stratum_df = df[(df['continent'] == stratum['continent']) & 
                       (df['emission_bin'] == stratum['emission_bin'])]
        
        # Add extra sample to first strata if remainder exists
        n_samples = base_per_stratum + (1 if idx < remainder else 0)
        n_samples = min(n_samples, len(stratum_df))
        
        if n_samples > 0:
            sampled = stratum_df.sample(n_samples, random_state=config.SAMPLING['random_state'] + idx)
            sampled_dfs.append(sampled)
            print(f"Sampled {n_samples} from {stratum['continent']} - {stratum['emission_bin']}")
    
    # Combine all samples
    sampled_df = pd.concat(sampled_dfs, ignore_index=False) if sampled_dfs else pd.DataFrame()
    
    print(f"\n--- Final Sample Summary ---")
    print(f"Total samples: {len(sampled_df)}")
    print(f"\nContinent distribution:")
    print(sampled_df['continent'].value_counts())
    print(f"\nEmission bin distribution:")
    if 'emission_bin' in sampled_df.columns:
        print(sampled_df['emission_bin'].value_counts())
    
    return sampled_df

# --- Call the stratified sampling function ---
try:
    # Use the stratified sampling function
    sampled_df = sample_emission_snapshots_stratified()
    
    if not sampled_df.empty:
        print("\n--- Stratified Sample Verification ---")
        print(f"Number of samples obtained: {len(sampled_df)}")
        print(f"Columns in sampled data: {list(sampled_df.columns)}")
        
        # Save the sampled data with stratification info
        output_csv_path = os.path.join(OUTPUT_FIGURE_DIR, 'stratified_sampled_emission_snapshots.csv')
        sampled_df.to_csv(output_csv_path, index=False)
        print(f"Saved stratified sample data to: {output_csv_path}")
        
        # Create a summary report
        summary_path = os.path.join(OUTPUT_FIGURE_DIR, 'stratification_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=== Stratified Sampling Summary ===\n\n")
            
            # Overall statistics
            f.write(f"Total samples: {len(sampled_df)}\n")
            f.write(f"Date: {pd.Timestamp.now()}\n\n")
            
            # Continent breakdown
            f.write("Continent Distribution:\n")
            f.write(sampled_df['continent'].value_counts().to_string())
            f.write("\n\n")
            
            # Emission bin breakdown
            if 'emission_bin' in sampled_df.columns:
                f.write("Emission Bin Distribution:\n")
                f.write(sampled_df['emission_bin'].value_counts().to_string())
                f.write("\n\n")
            
            # Emission statistics by continent
            f.write("Emission Statistics by Continent:\n")
            for continent in sampled_df['continent'].unique():
                cont_df = sampled_df[sampled_df['continent'] == continent]
                if 'annual_nox_emission' in cont_df.columns:
                    f.write(f"\n{continent}:\n")
                    f.write(f"  Mean: {cont_df['annual_nox_emission'].mean():.2f}\n")
                    f.write(f"  Median: {cont_df['annual_nox_emission'].median():.2f}\n")
                    f.write(f"  Min: {cont_df['annual_nox_emission'].min():.2f}\n")
                    f.write(f"  Max: {cont_df['annual_nox_emission'].max():.2f}\n")
            
            # Cross-tabulation
            if 'emission_bin' in sampled_df.columns:
                f.write("\n\nCross-tabulation: Continent x Emission Bin\n")
                cross_tab = pd.crosstab(sampled_df['continent'], sampled_df['emission_bin'])
                f.write(cross_tab.to_string())
                f.write("\n")
        
        print(f"Saved stratification summary to: {summary_path}")
        
        # If the column names are different, map them to expected names
        if 'location' not in sampled_df.columns and 'plant_id' in sampled_df.columns:
            sampled_df['location'] = sampled_df['plant_id']
        if 'country' not in sampled_df.columns and 'iso_code' in sampled_df.columns:
            sampled_df['country'] = sampled_df['iso_code']
        if 'utc_time' not in sampled_df.columns and 'time' in sampled_df.columns:
            sampled_df['utc_time'] = sampled_df['time']
        
        # Create visualization of the stratification
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Stratified Sampling Distribution', fontsize=16)
        
        # 1. Continent distribution
        ax = axes[0, 0]
        continent_counts = sampled_df['continent'].value_counts()
        continent_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Samples by Continent')
        ax.set_xlabel('Continent')
        ax.set_ylabel('Count')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Emission bin distribution
        ax = axes[0, 1]
        if 'emission_bin' in sampled_df.columns:
            emission_counts = sampled_df['emission_bin'].value_counts().sort_index()
            emission_counts.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title('Samples by Emission Bin')
            ax.set_xlabel('Emission Bin')
            ax.set_ylabel('Count')
        
        # 3. Emission distribution (log scale)
        ax = axes[1, 0]
        if 'annual_nox_emission' in sampled_df.columns:
            ax.hist(np.log10(sampled_df['annual_nox_emission'] + 1), bins=30, color='lightgreen', edgecolor='black')
            ax.set_xlabel('log10(NOx Emissions + 1)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of NOx Emissions (log scale)')
        
        # 4. Heatmap of continent vs emission bin
        ax = axes[1, 1]
        if 'emission_bin' in sampled_df.columns:
            cross_tab = pd.crosstab(sampled_df['continent'], sampled_df['emission_bin'])
            sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
            ax.set_title('Continent vs Emission Bin')
            ax.set_xlabel('Emission Bin')
            ax.set_ylabel('Continent')
        
        plt.tight_layout()
        vis_path = os.path.join(OUTPUT_FIGURE_DIR, 'stratification_visualization.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved stratification visualization to: {vis_path}")
        
    else:
        print("Stratified sampling resulted in an empty DataFrame. Exiting.")
        exit()
        
except Exception as e:
    print(f"\nAn error occurred during stratified sampling: {e}")
    import traceback
    traceback.print_exc()
    exit()

# === PARALLEL PROCESSING SECTION ===

def process_and_save_row(args):
    """
    Processes a single row from the sampled DataFrame and saves the plot.
    """
    row_tuple, row_index, locations_df_global, world_cities_df_global, plot_params, output_dir = args
    row = pd.Series(row_tuple, index=row_index)
    fig = None
    location_id = row.get('location', 'UnknownLocation')
    iso_code = row.get('country', 'UnknownISO')
    time = row.get('utc_time', 'UnknownTime')
    continent = row.get('continent', 'Unknown')
    emission_bin = row.get('emission_bin', 'Unknown')
    save_path = os.path.join(output_dir, f"location_{location_id}_{iso_code}_{time}_{continent}_{emission_bin}.png")
    
    try:
        # Ensure necessary dataframes are valid before passing
        loc_df_copy = locations_df_global.copy() if locations_df_global is not None else pd.DataFrame()
        city_df_copy = world_cities_df_global.copy() if world_cities_df_global is not None else pd.DataFrame()
        
        # Call the main processing function
        fig = process_zoomed_data(
            row=row,
            global_locations_df=loc_df_copy,
            cities_df=city_df_copy,
            **plot_params  # Use parameters from config
        )
        
        # Save the figure if generated successfully
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

# --- Main Execution Block ---
if __name__ == "__main__":
    # Use subset of locations_df for interference checking
    locations_subset_size = TROPOMIConfig.PROCESSING['locations_subset_size']
    
    if 'ID' in locations_df.columns:
        locations_df_subset = locations_df.iloc[:locations_subset_size].copy()
        print(f"\nUsing subset of locations_df (first {locations_subset_size} rows) for interference checks: Shape={locations_df_subset.shape}")
    else:
        print("\nWarning: 'ID' column not found in locations_df. Using full locations_df for interference checks, which might be slow.")
        locations_df_subset = locations_df.copy()
    
    # Get plotting parameters from configuration
    plotting_parameters = TROPOMIConfig.get_plotting_params()
    
    # Prepare Arguments for Parallel Processing
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
    
    # Determine Number of Workers
    max_workers = os.cpu_count()
    print(f"\nStarting parallel plot generation using up to {max_workers} workers...")
    
    # Execute in Parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_and_save_row, args_list), total=len(args_list)))
    
    # Report Results
    successful_plots = [res for res in results if res is not None]
    failed_count = len(results) - len(successful_plots)
    
    print(f"\n--- Parallel Processing Complete ---")
    print(f"Successfully generated {len(successful_plots)} plots.")
    if failed_count > 0:
        print(f"Failed to generate plots for {failed_count} locations (see error messages above).")
    
    # Create summary report by stratification
    print("\n--- Stratification Summary ---")
    for continent in sampled_df['continent'].unique():
        continent_df = sampled_df[sampled_df['continent'] == continent]
        continent_results = [r for r, row in zip(results, sampled_df.itertuples()) 
                           if row.continent == continent and r is not None]
        print(f"{continent}: {len(continent_results)}/{len(continent_df)} plots generated successfully")
    
    print("\nScript finished.")