import pandas as pd
import numpy as np
import netCDF4 as nc
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
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
OUTPUT_FIGURE_DIR = '/net/fs06/d3/rzhuang/TROPOMI_world/code/figure_snapshots_stratified/'

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

# --- Train/Val/Test Split Functions ---
def stratified_train_val_test_split(df, config, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split dataset into train, validation, and test sets with stratification.
    
    Modified order:
    1. First extract validation set
    2. Then split remaining into train and test
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    config : TROPOMIConfig
        Configuration object
    train_ratio : float
        Proportion for training set (default: 0.7)
    val_ratio : float
        Proportion for validation set (default: 0.15)
    test_ratio : float
        Proportion for test set (default: 0.15)
    
    Returns:
    --------
    train_df, val_df, test_df : tuple of pd.DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
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
        'GRL': 'North America', 'SPM': 'North America', 'CUW': 'North America',
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
    
    # Add continent column
    df['continent'] = df['ISO3'].map(iso3_to_continent)
    df.loc[df['continent'].isna(), 'continent'] = 'Unknown'
    
    # Filter out invalid emissions
    df = df[df['annual_nox_emission'] > 0].copy()
    
    # Create emission bins
    try:
        df['emission_bin'] = pd.qcut(df['annual_nox_emission'], 
                                     q=config.SAMPLING['n_emission_bins'], 
                                     labels=[f'E{i+1}' for i in range(config.SAMPLING['n_emission_bins'])],
                                     duplicates='drop')
    except:
        df['emission_bin'] = pd.cut(df['annual_nox_emission'], 
                                   bins=config.SAMPLING['n_emission_bins'], 
                                   labels=[f'E{i+1}' for i in range(config.SAMPLING['n_emission_bins'])],
                                   include_lowest=True)
    
    # Create stratification column
    df['stratify_col'] = df['continent'] + '_' + df['emission_bin'].astype(str)
    
    # Remove strata with too few samples
    strata_counts = df['stratify_col'].value_counts()
    valid_strata = strata_counts[strata_counts >= 3].index  # Need at least 3 samples per stratum
    df_filtered = df[df['stratify_col'].isin(valid_strata)].copy()
    
    print(f"Filtered dataset: {len(df_filtered)} rows (removed {len(df) - len(df_filtered)} rows from small strata)")
    
    # MODIFIED: First split - extract validation set
    print(f"\n--- Step 1: Extracting validation set ({val_ratio*100:.0f}%) ---")
    train_test_df, val_df = train_test_split(
        df_filtered, 
        test_size=val_ratio, 
        stratify=df_filtered['stratify_col'],
        random_state=config.SAMPLING['random_state']
    )
    print(f"Validation set: {len(val_df)} samples")
    print(f"Remaining (train+test): {len(train_test_df)} samples")
    
    # MODIFIED: Second split - split remaining into train and test
    print(f"\n--- Step 2: Splitting remaining into train and test ---")
    # Calculate relative test size from the remaining data
    relative_test_size = test_ratio / (train_ratio + test_ratio)
    train_df, test_df = train_test_split(
        train_test_df, 
        test_size=relative_test_size, 
        stratify=train_test_df['stratify_col'],
        random_state=config.SAMPLING['random_state'] + 1
    )
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Print split statistics
    print(f"\n--- Dataset Split Statistics ---")
    print(f"Total samples: {len(df_filtered)}")
    print(f"Train set: {len(train_df)} ({100*len(train_df)/len(df_filtered):.1f}%)")
    print(f"Val set: {len(val_df)} ({100*len(val_df)/len(df_filtered):.1f}%)")
    print(f"Test set: {len(test_df)} ({100*len(test_df)/len(df_filtered):.1f}%)")
    
    # Verify stratification
    print("\n--- Stratification Verification ---")
    for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{name} set continent distribution:")
        print(subset['continent'].value_counts())
    
    return train_df, val_df, test_df

def sample_from_test_set(test_df, n_samples, config):
    """
    Sample from the test set maintaining stratification.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataset
    n_samples : int
        Number of samples to draw
    config : TROPOMIConfig
        Configuration object
    
    Returns:
    --------
    sampled_df : pd.DataFrame
        Sampled data from test set
    """
    # Ensure we don't sample more than available
    n_samples = min(n_samples, len(test_df))
    
    # Get unique strata
    strata = test_df.groupby(['continent', 'emission_bin']).size().reset_index(name='count')
    strata = strata[strata['count'] > 0]
    n_strata = len(strata)
    
    print(f"\nSampling {n_samples} samples from test set ({len(test_df)} total)")
    print(f"Found {n_strata} unique strata in test set")
    
    # Calculate samples per stratum
    base_per_stratum = max(1, n_samples // n_strata)
    remainder = n_samples - (base_per_stratum * n_strata)
    
    # Sample from each stratum
    sampled_dfs = []
    np.random.seed(config.SAMPLING['random_state'] + 42)  # Different seed for test sampling
    
    for idx, (_, stratum) in enumerate(strata.iterrows()):
        stratum_df = test_df[(test_df['continent'] == stratum['continent']) & 
                            (test_df['emission_bin'] == stratum['emission_bin'])]
        
        # Add extra sample to first strata if remainder exists
        n_stratum_samples = base_per_stratum + (1 if idx < remainder else 0)
        n_stratum_samples = min(n_stratum_samples, len(stratum_df))
        
        if n_stratum_samples > 0:
            sampled = stratum_df.sample(n_stratum_samples, 
                                      random_state=config.SAMPLING['random_state'] + 42 + idx)
            sampled_dfs.append(sampled)
    
    # Combine all samples
    sampled_df = pd.concat(sampled_dfs, ignore_index=False) if sampled_dfs else pd.DataFrame()
    
    print(f"\n--- Test Set Sample Summary ---")
    print(f"Total samples drawn: {len(sampled_df)}")
    print(f"\nContinent distribution:")
    print(sampled_df['continent'].value_counts())
    print(f"\nEmission bin distribution:")
    print(sampled_df['emission_bin'].value_counts())
    
    return sampled_df


def perform_train_val_test_split_and_sample(df_path, config, output_dir):
    """
    Main function to perform train/val/test split and sample from test set.
    """
    # Load the full dataset
    print(f"Loading dataset from: {df_path}")
    df = pd.read_csv(df_path, low_memory=False)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # Handle column name variations
    if 'country' in df.columns and 'ISO3' not in df.columns:
        df['ISO3'] = df['country']
    if 'nox_emis_ty' in df.columns and 'annual_nox_emission' not in df.columns:
        df['annual_nox_emission'] = df['nox_emis_ty']
    
    # Perform stratified train/val/test split
    train_df, val_df, test_df = stratified_train_val_test_split(
        df, 
        config,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    # Save the splits
    train_path = os.path.join(output_dir, 'train_set.csv')
    val_path = os.path.join(output_dir, 'val_set.csv')
    test_path = os.path.join(output_dir, 'test_set.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved splits to:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")
    
    # Sample from test set
    test_samples = sample_from_test_set(
        test_df, 
        n_samples=config.SAMPLING['n_samples'],
        config=config
    )
    
    # Save test samples
    test_samples_path = os.path.join(output_dir, 'test_samples.csv')
    test_samples.to_csv(test_samples_path, index=False)
    print(f"\nSaved {len(test_samples)} test samples to: {test_samples_path}")
    
    # Create split summary
    summary_path = os.path.join(output_dir, 'split_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== Train/Val/Test Split Summary ===\n\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"After filtering: {len(train_df) + len(val_df) + len(test_df)}\n\n")
        
        f.write("Split sizes:\n")
        f.write(f"  Train: {len(train_df)} ({100*len(train_df)/(len(train_df)+len(val_df)+len(test_df)):.1f}%)\n")
        f.write(f"  Val: {len(val_df)} ({100*len(val_df)/(len(train_df)+len(val_df)+len(test_df)):.1f}%)\n")
        f.write(f"  Test: {len(test_df)} ({100*len(test_df)/(len(train_df)+len(val_df)+len(test_df)):.1f}%)\n\n")
        
        f.write("Test set samples:\n")
        f.write(f"  Total in test set: {len(test_df)}\n")
        f.write(f"  Samples drawn: {len(test_samples)}\n\n")
        
        # Cross-tabulation for each split
        for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df), ("Test Samples", test_samples)]:
            if 'emission_bin' in subset.columns:
                f.write(f"\n{name} - Continent x Emission Bin:\n")
                cross_tab = pd.crosstab(subset['continent'], subset['emission_bin'])
                f.write(cross_tab.to_string())
                f.write("\n")
    
    print(f"\nSaved split summary to: {summary_path}")
    
    return train_df, val_df, test_df, test_samples


# --- Perform Train/Val/Test Split and Sample from Test Set ---
try:
    # Perform the split and get test samples
    train_df, val_df, test_df, sampled_df = perform_train_val_test_split_and_sample(
        df_path=SNAPSHOT_DATA_PATH,
        config=TROPOMIConfig,
        output_dir=OUTPUT_FIGURE_DIR
    )
    
    if not sampled_df.empty:
        print("\n--- Test Set Sample Verification ---")
        print(f"Number of test samples obtained: {len(sampled_df)}")
        print(f"Columns in sampled data: {list(sampled_df.columns)}")
        
        # Map column names if necessary
        if 'location' not in sampled_df.columns and 'plant_id' in sampled_df.columns:
            sampled_df['location'] = sampled_df['plant_id']
        if 'country' not in sampled_df.columns and 'iso_code' in sampled_df.columns:
            sampled_df['country'] = sampled_df['iso_code']
        if 'utc_time' not in sampled_df.columns and 'time' in sampled_df.columns:
            sampled_df['utc_time'] = sampled_df['time']
        
        # Create visualization of the splits and test samples
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Train/Val/Test Split and Test Sample Distribution', fontsize=16)
        
        # Plot continent distributions for each split
        for idx, (name, df_split) in enumerate([("Train", train_df), ("Val", val_df), ("Test", test_df)]):
            ax = axes[0, idx]
            continent_counts = df_split['continent'].value_counts()
            continent_counts.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'lightgreen'][idx])
            ax.set_title(f'{name} Set by Continent')
            ax.set_xlabel('Continent')
            ax.set_ylabel('Count')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot emission distributions for each split
        for idx, (name, df_split) in enumerate([("Train", train_df), ("Val", val_df), ("Test", test_df)]):
            ax = axes[1, idx]
            if 'annual_nox_emission' in df_split.columns:
                ax.hist(np.log10(df_split['annual_nox_emission'] + 1), 
                       bins=30, alpha=0.7, edgecolor='black',
                       color=['skyblue', 'lightcoral', 'lightgreen'][idx])
                ax.set_xlabel('log10(NOx Emissions + 1)')
                ax.set_ylabel('Count')
                ax.set_title(f'{name} Set Emission Distribution')
        
        plt.tight_layout()
        split_vis_path = os.path.join(OUTPUT_FIGURE_DIR, 'train_val_test_split_visualization.png')
        plt.savefig(split_vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved split visualization to: {split_vis_path}")
        
        # Create test samples visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Test Set Samples Distribution', fontsize=16)
        
        # 1. Continent distribution comparison: full test vs sampled
        ax = axes[0, 0]
        test_continent = test_df['continent'].value_counts()
        sample_continent = sampled_df['continent'].value_counts()
        x = np.arange(len(test_continent))
        width = 0.35
        
        ax.bar(x - width/2, test_continent.values, width, label='Full Test Set', alpha=0.7)
        ax.bar(x + width/2, sample_continent.reindex(test_continent.index, fill_value=0).values, 
               width, label='Sampled', alpha=0.7)
        ax.set_xlabel('Continent')
        ax.set_ylabel('Count')
        ax.set_title('Test Set vs Sampled: Continent Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(test_continent.index, rotation=45, ha='right')
        ax.legend()
        
        # 2. Emission bin distribution
        ax = axes[0, 1]
        if 'emission_bin' in sampled_df.columns:
            emission_counts = sampled_df['emission_bin'].value_counts().sort_index()
            emission_counts.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title('Test Samples by Emission Bin')
            ax.set_xlabel('Emission Bin')
            ax.set_ylabel('Count')
        
        # 3. Emission distribution (log scale)
        ax = axes[1, 0]
        if 'annual_nox_emission' in sampled_df.columns:
            ax.hist(np.log10(sampled_df['annual_nox_emission'] + 1), 
                   bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
            ax.set_xlabel('log10(NOx Emissions + 1)')
            ax.set_ylabel('Count')
            ax.set_title('Test Samples: NOx Emissions Distribution')
        
        # 4. Heatmap of continent vs emission bin for test samples
        ax = axes[1, 1]
        if 'emission_bin' in sampled_df.columns:
            cross_tab = pd.crosstab(sampled_df['continent'], sampled_df['emission_bin'])
            sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
            ax.set_title('Test Samples: Continent vs Emission Bin')
            ax.set_xlabel('Emission Bin')
            ax.set_ylabel('Continent')
        
        plt.tight_layout()
        test_sample_vis_path = os.path.join(OUTPUT_FIGURE_DIR, 'test_samples_visualization.png')
        plt.savefig(test_sample_vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved test samples visualization to: {test_sample_vis_path}")
        
    else:
        print("Test set sampling resulted in an empty DataFrame. Exiting.")
        exit()
        
except Exception as e:
    print(f"\nAn error occurred during train/val/test splitting and sampling: {e}")
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