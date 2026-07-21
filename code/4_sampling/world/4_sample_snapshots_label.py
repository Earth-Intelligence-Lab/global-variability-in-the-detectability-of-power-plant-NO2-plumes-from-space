import pandas as pd
import numpy as np
import netCDF4 as nc
from sklearn.neighbors import BallTree
from plotting import process_zoomed_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import concurrent.futures
import os
from tqdm import tqdm
import torch
import torch.nn as nn

# Define MLP model class
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# Define features
FEATURES = [
    'annual_nox_emission', 'surface_altitude', 'surface_altitude_precision',
    'surface_classification', 'surface_pressure', 'surface_albedo',
    'surface_albedo_nitrogendioxide_window', 'cloud_pressure_crb',
    'cloud_fraction_crb', 'cloud_albedo_crb', 'scene_albedo',
    'apparent_scene_pressure', 'snow_ice_flag', 'aerosol_index_354_388', 
    'scaled_small_pixel_variance', 'tropospheric_NO2_column_number_density', 
    'sensor_altitude', 'sensor_azimuth_angle', 'sensor_zenith_angle', 
    'solar_azimuth_angle', 'solar_zenith_angle', 'nearby_plants_count_20km', 
    'total_emission_20km', 'percentage_emission_20km', 'nearby_plants_count_50km',
    'total_emission_50km', 'percentage_emission_50km', 'nearby_plants_count_100km', 
    'total_emission_100km', 'percentage_emission_100km', 'nearby_cities_count_20km',
    'nearby_cities_pop_20km', 'nearby_cities_count_50km', 'nearby_cities_pop_50km', 
    'nearby_cities_count_100km', 'nearby_cities_pop_100km', 'wind_speed', 't2m', 'tisr', 'tcwv',
    'no2_mean_radius', 'no2_std_radius', 'no2_frac_valid_radius', 'primary_fuel_type'
]

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
        'n_samples': 500,                     # Number of samples to process
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

# --- ML-Based Sampling Function with MLP ---
def sample_emission_snapshots_ml(df_path: str = None, config: type = TROPOMIConfig) -> pd.DataFrame:
    """
    Sample emission snapshots based on MLP predictions from TEST SET only.
    Selects 25 samples each from TP, FN, TN, FP categories.
    """
    # Load the ML dataset
    if df_path is None:
        df_path = '/net/fs06/d3/rzhuang/TROPOMI_world/data/Run_3/updated_tropomi_emissions_full_variables_with_fuel.csv'
    
    print(f"Loading ML dataset from: {df_path}")
    try:
        ml_df = pd.read_csv(df_path, low_memory=False)
        print(f"ML dataset loaded. Shape: {ml_df.shape}")
    except Exception as e:
        print(f"Error loading ML dataset: {e}")
        return pd.DataFrame()
    
    # Check for required columns
    if 'plume_label' not in ml_df.columns:
        raise ValueError("'plume_label' column not found in ML dataset")
    
    # Prepare features - exactly as in training
    le = LabelEncoder()
    ml_df['primary_fuel_type'] = le.fit_transform(ml_df['primary_fuel_type'])
    print(f"Fuel type mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    X = ml_df[FEATURES].values.astype(np.float32)
    y = ml_df['plume_label'].astype(int).values
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution - Plumes: {(y==1).sum()}, No plumes: {(y==0).sum()}")
    
    # Split data exactly as in training to get test indices
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTest set size: {len(test_idx)}")
    print(f"Test set plume distribution - Plumes: {(y_test==1).sum()}, No plumes: {(y_test==0).sum()}")
    
    # Balance training set (as done during training)
    ros = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
    
    # Scale features - fit on balanced training data, transform test data
    scaler = StandardScaler()
    scaler.fit(X_train_bal)  # Fit on training data only
    X_test_scaled = scaler.transform(X_test)
    
    # Load trained MLP model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MLP(len(FEATURES)).to(device)
    model_path = '/net/fs06/d3/rzhuang/TROPOMI_world/model/best_0.0005_features.pt'
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get predictions on TEST SET only
    print("Getting MLP predictions on test set...")
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 1024
        probs = []
        for i in range(0, len(X_test_scaled), batch_size):
            batch = X_test_scaled[i:i+batch_size]
            X_tensor = torch.from_numpy(batch).to(device)
            batch_probs = torch.sigmoid(model(X_tensor)).cpu().numpy()
            probs.extend(batch_probs)
    
    probs = np.array(probs)
    y_pred = (probs >= 0.5).astype(int).flatten()
    
    # Create confusion matrix categories for TEST SET
    true_labels = y_test
    pred_labels = y_pred
    
    # Get test dataframe
    test_df = ml_df.iloc[test_idx].copy()
    
    # Identify indices for each category within test set
    tp_mask = (true_labels == 1) & (pred_labels == 1)
    fn_mask = (true_labels == 1) & (pred_labels == 0)
    tn_mask = (true_labels == 0) & (pred_labels == 0)
    fp_mask = (true_labels == 0) & (pred_labels == 1)
    
    tp_indices = test_df.index[tp_mask].tolist()
    fn_indices = test_df.index[fn_mask].tolist()
    tn_indices = test_df.index[tn_mask].tolist()
    fp_indices = test_df.index[fp_mask].tolist()
    
    print(f"\nTest Set Confusion Matrix:")
    print(f"True Positives: {len(tp_indices)}")
    print(f"False Negatives: {len(fn_indices)}")
    print(f"True Negatives: {len(tn_indices)}")
    print(f"False Positives: {len(fp_indices)}")
    
    # Calculate test metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    print(f"\nTest Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, probs):.4f}")

    # Sample 100 from each category (or all if less than 100)
    n_samples_per_category = 100
    sampled_indices = []
    
    categories = [
        ('TP', tp_indices),
        ('FN', fn_indices),
        ('TN', tn_indices),
        ('FP', fp_indices)
    ]
    
    np.random.seed(config.SAMPLING['random_state'])
    
    for cat_name, indices in categories:
        if len(indices) >= n_samples_per_category:
            sampled = np.random.choice(indices, n_samples_per_category, replace=False)
        else:
            sampled = indices
            print(f"Warning: Only {len(indices)} samples available for {cat_name}")
        sampled_indices.extend(sampled)
        print(f"Sampled {len(sampled)} from {cat_name}")
    
    # Create sampled dataframe
    sampled_df = ml_df.loc[sampled_indices].copy()
    
    # Add prediction category and probability for reference
    sampled_df['ml_category'] = 'Unknown'
    sampled_df['prediction_probability'] = np.nan
    
    # Map back the predictions
    for idx, (test_idx_val, prob) in enumerate(zip(test_idx, probs)):
        if ml_df.index[test_idx_val] in sampled_indices:
            sampled_df.loc[ml_df.index[test_idx_val], 'prediction_probability'] = prob
    
    sampled_df.loc[sampled_df.index.isin(tp_indices), 'ml_category'] = 'TP'
    sampled_df.loc[sampled_df.index.isin(fn_indices), 'ml_category'] = 'FN'
    sampled_df.loc[sampled_df.index.isin(tn_indices), 'ml_category'] = 'TN'
    sampled_df.loc[sampled_df.index.isin(fp_indices), 'ml_category'] = 'FP'
    
    print(f"\nFinal sampled DataFrame shape: {sampled_df.shape}")
    print(f"Category distribution:")
    print(sampled_df['ml_category'].value_counts())
    
    # Add a flag to indicate these are test samples
    sampled_df['dataset_split'] = 'test'
    
    return sampled_df

# --- Call the ML-based sampling function ---
try:
    # Use the new ML-based sampling function
    sampled_df = sample_emission_snapshots_ml()
    
    if not sampled_df.empty:
        print("\n--- ML-Based Sample Verification ---")
        print(f"Number of samples obtained: {len(sampled_df)}")
        print(f"Columns in sampled data: {list(sampled_df.columns)}")
        
        # Save the sampled data with ML categories
        sampled_df.to_csv(os.path.join(OUTPUT_FIGURE_DIR, 'ml_sampled_emission_snapshots.csv'), index=False)
        
        # If the column names are different, map them to expected names
        if 'location' not in sampled_df.columns and 'plant_id' in sampled_df.columns:
            sampled_df['location'] = sampled_df['plant_id']
        if 'country' not in sampled_df.columns and 'iso_code' in sampled_df.columns:
            sampled_df['country'] = sampled_df['iso_code']
        if 'utc_time' not in sampled_df.columns and 'time' in sampled_df.columns:
            sampled_df['utc_time'] = sampled_df['time']
            
    else:
        print("ML sampling resulted in an empty DataFrame. Exiting.")
        exit()
        
except Exception as e:
    print(f"\nAn error occurred during ML sampling: {e}")
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
    ml_category = row.get('ml_category', 'Unknown')
    save_path = os.path.join(output_dir, f"sampled_location_{location_id}_{iso_code}_{time}_{ml_category}.png")
    
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
    
    # Create summary report by ML category
    if 'ml_category' in sampled_df.columns:
        print("\n--- ML Category Summary ---")
        for category in ['TP', 'FN', 'TN', 'FP']:
            category_df = sampled_df[sampled_df['ml_category'] == category]
            category_results = [r for r, row in zip(results, sampled_df.itertuples()) 
                              if row.ml_category == category and r is not None]
            print(f"{category}: {len(category_results)}/{len(category_df)} plots generated successfully")
    
    print("\nScript finished.")