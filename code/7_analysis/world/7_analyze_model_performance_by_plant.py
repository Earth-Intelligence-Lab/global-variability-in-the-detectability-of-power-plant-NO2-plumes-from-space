"""
Analyze trained GLOBAL model performance by power plant and identify key features.

This script:
1. Evaluates per-plant prediction accuracy for GLOBAL power plants
2. Identifies which plants are predicted more accurately
3. Determines which features contribute to better predictions
4. Correlates plant characteristics with model performance
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import json
from collections import defaultdict
from sklearn.neighbors import BallTree
from haversine import haversine, Unit
from math import radians, log10
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Interference zone constants
EARTH_RADIUS_KM = 6371.0
PLANT_RADIUS_BASE_KM = 20.0
PLANT_MAX_SEARCH_KM = 150.0
CITY_POP_THRESHOLD = 200000
CITY_RADIUS_SCALE = 9.0
CITY_RADIUS_BASE_KM = 10.0
CITY_RADIUS_MAX_KM = 90.0

# Helper functions for interference calculation
def _process_source_dataframe(df_raw, lat_col, lon_col, id_col=None, value_col=None):
    if df_raw is None or df_raw.empty:
        return None, None
    df = df_raw.copy()
    std_lat, std_lon, std_id, std_value = 'latitude', 'longitude', 'ID', 'value'
    col_map = {lat_col: std_lat, lon_col: std_lon}
    if id_col: col_map[id_col] = std_id
    if value_col: col_map[value_col] = std_value
    df = df[list(col_map.keys())].rename(columns=col_map)
    for c in [std_lat, std_lon, std_value]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=[std_lat, std_lon], inplace=True)
    if df.empty:
        return df, None
    df['lat_rad'] = np.radians(df[std_lat])
    df['lon_rad'] = np.radians(df[std_lon])
    tree = BallTree(df[['lat_rad', 'lon_rad']].values, metric='haversine')
    return df, tree

def identify_plants_in_interference_zones_global(plants_df, cities_df):
    df_plants = plants_df.copy()
    if 'nox_emis_ty' in df_plants.columns:
        df_plants['annual_nox_emission'] = df_plants['nox_emis_ty']
    else:
        df_plants['annual_nox_emission'] = -df_plants.index
    
    df_plants['lat_rad'] = np.radians(df_plants['latitude'])
    df_plants['lon_rad'] = np.radians(df_plants['longitude'])
    plant_tree = BallTree(df_plants[['lat_rad', 'lon_rad']].values, metric='haversine')
    
    cities_filtered = cities_df[cities_df['population'] >= CITY_POP_THRESHOLD].copy()
    if not cities_filtered.empty:
        cities_filtered['lat_rad'] = np.radians(cities_filtered['latitude'])
        cities_filtered['lon_rad'] = np.radians(cities_filtered['longitude'])
        city_tree = BallTree(cities_filtered[['lat_rad', 'lon_rad']].values, metric='haversine')
    else:
        city_tree = None
    
    interfered_plants = set()
    
    for idx, tgt in tqdm(df_plants.iterrows(), total=len(df_plants), desc="Checking global interference"):
        tid = tgt['ID']; tlat = tgt['latitude']; tlon = tgt['longitude']
        temis = tgt.get('annual_nox_emission', 0)
        if pd.isna(tlat) or pd.isna(tlon):
            continue
        tgt_rad = np.array([[radians(tlat), radians(tlon)]])
        
        near_idx = plant_tree.query_radius(tgt_rad, r=(PLANT_MAX_SEARCH_KM / EARTH_RADIUS_KM))[0]
        for j in near_idx:
            if j == idx: continue
            src = df_plants.iloc[j]
            semis = src.get('annual_nox_emission', 0)
            if pd.notna(semis) and semis > temis:
                d = haversine((tlat, tlon), (src['latitude'], src['longitude']), unit=Unit.KILOMETERS)
                if d < PLANT_RADIUS_BASE_KM:
                    interfered_plants.add(tid)
                    break
        
        if city_tree and tid not in interfered_plants:
            near_idx = city_tree.query_radius(tgt_rad, r=(CITY_RADIUS_MAX_KM / EARTH_RADIUS_KM))[0]
            for k in near_idx:
                city = cities_filtered.iloc[k]
                pop = city['population']
                radius = CITY_RADIUS_BASE_KM + (log10(max(1, pop)) * CITY_RADIUS_SCALE)
                rr = min(radius, CITY_RADIUS_MAX_KM)
                d = haversine((tlat, tlon), (city['latitude'], city['longitude']), unit=Unit.KILOMETERS)
                if d < rr:
                    interfered_plants.add(tid)
                    break
    
    return interfered_plants

# Constants and features (from global training script)
FEATURES = [
    'annual_nox_emission', 'surface_altitude', 'surface_altitude_precision',
    'surface_classification', 'surface_pressure', 'surface_albedo',
    'surface_albedo_nitrogendioxide_window', 'cloud_pressure_crb',
    'cloud_fraction_crb', 'cloud_albedo_crb', 'scene_albedo',
    'apparent_scene_pressure', 'snow_ice_flag', 'aerosol_index_354_388', 
    'scaled_small_pixel_variance',
    'sensor_altitude', 'sensor_azimuth_angle', 'sensor_zenith_angle', 
    'solar_azimuth_angle', 'solar_zenith_angle', 'wind_speed', 't2m', 'tisr', 'tcwv',
    'primary_fuel_type'
]

# Model definition (same as training script)
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

def calculate_per_plant_metrics(model, X, y, plant_ids, device):
    """Calculate prediction metrics for each plant individually."""
    model.eval()
    
    # Get predictions for all data
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        probs = torch.sigmoid(model(X_tensor)).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    
    # Calculate metrics per plant
    plant_metrics = {}
    unique_plants = np.unique(plant_ids)
    
    for plant_id in unique_plants:
        plant_mask = plant_ids == plant_id
        
        if plant_mask.sum() < 5:  # Skip plants with too few observations
            continue
            
        y_plant = y[plant_mask]
        preds_plant = preds[plant_mask]
        probs_plant = probs[plant_mask]
        
        # Skip if only one class present (can't calculate AUC)
        if len(np.unique(y_plant)) < 2:
            continue
        
        try:
            metrics = {
                'n_observations': int(plant_mask.sum()),
                'n_positive': int(y_plant.sum()),
                'n_negative': int((y_plant == 0).sum()),
                'accuracy': float(accuracy_score(y_plant, preds_plant)),
                'precision': float(precision_score(y_plant, preds_plant, zero_division=0)),
                'recall': float(recall_score(y_plant, preds_plant, zero_division=0)),
                'f1': float(f1_score(y_plant, preds_plant, zero_division=0)),
                'auc': float(roc_auc_score(y_plant, probs_plant))
            }
            
            # Add confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_plant, preds_plant).ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            plant_metrics[str(plant_id)] = metrics  # Keep as string for global
        except Exception as e:
            print(f"Error calculating metrics for plant {plant_id}: {e}")
            continue
    
    return plant_metrics

def calculate_gradient_feature_importance(model, X, y, feature_names, device):
    """Calculate feature importance using gradient-based method."""
    model.eval()
    
    X_tensor = torch.from_numpy(X).to(device).requires_grad_(True)
    y_tensor = torch.from_numpy(y).to(device).float()
    
    # Forward pass
    outputs = model(X_tensor)
    loss = nn.BCEWithLogitsLoss()(outputs, y_tensor)
    
    # Backward pass
    loss.backward()
    
    # Get gradients
    gradients = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
    
    # Create importance dictionary
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, gradients)}
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return importance_dict

def calculate_feature_statistics_by_performance(data_df, plant_metrics, feature_names):
    """Compare feature values between high and low performing plants."""
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(plant_metrics, orient='index')
    metrics_df['plant_id'] = metrics_df.index
    
    # Define high/low performers by F1 score median
    f1_median = metrics_df['f1'].median()
    high_performers = metrics_df[metrics_df['f1'] >= f1_median]['plant_id'].values
    low_performers = metrics_df[metrics_df['f1'] < f1_median]['plant_id'].values
    
    feature_comparison = {}
    
    for feature in feature_names:
        if feature not in data_df.columns:
            continue
            
        high_values = data_df[data_df['location'].isin(high_performers)][feature].dropna()
        low_values = data_df[data_df['location'].isin(low_performers)][feature].dropna()
        
        if len(high_values) == 0 or len(low_values) == 0:
            continue
        
        high_mean = high_values.mean()
        low_mean = low_values.mean()
        diff = high_mean - low_mean
        
        feature_comparison[feature] = {
            'high_mean': float(high_mean),
            'low_mean': float(low_mean),
            'difference': float(diff),
            'relative_difference': float(diff / (low_mean + 1e-10) * 100)
        }
    
    return feature_comparison, high_performers, low_performers

def main():
    print("="*60)
    print("GLOBAL POWER PLANT PREDICTION ANALYSIS")
    print("="*60)
    
    # Paths
    model_path = '/net/fs06/d3/rzhuang/TROPOMI_world/model/best_model_all_data_filtered_no_stats.pt'
    data_path = '/net/fs06/d3/rzhuang/TROPOMI_world/data/Run_3/updated_tropomi_emissions_full_variables_with_fuel.csv'
    power_plants_path = '/net/fs06/d3/rzhuang/TROPOMI_world/data/power_plant_location/power_plants_with_combined_nearby_stats.csv'
    cities_path = '/net/fs06/d3/rzhuang/TROPOMI_world/data/worldcities.csv'
    output_dir = '/net/fs06/d3/rzhuang/TROPOMI_world/results/'
    
    # Load data
    print("\n1. Loading data...")
    tropomi_df = pd.read_csv(data_path, low_memory=False).dropna()
    power_plants_df = pd.read_csv(power_plants_path)
    cities_df = pd.read_csv(cities_path)
    
    print(f"   Loaded TROPOMI data: {len(tropomi_df)} observations")
    print(f"   Unique plants in TROPOMI: {tropomi_df['location'].nunique()}")
    
    print(f"   Loaded TROPOMI data: {len(tropomi_df)} observations")
    print(f"   Unique plants in TROPOMI: {tropomi_df['location'].nunique()}")
    
    # Get all unique plant IDs from TROPOMI data
    print("\n2. Identifying plants to analyze...")
    all_plant_ids = tropomi_df['location'].unique()
    all_plants_for_interference = power_plants_df[power_plants_df['ID'].isin(all_plant_ids)].copy()
    
    print(f"   Total unique plants in TROPOMI: {len(all_plant_ids)}")
    print(f"   Matched in power plants database: {len(all_plants_for_interference)}")
    
    # Calculate interference zones
    print("\n3. Calculating interference zones...")
    global_interfered = identify_plants_in_interference_zones_global(all_plants_for_interference, cities_df)
    
    keep_ids = set(map(str, all_plant_ids)) - set(map(str, global_interfered))
    # Keep as strings for global (IDs like 'CoCO2_12278')
    kept_ids = keep_ids
    
    print(f"   Plants interfered: {len(global_interfered)}")
    print(f"   Plants for analysis (non-interfered): {len(kept_ids)}")
    
    # Filter TROPOMI data to kept IDs
    print("\n4. Filtering TROPOMI data...")
    print(f"   Total observations (before filtering): {len(tropomi_df)}")
    print(f"   Unique plants (before filtering): {tropomi_df['location'].nunique()}")
    
    tropomi_df['location'] = tropomi_df['location'].astype(str)
    tropomi_df = tropomi_df[tropomi_df['location'].isin(kept_ids)]
    
    print(f"   Total observations (after filtering): {len(tropomi_df)}")
    print(f"   Unique plants (after filtering): {tropomi_df['location'].nunique()}")
    
    # Encode primary_fuel_type (already in the data file)
    print("\n5. Encoding categorical variables...")
    le = LabelEncoder()
    tropomi_df['primary_fuel_type'] = le.fit_transform(tropomi_df['primary_fuel_type'].fillna('Unknown').astype(str))
    
    # Prepare features
    print("\n5. Preparing features...")
    X = tropomi_df[FEATURES].values.astype(np.float32)
    y = tropomi_df["plume_label"].astype(int).values
    plant_ids = tropomi_df['location'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Load model
    print("\n6. Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = MLP(len(FEATURES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("   Model loaded successfully")
    
    # Calculate per-plant metrics
    print("\n7. Calculating per-plant prediction metrics...")
    plant_metrics = calculate_per_plant_metrics(model, X_scaled, y, plant_ids, device)
    print(f"   Analyzed {len(plant_metrics)} plants with sufficient data (>= 5 obs, both classes)")
    
    # Calculate gradient-based feature importance
    print("\n8. Calculating feature importance (gradient-based)...")
    sample_size = min(10000, len(X_scaled))
    sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    gradient_importance = calculate_gradient_feature_importance(
        model, X_scaled[sample_idx], y[sample_idx], FEATURES, device
    )
    
    # Calculate feature statistics by performance
    print("\n9. Analyzing feature differences between high/low performers...")
    feature_comparison, high_performers, low_performers = calculate_feature_statistics_by_performance(
        tropomi_df, plant_metrics, FEATURES
    )
    
    print(f"   High performers: {len(high_performers)} plants")
    print(f"   Low performers: {len(low_performers)} plants")
    
    # Merge with power plant information
    print("\n10. Merging with power plant characteristics...")
    plant_metrics_df = pd.DataFrame.from_dict(plant_metrics, orient='index')
    plant_metrics_df['Plant_ID'] = plant_metrics_df.index
    
    # Filter power plants to only those in our analysis
    power_plants_filtered = power_plants_df[power_plants_df['ID'].astype(str).isin(kept_ids)].copy()
    
    # Merge with power plant data
    plant_analysis_df = plant_metrics_df.merge(
        power_plants_filtered, 
        left_on='Plant_ID', 
        right_on='ID', 
        how='left'
    )
    
    # Sort by F1 score
    plant_analysis_df = plant_analysis_df.sort_values('f1', ascending=False)
    
    # Save results
    print("\n11. Saving results...")
    plant_analysis_df.to_csv(f'{output_dir}per_plant_performance_metrics.csv', index=False)
    print(f"   Saved: per_plant_performance_metrics.csv")
    
    with open(f'{output_dir}feature_importance_gradient.json', 'w') as f:
        json.dump(gradient_importance, f, indent=2)
    print(f"   Saved: feature_importance_gradient.json")
    
    with open(f'{output_dir}feature_comparison_high_vs_low_performers.json', 'w') as f:
        json.dump(feature_comparison, f, indent=2)
    print(f"   Saved: feature_comparison_high_vs_low_performers.json")
    
    with open(f'{output_dir}high_low_performers.json', 'w') as f:
        json.dump({
            'high_performers': high_performers.tolist(),
            'low_performers': low_performers.tolist()
        }, f, indent=2)
    print(f"   Saved: high_low_performers.json")
    
    # Print summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    print("\n--- TOP 10 BEST PREDICTED PLANTS (by F1 score) ---\n")
    for idx, row in plant_analysis_df.head(10).iterrows():
        print(f"Plant ID: {row['Plant_ID']}")
        if 'name' in row and pd.notna(row['name']):
            print(f"  Name: {row['name']}")
        if 'country' in row and pd.notna(row['country']):
            print(f"  Country: {row['country']}")
        print(f"  F1 Score: {row['f1']:.4f}")
        print(f"  Accuracy: {row['accuracy']:.4f}")
        print(f"  AUC: {row['auc']:.4f}")
        print(f"  Observations: {row['n_observations']}")
        if 'nox_emis_ty' in row and pd.notna(row['nox_emis_ty']):
            print(f"  NOx Emissions: {row['nox_emis_ty']:,.0f} tons/year")
        print()
    
    print("\n--- FEATURE IMPORTANCE (Top 10) ---\n")
    for i, (feature, importance) in enumerate(list(gradient_importance.items())[:10], 1):
        print(f"{i:2d}. {feature:40s}: {importance:.6f}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()
