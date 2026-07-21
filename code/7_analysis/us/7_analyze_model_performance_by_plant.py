"""
Analyze trained model performance by power plant and identify key features.

This script:
1. Evaluates per-plant prediction accuracy
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
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
from sklearn.neighbors import BallTree
from haversine import haversine, Unit
from math import radians, log10
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Interference zone constants (from reference plotting code)
EARTH_RADIUS_KM = 6371.0
PLANT_RADIUS_BASE_KM = 20.0
PLANT_MAX_SEARCH_KM = 150.0
CITY_POP_THRESHOLD = 200000
CITY_RADIUS_SCALE = 9.0
CITY_RADIUS_BASE_KM = 10.0
CITY_RADIUS_MAX_KM = 90.0

# Helper functions for interference calculation (from reference plotting code)
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

def identify_us_interference_one_year(plants_year_df, cities_df):
    source_plants_df, plant_tree = _process_source_dataframe(
        plants_year_df, 'Latitude', 'Longitude',
        id_col='Facility ID', value_col='NOx Mass (short tons)'
    )
    source_cities_df, city_tree = _process_source_dataframe(
        cities_df, 'latitude', 'longitude',
        id_col='name', value_col='population'
    )
    
    plant_interfered_ids, city_interfered_ids = set(), set()
    
    if plants_year_df.empty:
        return plant_interfered_ids, city_interfered_ids
    
    for _, tgt in plants_year_df.iterrows():
        tid = tgt['Facility ID']
        tlat = tgt['Latitude']; tlon = tgt['Longitude']
        temis = tgt.get('NOx Mass (short tons)', 0)
        if pd.isna(tlat) or pd.isna(tlon) or pd.isna(temis):
            continue
        target_coords_rad = np.array([[radians(tlat), radians(tlon)]])
        
        if plant_tree is not None:
            nearby_idx = plant_tree.query_radius(target_coords_rad, r=(PLANT_MAX_SEARCH_KM / EARTH_RADIUS_KM))[0]
            for idx in nearby_idx:
                src = source_plants_df.iloc[idx]
                if src['ID'] == tid:
                    continue
                src_emis = src.get('value', 0)
                if pd.isna(src_emis) or src_emis < temis:
                    continue
                dist_km = haversine((tlat, tlon), (src['latitude'], src['longitude']), unit=Unit.KILOMETERS)
                if dist_km < PLANT_RADIUS_BASE_KM:
                    plant_interfered_ids.add(tid)
                    break
        
        if city_tree is not None:
            nearby_idx = city_tree.query_radius(target_coords_rad, r=(CITY_RADIUS_MAX_KM / EARTH_RADIUS_KM))[0]
            for idx in nearby_idx:
                c = source_cities_df.iloc[idx]
                pop = c.get('value', 0)
                if pd.isna(pop) or pop < CITY_POP_THRESHOLD:
                    continue
                radius = CITY_RADIUS_BASE_KM + (log10(max(1, pop)) * CITY_RADIUS_SCALE)
                interference_radius_km = min(radius, CITY_RADIUS_MAX_KM)
                dist_km = haversine((tlat, tlon), (c['latitude'], c['longitude']), unit=Unit.KILOMETERS)
                if dist_km < interference_radius_km:
                    city_interfered_ids.add(tid)
                    break
    
    return plant_interfered_ids, city_interfered_ids

# Constants and features (must match training script)
FEATURES = [
    'surface_altitude', 'surface_altitude_precision',
    'surface_classification', 'surface_pressure', 'surface_albedo',
    'surface_albedo_nitrogendioxide_window', 'cloud_pressure_crb',
    'cloud_fraction_crb', 'cloud_albedo_crb', 'scene_albedo',
    'apparent_scene_pressure', 'snow_ice_flag', 'aerosol_index_354_388', 
    'scaled_small_pixel_variance', 'sensor_altitude', 'sensor_azimuth_angle', 'sensor_zenith_angle', 
    'solar_azimuth_angle', 'solar_zenith_angle', 'wind_speed', 't2m', 'tisr', 'tcwv',
    'primary_fuel_type', 'NOx Mass (lbs)']

# Model definition (must match training script)
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
            
            plant_metrics[int(plant_id)] = metrics
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
    importance_dict = {feature_names[i]: float(gradients[i]) for i in range(len(feature_names))}
    
    return importance_dict

def calculate_feature_statistics_by_performance(df, plant_metrics, feature_names):
    """Calculate mean feature values for high vs low performing plants."""
    
    # Categorize plants by performance (using F1 score)
    f1_scores = {plant_id: metrics['f1'] for plant_id, metrics in plant_metrics.items()}
    median_f1 = np.median(list(f1_scores.values()))
    
    high_performers = [plant_id for plant_id, f1 in f1_scores.items() if f1 >= median_f1]
    low_performers = [plant_id for plant_id, f1 in f1_scores.items() if f1 < median_f1]
    
    # Calculate mean feature values
    feature_comparison = {}
    
    for feature in feature_names:
        if feature not in df.columns:
            continue
            
        high_mean = df[df['location'].isin(high_performers)][feature].mean()
        low_mean = df[df['location'].isin(low_performers)][feature].mean()
        diff = high_mean - low_mean
        
        feature_comparison[feature] = {
            'high_performers_mean': float(high_mean),
            'low_performers_mean': float(low_mean),
            'difference': float(diff),
            'relative_difference': float(diff / (low_mean + 1e-10) * 100)
        }
    
    return feature_comparison, high_performers, low_performers

def main():
    print("="*60)
    print("POWER PLANT PREDICTION ANALYSIS")
    print("="*60)
    
    # Paths
    model_path = '/net/fs06/d3/rzhuang/TROPOMI_US/model/best_model_hourly_all_data_filtered_yearly_no_stats.pt'
    data_path = '/net/fs06/d3/rzhuang/TROPOMI_US/data/Run_20250623_203825/updated_tropomi_hourly_emissions_full_variables.csv'
    power_plants_path = '/net/fs06/d3/rzhuang/TROPOMI_US/data/facility_emissions_by_plant_comprehensive.csv'
    annual_emissions_path = '/net/fs06/d3/rzhuang/TROPOMI_US/data/annual-emissions-facility-aggregation-2019-2024.csv'
    cities_path = '/net/fs06/d3/rzhuang/TROPOMI_world/data/worldcities.csv'
    output_dir = '/net/fs06/d3/rzhuang/TROPOMI_US/results/'
    
    # Load data
    print("\n1. Loading data...")
    tropomi_df = pd.read_csv(data_path).dropna()
    power_plants_df = pd.read_csv(power_plants_path)
    annual_emissions_df = pd.read_csv(annual_emissions_path)
    cities_df = pd.read_csv(cities_path)
    
    # Sort by NOx_Rank and get top 500
    print("   Selecting top 500 plants by NOx_Rank...")
    if 'NOx_Rank' in power_plants_df.columns:
        power_plants_df = power_plants_df.sort_values('NOx_Rank', ascending=True)
    top500_ids = set(power_plants_df.head(500)['Facility_ID'])
    print(f"   Top 500 plants selected: {len(top500_ids)}")
    
    # Filter annual emissions to top 500
    annual_top500 = annual_emissions_df[annual_emissions_df['Facility ID'].isin(top500_ids)].copy()
    
    # Keep only plants present in ALL 6 years (2019-2024)
    US_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
    present_counts = (annual_top500[annual_top500['Year'].isin(US_YEARS)]
                      .groupby('Facility ID')['Year'].nunique())
    complete_6y_ids = set(present_counts[present_counts == len(US_YEARS)].index)
    print(f"   Plants present in all 6 years: {len(complete_6y_ids)}")
    
    # Calculate interference zones year by year (same as reference plotting code)
    print("   Calculating interference zones year by year...")
    annual_top500 = annual_top500.dropna(subset=['Latitude', 'Longitude'])
    
    all_year_interfered = set()
    print("\n   ================ US Year-by-Year Interference ================")
    for yr in US_YEARS:
        df_y = annual_top500[(annual_top500['Year'] == yr) & 
                            (annual_top500['Facility ID'].isin(complete_6y_ids))].copy()
        if df_y.empty:
            continue
        plant_set, city_set = identify_us_interference_one_year(df_y, cities_df)
        union_set = set(map(str, plant_set.union(city_set)))
        all_year_interfered |= union_set
        print(f"   [{yr}] Plants interfered by plants: {len(plant_set)}; by cities: {len(city_set)}; unique union: {len(union_set)}")
    
    # Convert complete_6y_ids to strings for comparison (match reference code)
    complete_6y_ids_str = set(map(str, complete_6y_ids))
    
    # Final kept IDs: top 500 + in all 6 years + never interfered
    # This matches the reference code logic exactly
    kept_ids_str = complete_6y_ids_str - all_year_interfered
    kept_ids = set(int(x) for x in kept_ids_str)  # Convert back to int for filtering
    
    print(f"   US plants removed due to interference in any year: {len(all_year_interfered & complete_6y_ids_str)}")
    print(f"   US plants for analysis (6y & non-interfered): {len(kept_ids)}")
    
    # Filter TROPOMI data to kept IDs
    print("\n2. Filtering TROPOMI data...")
    print(f"   Total observations (before filtering): {len(tropomi_df)}")
    print(f"   Unique plants (before filtering): {tropomi_df['location'].nunique()}")
    
    tropomi_df['location'] = tropomi_df['location'].astype(int)
    tropomi_df = tropomi_df[tropomi_df['location'].isin(kept_ids)]
    
    print(f"   Total observations (after filtering): {len(tropomi_df)}")
    print(f"   Unique plants (after filtering): {tropomi_df['location'].nunique()}")
    
    print(f"\n   Total observations (after filtering): {len(tropomi_df)}")
    print(f"   Unique plants (after filtering): {tropomi_df['location'].nunique()}")
    
    # Encode primary_fuel_type (same as training)
    print("\n3. Encoding categorical variables...")
    le = LabelEncoder()
    tropomi_df['primary_fuel_type'] = le.fit_transform(tropomi_df['primary_fuel_type'])
    
    # Prepare features
    X = tropomi_df[FEATURES].values.astype(np.float32)
    y = tropomi_df["plume_label"].astype(int).values
    plant_ids = tropomi_df['location'].values
    
    # Scale features (using all data for consistency)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Load model
    print("\n4. Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = MLP(len(FEATURES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("   Model loaded successfully")
    
    # Calculate per-plant metrics
    print("\n5. Calculating per-plant prediction metrics...")
    plant_metrics = calculate_per_plant_metrics(model, X_scaled, y, plant_ids, device)
    print(f"   Analyzed {len(plant_metrics)} plants with sufficient data (>= 5 obs, both classes)")
    
    # Calculate gradient-based feature importance
    print("\n6. Calculating feature importance (gradient-based)...")
    # Use a sample for gradient calculation (faster)
    sample_size = min(10000, len(X_scaled))
    sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    gradient_importance = calculate_gradient_feature_importance(
        model, X_scaled[sample_idx], y[sample_idx], FEATURES, device
    )
    
    # Calculate feature statistics by performance
    print("\n5. Analyzing feature differences between high/low performers...")
    feature_comparison, high_performers, low_performers = calculate_feature_statistics_by_performance(
        tropomi_df, plant_metrics, FEATURES
    )
    
    print(f"   High performers: {len(high_performers)} plants")
    print(f"   Low performers: {len(low_performers)} plants")
    
    # Merge with power plant information
    print("\n6. Merging with power plant characteristics...")
    plant_metrics_df = pd.DataFrame.from_dict(plant_metrics, orient='index')
    plant_metrics_df['Facility_ID'] = plant_metrics_df.index
    
    # Merge with power plant data
    plant_analysis_df = plant_metrics_df.merge(
        power_plants_df, 
        left_on='Facility_ID', 
        right_on='Facility_ID', 
        how='left'
    )
    
    # Sort by F1 score
    plant_analysis_df = plant_analysis_df.sort_values('f1', ascending=False)
    
    # Save results
    print("\n9. Saving results...")
    
    # Save per-plant metrics
    plant_analysis_df.to_csv(f'{output_dir}per_plant_performance_metrics.csv', index=False)
    print(f"   Saved: per_plant_performance_metrics.csv")
    
    # Save feature importance
    with open(f'{output_dir}feature_importance_gradient.json', 'w') as f:
        json.dump(gradient_importance, f, indent=2)
    print(f"   Saved: feature_importance_gradient.json")
    
    # Save feature comparison
    with open(f'{output_dir}feature_comparison_high_vs_low_performers.json', 'w') as f:
        json.dump(feature_comparison, f, indent=2)
    print(f"   Saved: feature_comparison_high_vs_low_performers.json")
    
    # Save high/low performer lists
    performer_info = {
        'high_performers': [int(x) for x in high_performers],
        'low_performers': [int(x) for x in low_performers],
        'median_f1': float(np.median([m['f1'] for m in plant_metrics.values()]))
    }
    with open(f'{output_dir}high_low_performers.json', 'w') as f:
        json.dump(performer_info, f, indent=2)
    print(f"   Saved: high_low_performers.json")
    
    # Create summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    print("\n--- TOP 10 BEST PREDICTED PLANTS (by F1 score) ---")
    top_10 = plant_analysis_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"\nPlant ID: {row['Facility_ID']}")
        if 'Facility_Name' in row and pd.notna(row['Facility_Name']):
            print(f"  Name: {row['Facility_Name']}")
        print(f"  F1 Score: {row['f1']:.4f}")
        print(f"  Accuracy: {row['accuracy']:.4f}")
        print(f"  AUC: {row['auc']:.4f}")
        print(f"  Observations: {row['n_observations']}")
        if 'NOx Mass (lbs)' in row and pd.notna(row['NOx Mass (lbs)']):
            print(f"  NOx Emissions: {row['NOx Mass (lbs)']:.2f} lbs")
    
    print("\n--- TOP 10 WORST PREDICTED PLANTS (by F1 score) ---")
    bottom_10 = plant_analysis_df.tail(10)
    for idx, row in bottom_10.iterrows():
        print(f"\nPlant ID: {row['Facility_ID']}")
        if 'Facility_Name' in row and pd.notna(row['Facility_Name']):
            print(f"  Name: {row['Facility_Name']}")
        print(f"  F1 Score: {row['f1']:.4f}")
        print(f"  Accuracy: {row['accuracy']:.4f}")
        print(f"  AUC: {row['auc']:.4f}")
        print(f"  Observations: {row['n_observations']}")
        if 'NOx Mass (lbs)' in row and pd.notna(row['NOx Mass (lbs)']):
            print(f"  NOx Emissions: {row['NOx Mass (lbs)']:.2f} lbs")
    
    print("\n--- TOP 10 MOST IMPORTANT FEATURES (by gradient) ---")
    sorted_features = sorted(gradient_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        print(f"{i:2d}. {feature:<40s}: {importance:.6f}")
    
    print("\n--- FEATURES WITH LARGEST DIFFERENCES (High vs Low Performers) ---")
    sorted_diff = sorted(
        feature_comparison.items(), 
        key=lambda x: abs(x[1]['relative_difference']), 
        reverse=True
    )
    for i, (feature, stats) in enumerate(sorted_diff[:10], 1):
        print(f"\n{i:2d}. {feature}")
        print(f"    High performers mean: {stats['high_performers_mean']:.4f}")
        print(f"    Low performers mean:  {stats['low_performers_mean']:.4f}")
        print(f"    Relative difference:  {stats['relative_difference']:.2f}%")
    
    print("\n" + "="*60)
    print("Analysis complete! Check the results directory for detailed outputs.")
    print("="*60)

if __name__ == '__main__':
    main()
