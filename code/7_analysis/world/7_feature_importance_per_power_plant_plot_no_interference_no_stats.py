import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import shap
from joblib import Parallel, delayed
from sklearn.neighbors import BallTree
from haversine import haversine
from math import radians, log10
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- Interference Zone Constants ---
EARTH_RADIUS_KM = 6371
PLANT_RADIUS_BASE_KM = 20.0
CITY_POP_THRESHOLD = 200000
CITY_RADIUS_SCALE = 9.0
CITY_RADIUS_BASE_KM = 10.0
CITY_RADIUS_MAX_KM = 90.0

# Model definition
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

# --- Interference Zone Functions ---
def identify_plants_in_interference_zones_world(plants_df, cities_df, plant_subset_ids=None):
    """
    Identifies which plants are in interference zones of other plants or cities.
    Adapted for world data structure.
    """
    print("Identifying plants in interference zones (global)...")
    
    # Filter to subset if provided
    if plant_subset_ids is not None:
        plants_df = plants_df[plants_df['location'].isin(plant_subset_ids)].copy()
        print(f"Checking interference for {len(plants_df)} plants in subset")
    
    # Get unique plants for interference check
    unique_plants = plants_df.groupby('location').first().reset_index()
    
    # Ensure we have the required columns
    if 'annual_nox_emission' not in unique_plants.columns:
        if 'nox_emis_ty' in unique_plants.columns:
            unique_plants['annual_nox_emission'] = unique_plants['nox_emis_ty']
        else:
            print("Warning: No annual_nox_emission column found, using index as proxy")
            unique_plants['annual_nox_emission'] = -unique_plants.index
    
    # Build BallTree for plants
    unique_plants['lat_rad'] = np.radians(unique_plants['latitude'])
    unique_plants['lon_rad'] = np.radians(unique_plants['longitude'])
    plant_tree = BallTree(unique_plants[['lat_rad', 'lon_rad']].values, metric='haversine')
    
    # Build BallTree for cities
    cities_filtered = cities_df[cities_df['population'] >= CITY_POP_THRESHOLD].copy()
    cities_filtered['lat_rad'] = np.radians(cities_filtered['latitude'])
    cities_filtered['lon_rad'] = np.radians(cities_filtered['longitude'])
    
    if len(cities_filtered) > 0:
        city_tree = BallTree(cities_filtered[['lat_rad', 'lon_rad']].values, metric='haversine')
    else:
        city_tree = None
        print("Warning: No cities above population threshold")
    
    interfered_plants = set()
    
    for idx, target_plant in tqdm(unique_plants.iterrows(), total=len(unique_plants), desc="Checking interference"):
        target_id = target_plant['location']
        target_lat = target_plant['latitude']
        target_lon = target_plant['longitude']
        target_emissions = target_plant.get('annual_nox_emission', 0)
        
        if pd.isna(target_lat) or pd.isna(target_lon):
            continue
            
        target_coords_rad = np.array([[radians(target_lat), radians(target_lon)]])
        
        # Check interference from other plants
        search_radius = PLANT_RADIUS_BASE_KM / EARTH_RADIUS_KM
        nearby_plant_indices = plant_tree.query_radius(target_coords_rad, r=search_radius)[0]
        
        for plant_idx in nearby_plant_indices:
            if plant_idx == idx:  # Skip self
                continue
            source_plant = unique_plants.iloc[plant_idx]
            source_emissions = source_plant.get('annual_nox_emission', 0)
            
            # Only interfered if source has higher emissions
            if pd.notna(source_emissions) and source_emissions > target_emissions:
                distance_km = haversine(
                    (target_lat, target_lon),
                    (source_plant['latitude'], source_plant['longitude'])
                )
                if distance_km < PLANT_RADIUS_BASE_KM:
                    interfered_plants.add(target_id)
                    break
        
        # Check interference from cities
        if city_tree and target_id not in interfered_plants:
            search_radius = CITY_RADIUS_MAX_KM / EARTH_RADIUS_KM
            nearby_city_indices = city_tree.query_radius(target_coords_rad, r=search_radius)[0]
            
            for city_idx in nearby_city_indices:
                source_city = cities_filtered.iloc[city_idx]
                population = source_city['population']
                
                # Calculate city interference radius
                radius = CITY_RADIUS_BASE_KM + (log10(max(1, population)) * CITY_RADIUS_SCALE)
                interference_radius_km = min(radius, CITY_RADIUS_MAX_KM)
                
                distance_km = haversine(
                    (target_lat, target_lon),
                    (source_city['latitude'], source_city['longitude'])
                )
                
                if distance_km < interference_radius_km:
                    interfered_plants.add(target_id)
                    break
    
    print(f"Found {len(interfered_plants)} plants in interference zones out of {len(unique_plants)} checked")
    return interfered_plants

# Parallel processing function for a single plant
def process_single_plant_global(plant_data, explainer, features):
    """
    Process a single plant's SHAP analysis with location info
    """
    plant_samples, plant_id, plant_y, plant_X_original, lat, lon = plant_data
    
    # Sample if too many observations
    if len(plant_samples) > 50:
        idx = np.random.choice(len(plant_samples), 50, replace=False)
        plant_samples = plant_samples[idx]
        plant_y = plant_y[idx]
        plant_X_original = plant_X_original[idx]
    
    try:
        # Get SHAP values
        shap_values = explainer.shap_values(plant_samples)
        
        # Get predictions
        with torch.no_grad():
            predictions = explainer.expected_value + shap_values.sum(axis=1)
            predictions = 1 / (1 + np.exp(-predictions))  # Sigmoid
        
        # Calculate average contributions
        avg_shap = shap_values.mean(axis=0)
        abs_avg_shap = np.abs(avg_shap)
        
        # Calculate percentage contributions
        total_contribution = abs_avg_shap.sum()
        if total_contribution > 0:
            percentage_contributions = (abs_avg_shap / total_contribution) * 100
        else:
            percentage_contributions = np.zeros_like(abs_avg_shap)
        
        # Get top contributors
        top_indices = np.argsort(percentage_contributions)[::-1][:10]
        
        # Determine region
        if -130 <= lon <= -60 and 25 <= lat <= 70:
            region = 'North America'
        elif -10 <= lon <= 40 and 35 <= lat <= 70:
            region = 'Europe'
        elif 60 <= lon <= 150 and 0 <= lat <= 60:
            region = 'Asia'
        elif -80 <= lon <= -35 and -55 <= lat <= 15:
            region = 'South America'
        elif -20 <= lon <= 50 and -35 <= lat <= 35:
            region = 'Africa'
        elif 110 <= lon <= 180 and -45 <= lat <= -10:
            region = 'Oceania'
        else:
            region = 'Other'
        
        # Store results
        plant_result = {
            'plant_id': plant_id,
            'latitude': lat,
            'longitude': lon,
            'region': region,
            'true_plume': plant_y[0],
            'avg_prediction': predictions.mean(),
            'num_observations': len(plant_samples)
        }
        
        # Add top 10 feature contributions
        for rank, idx in enumerate(top_indices):
            plant_result[f'top_{rank+1}_feature'] = features[idx]
            plant_result[f'top_{rank+1}_contribution_pct'] = percentage_contributions[idx]
            plant_result[f'top_{rank+1}_shap_value'] = avg_shap[idx]
            plant_result[f'top_{rank+1}_direction'] = 'increases' if avg_shap[idx] > 0 else 'decreases'
            plant_result[f'top_{rank+1}_avg_value'] = plant_X_original[:, idx].mean()
        
        return plant_result
        
    except Exception as e:
        print(f"Error processing plant {plant_id}: {e}")
        return None

def analyze_all_global_plants(data, features, model_path, batch_size=6000, interfered_plants=None):
    """
    Analyze ALL plants globally using parallel processing, with optional interference filtering
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Filter out interfered plants if provided
    if interfered_plants is not None:
        print(f"Filtering out {len(interfered_plants)} interfered plants from analysis...")
        data = data[~data['location'].isin(interfered_plants)].copy()
        print(f"Remaining plants after filtering: {data['location'].nunique()}")
    
    # Prepare data
    le = LabelEncoder()
    data['primary_fuel_type'] = le.fit_transform(data['primary_fuel_type'])
    X = data[features].to_numpy(dtype=np.float32)
    y = data['plume_label'].to_numpy(dtype=int)
    plant_ids = data['location'].to_numpy()
    latitudes = data['latitude'].to_numpy()
    longitudes = data['longitude'].to_numpy()
    
    # Split and scale
    X_train, X_test, y_train, y_test, pid_train, pid_test, lat_train, lat_test, lon_train, lon_test = train_test_split(
        X, y, plant_ids, latitudes, longitudes, test_size=0.2, random_state=42
    )
    
    ros = RandomOverSampler(random_state=42)
    X_tr_bal, y_tr_bal = ros.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_bal)
    X_te_s = scaler.transform(X_test)
    
    # Load model
    model = MLP(len(features)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Model wrapper
    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(device)
            output = torch.sigmoid(model(x_tensor))
            return output.cpu().numpy()
    
    # Create explainer
    n_background = min(100, X_tr_s.shape[0])
    background = X_tr_s[np.random.choice(X_tr_s.shape[0], n_background, replace=False)]
    explainer = shap.KernelExplainer(model_predict, background)
    
    # Get all unique plants
    unique_plants = np.unique(pid_test)
    total_plants = len(unique_plants)
    
    print(f"Starting analysis of {total_plants} non-interfered plants globally...")
    
    # Prepare plant data for parallel processing
    plant_data_list = []
    for plant_id in unique_plants:
        plant_mask = pid_test == plant_id
        plant_samples = X_te_s[plant_mask]
        plant_y = y_test[plant_mask]
        plant_X_original = X_test[plant_mask]
        plant_lat = lat_test[plant_mask][0]
        plant_lon = lon_test[plant_mask][0]
        
        if len(plant_samples) > 0:
            plant_data_list.append((plant_samples, plant_id, plant_y, plant_X_original, plant_lat, plant_lon))
    
    # Process in batches for memory efficiency
    all_results = []
    n_batches = (len(plant_data_list) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(plant_data_list))
        batch_data = plant_data_list[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{n_batches} (plants {start_idx + 1}-{end_idx}/{total_plants})")
        
        # Parallel processing of batch
        batch_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_single_plant_global)(plant_data, explainer, features) 
            for plant_data in batch_data
        )
        
        # Filter out None results and add to all results
        batch_results = [r for r in batch_results if r is not None]
        all_results.extend(batch_results)
        
        print(f"Completed batch {batch_idx + 1}: {len(batch_results)} plants processed successfully")
    
    print(f"\nTotal plants successfully analyzed: {len(all_results)}/{total_plants}")
    
    return pd.DataFrame(all_results), scaler, explainer

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading global power plant data...")
    data = pd.read_csv('/net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m/Run_100m_20260428/updated_tropomi_emissions_full_variables_with_fuel_100mlabel.csv').dropna()
    
    print(f"Total records in dataset: {len(data)}")
    print(f"Initial unique power plants: {data['location'].nunique()}")
    
    # Load cities data for interference checking
    print("\nLoading cities data for interference checking...")
    try:
        # You'll need to provide the path to your cities dataset
        cities_df = pd.read_csv('/net/fs06/d3/rzhuang/TROPOMI/data/world/worldcities.csv')
        print(f"Loaded {len(cities_df)} cities")
    except:
        print("Warning: Could not load cities data. Creating empty cities dataframe.")
        cities_df = pd.DataFrame(columns=['latitude', 'longitude', 'population'])
    
    # Identify interfered plants
    print("\n" + "="*80)
    print("STEP 1: IDENTIFYING INTERFERED PLANTS")
    print("="*80)
    interfered_plants = identify_plants_in_interference_zones_world(data, cities_df)
    
    print(f"\nInterference Summary:")
    print(f"  Total unique plants: {data['location'].nunique()}")
    print(f"  Interfered plants: {len(interfered_plants)}")
    print(f"  Non-interfered plants: {data['location'].nunique() - len(interfered_plants)}")
    print(f"  Interference rate: {len(interfered_plants)/data['location'].nunique()*100:.1f}%")
    
    # Save list of interfered plants
    interfered_df = pd.DataFrame({
        'plant_id': list(interfered_plants),
        'interfered': True
    })
    interfered_df.to_csv('/net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m/Run_100m_20260428/training_no_stats_item/feature_importance/all_global_plants_contributions_filtered.csv', index=False)
    print(f"\nSaved list of interfered plants to all_global_plants_contributions_filtered.csv")

    # Define features (no NOx Mass for global)
    features = np.array([
        'annual_nox_emission', 'surface_altitude', 'surface_altitude_precision',
        'surface_classification', 'surface_pressure', 'surface_albedo',
        'surface_albedo_nitrogendioxide_window', 'cloud_pressure_crb',
        'cloud_fraction_crb', 'cloud_albedo_crb', 'scene_albedo',
        'apparent_scene_pressure', 'snow_ice_flag', 'aerosol_index_354_388', 
        'scaled_small_pixel_variance',
        'sensor_altitude', 'sensor_azimuth_angle', 'sensor_zenith_angle', 
        'solar_azimuth_angle', 'solar_zenith_angle', 'wind_speed', 't2m', 'tisr', 'tcwv',
        'primary_fuel_type'
    ])
    
    # Run analysis on non-interfered plants only
    print("\n" + "="*80)
    print("STEP 2: ANALYZING NON-INTERFERED PLANTS")
    print("="*80)
    plant_df, scaler, explainer = analyze_all_global_plants(
        data, features, 
        '/net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m/Run_100m_20260428/training_no_stats_item/best_model_all_data_filtered.pt',
        batch_size=6000,
        interfered_plants=interfered_plants  # Pass the interfered plants to filter them out
    )
    
    # Save results
    output_file = '/net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m/Run_100m_20260428/training_no_stats_item/feature_importance/non_interfered_plants_contributions.csv'
    plant_df.to_csv(output_file, index=False)
    print(f"\nSaved results for {len(plant_df)} non-interfered plants to {output_file}")
    
    # Comprehensive global analysis
    print("\n" + "="*100)
    print("COMPREHENSIVE ANALYSIS OF NON-INTERFERED POWER PLANTS")
    print("="*100)
    
    # Overall statistics
    print(f"\nTotal non-interfered plants analyzed: {len(plant_df)}")
    print(f"Plants with visible plumes: {len(plant_df[plant_df['true_plume']==1])} ({len(plant_df[plant_df['true_plume']==1])/len(plant_df)*100:.1f}%)")
    print(f"Plants without visible plumes: {len(plant_df[plant_df['true_plume']==0])} ({len(plant_df[plant_df['true_plume']==0])/len(plant_df)*100:.1f}%)")
    
    # Regional breakdown
    print("\n" + "-"*80)
    print("REGIONAL BREAKDOWN (NON-INTERFERED PLANTS ONLY)")
    print("-"*80)
    print(f"{'Region':<20} {'Total Plants':>15} {'With Plume':>15} {'Plume Rate %':>15}")
    print("-"*80)
    
    for region in plant_df['region'].unique():
        region_data = plant_df[plant_df['region'] == region]
        plume_count = len(region_data[region_data['true_plume'] == 1])
        plume_rate = plume_count / len(region_data) * 100 if len(region_data) > 0 else 0
        print(f"{region:<20} {len(region_data):>15} {plume_count:>15} {plume_rate:>15.1f}")
    
    # Feature importance statistics
    print("\n" + "-"*80)
    print("GLOBAL FEATURE IMPORTANCE STATISTICS (NON-INTERFERED PLANTS)")
    print("-"*80)
    
    # Collect feature statistics
    feature_stats = {}
    for _, row in plant_df.iterrows():
        for i in range(1, 6):  # Top 5 features
            if f'top_{i}_feature' in row:
                feature = row[f'top_{i}_feature']
                contribution = row[f'top_{i}_contribution_pct']
                region = row['region']
                
                if feature not in feature_stats:
                    feature_stats[feature] = {
                        'contributions': [],
                        'regional_contributions': {},
                        'appearances': 0,
                        'as_top_1': 0,
                        'as_top_3': 0
                    }
                
                feature_stats[feature]['contributions'].append(contribution)
                feature_stats[feature]['appearances'] += 1
                
                if region not in feature_stats[feature]['regional_contributions']:
                    feature_stats[feature]['regional_contributions'][region] = []
                feature_stats[feature]['regional_contributions'][region].append(contribution)
                
                if i == 1:
                    feature_stats[feature]['as_top_1'] += 1
                if i <= 3:
                    feature_stats[feature]['as_top_3'] += 1
    
    # Calculate summary statistics
    summary_stats = []
    for feature, stats in feature_stats.items():
        if len(stats['contributions']) >= 20:  # Only features that appear frequently
            # Regional averages
            regional_avgs = {}
            for region, contribs in stats['regional_contributions'].items():
                if len(contribs) >= 5:
                    regional_avgs[region] = np.mean(contribs)
            
            summary_stats.append({
                'feature': feature,
                'mean_contribution_global': np.mean(stats['contributions']),
                'std_contribution': np.std(stats['contributions']),
                'min_contribution': np.min(stats['contributions']),
                'max_contribution': np.max(stats['contributions']),
                'appearances': stats['appearances'],
                'times_as_top_1': stats['as_top_1'],
                'pct_plants_top_1': stats['as_top_1'] / len(plant_df) * 100,
                **{f'mean_{region.replace(" ", "_")}': avg for region, avg in regional_avgs.items()}
            })
    
    stats_summary_df = pd.DataFrame(summary_stats).sort_values('mean_contribution_global', ascending=False)
    
    print("\nTop 15 Features (Non-Interfered Plants):")
    print(f"{'Feature':<45} {'Global%':>8} {'Std%':>8} {'Range':>15} {'Top1%':>8}")
    print("-"*85)
    for _, row in stats_summary_df.head(15).iterrows():
        print(f"{row['feature']:<45} {row['mean_contribution_global']:>8.1f} {row['std_contribution']:>8.1f} "
              f"[{row['min_contribution']:>5.1f}-{row['max_contribution']:>5.1f}] "
              f"{row['pct_plants_top_1']:>7.1f}")
    
    # Save detailed statistics
    stats_summary_df.to_csv('/net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m/Run_100m_20260428/training_no_stats_item/feature_importance/all_global_plants_feature_filtered_statistics.csv', index=False)
    
    print("\n" + "="*100)
    print("NON-INTERFERED PLANTS ANALYSIS COMPLETE!")
    print("="*100)
    print(f"\nFiles saved:")
    print(f"- interfered_plants.csv: List of {len(interfered_plants)} interfered plants")
    print(f"- non_interfered_plants_contributions.csv: Individual contributions for {len(plant_df)} non-interfered plants")
    print(f"- non_interfered_plants_feature_statistics.csv: Feature statistics for non-interfered plants")