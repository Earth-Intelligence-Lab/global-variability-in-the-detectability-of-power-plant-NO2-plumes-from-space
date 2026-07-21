#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# optional plotting libs are imported by your template, but we won't plot
import matplotlib.pyplot as plt
import seaborn as sns

# ML / SHAP
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import BallTree
from haversine import haversine, Unit
from math import radians, log10
from imblearn.over_sampling import RandomOverSampler
import shap
from joblib import Parallel, delayed

# ===========
# CONSTANTS
# ===========
EARTH_RADIUS_KM = 6371.0
PLANT_RADIUS_BASE_KM = 20.0           # plant interferes if within 20 km AND has higher emissions
PLANT_MAX_SEARCH_KM = 150.0           # neighbor search window (BallTree)
CITY_POP_THRESHOLD = 200000
CITY_RADIUS_SCALE = 9.0
CITY_RADIUS_BASE_KM = 10.0
CITY_RADIUS_MAX_KM = 90.0
US_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# ===========
# PATHS
# ===========
TROPOMI_CSV = '/net/fs06/d3/rzhuang/TROPOMI/pipeline_100m_run/Run_100m_20260414/updated_tropomi_hourly_emissions_full_variables_augmented_localtz.csv'
PLANTS_CSV = '/net/fs06/d3/rzhuang/TROPOMI/data/us/facility_emissions_by_plant_comprehensive.csv'
ANNUAL_CSV = '/net/fs06/d3/rzhuang/TROPOMI/data/us/annual-emissions-facility-aggregation-2019-2024.csv'
CITIES_CSV = '/net/fs06/d3/rzhuang/TROPOMI/data/world/worldcities.csv'

OUT_DIR = '/net/fs06/d3/rzhuang/TROPOMI/data/us/Run_100m_20260414/training_no_stats_item/feature_importance'
OUT_ORIG_CSV = f'{OUT_DIR}/all_plants_contributions_original.csv'
OUT_CONST_CSV = f'{OUT_DIR}/all_plants_contributions_constant_clean.csv'
OUT_STATS_ORIG = f'{OUT_DIR}/feature_stats_original.csv'
OUT_STATS_CONST = f'{OUT_DIR}/feature_stats_constant_clean.csv'
OUT_INTERF_JSON = f'{OUT_DIR}/interference_summary_constant_clean.json'

MODEL_ORIG = '/net/fs06/d3/rzhuang/TROPOMI/data/us/Run_100m_20260414/training_no_stats_item/best_model_all_data.pt'
MODEL_CONST = '/net/fs06/d3/rzhuang/TROPOMI/data/us/Run_100m_20260414/training_no_stats_item/best_model_all_data_filtered_yearly.pt'

# ===========
# HELPERS
# ===========
def _norm(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalnum())

def _pick_col(df: pd.DataFrame, prefer_exact, fuzzy_contains):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for p in prefer_exact:
        if p in cols:
            return p
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    for c in cols:
        lc = _norm(c)
        if any(tok in lc for tok in fuzzy_contains):
            return c
    return None

def ensure_lat_lon(df: pd.DataFrame, target_case='upper') -> pd.DataFrame:
    """
    Robustly find latitude/longitude and rename:
      target_case='upper' -> 'Latitude','Longitude'
      target_case='lower' -> 'latitude','longitude'
    """
    if df is None or df.empty:
        return df
    lat_col = _pick_col(
        df,
        prefer_exact=['Latitude','latitude','LATITUDE','Latitude_y','Latitude_x','lat','Lat'],
        fuzzy_contains=['lat']
    )
    lon_col = _pick_col(
        df,
        prefer_exact=['Longitude','longitude','LONGITUDE','Longitude_y','Longitude_x','Lng','lng','long'],
        fuzzy_contains=['lon','long','lng']
    )
    if lat_col is None or lon_col is None:
        raise KeyError(f"Could not find latitude/longitude columns. Sample columns: {list(df.columns)[:15]}")

    if target_case == 'upper':
        target_lat, target_lon = 'Latitude', 'Longitude'
    else:
        target_lat, target_lon = 'latitude', 'longitude'

    rename_map = {}
    if lat_col != target_lat: rename_map[lat_col] = target_lat
    if lon_col != target_lon: rename_map[lon_col] = target_lon
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def process_source_dataframe(df_raw, lat_col, lon_col, id_col=None, value_col=None):
    """Standardize source dataframe to: latitude/Longitude -> 'latitude','longitude'; ID->'ID'; value->'value'."""
    if df_raw is None or df_raw.empty:
        return None, None
    df = df_raw.copy()

    # ensure given lat/lon exist (or auto-discover), then convert to lower-case names for BallTree routine
    if lat_col not in df.columns or lon_col not in df.columns:
        df = ensure_lat_lon(df, target_case='upper')
        lat_col, lon_col = 'Latitude','Longitude'
    if lat_col == 'Latitude' and lon_col == 'Longitude':
        df = ensure_lat_lon(df, target_case='lower')
        lat_col, lon_col = 'latitude','longitude'

    std_lat, std_lon, std_id, std_value = 'latitude', 'longitude', 'ID', 'value'
    col_map = {lat_col: std_lat, lon_col: std_lon}
    if id_col and id_col in df.columns: col_map[id_col] = std_id
    if value_col and value_col in df.columns: col_map[value_col] = std_value

    keep = [c for c in col_map.keys() if c in df.columns]
    if not keep:
        return None, None

    df = df[keep].rename(columns=col_map)
    if std_id in df.columns:
        df[std_id] = df[std_id].astype(str)

    for c in [std_lat, std_lon, std_value]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df.dropna(subset=[std_lat, std_lon], inplace=True)
    if df.empty:
        return df, None

    df['lat_rad'] = np.radians(df[std_lat])
    df['lon_rad'] = np.radians(df[std_lon])
    tree = BallTree(df[['lat_rad','lon_rad']].values, metric='haversine')
    return df, tree

# ===========
# YEAR EXTRACTION (TROPOMI)
# ===========
def extract_year_from_utc_time(df):
    """Ensure a 'year' column exists using 'utc_time' if needed."""
    if 'year' in df.columns:
        print(f"Found 'year' column directly. Years present: {sorted(df['year'].unique())}")
        return df

    if 'utc_time' in df.columns:
        try:
            try:
                df['year'] = pd.to_datetime(df['utc_time'], utc=True).dt.year
            except:
                df['year'] = pd.to_datetime(df['utc_time']).dt.year
        except Exception as e:
            print(f"Error parsing utc_time: {e}")
            df['year'] = df['utc_time'].astype(str).str[:4].astype(int)

        years_in_data = sorted(df['year'].unique())
        print(f"Successfully extracted years: {years_in_data}")
        print(f"\nYear distribution in TROPOMI data:")
        for year in years_in_data:
            count = len(df[df['year'] == year])
            print(f"  Year {year}: {count:,} observations")
    else:
        print("ERROR: 'utc_time' column not found, defaulting year=2024")
        df['year'] = 2024
    return df

# ===========
# INTERFERENCE
# ===========
def identify_us_interference_one_year(plants_year_df, cities_df):
    """
    plants_year_df must contain:
      - 'Facility ID' (string), 'Latitude', 'Longitude', 'NOx Mass (short tons)'
    Returns: (set_of_ids_interfered_by_plants, set_of_ids_interfered_by_cities) as strings.
    """
    # Ensure lat/lon present and named 'Latitude'/'Longitude'
    plants_year_df = ensure_lat_lon(plants_year_df, target_case='upper')

    # Standardize sources for BallTree
    src_plants_df, plant_tree = process_source_dataframe(
        plants_year_df, 'Latitude', 'Longitude',
        id_col='Facility ID', value_col='NOx Mass (short tons)'
    )
    src_cities_df, city_tree = process_source_dataframe(
        cities_df, 'latitude', 'longitude',
        id_col='name', value_col='population'
    )

    plant_interfered_ids, city_interfered_ids = set(), set()
    if plants_year_df.empty:
        return plant_interfered_ids, city_interfered_ids

    for _, tgt in plants_year_df.iterrows():
        tid = str(tgt['Facility ID'])
        tlat = tgt['Latitude']; tlon = tgt['Longitude']
        temis = tgt.get('NOx Mass (short tons)', 0)
        if pd.isna(tlat) or pd.isna(tlon) or pd.isna(temis):
            continue
        target_coords_rad = np.array([[radians(tlat), radians(tlon)]])

        # A) higher-emitting plants within 20 km
        if plant_tree is not None and src_plants_df is not None and not src_plants_df.empty:
            nearby_idx = plant_tree.query_radius(target_coords_rad, r=(PLANT_MAX_SEARCH_KM / EARTH_RADIUS_KM))[0]
            for idx in nearby_idx:
                src = src_plants_df.iloc[idx]
                if src['ID'] == tid:
                    continue
                src_emis = src.get('value', 0)
                if pd.isna(src_emis) or src_emis < temis:
                    continue
                dist_km = haversine((tlat, tlon), (src['latitude'], src['longitude']), unit=Unit.KILOMETERS)
                if dist_km < PLANT_RADIUS_BASE_KM:
                    plant_interfered_ids.add(tid)
                    break

        # B) large cities within variable radius
        if city_tree is not None and src_cities_df is not None and not src_cities_df.empty:
            nearby_idx = city_tree.query_radius(target_coords_rad, r=(CITY_RADIUS_MAX_KM / EARTH_RADIUS_KM))[0]
            for idx in nearby_idx:
                c = src_cities_df.iloc[idx]
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

def build_year_interference_dict_and_constant_clean(plants_locations_df, annual_df, cities_df):
    """
    Returns:
      - year_interfered_dict: {year: set(str Facility IDs interfered by plants or cities)}
      - complete_6y_ids: set(str) of plants observed in all 6 years (from annual_df)
      - kept_us_ids: set(str) = complete_6y_ids minus union of interfered across 6 years (expected 171)
    """
    # Prep locations & annual
    locs = plants_locations_df.copy()
    locs['Facility_ID'] = locs['Facility_ID'].astype(str)
    locs = ensure_lat_lon(locs, target_case='upper')

    annual = annual_df.copy()
    annual['Facility ID'] = annual['Facility ID'].astype(str)

    # If you want the top-500 cohort:
    if 'NOx_Rank' in locs.columns:
        locs = locs.sort_values('NOx_Rank', ascending=True)
    locs_top500 = locs.head(500).copy()

    top500_ids = set(locs_top500['Facility_ID'])
    annual_top500 = annual[annual['Facility ID'].isin(top500_ids) & annual['Year'].isin(US_YEARS)].copy()

    # complete 6-year IDs
    present_counts = annual_top500.groupby('Facility ID')['Year'].nunique()
    complete_6y_ids = set(present_counts[present_counts == len(US_YEARS)].index)

    # merge lat/lon into annual
    merge_right = locs_top500.drop(columns=['State','Facility_Name','Primary_Fuel_Type'], errors='ignore').copy()
    merge_right = merge_right.rename(columns={'Facility_ID':'Facility ID'})
    merge_right = ensure_lat_lon(merge_right, target_case='upper')

    annual_top500 = pd.merge(annual_top500, merge_right, on='Facility ID', how='left')
    # make sure lat/lon exist even with suffixes
    try:
        annual_top500 = ensure_lat_lon(annual_top500, target_case='upper')
    except KeyError:
        rename_guess = {}
        for c in annual_top500.columns:
            cl = c.lower()
            if cl in ('latitude_x','latitude_y'): rename_guess[c] = 'Latitude'
            if cl in ('longitude_x','longitude_y'): rename_guess[c] = 'Longitude'
        if rename_guess:
            annual_top500 = annual_top500.rename(columns=rename_guess)
        annual_top500 = ensure_lat_lon(annual_top500, target_case='upper')

    annual_top500['Latitude'] = pd.to_numeric(annual_top500['Latitude'], errors='coerce')
    annual_top500['Longitude'] = pd.to_numeric(annual_top500['Longitude'], errors='coerce')
    annual_top500 = annual_top500.dropna(subset=['Latitude','Longitude'])

    # per-year interference
    year_interfered_dict = {}
    print("\n================ US Year-by-Year Interference ================")
    for yr in US_YEARS:
        # df_y = annual_top500[annual_top500['Year'] == yr].copy()
        df_y = annual_top500[(annual_top500['Year'] == yr) & 
                     (annual_top500['Facility ID'].isin(complete_6y_ids))].copy()
        if df_y.empty:
            print(f"[{yr}] No records.")
            year_interfered_dict[yr] = set()
            continue
        plant_set, city_set = identify_us_interference_one_year(df_y, cities_df)
        union_set = set(map(str, plant_set.union(city_set)))
        year_interfered_dict[yr] = union_set
        print(f"[{yr}] Plants interfered by plants: {len(plant_set)}; by cities: {len(city_set)}; unique union: {len(union_set)}")

    # constant-clean set
    complete_6y_ids_str = set(map(str, complete_6y_ids))
    union_all_years = set().union(*[year_interfered_dict.get(y, set()) for y in US_YEARS])
    kept_us_ids = complete_6y_ids_str - union_all_years

    print(f"\nUS plants in all 6 years: {len(complete_6y_ids)}")
    print(f"US plants removed due to interference in any year: {len(union_all_years & complete_6y_ids_str)}")
    print(f"US plants for analysis (6y & non-interfered): {len(kept_us_ids)}  # expected 171")

    return year_interfered_dict, complete_6y_ids_str, kept_us_ids

def filter_data_by_year_interference(tropomi_df, year_interfered_dict):
    filtered_dfs = []
    years_in_data = sorted(tropomi_df['year'].unique())
    print(f"\nFiltering data by year-specific interference zones...")
    total_removed = 0
    for year in years_in_data:
        year_data = tropomi_df[tropomi_df['year'] == year].copy()
        interfered_ids = year_interfered_dict.get(year, set())
        year_data_filtered = year_data[~year_data['location'].astype(str).isin(interfered_ids)]
        filtered_dfs.append(year_data_filtered)

        removed_obs = len(year_data) - len(year_data_filtered)
        total_removed += removed_obs
        # NEW: report plant count like your “right results”
        print(f"Year {year}: {len(year_data):,} -> {len(year_data_filtered):,} observations after filtering ({len(interfered_ids)} plants removed)")

    print(f"\nTotal observations removed: {total_removed:,}")
    return pd.concat(filtered_dfs, ignore_index=True) if filtered_dfs else pd.DataFrame()

# ===========
# MODEL
# ===========
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

# ===========
# SHAP per-plant
# ===========
def process_single_plant(plant_data, explainer, features):
    """
    plant_data: (plant_samples, plant_id, plant_y, plant_X_original, lat, lon)
    """
    plant_samples, plant_id, plant_y, plant_X_original, lat, lon = plant_data

    # sample to at most 20 per plant
    if len(plant_samples) > 50:
        idx = np.random.choice(len(plant_samples), 50, replace=False)
        plant_samples = plant_samples[idx]
        plant_y = plant_y[idx]
        plant_X_original = plant_X_original[idx]

    try:
        shap_values = explainer.shap_values(plant_samples)
        with torch.no_grad():
            predictions = explainer.expected_value + shap_values.sum(axis=1)
            predictions = 1 / (1 + np.exp(-predictions))  # sigmoid

        avg_shap = shap_values.mean(axis=0)
        abs_avg_shap = np.abs(avg_shap)
        total_contribution = abs_avg_shap.sum()
        if total_contribution > 0:
            pct = (abs_avg_shap / total_contribution) * 100.0
        else:
            pct = np.zeros_like(abs_avg_shap)

        top_idx = np.argsort(pct)[::-1][:10]
        res = {
            'plant_id': plant_id,
            'latitude': lat,
            'longitude': lon,
            'true_plume': plant_y[0],
            'avg_prediction': predictions.mean(),
            'num_observations': len(plant_samples)
        }
        for rank, fi in enumerate(top_idx):
            res[f'top_{rank+1}_feature'] = features[fi]
            res[f'top_{rank+1}_contribution_pct'] = pct[fi]
            res[f'top_{rank+1}_shap_value'] = avg_shap[fi]
            res[f'top_{rank+1}_direction'] = 'increases' if avg_shap[fi] > 0 else 'decreases'
            res[f'top_{rank+1}_avg_value'] = plant_X_original[:, fi].mean()
        return res
    except Exception as e:
        print(f"Error processing plant {plant_id}: {e}")
        return None

def analyze_all_plants(data, features, model_path, batch_size=10000, dataset_name=""):
    """
    Analyze ALL plants in the dataset using parallel processing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure tropomi lat/lon exist (lower-case) for storing in results
    try:
        data = ensure_lat_lon(data, target_case='lower')
    except Exception:
        pass

    # Prepare data
    le = LabelEncoder()
    if 'primary_fuel_type' in data.columns:
        data['primary_fuel_type'] = le.fit_transform(data['primary_fuel_type'])
    X = data[features].to_numpy(dtype=np.float32)
    y = data['plume_label'].to_numpy(dtype=int)
    plant_ids = data['location'].astype(str).to_numpy()
    latitudes = data['latitude'].to_numpy() if 'latitude' in data.columns else np.full(len(data), np.nan)
    longitudes = data['longitude'].to_numpy() if 'longitude' in data.columns else np.full(len(data), np.nan)

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

    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(device)
            output = torch.sigmoid(model(x_tensor))
            return output.cpu().numpy()

    # SHAP explainer (KernelExplainer w/ background subset)
    n_background = min(100, X_tr_s.shape[0])
    bg_idx = np.random.choice(X_tr_s.shape[0], n_background, replace=False)
    background = X_tr_s[bg_idx]
    explainer = shap.KernelExplainer(model_predict, background)

    # plants to process
    unique_plants = np.unique(pid_test)
    total_plants = len(unique_plants)
    print(f"\nStarting SHAP analysis of {total_plants} plants in {dataset_name}...")

    plant_data_list = []
    for plant_id in unique_plants:
        mask = pid_test == plant_id
        samples = X_te_s[mask]
        yy = y_test[mask]
        X_orig = X_test[mask]
        if len(samples) > 0:
            plat = lat_test[mask][0] if len(lat_test[mask]) else np.nan
            plon = lon_test[mask][0] if len(lon_test[mask]) else np.nan
            plant_data_list.append((samples, plant_id, yy, X_orig, plat, plon))

    all_results = []
    n_batches = (len(plant_data_list) + batch_size - 1) // batch_size
    for b in range(n_batches):
        s = b * batch_size
        e = min((b+1) * batch_size, len(plant_data_list))
        batch = plant_data_list[s:e]
        print(f"  Processing batch {b+1}/{n_batches} (plants {s+1}-{e}/{total_plants})")
        batch_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(process_single_plant)(pd_tuple, explainer, features) for pd_tuple in batch
        )
        batch_results = [r for r in batch_results if r is not None]
        all_results.extend(batch_results)

    print(f"  Total plants successfully analyzed: {len(all_results)}/{total_plants}")
    return pd.DataFrame(all_results), scaler, explainer

# ===========
# FEATURE STATS SUMMARY
# ===========
def summarize_feature_stats(df):
    feature_stats = {}
    for _, row in df.iterrows():
        for i in range(1, 6):  # Top 5 features
            fi = f'top_{i}_feature'
            ci = f'top_{i}_contribution_pct'
            if fi in row and ci in row:
                feature = row[fi]
                contribution = row[ci]
                if feature not in feature_stats:
                    feature_stats[feature] = {'contributions': [], 'as_top_1': 0, 'as_top_3': 0}
                feature_stats[feature]['contributions'].append(contribution)
                if i == 1:
                    feature_stats[feature]['as_top_1'] += 1
                if i <= 3:
                    feature_stats[feature]['as_top_3'] += 1

    summary = []
    for feature, stats in feature_stats.items():
        if len(stats['contributions']) >= 10:
            summary.append({
                'feature': feature,
                'mean_contribution': np.mean(stats['contributions']),
                'pct_plants_top_1': stats['as_top_1'] / len(df) * 100.0,
                'pct_plants_top_3': stats['as_top_3'] / len(df) * 100.0
            })
    return pd.DataFrame(summary).sort_values('mean_contribution', ascending=False)

# ===========
# MAIN
# ===========
if __name__ == '__main__':
    print("="*100)
    print("SHAP + YEAR-BY-YEAR INTERFERENCE with CONSTANT-AVAILABILITY CLEAN DEFINITION")
    print("="*100)

    # Load
    print("\nLoading data...")
    tropomi = pd.read_csv(TROPOMI_CSV, low_memory=False).dropna()
    plants_all = pd.read_csv(PLANTS_CSV)
    annual = pd.read_csv(ANNUAL_CSV)
    cities = pd.read_csv(CITIES_CSV)

    print(f"Total records: {len(tropomi):,}")
    print(f"Unique plants in TROPOMI: {tropomi['location'].nunique()}")

    # Ensure types & year
    tropomi['location'] = tropomi['location'].astype(str)
    tropomi = extract_year_from_utc_time(tropomi)

    # Build per-year interference + constant-clean set (expected 171)
    year_interfered_dict, complete_6y_ids, kept_us_ids = build_year_interference_dict_and_constant_clean(
        plants_all, annual, cities
    )

    # Extra reporting (constant-availability from TROPOMI itself)
    required_years = set(US_YEARS)
    plant_years = tropomi.groupby('location')['year'].agg(lambda s: set(s))
    constant_avail_from_tropomi = {pid for pid, yrs in plant_years.items() if required_years.issubset(yrs)}
    print(f"Constant-availability plants (observed in all 6 years in TROPOMI): {len(constant_avail_from_tropomi)}")

    # Year-specific filtered dataset (not restricted to constant availability)
    tropomi_year_filtered = filter_data_by_year_interference(tropomi, year_interfered_dict)

    # Final CONSTANT-CLEAN dataset (6y & never interfered across any of the 6 years)
    tropomi_constant_clean = tropomi_year_filtered[tropomi_year_filtered['location'].isin(kept_us_ids)].copy()

    # Summary
    all_interfered_union = set().union(*[year_interfered_dict.get(y, set()) for y in US_YEARS])
    print(f"\nDataset Summary:")
    print(f"  Total unique plants (TROPOMI): {tropomi['location'].nunique()}")
    print(f"  Ever interfered plants (union 2019–2024): {len(all_interfered_union)}")
    print(f"  Never interfered plants (among TROPOMI): {tropomi['location'].nunique() - len(all_interfered_union)}")
    print(f"  Original observations: {len(tropomi):,}")
    print(f"  Year-filtered observations: {len(tropomi_year_filtered):,}")
    print(f"  CONSTANT-CLEAN observations (171 plants): {len(tropomi_constant_clean):,}")

    # =========================
    # SHAP features
    # =========================
    features = np.array([
        'surface_altitude', 'surface_altitude_precision',
        'surface_classification', 'surface_pressure', 'surface_albedo',
        'surface_albedo_nitrogendioxide_window', 'cloud_pressure_crb',
        'cloud_fraction_crb', 'cloud_albedo_crb', 'scene_albedo',
        'apparent_scene_pressure', 'snow_ice_flag', 'aerosol_index_354_388',
        'scaled_small_pixel_variance',
        'sensor_altitude', 'sensor_azimuth_angle', 'sensor_zenith_angle',
        'solar_azimuth_angle', 'solar_zenith_angle', 'wind_speed', 't2m', 'tisr', 'tcwv',
        'primary_fuel_type', 'NOx Mass (lbs)'
    ])

    # =========================
    # SHAP on ORIGINAL (full) dataset
    # =========================
    # print("\n" + "="*80)
    # print("ANALYZING ORIGINAL DATA (INCLUDING INTERFERED PLANTS)")
    # print("="*80)
    # plant_df_original, scaler_orig, explainer_orig = analyze_all_plants(
    #     tropomi, features, MODEL_ORIG, batch_size=10000, dataset_name="ORIGINAL DATA"
    # )

    # # Tag if plant is in the interfered union
    # plant_df_original['is_ever_interfered'] = plant_df_original['plant_id'].astype(str).isin(all_interfered_union)
    # plant_df_original.to_csv(OUT_ORIG_CSV, index=False)
    # print(f"Saved: {OUT_ORIG_CSV} ({len(plant_df_original)} plants)")

    # =========================
    # SHAP on CONSTANT-CLEAN dataset (171 plants)
    # =========================
    print("\n" + "="*80)
    print("ANALYZING CONSTANT-CLEAN DATA (6y & never-interfered)")
    print("="*80)
    plant_df_const, scaler_const, explainer_const = analyze_all_plants(
        tropomi_constant_clean, features, MODEL_CONST, batch_size=10000, dataset_name="CONSTANT-CLEAN DATA"
    )
    # Helpful tag
    plant_df_const['in_constant_clean'] = True
    plant_df_const.to_csv(OUT_CONST_CSV, index=False)
    print(f"Saved: {OUT_CONST_CSV} ({len(plant_df_const)} plants)")

    # =========================
    # Summaries (no plots)
    # =========================
    print("\n" + "="*100)
    print("COMPARATIVE SUMMARY: ORIGINAL vs CONSTANT-CLEAN")
    print("="*100)

    def acc_line(name, df):
        correct = ((df['avg_prediction'] > 0.5) == df['true_plume']).sum()
        total = len(df)
        print(f"{name:16}: {correct}/{total} correct ({(correct/total*100 if total else 0):.1f}% accuracy)")

    print("\n--- Model Performance (per-plant majority proxy) ---")
    # acc_line("Original", plant_df_original)
    acc_line("Constant-clean", plant_df_const)

    # Feature summaries
    # stats_original = summarize_feature_stats(plant_df_original)
    stats_const = summarize_feature_stats(plant_df_const)

    # stats_original.to_csv(OUT_STATS_ORIG, index=False)
    stats_const.to_csv(OUT_STATS_CONST, index=False)

    print("\n--- Saved Feature Stats ---")
    # print(f"Original feature stats : {OUT_STATS_ORIG} (rows={len(stats_original)})")
    print(f"Constant-clean stats   : {OUT_STATS_CONST} (rows={len(stats_const)})")

    # Year-by-year interference info (for provenance)
    interference_summary = {
        'years': US_YEARS,
        'complete_6y_ids_count': len(complete_6y_ids),
        'constant_clean_ids_count': len(kept_us_ids),
        'year_specific_interference_counts': {str(y): len(year_interfered_dict.get(y, set())) for y in US_YEARS},
        'ever_interfered_union_count': len(all_interfered_union)
    }
    with open(OUT_INTERF_JSON, 'w') as f:
        json.dump(interference_summary, f, indent=2)
    print(f"\nSaved interference summary: {OUT_INTERF_JSON}")

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    print(f"Files saved:\n- {OUT_ORIG_CSV}\n- {OUT_CONST_CSV}\n- {OUT_STATS_ORIG}\n- {OUT_STATS_CONST}\n- {OUT_INTERF_JSON}")