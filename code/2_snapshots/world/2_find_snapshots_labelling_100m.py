import os
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from sklearn.neighbors import BallTree
from datetime import datetime, timedelta
import argparse
import time
import multiprocessing as mp
import logging
from pyproj import Geod
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import warnings
from haversine import haversine, Unit
from plotting import prepare_interfering_sources_build_trees_inside, label_no2_plume_flexible_interference, calculate_areas
from math import radians, log10, sqrt
import json


# ─── ERA5 100m wind globals (loaded once in main, shared via fork COW) ────
# Identical integration as code/2_snapshots/us/2_find_snapshots_labelling_100m.py
_ERA5_U100 = None            # (T, lat, lon) float32
_ERA5_V100 = None
_ERA5_TIMES_NS = None        # (T,) int64 nanoseconds since epoch
_ERA5_LATS = None            # ascending
_ERA5_LONS = None            # ascending


def load_era5_100m_wind(era5_dir, years=range(2018, 2019)):
    """Load u100/v100 yearly ERA5 files into global cubes (shared via fork COW)."""
    global _ERA5_U100, _ERA5_V100, _ERA5_TIMES_NS, _ERA5_LATS, _ERA5_LONS
    u_parts, v_parts, t_parts = [], [], []
    for y in years:
        up = os.path.join(era5_dir, f'u100_{y}.nc')
        vp = os.path.join(era5_dir, f'v100_{y}.nc')
        if not (os.path.exists(up) and os.path.exists(vp)):
            continue
        with xr.open_dataset(up, engine='netcdf4') as du:
            u_parts.append(du['u100'].values.astype(np.float32))
            t_parts.append(du.time.values.astype('datetime64[ns]').astype(np.int64))
            if _ERA5_LATS is None:
                _ERA5_LATS = du.latitude.values.astype(np.float64)
                _ERA5_LONS = du.longitude.values.astype(np.float64)
        with xr.open_dataset(vp, engine='netcdf4') as dv:
            v_parts.append(dv['v100'].values.astype(np.float32))
    if not u_parts:
        raise FileNotFoundError(f'No u100/v100 yearly files in {era5_dir}')
    U = np.concatenate(u_parts, axis=0)
    V = np.concatenate(v_parts, axis=0)
    T = np.concatenate(t_parts)
    order = np.argsort(T)
    _ERA5_TIMES_NS = T[order]
    _ERA5_U100 = U[order]; _ERA5_V100 = V[order]
    if _ERA5_LATS[0] > _ERA5_LATS[-1]:
        _ERA5_LATS = _ERA5_LATS[::-1]
        _ERA5_U100 = _ERA5_U100[:, ::-1, :]
        _ERA5_V100 = _ERA5_V100[:, ::-1, :]
    if _ERA5_LONS[0] > _ERA5_LONS[-1]:
        _ERA5_LONS = _ERA5_LONS[::-1]
        _ERA5_U100 = _ERA5_U100[:, :, ::-1]
        _ERA5_V100 = _ERA5_V100[:, :, ::-1]
    print(f'[era5] loaded 100m wind: shape {_ERA5_U100.shape}, '
          f'{len(_ERA5_TIMES_NS):,} hours, '
          f'lat {_ERA5_LATS[0]:.2f}..{_ERA5_LATS[-1]:.2f}, '
          f'lon {_ERA5_LONS[0]:.2f}..{_ERA5_LONS[-1]:.2f}', flush=True)


def _utc_to_ns(utc_str_or_np):
    if isinstance(utc_str_or_np, bytes):
        utc_str_or_np = utc_str_or_np.decode()
    if isinstance(utc_str_or_np, str):
        return np.datetime64(utc_str_or_np.replace('Z', '').split('.')[0], 'ns').astype(np.int64)
    return np.datetime64(utc_str_or_np, 'ns').astype(np.int64)


def interp_era5_100m(plant_lat, plant_lon, utc_ns):
    """Bilinear-in-space, nearest-hour-in-time interp of ERA5 100m wind.
    Returns (u100, v100) scalars at the plant location."""
    if _ERA5_U100 is None:
        raise RuntimeError('ERA5 100m cubes not loaded')
    left = np.clip(np.searchsorted(_ERA5_TIMES_NS, utc_ns) - 1, 0, len(_ERA5_TIMES_NS) - 1)
    right = np.clip(left + 1, 0, len(_ERA5_TIMES_NS) - 1)
    t_snap = left if abs(utc_ns - _ERA5_TIMES_NS[left]) <= abs(utc_ns - _ERA5_TIMES_NS[right]) else right
    iu = RegularGridInterpolator((_ERA5_LATS, _ERA5_LONS), _ERA5_U100[int(t_snap)],
                                 method='linear', bounds_error=False, fill_value=np.nan)
    iv = RegularGridInterpolator((_ERA5_LATS, _ERA5_LONS), _ERA5_V100[int(t_snap)],
                                 method='linear', bounds_error=False, fill_value=np.nan)
    return float(iu([plant_lat, plant_lon])[0]), float(iv([plant_lat, plant_lon])[0])

# --- Constants ---
EARTH_RADIUS_KM = 6371

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tropomi_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# --- Helper Functions ---

def calculate_pixel_areas(plant_lat, plant_lon, tree, valid_no2, radius_rad, lats, lons):
    """Calculate area of valid TROPOMI pixels within radius."""
    
    plant_rad = np.radians([plant_lat, plant_lon]).reshape(1, -1)
    indices = tree.query_radius(plant_rad, r=radius_rad)[0]
    if len(indices) == 0:
        return 0
    
    valid_indices = indices[valid_no2[indices]]
    total_indices = len(indices)
    if total_indices == 0:
        return 0

    # Build mask for calculate_areas
    mask = np.zeros(lats.shape, dtype=bool)
    mask.flat[valid_indices] = True

    valid_area = calculate_areas(mask, lats, lons)
    return valid_area

def plant_valid(plant_lat, plant_lon, lats, lons, tree, valid_no2, radius_rad, valid_ratio=0.5):
    """Check if a plant has sufficient valid TROPOMI data coverage."""
    valid_area = calculate_pixel_areas(
        plant_lat, plant_lon, tree, valid_no2, radius_rad, lats, lons
    )
    total_area = np.pi * (50**2)  # Circle with 50km radius
    return (valid_area/total_area) > valid_ratio

def save_config(config, output_dir):
    """Save configuration parameters to a file."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to: {config_path}")
    return config_path

def process_file(file_path, emissions_df, cities_df, process_id=0, config=None):
    """Process a single TROPOMI file."""
    file_start_time = time.time()
    file_name = os.path.basename(file_path)
    valid_plants = []
    
    logger.info(f"[Process {process_id}] Processing file: {file_name}")
     
    try:
        # Read TROPOMI data from NetCDF file
        with nc.Dataset(file_path) as ds:
            lats = ds['PRODUCT/latitude'][:][0]
            lons = ds['PRODUCT/longitude'][:][0]
            no2_var = ds['PRODUCT/nitrogendioxide_tropospheric_column']
            no2_data_raw = no2_var[:][0]
            wind_u = ds['PRODUCT/SUPPORT_DATA/INPUT_DATA/eastward_wind'][:][0]
            wind_v = ds['PRODUCT/SUPPORT_DATA/INPUT_DATA/northward_wind'][:][0]
            utc_time_data = ds['PRODUCT/time_utc'][:][0]
            
            # Apply fill value - FIXED
            fill_value = no2_var._FillValue if hasattr(no2_var, '_FillValue') else None
            if fill_value is not None:
                no2_masked = np.ma.masked_values(no2_data_raw, fill_value)
            else:
                no2_masked = np.ma.masked_array(no2_data_raw)

            no2 = np.array(np.ma.filled(no2_masked, np.nan))
            
            # ← Add this line to read the QA values
            qa = ds['PRODUCT'].variables['qa_value'][:][0]
            
        # Flatten arrays and convert to radians for BallTree
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()
        points_rad = np.radians(np.column_stack((lats_flat, lons_flat)))
        tree = BallTree(points_rad, metric='haversine')
        radius_rad = 50 / 6371  # 50 km in radians

        # Prepare valid pixel mask: require finite NO₂ *and* qa_value > 0.75
        qa_flat   = qa.flatten()
        no2_flat  = no2.flatten()
        valid_no2 = (qa_flat > 0.75) & np.isfinite(no2_flat)
        
        # Process each power plant sequentially
        file_valid_count = 0
        plant_count = 0
        
        # Process plants in this file
        for idx, emission in emissions_df.iterrows():
            plant_lat = emission['latitude']
            plant_lon = emission['longitude']
            primary_fuel_type = emission['fuel']
            
            plant_count += 1
            if plant_count % 1000 == 0:
                logger.info(f"[Process {process_id}] Checked {plant_count} plants for file: {file_name}")
            
            # Skip plants with invalid coordinates
            if pd.isna(plant_lat) or pd.isna(plant_lon):
                continue
                
            # Check if this plant has valid TROPOMI data
            is_valid = plant_valid(
                plant_lat, plant_lon, lats, lons, tree, valid_no2, radius_rad, valid_ratio=0.5
            )
            
            if not is_valid:
                continue
                
            file_valid_count += 1
            
            # Find the nearest TROPOMI pixel
            plant_rad = np.radians([plant_lat, plant_lon]).reshape(1, -1)
            _, nearest_idx = tree.query(plant_rad, k=1)
            nearest_idx = nearest_idx[0][0]
            
            # Get the grid indices for the nearest pixel
            grid_idx = np.unravel_index(nearest_idx, lats.shape)

            # Handle UTC time based on dimensions
            if utc_time_data.ndim == 1:
                local_utc_time = utc_time_data[grid_idx[0]] if len(grid_idx) > 0 else None
            elif utc_time_data.ndim == 2:
                local_utc_time = utc_time_data[grid_idx[0], grid_idx[1]] if len(grid_idx) > 1 else None
            else:
                try:
                    local_utc_time = utc_time_data[grid_idx]
                except IndexError:
                    local_utc_time = utc_time_data[grid_idx[0]] if len(grid_idx) > 0 else None

            # Wind driving plume detection: ERA5 100m at the plant location.
            # Replaces TROPOMI-embedded 10m operational-forecast wind. Same
            # integration as code/2_snapshots/us/2_find_snapshots_labelling_100m.py.
            try:
                utc_ns = _utc_to_ns(local_utc_time)
                local_wind_u, local_wind_v = interp_era5_100m(plant_lat, plant_lon, utc_ns)
            except Exception as e:
                logger.warning(f'[Process {process_id}] ERA5 100m wind failed for '
                               f'plant {emission.get("ID","?")}: {e}; falling back to TROPOMI 10m')
                local_wind_u = float(np.ma.filled(wind_u[grid_idx], np.nan))
                local_wind_v = float(np.ma.filled(wind_v[grid_idx], np.nan))
                
            # --- Extract primary data ---
            facility_id = emission['ID']
            annual_nox = emission.get('nox_emis_ty', 0)
            country_code = emission.get('ISO3', 0)

            # --- Run Plume Detection with parameters from config ---
            interfering_sources = prepare_interfering_sources_build_trees_inside(
                target_lat=plant_lat, target_lon=plant_lon,
                all_plants_df_raw=emissions_df, cities_df_raw=cities_df,
                max_distance_km=config['interf_max_distance_km'], target_plant_id=facility_id,
                plant_in_id_col="ID", plant_in_lat_col="latitude", plant_in_lon_col="longitude",
                plant_in_emission_col="nox_emis_ty", city_in_lat_col="latitude", city_in_lon_col="longitude",
                city_in_pop_col="population", city_in_name_col="name", city_in_country_col="country",
                city_base_radius=config['city_base_radius'], city_pop_scale=config['city_pop_scale'], 
                city_radius_min=config['city_radius_min'], city_radius_max=config['city_radius_max'],
                plant_base_radius=config['plant_base_radius'], plant_emission_scale=config['plant_emission_scale'],
                plant_radius_min=config['plant_radius_min'], plant_radius_max=config['plant_radius_max'],
                min_city_population_threshold=config['interf_city_pop_thresh'],
                min_plant_emission_threshold=config['interf_plant_emis_thresh'] * annual_nox,
            )
            
            # Run plume detection with config parameters
            results = label_no2_plume_flexible_interference(
                full_no2=no2, full_lon=lons, full_lat=lats,
                wind_u=local_wind_u, wind_v=local_wind_v, 
                plant_lon=plant_lon, plant_lat=plant_lat,
                interfering_sources=interfering_sources,
                zoom_radius_km=config['zoom_radius_km'], 
                threshold_factor=config['threshold_factor'],
                threshold_abs_min=config['threshold_abs_min'], 
                threshold_radius_km=config['threshold_radius_km'],
                max_distance_km=config['max_distance_km'], 
                close_distance_km=config['close_distance_km'],
                max_angle_diff=config['max_angle_diff'], 
                max_angle_diff_mask=config['max_angle_diff_mask'], 
                close_distance_km_mask=config['close_distance_km_mask'],
                flagged_area=config['flagged_area'], 
                background_mode=config['background_mode'], 
                upwind_angle_tolerance=config['upwind_angle_tolerance'],
                background_dist_min_km=config['background_dist_min_km'], 
                background_dist_max_km=config['background_dist_max_km'],
                sigma=config['sigma'],
                stat_radius=config['stat_radius']
            )
            
            # Add to results
            valid_plants.append({
                'location': facility_id,
                'latitude': plant_lat,
                'longitude': plant_lon,
                'utc_time': local_utc_time,
                'wind_u': float(np.ma.filled(local_wind_u, np.nan)),
                'wind_v': float(np.ma.filled(local_wind_v, np.nan)),
                'plume_label': results['is_significant'],
                'file_path': file_path,
                'country': country_code,
                'annual_nox_emission': annual_nox,
                'primary_fuel_type': primary_fuel_type,
                'no2_mean_radius': results['no2_mean_radius'],
                'no2_std_radius': results['no2_std_radius'],
                'no2_frac_valid_radius': results['no2_frac_valid_radius'],
            })

        file_processing_time = time.time() - file_start_time
        logger.info(f"[Process {process_id}] Completed file: {file_name} in {file_processing_time:.2f} seconds")
        logger.info(f"[Process {process_id}] Found {file_valid_count} valid plants out of {plant_count} checked")
        
    except Exception as e:
        logger.error(f"[Process {process_id}] Error processing file {file_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    return valid_plants

def process_file_wrapper(args):
    """Wrapper function for process_file to handle multiple arguments for pool.map."""
    if len(args) == 5:
        file_path, emissions_df, cities_df, process_id, config = args
        return process_file(file_path, emissions_df, cities_df, process_id, config)
    else:
        file_path, emissions_df, cities_df, process_id = args
        return process_file(file_path, emissions_df, cities_df, process_id)

def parse_arguments():
    """Parse command line arguments for different configuration runs."""
    parser = argparse.ArgumentParser(description='TROPOMI processing with configurable parameters')
    
    # CPU allocation
    parser.add_argument('--num_cpus', type=int, default=None, 
                       help='Number of CPUs to use for processing')
    
    # Run identification
    parser.add_argument('--run_id', type=str, help='Run identifier (e.g., 136)')
    parser.add_argument('--base_output_dir', type=str, default='../data', help='Base output directory')
    
    # Add all configuration parameters as arguments
    parser.add_argument('--zoom_radius_km', type=float, default=100)
    parser.add_argument('--threshold_factor', type=float, default=2)
    parser.add_argument('--threshold_abs_min', type=float, default=5e-6)
    parser.add_argument('--max_distance_km', type=float, default=20.0)
    parser.add_argument('--close_distance_km', type=float, default=5.0)
    parser.add_argument('--max_angle_diff', type=float, default=25.0)
    parser.add_argument('--flagged_area', type=float, default=25.0)
    parser.add_argument('--threshold_radius_km', type=float, default=50.0)    
    
    parser.add_argument('--background_mode', type=str, default='directional')
    parser.add_argument('--upwind_angle_tolerance', type=float, default=60)
    parser.add_argument('--background_dist_min_km', type=float, default=10)
    parser.add_argument('--background_dist_max_km', type=float, default=100)
    parser.add_argument('--sigma', type=float, default=10)
    
    parser.add_argument('--max_angle_diff_mask', type=float, default=0.0)
    parser.add_argument('--close_distance_km_mask', type=float, default=20.0)

    parser.add_argument('--interf_max_distance_km', type=float, default=150.0)
    parser.add_argument('--interf_city_pop_thresh', type=float, default=200000.0)
    parser.add_argument('--interf_plant_emis_thresh', type=float, default=1.0)

    parser.add_argument('--city_base_radius', type=float, default=0.0)
    parser.add_argument('--city_pop_scale', type=float, default=9.0)
    parser.add_argument('--city_radius_min', type=float, default=10.0)
    parser.add_argument('--city_radius_max', type=float, default=90.0)
    
    parser.add_argument('--plant_base_radius', type=float, default=0.0)
    parser.add_argument('--plant_emission_scale', type=float, default=0.0)
    parser.add_argument('--plant_radius_min', type=float, default=0.0)
    parser.add_argument('--plant_radius_max', type=float, default=0.0)
    
    parser.add_argument('--stat_radius', type=float, default=50.0)
    
    
    return parser.parse_args()

def main():
    # Track total processing time
    total_start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create run ID if not provided
    if not args.run_id:
        args.run_id = f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        args.run_id = f"Run_{args.run_id}"
    
    # Create output directory
    output_dir = os.path.join(args.base_output_dir, args.run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging for this specific run
    log_file = os.path.join(output_dir, f"{args.run_id}_processing.log")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("=" * 80)
    logger.info(f"Starting TROPOMI processing for {args.run_id}")
    logger.info("=" * 80)

    # Build config from command line arguments
    config = {
        'zoom_radius_km': args.zoom_radius_km,
        'threshold_factor': args.threshold_factor,
        'threshold_abs_min': args.threshold_abs_min,
        'max_distance_km': args.max_distance_km,
        'close_distance_km': args.close_distance_km,
        'max_angle_diff': args.max_angle_diff,
        'flagged_area': args.flagged_area,
        'sigma': args.sigma,
        'threshold_radius_km': args.threshold_radius_km,
        'background_mode': args.background_mode,
        'upwind_angle_tolerance': args.upwind_angle_tolerance,
        'background_dist_min_km': args.background_dist_min_km,
        'background_dist_max_km': args.background_dist_max_km,
        'max_angle_diff_mask': args.max_angle_diff_mask,
        'close_distance_km_mask': args.close_distance_km_mask,
        'interf_max_distance_km': args.interf_max_distance_km,
        'interf_city_pop_thresh': args.interf_city_pop_thresh,
        'interf_plant_emis_thresh': args.interf_plant_emis_thresh,
        'city_base_radius': args.city_base_radius,
        'city_pop_scale': args.city_pop_scale,
        'city_radius_min': args.city_radius_min,
        'city_radius_max': args.city_radius_max,
        'plant_base_radius': args.plant_base_radius,
        'plant_emission_scale': args.plant_emission_scale,
        'plant_radius_min': args.plant_radius_min,
        'plant_radius_max': args.plant_radius_max,
        'stat_radius': args.stat_radius,
    }

    # Log the configuration
    logger.info(f"Configuration for {args.run_id}:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    # Load emissions data
    emissions_path = '/net/fs06/d3/rzhuang/TROPOMI/data/world/power_plant_location/power_plants_with_combined_nearby_stats.csv'
    citys_path = '/net/fs06/d3/rzhuang/TROPOMI/data/world/worldcities.csv'
    logger.info(f"Loading emissions data from {emissions_path}")
    emissions = pd.read_csv(emissions_path)
    emissions = emissions[:6000]
    cities = pd.read_csv(citys_path)
    logger.info(f"Loaded {len(emissions)} emission sources")

    # Find TROPOMI data files
    file_path = '/net/fs06/d3/rzhuang/TROPOMI/data/world/TROPOMI_2018_data'
    files = sorted([os.path.join(file_path, item) for item in os.listdir(file_path) if item.endswith('.nc')])
    logger.info(f"Found {len(files)} TROPOMI files to process")

    # Load ERA5 100m wind BEFORE Pool fork so workers inherit via COW.
    # Same integration as code/2_snapshots/us/2_find_snapshots_labelling_100m.py.
    era5_dir = '/net/fs06/d3/rzhuang/TROPOMI/data/world/era5/expanded'
    logger.info(f"Loading ERA5 100m wind from {era5_dir}")
    load_era5_100m_wind(era5_dir, years=range(2018, 2019))
    
    # Save config file to the run-specific directory
    save_config(config, output_dir)
    logger.info(f"Saved configuration to {output_dir}")

    # Create list to store results
    valid_tropomi_list = []
    
    # Set number of processes based on args or available CPUs
    if args.num_cpus:
        num_processes = min(args.num_cpus, mp.cpu_count())
        logger.info(f"Using {num_processes} CPUs as specified (system has {mp.cpu_count()} CPUs)")
    else:
        num_processes = mp.cpu_count()
        logger.info(f"Using all available CPUs: {num_processes}")
    
    # Create argument list for multiprocessing with config
    process_args = [(file, emissions, cities, i % num_processes, config) for i, file in enumerate(files)]
    
    # Use smaller chunks to get more frequent updates
    with mp.Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file_wrapper, process_args, chunksize=1)):
            valid_tropomi_list.extend(result)
            logger.info(f"Completed {i+1}/{len(files)} files. Total valid plants so far: {len(valid_tropomi_list)}")
    
    # Convert to DataFrame
    logger.info("Creating final DataFrame")
    valid_tropomi_df = pd.DataFrame(valid_tropomi_list)
    
    valid_tropomi_df.dropna(inplace=True)
    # Save the final dataframe to the run-specific directory
    output_path = os.path.join(output_dir, f"valid_tropomi_emissions_with_qa.csv")
    logger.info(f"Writing {len(valid_tropomi_df)} data points to {output_path}")
    valid_tropomi_df.to_csv(output_path, index=False)

    total_processing_time = time.time() - total_start_time
    logger.info("=" * 80)
    logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
    logger.info(f"Total valid TROPOMI data points: {len(valid_tropomi_df)}")
    logger.info(f"Data saved to: {output_path}")
    logger.info(f"Configuration saved to: {os.path.join(output_dir, 'config.json')}")
    logger.info("=" * 80)
    
if __name__ == "__main__":
    main()