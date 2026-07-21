import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import time
from datetime import datetime
from sklearn.neighbors import BallTree
import multiprocessing as mp

def process_file_group(args):
    file_path, records = args
    t0 = time.time()
    print(f"▶ Processing {os.path.basename(file_path)} ({len(records)} recs)")
    
    ds = nc.Dataset(file_path)
    prod = ds.groups['PRODUCT']
    inp  = prod.groups['SUPPORT_DATA'].groups['INPUT_DATA']
    geo  = prod.groups['SUPPORT_DATA'].groups['GEOLOCATIONS']

    # load lat/lon and build BallTree
    lats = prod.variables['latitude'][0]
    lons = prod.variables['longitude'][0]
    flat = np.radians(np.column_stack((lats.flatten(), lons.flatten())))
    tree = BallTree(flat, metric='haversine')

    # slice off time=0 for each variable
    vars_2d = {
        'surface_altitude':                       inp.variables['surface_altitude'][0],
        'surface_altitude_precision':             inp.variables['surface_altitude_precision'][0],
        'surface_classification':                 inp.variables['surface_classification'][0],
        'surface_pressure':                       inp.variables['surface_pressure'][0],
        'surface_albedo':                         inp.variables['surface_albedo'][0],
        'surface_albedo_nitrogendioxide_window':  inp.variables['surface_albedo_nitrogendioxide_window'][0],
        'cloud_pressure_crb':                     inp.variables['cloud_pressure_crb'][0],
        'cloud_fraction_crb':                     inp.variables['cloud_fraction_crb'][0],
        'cloud_albedo_crb':                       inp.variables['cloud_albedo_crb'][0],
        'scene_albedo':                           inp.variables['scene_albedo'][0],
        'apparent_scene_pressure':                inp.variables['apparent_scene_pressure'][0],
        'snow_ice_flag':                          inp.variables['snow_ice_flag'][0],
        'aerosol_index_354_388':                  inp.variables['aerosol_index_354_388'][0],
        'eastward_wind':                          inp.variables['eastward_wind'][0],
        'northward_wind':                         inp.variables['northward_wind'][0],
        'scaled_small_pixel_variance':            inp.variables['scaled_small_pixel_variance'][0],
        'tropospheric_NO2_column_number_density': prod.variables['nitrogendioxide_tropospheric_column'][0],
        'sensor_altitude':                        np.broadcast_to(geo.variables['satellite_altitude'][0][:,None], lats.shape),
        'sensor_azimuth_angle':                   geo.variables['viewing_azimuth_angle'][0],
        'sensor_zenith_angle':                    geo.variables['viewing_zenith_angle'][0],
        'solar_azimuth_angle':                    geo.variables['solar_azimuth_angle'][0],
        'solar_zenith_angle':                     geo.variables['solar_zenith_angle'][0],
    }
    print(f" • Loaded vars: {', '.join(vars_2d.keys())}")

    out = []
    for rec in records:
        lat = rec['latitude']
        lon = rec['longitude']
        point = np.radians([[lat, lon]])
        idx = tree.query(point, k=1)[1][0][0]
        gi = np.unravel_index(idx, lats.shape)

        new_rec = rec.copy()
        for name, arr2d in vars_2d.items():
            val = arr2d[gi]
            new_rec[name] = float(np.ma.filled(val, np.nan))
        out.append(new_rec)

    ds.close()

    elapsed = time.time() - t0
    print(f"✅ Done {os.path.basename(file_path)} in {elapsed:.1f}s")
    
    return out

def main():
    mp.set_start_method('fork')
    input_csv  = '/net/fs06/d3/rzhuang/TROPOMI_US/data/Run_600/processed_valid_tropomi_emissions_with_qa_updated.csv'
    output_csv = '/net/fs06/d3/rzhuang/TROPOMI_US/data/Run_600/valid_tropomi_emissions_with_qa_with_all_vars.csv'
    df = pd.read_csv(input_csv)
    groups = []
    for file_path, grp in df.groupby('file_path'):
        records = grp.to_dict('records')
        groups.append((file_path, records))
    print(f"Starting pool with {mp.cpu_count()} workers")
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_file_group, groups)
    # flatten list of lists
    all_records = [rec for sub in results for rec in sub]
    out_df = pd.DataFrame(all_records)
    out_df.to_csv(output_csv, index=False)

    print(f"Written {len(all_records)} records")

if __name__ == '__main__':
    main()