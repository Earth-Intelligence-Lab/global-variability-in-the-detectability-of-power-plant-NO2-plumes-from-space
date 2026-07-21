import numpy as np
import pandas as pd # Need pandas for pd.notna
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Polygon, Patch, Wedge # Import Polygon and Patch, Wedge for sector
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter # Import ticker for formatting coordinate labels
from pyproj import Geod # <--- Need Geod for accurate circles/sectors
from scipy.ndimage import gaussian_filter
import contextily as ctx
import os
import warnings # Import the standard warnings module
from haversine import haversine, Unit # <--- Need haversine for nearby plant filtering
from sklearn.neighbors import BallTree # <--- Need BallTree for interference prep
from math import radians, log10, sqrt # <--- Need math functions for interference prep
from sklearn.linear_model import RANSACRegressor

# --- Constants ---
EARTH_RADIUS_KM = 6371

# --- Matplotlib Default Style Reset ---
plt.rcdefaults() # Reset styles to default to avoid conflicts if run multiple times

# --- Helper Functions (compact, same output) ---
def calculate_areas(mask, lat_2d, lon_2d):
    """Estimate total plume area (km²) for TROPOMI pixels flagged True in mask."""
    plume_y, plume_x = np.where(mask)
    pixel_areas = []
    deg2km = 111.14

    for y, x in zip(plume_y, plume_x):
        area_est = []
        for dx in (1, -1):                         # right / left neighbours for dlon
            nx = x + dx
            if 0 <= nx < lon_2d.shape[1]:
                dlon = abs(lon_2d[y, nx] - lon_2d[y, x])
                for dy in (1, -1):                 # bottom / top neighbours for dlat
                    ny = y + dy
                    if 0 <= ny < lat_2d.shape[0]:
                        dlat = abs(lat_2d[ny, x] - lat_2d[y, x])
                        lat_km = dlat * deg2km
                        lon_km = dlon * deg2km * np.cos(np.radians(lat_2d[y, x]))
                        area_est.append(lat_km * lon_km)
        pixel_areas.append(np.mean(area_est) if area_est else 25.0)  # fallback 25 km²

    return float(np.sum(pixel_areas)) if pixel_areas else 0.0


def create_geodesic_circle(center_lon, center_lat, radius_km, geod_obj, num_points=100):
    """Return lon/lat arrays of points on a geodesic circle."""
    radius_m = radius_km * 1000.0
    az = np.linspace(0, 360, num_points)
    lons, lats, _ = geod_obj.fwd(np.full(num_points, center_lon),
                                 np.full(num_points, center_lat),
                                 az,
                                 np.full(num_points, radius_m))
    return lons.tolist(), lats.tolist()


# --- Interference Prep Helper (compact, same output) ---

def _process_source_dataframe(df_raw, lat_col, lon_col, id_col, value_col,
                              std_lat, std_lon, std_id, std_value,
                              value_required=False):
    """Standardise dataframe and build BallTree for interference calculations."""
    if df_raw is None or df_raw.empty:
        return None, None

    try:
        df = df_raw.copy()
        col_map = {lat_col: std_lat, lon_col: std_lon}
        essentials = [lat_col, lon_col]

        if id_col and id_col in df:          # optional ID
            col_map[id_col] = std_id
        if value_col and value_col in df:    # optional value
            col_map[value_col] = std_value
            if value_required:
                essentials.append(value_col)

        missing = [c for c in essentials if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[[c for c in col_map]].rename(columns=col_map)

        # numeric conversion
        for c in (std_lat, std_lon, std_value):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df.dropna(subset=[std_lat, std_lon] + ([std_value] if value_required and std_value in df else []),
                  inplace=True)
        if df.empty:
            return df, None

        # radians + BallTree
        df['lat_rad'] = np.radians(df[std_lat])
        df['lon_rad'] = np.radians(df[std_lon])
        tree = BallTree(df[['lat_rad', 'lon_rad']].values, metric='haversine')
        return df, tree

    except Exception as e:
        print(f"Warn: Error processing source dataframe: {e}")
        return None, None

# --- Interference Prep Function (Refactored, ID‑based exclusion only) ---
def prepare_interfering_sources_build_trees_inside(
    target_lat, target_lon,
    all_plants_df_raw, cities_df_raw,
    max_distance_km=150,
    target_plant_id=None,
    plant_in_id_col='location', plant_in_lat_col='latitude', plant_in_lon_col='longitude',
    plant_in_emission_col='nox_emis_ty',
    city_in_lat_col='latitude', city_in_lon_col='longitude', city_in_pop_col='population',
    city_in_name_col='name', city_in_country_col='country',
    city_base_radius=5.0, city_pop_scale=3, city_radius_min=30.0, city_radius_max=50.0,
    plant_base_radius=0.0, plant_emission_scale=0.5, plant_radius_min=40.0, plant_radius_max=50.0,
    use_plant_emission_scaling=True,
    min_city_population_threshold=200000,
    min_plant_emission_threshold=0.5
):
    """
    Identifies potential interfering sources (cities and other plants) near a target location.
    Builds BallTrees internally for efficient spatial querying. Calculates interference radii.
    Removes plants whose ID matches the target plant ID.
    """
    if pd.isna(target_lat) or pd.isna(target_lon):
        print("Error: Target coords invalid.")
        return []
    if (all_plants_df_raw is None or all_plants_df_raw.empty) and (cities_df_raw is None or cities_df_raw.empty):
        print("Warn: Both input DFs None or empty. No interference calculated.")
        return []

    plant_std_lat, plant_std_lon, plant_std_id, plant_std_emis = 'latitude', 'longitude', 'ID', 'nox_emis_ty'
    city_std_lat, city_std_lon, city_std_name, city_std_pop, city_std_country = \
        'latitude', 'longitude', 'name', 'population', 'country'

    plants_df, plant_tree = _process_source_dataframe(
        all_plants_df_raw, plant_in_lat_col, plant_in_lon_col, plant_in_id_col, plant_in_emission_col,
        plant_std_lat, plant_std_lon, plant_std_id, plant_std_emis, value_required=False
    )
    cities_df, city_tree = _process_source_dataframe(
        cities_df_raw, city_in_lat_col, city_in_lon_col, city_in_name_col, city_in_pop_col,
        city_std_lat, city_std_lon, city_std_name, city_std_pop, value_required=False
    )
    if cities_df is not None and not cities_df.empty and cities_df_raw is not None and city_in_country_col in cities_df_raw.columns:
        try:
            cities_df[city_std_country] = cities_df_raw[city_in_country_col].reindex(cities_df.index).fillna('N/A')
        except Exception:
            cities_df[city_std_country] = 'N/A'

    target_rad = np.array([[radians(target_lat), radians(target_lon)]])
    max_rad = max_distance_km / EARTH_RADIUS_KM

    def clamp(r, lo, hi):
        return max(lo, min(r, hi))

    results = []

    configs = []
    if city_tree is not None and not cities_df.empty:
        configs.append({
            'df': cities_df, 'tree': city_tree, 'type': 'city',
            'lat_col': city_std_lat, 'lon_col': city_std_lon,
            'id_col': city_std_name, 'extra_cols': [city_std_country],
            'threshold_col': city_std_pop, 'threshold': min_city_population_threshold,
            'base': city_base_radius, 'scale_fn': lambda v: city_pop_scale * log10(max(1, v)),
            'min_r': city_radius_min, 'max_r': city_radius_max
        })
    if plant_tree is not None and not plants_df.empty:
        configs.append({
            'df': plants_df, 'tree': plant_tree, 'type': 'plant',
            'lat_col': plant_std_lat, 'lon_col': plant_std_lon,
            'id_col': plant_std_id, 'extra_cols': [],
            'threshold_col': plant_std_emis, 'threshold': min_plant_emission_threshold,
            'base': plant_base_radius,
            'scale_fn': lambda v: (sqrt(max(0, v)) * plant_emission_scale) if use_plant_emission_scaling else 0,
            'min_r': plant_radius_min, 'max_r': plant_radius_max,
            'exclude_target_id': target_plant_id
        })

    for cfg in configs:
        idx = cfg['tree'].query_radius(target_rad, r=max_rad)[0]
        if len(idx) == 0:
            continue
        df_near = cfg['df'].iloc[idx].copy()

        if cfg['type'] == 'plant' and cfg.get('exclude_target_id') is not None:
            df_near = df_near[df_near[cfg['id_col']].astype(str) != str(cfg['exclude_target_id'])]

        if cfg['threshold_col'] in df_near.columns:
            vals = pd.to_numeric(df_near[cfg['threshold_col']], errors='coerce').fillna(0)
            df_near = df_near[vals >= cfg['threshold']]

        for _, row in df_near.iterrows():
            val = row.get(cfg['threshold_col'], 0) or 0
            r = cfg['base']
            if val > 0:
                r += cfg['scale_fn'](val)
            final_r = clamp(r, cfg['min_r'], cfg['max_r'])
            src = {
                'type': cfg['type'],
                'lat': row[cfg['lat_col']],
                'lon': row[cfg['lon_col']],
                'radius_km': final_r,
                'name': row.get(cfg['id_col'], 'Unknown')
            }
            for col in cfg.get('extra_cols', []):
                src[col] = row.get(col)
            results.append(src)

    return results


# --- Plume Labeling Function (Refactored for Geometry Calculation) ---
def label_no2_plume_flexible_interference(
    full_no2, full_lon, full_lat,
    wind_u, wind_v, plant_lon, plant_lat,
    interfering_sources=None, # Used to generate interference_mask
    zoom_radius_km=100,
    threshold_factor=4.5, threshold_abs_min=1e-5, threshold_radius_km=25.0,
    max_distance_km=20, close_distance_km=8, max_angle_diff=30, max_angle_diff_mask=90, close_distance_km_mask=2,
    flagged_area=10.0, background_mode='directional', upwind_angle_tolerance=90,
    background_dist_min_km=10, background_dist_max_km=75,
    sigma=10, stat_radius=50.0
    ):
    """
    Enhanced plume labeling using flexible interference handling.
    REFACTORED: Calculates geometry (distance/azimuth) from target plant to grid points once.
    FIXED: Calculates wind direction BEFORE interference mask calculation.
    FIXED: Uses wind_dir_to_deg and max_angle_diff_mask for plant interference cone mask.
    FIXED: Ensures the final plume mask explicitly excludes pixels within interference zones.
    Returns a dictionary containing results needed for visualization.
    """
    # ------------------------------------------------------------------
    # helpers -----------------------------------------------------------
    # ------------------------------------------------------------------
    geod = Geod(ellps="WGS84")
    tol = 1e-12

    def _norm(a):  # angle → [0,360)
        return (a + 360) % 360

    def _ang_diff(a, b):  # |shortest signed diff|
        return np.abs(((a - b + 180) % 360) - 180)

    def _safe_stat(arr, fn=np.nanmedian):
        """median fallback → mean; returns NaN if both fail"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            val = fn(arr)
            if pd.isna(val):
                val = np.nanmean(arr)
        return val

    def _zoom_indices(mask, pad, shp):
        if not np.any(mask):
            return None
        r, c = np.where(mask)
        return (
            max(0, r.min() - pad), min(shp[0] - 1, r.max() + pad),
            max(0, c.min() - pad), min(shp[1] - 1, c.max() + pad),
        )

    # ------------------------------------------------------------------
    # default return dict ----------------------------------------------
    # ------------------------------------------------------------------
    default = {
        "is_significant": False,
        "plume_mask": np.array([[]], bool),
        "background": np.array([[]]),
        "anomalies": np.array([[]]),
        "threshold": threshold_abs_min,
        "plume_area_km2": 0.0,
        "plume_direction_to_deg": np.nan,
        "wind_direction_from_deg": np.nan,
        "zoomed_lats": np.array([[]]),
        "zoomed_lons": np.array([[]]),
        "zoomed_no2": np.array([[]]),
        "lon_min": np.nan,
        "lon_max": np.nan,
        "lat_min": np.nan,
        "lat_max": np.nan,
        "interference_mask": np.array([[]], bool),
        "background_mode_used": background_mode,
        "bg_params": {
            "upwind_angle_tolerance": upwind_angle_tolerance,
            "background_dist_min_km": background_dist_min_km,
            "background_dist_max_km": background_dist_max_km,
            "sigma": sigma,
        },
        # placeholders for radius km NO₂ stats
        "no2_mean_radius": np.nan,
        "no2_std_radius": np.nan,
        "no2_frac_valid_radius": np.nan,
    }

    try:
        # ------------------------------------------------------------------
        # (0) basic validation ---------------------------------------------
        # ------------------------------------------------------------------
        if not all(isinstance(x, np.ndarray) for x in (full_lat, full_lon, full_no2)):
            raise ValueError("Inputs must be numpy arrays.")
        if not (full_lat.shape == full_lon.shape == full_no2.shape):
            raise ValueError("Input shapes mismatch.")
        if pd.isna(plant_lat) or pd.isna(plant_lon):
            raise ValueError("Plant coordinates invalid.")

        # ------------------------------------------------------------------
        # (1) zoom ----------------------------------------------------------
        # ------------------------------------------------------------------
        dlat = zoom_radius_km / 111.0
        dlon = zoom_radius_km / (111.0 * max(np.cos(np.radians(plant_lat)), 1e-9))
        box_mask = (
            (full_lat >= plant_lat - dlat)
            & (full_lat <= plant_lat + dlat)
            & (full_lon >= plant_lon - dlon)
            & (full_lon <= plant_lon + dlon)
        )
        idx = _zoom_indices(box_mask, pad=2, shp=full_no2.shape)
        if idx is None:
            print("Warn: No data in zoom box.")
            return default
        i0, i1, j0, j1 = idx
        no2 = full_no2[i0 : i1 + 1, j0 : j1 + 1].copy()
        lat = full_lat[i0 : i1 + 1, j0 : j1 + 1].copy()
        lon = full_lon[i0 : i1 + 1, j0 : j1 + 1].copy()
        if no2.size == 0:
            return default
        h, w = no2.shape
        default.update(
            {
                "zoomed_no2": no2,
                "zoomed_lats": lat,
                "zoomed_lons": lon,
                "lat_min": np.nanmin(lat),
                "lat_max": np.nanmax(lat),
                "lon_min": np.nanmin(lon),
                "lon_max": np.nanmax(lon),
            }
        )

        # ------------------------------------------------------------------
        # (1a) Compute radius km NO₂ stats on the zoomed patch only -------------
        # ------------------------------------------------------------------
        flat_no2  = no2.ravel()
        flat_lats = lat.ravel()
        flat_lons = lon.ravel()
        tol_stat  = tol
        _, _, d_m = geod.inv(
            np.full_like(flat_lons, plant_lon),
            np.full_like(flat_lats, plant_lat),
            flat_lons,
            flat_lats
        )
        d_km         = d_m / 1000.0
        within_radius     = d_km <= stat_radius
        valid_pixels = np.isfinite(flat_no2) & (np.abs(flat_no2) > tol_stat)
        vals_radius       = flat_no2[within_radius & valid_pixels]
        mask2d       = (within_radius & valid_pixels).reshape(no2.shape)
        valid_area_radius = calculate_areas(mask2d, lat, lon)
        total_area_radius = np.pi * stat_radius**2
        mean_radius       = np.nan if vals_radius.size == 0 else np.mean(vals_radius)
        frac_radius       = valid_area_radius / total_area_radius if total_area_radius > 0 else np.nan
        # Apply RANSAC to remove outliers and compute std
        if vals_radius.size == 0:
            std_radius = np.nan
        else:
            X = np.arange(len(vals_radius)).reshape(-1, 1)
            y = vals_radius
            ransac = RANSACRegressor(min_samples=0.5, max_trials=100, random_state=42)
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            std_radius = np.std(y[inlier_mask]) if np.sum(inlier_mask) > 0 else np.nan
        default.update({
            "no2_mean_radius": mean_radius,
            "no2_std_radius": std_radius,
            "no2_frac_valid_radius": frac_radius
        })
        
        # ------------------------------------------------------------------
        # (2) geometry ------------------------------------------------------
        # ------------------------------------------------------------------
        geo_mask = ~np.isnan(lat) & ~np.isnan(lon)
        dist_km = np.full_like(no2, np.nan, float)
        azm = np.full_like(no2, np.nan, float)
        if np.any(geo_mask):
            # Fix: Convert to flat arrays with correct length matching
            lat_flat = lat[geo_mask].flatten()
            lon_flat = lon[geo_mask].flatten()
            n_points = len(lat_flat)
            
            # Create arrays of constant values with the same length
            plant_lons = np.full(n_points, plant_lon)
            plant_lats = np.full(n_points, plant_lat)
            
            fwd, _, d_m = geod.inv(
                plant_lons,
                plant_lats,
                lon_flat,
                lat_flat
            )
            dist_km[geo_mask] = d_m / 1000.0
            azm[geo_mask] = _norm(fwd)

        # ------------------------------------------------------------------
        # (3) wind direction -----------------------------------------------
        # ------------------------------------------------------------------
        u = np.nanmean(wind_u) if isinstance(wind_u, (np.ndarray, list, pd.Series)) else wind_u
        v = np.nanmean(wind_v) if isinstance(wind_v, (np.ndarray, list, pd.Series)) else wind_v
        if pd.isna(u) or pd.isna(v) or (abs(u) < tol and abs(v) < tol):
            wind_to, wind_from = np.nan, np.nan
        else:
            wind_to = _norm(np.degrees(np.arctan2(u, v)))
            wind_from = _norm(wind_to + 180)
        default["plume_direction_to_deg"] = wind_to
        default["wind_direction_from_deg"] = wind_from

        # ------------------------------------------------------------------
        # (4) masks ---------------------------------------------------------
        # ------------------------------------------------------------------
        valid_no2 = (~np.isnan(no2)) & (np.abs(no2) > tol) & geo_mask
        interference = np.ones_like(no2, bool)  # True → allowed

        if interfering_sources and np.any(geo_mask):
            lon_flat, lat_flat = lon[geo_mask], lat[geo_mask]
            for src in interfering_sources:
                slat, slon, rad = src.get("lat"), src.get("lon"), src.get("radius_km")
                if pd.isna(slat) or pd.isna(slon) or pd.isna(rad):
                    continue
                az1, _, d_m = geod.inv(np.full_like(lon_flat, slon), np.full_like(lat_flat, slat), lon_flat, lat_flat)
                d_km = d_m / 1000.0
                in_rad = d_km <= rad
                if src.get("type", "city").lower() == "plant":
                    close = d_km <= close_distance_km_mask
                    if pd.notna(wind_to):
                        in_cone = _ang_diff(az1, wind_to) <= max_angle_diff_mask
                        in_rad = (in_rad & in_cone) | close
                    else:
                        in_rad = in_rad | close
                interference[geo_mask] &= ~in_rad
        default["interference_mask"] = interference

        calc_mask = valid_no2 & interference
        no2_calc = no2.copy()
        no2_calc[~calc_mask] = np.nan

        # ------------------------------------------------------------------
        # (5) background ----------------------------------------------------
        # ------------------------------------------------------------------
        mode = background_mode
        if mode == "directional" and pd.isna(wind_from):
            print("Warn: No wind dir – falling back to gaussian background.")
            mode = "gaussian"
        default["background_mode_used"] = mode

        background = np.full_like(no2, np.nan, float)
        flat_valid_idx = np.where(calc_mask.ravel())[0]

        if flat_valid_idx.size == 0:
            # all blocked – fallback to median of any valid NO2
            fallback = _safe_stat(no2[valid_no2])
            if pd.notna(fallback):
                background.fill(fallback)
        else:
            dist_v = dist_km.ravel()[flat_valid_idx]
            az_v = azm.ravel()[flat_valid_idx]
            no2_v = no2_calc.ravel()[flat_valid_idx]
            geom_ok = (~np.isnan(dist_v)) & (~np.isnan(az_v))

            if mode == "directional" and np.any(geom_ok):
                dist_ok, az_ok, no2_ok = dist_v[geom_ok], az_v[geom_ok], no2_v[geom_ok]
                sector = (
                    (_ang_diff(az_ok, wind_from) <= upwind_angle_tolerance / 2)
                    & (dist_ok >= background_dist_min_km)
                    & (dist_ok <= background_dist_max_km)
                )
                bg_val = _safe_stat(no2_ok[sector]) if sector.any() else np.nan
                if pd.isna(bg_val):
                    bg_val = _safe_stat(no2_v)
                if pd.notna(bg_val):
                    background.fill(bg_val)
            if mode == "gaussian" or pd.isna(background).all():
                msk = calc_mask.astype(float)
                nz = np.nan_to_num(no2_calc)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    wsum = gaussian_filter(nz * msk, sigma, mode="constant", cval=0.0)
                    wnorm = gaussian_filter(msk, sigma, mode="constant", cval=0.0)
                    bg = np.divide(wsum, wnorm, out=np.full_like(wsum, np.nan), where=wnorm > 1e-9)
                fill_val = _safe_stat(bg)
                if pd.notna(fill_val):
                    bg[np.isnan(bg)] = fill_val
                background = bg if not np.isnan(bg).all() else background

        default["background"] = background

        # ------------------------------------------------------------------
        # (6) anomalies & threshold ----------------------------------------
        # ------------------------------------------------------------------
        anomalies = no2 - background
        anomalies[~valid_no2] = np.nan
        default["anomalies"] = anomalies

        loc_mask = (dist_km <= threshold_radius_km) & ~np.isnan(dist_km) & ~np.isnan(anomalies)
        local_vals = anomalies[loc_mask]
        if local_vals.size > 5:
            med = _safe_stat(local_vals)
            mad = _safe_stat(np.abs(local_vals - med))
            if mad > tol:
                thresh = max(med + threshold_factor * mad * 1.4826, threshold_abs_min)
            else:
                std = np.nanstd(local_vals)
                thresh = max(med + threshold_factor * std, threshold_abs_min)
        else:
            thresh = threshold_abs_min
        default["threshold"] = thresh

        # ------------------------------------------------------------------
        # (7) plume detection ----------------------------------------------
        # ------------------------------------------------------------------
        plume_mask = np.zeros_like(no2, bool)
        valid_idx = np.where(valid_no2.ravel())[0]
        if valid_idx.size:
            d_all = dist_km.ravel()[valid_idx]
            az_all = azm.ravel()[valid_idx]
            an_all = anomalies.ravel()[valid_idx]
            usable = (~np.isnan(d_all)) & (~np.isnan(az_all)) & (~np.isnan(an_all))
            if usable.any():
                d_u, az_u, an_u = d_all[usable], az_all[usable], an_all[usable]
                if pd.notna(wind_to):
                    cond = (
                        ((d_u <= max_distance_km) & (an_u > thresh) & (_ang_diff(az_u, wind_to) <= max_angle_diff))
                        | ((d_u <= close_distance_km) & (an_u > thresh))
                    )
                else:
                    cond = (d_u <= close_distance_km) & (an_u > thresh)
                plume_idx = valid_idx[usable][cond]
                plume_mask.ravel()[plume_idx] = True
        plume_mask &= interference  # ensure not in interference zones
        default["plume_mask"] = plume_mask

        # ------------------------------------------------------------------
        # (8) area & significance ------------------------------------------
        # ------------------------------------------------------------------
        area = calculate_areas(plume_mask, lat, lon)
        default["plume_area_km2"] = area
        default["is_significant"] = area >= flagged_area

        return default

    except Exception as e:
        print(f"Error during plume labeling: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return default


def _load_s5p_no2(file_path: str):
    """Load NO₂, lat, lon from a S5P NetCDF file (handles PRODUCT / root groups)."""
    with nc.Dataset(file_path) as ds:
        if "PRODUCT" in ds.groups and "latitude" in ds.groups["PRODUCT"].variables:
            grp = ds.groups["PRODUCT"]
        else:
            grp = ds
        lat = grp["latitude"][:]
        lon = grp["longitude"][:]
        no2 = grp["nitrogendioxide_tropospheric_column"][:]

        # remove leading singleton dims (time / scanline)
        while lat.ndim > 2 and lat.shape[0] == 1:
            lat, lon, no2 = lat[0], lon[0], no2[0]

        if lat.ndim != 2:
            raise ValueError("Expected 2‑D lat/lon.")

        fv = getattr(grp["nitrogendioxide_tropospheric_column"], "_FillValue", None)
        no2 = np.ma.masked_values(no2, fv) if fv is not None else np.ma.masked_invalid(no2)
        return np.array(no2.filled(np.nan)), np.array(lat), np.array(lon)

# --- Visualization Function (MODIFIED TO USE CORRECT WIND DIRECTIONS) ---
def create_zoomed_plume_visualization(file_path, plant_lat, plant_lon, wind_u, wind_v, utc_time,
                                      annual_nox_ty,
                                      nearby_plant_stats,
                                      nearby_city_stats,
                                      all_locations_df=None,
                                      cities_df=None,
                                      zoom_radius_km=100,  facility_id=None, facility_country=None,
                                      threshold_factor=4.5, threshold_abs_min=1e-5,
                                      max_distance_km=20, close_distance_km=8,
                                      max_angle_diff=30, max_angle_diff_mask=90, close_distance_km_mask=2,
                                      flagged_area=10.0, sigma=10,
                                      threshold_radius_km=25.0, background_mode='directional',
                                      upwind_angle_tolerance=90, background_dist_min_km=10, background_dist_max_km=75,
                                      interf_max_distance_km=150,
                                      interf_city_pop_thresh=200000,
                                      interf_plant_emis_thresh=0.5,
                                      city_base_radius=5.0, city_pop_scale=3,
                                      city_radius_min=30.0, city_radius_max=50.0,
                                      plant_base_radius=0.0, plant_emission_scale=0.5,
                                      plant_radius_min=40.0, plant_radius_max=50.0,
                                      stat_radius=50.0, # Radius for radius km NO₂ stats
                                      # New parameter to control interference plotting
                                      plot_interference_zones=True, # Changed name for clarity
                                      plot_dpi=200):
    """
    Create a comprehensive 6-plot visualization including NO2 concentration,
    plume overlay, background, anomalies, satellite, and urban maps.
    MODIFIED: Plots plant interference zones as CONES (sector outlines) using wind direction
              and city interference zones as CIRCLES on the Background and Anomalies plots.
    MODIFIED: Uses correctly calculated wind FROM and TO directions retrieved from the results dict.
    """
    geod = Geod(ellps="WGS84")

    # --- Load satellite data ---
    try:
        no2, lats, lons = _load_s5p_no2(file_path)
    except Exception as exc:
        fig, axs = plt.subplots(2, 3, figsize=(18, 11), dpi=plot_dpi)
        fig.suptitle(f"Error loading data for {facility_name or 'Facility'}: {exc}", color="red")
        for ax in axs.flat:
            ax.axis("off")
        plt.close(fig)
        return None

    facility_name = (facility_id if not facility_country or facility_id == facility_country
                      else f"{facility_id} ({facility_country})")
    datetime_str = utc_time or "Unknown Date"

    # --- Process interference sources and detect plume ---
    interfering_sources = prepare_interfering_sources_build_trees_inside(
        target_lat=plant_lat, target_lon=plant_lon,
        all_plants_df_raw=all_locations_df, cities_df_raw=cities_df,
        max_distance_km=interf_max_distance_km, target_plant_id=facility_id,
        plant_in_id_col="ID", plant_in_lat_col="latitude", plant_in_lon_col="longitude",
        plant_in_emission_col="nox_emis_ty", city_in_lat_col="latitude", city_in_lon_col="longitude",
        city_in_pop_col="population", city_in_name_col="name", city_in_country_col="country",
        city_base_radius=city_base_radius, city_pop_scale=city_pop_scale, 
        city_radius_min=city_radius_min, city_radius_max=city_radius_max,
        plant_base_radius=plant_base_radius, plant_emission_scale=plant_emission_scale,
        plant_radius_min=plant_radius_min, plant_radius_max=plant_radius_max,
        min_city_population_threshold=interf_city_pop_thresh,
        min_plant_emission_threshold=interf_plant_emis_thresh * annual_nox_ty,
    )

    results = label_no2_plume_flexible_interference(
        full_no2=no2, full_lon=lons, full_lat=lats,
        wind_u=wind_u, wind_v=wind_v, plant_lon=plant_lon, plant_lat=plant_lat,
        interfering_sources=interfering_sources, zoom_radius_km=zoom_radius_km,
        threshold_factor=threshold_factor, threshold_abs_min=threshold_abs_min,
        threshold_radius_km=threshold_radius_km, max_distance_km=max_distance_km,
        close_distance_km=close_distance_km, max_angle_diff=max_angle_diff,
        max_angle_diff_mask=max_angle_diff_mask, close_distance_km_mask=close_distance_km_mask,
        flagged_area=flagged_area, background_mode=background_mode,
        upwind_angle_tolerance=upwind_angle_tolerance,
        background_dist_min_km=background_dist_min_km, background_dist_max_km=background_dist_max_km,
        sigma=sigma, stat_radius=stat_radius
    )

    # --- Extract and validate data from results ---
    zoomed_no2 = results.get("zoomed_no2")
    zoomed_lats = results.get("zoomed_lats")
    zoomed_lons = results.get("zoomed_lons")
    if not all(isinstance(x, np.ndarray) and x.size for x in (zoomed_no2, zoomed_lats, zoomed_lons)):
        fig, axs = plt.subplots(2, 3, figsize=(18, 11), dpi=plot_dpi)
        fig.suptitle(f"No valid TROPOMI data for {facility_name}", color="orange")
        for ax in axs.flat:
            ax.axis("off")
        plt.close(fig)
        return None

    # --- Setup figure and extract key parameters ---
    fig, axs = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True, dpi=plot_dpi)
    wind_u_val = np.nanmean(wind_u) if isinstance(wind_u, (np.ndarray, list, pd.Series)) else wind_u
    wind_v_val = np.nanmean(wind_v) if isinstance(wind_v, (np.ndarray, list, pd.Series)) else wind_v
    wind_speed_val = np.sqrt(wind_u_val**2 + wind_v_val**2) if pd.notna(wind_u_val) and pd.notna(wind_v_val) else np.nan

    wind_dir_plume_to_deg = results.get("plume_direction_to_deg", np.nan)
    wind_dir_from_deg = results.get("wind_direction_from_deg", np.nan)
    is_significant = results.get("is_significant", False)
    plume_area_km2 = results.get("plume_area_km2", 0.0)
    plume_threshold = results.get("threshold", threshold_abs_min)
    bg_mode_used = results.get("background_mode_used", background_mode)

    # --- Figure title ---
    title = (
        f"Facility: {facility_name} | {datetime_str}\n"
        f"{f'Annual NOx: {annual_nox_ty:.1f} t/yr | ' if annual_nox_ty is not None else 'Annual NOx: N/A t/yr | '}"
        f"Wind: {wind_dir_from_deg:.1f}° @ {wind_speed_val:.2f} m/s | "
        f"{'Plume Detected' if is_significant else 'No Significant Plume'} "
        f"(Area {plume_area_km2:.1f} km², Thresh {plume_threshold:.2e}) "
    )
    fig.suptitle(title, fontsize=12)

    # --- Extract data and setup color scales ---
    background = results.get("background", np.full_like(zoomed_no2, np.nan))
    anomalies = results.get("anomalies", np.full_like(zoomed_no2, np.nan))
    lon_min, lon_max = results.get("lon_min"), results.get("lon_max")
    lat_min, lat_max = results.get("lat_min"), results.get("lat_max")

    valid = np.concatenate([arr[np.isfinite(arr)] for arr in (zoomed_no2, background) if isinstance(arr, np.ndarray)])
    vmin_no2, vmax_no2 = (np.percentile(valid, [5, 99.5]) if valid.size else (0, 1e-5))
    if vmin_no2 >= vmax_no2:
        eps = abs(vmin_no2 * 0.1) or 1e-9
        vmin_no2, vmax_no2 = vmin_no2 - eps, vmax_no2 + eps

    norm_no2 = colors.Normalize(vmin=vmin_no2, vmax=vmax_no2)
    an_lim = abs(plume_threshold) or 1e-6
    norm_anom = colors.Normalize(vmin=-an_lim, vmax=an_lim)
    cmap_no2, cmap_anom = 'viridis', 'coolwarm'
    plot_extent_valid = np.isfinite(lon_min) and np.isfinite(lon_max) and np.isfinite(lat_min) and np.isfinite(lat_max)

    # --- Background parameters ---
    bg_params = results.get('bg_params', {
        'upwind_angle_tolerance': upwind_angle_tolerance,
        'background_dist_min_km': background_dist_min_km,
        'background_dist_max_km': background_dist_max_km,
        'sigma': sigma
    })
    if not isinstance(bg_params, dict):
        bg_params = {
            'upwind_angle_tolerance': upwind_angle_tolerance,
            'background_dist_min_km': background_dist_min_km,
            'background_dist_max_km': background_dist_max_km,
            'sigma': sigma
        }

    # --- Helper function for plotting data ---
    def plot_data_map(ax, data, cmap, norm, title, cbar_label):
        if not isinstance(data, np.ndarray) or data.size == 0 or data.ndim != 2:
            ax.set_title(f"{title}\n(No Data)", fontsize=10, color='red')
            if plot_extent_valid: ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
            return None
        
        data_masked = np.ma.masked_invalid(data)
        if data_masked.mask.all():
            ax.set_title(f"{title}\n(All Masked)", fontsize=10, color='orange')
            if plot_extent_valid: ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
            return None
            
        im = ax.pcolormesh(zoomed_lons, zoomed_lats, data_masked, cmap=cmap, norm=norm, shading='auto', zorder=2)
        ax.set_title(title, fontsize=10)
        cb = plt.colorbar(im, ax=ax, label=cbar_label, orientation='vertical', pad=0.02, fraction=0.046)
        cb.ax.tick_params(labelsize=8); cb.set_label(cbar_label, size=9)
        return im

    # --- Plot 1: NO2 Concentration & Stats Text ---
    ax_no2 = axs[0, 0]
    plot_data_map(ax_no2, zoomed_no2, cmap_no2, norm_no2, 'NO2 Concentration', 'NO2 (mol/m²)')

    # Build stats text
    stats_text = f"Emission (This Plant): {annual_nox_ty:,.1f} t/yr\n" if pd.notna(annual_nox_ty) else "Emission (This Plant): N/A\n"
    stats_text += "Nearby Plants (Count | Total Emis. | % of Total):\n"

    # Add plant stats
    if nearby_plant_stats:
        for r in [20, 50, 100]:
            stats = nearby_plant_stats.get(r, {})
            count = stats.get('count', np.nan)
            total_emis = stats.get('total_emission', np.nan)
            percent = stats.get('percentage', np.nan)
            count_str = f"{int(count):,}" if pd.notna(count) else "N/A"
            total_emis_str = f"{total_emis:,.1f} t/yr" if pd.notna(total_emis) else "N/A"
            percent_str = f"{percent:.1f}%" if pd.notna(percent) else "N/A"
            stats_text += f"<{r}km: {count_str} | {total_emis_str} | {percent_str}\n"
    else:
        stats_text += "  Stats Unavailable\n"

    # Add city stats
    stats_text += "Nearby Cities (>200k Pop) (Count | Total Pop.):\n"
    if nearby_city_stats:
        for r in [20, 50, 100, 200]:
            stats = nearby_city_stats.get(r, {})
            count = stats.get('count', np.nan)
            pop = stats.get('pop', np.nan)
            count_str = f"{int(count):,}" if pd.notna(count) else "N/A"
            if pd.notna(pop):
                pop_str = f"{pop/1e6:,.1f}M" if pop >= 1e6 else f"{pop/1e3:,.1f}k" if pop >= 1e3 else f"{int(pop):,}"
            else:
                pop_str = "N/A"
            stats_text += f"<{r}km: {count_str} | {pop_str}\n"
    else:
        stats_text += "  Stats Unavailable\n"

    # Add stats text to plot if data is available
    if ax_no2.has_data():
        ax_no2.text(0.03, 0.97, stats_text.strip(), transform=ax_no2.transAxes, fontsize=6,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75), zorder=20)

    # --- Remaining maps ---
    # Plot 4: NO2 Background
    plot_data_map(axs[1, 0], background, cmap_no2, norm_no2, 'NO2 Background', 'NO2 (mol/m²)')

    # Plot 5: NO2 Anomalies
    anom_title = f'NO2 Anomalies (Range ±{an_lim:.1e})'
    plot_data_map(axs[1, 1], anomalies, cmap_anom, norm_anom, anom_title, 'Anomaly (mol/m²)')

    # Plot 2: Detected Plume & Wind Cone
    im2 = plot_data_map(axs[0, 1], zoomed_no2, cmap_no2, norm_no2, 'Detected Plume & Wind Cone', 'NO2 (mol/m²)')

    # Add plume overlay if available
    plume_mask_plot = results.get('plume_mask', np.array([[]], dtype=bool))
    if isinstance(plume_mask_plot, np.ndarray) and plume_mask_plot.shape == zoomed_no2.shape and np.any(plume_mask_plot) and im2 is not None:
        plume_overlay = np.ma.masked_where(~plume_mask_plot, plume_mask_plot)
        axs[0, 1].pcolormesh(zoomed_lons, zoomed_lats, plume_overlay, 
                            cmap=colors.ListedColormap(['lime']), 
                            alpha=0.7, shading='auto', zorder=5, vmin=0, vmax=1)
        
        # Annotate plume centroid if significant
        if is_significant and len(np.where(plume_mask_plot)[0]) > 0:
            y_indices, x_indices = np.where(plume_mask_plot)
            centroid_y_idx, centroid_x_idx = np.mean(y_indices), np.mean(x_indices)
            c_lat_idx = np.clip(int(round(centroid_y_idx)), 0, zoomed_lats.shape[0] - 1)
            c_lon_idx = np.clip(int(round(centroid_x_idx)), 0, zoomed_lats.shape[1] - 1)
            centroid_lat = zoomed_lats[c_lat_idx, c_lon_idx]
            centroid_lon = zoomed_lons[c_lat_idx, c_lon_idx]
            if np.isfinite(centroid_lat) and np.isfinite(centroid_lon):
                axs[0, 1].annotate('Plume', xy=(centroid_lon, centroid_lat), 
                                xytext=(5, 5), textcoords='offset points', 
                                fontsize=9, fontweight='bold', color='black',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7), zorder=10)

    # --- Add basemaps ---
    for i, j, provider, title_str in [(0, 2, ctx.providers.Esri.WorldImagery, 'Satellite Background'),
                                    (1, 2, ctx.providers.OpenStreetMap.Mapnik, 'Urban Area Map')]:
        ax_map = axs[i, j]
        ax_map.set_title(title_str, fontsize=10)
        if plot_extent_valid:
            ax_map.set_xlim(lon_min, lon_max)
            ax_map.set_ylim(lat_min, lat_max)
            try:
                ctx.add_basemap(ax_map, crs='EPSG:4326', source=provider, zoom='auto', attribution_size=5)
            except Exception as e:
                ax_map.text(0.5, 0.5, f'{title_str}\nUnavailable', 
                        ha='center', va='center', transform=ax_map.transAxes, color='red', fontsize=8)
        else:
            ax_map.text(0.5, 0.5, f'{title_str}\n(Invalid Extent)', 
                    ha='center', va='center', transform=ax_map.transAxes, color='red', fontsize=8)

    # --- Filter nearby locations for plotting ---
    nearby_plants_toplot = pd.DataFrame()
    cities_in_extent_df = pd.DataFrame()
    plot_nearby_plant_radius_km = 200

    # Filter plants
    if (all_locations_df is not None and not all_locations_df.empty and 
        'latitude' in all_locations_df.columns and 'longitude' in all_locations_df.columns and 
        pd.notna(plant_lat) and pd.notna(plant_lon) and plot_extent_valid):
        
        temp_plants_df = all_locations_df.copy()
        temp_plants_df['latitude'] = pd.to_numeric(temp_plants_df['latitude'], errors='coerce')
        temp_plants_df['longitude'] = pd.to_numeric(temp_plants_df['longitude'], errors='coerce')
        temp_plants_df.dropna(subset=['latitude', 'longitude'], inplace=True)
        
        if not temp_plants_df.empty:
            # Quick bounding box filter
            lat_pad = plot_nearby_plant_radius_km / 111.0
            lon_pad_denom = abs(np.cos(np.radians(plant_lat)))
            lon_pad = plot_nearby_plant_radius_km / (111.1 * lon_pad_denom if lon_pad_denom > 1e-9 else 111.1)
            
            mask = ((temp_plants_df['longitude'] >= lon_min - lon_pad) & 
                (temp_plants_df['longitude'] <= lon_max + lon_pad) &
                (temp_plants_df['latitude'] >= lat_min - lat_pad) & 
                (temp_plants_df['latitude'] <= lat_max + lat_pad))
            temp_plants_df = temp_plants_df[mask].copy()
            
            if not temp_plants_df.empty:
                try:
                    plant_coords = (plant_lat, plant_lon)
                    temp_plants_df['distance_km'] = temp_plants_df.apply(
                        lambda row: haversine(plant_coords, (row['latitude'], row['longitude']), 
                                            unit=Unit.KILOMETERS) if pd.notna(row['latitude']) and 
                                                                    pd.notna(row['longitude']) else np.nan,
                        axis=1
                    )
                    nearby_mask = temp_plants_df['distance_km'] <= plot_nearby_plant_radius_km
                    main_facility_mask = np.isclose(temp_plants_df['latitude'], plant_lat, atol=1e-5) & \
                                        np.isclose(temp_plants_df['longitude'], plant_lon, atol=1e-5)
                    nearby_plants_toplot = temp_plants_df[nearby_mask & ~main_facility_mask].copy()
                except Exception:
                    pass

    # Filter cities
    if (cities_df is not None and not cities_df.empty and 
        'latitude' in cities_df.columns and 'longitude' in cities_df.columns and plot_extent_valid):
        
        temp_cities_df = cities_df.copy()
        temp_cities_df['latitude'] = pd.to_numeric(temp_cities_df['latitude'], errors='coerce')
        temp_cities_df['longitude'] = pd.to_numeric(temp_cities_df['longitude'], errors='coerce')
        temp_cities_df.dropna(subset=['latitude', 'longitude'], inplace=True)
        
        cities_in_extent_df = temp_cities_df[
            (temp_cities_df['longitude'] >= lon_min) & (temp_cities_df['longitude'] <= lon_max) &
            (temp_cities_df['latitude'] >= lat_min) & (temp_cities_df['latitude'] <= lat_max)
        ].copy()

    # --- Define styles ---
    facility_marker_style = dict(marker='X', s=60, color='red', edgecolor='white', linewidth=0.7, label='Facility', zorder=15)
    nearby_marker_style = dict(marker='^', s=30, color='blue', edgecolor='white', linewidth=0.5, label=f'Other Facility (<{plot_nearby_plant_radius_km}km)', zorder=14, alpha=0.8)
    city_marker_style = dict(marker='.', s=20, color='yellow', edgecolor='black', linewidth=0.3, label='City (in view)', zorder=13, alpha=0.9)
    close_circle_style = dict(color='white', linestyle='-', linewidth=1.5, label=f'{close_distance_km} km Radius', zorder=10)
    max_circle_style = dict(color='grey', linestyle='--', linewidth=1.5, label=f'{max_distance_km} km Radius', zorder=10)
    city_interference_style = dict(color='cyan', linestyle=':', linewidth=1.2, alpha=0.8, zorder=6, label='City Interference Zone')
    plant_interference_style = dict(color='magenta', linestyle=':', linewidth=1.2, alpha=0.8, zorder=6, label='Plant Interference Zone')
    bg_sector_style = {'facecolor':'grey', 'edgecolor':'white', 'alpha':0.15, 'linewidth':0.7, 'linestyle':'-', 'label':'BG Calc Area (Dir.)', 'zorder':4}

    # --- Pre-calculate geometrical elements ---
    close_circle_lons, close_circle_lats = [], []
    max_circle_lons, max_circle_lats = [], []
    outer_circle_lons, outer_circle_lats = [], []
    arrow_dx, arrow_dy = 0, 0
    arrow_style = {}
    plume_cone_poly_verts = []
    plume_cone_style = {}
    bg_poly_verts = []

    if pd.notna(plant_lat) and pd.notna(plant_lon):
        # Distance circles
        close_circle_lons, close_circle_lats = create_geodesic_circle(plant_lon, plant_lat, close_distance_km, geod)
        max_circle_lons, max_circle_lats = create_geodesic_circle(plant_lon, plant_lat, max_distance_km, geod)
        outer_circle_lons, outer_circle_lats = create_geodesic_circle(plant_lon, plant_lat, background_dist_max_km, geod)

        # Wind arrow
        if pd.notna(wind_dir_plume_to_deg) and pd.notna(wind_speed_val) and wind_speed_val > 0.1 and plot_extent_valid:
            cos_plant_lat_rad = max(1e-9, abs(np.cos(np.radians(plant_lat))))
            arrow_len_km = max(close_distance_km * 0.5, min(max_distance_km, zoom_radius_km * 0.1))
            
            arrow_end_lon, arrow_end_lat, _ = geod.fwd(plant_lon, plant_lat, wind_dir_plume_to_deg, arrow_len_km * 1000)
            arrow_dx = arrow_end_lon - plant_lon
            arrow_dy = arrow_end_lat - plant_lat
            
            arrow_len_deg = sqrt((arrow_dx*cos_plant_lat_rad)**2 + arrow_dy**2)
            arrow_head_width = arrow_len_deg * 0.3
            arrow_head_length = arrow_len_deg * 0.5
            
            arrow_style = dict(head_width=max(1e-4, arrow_head_width), 
                            head_length=max(1e-4, arrow_head_length),
                            fc='white', ec='black', linewidth=1, 
                            label='Wind Direction (TO)', length_includes_head=True, zorder=12)

        # Plume cone
        if pd.notna(wind_dir_plume_to_deg):
            start_angle_cone = wind_dir_plume_to_deg - max_angle_diff
            end_angle_cone = wind_dir_plume_to_deg + max_angle_diff
            cone_angles_deg = np.linspace(start_angle_cone, end_angle_cone, 50)
            
            cone_arc_lons, cone_arc_lats, _ = geod.fwd(
                np.full_like(cone_angles_deg, plant_lon), 
                np.full_like(cone_angles_deg, plant_lat),
                cone_angles_deg, 
                np.full_like(cone_angles_deg, max_distance_km * 1000.0)
            )
            
            plume_cone_poly_verts = [(plant_lon, plant_lat)] + list(zip(cone_arc_lons, cone_arc_lats)) + [(plant_lon, plant_lat)]
            plume_cone_style = dict(alpha=0.20, color='lightgrey', ec='darkgrey', lw=0.5, label=f'Plume Angle ({max_angle_diff*2}°)', zorder=3)

        # Background sector
        if bg_mode_used == 'directional' and pd.notna(wind_dir_from_deg):
            try:
                upwind_angle_tol = bg_params.get('upwind_angle_tolerance', upwind_angle_tolerance)
                bg_dist_min = bg_params.get('background_dist_min_km', background_dist_min_km)
                bg_dist_max = bg_params.get('background_dist_max_km', background_dist_max_km)
                
                start_angle = wind_dir_from_deg - upwind_angle_tol / 2.0
                end_angle = wind_dir_from_deg + upwind_angle_tol / 2.0
                angles_deg = np.linspace(start_angle, end_angle, 50)
                
                inner_lons, inner_lats, _ = geod.fwd(
                    np.full_like(angles_deg, plant_lon), 
                    np.full_like(angles_deg, plant_lat), 
                    angles_deg, 
                    np.full_like(angles_deg, bg_dist_min * 1000)
                )
                
                outer_lons, outer_lats, _ = geod.fwd(
                    np.full_like(angles_deg, plant_lon), 
                    np.full_like(angles_deg, plant_lat), 
                    angles_deg[::-1], 
                    np.full_like(angles_deg, bg_dist_max * 1000)
                )
                
                poly_lons = np.concatenate([inner_lons, outer_lons, [inner_lons[0]]])
                poly_lats = np.concatenate([inner_lats, outer_lats, [inner_lats[0]]])
                bg_poly_verts = list(zip(poly_lons, poly_lats))
                
                if len(bg_poly_verts) <= 2:
                    bg_poly_verts = []
            except Exception:
                bg_poly_verts = []

    # --- Plotting loop with tracking flags ---
    bg_sector_polygon_added = False
    plume_cone_polygon_added = False
    city_interference_plotted = False
    plant_interference_plotted = False

    # Helper function for drawing interference zones
    def draw_interference(ax, source_info, is_city=True):
        nonlocal city_interference_plotted, plant_interference_plotted
        
        interf_lat = source_info.get('lat')
        interf_lon = source_info.get('lon')
        interf_rad = source_info.get('radius_km')
        
        if not (pd.notna(interf_lat) and pd.notna(interf_lon)):
            return
            
        try:
            if is_city:
                # Circle for city
                circle_lons, circle_lats = create_geodesic_circle(
                    interf_lon, interf_lat, interf_rad, geod, num_points=60
                )
                if circle_lons:
                    style = city_interference_style.copy()
                    style.pop('label', None)
                    ax.plot(circle_lons, circle_lats, **style)
                    city_interference_plotted = True
            else:
                # Cone for plant with wind direction
                start_angle = wind_dir_plume_to_deg - max_angle_diff_mask
                end_angle = wind_dir_plume_to_deg + max_angle_diff_mask
                radius_m = interf_rad * 1000.0
                
                # Arc points
                arc_azimuths = np.linspace(start_angle, end_angle, 30)
                arc_lons, arc_lats, _ = geod.fwd(
                    np.full(30, interf_lon), 
                    np.full(30, interf_lat), 
                    arc_azimuths, 
                    np.full(30, radius_m)
                )
                
                # Radial lines
                lon1, lat1, _ = geod.fwd(interf_lon, interf_lat, start_angle, radius_m)
                lon2, lat2, _ = geod.fwd(interf_lon, interf_lat, end_angle, radius_m)
                
                # Plot cone
                style = plant_interference_style.copy()
                style.pop('label', None)
                ax.plot([interf_lon, lon1], [interf_lat, lat1], **style)
                ax.plot([interf_lon, lon2], [interf_lat, lat2], **style)
                ax.plot(arc_lons, arc_lats, **style)
                
                # Close distance circle
                circle_lons, circle_lats = create_geodesic_circle(
                    interf_lon, interf_lat, close_distance_km_mask, geod, num_points=60
                )
                if circle_lons:
                    ax.plot(circle_lons, circle_lats, **style)
                
                plant_interference_plotted = True
        except Exception:
            pass

    # Plot on each axis
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            ax = axs[i, j]
            
            # Skip if no data was plotted on this axis
            if not ax.has_data() and not ((i == 0 and j == 2) or (i == 1 and j == 2)):
                continue
                
            # Plot markers
            if pd.notna(plant_lat) and pd.notna(plant_lon):
                ax.scatter(plant_lon, plant_lat, **facility_marker_style)
            if not nearby_plants_toplot.empty:
                ax.scatter(nearby_plants_toplot['longitude'], nearby_plants_toplot['latitude'], **nearby_marker_style)
            if not cities_in_extent_df.empty:
                ax.scatter(cities_in_extent_df['longitude'], cities_in_extent_df['latitude'], **city_marker_style)
            
            # Determine plot type
            is_basemap = (i == 0 and j == 2) or (i == 1 and j == 2)
            is_bg_plot = (i == 1 and j == 0)
            is_anomaly_plot = (i == 1 and j == 1)
            is_plume_plot = (i == 0 and j == 1)
            
            # Draw interference zones
            if plot_interference_zones and (is_bg_plot or is_anomaly_plot) and interfering_sources:
                for source in interfering_sources:
                    draw_interference(ax, source, is_city=(source.get('type') == 'city'))
            
            # Add background sector polygon
            if (is_bg_plot or is_anomaly_plot) and bg_poly_verts:
                bg_sector_polygon = Polygon(bg_poly_verts, closed=True, **bg_sector_style)
                ax.add_patch(bg_sector_polygon)
                bg_sector_polygon_added = True
                
            # Plot circles, plume cone, and arrow on data plots
            if not is_basemap and pd.notna(plant_lat) and pd.notna(plant_lon):
                if i < 2 and j < 2:  # Data plots only
                    if close_circle_lons:
                        ax.plot(close_circle_lons, close_circle_lats, **close_circle_style)
                    if max_circle_lons:
                        ax.plot(max_circle_lons, max_circle_lats, **max_circle_style)
                    if outer_circle_lons:
                        ax.plot(outer_circle_lons, outer_circle_lats, color='grey', linestyle=':', linewidth=0.7, alpha=0.6, zorder=9)
                
                # Wind arrow on first plot only
                if i == 0 and j == 0 and arrow_style and arrow_dx != 0 and arrow_dy != 0:
                    ax.arrow(plant_lon, plant_lat, arrow_dx, arrow_dy, **arrow_style)
                
                # Plume cone on plume and anomaly plots
                if (is_plume_plot or is_anomaly_plot) and plume_cone_poly_verts:
                    cone = Polygon(plume_cone_poly_verts, **plume_cone_style)
                    ax.add_patch(cone)
                    plume_cone_polygon_added = True
            
            # Configure axis appearance
            if plot_extent_valid:
                ax.set_xlim(lon_min, lon_max)
                ax.set_ylim(lat_min, lat_max)
            
            ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6, zorder=1)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            show_xlabel = (i == 1)
            show_ylabel = (j == 0)
            ax.tick_params(axis='x', labelsize=8, labelbottom=show_xlabel, direction='in')
            ax.tick_params(axis='y', labelsize=8, labelleft=show_ylabel, direction='in')
            
            if show_xlabel:
                ax.set_xlabel('Longitude', fontsize=9)
            if show_ylabel:
                ax.set_ylabel('Latitude', fontsize=9)

    # --- Create legend ---
    legend_elements = []

    # Add elements that were actually plotted
    if pd.notna(plant_lat) and pd.notna(plant_lon):
        legend_elements.append(Line2D([0], [0], marker=facility_marker_style['marker'], color='w', 
                                    markerfacecolor=facility_marker_style['color'], 
                                    markeredgecolor=facility_marker_style['edgecolor'], 
                                    markersize=8, label=facility_marker_style['label'], linestyle='None'))

    if not nearby_plants_toplot.empty:
        legend_elements.append(Line2D([0], [0], marker=nearby_marker_style['marker'], color='w', 
                                    markerfacecolor=nearby_marker_style['color'], 
                                    markeredgecolor=nearby_marker_style['edgecolor'], 
                                    markersize=7, label=nearby_marker_style['label'], linestyle='None'))

    if not cities_in_extent_df.empty:
        legend_elements.append(Line2D([0], [0], marker=city_marker_style['marker'], color='w', 
                                    markerfacecolor=city_marker_style['color'], 
                                    markeredgecolor=city_marker_style['edgecolor'], 
                                    markersize=6, label=city_marker_style['label'], linestyle='None'))

    if close_circle_lons:
        legend_elements.append(Line2D([0], [0], color=close_circle_style['color'], 
                                    lw=close_circle_style['linewidth'], 
                                    linestyle=close_circle_style['linestyle'], 
                                    label=close_circle_style['label']))

    if max_circle_lons:
        legend_elements.append(Line2D([0], [0], color=max_circle_style['color'], 
                                    lw=max_circle_style['linewidth'], 
                                    linestyle=max_circle_style['linestyle'], 
                                    label=max_circle_style['label']))

    if is_significant or (isinstance(plume_mask_plot, np.ndarray) and np.any(plume_mask_plot)):
        legend_elements.append(Patch(facecolor='lime', alpha=0.7, label='Detected Plume'))
        
    if arrow_style and arrow_dx != 0 and arrow_dy != 0:
        legend_elements.append(Line2D([0], [0], color='black', markerfacecolor='white', marker='>', 
                                    markersize=7, label=arrow_style['label'], linestyle='None', 
                                    markeredgewidth=1, markeredgecolor='black'))
        
    if plume_cone_polygon_added:
        legend_elements.append(Patch(facecolor=plume_cone_style['color'], 
                                    alpha=plume_cone_style['alpha'], 
                                    edgecolor=plume_cone_style['ec'], 
                                    label=plume_cone_style.get('label', 'Plume Angle')))

    if city_interference_plotted:
        legend_elements.append(Line2D([0], [0], **city_interference_style))
        
    if plant_interference_plotted:
        legend_elements.append(Line2D([0], [0], **plant_interference_style))
        
    if bg_sector_polygon_added:
        legend_elements.append(Patch(facecolor=bg_sector_style['facecolor'], 
                                    alpha=bg_sector_style['alpha'], 
                                    edgecolor=bg_sector_style['edgecolor'], 
                                    label=bg_sector_style.get('label', 'BG Calc Area (Dir.)')))

    # Add legend if there are elements to show
    if legend_elements:
        ncols = min(5, (len(legend_elements) + 2) // 2 if len(legend_elements) > 6 else len(legend_elements))
        fig.legend(handles=legend_elements, loc='lower center', ncol=ncols, 
                bbox_to_anchor=(0.5, 0.01), fontsize=8, frameon=False)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.subplots_adjust(hspace=0.25, wspace=0.18)

    return fig


# --- process_zoomed_data (refactored for reduced redundancy, unchanged logic/outputs) ---
def process_zoomed_data(
    row,
    global_locations_df,
    cities_df,
    zoom_radius_km=100, threshold_factor=4.5, threshold_abs_min=1e-5,
    max_distance_km=20, close_distance_km=8,
    max_angle_diff=30, max_angle_diff_mask=90, close_distance_km_mask=2,
    flagged_area=10.0, sigma=10, plot_dpi=200,
    interf_max_distance_km=150, interf_city_pop_thresh=200000,
    interf_plant_emis_thresh=0.5, threshold_radius_km=25,
    background_mode='directional', upwind_angle_tolerance=90,
    background_dist_min_km=10, background_dist_max_km=75,
    plot_interference_zones=True,
    city_base_radius=5.0, city_pop_scale=3,
    city_radius_min=30.0, city_radius_max=50.0,
    plant_base_radius=0.0, plant_emission_scale=0.5,
    plant_radius_min=40.0, plant_radius_max=50.0,
    stat_radius=50.0, # Radius for radius km NO₂ stats
):
    """Generate plume visualization for a single facility snapshot."""
    import pandas as pd, numpy as np, matplotlib.pyplot as plt

    def log(msg, warn=False):
        print(("Warn: " if warn else "Error: ") + msg)

    # --- Basic validation ---------------------------------------------------
    if not isinstance(row, pd.Series):
        log(f"'row' must be pd.Series, got {type(row)}"); return None

    req_row = ['file_path', 'latitude', 'longitude', 'utc_time', 'location']
    if (missing := [k for k in req_row if k not in row or pd.isna(row[k])]):
        log(f"row missing/NaN keys: {missing}"); return None

    if global_locations_df is None or not isinstance(global_locations_df, pd.DataFrame):
        log("`global_locations_df` missing/invalid"); return None
    req_glob = ['ID', 'latitude', 'longitude', 'nox_emis_ty']
    if (missing := [c for c in req_glob if c not in global_locations_df.columns]):
        log(f"`global_locations_df` missing cols: {missing}"); return None

    if cities_df is None:
        log("`cities_df` is None; city interference disabled", warn=True)
        cities_df = pd.DataFrame()
    elif not isinstance(cities_df, pd.DataFrame):
        log("`cities_df` must be DataFrame"); return None
    else:
        req_city = ['latitude', 'longitude', 'population', 'name', 'country']
        if (missing := [c for c in req_city if c not in cities_df.columns]):
            log(f"`cities_df` missing cols: {missing}", warn=True)

    if pd.isna(row.get('wind_u')) or pd.isna(row.get('wind_v')):
        log(f"ID {row['location']}: wind is NaN; using fallbacks", warn=True)

    # --- Extract primary fields --------------------------------------------
    file_path, plant_lat, plant_lon, utc_time = row[['file_path', 'latitude', 'longitude', 'utc_time']]
    wind_u, wind_v   = row.get('wind_u'), row.get('wind_v')
    facility_id      = row['location']
    annual_nox       = row.get('annual_nox_emission', row.get('nox_emis_ty', np.nan))
    country_code     = row.get('ISO3', row.get('country', ''))

    # --- Nearby‑stats lookup ------------------------------------------------
    nearby_plant_stats = {r: {'count': np.nan, 'total_emission': np.nan, 'percentage': np.nan}
                          for r in (20, 50, 100)}
    nearby_city_stats  = {r: {'count': np.nan, 'pop': np.nan}
                          for r in (20, 50, 100, 200)}
    try:
        lookup_id = (pd.to_numeric(facility_id, errors='coerce')
                     if pd.api.types.is_numeric_dtype(global_locations_df['ID'])
                     else str(facility_id))
        match = global_locations_df[global_locations_df['ID'] == lookup_id]
        if len(match) > 1: log(f"ID '{facility_id}' matched multiple rows; using first", warn=True)
        if not match.empty:
            g = match.iloc[0]
            for r in nearby_plant_stats:
                nearby_plant_stats[r].update({
                    'count': g.get(f'nearby_plants_count_{r}km', np.nan),
                    'total_emission': g.get(f'total_emission_{r}km', np.nan),
                    'percentage': g.get(f'percentage_emission_{r}km', np.nan)
                })
            for r in nearby_city_stats:
                nearby_city_stats[r].update({
                    'count': g.get(f'nearby_cities_count_{r}km', np.nan),
                    'pop':   g.get(f'nearby_cities_pop_{r}km', np.nan)
                })
        else:
            log(f"ID '{facility_id}' not found in global_locations_df", warn=True)
    except Exception as e:
        log(f"Stats lookup failed for '{facility_id}': {e}", warn=True)

    # --- Visualization ------------------------------------------------------
    try:
        return create_zoomed_plume_visualization(
            file_path=file_path, plant_lat=plant_lat, plant_lon=plant_lon,
            wind_u=wind_u, wind_v=wind_v, utc_time=utc_time,
            annual_nox_ty=annual_nox,
            nearby_plant_stats=nearby_plant_stats,
            nearby_city_stats=nearby_city_stats,
            all_locations_df=global_locations_df.copy(),
            cities_df=cities_df.copy(),
            zoom_radius_km=zoom_radius_km,
            facility_id=facility_id,
            facility_country=country_code,
            threshold_factor=threshold_factor, threshold_abs_min=threshold_abs_min,
            max_distance_km=max_distance_km, close_distance_km=close_distance_km,
            max_angle_diff=max_angle_diff, max_angle_diff_mask=max_angle_diff_mask,
            close_distance_km_mask=close_distance_km_mask,
            flagged_area=flagged_area, sigma=sigma, plot_dpi=plot_dpi,
            threshold_radius_km=threshold_radius_km,
            interf_max_distance_km=interf_max_distance_km,
            interf_city_pop_thresh=interf_city_pop_thresh,
            interf_plant_emis_thresh=interf_plant_emis_thresh,
            background_mode=background_mode,
            upwind_angle_tolerance=upwind_angle_tolerance,
            background_dist_min_km=background_dist_min_km,
            background_dist_max_km=background_dist_max_km,
            plot_interference_zones=plot_interference_zones,
            city_base_radius=city_base_radius, city_pop_scale=city_pop_scale,
            city_radius_min=city_radius_min, city_radius_max=city_radius_max,
            plant_base_radius=plant_base_radius, plant_emission_scale=plant_emission_scale,
            plant_radius_min=plant_radius_min, plant_radius_max=plant_radius_max,
            stat_radius=stat_radius, # Radius for radius km NO₂ stats
        )
    except Exception as e:
        log(f"Visualization failed for '{facility_id}': {type(e).__name__} - {e}")
        import traceback; traceback.print_exc(); plt.close('all')
        return None