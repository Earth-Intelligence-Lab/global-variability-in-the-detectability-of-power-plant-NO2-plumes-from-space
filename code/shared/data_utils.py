"""
Shared data utilities for TROPOMI pipeline.

Functions extracted from previously-duplicated us/world scripts so that
unified pipeline scripts (driven by config/) can call them with region-agnostic
arguments.

Sections:
1. TROPOMI NetCDF loading + BallTree matching (used by step 2 + 5a)
2. ERA5 nearest-neighbor interpolation (used by step 5b)
3. Plant validity check (used by step 2)
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import netCDF4 as nc
import xarray as xr
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


# ──────────────────────────────────────────────────────────────────────────────
# Section 1: TROPOMI loading + spatial matching
# ──────────────────────────────────────────────────────────────────────────────
def open_tropomi_basics(file_path: str) -> Dict[str, np.ndarray]:
    """Open a TROPOMI NetCDF and return latitude/longitude/no2/wind/utc_time arrays.

    Used by snapshot finder (step 2).
    """
    with nc.Dataset(file_path) as ds:
        lats = ds['PRODUCT/latitude'][:]
        lons = ds['PRODUCT/longitude'][:]
        no2 = ds['PRODUCT/nitrogendioxide_tropospheric_column'][:]
        wind_u = ds['PRODUCT/SUPPORT_DATA/INPUT_DATA/eastward_wind'][:]
        wind_v = ds['PRODUCT/SUPPORT_DATA/INPUT_DATA/northward_wind'][:]
        utc_time = ds['PRODUCT/time_utc'][:]
    return {
        'lats': lats, 'lons': lons, 'no2': no2,
        'wind_u': wind_u, 'wind_v': wind_v, 'utc_time': utc_time,
    }


def load_tropomi_full_vars(ds: nc.Dataset) -> Dict[str, np.ndarray]:
    """Load the full set of 22 2D variables used by the table-generation step.

    Slices time=0 from each variable.  Pulled directly from
    5_generate_whole_table_TROPOMI.py.
    """
    prod = ds.groups['PRODUCT']
    inp = prod.groups['SUPPORT_DATA'].groups['INPUT_DATA']
    geo = prod.groups['SUPPORT_DATA'].groups['GEOLOCATIONS']

    lats = prod.variables['latitude'][0]

    return {
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
        'sensor_altitude':                        np.broadcast_to(
            geo.variables['satellite_altitude'][0][:, None], lats.shape
        ),
        'sensor_azimuth_angle':                   geo.variables['viewing_azimuth_angle'][0],
        'sensor_zenith_angle':                    geo.variables['viewing_zenith_angle'][0],
        'solar_azimuth_angle':                    geo.variables['solar_azimuth_angle'][0],
        'solar_zenith_angle':                     geo.variables['solar_zenith_angle'][0],
    }


def build_balltree(lats: np.ndarray, lons: np.ndarray) -> BallTree:
    """Build a haversine BallTree from a 2D lat/lon grid."""
    points_rad = np.radians(np.column_stack((lats.flatten(), lons.flatten())))
    return BallTree(points_rad, metric='haversine')


def plant_valid(
    plant_lat: float, plant_lon: float,
    min_lat: float, max_lat: float, min_lon: float, max_lon: float,
    tree: BallTree, valid_no2: np.ndarray,
    radius_km: float = 50.0, valid_ratio: float = 0.5,
) -> bool:
    """Check whether a plant has enough valid TROPOMI pixels around it.

    Behavior identical to the original plant_valid() in 2_find_snapshots.py.
    """
    if not (min_lat <= plant_lat <= max_lat and min_lon <= plant_lon <= max_lon):
        return False
    plant_rad = np.radians([plant_lat, plant_lon]).reshape(1, -1)
    radius_rad = radius_km / EARTH_RADIUS_KM
    indices = tree.query_radius(plant_rad, r=radius_rad)[0]
    if len(indices) == 0:
        return False
    return (np.sum(valid_no2[indices]) / len(indices)) >= valid_ratio


def nearest_pixel_index(
    tree: BallTree, lat: float, lon: float, grid_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Return the (row, col) index of the nearest TROPOMI pixel to (lat, lon)."""
    point_rad = np.radians([[lat, lon]])
    flat_idx = tree.query(point_rad, k=1)[1][0][0]
    return np.unravel_index(flat_idx, grid_shape)


def get_local_utc_time(utc_time_data: np.ndarray, grid_idx):
    """Pull the UTC time string for a given grid index, robust to ndim.

    Behavior identical to the original logic in 2_find_snapshots.py
    (handles 1D, 2D, and N-D shapes).
    """
    if utc_time_data.ndim == 1:
        return utc_time_data[grid_idx[0]] if len(grid_idx) > 0 else None
    if utc_time_data.ndim == 2:
        return utc_time_data[grid_idx[0], grid_idx[1]] if len(grid_idx) > 1 else None
    try:
        return utc_time_data[grid_idx]
    except IndexError:
        return utc_time_data[grid_idx[0]] if len(grid_idx) > 0 else None


def get_valid_no2_mask(no2: np.ndarray) -> np.ndarray:
    """Return a flat boolean mask of valid (non-masked, finite) NO2 pixels."""
    if np.ma.is_masked(no2):
        return (~no2.mask).flatten()
    return np.isfinite(no2.flatten())


# ──────────────────────────────────────────────────────────────────────────────
# Section 2: ERA5 nearest-neighbor interpolation
# ──────────────────────────────────────────────────────────────────────────────
def wrap_lons_to_grid(
    raw_lons: np.ndarray, lon0: float, lon1: float,
) -> np.ndarray:
    """Wrap longitudes into whichever interval the ERA5 file uses.

    Pulled verbatim from 5_generate_whole_table_era5.py — handles both
    [0, 360) and [-180, 180) systems.
    """
    if lon0 >= 0:
        # 0…360 system
        wrapped = np.where(raw_lons < 0, raw_lons + 360, raw_lons)
        lons = np.clip(wrapped, lon0, lon1)
        mask = wrapped > lon1
        if mask.any():
            d0 = np.minimum(wrapped[mask], 360 - wrapped[mask])
            diff = np.abs(wrapped[mask] - lon1)
            d1 = np.minimum(diff, 360 - diff)
            lons[mask] = np.where(d0 < d1, lon0, lon1)
        return lons
    if lon1 <= 180:
        return ((raw_lons + 180) % 360) - 180
    return raw_lons.copy()


def era5_get_grid_bounds(file_path: str) -> Tuple[float, float, float, float]:
    """Return (lon0, lon1, lat0, lat1) for an ERA5 NetCDF file."""
    with xr.open_dataset(file_path, engine='netcdf4') as ds:
        return (
            ds.longitude.min().item(), ds.longitude.max().item(),
            ds.latitude.min().item(), ds.latitude.max().item(),
        )


def interp_era5_var(
    da: xr.DataArray, time_dim: str,
    times: np.ndarray, lats: np.ndarray, lons: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor interpolate a DataArray to (time, lat, lon) tuples."""
    out = da.interp(
        **{time_dim: ("points", times)},
        latitude=("points", lats),
        longitude=("points", lons),
        method="nearest",
    )
    return out.compute().values
