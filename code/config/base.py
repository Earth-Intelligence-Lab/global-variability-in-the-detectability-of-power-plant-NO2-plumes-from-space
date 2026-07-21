"""
Base configuration for TROPOMI pipeline.

RegionConfig defines all parameters that may vary between US and World pipelines:
data paths, column names, processing parameters, and feature lists. Subclasses
override only what differs.

TROPOMIConfig holds the plume detection / visualization algorithm parameters
(migrated from the world sampling script). It does not depend on region.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Plume detection algorithm parameters (region-independent)
# ──────────────────────────────────────────────────────────────────────────────
class TROPOMIConfig:
    """Centralized configuration for TROPOMI NO2 plume detection and visualization.

    Migrated from 4_sampling/world/4_sample_snapshots_label_train_val_test_split_continent_emission.py
    so it is shared across all pipeline scripts.
    """

    PLUME_DETECTION = {
        'zoom_radius_km': 100,
        'threshold_factor': 2.0,
        'threshold_abs_min': 5e-6,
        'max_distance_km': 20.0,
        'close_distance_km': 5.0,
        'max_angle_diff': 25.0,
        'flagged_area': 25.0,
        'stat_radius': 50.0,
        'threshold_radius_km': 50.0,
    }

    BACKGROUND = {
        'mode': 'directional',
        'upwind_angle_tolerance': 60,
        'dist_min_km': 10,
        'dist_max_km': 100,
        'gaussian_sigma': 10,
    }

    PLANT_MASK = {
        'max_angle_diff_mask': 0,
        'close_distance_km_mask': 20,
    }

    INTERFERENCE = {
        'max_distance_km': 150,
        'city': {
            'base_radius': 0.0,
            'pop_scale': 9.0,
            'radius_min': 10.0,
            'radius_max': 90.0,
            'min_population': 200000,
        },
        'plant': {
            'base_radius': 0.0,
            'emission_scale': 0.0,
            'radius_min': 0.0,
            'radius_max': 0.0,
            'min_emission_threshold': 1.0,
            'use_emission_scaling': True,
        },
    }

    VISUALIZATION = {
        'plot_dpi': 200,
        'plot_interference_zones': True,
        'nearby_plant_radius_km': 200,
        'basemap_zoom': 'auto',
        'colormap_no2': 'viridis',
        'colormap_anomaly': 'coolwarm',
    }

    PROCESSING = {
        'min_city_population': 50000,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Common feature lists (shared across regions)
# ──────────────────────────────────────────────────────────────────────────────
TROPOMI_VARS = [
    'surface_altitude', 'surface_altitude_precision', 'surface_classification',
    'surface_pressure', 'surface_albedo', 'surface_albedo_nitrogendioxide_window',
    'cloud_pressure_crb', 'cloud_fraction_crb', 'cloud_albedo_crb',
    'scene_albedo', 'apparent_scene_pressure', 'snow_ice_flag',
    'aerosol_index_354_388', 'eastward_wind', 'northward_wind',
    'scaled_small_pixel_variance', 'tropospheric_NO2_column_number_density',
    'sensor_altitude', 'sensor_azimuth_angle', 'sensor_zenith_angle',
    'solar_azimuth_angle', 'solar_zenith_angle',
]

# Spatial-context columns common to both regions (added in step 5/merge)
SPATIAL_CONTEXT_VARS = [
    'nearby_plants_count_20km', 'total_emission_20km', 'percentage_emission_20km',
    'nearby_plants_count_50km', 'total_emission_50km', 'percentage_emission_50km',
    'nearby_plants_count_100km', 'total_emission_100km', 'percentage_emission_100km',
    'nearby_cities_count_20km', 'nearby_cities_pop_20km',
    'nearby_cities_count_50km', 'nearby_cities_pop_50km',
    'nearby_cities_count_100km', 'nearby_cities_pop_100km',
]

PLUME_METRIC_VARS = [
    'no2_mean_radius', 'no2_std_radius', 'no2_frac_valid_radius',
]


# ──────────────────────────────────────────────────────────────────────────────
# Region-specific configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RegionConfig:
    """Base config; subclasses override paths, column names and feature filters.

    All paths are absolute so scripts can be run from any working directory.
    """

    # ── identity ─────────────────────────────────────────────────────────────
    name: str = "base"

    # ── data root paths ──────────────────────────────────────────────────────
    data_dir: str = ""              # e.g. /net/fs06/d3/rzhuang/TROPOMI_US/data
    tropomi_dir: str = ""           # subdir or absolute path of NetCDF files
    era5_dir: str = ""

    # ── input csvs ───────────────────────────────────────────────────────────
    emissions_csv: str = ""         # plant location/emissions csv
    power_plants_csv: str = ""      # plant-with-stats csv (used by training)

    # ── default run paths (can be overridden via CLI) ───────────────────────
    default_run_dir: str = ""
    snapshots_csv: str = ""         # raw snapshots from step 2
    tropomi_table_csv: str = ""     # output of step 5a
    era5_table_csv: str = ""        # output of step 5b
    final_table_csv: str = ""       # used by training

    # ── csv column mappings ──────────────────────────────────────────────────
    loc_id_col: str = "ID"          # plant id column in emissions_csv
    lat_col: str = "latitude"
    lon_col: str = "longitude"

    # ── data filtering hook (optional) ──────────────────────────────────────
    # A function that takes the loaded emissions DataFrame and returns the
    # subset to use. Default is identity (no filter).
    data_filter: Callable[[pd.DataFrame], pd.DataFrame] = field(
        default=lambda df: df
    )

    # ── processing parameters ────────────────────────────────────────────────
    num_processes: int = 288
    radius_km: float = 50.0
    valid_ratio: float = 0.5

    # ── ERA5 file list ───────────────────────────────────────────────────────
    era5_files: List[str] = field(default_factory=list)

    # ── training feature list ────────────────────────────────────────────────
    # Includes 'annual_nox_emission' + TROPOMI vars + spatial context + ERA5 + plume metrics
    training_features: List[str] = field(default_factory=list)
    target_col: str = "plume_label"
    fuel_type_col: Optional[str] = "primary_fuel_type"

    # ── output directory for trained models ─────────────────────────────────
    model_output_dir: str = ""
