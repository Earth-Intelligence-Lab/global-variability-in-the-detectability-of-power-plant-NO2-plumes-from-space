"""
Central configuration for TROPOMI processing pipeline.

Usage:
    from config import get_config, INTERFERENCE, TRAINING, SPATIAL, CITIES_CSV
    cfg = get_config("us")   # or "world"
    print(cfg["lat_col"])    # "Latitude" for US, "latitude" for world
"""

# ─── Region Profiles ─────────────────────────────────────────────────────────

REGION_PROFILES = {
    "us": {
        # --- Paths ---
        "base_dir": "/net/fs06/d3/rzhuang/TROPOMI_US",
        "data_dir": "/net/fs06/d3/rzhuang/TROPOMI_US/data",
        "plant_csv": "facility_emissions_by_plant_comprehensive.csv",
        "tropomi_csv_name": "updated_tropomi_hourly_emissions_full_variables.csv",
        "city_csv": "worldcities.csv",
        "tropomi_dir": "TROPOMI_2019-2024",
        "era5_dir": "era5",

        # --- Column names ---
        "lat_col": "Latitude",
        "lon_col": "Longitude",
        "id_col": "Facility_ID",
        "emission_col": "CO2 Mass (short tons)",
        "nox_col": "NOx Mass (lbs)",
        "annual_emission_col": "annual_nox_emission",
        "location_col": "location",
        "country_col": "State",
        "fuel_col": "primary_fuel_type",

        # --- Processing params ---
        "top_n_filter": 20,
        "n_jobs": 48,
        "has_hourly_nox": True,
        "years": [2019, 2020, 2021, 2022, 2023, 2024],

        # --- ERA5 files (order matters) ---
        "era5_files": [
            "temperature.nc",
            "TOA_incident_solar_radiation.nc",
            "Total_column_water_vapour.nc",
        ],
    },
    "world": {
        # --- Paths ---
        "base_dir": "/net/fs06/d3/rzhuang/TROPOMI_world",
        "data_dir": "/net/fs06/d3/rzhuang/TROPOMI_world/data",
        "plant_csv": "power_plant_location/filtered_power_plants.csv",
        "tropomi_csv_name": "updated_tropomi_emissions_full_variables_with_fuel.csv",
        "city_csv": "worldcities.csv",
        "tropomi_dir": "TROPOMI_2018_data",
        "era5_dir": "era5_compact",

        # --- Column names ---
        "lat_col": "latitude",
        "lon_col": "longitude",
        "id_col": "ID",
        "emission_col": "nox_emis_ty",
        "nox_col": "nox_emis_ty",
        "annual_emission_col": "annual_nox_emission",
        "location_col": "location",
        "country_col": "ISO3",
        "fuel_col": "primary_fuel_type",

        # --- Processing params ---
        "top_n_filter": None,
        "n_jobs": 288,
        "has_hourly_nox": False,
        "years": None,

        # --- ERA5 files (order matters — different from US) ---
        "era5_files": [
            "TOA_incident_solar_radiation.nc",
            "Total_column_water_vapour.nc",
            "temperature.nc",
        ],
    },
}

# ─── Interference Zone Constants ─────────────────────────────────────────────
# Used by training scripts (*_no_interference_plant.py) and analysis scripts.
# Note: training scripts and interference_zone_analysis.py define overlapping
# but slightly different sets of constants — all are captured here.

INTERFERENCE = {
    "EARTH_RADIUS_KM": 6371.0,

    # Plant interference (training scripts)
    "PLANT_RADIUS_BASE_KM": 20.0,
    "CITY_POP_THRESHOLD": 200000,
    "CITY_RADIUS_SCALE": 9.0,
    "CITY_RADIUS_BASE_KM": 10.0,
    "CITY_RADIUS_MAX_KM": 90.0,

    # Plant interference (interference_zone_analysis.py)
    "PLANT_MAX_SEARCH_KM": 150.0,
    "PLANT_RADIUS_SCALE": 0,        # interference_zone_analysis.py uses 0
    "PLANT_RADIUS_MAX_KM": 50.0,    # interference_zone_analysis.py

    # City interference (interference_zone_analysis.py)
    "CITY_MAX_SEARCH_KM": 150.0,

    # Plume detection interference (plot_pipeline_concise_new.py uses different values)
    "PLUME_PLANT_EMISSION_SCALE": 0.4,
    "PLUME_PLANT_RADIUS_MIN_KM": 5.0,
    "PLUME_PLANT_RADIUS_MAX_KM": 30.0,
    "PLUME_PLANT_EMISSION_THRESHOLD_TY": 0.5,
}

# ─── Training Defaults ───────────────────────────────────────────────────────

TRAINING = {
    # Model architecture
    "hidden_dims": [256, 128, 64, 32],
    "dropout": 0.3,

    # Training loop
    "batch_size": 32,
    "num_epochs": 100,
    "patience": 5,
    "optimizer": "adam",
    "loss": "BCEWithLogitsLoss",
    "prediction_threshold": 0.5,

    # Hyperparameter search (full_sweep mode)
    "learning_rates": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "num_runs": 5,

    # Default learning rate (simple mode)
    "default_lr": 1e-3,

    # Data split ratios (full_sweep mode: train/val/test)
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2,

    # Data split ratio (simple mode: train/test)
    "simple_test_size": 0.2,

    # Random seeds
    # full_sweep: data split uses 345 (+run for some variants)
    # simple: data split uses 42
    # oversampling: 42 (simple) or 42+run (full_sweep)
    "split_seed_simple": 42,
    "split_seed_sweep": 345,
    "oversample_seed_base": 42,

    # QA threshold for TROPOMI data quality
    "qa_threshold": 0.75,
}

# ─── Spatial Analysis Defaults ────────────────────────────────────────────────

SPATIAL = {
    "plant_radii_km": [20, 50, 100],
    "city_radii_km": [20, 50, 100, 200],
    "min_city_population": 50000,       # for data prep (Stage 1)
}

# ─── Plume Detection Defaults ────────────────────────────────────────────────
# These match the argparse defaults in 2_find_snapshots_labelling_100m.py

PLUME_DETECTION = {
    "zoom_radius_km": 100,
    "max_distance_km": 20.0,
    "close_distance_km": 5.0,
    "threshold_radius_km": 50.0,
    "stat_radius": 50.0,
    "threshold_factor": 2,
    "threshold_abs_min": 5e-6,
    "max_angle_diff": 25.0,
    "flagged_area": 25.0,
    "background_mode": "directional",
    "upwind_angle_tolerance": 60,
    "background_dist_min_km": 10,
    "background_dist_max_km": 100,
    "sigma": 10,
    "max_angle_diff_mask": 0.0,
    "close_distance_km_mask": 20.0,
    "interf_max_distance_km": 150.0,
    "interf_city_pop_thresh": 200000.0,
    "interf_plant_emis_thresh": 1.0,
    "city_base_radius": 0.0,
    "city_pop_scale": 9.0,
    "city_radius_min": 10.0,
    "city_radius_max": 90.0,
    "plant_base_radius": 0.0,
    "plant_emission_scale": 0.0,
    "plant_radius_min": 0.0,
    "plant_radius_max": 0.0,
}

# ─── Cities Data (shared between US and World) ───────────────────────────────

CITIES_CSV = "/net/fs06/d3/rzhuang/TROPOMI_world/data/worldcities.csv"


# ─── Helper ───────────────────────────────────────────────────────────────────

def get_config(region):
    """Return the config dict for a region.

    Args:
        region: "us" or "world" (case-insensitive)

    Returns:
        dict with all region-specific configuration

    Raises:
        ValueError: if region is not recognized
    """
    region = region.lower()
    if region not in REGION_PROFILES:
        raise ValueError(
            f"Unknown region '{region}'. Choose from: {list(REGION_PROFILES.keys())}"
        )
    return REGION_PROFILES[region]
