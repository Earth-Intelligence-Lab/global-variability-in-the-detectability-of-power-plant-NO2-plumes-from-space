"""Feature list definitions for TROPOMI MLP training.

Verified programmatically from ALL 26 training scripts. Feature counts:

  feature_set       | region | data_type | Count | Scripts
  ------------------+--------+-----------+-------+--------------------------------------
  baseline          | US     | any       |    45 | 6_MLP_training.py, _power_plant_split
  baseline          | world  | any       |    44 | 6_MLP_training.py, _power_plant_split
  baseline_wo_stats | US     | any       |    30 | 6_MLP_training_wo_stats.py
  baseline_wo_stats | world  | any       |    29 | 6_MLP_training_wo_stats.py
  full              | US     | annual    |    40 | *_train_val_test_split_annual.py
  full              | US     | hourly    |    40 | *_train_val_test_split_hourly.py
  full              | world  | any       |    40 | *_train_val_test_split.py
  no_stats          | any    | any       |    25 | *_no_stats.py

Note: "baseline" vs "full" differ in whether tropospheric_NO2, no2_radius,
and NOx Mass are included. Whether this was intentional is TBD.

Total unique features across codebase: 45
"""

# ─── Sensor + satellite features (19) ─────────────────────────────────────────
# surface_altitude != sensor_altitude (surface = ground, sensor = satellite)

SENSOR_FEATURES = [
    'surface_altitude',
    'surface_altitude_precision',
    'surface_classification',
    'surface_pressure',
    'surface_albedo',
    'surface_albedo_nitrogendioxide_window',
    'cloud_pressure_crb',
    'cloud_fraction_crb',
    'cloud_albedo_crb',
    'scene_albedo',
    'apparent_scene_pressure',
    'snow_ice_flag',
    'aerosol_index_354_388',
    'scaled_small_pixel_variance',
    'sensor_altitude',
    'sensor_azimuth_angle',
    'sensor_zenith_angle',
    'solar_azimuth_angle',
    'solar_zenith_angle',
]

# ─── ERA5 weather features (4) ───────────────────────────────────────────────

ERA5_FEATURES = [
    'wind_speed',
    't2m',
    'tisr',
    'tcwv',
]

# ─── Nearby plant/city statistics (15) ────────────────────────────────────────

STATS_FEATURES = [
    'nearby_plants_count_20km',
    'total_emission_20km',
    'percentage_emission_20km',
    'nearby_plants_count_50km',
    'total_emission_50km',
    'percentage_emission_50km',
    'nearby_plants_count_100km',
    'total_emission_100km',
    'percentage_emission_100km',
    'nearby_cities_count_20km',
    'nearby_cities_pop_20km',
    'nearby_cities_count_50km',
    'nearby_cities_pop_50km',
    'nearby_cities_count_100km',
    'nearby_cities_pop_100km',
]

# ─── NO2 radius statistics (3) ───────────────────────────────────────────────

NO2_RADIUS_FEATURES = [
    'no2_mean_radius',
    'no2_std_radius',
    'no2_frac_valid_radius',
]

# ─── Individual features ─────────────────────────────────────────────────────

ANNUAL_NOX_EMISSION = 'annual_nox_emission'
TROPOSPHERIC_NO2 = 'tropospheric_NO2_column_number_density'
US_NOX_MASS = 'NOx Mass (lbs)'
FUEL_TYPE = 'primary_fuel_type'


def get_feature_list(feature_set, region="us", data_type="annual"):
    """Build feature list matching the exact original script behavior.

    Feature ORDER matches the originals (verified from source code).

    Args:
        feature_set: "baseline", "baseline_wo_stats", "full", or "no_stats"
        region: "us" or "world"
        data_type: "annual" or "hourly"

    Returns:
        list of feature column names

    Examples:
        >>> len(get_feature_list("baseline", "us"))     # 45
        45
        >>> len(get_feature_list("baseline", "world"))   # 44
        44
        >>> len(get_feature_list("baseline_wo_stats", "us"))  # 30
        30
        >>> len(get_feature_list("full", "us", "annual"))     # 40
        40
        >>> len(get_feature_list("full", "us", "hourly"))     # 40
        40
        >>> len(get_feature_list("no_stats"))                  # 25
        25
    """
    if feature_set == "baseline":
        # 6_MLP_training.py, 6_MLP_training_power_plant_split.py
        # US: 45, World: 44
        # Order: annual_nox, sensor[:14], trop_NO2, sensor[14:],
        #        stats, era5, no2_radius, fuel [, NOx Mass]
        features = [ANNUAL_NOX_EMISSION]
        features += SENSOR_FEATURES[:14]
        features.append(TROPOSPHERIC_NO2)
        features += SENSOR_FEATURES[14:]
        features += STATS_FEATURES
        features += ERA5_FEATURES
        features += NO2_RADIUS_FEATURES
        features.append(FUEL_TYPE)
        if region == "us":
            features.append(US_NOX_MASS)
        return features

    elif feature_set == "baseline_wo_stats":
        # 6_MLP_training_wo_stats.py
        # US: 30, World: 29
        # Same as baseline minus 15 stats features
        features = [ANNUAL_NOX_EMISSION]
        features += SENSOR_FEATURES[:14]
        features.append(TROPOSPHERIC_NO2)
        features += SENSOR_FEATURES[14:]
        features += ERA5_FEATURES
        features += NO2_RADIUS_FEATURES
        features.append(FUEL_TYPE)
        if region == "us":
            features.append(US_NOX_MASS)
        return features

    elif feature_set == "full":
        # *_train_val_test_split_*.py, *_train_val_split_*.py,
        # *_no_interference_plant.py (the variants WITH stats)
        # All: 40
        if data_type == "hourly" and region == "us":
            # US hourly: NO annual_nox_emission, YES NOx Mass
            features = list(SENSOR_FEATURES)
            features += STATS_FEATURES
            features += ERA5_FEATURES
            features.append(FUEL_TYPE)
            features.append(US_NOX_MASS)
        else:
            # US annual, World annual/hourly
            features = [ANNUAL_NOX_EMISSION]
            features += SENSOR_FEATURES
            features += STATS_FEATURES
            features += ERA5_FEATURES
            features.append(FUEL_TYPE)
        return features

    elif feature_set == "no_stats":
        # *_no_stats.py variants
        # US hourly: 25 (sensor + ERA5 + fuel + NOx Mass)
        # US annual: 25 (sensor + ERA5 + fuel + annual_nox_emission)
        # World any:  25 (annual_nox + sensor + ERA5 + fuel)
        if region == "us":
            features = list(SENSOR_FEATURES)
            features += ERA5_FEATURES
            features.append(FUEL_TYPE)
            if data_type == "annual":
                features.append(ANNUAL_NOX_EMISSION)
            else:
                features.append(US_NOX_MASS)
        else:
            features = [ANNUAL_NOX_EMISSION]
            features += SENSOR_FEATURES
            features += ERA5_FEATURES
            features.append(FUEL_TYPE)
        return features

    else:
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. "
            "Choose from: baseline, baseline_wo_stats, full, no_stats"
        )
