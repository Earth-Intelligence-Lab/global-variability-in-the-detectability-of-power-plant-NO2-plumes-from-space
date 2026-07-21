"""US-specific configuration."""
from dataclasses import dataclass, field
import os

from .base import (
    RegionConfig,
    TROPOMI_VARS,
    SPATIAL_CONTEXT_VARS,
    PLUME_METRIC_VARS,
)


_US_DATA = "/net/fs06/d3/rzhuang/TROPOMI/data/us"
_US_RUN = os.path.join(_US_DATA, "Run_20250623_203825")


@dataclass
class USConfig(RegionConfig):
    name: str = "us"

    # data roots
    data_dir: str = _US_DATA
    tropomi_dir: str = os.path.join(_US_DATA, "TROPOMI_2019-2024")
    era5_dir: str = os.path.join(_US_DATA, "era5")

    # input csvs
    emissions_csv: str = os.path.join(_US_DATA, "top-CAMPD-sources-2019-2024.csv")
    power_plants_csv: str = os.path.join(
        _US_DATA, "facility_emissions_by_plant_comprehensive.csv"
    )

    # default run paths
    default_run_dir: str = _US_RUN
    snapshots_csv: str = os.path.join(_US_RUN, "valid_tropomi_emissions.csv")
    tropomi_table_csv: str = os.path.join(
        _US_RUN, "valid_tropomi_emissions_with_qa_with_all_vars.csv"
    )
    era5_table_csv: str = os.path.join(
        _US_RUN, "processed_valid_tropomi_emissions_with_qa_updated.csv"
    )
    final_table_csv: str = os.path.join(
        _US_RUN, "updated_tropomi_hourly_emissions_full_variables.csv"
    )

    # csv column mappings (US uses CAMPD's "Latitude"/"Longitude"/"Facility ID")
    loc_id_col: str = "Facility ID"
    lat_col: str = "Latitude"
    lon_col: str = "Longitude"

    # US uses top-480 facilities
    data_filter: callable = field(default=lambda df: df.iloc[:480])

    # ERA5 (current US set has only 3 vars; B0 will expand this)
    era5_files: list = field(default_factory=lambda: [
        "temperature.nc",
        "TOA_incident_solar_radiation.nc",
        "Total_column_water_vapour.nc",
    ])

    # Training features (43 total — matches existing US training script)
    training_features: list = field(default_factory=lambda: (
        ['annual_nox_emission']
        + TROPOMI_VARS
        + SPATIAL_CONTEXT_VARS
        + ['wind_speed', 't2m', 'tisr', 'tcwv']
        + PLUME_METRIC_VARS
        + ['primary_fuel_type', 'NOx Mass (lbs)']
    ))

    fuel_type_col: str = "primary_fuel_type"

    model_output_dir: str = _US_RUN
