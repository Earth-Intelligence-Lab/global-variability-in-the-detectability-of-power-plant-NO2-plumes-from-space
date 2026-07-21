"""World-specific configuration."""
from dataclasses import dataclass, field
import os

from .base import (
    RegionConfig,
    TROPOMI_VARS,
    SPATIAL_CONTEXT_VARS,
    PLUME_METRIC_VARS,
)


_W_DATA = "/net/fs06/d3/rzhuang/TROPOMI/data/world"
_W_RUN = os.path.join(_W_DATA, "Run_3")


@dataclass
class WorldConfig(RegionConfig):
    name: str = "world"

    # data roots
    data_dir: str = _W_DATA
    tropomi_dir: str = os.path.join(_W_DATA, "TROPOMI_2018_data")
    era5_dir: str = os.path.join(_W_DATA, "era5_compact")

    # input csvs
    emissions_csv: str = os.path.join(_W_DATA, "coco2_ps_catalogue_v2.0.csv")
    power_plants_csv: str = os.path.join(
        _W_DATA, "power_plant_location",
        "power_plants_with_combined_nearby_stats.csv",
    )

    # default run paths
    default_run_dir: str = _W_RUN
    snapshots_csv: str = os.path.join(_W_RUN, "valid_tropomi_emissions.csv")
    tropomi_table_csv: str = os.path.join(
        _W_RUN, "valid_tropomi_emissions_with_qa_with_all_vars.csv"
    )
    era5_table_csv: str = os.path.join(
        _W_RUN, "processed_valid_tropomi_emissions_with_qa_with_all_vars.csv"
    )
    final_table_csv: str = os.path.join(
        _W_RUN, "updated_tropomi_emissions_full_variables_with_fuel.csv"
    )

    # csv column mappings (COCO2 uses lowercase + "ID")
    loc_id_col: str = "ID"
    lat_col: str = "latitude"
    lon_col: str = "longitude"

    # World uses all rows
    # (default identity filter from base)

    era5_files: list = field(default_factory=lambda: [
        "TOA_incident_solar_radiation.nc",
        "Total_column_water_vapour.nc",
        "temperature.nc",
    ])

    # Training features (42 — World version has no 'NOx Mass (lbs)')
    training_features: list = field(default_factory=lambda: (
        ['annual_nox_emission']
        + TROPOMI_VARS
        + SPATIAL_CONTEXT_VARS
        + ['wind_speed', 't2m', 'tisr', 'tcwv']
        + PLUME_METRIC_VARS
        + ['primary_fuel_type']
    ))

    fuel_type_col: str = "primary_fuel_type"

    model_output_dir: str = _W_RUN
