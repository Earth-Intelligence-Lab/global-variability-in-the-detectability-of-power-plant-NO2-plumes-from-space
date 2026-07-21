# TROPOMI Power-Plant NO₂ Plume Detection & Emission Analysis

Processing and analysis code for automated detection of NO₂ plumes from
individual power plants in TROPOMI/Sentinel-5P observations, with emission
attribution for U.S. (EPA CEMS, 2019–2024) and global power-plant datasets.

> **Paper:** Huang, R. and Wang, S.: *Global variability in the detectability of power plant NO₂ plumes from space*, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2025-6008, 2026.
>
> If you use this code, please cite the paper above.

## Repository structure

The pipeline is organized as numbered stages under `code/`:

| Stage | Directory | Purpose |
|---|---|---|
| 1 | `1_data_prep/` | TROPOMI L2 NO₂ download and preprocessing (`us/`, `world/`) |
| 2 | `2_snapshots/` | Per-plant overpass snapshot extraction |
| 3 | `3_external_data/` | External data: EPA CEMS hourly emissions, ERA5 winds |
| 4 | `4_sampling/` | Observation sampling / QA filtering |
| 5 | `5_tables/` | Aggregated per-observation tables |
| 6 | `6_training/` | Detectability (POD) model training |
| 7 | `7_analysis/` | Statistical analyses |
| 8 | `8_visualization/` | Diagnostic visualizations |
| 9 | `9_paper_figures/` | Notebooks reproducing every figure in the paper |
| — | `shared/` | Shared plume-detection and plotting library |
| — | `config/` | Pipeline configuration |
| — | `slurm/` | SLURM batch scripts used on our cluster |
| — | `notebooks/` | Exploratory notebooks |

## The plume-detection algorithm

The core automated plume detection (downwind-sector NO₂ enhancement test with
interference masking of nearby cities and other plants) lives in
`code/shared/` (`label_no2_plume_flexible_interference`). Figure notebooks in
`code/9_paper_figures/` show it applied end-to-end.

## Data

### Processed data (Zenodo)

The aggregated per-observation tables and plant lists used by the analysis
and figure notebooks are archived at
[doi:10.5281/zenodo.21466576](https://doi.org/10.5281/zenodo.21466576)
(CC BY 4.0):

| Zenodo file | Contents |
|---|---|
| `us-tropomi-observation-with-variables.csv` | U.S. per-overpass table (top-500 plants, 2019–2024): plume labels, 100 m ERA5 wind, TROPOMI variables, EPA CEMS hourly NOx |
| `global-tropomi-observation-with-variables.csv` | Global per-overpass table (top-6000 plants): plume labels (100 m wind), full TROPOMI variables, fuel type |
| `us-power-plant-list.csv` | U.S. facility list with locations and emissions |
| `global-power-plant-list.csv` | Global plant list with locations, emissions, nearby-source statistics |
| `world-cities.csv` | City locations/populations used for interference masking |
| S5P sample file | One example TROPOMI L2 NO₂ granule |

### Raw input sources

Raw inputs are **not** included (≈12 TB); they can be re-obtained from:

- **TROPOMI L2 NO₂** — [Copernicus Data Space](https://dataspace.copernicus.eu/)
- **EPA CEMS hourly emissions** — [EPA CAMPD API](https://campd.epa.gov/)
  (requires an API key; set `EPA_API_KEYS` in a `.env` file, see below)
- **ERA5 100 m winds** — [Copernicus CDS](https://cds.climate.copernicus.eu/)
- **World cities** — simplemaps world cities database

Paths to local data roots are configured at the top of each stage script and
in `code/config/`.

## Setup

```bash
# Python ≥ 3.10; typical scientific stack
pip install numpy pandas scipy matplotlib cartopy geopandas scikit-learn \
    netCDF4 xarray haversine tqdm python-dotenv

# EPA API access (stage 3, US only)
echo "EPA_API_KEYS=key1,key2" > .env
```

## Reproducing the paper figures

Each notebook in `code/9_paper_figures/us/` and `code/9_paper_figures/world/`
regenerates one figure (or figure panel set) from the aggregated stage-5
tables. Notebooks are committed with outputs, so results are inspectable
without rerunning.

## License

[MIT](LICENSE)
