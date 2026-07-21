"""
Download all 10 ERA5 single-level variables from AWS Open Data and crop to
the US bounding box (CONUS + Alaska).

Source: s3://nsf-ncar-era5/  (NSF NCAR ERA5 mirror, public, no credentials)

Strategy: For each (variable, monthly file)
    1. Multipart parallel download to a local temp file (~14 MB/s)
    2. Open with xarray, crop lat/lon to US, write compressed NetCDF
    3. Delete the raw monthly file
    4. After all months for a var/year are processed, concat into one yearly file

Variables (12 total):

  short  AWS code        prefix         CDS variable
  ─────  ──────────────  ─────────────  ──────────────────────────────
  t2m    128_167_2t      an.sfc         2m_temperature
  d2m    128_168_2d      an.sfc         2m_dewpoint_temperature
  tcwv   128_137_tcwv    an.sfc         total_column_water_vapour
  blh    128_159_blh     an.sfc         boundary_layer_height
  hcc    128_188_hcc     an.sfc         high_cloud_cover
  mcc    128_187_mcc     an.sfc         medium_cloud_cover
  lcc    128_186_lcc     an.sfc         low_cloud_cover
  u100   228_246_100u    an.sfc         100m_u_component_of_wind
  v100   228_247_100v    an.sfc         100m_v_component_of_wind
  tisr   128_212_tisr    fc.sfc.accumu  TOA_incident_solar_radiation (accumulated)
  slhf   128_147_slhf    fc.sfc.accumu  surface_latent_heat_flux     (accumulated)
  sshf   128_146_sshf    fc.sfc.accumu  surface_sensible_heat_flux   (accumulated)

Bounding box: lat 17–64°N, lon -170 to -66°E.
Years: 2019–2024 by default.

Output: <era5>/expanded/<short>_<year>.nc

Usage:
    python download_era5_aws.py                              # all vars × all years
    python download_era5_aws.py --years 2019 2020 --workers 4
    python download_era5_aws.py --var blh --year 2019
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing.pool import ThreadPool

import boto3
import xarray as xr
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), os.pardir))
from config import USConfig, WorldConfig  # noqa: E402

US_BBOX    = dict(lat_min=17.0,  lat_max=64.0, lon_min=-170.0, lon_max=-66.0)
WORLD_BBOX = dict(lat_min=-90.0, lat_max=90.0, lon_min=-180.0, lon_max=180.0)
DEFAULT_YEARS = list(range(2019, 2025))

BUCKET = "nsf-ncar-era5"
PREFIX_AN_SFC = "e5.oper.an.sfc"
PREFIX_FC_ACCUMU = "e5.oper.fc.sfc.accumu"

VAR_SPECS = {
    # short_name → (AWS code substring, S3 prefix, native NetCDF varname)
    # Instantaneous surface analyses
    "t2m":  ("128_167_2t",   PREFIX_AN_SFC,    "VAR_2T"),
    "d2m":  ("128_168_2d",   PREFIX_AN_SFC,    "VAR_2D"),
    "tcwv": ("128_137_tcwv", PREFIX_AN_SFC,    "TCWV"),
    "blh":  ("128_159_blh",  PREFIX_AN_SFC,    "BLH"),
    "hcc":  ("128_188_hcc",  PREFIX_AN_SFC,    "HCC"),
    "mcc":  ("128_187_mcc",  PREFIX_AN_SFC,    "MCC"),
    "lcc":  ("128_186_lcc",  PREFIX_AN_SFC,    "LCC"),
    "u100": ("228_246_100u", PREFIX_AN_SFC,    "VAR_100U"),
    "v100": ("228_247_100v", PREFIX_AN_SFC,    "VAR_100V"),
    # Accumulated forecast surface
    "tisr": ("128_212_tisr", PREFIX_FC_ACCUMU, "TISR"),
    "slhf": ("128_147_slhf", PREFIX_FC_ACCUMU, "SLHF"),
    "sshf": ("128_146_sshf", PREFIX_FC_ACCUMU, "SSHF"),
}

AUX_VARS_TO_DROP = {"utc_date"}

TRANSFER_CFG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    max_concurrency=16,
    multipart_chunksize=8 * 1024 * 1024,
    use_threads=True,
)


def _flatten_accumu_to_time(ds: xr.Dataset) -> xr.Dataset:
    """Flatten an accumulated-forecast Dataset from
       (forecast_initial_time, forecast_hour, lat, lon) → (time, lat, lon).

    valid_time = forecast_initial_time + forecast_hour * 1h
    The values are kept as raw accumulations (NCAR ERA5 convention: each
    forecast_hour gives the running accumulation since the forecast start at
    forecast_initial_time).  Downstream code can convert to instantaneous by
    differencing if needed.
    """
    import numpy as np
    fit = ds["forecast_initial_time"].values  # (30,) datetime64
    fh = ds["forecast_hour"].values            # (12,) int
    # broadcast: (30, 12)
    valid = fit[:, None] + (fh.astype("timedelta64[h]"))[None, :]
    valid_flat = valid.reshape(-1)             # (360,)

    # Manually reshape: collapse the two leading dims into one
    new_vars = {}
    for v in ds.data_vars:
        a = ds[v]
        if "forecast_initial_time" in a.dims and "forecast_hour" in a.dims:
            arr = a.transpose("forecast_initial_time", "forecast_hour", ...).values
            shape = (arr.shape[0] * arr.shape[1],) + arr.shape[2:]
            arr = arr.reshape(shape)
            new_dims = ("time",) + tuple(d for d in a.dims
                                          if d not in ("forecast_initial_time", "forecast_hour"))
            new_vars[v] = (new_dims, arr)
        else:
            new_vars[v] = (a.dims, a.values)
    out = xr.Dataset(
        new_vars,
        coords={
            "time": ("time", valid_flat),
            "latitude": ds["latitude"],
            "longitude": ds["longitude"],
        },
    )
    return out.sortby("time")


def _build_encoding(ds: xr.Dataset) -> dict:
    """Force (1, lat, lon) chunk layout on every 3D data var.

    NCAR's accumu source files ship with chunks=(1460, 32, 70) which is great
    for long time-series queries over a small region but ~5× slower than
    (1, nlat, nlon) for the whole-swath-per-hour read pattern used by
    extract_era5_fields.py. This normalizes all vars to the same layout.
    """
    enc = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        e = {"zlib": True, "complevel": 4}
        if "time" in dims and "latitude" in dims and "longitude" in dims:
            nlat = ds.sizes["latitude"]
            nlon = ds.sizes["longitude"]
            e["chunksizes"] = tuple(
                1 if d == "time" else (nlat if d == "latitude" else nlon)
                for d in dims
            )
        enc[v] = e
    return enc


def rechunk_file(path: str) -> bool:
    """Rewrite a yearly file in place if its chunks don't match (1, lat, lon).

    Returns True if the file was rewritten, False if it was already correct.
    """
    import h5py
    with h5py.File(path, "r") as f:
        for name, obj in f.items():
            if isinstance(obj, h5py.Dataset) and obj.chunks is not None \
                    and len(obj.shape) == 3:
                shape, chunks = obj.shape, obj.chunks
                if chunks == (1, shape[1], shape[2]):
                    return False   # already correct
                break
    tmp = path + ".rechunk.tmp"
    ds = xr.open_dataset(path, engine="netcdf4").load()
    ds.to_netcdf(tmp, encoding=_build_encoding(ds))
    ds.close()
    os.replace(tmp, path)
    return True


def rechunk_existing(out_dir: str, workers: int = 1):
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, "*.nc")))
    print(f"Rechunk pass over {len(files)} files in {out_dir} "
          f"(workers={workers})", flush=True)

    def _one(p):
        t0 = time.time()
        try:
            changed = rechunk_file(p)
            tag = "rewrote" if changed else "skip   "
            print(f"  {tag} {os.path.basename(p)}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  [error] {p}: {e}", flush=True)

    if workers > 1:
        with ThreadPool(workers) as pool:
            pool.map(_one, files)
    else:
        for p in files:
            _one(p)


def s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_monthly_keys(client, var_code: str, prefix: str, year: int, month: int):
    yyyymm = f"{year:04d}{month:02d}"
    s3_prefix = f"{prefix}/{yyyymm}/"
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if var_code in k and k.endswith(".nc"):
                keys.append(k)
    return sorted(keys)


def download_and_crop(client, key: str, raw_dir: str,
                      sname: str = None,
                      bbox: dict = None) -> xr.Dataset:
    raw_path = os.path.join(raw_dir, os.path.basename(key))
    t0 = time.time()
    client.download_file(BUCKET, key, raw_path, Config=TRANSFER_CFG)
    dl_t = time.time() - t0

    ds = xr.open_dataset(raw_path, engine="netcdf4")

    if float(ds.longitude.min()) >= 0:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby("longitude")

    if bbox is None:
        bbox = US_BBOX
    lat_max, lat_min = bbox["lat_max"], bbox["lat_min"]
    if float(ds.latitude[0]) > float(ds.latitude[-1]):
        ds = ds.sel(latitude=slice(lat_max, lat_min))
    else:
        ds = ds.sel(latitude=slice(lat_min, lat_max))
    ds = ds.sel(longitude=slice(bbox["lon_min"], bbox["lon_max"]))

    # Drop aux vars (e.g. utc_date) and rename native var to short_name
    drop = [v for v in ds.data_vars if v in AUX_VARS_TO_DROP]
    if drop:
        ds = ds.drop_vars(drop)
    if sname is not None and sname in VAR_SPECS:
        native = VAR_SPECS[sname][2]
        if native in ds.data_vars:
            ds = ds.rename({native: sname})

    # Accumulated forecast files use (forecast_initial_time, forecast_hour)
    # instead of (time,).  Flatten by computing valid_time = fc_init + fc_hour*1h
    # and stacking into a single hourly time dimension.
    if "forecast_initial_time" in ds.dims and "forecast_hour" in ds.dims:
        ds = _flatten_accumu_to_time(ds)

    cropped = ds.load()
    ds.close()
    try:
        os.remove(raw_path)
    except OSError:
        pass

    crop_t = time.time() - t0 - dl_t
    print(f"     {os.path.basename(key)}  dl={dl_t:.0f}s crop={crop_t:.0f}s "
          f"shape={dict(cropped.sizes)}", flush=True)
    return cropped


def download_var_year(client, sname: str, year: int, out_dir: str, raw_dir: str,
                      bbox: dict = None):
    if sname not in VAR_SPECS:
        print(f"  [skip] unknown var {sname}")
        return
    var_code, prefix, _native = VAR_SPECS[sname]
    out_path = os.path.join(out_dir, f"{sname}_{year}.nc")
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} exists")
        return

    print(f"\n→ {sname} ({var_code}) {year}", flush=True)
    monthly = []
    for month in range(1, 13):
        keys = list_monthly_keys(client, var_code, prefix, year, month)
        if not keys:
            print(f"   {year}-{month:02d}: no files")
            continue
        for key in keys:
            try:
                monthly.append(download_and_crop(client, key, raw_dir,
                                                 sname=sname, bbox=bbox))
            except Exception as e:
                print(f"     [error] {key}: {e}", flush=True)

    if not monthly:
        print(f"  [warn] nothing for {sname} {year}")
        return

    merged = xr.concat(monthly, dim="time").sortby("time")
    merged.to_netcdf(out_path, encoding=_build_encoding(merged))
    print(f"  ✓ wrote {out_path}  ({dict(merged.sizes)})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us", choices=["us", "world"],
                        help="us → CONUS+Alaska bbox; world → full globe (no crop)")
    parser.add_argument("--year", type=int, action="append", default=None)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--var", action="append", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--raw-dir", default="/tmp/era5_aws_raw",
                        help="Temp dir for raw monthly downloads (deleted after crop)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel (var, year) downloads (default: 1)")
    parser.add_argument("--rechunk-only", action="store_true",
                        help="Skip download; rewrite existing yearly files to "
                             "(1, lat, lon) chunk layout and exit")
    args = parser.parse_args()

    cfg = WorldConfig() if args.region == "world" else USConfig()
    bbox = WORLD_BBOX if args.region == "world" else US_BBOX
    out_dir = args.out_dir or os.path.join(cfg.era5_dir, "expanded")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)
    print(f"Region:     {args.region}")
    print(f"Output dir: {out_dir}")
    print(f"Raw temp:   {args.raw_dir}")

    if args.rechunk_only:
        rechunk_existing(out_dir, workers=args.workers)
        return

    years = args.years or args.year or DEFAULT_YEARS
    selected = args.var or list(VAR_SPECS.keys())

    print(f"Variables: {selected}")
    print(f"Years:     {years}")
    print(f"BBox:      lat [{bbox['lat_min']}, {bbox['lat_max']}], "
          f"lon [{bbox['lon_min']}, {bbox['lon_max']}]")
    print(f"Workers:   {args.workers}")

    tasks = [(s, y) for y in years for s in selected]
    print(f"Total tasks: {len(tasks)}")

    def _run(task):
        sname, year = task
        client = s3_client()
        try:
            download_var_year(client, sname, year, out_dir, args.raw_dir, bbox=bbox)
        except Exception as e:
            print(f"  [error] {sname} {year}: {e}")

    if args.workers > 1:
        with ThreadPool(args.workers) as pool:
            pool.map(_run, tasks)
    else:
        for t in tasks:
            _run(t)


if __name__ == "__main__":
    main()
