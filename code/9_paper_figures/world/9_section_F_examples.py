"""Publication-quality Section F examples — consolidated multi-panel figure.

Replaces the previous nine-figure Appendix-F gallery (each a 6-panel 2×3
diagnostic grid) with a single multi-panel publication figure.

Each panel shows ONE (facility, overpass) example as a single axes:
    - NO₂ tropospheric column (viridis pcolormesh)
    - Detected plume contour (red, traced along TROPOMI pixel edges)
    - Cyan circles for city interference zones
    - Magenta circles for nearby-plant interference zones
    - Cyan fancy arrow showing wind-to direction
    - Red star for the target facility
    - Compact title: facility ID, country, date, wind, NOx, plume area

Reviewer-driven changes vs the original 6-panel gallery:
    1) one figure replaces nine (panels = ~6 examples)
    2) all symbols enlarged ≥ 3×: facility star markersize=400, other plants
       s=250, cities s=250, wind arrow visible (head_length=1.6)
    3) explicit cyan wind arrow (was missing in the original render)
    4) Background and Anomaly sub-panels dropped (redundant for an Appendix
       gallery; the same information is summarised in the panel title)
    5) Satellite/Urban basemaps dropped — out of scope for an algorithm
       illustration; readers can identify location from the title
"""
from __future__ import annotations
import os
import warnings
from collections import OrderedDict
from math import cos, log10, radians, sqrt

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from pyproj import Geod
from scipy.ndimage import gaussian_filter, label as ndi_label
from scipy.ndimage import label as _ndi_label

# ── Paths ─────────────────────────────────────────────────────────────────
WORLD_CSV = ('/net/fs06/d3/rzhuang/TROPOMI/data/world/'
             'pipeline_test_labelling_100m/Run_100m_20260428/'
             'valid_tropomi_emissions_with_qa.csv')
PLANTS    = ('/net/fs06/d3/rzhuang/TROPOMI/data/world/'
             'power_plant_location/power_plants_with_combined_nearby_stats.csv')
CITIES    = '/net/fs06/d3/rzhuang/TROPOMI/data/world/worldcities.csv'
OUT_DIR   = '/net/fs06/d3/rzhuang/TROPOMI/results/paper_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Algorithm constants (paper §3.1, world version) ───────────────────────
INTERF_MAX_DIST_KM      = 150.0
CITY_POP_THRESH         = 200_000
CITY_RADIUS_BASE        = 0.0
CITY_RADIUS_SCALE       = 9.0       # original world setting
CITY_RADIUS_MIN         = 10.0
CITY_RADIUS_MAX         = 90.0
CLOSE_DISTANCE_KM_MASK  = 20.0
THRESHOLD_FACTOR        = 2.0
THRESHOLD_ABS_MIN       = 5e-6
THRESHOLD_RADIUS_KM     = 50.0
MAX_DISTANCE_KM         = 20.0
CLOSE_DISTANCE_KM       = 5.0
MAX_ANGLE_DIFF          = 25.0
FLAGGED_AREA            = 25.0
BG_DIST_MIN_KM          = 10.0
BG_DIST_MAX_KM          = 100.0
BG_ANGLE_TOL            = 60.0
ZOOM_RADIUS_KM_PLOT     = 160.0     # algorithm zoom (sets analysis box)
PLOT_BOX_DEG            = 1.0        # display half-window in degrees (~110 km)
KM_PER_DEG_LAT          = 111.1

# ── Style (Nimbus Roman) ──────────────────────────────────────────────────
plt.rcdefaults()
plt.rcParams['figure.facecolor'] = 'white'
nimbus_path = None
for path in fm.findSystemFonts():
    pl = path.lower()
    if (('nimbusroman' in pl or 'nimbus_roman' in pl)
            and 'bold' not in pl and 'italic' not in pl and 'oblique' not in pl):
        nimbus_path = path
        break
TITLE_FP    = fm.FontProperties(fname=nimbus_path, size=15) if nimbus_path else fm.FontProperties(size=15)
LEGEND_FP   = fm.FontProperties(fname=nimbus_path, size=14) if nimbus_path else fm.FontProperties(size=14)
TICK_FP     = fm.FontProperties(fname=nimbus_path, size=12) if nimbus_path else fm.FontProperties(size=12)
CBAR_LBL_FP = fm.FontProperties(fname=nimbus_path, size=15) if nimbus_path else fm.FontProperties(size=15)


# ── Plume detection (compact, mirrors `label_no2_plume_flexible_interference`)
def calc_areas(mask, lat2d, lon2d):
    ys, xs = np.where(mask)
    if not len(ys): return 0.0
    deg2km = 111.14
    out = []
    for y, x in zip(ys, xs):
        a = []
        for dx in (1, -1):
            nx = x + dx
            if 0 <= nx < lon2d.shape[1]:
                dlon = abs(lon2d[y, nx] - lon2d[y, x])
                for dy in (1, -1):
                    ny = y + dy
                    if 0 <= ny < lat2d.shape[0]:
                        dlat = abs(lat2d[ny, x] - lat2d[y, x])
                        a.append(dlat * deg2km * dlon * deg2km
                                 * np.cos(np.radians(lat2d[y, x])))
        out.append(np.mean(a) if a else 25.0)
    return float(np.sum(out))


def create_geodesic_circle(center_lon, center_lat, radius_km, num_points=120):
    geod = Geod(ellps='WGS84')
    az = np.linspace(0, 360, num_points)
    lons, lats, _ = geod.fwd(np.full(num_points, center_lon),
                             np.full(num_points, center_lat),
                             az, np.full(num_points, radius_km * 1000.0))
    return lons, lats


def prepare_interfering(target_lat, target_lon, target_id, plants_df, cities_df,
                        target_emission):
    """World-version helper: target plant & city interference identification.

    Each entry now carries `name` for plotting captions:
        cities → city name (e.g. 'Tokyo')
        plants → facility ID (e.g. 'CoCO2_00224')
    """
    geod = Geod(ellps='WGS84')
    out = []
    if cities_df is not None and len(cities_df):
        in_box = cities_df[(cities_df['latitude'].between(target_lat-2, target_lat+2)) &
                           (cities_df['longitude'].between(target_lon-2, target_lon+2))]
        for _, row in in_box.iterrows():
            if row['population'] < CITY_POP_THRESH:
                continue
            d_km = geod.inv(target_lon, target_lat,
                            row['longitude'], row['latitude'])[2] / 1000.0
            if d_km > INTERF_MAX_DIST_KM: continue
            r = CITY_RADIUS_BASE + CITY_RADIUS_SCALE * log10(max(1, row['population']))
            r = max(CITY_RADIUS_MIN, min(r, CITY_RADIUS_MAX))
            out.append({'type': 'city', 'lat': row['latitude'],
                        'lon': row['longitude'], 'radius_km': r,
                        'name': str(row.get('name', ''))})

    others = plants_df[plants_df['ID'] != target_id]
    in_box = others[(others['latitude'].between(target_lat-2, target_lat+2)) &
                    (others['longitude'].between(target_lon-2, target_lon+2))]
    for _, row in in_box.iterrows():
        v = row.get('nox_emis_ty', 0)
        if pd.isna(v) or v < target_emission:
            continue
        d_km = geod.inv(target_lon, target_lat,
                        row['longitude'], row['latitude'])[2] / 1000.0
        if d_km > INTERF_MAX_DIST_KM: continue
        out.append({'type': 'plant', 'lat': row['latitude'],
                    'lon': row['longitude'], 'radius_km': 0,
                    'name': str(row.get('ID', ''))})
    return out


def detect_plume(no2_full, lat_full, lon_full, plant_lat, plant_lon,
                 wind_u, wind_v, interfering):
    """Return dict with plume_mask, area, threshold, zoomed lat/lon/no2."""
    geod = Geod(ellps='WGS84')
    tol = 1e-12

    dlat = ZOOM_RADIUS_KM_PLOT / 111.1
    dlon = ZOOM_RADIUS_KM_PLOT / (111.1 * max(cos(radians(plant_lat)), 1e-9))
    box = ((lat_full >= plant_lat-dlat) & (lat_full <= plant_lat+dlat) &
           (lon_full >= plant_lon-dlon) & (lon_full <= plant_lon+dlon))
    if not box.any():
        return None
    rs, cs = np.where(box)
    r0, r1 = max(rs.min()-2, 0), min(rs.max()+2, lat_full.shape[0]-1)
    c0, c1 = max(cs.min()-2, 0), min(cs.max()+2, lon_full.shape[1]-1)
    no2 = no2_full[r0:r1+1, c0:c1+1].copy()
    lat = lat_full[r0:r1+1, c0:c1+1].copy()
    lon = lon_full[r0:r1+1, c0:c1+1].copy()

    geo = ~np.isnan(lat) & ~np.isnan(lon)
    dist_km = np.full_like(no2, np.nan, float)
    azm = np.full_like(no2, np.nan, float)
    if geo.any():
        latf = lat[geo].flatten()
        lonf = lon[geo].flatten()
        fwd, _, dm = geod.inv(np.full_like(lonf, plant_lon),
                              np.full_like(latf, plant_lat), lonf, latf)
        dist_km[geo] = dm / 1000.0
        azm[geo] = (fwd + 360) % 360

    if pd.isna(wind_u) or pd.isna(wind_v) or (abs(wind_u) < tol and abs(wind_v) < tol):
        wind_to = wind_from = np.nan
    else:
        wind_to = (np.degrees(np.arctan2(wind_u, wind_v)) + 360) % 360
        wind_from = (wind_to + 180) % 360

    valid = (~np.isnan(no2)) & (np.abs(no2) > tol) & geo
    interf_mask = np.ones_like(no2, bool)
    if interfering and geo.any():
        lonf = lon[geo]
        latf = lat[geo]
        for src in interfering:
            slat, slon, srad = src['lat'], src['lon'], src['radius_km']
            _, _, dm = geod.inv(np.full_like(lonf, slon),
                                np.full_like(latf, slat), lonf, latf)
            d_km = dm / 1000.0
            in_rad = d_km <= srad
            if src['type'] == 'plant':
                in_rad = in_rad | (d_km <= CLOSE_DISTANCE_KM_MASK)
            tm = np.zeros_like(no2, dtype=bool)
            tm[geo] = in_rad
            interf_mask &= ~tm

    calc_mask = valid & interf_mask
    no2_calc = no2.copy()
    no2_calc[~calc_mask] = np.nan

    # Background (directional, fall back to gaussian if no wind)
    flat_idx = np.where(calc_mask.ravel())[0]
    background = np.full_like(no2, np.nan, float)
    if flat_idx.size and pd.notna(wind_from):
        d_v = dist_km.ravel()[flat_idx]
        a_v = azm.ravel()[flat_idx]
        n_v = no2_calc.ravel()[flat_idx]
        ok = ~np.isnan(d_v) & ~np.isnan(a_v)
        sector = (np.abs(((a_v[ok] - wind_from + 180) % 360) - 180) <= BG_ANGLE_TOL/2) & \
                 (d_v[ok] >= BG_DIST_MIN_KM) & (d_v[ok] <= BG_DIST_MAX_KM)
        bg_pixels = n_v[ok][sector]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            bg_val = np.nanmedian(bg_pixels) if bg_pixels.size else np.nanmedian(n_v)
        background.fill(bg_val if pd.notna(bg_val) else 0.0)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            background.fill(np.nanmedian(no2[valid]) if valid.any() else 0.0)

    anomalies = no2 - background
    anomalies[~valid] = np.nan

    loc = (dist_km <= THRESHOLD_RADIUS_KM) & ~np.isnan(dist_km) & ~np.isnan(anomalies)
    locv = anomalies[loc]
    if locv.size > 5:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(locv)
            mad = np.nanmedian(np.abs(locv - med))
            sigma = mad * 1.4826 if mad > tol else np.nanstd(locv)
        stat_thresh = med + THRESHOLD_FACTOR * sigma
        thresh = max(stat_thresh, THRESHOLD_ABS_MIN)
    else:
        thresh = THRESHOLD_ABS_MIN

    plume_mask = np.zeros_like(no2, bool)
    valid_idx = np.where(valid.ravel())[0]
    if valid_idx.size and pd.notna(wind_to):
        d_all = dist_km.ravel()[valid_idx]
        az_all = azm.ravel()[valid_idx]
        an_all = anomalies.ravel()[valid_idx]
        use = ~np.isnan(d_all) & ~np.isnan(az_all) & ~np.isnan(an_all)
        if use.any():
            d_u, az_u, an_u = d_all[use], az_all[use], an_all[use]
            in_close = d_u <= CLOSE_DISTANCE_KM
            ang_diff = np.abs(((az_u - wind_to + 180) % 360) - 180)
            in_cone = (d_u <= MAX_DISTANCE_KM) & (ang_diff <= MAX_ANGLE_DIFF)
            avail = (in_close | in_cone)
            stat_pass = an_u > thresh
            abs_pass = an_u > THRESHOLD_ABS_MIN
            cond = avail & stat_pass & abs_pass
            plume_mask.ravel()[valid_idx[use][cond]] = True
    plume_mask &= interf_mask

    area = calc_areas(plume_mask, lat, lon)
    return dict(no2=no2, lat=lat, lon=lon, plume_mask=plume_mask,
                area_km2=area, threshold=thresh,
                wind_to=wind_to, wind_from=wind_from)


# ── Pixel-edge plume contour drawing ──────────────────────────────────────
def _pixel_corners(z_lon, z_lat):
    rows, cols = z_lon.shape
    lon_c = np.zeros((rows + 1, cols + 1))
    lat_c = np.zeros((rows + 1, cols + 1))
    for i in range(1, rows):
        for j in range(1, cols):
            lon_c[i, j] = 0.25*(z_lon[i-1, j-1]+z_lon[i-1, j]+z_lon[i, j-1]+z_lon[i, j])
            lat_c[i, j] = 0.25*(z_lat[i-1, j-1]+z_lat[i-1, j]+z_lat[i, j-1]+z_lat[i, j])
    for j in range(1, cols):
        lon_c[0, j] = 0.5*(z_lon[0, j-1] + z_lon[0, j]) - 0.5*(lon_c[1, j] - z_lon[0, j-1:j+1].mean())
        lat_c[0, j] = 0.5*(z_lat[0, j-1] + z_lat[0, j]) - 0.5*(lat_c[1, j] - z_lat[0, j-1:j+1].mean())
        lon_c[rows, j] = 0.5*(z_lon[rows-1, j-1]+z_lon[rows-1, j]) + 0.5*(z_lon[rows-1, j-1:j+1].mean() - lon_c[rows-1, j])
        lat_c[rows, j] = 0.5*(z_lat[rows-1, j-1]+z_lat[rows-1, j]) + 0.5*(z_lat[rows-1, j-1:j+1].mean() - lat_c[rows-1, j])
    for i in range(1, rows):
        lon_c[i, 0] = 0.5*(z_lon[i-1, 0]+z_lon[i, 0]) - 0.5*(lon_c[i, 1] - z_lon[i-1:i+1, 0].mean())
        lat_c[i, 0] = 0.5*(z_lat[i-1, 0]+z_lat[i, 0]) - 0.5*(lat_c[i, 1] - z_lat[i-1:i+1, 0].mean())
        lon_c[i, cols] = 0.5*(z_lon[i-1, cols-1]+z_lon[i, cols-1]) + 0.5*(z_lon[i-1:i+1, cols-1].mean() - lon_c[i, cols-1])
        lat_c[i, cols] = 0.5*(z_lat[i-1, cols-1]+z_lat[i, cols-1]) + 0.5*(z_lat[i-1:i+1, cols-1].mean() - lat_c[i, cols-1])
    lon_c[0, 0] = z_lon[0, 0] - 0.5*(lon_c[1, 1] - z_lon[0, 0])
    lat_c[0, 0] = z_lat[0, 0] - 0.5*(lat_c[1, 1] - z_lat[0, 0])
    lon_c[0, cols] = z_lon[0, cols-1] + 0.5*(z_lon[0, cols-1] - lon_c[1, cols-1])
    lat_c[0, cols] = z_lat[0, cols-1] + 0.5*(z_lat[0, cols-1] - lat_c[1, cols-1])
    lon_c[rows, 0] = z_lon[rows-1, 0] + 0.5*(z_lon[rows-1, 0] - lon_c[rows-1, 1])
    lat_c[rows, 0] = z_lat[rows-1, 0] + 0.5*(z_lat[rows-1, 0] - lat_c[rows-1, 1])
    lon_c[rows, cols] = z_lon[rows-1, cols-1] + 0.5*(z_lon[rows-1, cols-1] - lon_c[rows-1, cols-1])
    lat_c[rows, cols] = z_lat[rows-1, cols-1] + 0.5*(z_lat[rows-1, cols-1] - lat_c[rows-1, cols-1])
    return lon_c, lat_c


def draw_plume_contour(ax, plume_mask, z_lon, z_lat, color='red', lw=3.5, zorder=8):
    if not np.any(plume_mask): return
    rows, cols = plume_mask.shape
    lon_c, lat_c = _pixel_corners(z_lon, z_lat)
    labeled, n_feat = _ndi_label(plume_mask)
    for fid in range(1, n_feat + 1):
        feat = (labeled == fid)
        edges = []
        for i in range(rows):
            for j in range(cols):
                if not feat[i, j]: continue
                if i == 0 or not feat[i-1, j]:
                    edges.append(((lon_c[i, j],     lat_c[i, j]),
                                  (lon_c[i, j+1],   lat_c[i, j+1])))
                if i == rows-1 or not feat[i+1, j]:
                    edges.append(((lon_c[i+1, j],   lat_c[i+1, j]),
                                  (lon_c[i+1, j+1], lat_c[i+1, j+1])))
                if j == 0 or not feat[i, j-1]:
                    edges.append(((lon_c[i, j],     lat_c[i, j]),
                                  (lon_c[i+1, j],   lat_c[i+1, j])))
                if j == cols-1 or not feat[i, j+1]:
                    edges.append(((lon_c[i, j+1],   lat_c[i, j+1]),
                                  (lon_c[i+1, j+1], lat_c[i+1, j+1])))
        for (x1, y1), (x2, y2) in edges:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=zorder)


# ── One panel renderer ────────────────────────────────────────────────────
def render_satellite_panel(ax, plant_lat, plant_lon, plant_id, country, utc_time,
                           interfering=None, all_plants_df=None, all_cities_df=None):
    """Satellite panel: imagery + ALL cities (○) + ALL other plants (▲) within
    the plotting window. Interfering subset (those that drove the masks in
    the NO₂ panel) is drawn with bigger / thicker markers to stand out."""
    x_min, x_max = plant_lon - PLOT_BOX_DEG, plant_lon + PLOT_BOX_DEG
    y_min, y_max = plant_lat - PLOT_BOX_DEG, plant_lat + PLOT_BOX_DEG
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('lightgray')
    try:
        ctx.add_basemap(ax, crs='EPSG:4326',
                        source=ctx.providers.Esri.WorldImagery,
                        zoom='auto', attribution=False, zorder=0)
    except Exception as e:
        print(f"  satellite basemap failed for {plant_id}: "
              f"{type(e).__name__}: {e}", flush=True)
        ax.text(0.5, 0.5, '(satellite tiles unavailable)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray')

    # Set of (lon, lat) tuples that are flagged as interfering
    interf_keys = set()
    if interfering:
        for src in interfering:
            interf_keys.add((round(src['lon'], 4), round(src['lat'], 4),
                             src['type']))

    # ALL cities in the plot window — small base marker, big for interfering
    if all_cities_df is not None and len(all_cities_df):
        cw = all_cities_df[(all_cities_df['latitude'].between(y_min, y_max)) &
                           (all_cities_df['longitude'].between(x_min, x_max))]
        for _, row in cw.iterrows():
            key = (round(row['longitude'], 4), round(row['latitude'], 4), 'city')
            is_interf = key in interf_keys
            ax.scatter(row['longitude'], row['latitude'],
                       s=(240 if is_interf else 80),
                       marker='o', facecolor='orange',
                       edgecolor='black',
                       linewidth=(1.5 if is_interf else 0.8),
                       alpha=(1.0 if is_interf else 0.85),
                       zorder=(8 if is_interf else 6))

    # ALL other power plants in the plot window — small base, big if interfering
    if all_plants_df is not None and len(all_plants_df):
        pw = all_plants_df[(all_plants_df['latitude'].between(y_min, y_max)) &
                           (all_plants_df['longitude'].between(x_min, x_max)) &
                           (all_plants_df['ID'] != plant_id)]
        for _, row in pw.iterrows():
            key = (round(row['longitude'], 4), round(row['latitude'], 4), 'plant')
            is_interf = key in interf_keys
            ax.scatter(row['longitude'], row['latitude'],
                       s=(260 if is_interf else 90),
                       marker='^', facecolor='#00FF66',
                       edgecolor='black',
                       linewidth=(1.5 if is_interf else 0.8),
                       alpha=(1.0 if is_interf else 0.85),
                       zorder=(8 if is_interf else 6))

    # Target plant — red star (drawn last, on top)
    ax.plot(plant_lon, plant_lat, 'r*', markersize=24,
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax.set_title(f'Satellite — {plant_id} ({country})',
                 fontproperties=TITLE_FP, pad=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(TICK_FP)


def add_locator_inset(ax, plant_lon, plant_lat,
                      pos=(0.015, 0.66, 0.34, 0.26)):
    """Add a small world-map inset to `ax` showing the facility location.

    `pos` is (x, y, w, h) in axes-relative coordinates.
    """
    inset = ax.inset_axes(pos, projection=ccrs.PlateCarree())
    inset.set_global()
    inset.add_feature(cfeature.LAND.with_scale('110m'),
                      facecolor='lightgray', edgecolor='none', zorder=1)
    inset.add_feature(cfeature.OCEAN.with_scale('110m'),
                      facecolor='white', edgecolor='none', zorder=0)
    inset.add_feature(cfeature.COASTLINE.with_scale('110m'),
                      linewidth=0.35, edgecolor='black', zorder=2)
    inset.scatter(plant_lon, plant_lat, transform=ccrs.PlateCarree(),
                  s=55, marker='*', facecolor='red',
                  edgecolor='black', linewidth=0.8, zorder=5)
    inset.set_xticks([]); inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
    inset.patch.set_facecolor('white')
    inset.patch.set_alpha(0.92)
    return inset


def render_panel(ax, no2_full, lat_full, lon_full,
                 plant_lat, plant_lon, plant_id, country, utc_time,
                 wind_u, wind_v, annual_nox, plants_df, cities_df,
                 target_emission, vmin=1e-5, vmax=5e-5, category=None):
    """Render one publication-style example panel onto `ax`."""
    interf = prepare_interfering(plant_lat, plant_lon, plant_id,
                                 plants_df, cities_df, target_emission)
    res = detect_plume(no2_full, lat_full, lon_full,
                       plant_lat, plant_lon, wind_u, wind_v, interf)
    if res is None:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center')
        return None
    z_no2, z_lat, z_lon = res['no2'], res['lat'], res['lon']

    # Lock axis extent to the plotting box
    x_min, x_max = plant_lon - PLOT_BOX_DEG, plant_lon + PLOT_BOX_DEG
    y_min, y_max = plant_lat - PLOT_BOX_DEG, plant_lat + PLOT_BOX_DEG
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')

    # Per-panel auto vmin/vmax: 5–95 percentile range
    valid_vals = z_no2[np.isfinite(z_no2)]
    if valid_vals.size > 10:
        v_lo, v_hi = np.percentile(valid_vals, [5, 95])
        if v_hi <= v_lo:
            v_lo, v_hi = vmin, vmax
    else:
        v_lo, v_hi = vmin, vmax

    # Solid NO₂ pcolormesh (satellite is now a separate panel)
    im = ax.pcolormesh(z_lon, z_lat, z_no2, cmap='viridis',
                       shading='auto', vmin=v_lo, vmax=v_hi, zorder=1)

    # Interference zones (cyan = city, magenta = plant) — plotted UNDER plume
    # Names/markers for the actual city and plant positions go in the
    # satellite panel; here we only show the masking zones.
    city_seen = plant_seen = False
    for src in interf:
        rad = src['radius_km'] if src['type'] == 'city' else CLOSE_DISTANCE_KM_MASK
        cl, ca = create_geodesic_circle(src['lon'], src['lat'], rad)
        col = 'cyan' if src['type'] == 'city' else 'magenta'
        ax.fill(cl, ca, color=col, alpha=0.18, zorder=2)
        ax.plot(cl, ca, color=col, linewidth=2.5, alpha=0.9, zorder=3)
        if src['type'] == 'city':
            city_seen = True
        else:
            plant_seen = True

    # Plume contour (red, traces pixel edges)
    if res['area_km2'] >= FLAGGED_AREA:
        draw_plume_contour(ax, res['plume_mask'], z_lon, z_lat,
                           color='red', lw=3.0, zorder=7)

    # Wind arrow (cyan, big fancy)
    arrow_drawn = False
    if pd.notna(wind_u) and pd.notna(wind_v) and not (wind_u == 0 and wind_v == 0):
        wind_to_rad = np.arctan2(wind_v, wind_u)   # math convention
        arrow_len_km = 35.0
        arrow_len_deg = arrow_len_km / KM_PER_DEG_LAT
        end_x = plant_lon + arrow_len_deg * np.cos(wind_to_rad)
        end_y = plant_lat + arrow_len_deg * np.sin(wind_to_rad)
        ax.annotate('', xy=(end_x, end_y), xytext=(plant_lon, plant_lat),
                    arrowprops=dict(arrowstyle='fancy,head_length=1.6,head_width=1.2,tail_width=0.7',
                                    facecolor='cyan', edgecolor='black', linewidth=1.8,
                                    shrinkA=12, shrinkB=8),
                    zorder=9)
        arrow_drawn = True

    # Target plant — big red star
    ax.plot(plant_lon, plant_lat, 'r*', markersize=24,
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    # World-map locator inset (top-left corner)
    add_locator_inset(ax, plant_lon, plant_lat)

    # Title (compact, multi-line)
    wind_speed = sqrt(wind_u**2 + wind_v**2)
    wind_from_deg = (np.degrees(np.arctan2(wind_u, wind_v)) + 180 + 360) % 360
    detected = res['area_km2'] >= FLAGGED_AREA
    plume_str = f"Plume detected — {res['area_km2']:.0f} km²" if detected else "No plume detected"
    cat_str = f"[{category}]  " if category else ""
    # 3-line title — keeps long lines from being clipped at narrow panel widths
    title = (f"{cat_str}{plant_id} ({country}) — {utc_time[:10]}\n"
             f"NOx {annual_nox:.0f} t/yr · Wind {wind_from_deg:.0f}° @ {wind_speed:.1f} m/s\n"
             f"{plume_str}")
    ax.set_title(title, fontproperties=TITLE_FP, pad=8)

    # Tick labels
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(TICK_FP)

    return dict(im=im, city_seen=city_seen, plant_seen=plant_seen,
                arrow_drawn=arrow_drawn, detected=detected,
                interfering=interf)


# ── Master figure ─────────────────────────────────────────────────────────
def make_master_figure(snapshots, out_path, panel_height=7.0, panel_width=7.5):
    """One row per example, two columns: [NO₂ + plume + wind] | [Satellite].

    snapshots: list of dicts with keys
        plant_id, country, utc_time, plant_lat, plant_lon,
        wind_u, wind_v, annual_nox, file_path, category.
    """
    print(f'Loading auxiliary data ...', flush=True)
    plants_df = pd.read_csv(PLANTS)
    cities_df = pd.read_csv(CITIES)
    cities_df['population'] = pd.to_numeric(cities_df['population'], errors='coerce').fillna(0)

    n = len(snapshots)
    # Satellite panel made wider than the NO₂ panel via gridspec width_ratios.
    # Total figure width also bumped slightly to accommodate.
    fig, axes = plt.subplots(
        n, 2, figsize=(2.4 * panel_width, n * panel_height),
        dpi=150, constrained_layout=False, squeeze=False,
        gridspec_kw={'width_ratios': [1.0, 1.4]},
    )

    legend_state = {'city': False, 'plant': False, 'arrow': False, 'plume': False}
    for i, snap in enumerate(snapshots):
        print(f"  rendering [{snap.get('category','')}] {snap['plant_id']} "
              f"({snap['country']}) ...", flush=True)
        no2_full, lat_full, lon_full = _load_swath(snap['file_path'])

        # Left: NO₂ + plume + wind + interference + locator inset
        info = render_panel(
            axes[i, 0], no2_full, lat_full, lon_full,
            plant_lat=snap['plant_lat'], plant_lon=snap['plant_lon'],
            plant_id=snap['plant_id'], country=snap['country'],
            utc_time=snap['utc_time'],
            wind_u=snap['wind_u'], wind_v=snap['wind_v'],
            annual_nox=snap['annual_nox'],
            plants_df=plants_df, cities_df=cities_df,
            target_emission=snap['annual_nox'],
            category=snap.get('category'),
        )
        if info is not None:
            if info['city_seen']:   legend_state['city']  = True
            if info['plant_seen']:  legend_state['plant'] = True
            if info['arrow_drawn']: legend_state['arrow'] = True
            if info['detected']:    legend_state['plume'] = True

            # Per-panel colorbar attached to the right edge of the NO₂ axis
            cbar = fig.colorbar(info['im'], ax=axes[i, 0],
                                orientation='vertical', pad=0.02,
                                fraction=0.046, shrink=0.95)
            cbar.set_label('NO$_2$ (mol/m²)', fontproperties=CBAR_LBL_FP)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_major_formatter(formatter)
            cbar.ax.yaxis.get_offset_text().set_fontproperties(TICK_FP)
            for lbl in cbar.ax.get_yticklabels():
                lbl.set_fontproperties(TICK_FP)

        # Right: Satellite + ALL cities (○) and ALL other plants (▲)
        render_satellite_panel(
            axes[i, 1],
            plant_lat=snap['plant_lat'], plant_lon=snap['plant_lon'],
            plant_id=snap['plant_id'], country=snap['country'],
            utc_time=snap['utc_time'],
            interfering=info['interfering'] if info else None,
            all_plants_df=plants_df,
            all_cities_df=cities_df,
        )

    # Combined legend at bottom
    handles = [Line2D([0], [0], marker='*', color='black',
                      markerfacecolor='red', markersize=20, linestyle='None',
                      label='Target plant')]
    if legend_state['arrow']:
        handles.append(Line2D([0], [0], marker='>', color='cyan',
                              markersize=14, markeredgecolor='black',
                              linestyle='None', label='Wind direction'))
    if legend_state['plume']:
        handles.append(Line2D([0], [0], color='red', linewidth=3,
                              label='Detected plume contour'))
    if legend_state['city']:
        handles.append(Patch(facecolor='cyan', alpha=0.2,
                             edgecolor='cyan', linewidth=2.5,
                             label='Zone of no plume (city)'))
        handles.append(Line2D([0], [0], marker='o', color='black',
                              markerfacecolor='orange', markersize=14,
                              linestyle='None', label='City centre'))
    if legend_state['plant']:
        handles.append(Patch(facecolor='magenta', alpha=0.2,
                             edgecolor='magenta', linewidth=2.5,
                             label='Zone of no plume (other plant)'))
        handles.append(Line2D([0], [0], marker='^', color='black',
                              markerfacecolor='#00FF66', markersize=14,
                              linestyle='None', label='Other power plant'))
    # Wrap legend to 2 rows when there are >4 entries (keeps it compact)
    ncol = (len(handles) + 1) // 2 if len(handles) > 4 else len(handles)
    fig.legend(handles=handles, loc='lower center', ncol=ncol,
               frameon=True, facecolor='white', framealpha=0.9,
               prop=LEGEND_FP, bbox_to_anchor=(0.5, 0.0))

    fig.subplots_adjust(left=0.04, right=0.97, top=0.96, bottom=0.10,
                        wspace=0.30, hspace=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {out_path}', flush=True)


def _load_swath(file_path):
    with nc.Dataset(file_path) as ds:
        grp = ds.groups.get('PRODUCT', ds)
        lat = grp.variables['latitude'][:]
        lon = grp.variables['longitude'][:]
        no2 = grp.variables['nitrogendioxide_tropospheric_column'][:]
        if lat.ndim > 2:
            lat, lon, no2 = lat[0], lon[0], no2[0]
    fv = getattr(grp.variables['nitrogendioxide_tropospheric_column'], '_FillValue', None)
    if fv is not None:
        no2 = np.where(no2 == fv, np.nan, no2)
    return np.array(no2, dtype=float), np.array(lat, dtype=float), np.array(lon, dtype=float)


# ── Entry point ───────────────────────────────────────────────────────────
def lookup_snapshots(targets):
    """targets: list of (plant_id, day_str[, category]). Returns list of snapshot dicts."""
    df = pd.read_csv(WORLD_CSV, low_memory=False)
    snaps = []
    for t in targets:
        if len(t) == 2:
            fid, day = t; cat = None
        else:
            fid, day, cat = t
        m = (df['location'].astype(str).str.contains(fid, na=False) &
             df['utc_time'].astype(str).str.startswith(day))
        if not m.any():
            print(f'  WARN: {fid} on {day} not found in world CSV', flush=True)
            continue
        r = df[m].iloc[0]
        snaps.append(dict(
            plant_id=str(r['location']),
            country=str(r.get('country', '?')),
            utc_time=str(r['utc_time']),
            plant_lat=float(r['latitude']),
            plant_lon=float(r['longitude']),
            wind_u=float(r['wind_u']),
            wind_v=float(r['wind_v']),
            annual_nox=float(r.get('annual_nox_emission', np.nan)),
            file_path=str(r['file_path']),
            category=cat,
        ))
    return snaps


if __name__ == '__main__':
    # One figure per confusion-matrix category — 4 figures total replace 12.
    # ── Fig F1: True Positives (1×2) ──────────────────────────────────────
    targets_tp = [
        ('CoCO2_00224', '2018-09-23', 'TP'),   # AUS, 173.6 km² plume @ 4886 t/yr
        ('CoCO2_00341', '2018-09-13', 'TP'),   # AUS,  60.0 km² plume @  963 t/yr
    ]
    make_master_figure(lookup_snapshots(targets_tp),
                       os.path.join(OUT_DIR, 'section_F_TP.png'))

    # ── Fig F2: True Negatives (1×2) ──────────────────────────────────────
    targets_tn = [
        ('CoCO2_00042', '2018-11-21', 'TN'),   # ARG,  418 t/yr, no plume
        ('CoCO2_07946', '2018-10-03', 'TN'),   # JPN,  345 t/yr, no plume
    ]
    make_master_figure(lookup_snapshots(targets_tn),
                       os.path.join(OUT_DIR, 'section_F_TN.png'))

    # ── Fig F3a / F3b: False Positives — 4 examples split into 2 figures ──
    # (matches the 2-example-per-figure pacing of TP / TN / FN; avoids the
    #  single tall figure overflowing a LaTeX page when [H] is used.)
    targets_fp_a = [
        ('CoCO2_04813', '2018-11-26', 'FP'),   # CHN
        ('CoCO2_15320', '2018-05-11', 'FP'),   # IRL
    ]
    make_master_figure(lookup_snapshots(targets_fp_a),
                       os.path.join(OUT_DIR, 'section_F_FP_part1.png'))

    targets_fp_b = [
        ('CoCO2_00629', '2018-09-22', 'FP'),   # BRA
        ('CoCO2_00346', '2018-07-14', 'FP'),   # AUS
    ]
    make_master_figure(lookup_snapshots(targets_fp_b),
                       os.path.join(OUT_DIR, 'section_F_FP_part2.png'))

    # ── Fig F4: False Negatives — 2 examples (USA-11845 dropped) ──────────
    targets_fn = [
        ('CoCO2_09862', '2018-09-23', 'FN'),   # PAK,  3598 t/yr missed
        ('CoCO2_12006', '2018-11-10', 'FN'),   # USA,  7303 t/yr missed
    ]
    make_master_figure(lookup_snapshots(targets_fn),
                       os.path.join(OUT_DIR, 'section_F_FN.png'))
