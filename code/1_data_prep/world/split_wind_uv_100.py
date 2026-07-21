import xarray as xr, numpy as np, time, os
SRC = '/net/fs06/d3/rzhuang/TROPOMI/data/world/era5/wind_uv_100.nc'
DST_DIR = '/net/fs06/d3/rzhuang/TROPOMI/data/world/era5/expanded'
os.makedirs(DST_DIR, exist_ok=True)

print('opening...', flush=True)
t0 = time.time()
ds = xr.open_dataset(SRC, engine='netcdf4')
tvar = 'valid_time' if 'valid_time' in ds.coords else 'time'
year = int(str(ds[tvar].values[0])[:4])
print(f'year={year}, time={tvar}, sizes={dict(ds.sizes)}', flush=True)

# Drop scalar coords
drop_coords = [c for c in ds.coords if c not in ('latitude', 'longitude', tvar)
               and ds[c].ndim == 0]
ds = ds.drop_vars(drop_coords, errors='ignore')

# Normalize lon 0..360 → -180..180 to match US AWS-format convention
if float(ds.longitude.min()) >= 0:
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby('longitude')
    print(f'lon normalized to {float(ds.longitude.min()):.1f}..{float(ds.longitude.max()):.1f}',
          flush=True)

for var in ['u100', 'v100']:
    out = f'{DST_DIR}/{var}_{year}.nc'
    print(f'\nwriting {out}...', flush=True)
    sub = ds[[var]].rename({tvar: 'time'})
    enc = {var: {"zlib": True, "complevel": 4,
                  "chunksizes": (1, sub.sizes['latitude'], sub.sizes['longitude'])}}
    sub.to_netcdf(out, encoding=enc)
    print(f'  {os.path.getsize(out)/1024/1024:.0f} MB  ({time.time()-t0:.0f}s elapsed)', flush=True)
ds.close()
print('done', flush=True)
