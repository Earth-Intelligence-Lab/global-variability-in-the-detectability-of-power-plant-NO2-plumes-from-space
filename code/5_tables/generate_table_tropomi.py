"""
Unified TROPOMI variable extraction (step 5a).

Replaces us/5_generate_whole_table_TROPOMI.py and world/5_generate_whole_table_TROPOMI.py.
The two original scripts were 99% identical (only the input/output CSV paths
differed). Both are now driven by config/{us,world}.py.

Usage:
    python generate_table_tropomi.py --region us
    python generate_table_tropomi.py --region world

Behavior is intended to be byte-equivalent to the originals.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import netCDF4 as nc
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from config import get_config  # noqa: E402
from shared.data_utils import build_balltree, load_tropomi_full_vars  # noqa: E402


def process_file_group(args):
    """Process all records sharing the same TROPOMI source file."""
    file_path, records = args
    t0 = time.time()
    print(f"▶ Processing {os.path.basename(file_path)} ({len(records)} recs)")

    ds = nc.Dataset(file_path)
    prod = ds.groups['PRODUCT']
    lats = prod.variables['latitude'][0]

    tree = build_balltree(lats, prod.variables['longitude'][0])
    vars_2d = load_tropomi_full_vars(ds)
    print(f" • Loaded vars: {', '.join(vars_2d.keys())}")

    out = []
    for rec in records:
        lat = rec['latitude']
        lon = rec['longitude']
        point = np.radians([[lat, lon]])
        idx = tree.query(point, k=1)[1][0][0]
        gi = np.unravel_index(idx, lats.shape)

        new_rec = rec.copy()
        for name, arr2d in vars_2d.items():
            val = arr2d[gi]
            new_rec[name] = float(np.ma.filled(val, np.nan))
        out.append(new_rec)

    ds.close()
    print(f"✅ Done {os.path.basename(file_path)} in {time.time()-t0:.1f}s")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=["us", "world"])
    parser.add_argument("--input", default=None,
                        help="Input CSV (default: cfg.snapshots_csv)")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: cfg.tropomi_table_csv)")
    args = parser.parse_args()

    cfg = get_config(args.region)
    input_csv = args.input or cfg.snapshots_csv
    output_csv = args.output or cfg.tropomi_table_csv

    mp.set_start_method('fork')

    df = pd.read_csv(input_csv)
    groups = []
    for file_path, grp in df.groupby('file_path'):
        records = grp.to_dict('records')
        groups.append((file_path, records))

    print(f"Loaded {len(df)} records, {len(groups)} unique files")
    print(f"Starting pool with {mp.cpu_count()} workers")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_file_group, groups)

    all_records = [rec for sub in results for rec in sub]
    out_df = pd.DataFrame(all_records)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Written {len(all_records)} records → {output_csv}")


if __name__ == '__main__':
    main()
