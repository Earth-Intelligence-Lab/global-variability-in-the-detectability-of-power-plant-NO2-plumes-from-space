"""
Equivalence verification for the unified refactor.

Compares the output CSVs from the new unified scripts (find_snapshots.py,
generate_table_tropomi.py, generate_table_era5.py, merge_tables.py) against
the outputs of the original us/ + world/ scripts and reports any differences.

Usage:
    # Compare an existing legacy output against the new unified output
    python verify_refactor.py --legacy /path/to/legacy.csv --new /path/to/new.csv

The script:
  1. Loads both CSVs
  2. Checks shape matches
  3. Checks column sets match
  4. For numeric columns: max abs diff per column (should be ~0)
  5. For string columns: number of mismatched rows (should be 0)

Note: this is offline verification. To actually verify a step end-to-end you
need to run the unified script first (--output to a temporary path) and then
point this script at the legacy CSV the original produced.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def compare(legacy_path: str, new_path: str, atol: float = 1e-9) -> bool:
    print(f"Loading legacy: {legacy_path}")
    a = pd.read_csv(legacy_path)
    print(f"Loading new:    {new_path}")
    b = pd.read_csv(new_path)

    print(f"\nShapes: legacy={a.shape}, new={b.shape}")
    if a.shape != b.shape:
        print("✗ shape mismatch")
        return False

    cols_a = set(a.columns)
    cols_b = set(b.columns)
    if cols_a != cols_b:
        print(f"✗ column set differs:")
        print(f"  legacy only: {cols_a - cols_b}")
        print(f"  new only:    {cols_b - cols_a}")
        return False
    print(f"✓ {len(cols_a)} columns match")

    # Sort both by a stable key to make row-wise comparison meaningful
    sort_keys = [c for c in ("location", "utc_time", "latitude", "longitude")
                 if c in a.columns]
    if sort_keys:
        a = a.sort_values(sort_keys).reset_index(drop=True)
        b = b.sort_values(sort_keys).reset_index(drop=True)

    ok = True
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            diff = np.abs(a[col].fillna(0).values - b[col].fillna(0).values)
            max_d = diff.max() if len(diff) else 0.0
            nan_a = a[col].isna().sum()
            nan_b = b[col].isna().sum()
            if max_d > atol or nan_a != nan_b:
                print(f"✗ {col}: max_abs_diff={max_d:.2e}, nan_legacy={nan_a}, nan_new={nan_b}")
                ok = False
            else:
                print(f"✓ {col}: max_abs_diff={max_d:.2e}")
        else:
            mism = (a[col].astype(str) != b[col].astype(str)).sum()
            if mism > 0:
                print(f"✗ {col}: {mism} string mismatches")
                ok = False
            else:
                print(f"✓ {col}: identical")

    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", required=True, help="Original CSV")
    parser.add_argument("--new", required=True, help="Refactored CSV")
    parser.add_argument("--atol", type=float, default=1e-9,
                        help="Absolute tolerance for numeric diffs")
    args = parser.parse_args()

    ok = compare(args.legacy, args.new, atol=args.atol)
    print("\n" + ("ALL CHECKS PASSED ✅" if ok else "MISMATCHES FOUND ❌"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
