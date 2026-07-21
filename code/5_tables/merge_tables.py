"""
Unified merge: combine reference TROPOMI table with run-specific plume labels.

Replaces world/5_merge.py and adds region support so US can use the same flow.
Merges on the composite key (location, latitude, longitude, utc_time) and
copies plume_label + radius statistics from the new run into the reference.

Usage:
    python merge_tables.py --region world --run-id Run_4 --data-type annual
    python merge_tables.py --region us    --run-id Run_20250623_203825 --data-type hourly

Behavior intended to be byte-equivalent to the original world/5_merge.py.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from config import get_config  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_match_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["location"].astype(str) + "_"
        + df["latitude"].astype(str) + "_"
        + df["longitude"].astype(str) + "_"
        + df["utc_time"].astype(str)
    )


def merge_run(reference_csv: str, run_csv: str, output_csv: str,
              include_radius: bool = True) -> pd.DataFrame:
    """Merge plume_label + radius columns from run_csv into reference_csv."""
    if os.path.exists(output_csv):
        logger.info(f"Already merged: {output_csv}")
        return pd.read_csv(output_csv)

    orig = pd.read_csv(reference_csv)
    new = pd.read_csv(run_csv)

    orig["match_key"] = _make_match_key(orig)
    new["match_key"] = _make_match_key(new)

    radius_cols = ["no2_mean_radius", "no2_std_radius", "no2_frac_valid_radius"]

    lookup = {}
    for _, r in new.iterrows():
        d = {"plume_label": r["plume_label"]}
        if include_radius:
            for c in radius_cols:
                if c in r:
                    d[c] = r[c]
        lookup[r["match_key"]] = d

    cnt = 0
    for i, r in orig.iterrows():
        mk = r["match_key"]
        if mk in lookup:
            cnt += 1
            orig.at[i, "plume_label"] = lookup[mk]["plume_label"]
            if include_radius:
                for c in radius_cols:
                    if c in lookup[mk]:
                        orig.at[i, c] = lookup[mk][c]

    logger.info(f"Updated {cnt}/{len(orig)} rows")
    orig.drop(columns=["no2_var_50km", "match_key"], errors="ignore", inplace=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    orig.to_csv(output_csv, index=False)
    return orig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=["us", "world"])
    parser.add_argument("--run-id", required=True,
                        help="Run directory name under data_dir/")
    parser.add_argument("--data-type", choices=["annual", "hourly"], default="annual")
    parser.add_argument("--reference-csv", default=None,
                        help="Reference TROPOMI table (default: cfg.era5_table_csv)")
    parser.add_argument("--no-radius", action="store_true",
                        help="Skip radius columns when merging")
    args = parser.parse_args()

    cfg = get_config(args.region)
    run_dir = os.path.join(cfg.data_dir, args.run_id)

    outfile = (
        "updated_tropomi_emissions_full_variables.csv"
        if args.data_type == "annual"
        else "updated_tropomi_hourly_emissions_full_variables.csv"
    )

    reference_csv = args.reference_csv or cfg.era5_table_csv
    run_csv = os.path.join(run_dir, "valid_tropomi_emissions_with_qa.csv")
    output_csv = os.path.join(run_dir, outfile)

    logger.info(f"Reference: {reference_csv}")
    logger.info(f"Run csv:   {run_csv}")
    logger.info(f"Output:    {output_csv}")

    df = merge_run(reference_csv, run_csv, output_csv,
                   include_radius=not args.no_radius)
    print(f"{args.run_id} ({args.data_type}): {len(df)} rows → {output_csv}")


if __name__ == "__main__":
    main()
