import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_and_save_dataframes(run_id: str, data_type: str = "annual"):
    """
    Merge the reference TROPOMI CSV with the run‑specific plume labels,
    then cache the merged copy.
    """
    if data_type == "annual":
        ref = "/net/fs06/d3/rzhuang/TROPOMI_world/data/Run_1/processed_valid_tropomi_emissions_with_qa_with_all_vars.csv"
        outfile = "updated_tropomi_emissions_full_variables.csv"
        include_radius = True
    else:                                        # hourly
        ref = "/net/fs06/d3/rzhuang/TROPOMI_world/data/Run_1/processed_valid_tropomi_hourly_emissions_with_qa_with_all_vars.csv"
        outfile = "updated_tropomi_hourly_emissions_full_variables.csv"
        include_radius = True

    run_dir = f"../data/{run_id}"
    merged_path = os.path.join(run_dir, outfile)

    if os.path.exists(merged_path):
        logger.info(f"[{run_id}]  {data_type} already merged.")
        return merged_path, pd.read_csv(merged_path)

    orig = pd.read_csv(ref)
    new  = pd.read_csv(os.path.join(run_dir, "valid_tropomi_emissions_with_qa.csv"))

    for df in (orig, new):
        df["match_key"] = (
            df["location"].astype(str) + "_" +
            df["latitude"].astype(str) + "_" +
            df["longitude"].astype(str) + "_" +
            df["utc_time"].astype(str)
        )

    lookup = {}
    for _, r in new.iterrows():
        d = {"plume_label": r["plume_label"]}
        if include_radius and "no2_mean_radius" in r:
            d["no2_mean_radius"] = r["no2_mean_radius"]
        if include_radius and "no2_std_radius" in r:
            d["no2_std_radius"] = r["no2_std_radius"]
        if include_radius and "no2_frac_valid_radius" in r:
            d["no2_frac_valid_radius"] = r["no2_frac_valid_radius"]
        lookup[r["match_key"]] = d

    cnt = 0
    for i, r in orig.iterrows():
        mk = r["match_key"]
        if mk in lookup:
            cnt += 1
            orig.at[i, "plume_label"] = lookup[mk]["plume_label"]
            if include_radius and "no2_std_radius" in lookup[mk]:
                orig.at[i, "no2_std_radius"] = lookup[mk]["no2_std_radius"]
            if include_radius and "no2_mean_radius" in lookup[mk]:
                orig.at[i, "no2_mean_radius"] = lookup[mk]["no2_mean_radius"]
            if include_radius and "no2_frac_valid_radius" in lookup[mk]:
                orig.at[i, "no2_frac_valid_radius"] = lookup[mk]["no2_frac_valid_radius"]

    logger.info(f"[{run_id}]  {data_type}: updated {cnt}/{len(orig)} rows")
    orig.drop(columns=["no2_var_50km", "match_key"], errors="ignore", inplace=True)
    orig.dropna(inplace=True)
    orig.to_csv(merged_path, index=False)
    return merged_path, orig

# Execute the function
if __name__ == "__main__":
    # Process multiple runs
    run_ids = ["Run_4"]  # Add your actual run IDs
    
    results = {}
    
    for run_id in run_ids:
        # Process annual data
        annual_path, annual_df = merge_and_save_dataframes(run_id, "annual")
        
        # # Process hourly data
        # hourly_path, hourly_df = merge_and_save_dataframes(run_id, "hourly")
        
        results[run_id] = {
            "annual": {"path": annual_path, "rows": len(annual_df)},
            # "hourly": {"path": hourly_path, "rows": len(hourly_df)}
        }
    
    # Summary
    for run_id, data in results.items():
        print(f"{run_id}: Annual={data['annual']['rows']} rows")
        # print(f"{run_id}: Hourly={data['hourly']['rows']} rows")