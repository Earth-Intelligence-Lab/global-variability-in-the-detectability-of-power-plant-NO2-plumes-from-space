"""Permutation feature importance for the latest item-split MLP models.

For each (region, dataset) the best-of-5-runs `best_model_*.pt` was selected
by max test AUC across runs that each used a different item-split seed
(seed=345+run). To make permutation importance match the saved model exactly,
we parse the training log to identify the best run per dataset, replay that
run's split, load the saved best model + scaler, and compute permutation
importance on its test fold.

Outputs (under <model_dir>/feature_importance/):
  feature_importance_permutation_<dataset>.csv
  feature_importance_permutation_top<TOPK>_<dataset>.png
  permutation_summary.csv

Usage:
  python feature_importance_permutation.py --region us
  python feature_importance_permutation.py --region world
"""

import argparse
import os
import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path('/net/fs06/d3/rzhuang/TROPOMI/code')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '6_training'))

from config import TRAINING                                              # noqa: E402
from shared.features import get_feature_list                             # noqa: E402
from shared.interference import (                                        # noqa: E402
    identify_interference_us_by_year, identify_interference_world,
    extract_year_from_datetime,
)
from shared.model import MLP                                             # noqa: E402

US_PATHS = {
    'data_csv':            '/net/fs06/d3/rzhuang/TROPOMI/pipeline_100m_run/Run_100m_20260414/'
                           'updated_tropomi_hourly_emissions_full_variables_augmented_localtz.csv',
    'plant_csv':           '/net/fs06/d3/rzhuang/TROPOMI/data/us/'
                           'facility_emissions_by_plant_comprehensive.csv',
    'annual_emissions_csv': '/net/fs06/d3/rzhuang/TROPOMI/data/us/'
                           'annual-emissions-facility-aggregation-2019-2024.csv',
    'cities_csv':          '/net/fs06/d3/rzhuang/TROPOMI/data/world/worldcities.csv',
    'model_dir':           '/net/fs06/d3/rzhuang/TROPOMI/data/us/Run_100m_20260414/'
                           'training_no_stats_item',
    'log':                 '/net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs/train_us_item.out',
    'datasets':            ['all_data_filtered_yearly', 'top100_filtered_yearly',
                            'top50_filtered_yearly', 'top20_filtered_yearly'],
    'data_type':           'hourly',
}
WORLD_PATHS = {
    'data_csv':            '/net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m/'
                           'Run_100m_20260428/updated_tropomi_emissions_full_variables_with_fuel_100mlabel.csv',
    'plant_csv':           '/net/fs06/d3/rzhuang/TROPOMI/data/world/power_plant_location/'
                           'power_plants_with_combined_nearby_stats.csv',
    'cities_csv':          '/net/fs06/d3/rzhuang/TROPOMI/data/world/worldcities.csv',
    'model_dir':           '/net/fs06/d3/rzhuang/TROPOMI/data/world/pipeline_test_labelling_100m/'
                           'Run_100m_20260428/training_no_stats_item',
    'log':                 '/net/fs06/d3/rzhuang/TROPOMI/code/slurm/training/logs/train_world_item.out',
    'datasets':            ['all_data_filtered', 'top100_filtered',
                            'top50_filtered', 'top20_filtered'],
    'data_type':           'annual',
}

N_REPEATS = 5
TOPK_PLOT = 15
BATCH_SIZE = 50_000


# ─── Log parsing ──────────────────────────────────────────────────────────────

def parse_best_runs(log_path):
    """Return {ds_name: best_run_idx (0..4)} parsed from train_mlp.py log."""
    text = Path(log_path).read_text()
    blocks = re.split(r'=+\n([a-zA-Z_0-9]+):\s*\d+\s*samples', text)
    out = {}
    for i in range(1, len(blocks), 2):
        ds = blocks[i].strip()
        body = blocks[i+1] if i+1 < len(blocks) else ''
        runs = re.findall(r'--- Run (\d)/5 ---.*?Test AUC: (\d+\.\d+)', body, re.S)
        if not runs:
            continue
        runs = [(int(r) - 1, float(a)) for r, a in runs]
        best_run, best_auc = max(runs, key=lambda x: x[1])
        out[ds] = {'best_run': best_run, 'best_auc': best_auc, 'all_runs': runs}
    return out


# ─── Dataset reconstruction (mirrors train_mlp.py load_datasets) ──────────────

def load_us_datasets(features):
    p = US_PATHS
    tropomi = pd.read_csv(p['data_csv'])
    tropomi = tropomi.dropna()

    le = LabelEncoder()
    tropomi['primary_fuel_type'] = le.fit_transform(tropomi['primary_fuel_type'])

    plants = pd.read_csv(p['plant_csv'])
    if 'NOx_Rank' in plants.columns:
        plants.sort_values('NOx_Rank', ascending=True, inplace=True)
    cities = pd.read_csv(p['cities_csv'])
    annual_emis = pd.read_csv(p['annual_emissions_csv'])
    tropomi = extract_year_from_datetime(tropomi)

    out = {}
    US_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
    for n, label in [(None, 'all_data'), (100, 'top100'), (50, 'top50'), (20, 'top20')]:
        pids = (tropomi['location'].unique().tolist() if n is None
                else plants.head(n)['Facility_ID'].tolist())
        emis_top = annual_emis[annual_emis['Facility ID'].isin(pids)]
        present_yrs = (emis_top[emis_top['Year'].isin(US_YEARS)]
                       .groupby('Facility ID')['Year'].nunique())
        complete_6y_ids = set(present_yrs[present_yrs == len(US_YEARS)].index)
        pids_strict = [p for p in pids if p in complete_6y_ids]
        pdf_strict = plants[plants['Facility_ID'].isin(pids_strict)]
        emis_strict = emis_top[emis_top['Facility ID'].isin(pids_strict)]
        interf = identify_interference_us_by_year(pdf_strict, emis_strict, cities,
                                                   plant_subset_ids=pids_strict)
        base = tropomi if n is None else tropomi[tropomi['location'].isin(pids)]
        kept_ids = complete_6y_ids - set().union(*[set(v) for v in interf.values()])
        out[f'{label}_filtered_yearly'] = base[base['location'].isin(kept_ids)].copy()
    return out


def load_world_datasets(features):
    p = WORLD_PATHS
    tropomi = pd.read_csv(p['data_csv'])
    tropomi = tropomi.dropna(subset=features + ['plume_label', 'location'])

    le = LabelEncoder()
    tropomi['primary_fuel_type'] = le.fit_transform(tropomi['primary_fuel_type'])

    plants = pd.read_csv(p['plant_csv'])
    cities = pd.read_csv(p['cities_csv'])

    out = {}
    for n, label in [(None, 'all_data'), (100, 'top100'), (50, 'top50'), (20, 'top20')]:
        pids = (tropomi['location'].unique().tolist() if n is None
                else plants.head(n)['ID'].tolist())
        pdf = plants[plants['ID'].isin(pids)] if n is None else plants.head(n)
        interf = identify_interference_world(pdf, cities, plant_subset_ids=pids)
        base = tropomi if n is None else tropomi[tropomi['location'].isin(pids)]
        out[f'{label}_filtered'] = base[~base['location'].isin(interf)].copy()
    return out


# ─── Item split (matches train_mlp.py:_split_item exactly) ────────────────────

def split_item_for_run(X, y, run):
    seed = TRAINING['split_seed_sweep'] + run                    # 345 + run
    val_r = TRAINING['val_ratio']
    tr_r, te_r = TRAINING['train_ratio'], TRAINING['test_ratio']
    X_tt, X_val, y_tt, y_val = train_test_split(X, y, test_size=val_r, random_state=seed)
    rel_te = te_r / (tr_r + te_r)
    X_tr, X_te, y_tr, y_te = train_test_split(X_tt, y_tt, test_size=rel_te, random_state=seed + 1)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ─── Inference + permutation ──────────────────────────────────────────────────

def predict_probs(model, X, device, batch=BATCH_SIZE):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device).float()
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, p))


def permutation_importance(model, X_test, y_test, features, device,
                            n_repeats=N_REPEATS, seed=0):
    rng = np.random.default_rng(seed)
    base = safe_auc(y_test, predict_probs(model, X_test, device))
    importances = np.zeros(len(features), dtype=np.float32)
    for i, fname in enumerate(features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, i])
            drops.append(base - safe_auc(y_test, predict_probs(model, X_perm, device)))
        importances[i] = float(np.mean(drops))
        print(f'    [{i+1:>2}/{len(features)}] {fname:<45s} drop={importances[i]:+.4f}', flush=True)
    return base, importances


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(region: str):
    paths = US_PATHS if region == 'us' else WORLD_PATHS
    out_dir = Path(paths['model_dir']) / 'feature_importance'
    out_dir.mkdir(parents=True, exist_ok=True)

    features = get_feature_list('no_stats', region, paths['data_type'])
    print(f'Region={region}, features ({len(features)}):')
    for f in features:
        print(f'  - {f}')

    print('\nParsing log to identify best run per dataset ...')
    best_runs = parse_best_runs(paths['log'])
    for ds, info in best_runs.items():
        all_str = ' '.join(f'r{r}={a:.4f}' for r, a in info['all_runs'])
        print(f'  {ds:<28s}  best_run={info["best_run"]} (auc={info["best_auc"]:.4f})  | {all_str}')

    print('\nReconstructing datasets ...')
    datasets = (load_us_datasets(features) if region == 'us'
                else load_world_datasets(features))
    for n, d in datasets.items():
        print(f'  {n}: {len(d):,} rows')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}\n')

    summary_rows = []
    for ds_name in paths['datasets']:
        print(f'\n{"="*72}\n=== {ds_name} ===\n{"="*72}')
        if ds_name not in datasets:
            print(f'  skip: not loaded'); continue
        if ds_name not in best_runs:
            print(f'  skip: best run not in log'); continue

        best_run = best_runs[ds_name]['best_run']
        log_auc  = best_runs[ds_name]['best_auc']
        print(f'  Best run from log: r{best_run} (Test AUC = {log_auc:.4f})')

        ds = datasets[ds_name]
        X = ds[features].values.astype(np.float32)
        y = ds['plume_label'].astype(int).values

        # Replay run-best split
        _, _, X_te, _, _, y_te = split_item_for_run(X, y, best_run)
        print(f'  test rows: {len(X_te):,} | pos = {y_te.sum()} ({y_te.mean()*100:.1f}%)')

        scaler_path = Path(paths['model_dir']) / f'best_scaler_{ds_name}.pkl'
        model_path  = Path(paths['model_dir']) / f'best_model_{ds_name}.pt'
        if not scaler_path.exists() or not model_path.exists():
            print(f'  skip: missing {scaler_path.name} or {model_path.name}'); continue
        scaler = joblib.load(scaler_path)
        X_te_s = scaler.transform(X_te).astype(np.float32)

        model = MLP(input_dim=len(features)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        probs = predict_probs(model, X_te_s, device)
        base_auc = safe_auc(y_te, probs)
        pr_auc   = float(average_precision_score(y_te, probs))
        print(f'  baseline test ROC-AUC = {base_auc:.4f} | PR-AUC = {pr_auc:.4f}  '
              f'(log reported {log_auc:.4f})')

        print(f'\n  Computing permutation importance (N_REPEATS={N_REPEATS}) ...')
        _, imps = permutation_importance(model, X_te_s, y_te, features, device, n_repeats=N_REPEATS)

        df = (pd.DataFrame({'feature': features, 'importance_auc_drop': imps})
                .sort_values('importance_auc_drop', ascending=False).reset_index(drop=True))
        csv_path = out_dir / f'feature_importance_permutation_{ds_name}.csv'
        df.to_csv(csv_path, index=False)
        print(f'\n  Saved CSV: {csv_path}')

        top = df.head(TOPK_PLOT).iloc[::-1]
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top)), top['importance_auc_drop'], color='#4ECDC4')
        plt.yticks(range(len(top)), top['feature'])
        plt.xlabel('Permutation Importance (AUC drop)')
        plt.title(f'Top {TOPK_PLOT} Features — {region.upper()} {ds_name}\n'
                  f'(best run r{best_run}, baseline AUC={base_auc:.4f})')
        plt.tight_layout()
        png_path = out_dir / f'feature_importance_permutation_top{TOPK_PLOT}_{ds_name}.png'
        plt.savefig(png_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'  Saved PNG: {png_path}')

        summary_rows.append({
            'dataset': ds_name, 'best_run': best_run,
            'log_test_auc': log_auc, 'replayed_test_auc': base_auc,
            'pr_auc': pr_auc, 'n_test': len(X_te),
            'top_feature': df.iloc[0]['feature'],
            'top_importance': float(df.iloc[0]['importance_auc_drop']),
        })

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(out_dir / 'permutation_summary.csv', index=False)
        print(f'\n{"="*72}\nSummary (saved to permutation_summary.csv):')
        print(sdf.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', required=True, choices=['us', 'world'])
    args = ap.parse_args()
    run(args.region)
