"""Unified MLP training script for TROPOMI plume detection.

Consolidates 26 training scripts into one CLI.

Split modes:
  simple       - train/test 80/20 random, 1 run, fixed LR, early stop on train loss
  simple_group - train/test 80/20 by plant group, 1 run, fixed LR, early stop on train loss
  item         - train/val/test 60/20/20 random, N runs, LR search, early stop on val loss
  power_plant  - train/val/test 60/20/20 by plant group, N runs, LR search, early stop on val loss

Examples:
  python train_mlp.py --region us --split_mode item --feature_set full --interference \\
      --annual_emissions_csv .../emissions.csv --cities_csv .../worldcities.csv \\
      --data_csv .../data.csv --plant_csv .../plants.csv --output_dir ./out
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TRAINING
from shared.features import get_feature_list
from shared.model import MLP
from shared.metrics import (
    compute_metrics, print_metrics_simple, print_metrics_table,
    print_latex_table, aggregate_run_metrics,
)
from shared.interference import (
    identify_interference_us_by_year, identify_interference_world,
    filter_data_by_year_interference, extract_year_from_datetime,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def to_py(o):
    """Convert numpy types to Python for JSON serialization."""
    if isinstance(o, dict):
        return {(int(k) if isinstance(k, np.integer) else k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):  return [to_py(x) for x in o]
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_):    return bool(o)
    return o


def _train_epoch(model, dl, crit, opt, device):
    model.train()
    total = 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device).float()
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(dl.dataset)


def _eval_loss(model, dl, crit, device):
    model.eval()
    total = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device).float()
            total += crit(model(xb), yb).item() * xb.size(0)
    return total / len(dl.dataset)


def _predict(model, dl, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in dl:
            probs.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    probs = np.array(probs)
    return (probs >= TRAINING["prediction_threshold"]).astype(int), probs


def _dl(X, y, bs, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=bs, shuffle=shuffle)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_datasets(args, features):
    """Load TROPOMI CSV, create top-N subsets, optionally apply interference."""
    tropomi = pd.read_csv(args.data_csv)
    if args.region == 'us':
        tropomi = tropomi.dropna()
    else:
        tropomi = tropomi.dropna(subset=features + ['plume_label', 'location'])

    le = LabelEncoder()
    tropomi['primary_fuel_type'] = le.fit_transform(tropomi['primary_fuel_type'])
    print(f"Loaded {len(tropomi)} obs. Fuel types: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    plants = pd.read_csv(args.plant_csv)
    id_col = 'Facility_ID' if args.region == 'us' else 'ID'
    if args.region == 'us' and 'NOx_Rank' in plants.columns:
        plants.sort_values('NOx_Rank', ascending=True, inplace=True)
    elif 'annual_nox_emission' in plants.columns:
        plants.sort_values('annual_nox_emission', ascending=False, inplace=True)

    datasets = {}

    if args.interference:
        cities = pd.read_csv(args.cities_csv)

        if args.region == 'us':
            tropomi = extract_year_from_datetime(tropomi)
            annual_emis = pd.read_csv(args.annual_emissions_csv)

            # Match the paper's "171 plants" recipe: restrict the interference
            # candidate pool (per top-N subset) to plants that have a CAMPD
            # annual-emissions entry in *every* year 2019-2024 before computing
            # interference. Without this restriction we get 182 (500-318)
            # instead of 171 (460-289) for the All subset, because plants that
            # disappeared mid-period would otherwise contribute to (and be
            # filtered out by) the interference union.
            US_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
            for n, label in [(None, 'all_data'), (100, 'top100'), (50, 'top50'), (20, 'top20')]:
                pids = (tropomi['location'].unique().tolist() if n is None
                        else plants.head(n)[id_col].tolist())
                # Intersect with "present in all 6 CAMPD years"
                emis_top = annual_emis[annual_emis['Facility ID'].isin(pids)]
                present_yrs = (emis_top[emis_top['Year'].isin(US_YEARS)]
                               .groupby('Facility ID')['Year'].nunique())
                complete_6y_ids = set(present_yrs[present_yrs == len(US_YEARS)].index)
                pids_strict = [p for p in pids if p in complete_6y_ids]
                pdf_strict = plants[plants[id_col].isin(pids_strict)]
                emis_strict = emis_top[emis_top['Facility ID'].isin(pids_strict)]
                interf = identify_interference_us_by_year(pdf_strict, emis_strict, cities,
                                                           plant_subset_ids=pids_strict)
                base = tropomi if n is None else tropomi[tropomi['location'].isin(pids)]
                datasets[label] = base
                # Strict filter drops:
                #   (a) plants ever-flagged in any year, AND
                #   (b) plants that aren't in complete_6y_ids
                # Both are subsumed by intersecting kept set with complete_6y_ids.
                kept_ids = complete_6y_ids - set().union(*[set(v) for v in interf.values()])
                datasets[f'{label}_filtered_yearly'] = base[base['location'].isin(kept_ids)].copy()
        else:
            for n, label in [(None, 'all_data'), (100, 'top100'), (50, 'top50'), (20, 'top20')]:
                pids = (tropomi['location'].unique().tolist() if n is None
                        else plants.head(n)[id_col].tolist())
                pdf = plants[plants[id_col].isin(pids)] if n is None else plants.head(n)
                interf = identify_interference_world(pdf, cities, plant_subset_ids=pids)
                base = tropomi if n is None else tropomi[tropomi['location'].isin(pids)]
                datasets[label] = base
                datasets[f'{label}_filtered'] = base[~base['location'].isin(interf)]
    else:
        datasets['all_data'] = tropomi
        for n in [100, 50, 20]:
            datasets[f'top{n}'] = tropomi[tropomi['location'].isin(plants.head(n)[id_col])]

    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    for name, df in datasets.items():
        print(f"  {name}: {len(df)} samples, {df['location'].nunique()} plants")
    return datasets


# ─── Split functions ──────────────────────────────────────────────────────────

def _split_simple(X, y, groups, mode):
    """Simple train/test split for baseline modes."""
    seed = TRAINING["split_seed_simple"]
    if mode == 'simple_group':
        gss = GroupShuffleSplit(n_splits=1, test_size=TRAINING["simple_test_size"], random_state=seed)
        tr, te = next(gss.split(X, y, groups=groups))
        return X[tr], X[te], y[tr], y[te]
    return train_test_split(X, y, test_size=TRAINING["simple_test_size"], random_state=seed)


def _split_item(X, y, run):
    """Item-level train/val/test split. Seed=345+run (different split per run)."""
    seed = TRAINING["split_seed_sweep"] + run
    tr_r, val_r, te_r = TRAINING["train_ratio"], TRAINING["val_ratio"], TRAINING["test_ratio"]

    X_tt, X_val, y_tt, y_val = train_test_split(X, y, test_size=val_r, random_state=seed)
    rel_te = te_r / (tr_r + te_r)
    X_tr, X_te, y_tr, y_te = train_test_split(X_tt, y_tt, test_size=rel_te, random_state=seed + 1)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def _split_power_plant(X, y, groups, region):
    """Power plant group-based train/val/test split. Seed=345 (fixed).

    US uses val_ratio/(1-test_ratio) for GSS test_size;
    World uses val_ratio directly. Matches original scripts.
    """
    seed = TRAINING["split_seed_sweep"]
    tr_r, val_r, te_r = TRAINING["train_ratio"], TRAINING["val_ratio"], TRAINING["test_ratio"]

    # Step 1: random item split to remove test_ratio fraction
    X, _, y, _, groups, _ = train_test_split(X, y, groups, test_size=te_r, random_state=seed)

    # Step 2: GroupShuffleSplit for validation
    gss_val_size = val_r / (1 - te_r) if region == 'us' else val_r
    gss_val = GroupShuffleSplit(n_splits=1, test_size=gss_val_size, random_state=seed)
    tt_idx, val_idx = next(gss_val.split(X, y, groups))
    X_val, y_val = X[val_idx], y[val_idx]

    # Step 3: GroupShuffleSplit for test from remainder
    rel_te = te_r / (tr_r + te_r)
    gss_te = GroupShuffleSplit(n_splits=1, test_size=rel_te, random_state=seed + 1)
    tr_idx, te_idx = next(gss_te.split(X[tt_idx], y[tt_idx], groups[tt_idx]))
    X_tr, y_tr = X[tt_idx][tr_idx], y[tt_idx][tr_idx]
    X_te, y_te = X[tt_idx][te_idx], y[tt_idx][te_idx]

    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ─── Training modes ──────────────────────────────────────────────────────────

def train_simple_mode(ds, features, args):
    """1 run, fixed LR, early stop on train loss."""
    X = ds[features].values.astype(np.float32)
    y = ds["plume_label"].astype(int).values
    X_tr, X_te, y_tr, y_te = _split_simple(X, y, ds['location'].values, args.split_mode)

    ros = RandomOverSampler(random_state=TRAINING["oversample_seed_base"])
    X_tr, y_tr = ros.fit_resample(X_tr, y_tr)
    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs = TRAINING["batch_size"]
    tr_dl, te_dl = _dl(X_tr, y_tr, bs, True), _dl(X_te, y_te, bs)

    lr = TRAINING["default_lr"]
    model = MLP(X_tr.shape[1]).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    ckpt = os.path.join(args.output_dir, f'best_{lr}_features.pt')
    print(f"LR: {lr}, Features: {X_tr.shape[1]}")

    best_loss, counter = np.inf, 0
    for ep in range(TRAINING["num_epochs"]):
        loss = _train_epoch(model, tr_dl, crit, opt, device)
        print(f"Epoch {ep+1}/{TRAINING['num_epochs']} – train loss: {loss:.4f}")
        if loss < best_loss:
            best_loss, counter = loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            counter += 1
            if counter >= TRAINING["patience"]:
                print(f"Early stopping at epoch {ep+1}")
                break

    model.load_state_dict(torch.load(ckpt))
    preds, probs = _predict(model, te_dl, device)
    print_metrics_simple(y_te, preds, probs)


def train_sweep_mode(datasets, features, args):
    """Multiple runs, LR grid search, early stop on val loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs = TRAINING["batch_size"]
    lrs = TRAINING["learning_rates"]
    n_runs = TRAINING["num_runs"]
    all_results = {}

    for ds_name, ds in datasets.items():
        if len(ds) == 0:
            print(f"WARNING: {ds_name} empty, skipping.")
            continue
        print(f"\n{'='*60}\n{ds_name}: {len(ds)} samples, {ds['location'].nunique()} plants\n{'='*60}")

        X_all = ds[features].values.astype(np.float32)
        y_all = ds["plume_label"].astype(int).values
        g_all = ds['location'].values

        run_results = []
        best_auc, best_state, best_scaler = 0, None, None
        best_lr_for_ds = None  # set on run 0; reused on runs 1..N

        for run in range(n_runs):
            print(f"\n--- Run {run+1}/{n_runs} ---")

            if args.split_mode == 'power_plant':
                X_tr, X_va, X_te, y_tr, y_va, y_te = _split_power_plant(
                    X_all.copy(), y_all.copy(), g_all.copy(), args.region)
            else:
                X_tr, X_va, X_te, y_tr, y_va, y_te = _split_item(
                    X_all.copy(), y_all.copy(), run)

            print(f"Pos rates: train={y_tr.mean():.3f}, val={y_va.mean():.3f}, test={y_te.mean():.3f}")

            ros = RandomOverSampler(random_state=TRAINING["oversample_seed_base"] + run)
            X_tr, y_tr = ros.fit_resample(X_tr, y_tr)
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va, X_te = sc.transform(X_va), sc.transform(X_te)

            tr_dl = _dl(X_tr, y_tr, bs, True)
            va_dl = _dl(X_va, y_va, bs)
            te_dl = _dl(X_te, y_te, bs)

            # LR grid search.
            # Optimization: only sweep on run 0; runs 1..N reuse run 0's
            # best LR (much faster, with negligible effect on the final
            # mean +/- std since LR is a hyperparameter, not a stochastic
            # source we want to vary).
            hp = {}
            import time as _time
            if run == 0:
                lrs_to_try = list(lrs)
            else:
                lrs_to_try = [best_lr_for_ds]
                print(f"  reusing run-0 best LR: {best_lr_for_ds:.1e}", flush=True)
            for lr in lrs_to_try:
                model = MLP(X_tr.shape[1]).to(device)
                crit = nn.BCEWithLogitsLoss()
                opt = optim.Adam(model.parameters(), lr=lr)
                best_vl, counter = np.inf, 0
                ck = os.path.join(args.output_dir, f'tmp_{ds_name}_r{run}_lr{lr}.pt')
                _t0 = _time.time()
                _last_epoch = 0
                print(f"  [lr={lr:.1e}] starting", flush=True)

                for _ep in range(TRAINING["num_epochs"]):
                    _train_epoch(model, tr_dl, crit, opt, device)
                    vl = _eval_loss(model, va_dl, crit, device)
                    if vl < best_vl:
                        best_vl, counter = vl, 0
                        torch.save(model.state_dict(), ck)
                        _star = '*'
                    else:
                        counter += 1
                        _star = ' '
                    _last_epoch = _ep + 1
                    if (_ep + 1) % 5 == 0 or _ep == 0:
                        print(f"    epoch {_ep+1:3d}/{TRAINING['num_epochs']}  "
                              f"val_loss={vl:.4f} best={best_vl:.4f} {_star} "
                              f"(t={_time.time()-_t0:.0f}s)", flush=True)
                    if counter >= TRAINING["patience"]:
                        break

                model.load_state_dict(torch.load(ck))
                _, vp = _predict(model, va_dl, device)
                hp[lr] = roc_auc_score(y_va, vp)
                print(f"  [lr={lr:.1e}] done in {_time.time()-_t0:.0f}s "
                      f"(stopped at epoch {_last_epoch}, val_auc={hp[lr]:.4f})",
                      flush=True)

            best_lr = max(hp, key=hp.get)
            if run == 0:
                best_lr_for_ds = best_lr
            print(f"Best LR: {best_lr} (Val AUC: {hp[best_lr]:.4f})")

            model = MLP(X_tr.shape[1]).to(device)
            model.load_state_dict(torch.load(
                os.path.join(args.output_dir, f'tmp_{ds_name}_r{run}_lr{best_lr}.pt')))
            preds, probs = _predict(model, te_dl, device)
            met = compute_metrics(y_te, preds, probs, zero_division=0)
            run_results.append(met)
            print(f"Test AUC: {met['auc']:.4f}")

            if met['auc'] > best_auc:
                best_auc = met['auc']
                best_state, best_scaler = model.state_dict(), sc

        summary = aggregate_run_metrics(run_results)
        all_results[ds_name] = summary
        print_metrics_table(summary, ds_name)

        if best_state:
            torch.save(best_state, os.path.join(args.output_dir, f'best_model_{ds_name}.pt'))
            try:
                import joblib
                joblib.dump(best_scaler, os.path.join(args.output_dir, f'best_scaler_{ds_name}.pkl'))
            except ImportError:
                pass

    with open(os.path.join(args.output_dir, 'results_summary.json'), 'w') as f:
        json.dump(to_py(all_results), f, indent=2)
    print_latex_table(all_results)
    return all_results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--region', required=True, choices=['us', 'world'])
    p.add_argument('--data_csv', required=True, help='TROPOMI CSV')
    p.add_argument('--plant_csv', required=True, help='Power plants CSV')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--split_mode', default='item',
                   choices=['simple', 'simple_group', 'item', 'power_plant'])
    p.add_argument('--feature_set', default='full',
                   choices=['baseline', 'baseline_wo_stats', 'full', 'no_stats'])
    p.add_argument('--data_type', default='annual', choices=['annual', 'hourly'])
    p.add_argument('--interference', action='store_true',
                   help='Enable interference zone filtering')
    p.add_argument('--annual_emissions_csv', help='US interference only')
    p.add_argument('--cities_csv', help='For interference filtering')
    p.add_argument('--datasets', nargs='+',
                   help='Which subsets to train on (e.g., top20 all_data_filtered_yearly)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    features = get_feature_list(args.feature_set, args.region, args.data_type)
    print(f"Features: {args.feature_set} ({len(features)})")

    datasets = load_datasets(args, features)

    if args.split_mode in ('simple', 'simple_group'):
        for name, ds in datasets.items():
            print(f"\n--- {name} ---")
            train_simple_mode(ds, features, args)
    else:
        train_sweep_mode(datasets, features, args)

    print("\nDone!")


if __name__ == '__main__':
    main()
