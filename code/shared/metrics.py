"""Shared evaluation metrics for TROPOMI MLP models.

Extracted from the metric computation code duplicated across 30+ training
and analysis scripts. Preserves exact behavior including:
- World scripts use zero_division=0 in precision_score
- Simple scripts print individual metrics
- Full sweep scripts aggregate mean/std across runs
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred, y_prob=None, zero_division=0):
    """Compute all classification metrics.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels (0/1)
        y_prob: predicted probabilities (for AUC)
        zero_division: value for precision/recall when no positive predictions
            (World scripts use 0; US simple baseline uses default 'warn')

    Returns:
        dict with accuracy, precision, recall, f1, kappa, and optionally auc
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=zero_division),
        'recall': recall_score(y_true, y_pred, zero_division=zero_division),
        'f1': f1_score(y_true, y_pred, zero_division=zero_division),
        'kappa': cohen_kappa_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = float('nan')
    return metrics


def print_metrics_simple(y_true, y_pred, y_prob=None):
    """Print metrics in the format used by simple baseline scripts.

    Matches the exact output format of 6_MLP_training.py (US/World baseline).
    """
    print("\n=== Model Performance ===")
    print("Accuracy:          ", accuracy_score(y_true, y_pred))
    print("Precision:         ", precision_score(y_true, y_pred))
    print("Recall:            ", recall_score(y_true, y_pred))
    print("F1 Score:          ", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    if y_prob is not None:
        print("AUC:               ", roc_auc_score(y_true, y_prob))
    print("Classification Rep:\n", classification_report(y_true, y_pred))
    print("Cohen's Kappa:     ", cohen_kappa_score(y_true, y_pred))


def print_metrics_table(results_summary, dataset_name=""):
    """Print metrics in table format used by full_sweep scripts.

    Args:
        results_summary: dict with metric -> {'mean': float, 'std': float}
        dataset_name: label for the dataset (e.g., "top20", "all_data")
    """
    if dataset_name:
        print(f"\n=== Results for {dataset_name} ===")
    print(f"{'Metric':<15} {'Mean':>10} {'Std':>10}")
    print("-" * 37)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'kappa']:
        if metric in results_summary:
            mean_val = results_summary[metric]['mean']
            std_val = results_summary[metric]['std']
            print(f"{metric.capitalize():<15} {mean_val:>10.4f} {std_val:>10.4f}")


def print_latex_table(all_results, dataset_names=None):
    """Print results in LaTeX table format used by some training scripts.

    Args:
        all_results: dict of dataset_name -> aggregated metric results
        dataset_names: optional list to control column order
    """
    if dataset_names is None:
        dataset_names = list(all_results.keys())

    metrics_order = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'kappa']
    print("\n% LaTeX Table")
    print("\\begin{tabular}{l" + "c" * len(dataset_names) + "}")
    print("\\toprule")
    print("Metric & " + " & ".join(dataset_names) + " \\\\")
    print("\\midrule")
    for metric in metrics_order:
        row = f"{metric.capitalize()}"
        for name in dataset_names:
            if name in all_results and metric in all_results[name]:
                m = all_results[name][metric]
                row += f" & {m['mean']:.4f} $\\pm$ {m['std']:.4f}"
            else:
                row += " & --"
        row += " \\\\"
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}")


def aggregate_run_metrics(all_run_metrics):
    """Compute mean and std across multiple runs.

    Args:
        all_run_metrics: list of dicts from compute_metrics()

    Returns:
        dict with metric -> {'mean': float, 'std': float, 'values': list}
    """
    if not all_run_metrics:
        return {}

    keys = all_run_metrics[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in all_run_metrics
                if k in m and not (isinstance(m[k], float) and np.isnan(m[k]))]
        result[k] = {
            'mean': float(np.mean(vals)) if vals else float('nan'),
            'std': float(np.std(vals)) if vals else float('nan'),
            'values': [float(v) for v in vals],
        }
    return result
