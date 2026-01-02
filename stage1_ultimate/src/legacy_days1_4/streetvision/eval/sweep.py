"""
Threshold sweep utilities (Day 6)

Features:
- Threshold sweep CSV export
- Sweep curve data generation
- ROC/PR curve data (optional)

2025-12-30 best practices:
- CSV format for easy plotting
- Atomic writes
- Type-safe
"""

import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch

from streetvision.eval.metrics import compute_mcc, compute_all_metrics


def compute_threshold_sweep(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int = 100,
) -> List[Dict[str, float]]:
    """
    Compute metrics at different confidence thresholds

    Args:
        logits: Model logits (N, 2)
        labels: Ground truth labels (N,)
        n_thresholds: Number of thresholds to sweep

    Returns:
        List of dictionaries with threshold and metrics:
        [
            {"threshold": 0.0, "mcc": ..., "accuracy": ..., ...},
            {"threshold": 0.01, "mcc": ..., "accuracy": ..., ...},
            ...
        ]
    """
    # Convert to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    # Compute probabilities
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pos_probs = probs[:, 1]  # Probability of positive class

    # Generate thresholds
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    # Sweep results
    sweep_results = []

    for threshold in thresholds:
        # Apply threshold
        preds = (pos_probs >= threshold).astype(int)

        # Compute metrics
        metrics = compute_all_metrics(labels, preds)

        # Add threshold to result
        result = {"threshold": float(threshold)}
        result.update(metrics)

        sweep_results.append(result)

    return sweep_results


def export_threshold_sweep_csv(
    logits: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    n_thresholds: int = 100,
) -> None:
    """
    Export threshold sweep results to CSV file

    Args:
        logits: Model logits
        labels: Ground truth labels
        output_path: Path to save threshold_sweep.csv
        n_thresholds: Number of thresholds

    CSV Format:
        threshold,mcc,accuracy,precision,recall,f1,fnr,fpr,tp,tn,fp,fn
        0.00,0.123,0.456,...
        0.01,0.124,0.457,...
        ...
    """
    # Compute sweep
    sweep_results = compute_threshold_sweep(logits, labels, n_thresholds)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV (atomic write using temp file)
    temp_path = output_path.with_suffix(".csv.tmp")

    with open(temp_path, "w", newline="") as f:
        if len(sweep_results) == 0:
            return

        # Get field names from first result
        fieldnames = list(sweep_results[0].keys())

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_results)

    # Atomic replace
    import os
    os.replace(temp_path, output_path)


def find_optimal_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "mcc",
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Find optimal threshold that maximizes a given metric

    Args:
        logits: Model logits
        labels: Ground truth labels
        metric: Metric to optimize ("mcc", "f1", "accuracy", etc.)
        n_thresholds: Number of thresholds to try

    Returns:
        (optimal_threshold, metric_value)
    """
    # Compute sweep
    sweep_results = compute_threshold_sweep(logits, labels, n_thresholds)

    # Find best threshold
    best_threshold = 0.5
    best_value = 0.0

    for result in sweep_results:
        if result[metric] > best_value:
            best_value = result[metric]
            best_threshold = result["threshold"]

    return best_threshold, best_value


def compute_roc_curve_data(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_points: int = 100,
) -> Dict[str, List[float]]:
    """
    Compute ROC curve data (TPR vs FPR)

    Args:
        logits: Model logits
        labels: Ground truth labels
        n_points: Number of points on curve

    Returns:
        Dictionary with ROC curve data:
        {
            "fpr": [0.0, 0.01, ...],
            "tpr": [0.0, 0.02, ...],
            "thresholds": [1.0, 0.99, ...],
        }
    """
    # Compute sweep
    sweep_results = compute_threshold_sweep(logits, labels, n_points)

    # Extract TPR and FPR
    fpr_values = [result["fpr"] for result in sweep_results]
    tpr_values = [result["recall"] for result in sweep_results]  # TPR = Recall
    thresholds = [result["threshold"] for result in sweep_results]

    return {
        "fpr": fpr_values,
        "tpr": tpr_values,
        "thresholds": thresholds,
    }


def compute_pr_curve_data(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_points: int = 100,
) -> Dict[str, List[float]]:
    """
    Compute Precision-Recall curve data

    Args:
        logits: Model logits
        labels: Ground truth labels
        n_points: Number of points on curve

    Returns:
        Dictionary with PR curve data:
        {
            "recall": [0.0, 0.01, ...],
            "precision": [1.0, 0.99, ...],
            "thresholds": [1.0, 0.99, ...],
        }
    """
    # Compute sweep
    sweep_results = compute_threshold_sweep(logits, labels, n_points)

    # Extract precision and recall
    recall_values = [result["recall"] for result in sweep_results]
    precision_values = [result["precision"] for result in sweep_results]
    thresholds = [result["threshold"] for result in sweep_results]

    return {
        "recall": recall_values,
        "precision": precision_values,
        "thresholds": thresholds,
    }


__all__ = [
    "compute_threshold_sweep",
    "export_threshold_sweep_csv",
    "find_optimal_threshold",
    "compute_roc_curve_data",
    "compute_pr_curve_data",
]
