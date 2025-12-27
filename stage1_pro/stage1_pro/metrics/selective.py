import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .bootstrap import BootstrapECE, BootstrapConfig


def compute_selective_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    gate_probs: Optional[np.ndarray] = None,
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """
    Compute selective classification metrics with bootstrap CIs.

    Modern 2025 implementation with comprehensive metrics.

    Args:
        probs: (N, 2) class probabilities
        labels: (N,) ground truth
        gate_probs: (N,) gate probabilities (Phase 2+)
        thresholds: List of thresholds to evaluate

    Returns:
        Dictionary of metrics with confidence intervals
    """
    results = {}

    # Basic metrics
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    results["accuracy"] = float((preds == labels).mean())

    # ECE with bootstrap
    bootstrap = BootstrapECE(BootstrapConfig(n_bootstrap=50))
    ece_results = bootstrap.compute(probs, labels)
    results.update({f"ece_{k}": v for k, v in ece_results.items()})

    # Per-class accuracy
    for class_idx in range(probs.shape[1]):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_acc = (preds[mask] == labels[mask]).mean()
            results[f"acc_class_{class_idx}"] = float(class_acc)

    # Threshold sweep
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.95, 10)

    threshold_results = []
    for t in thresholds:
        exit_mask = max_probs >= t

        if exit_mask.sum() == 0:
            threshold_results.append(
                {
                    "threshold": float(t),
                    "coverage": 0.0,
                    "exit_accuracy": 0.0,
                    "fnr_on_exits": 1.0,
                    "fpr_on_exits": 0.0,
                }
            )
            continue

        exit_labels = labels[exit_mask]
        exit_preds = preds[exit_mask]

        exit_acc = float((exit_preds == exit_labels).mean())
        coverage = float(exit_mask.mean())

        # FNR on exits
        positive_mask = exit_labels == 1
        if positive_mask.sum() > 0:
            fnr = float(
                ((exit_labels == 1) & (exit_preds == 0)).sum() / positive_mask.sum()
            )
        else:
            fnr = 1.0

        # FPR on exits
        negative_mask = exit_labels == 0
        if negative_mask.sum() > 0:
            fpr = float(
                ((exit_labels == 0) & (exit_preds == 1)).sum() / negative_mask.sum()
            )
        else:
            fpr = 0.0

        threshold_results.append(
            {
                "threshold": float(t),
                "coverage": coverage,
                "exit_accuracy": exit_acc,
                "fnr_on_exits": fnr,
                "fpr_on_exits": fpr,
            }
        )

    results["threshold_sweep"] = threshold_results

    # Find optimal threshold for target FNR
    target_fnr = 0.02
    valid = [t for t in threshold_results if t["fnr_on_exits"] <= target_fnr]

    if valid:
        best = max(valid, key=lambda x: x["coverage"])
        results["optimal_threshold"] = best["threshold"]
        results["best_coverage"] = best["coverage"]
        results["best_exit_accuracy"] = best["exit_accuracy"]
        results["best_fnr_on_exits"] = best["fnr_on_exits"]
    else:
        results["optimal_threshold"] = None

    # Gate metrics
    if gate_probs is not None:
        gate_mask = gate_probs >= 0.5
        if gate_mask.sum() > 0:
            gate_labels = labels[gate_mask]
            gate_preds = preds[gate_mask]
            results["gate_coverage"] = float(gate_mask.mean())
            results["gate_accuracy"] = float((gate_preds == gate_labels).mean())
        else:
            results["gate_coverage"] = 0.0
            results["gate_accuracy"] = 0.0

    return results


def compute_auroc(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute AUROC (Area Under ROC Curve)."""
    from sklearn.metrics import roc_auc_score

    # Use roadwork class probability
    auroc = roc_auc_score(labels, probs[:, 1])
    return float(auroc)


def compute_precision_recall(
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute precision-recall metrics."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    preds = np.argmax(probs, axis=1)

    pr_curve = precision_recall_curve(labels, probs[:, 1])
    avg_precision = average_precision_score(labels, probs[:, 1])

    return {
        "average_precision": float(avg_precision),
        "precision_at_50": float(
            (preds[probs[:, 1] >= 0.5] == labels[probs[:, 1] >= 0.5]).mean()
        )
        if (probs[:, 1] >= 0.5).sum() > 0
        else 0.0,
    }
