"""
Centralized metric computation

All phases MUST use these functions to compute MCC and other metrics.
This prevents drift and ensures consistency across the pipeline.

2025 best practices:
- Type hints for numpy/torch arrays
- Explicit return types (float, not np.float64)
- Handles edge cases (all zeros, all ones)
- Vectorized operations (no loops)
"""

from typing import Dict, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, torch.Tensor, list]


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    """Convert torch.Tensor or list to numpy array"""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def compute_mcc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Matthews Correlation Coefficient (MCC)

    THE metric for imbalanced binary classification.
    Range: [-1, +1]
    - +1: Perfect prediction
    -  0: Random prediction
    - -1: Perfect inverse prediction

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        MCC score as Python float

    Why MCC:
        - Balanced metric for imbalanced datasets
        - Takes all confusion matrix cells into account
        - More informative than accuracy or F1
        - Standard metric in NATIX roadwork detection

    Example:
        >>> y_true = [0, 0, 1, 1, 1]
        >>> y_pred = [0, 0, 1, 1, 0]
        >>> compute_mcc(y_true, y_pred)
        0.6666666666666666
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    # Handle edge case: all predictions are the same
    if len(np.unique(y_pred)) == 1:
        # If all predictions match ground truth, MCC = 1
        # Otherwise, MCC = -1 or 0 depending on ground truth distribution
        pass  # sklearn handles this correctly

    return float(matthews_corrcoef(y_true, y_pred))


def compute_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Classification accuracy

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy as Python float (0.0 to 1.0)

    Example:
        >>> y_true = [0, 0, 1, 1, 1]
        >>> y_pred = [0, 0, 1, 1, 0]
        >>> compute_accuracy(y_true, y_pred)
        0.8
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(accuracy_score(y_true, y_pred))


def compute_precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Precision: TP / (TP + FP)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Precision as Python float (0.0 to 1.0)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(precision_score(y_true, y_pred, zero_division=0.0))


def compute_recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Recall (Sensitivity, TPR): TP / (TP + FN)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Recall as Python float (0.0 to 1.0)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(recall_score(y_true, y_pred, zero_division=0.0))


def compute_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    F1 score: Harmonic mean of precision and recall

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        F1 score as Python float (0.0 to 1.0)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(f1_score(y_true, y_pred, zero_division=0.0))


def compute_confusion(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, int]:
    """
    Confusion matrix components

    Returns:
        Dict with keys: TP, TN, FP, FN (all Python ints)

    Example:
        >>> y_true = [0, 0, 1, 1, 1]
        >>> y_pred = [0, 0, 1, 1, 0]
        >>> compute_confusion(y_true, y_pred)
        {'TP': 2, 'TN': 2, 'FP': 0, 'FN': 1}
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    # confusion_matrix returns [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        # Edge case: only one class in predictions
        if np.unique(y_true)[0] == np.unique(y_pred)[0]:
            # All correct
            if np.unique(y_true)[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            # All wrong
            if np.unique(y_true)[0] == 0:
                tn, fp, fn, tp = 0, 0, cm[0, 0], 0
            else:
                tn, fp, fn, tp = 0, cm[0, 0], 0, 0
    else:
        # Unexpected shape
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def compute_fnr(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    False Negative Rate (FNR): FN / (FN + TP)

    Also called "Miss Rate" or "Type II Error Rate".
    Critical for roadwork detection: minimizing missed potholes.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        FNR as Python float (0.0 to 1.0)

    Example:
        >>> y_true = [0, 0, 1, 1, 1]
        >>> y_pred = [0, 0, 1, 1, 0]
        >>> compute_fnr(y_true, y_pred)
        0.3333333333333333
    """
    conf = compute_confusion(y_true, y_pred)
    denominator = conf["FN"] + conf["TP"]

    if denominator == 0:
        return 0.0  # No positive samples

    return float(conf["FN"] / denominator)


def compute_fpr(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    False Positive Rate (FPR): FP / (FP + TN)

    Also called "Fall-out" or "Type I Error Rate".

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        FPR as Python float (0.0 to 1.0)
    """
    conf = compute_confusion(y_true, y_pred)
    denominator = conf["FP"] + conf["TN"]

    if denominator == 0:
        return 0.0  # No negative samples

    return float(conf["FP"] / denominator)


def compute_all_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """
    Compute all standard metrics at once

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dict with keys:
        - mcc: Matthews Correlation Coefficient
        - accuracy: Classification accuracy
        - precision: Precision
        - recall: Recall (TPR)
        - f1: F1 score
        - fnr: False Negative Rate
        - fpr: False Positive Rate
        - tp, tn, fp, fn: Confusion matrix components

    Example:
        >>> y_true = [0, 0, 1, 1, 1]
        >>> y_pred = [0, 0, 1, 1, 0]
        >>> metrics = compute_all_metrics(y_true, y_pred)
        >>> metrics['mcc']
        0.6666666666666666
        >>> metrics['accuracy']
        0.8
    """
    # Compute confusion matrix first (reused by other metrics)
    conf = compute_confusion(y_true, y_pred)

    metrics = {
        "mcc": compute_mcc(y_true, y_pred),
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred),
        "recall": compute_recall(y_true, y_pred),
        "f1": compute_f1(y_true, y_pred),
        "fnr": compute_fnr(y_true, y_pred),
        "fpr": compute_fpr(y_true, y_pred),
        # Include confusion matrix
        "tp": conf["TP"],
        "tn": conf["TN"],
        "fp": conf["FP"],
        "fn": conf["FN"],
    }

    return metrics
