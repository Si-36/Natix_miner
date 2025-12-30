"""
Centralized evaluation functions

Single source of truth for:
- MCC (Matthews Correlation Coefficient)
- Accuracy, Precision, Recall, F1
- Confusion matrix metrics (TP, TN, FP, FN)
- False Negative Rate (FNR)
- Threshold selection (max MCC)

Why centralized:
- Prevents metric drift across phases
- Ensures all phases compute MCC identically
- Single place to update metric computation
- Type-safe with numpy/torch arrays
"""

from .metrics import (
    compute_mcc,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_confusion,
    compute_fnr,
    compute_fpr,
    compute_all_metrics,
)
from .thresholds import (
    select_threshold_max_mcc,
    sweep_thresholds_binary,
    plot_threshold_curve,
)

__all__ = [
    # Core metrics
    "compute_mcc",
    "compute_accuracy",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_confusion",
    "compute_fnr",
    "compute_fpr",
    "compute_all_metrics",
    # Threshold selection
    "select_threshold_max_mcc",
    "sweep_thresholds_binary",
    "plot_threshold_curve",
]
