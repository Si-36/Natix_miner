"""
Centralized evaluation functions

Single source of truth for:
- MCC (Matthews Correlation Coefficient)
- Accuracy, Precision, Recall, F1
- Confusion matrix metrics (TP, TN, FP, FN)
- False Negative Rate (FNR)
- Threshold selection (max MCC)
- Evaluation reports (Day 6)
- Hard examples identification (Day 6)
- Threshold sweep CSV export (Day 6)

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
from .reports import (
    compute_confusion_matrix_dict,
    export_confusion_matrix,
    compute_per_class_metrics,
    identify_hard_examples,
    export_hard_examples,
    create_eval_report,
    export_eval_report,
)
from .sweep import (
    compute_threshold_sweep,
    export_threshold_sweep_csv,
    find_optimal_threshold,
    compute_roc_curve_data,
    compute_pr_curve_data,
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
    # Day 6: Evaluation reports
    "compute_confusion_matrix_dict",
    "export_confusion_matrix",
    "compute_per_class_metrics",
    "identify_hard_examples",
    "export_hard_examples",
    "create_eval_report",
    "export_eval_report",
    # Day 6: Threshold sweep
    "compute_threshold_sweep",
    "export_threshold_sweep_csv",
    "find_optimal_threshold",
    "compute_roc_curve_data",
    "compute_pr_curve_data",
]
