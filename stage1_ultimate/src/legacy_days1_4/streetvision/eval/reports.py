"""
Evaluation reporting utilities (Day 6)

Features:
- Confusion matrix export (JSON + CSV)
- Per-class metrics breakdown
- Hard examples identification
- Standard evaluation reports

2025-12-30 best practices:
- Type-safe with numpy/torch compatibility
- JSON-serializable outputs
- Atomic writes for all exports
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch

from streetvision.eval.metrics import (
    compute_mcc,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_confusion,
)
from streetvision.io.atomic import write_json_atomic


ArrayLike = Union[np.ndarray, torch.Tensor, List]


def compute_confusion_matrix_dict(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute confusion matrix with metadata

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional class names (defaults to ["negative", "positive"])

    Returns:
        Dictionary with confusion matrix data:
        {
            "matrix": [[TN, FP], [FN, TP]],
            "class_names": ["negative", "positive"],
            "counts": {"TP": ..., "TN": ..., "FP": ..., "FN": ...},
            "rates": {"TPR": ..., "TNR": ..., "FPR": ..., "FNR": ...},
        }
    """
    if class_names is None:
        class_names = ["negative", "positive"]

    # Compute confusion components
    conf = compute_confusion(y_true, y_pred)

    # Build matrix (2×2 for binary classification)
    matrix = [
        [conf["TN"], conf["FP"]],  # Row 0: true negative class
        [conf["FN"], conf["TP"]],  # Row 1: true positive class
    ]

    # Compute rates
    total_pos = conf["TP"] + conf["FN"]
    total_neg = conf["TN"] + conf["FP"]

    tpr = conf["TP"] / total_pos if total_pos > 0 else 0.0  # Recall / Sensitivity
    tnr = conf["TN"] / total_neg if total_neg > 0 else 0.0  # Specificity
    fpr = conf["FP"] / total_neg if total_neg > 0 else 0.0  # False Positive Rate
    fnr = conf["FN"] / total_pos if total_pos > 0 else 0.0  # False Negative Rate

    return {
        "matrix": matrix,
        "class_names": class_names,
        "counts": {
            "TP": int(conf["TP"]),
            "TN": int(conf["TN"]),
            "FP": int(conf["FP"]),
            "FN": int(conf["FN"]),
        },
        "rates": {
            "TPR": float(tpr),  # True Positive Rate (Recall)
            "TNR": float(tnr),  # True Negative Rate (Specificity)
            "FPR": float(fpr),  # False Positive Rate
            "FNR": float(fnr),  # False Negative Rate
        },
        "total_samples": int(conf["TP"] + conf["TN"] + conf["FP"] + conf["FN"]),
    }


def export_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    output_path: Path,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Export confusion matrix to JSON file (atomic write)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        output_path: Path to save confusion.json
        class_names: Optional class names

    Returns:
        SHA256 checksum of written file
    """
    confusion_dict = compute_confusion_matrix_dict(y_true, y_pred, class_names)
    return write_json_atomic(output_path, confusion_dict)


def compute_per_class_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics (precision, recall, F1)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional class names

    Returns:
        Dictionary mapping class names to metrics:
        {
            "negative": {"precision": ..., "recall": ..., "f1": ..., "support": ...},
            "positive": {"precision": ..., "recall": ..., "f1": ..., "support": ...},
        }
    """
    if class_names is None:
        class_names = ["negative", "positive"]

    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Compute confusion matrix
    conf = compute_confusion(y_true, y_pred)

    # Compute per-class metrics
    # Class 0 (negative)
    class0_tp = conf["TN"]  # For negative class, TN is TP
    class0_fp = conf["FN"]  # FN for positive class is FP for negative class
    class0_fn = conf["FP"]  # FP for positive class is FN for negative class
    class0_support = class0_tp + class0_fn

    class0_precision = class0_tp / (class0_tp + class0_fp) if (class0_tp + class0_fp) > 0 else 0.0
    class0_recall = class0_tp / (class0_tp + class0_fn) if (class0_tp + class0_fn) > 0 else 0.0
    class0_f1 = 2 * (class0_precision * class0_recall) / (class0_precision + class0_recall) if (class0_precision + class0_recall) > 0 else 0.0

    # Class 1 (positive)
    class1_tp = conf["TP"]
    class1_fp = conf["FP"]
    class1_fn = conf["FN"]
    class1_support = class1_tp + class1_fn

    class1_precision = class1_tp / (class1_tp + class1_fp) if (class1_tp + class1_fp) > 0 else 0.0
    class1_recall = class1_tp / (class1_tp + class1_fn) if (class1_tp + class1_fn) > 0 else 0.0
    class1_f1 = 2 * (class1_precision * class1_recall) / (class1_precision + class1_recall) if (class1_precision + class1_recall) > 0 else 0.0

    return {
        class_names[0]: {
            "precision": float(class0_precision),
            "recall": float(class0_recall),
            "f1": float(class0_f1),
            "support": int(class0_support),
        },
        class_names[1]: {
            "precision": float(class1_precision),
            "recall": float(class1_recall),
            "f1": float(class1_f1),
            "support": int(class1_support),
        },
    }


def identify_hard_examples(
    logits: torch.Tensor,
    labels: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    top_k: int = 100,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identify hard examples (misclassified, low confidence)

    Args:
        logits: Model logits (N, num_classes)
        labels: Ground truth labels (N,)
        indices: Optional sample indices (N,)
        top_k: Number of hard examples per category

    Returns:
        Dictionary with hard example categories:
        {
            "false_positives": [...],  # High confidence but wrong (0→1)
            "false_negatives": [...],  # High confidence but wrong (1→0)
            "low_confidence_correct": [...],  # Correct but uncertain
        }
    """
    # Convert to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    # Compute predictions and confidences
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = np.argmax(logits, axis=1)
    confidences = np.max(probs, axis=1)

    # Generate indices if not provided
    if indices is None:
        indices = np.arange(len(labels))
    elif isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()

    # Identify categories
    false_positives = []  # Predicted 1, actually 0
    false_negatives = []  # Predicted 0, actually 1
    low_confidence_correct = []  # Correct but low confidence

    for i in range(len(labels)):
        example = {
            "index": int(indices[i]),
            "true_label": int(labels[i]),
            "pred_label": int(preds[i]),
            "confidence": float(confidences[i]),
            "logits": logits[i].tolist(),
        }

        if labels[i] == 0 and preds[i] == 1:
            # False positive
            false_positives.append(example)
        elif labels[i] == 1 and preds[i] == 0:
            # False negative
            false_negatives.append(example)
        elif labels[i] == preds[i] and confidences[i] < 0.7:
            # Correct but low confidence
            low_confidence_correct.append(example)

    # Sort by confidence (descending for FP/FN, ascending for low-conf)
    false_positives = sorted(false_positives, key=lambda x: x["confidence"], reverse=True)[:top_k]
    false_negatives = sorted(false_negatives, key=lambda x: x["confidence"], reverse=True)[:top_k]
    low_confidence_correct = sorted(low_confidence_correct, key=lambda x: x["confidence"])[:top_k]

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "low_confidence_correct": low_confidence_correct,
        "counts": {
            "total_false_positives": len([1 for i in range(len(labels)) if labels[i] == 0 and preds[i] == 1]),
            "total_false_negatives": len([1 for i in range(len(labels)) if labels[i] == 1 and preds[i] == 0]),
            "total_low_confidence_correct": len([1 for i in range(len(labels)) if labels[i] == preds[i] and confidences[i] < 0.7]),
        },
    }


def export_hard_examples(
    logits: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    indices: Optional[torch.Tensor] = None,
    top_k: int = 100,
) -> str:
    """
    Export hard examples to JSON file (atomic write)

    Args:
        logits: Model logits
        labels: Ground truth labels
        output_path: Path to save hard_examples.json
        indices: Optional sample indices
        top_k: Number of examples per category

    Returns:
        SHA256 checksum of written file
    """
    hard_examples = identify_hard_examples(logits, labels, indices, top_k)
    return write_json_atomic(output_path, hard_examples)


def create_eval_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    phase_name: str,
    additional_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Create standard evaluation report

    Args:
        logits: Model logits
        labels: Ground truth labels
        phase_name: Phase name (e.g., "phase1_baseline")
        additional_metrics: Optional additional metrics

    Returns:
        Standard eval report dictionary
    """
    # Convert to numpy
    if isinstance(logits, torch.Tensor):
        logits_np = logits.cpu().numpy()
    else:
        logits_np = np.asarray(logits)

    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    # Compute predictions
    preds = np.argmax(logits_np, axis=1)

    # Compute all metrics
    mcc = compute_mcc(labels_np, preds)
    accuracy = compute_accuracy(labels_np, preds)
    precision = compute_precision(labels_np, preds)
    recall = compute_recall(labels_np, preds)
    f1 = compute_f1(labels_np, preds)
    confusion = compute_confusion(labels_np, preds)

    # Per-class metrics
    per_class = compute_per_class_metrics(labels_np, preds)

    # Confusion matrix
    confusion_matrix = compute_confusion_matrix_dict(labels_np, preds)

    # Build report
    report = {
        "phase": phase_name,
        "metrics": {
            "mcc": float(mcc),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "confusion_matrix": confusion_matrix,
        "per_class_metrics": per_class,
        "sample_counts": {
            "total": int(len(labels_np)),
            "positive": int(np.sum(labels_np == 1)),
            "negative": int(np.sum(labels_np == 0)),
        },
    }

    # Add additional metrics if provided
    if additional_metrics:
        report["metrics"].update(additional_metrics)

    return report


def export_eval_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    phase_name: str,
    additional_metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Export evaluation report to JSON file (atomic write)

    Args:
        logits: Model logits
        labels: Ground truth labels
        output_path: Path to save eval_report.json
        phase_name: Phase name
        additional_metrics: Optional additional metrics

    Returns:
        SHA256 checksum of written file
    """
    report = create_eval_report(logits, labels, phase_name, additional_metrics)
    return write_json_atomic(output_path, report)


__all__ = [
    "compute_confusion_matrix_dict",
    "export_confusion_matrix",
    "compute_per_class_metrics",
    "identify_hard_examples",
    "export_hard_examples",
    "create_eval_report",
    "export_eval_report",
]
