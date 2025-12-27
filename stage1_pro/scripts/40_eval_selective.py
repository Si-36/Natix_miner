#!/usr/bin/env python3
"""
Phase 2: Evaluate selective metrics on validation set.

Computes:
- Risk@Coverage
- Coverage@Risk
- FNR on exits
- Average set size (for prediction sets)
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

import torch
import numpy as np
import argparse
import json
from pathlib import Path


def compute_selective_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    gate_probs: np.ndarray = None,
    thresholds: list = None,
) -> dict:
    """
    Compute selective classification metrics.

    Args:
        probs: (N, 2) class probabilities
        labels: (N,) ground truth labels
        gate_probs: (N,) gate probabilities (Phase 2+)
        thresholds: List of thresholds to evaluate

    Returns:
        Dictionary of selective metrics
    """
    metrics = {}

    # Get max class probability and prediction
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    # Accuracy
    metrics["accuracy"] = (preds == labels).mean()

    # ECE
    confidences = max_probs
    accuracies = (preds == labels).astype(float)

    n_bins = 10
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += (bin_size / len(labels)) * abs(avg_confidence - avg_accuracy)

    metrics["ece"] = ece

    # Evaluate at different thresholds
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95]

    threshold_metrics = []

    for threshold in thresholds:
        exit_mask = max_probs >= threshold

        if exit_mask.sum() == 0:
            threshold_metrics.append(
                {
                    "threshold": threshold,
                    "coverage": 0.0,
                    "exit_accuracy": 0.0,
                    "fnr_on_exits": 1.0,
                }
            )
            continue

        exit_labels = labels[exit_mask]
        exit_preds = preds[exit_mask]

        exit_accuracy = (exit_preds == exit_labels).mean()
        coverage = exit_mask.mean()

        # FNR on exits
        if (exit_labels == 1).sum() > 0:
            fnr_on_exits = ((exit_labels == 1) & (exit_preds == 0)).sum() / (
                exit_labels == 1
            ).sum()
        else:
            fnr_on_exits = 1.0

        threshold_metrics.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "exit_accuracy": exit_accuracy,
                "fnr_on_exits": fnr_on_exits,
            }
        )

    metrics["threshold_sweep"] = threshold_metrics

    # Gate metrics (Phase 2+)
    if gate_probs is not None:
        gate_threshold = 0.5
        gate_mask = gate_probs >= gate_threshold

        if gate_mask.sum() > 0:
            gate_coverage = gate_mask.mean()
            gate_labels = labels[gate_mask]
            gate_preds = preds[gate_mask]
            gate_accuracy = (gate_preds == gate_labels).mean()

            metrics["gate_coverage"] = gate_coverage
            metrics["gate_accuracy"] = gate_accuracy
        else:
            metrics["gate_coverage"] = 0.0
            metrics["gate_accuracy"] = 0.0

    # Risk@Coverage
    metrics["risk_at_coverage"] = {}

    valid_thresholds = [t for t in threshold_metrics if t["coverage"] > 0]

    for coverage_target in [0.5, 0.6, 0.7, 0.8, 0.9]:
        # Find threshold with coverage >= target
        valid = [t for t in valid_thresholds if t["coverage"] >= coverage_target]

        if len(valid) > 0:
            # Among valid, minimize FNR
            best = min(valid, key=lambda x: x["fnr_on_exits"])
            metrics["risk_at_coverage"][f"cov_{int(coverage_target * 100)}"] = {
                "fnr": best["fnr_on_exits"],
                "threshold": best["threshold"],
                "exit_accuracy": best["exit_accuracy"],
            }
        else:
            metrics["risk_at_coverage"][f"cov_{int(coverage_target * 100)}"] = None

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_probs", type=str, required=True)
    parser.add_argument("--val_labels", type=str, required=True)
    parser.add_argument("--val_gate_probs", type=str, default=None)
    parser.add_argument("--thresholds", type=str, default=None)
    parser.add_argument("--output", type=str, default="selective_metrics.json")
    args = parser.parse_args()

    print("Loading validation data...")
    val_probs = torch.load(args.val_probs)
    val_labels = torch.load(args.val_labels)

    val_gate_probs = None
    if args.val_gate_probs:
        val_gate_probs = torch.load(args.val_gate_probs)

    # Convert to numpy
    val_probs_np = (
        val_probs.numpy() if isinstance(val_probs, torch.Tensor) else val_probs
    )
    val_labels_np = (
        val_labels.numpy() if isinstance(val_labels, torch.Tensor) else val_labels
    )

    if val_gate_probs is not None:
        val_gate_probs_np = (
            val_gate_probs.numpy()
            if isinstance(val_gate_probs, torch.Tensor)
            else val_gate_probs
        )
    else:
        val_gate_probs_np = None

    thresholds = eval(args.thresholds) if args.thresholds else None

    print("Computing selective metrics...")
    metrics = compute_selective_metrics(
        val_probs_np, val_labels_np, val_gate_probs_np, thresholds
    )

    # Save metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSelective Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")

    if "gate_coverage" in metrics:
        print(f"  Gate Coverage: {metrics['gate_coverage']:.4f}")
        print(f"  Gate Accuracy: {metrics['gate_accuracy']:.4f}")

    print(f"\nThreshold Sweep:")
    for t in metrics["threshold_sweep"]:
        print(
            f"  Threshold {t['threshold']:.2f}: "
            f"Cov={t['coverage']:.3f}, "
            f"Acc={t['exit_accuracy']:.3f}, "
            f"FNR={t['fnr_on_exits']:.3f}"
        )

    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
