"""
Visualization utilities for Phase 2+ metrics

Risk-coverage curves, AUGRC distributions, calibration curves.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional


def plot_risk_coverage_curve(
    risk_coverage_dict: Dict,
    output_path: str,
    bootstrap_ci: Optional[Dict] = None
):
    """
    Plot risk-coverage curve with bootstrap CI bands (Phase 2.9).
    
    Args:
        risk_coverage_dict: Dictionary with 'coverage', 'risk', 'thresholds'
        output_path: Path to save plot
        bootstrap_ci: Optional bootstrap CI data
    """
    coverage = risk_coverage_dict['coverage']
    risk = risk_coverage_dict['risk']
    
    plt.figure(figsize=(10, 6))
    
    if bootstrap_ci is not None:
        # Plot CI bands
        coverage_ci_lower = bootstrap_ci.get('coverage_ci_lower', coverage)
        coverage_ci_upper = bootstrap_ci.get('coverage_ci_upper', coverage)
        risk_ci_lower = bootstrap_ci.get('risk_ci_lower', risk)
        risk_ci_upper = bootstrap_ci.get('risk_ci_upper', risk)
        
        plt.fill_between(coverage, risk_ci_lower, risk_ci_upper, alpha=0.3, label='95% CI')
    
    plt.plot(coverage, risk, 'b-', linewidth=2, label='Risk-Coverage Curve')
    plt.xlabel('Coverage', fontsize=12)
    plt.ylabel('Risk (Error Rate)', fontsize=12)
    plt.title('Risk-Coverage Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved risk-coverage curve to {output_path}")


def plot_augrc_distribution(
    augrc_samples: np.ndarray,
    augrc_mean: float,
    augrc_ci: tuple,
    output_path: str
):
    """
    Plot AUGRC distribution across bootstrap samples (Phase 2.10).
    
    Args:
        augrc_samples: Bootstrap AUGRC samples
        augrc_mean: Mean AUGRC
        augrc_ci: (ci_lower, ci_upper) tuple
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(augrc_samples, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(augrc_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {augrc_mean:.4f}')
    plt.axvline(augrc_ci[0], color='orange', linestyle='--', linewidth=1.5, label=f'95% CI: [{augrc_ci[0]:.4f}, {augrc_ci[1]:.4f}]')
    plt.axvline(augrc_ci[1], color='orange', linestyle='--', linewidth=1.5)
    
    plt.xlabel('AUGRC', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('AUGRC Distribution (Bootstrap)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved AUGRC distribution to {output_path}")


def plot_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    output_path: str = None
):
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        probs: (N, num_classes) predicted probabilities
        labels: (N,) ground truth labels
        n_bins: Number of bins
        output_path: Path to save plot
    """
    confidences = np.max(probs, axis=1) if probs.ndim > 1 else probs
    predictions = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs > 0.5).astype(int)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_counts.append(in_bin.sum())
        if in_bin.sum() > 0:
            bin_confidences.append(confidences[in_bin].mean())
            bin_accuracies.append(accuracies[in_bin].mean())
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Mean Actual Accuracy', fontsize=12)
    plt.title('Calibration Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"✅ Saved calibration curve to {output_path}")
    else:
        plt.close()

