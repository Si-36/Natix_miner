"""
Visualization Module - Phase 2.9-2.10: Metrics Visualization (Dec 2025 Best Practice)

Creates plots for:
- Risk-coverage curves with bootstrap CI bands
- AUGRC distribution
- Calibration curves
- FNR/coverage distributions

Uses matplotlib for static plotting (no interactive backends).
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (Dec 2025 best practice)

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os


def plot_risk_coverage_curve(
    coverage_array: np.ndarray,
    risk_array: np.ndarray,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot risk-coverage curve with bootstrap CI bands (Phase 2.9).
    
    Args:
        coverage_array: Array of coverage values [0.0, ..., 1.0]
        risk_array: Array of risk values corresponding to coverage
        ci_lower: Lower CI bound for each coverage (optional)
        ci_upper: Upper CI bound for each coverage (optional)
        save_path: Path to save plot
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot main curve
    ax.plot(coverage_array, risk_array, 'b-', linewidth=2, label='Risk-Coverage', color='#1f77b4')
    
    # Plot CI bands if provided
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(coverage_array, ci_lower, ci_upper, alpha=0.3, color='#1f77b4', label='95% CI')
    
    # Formatting
    ax.set_xlabel('Coverage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk (Error Rate)', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Coverage Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Risk-coverage curve saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_augrc_distribution(
    augrc_samples: np.ndarray,
    augrc_mean: float,
    ci_lower: float,
    ci_upper: float,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot AUGRC distribution across bootstrap samples (Phase 2.10).
    
    Args:
        augrc_samples: Array of AUGRC values from bootstrap samples
        augrc_mean: Mean AUGRC
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound
        save_path: Path to save plot
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(augrc_samples, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
    
    # Plot mean line
    ax.axvline(augrc_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {augrc_mean:.4f}', alpha=0.8)
    
    # Plot CI bounds
    ax.axvline(ci_lower, color='orange', linestyle=':', linewidth=2, label=f'95% CI Lower: {ci_lower:.4f}', alpha=0.6)
    ax.axvline(ci_upper, color='orange', linestyle=':', linewidth=2, label=f'95% CI Upper: {ci_upper:.4f}', alpha=0.6)
    
    # Formatting
    ax.set_xlabel('AUGRC', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('AUGRC Distribution (Bootstrap Samples)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ AUGRC distribution saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        n_bins: Number of calibration bins
        save_path: Path to save plot
        show: Whether to display plot
    """
    # Get max probabilities and predicted labels
    max_probs = np.max(probs, axis=1)
    predicted = np.argmax(probs, axis=1)
    
    # Create bins
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    
    # Compute calibration metrics per bin
    confidences = []
    accuracies = []
    counts = []
    
    for i in range(n_bins):
        # Get samples in this bin
        in_bin = (max_probs >= bin_lowers[i]) & (max_probs < bin_boundaries[i + 1])
        
        if in_bin.sum() == 0:
            confidences.append(np.nan)
            accuracies.append(np.nan)
            counts.append(0)
            continue
        
        # Average confidence in bin
        avg_conf = max_probs[in_bin].mean()
        confidences.append(avg_conf)
        
        # Accuracy in bin
        acc = (predicted[in_bin] == labels[in_bin]).mean()
        accuracies.append(acc)
        
        # Count
        counts.append(in_bin.sum())
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', color='gray', alpha=0.8)
    
    # Plot calibration curve
    ax.plot(confidences, accuracies, 'b-', linewidth=2, label='Model Calibration', color='#1f77b4', markersize=8)
    
    # Formatting
    ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Show bin sizes as annotations
    for i, count in enumerate(counts):
        if count > 0 and i % 3 == 0:
            ax.annotate(f'n={count}', xy=(confidences[i], accuracies[i]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Calibration curve saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_fnr_coverage_distribution(
    fnr_values: np.ndarray,
    coverage_values: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot FNR vs Coverage distribution (complementary to risk-coverage curve).
    
    Args:
        fnr_values: Array of FNR values
        coverage_values: Array of coverage values
        save_path: Path to save plot
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scatter
    ax.scatter(coverage_values, fnr_values, alpha=0.6, color='#e74c3c', s=50, edgecolors='black', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Coverage', fontsize=12, fontweight='bold')
    ax.set_ylabel('FNR (Positive Class)', fontsize=12, fontweight='bold')
    ax.set_title('FNR vs Coverage Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ FNR-coverage distribution saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

