"""
Calibration Metrics Module - Phase 2.8: NLL/Brier Computation (Dec 2025 Best Practice)

Implements:
- Negative Log Likelihood (NLL)
- Brier Score
- Bootstrap confidence intervals for both metrics
"""

import torch
import numpy as np
from typing import Dict


def compute_nll(
    probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-15
) -> float:
    """
    Compute Negative Log Likelihood (NLL) for calibrated probabilities.
    
    NLL = -1/N * sum(log(p_y)) where p_y is predicted probability for true class.
    
    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        eps: Small epsilon to avoid log(0)
    
    Returns:
        NLL score (lower is better)
    """
    # Get probabilities for true labels
    probs_np = probs.cpu().numpy() if hasattr(probs, 'cpu') else probs
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
    
    # Get p_y (probability of true class)
    p_y = probs_np[np.arange(len(labels_np)), labels_np]
    
    # Add epsilon to avoid log(0)
    p_y = np.clip(p_y, eps, 1.0 - eps)
    
    # Compute NLL
    nll = -np.mean(np.log(p_y))
    
    return float(nll)


def compute_brier_score(
    probs: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Compute Brier Score for calibrated probabilities.
    
    Brier = 1/N * sum((p_y - y)^2) where p_y is predicted probability for true class.
    
    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
    
    Returns:
        Brier score (lower is better, range [0, 1])
    """
    # Get probabilities for true labels
    probs_np = probs.cpu().numpy() if hasattr(probs, 'cpu') else probs
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
    
    # Get p_y (probability of true class)
    p_y = probs_np[np.arange(len(labels_np)), labels_np]
    
    # Convert labels to 0/1
    y = labels_np.astype(np.float64)
    
    # Compute Brier score
    brier = np.mean((p_y - y) ** 2)
    
    return float(brier)


def compute_nll_brier_bootstrap(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Compute NLL and Brier with bootstrap confidence intervals (Phase 2.8).
    
    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        n_bootstrap: Number of bootstrap samples (default 1000)
        confidence: Confidence level (default 0.95)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dict with nll_mean, nll_std, nll_ci_lower, nll_ci_upper,
             brier_mean, brier_std, brier_ci_lower, brier_ci_upper
    """
    # Set seed for reproducibility
    np.random.seed(random_seed)
    
    # Convert to numpy
    probs_np = probs.cpu().numpy() if hasattr(probs, 'cpu') else probs
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
    
    n_samples = len(labels_np)
    
    # Bootstrap samples
    nll_samples = []
    brier_samples = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        resampled_probs = probs_np[indices]
        resampled_labels = labels_np[indices]
        
        # Compute NLL
        p_y = resampled_probs[np.arange(len(resampled_labels)), resampled_labels]
        eps = 1e-15
        p_y = np.clip(p_y, eps, 1.0 - eps)
        nll = -np.mean(np.log(p_y))
        nll_samples.append(nll)
        
        # Compute Brier
        y = resampled_labels.astype(np.float64)
        brier = np.mean((p_y - y) ** 2)
        brier_samples.append(brier)
    
    nll_samples = np.array(nll_samples)
    brier_samples = np.array(brier_samples)
    
    # Compute NLL statistics
    nll_mean = np.mean(nll_samples)
    nll_std = np.std(nll_samples)
    
    # Percentile CI for NLL
    alpha = 1.0 - confidence
    nll_ci_lower = np.percentile(nll_samples, 100 * alpha / 2)
    nll_ci_upper = np.percentile(nll_samples, 100 * (1.0 - alpha / 2))
    
    # Compute Brier statistics
    brier_mean = np.mean(brier_samples)
    brier_std = np.std(brier_samples)
    
    # Percentile CI for Brier
    brier_ci_lower = np.percentile(brier_samples, 100 * alpha / 2)
    brier_ci_upper = np.percentile(brier_samples, 100 * (1.0 - alpha / 2))
    
    return {
        'nll_mean': float(nll_mean),
        'nll_std': float(nll_std),
        'nll_ci_lower': float(nll_ci_lower),
        'nll_ci_upper': float(nll_ci_upper),
        'brier_mean': float(brier_mean),
        'brier_std': float(brier_std),
        'brier_ci_lower': float(brier_ci_lower),
        'brier_ci_upper': float(brier_ci_upper)
    }


def compute_calibration_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and related metrics.
    
    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        n_bins: Number of calibration bins
    
    Returns:
        Dict with ece, mce, brier_score
    """
    # Convert to numpy
    probs_np = probs.cpu().numpy() if hasattr(probs, 'cpu') else probs
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
    
    # Get max probabilities and predictions
    max_probs = np.max(probs_np, axis=1)
    predicted = np.argmax(probs_np, axis=1)
    
    # Compute Brier score
    y = labels_np.astype(np.float64)
    brier = np.mean((max_probs - y) ** 2)
    
    # Compute ECE (Expected Calibration Error)
    ece = 0.0
    
    # Create bins
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    
    for i in range(n_bins):
        # Get samples in this bin
        in_bin = (max_probs >= bin_boundaries[i]) & (max_probs < bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            # Average confidence in bin
            avg_conf = max_probs[in_bin].mean()
            
            # Accuracy in bin
            acc_in_bin = (predicted[in_bin] == labels_np[in_bin]).mean()
            
            # ECE contribution
            bin_weight = in_bin.sum() / len(labels_np)
            ece += bin_weight * np.abs(avg_conf - acc_in_bin)
    
    return {
        'ece': float(ece),
        'brier_score': float(brier),
        'mce': float(np.mean(np.abs(max_probs - (predicted == labels_np).astype(float))))
    }
