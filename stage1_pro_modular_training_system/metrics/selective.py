"""
Selective Metrics Module - Phase 2: Selective Evaluation (Dec 2025 Best Practice)

Implements:
- Risk-coverage curves
- AUGRC (Area Under Generalized Risk Curve)
- Bootstrap confidence intervals
- Selective metrics (Risk@Coverage, Coverage@Risk)
"""

import torch
import numpy as np
from typing import Tuple, Dict, List
from collections import defaultdict


def compute_risk_coverage(
    probs: torch.Tensor,
    labels: torch.Tensor,
    thresholds: np.ndarray,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute risk-coverage curve (Phase 2.1).
    
    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        thresholds: Array of coverage thresholds [0.0, 0.1, ..., 1.0]
        device: Device (cpu/cuda)
    
    Returns:
        (coverage_array, risk_array, errors_array)
    """
    probs_np = probs.cpu().numpy() if hasattr(probs, 'cpu') else probs
    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
    
    # Get max probabilities
    max_probs = np.max(probs_np, axis=1)
    predicted_labels = np.argmax(probs_np, axis=1)
    
    coverage_array = []
    risk_array = []
    errors_array = []
    
    for threshold in thresholds:
        # Exit condition: max_prob >= threshold
        exit_mask = max_probs >= threshold
        
        if exit_mask.sum() == 0:
            # No samples exit - full risk
            coverage_array.append(0.0)
            risk_array.append(1.0)
            errors_array.append(len(labels_np))
        else:
            # Coverage
            coverage = exit_mask.sum() / len(labels_np)
            coverage_array.append(coverage)
            
            # Risk on accepted samples (error rate on exited)
            exit_labels = predicted_labels[exit_mask]
            true_labels = labels_np[exit_mask]
            errors = (exit_labels != true_labels).sum()
            risk = errors / exit_mask.sum() if exit_mask.sum() > 0 else 1.0
            
            risk_array.append(risk)
            errors_array.append(errors)
    
    return np.array(coverage_array), np.array(risk_array), np.array(errors_array)


def compute_augrc(
    coverage_array: np.ndarray,
    risk_array: np.ndarray,
    target_coverage: float = 0.9
) -> Dict[str, float]:
    """
    Compute AUGRC (Area Under Generalized Risk Curve) (Phase 2.2).
    
    Args:
        coverage_array: Array of coverage values [0.0, ..., 1.0]
        risk_array: Array of risk values corresponding to coverage
        target_coverage: Target coverage for R@C metric
    
    Returns:
        Dict with augrc, risk_at_target_coverage
    """
    # AUGRC: Area under risk-coverage curve (trapezoidal rule)
    # Use np.trapezoid (Dec 2025, np.trapz deprecated)
    augrc = np.trapezoid(risk_array, coverage_array)
    
    # Risk@TargetCoverage
    idx = np.argmin(np.abs(coverage_array - target_coverage))
    risk_at_target = risk_array[idx]
    
    return {
        'augrc': float(augrc),
        f'risk_at_coverage_{int(target_coverage*100)}': float(risk_at_target),
        'coverage_at_target': float(coverage_array[idx])
    }


def compute_bootstrap_cis(
    metric_array: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals (Phase 2.4).
    
    Uses percentile method for uncertainty estimation.
    
    Args:
        metric_array: Array of metric values from bootstrap samples
        n_bootstrap: Number of bootstrap samples (default 1000)
        confidence: Confidence level (default 0.95)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dict with mean, std, ci_lower, ci_upper
    """
    # Set seed for reproducibility
    np.random.seed(random_seed)
    
    # Bootstrap sampling
    bootstrap_samples = []
    n_samples = len(metric_array)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_samples.append(metric_array[indices])
    
    bootstrap_samples = np.array(bootstrap_samples)
    
    # Compute statistics
    mean = np.mean(bootstrap_samples)
    std = np.std(bootstrap_samples)
    
    # Percentile CI
    alpha = 1.0 - confidence
    ci_lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_samples, 100 * (1.0 - alpha / 2))
    
    return {
        'mean': float(mean),
        'std': float(std),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }


def compute_selective_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    coverages: np.ndarray = np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
    risks: np.ndarray = np.array([0.01, 0.02, 0.05, 0.10]),
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Compute full selective metrics suite (Phase 2.3).
    
    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        coverages: Coverage thresholds for Risk@Coverage
        risks: Risk thresholds for Coverage@Risk
        device: Device (cpu/cuda)
    
    Returns:
        Dict with Risk@Coverage(c) and Coverage@Risk(r) for all thresholds
    """
    # Get max probabilities and predictions
    max_probs = probs.max(dim=1)[0].cpu().numpy()
    predicted = probs.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    metrics = {}
    
    # Risk@Coverage(c): Find minimum risk for given coverage
    for coverage in coverages:
        # Find threshold that gives this coverage
        threshold = np.percentile(max_probs, (1.0 - coverage) * 100)
        
        # Exit mask
        exit_mask = max_probs >= threshold
        
        if exit_mask.sum() > 0:
            exit_labels = predicted[exit_mask]
            true_labels = labels_np[exit_mask]
            risk = (exit_labels != true_labels).sum() / exit_mask.sum()
        else:
            risk = 1.0  # Worst case
        
        metrics[f'risk_at_coverage_{int(coverage*100)}'] = float(risk)
    
    # Coverage@Risk(r): Find maximum coverage for given risk
    for risk in risks:
        # Find threshold that gives this risk
        # Binary search for threshold
        thresholds = np.sort(max_probs)[::-1]  # Descending
        
        best_coverage = 0.0
        for threshold in thresholds:
            exit_mask = max_probs >= threshold
            
            if exit_mask.sum() > 0:
                exit_labels = predicted[exit_mask]
                true_labels = labels_np[exit_mask]
                current_risk = (exit_labels != true_labels).sum() / exit_mask.sum()
                
                if current_risk <= risk:
                    best_coverage = exit_mask.sum() / len(labels_np)
                else:
                    break
            else:
                break
        
        metrics[f'coverage_at_risk_{int(risk*100)}'] = float(best_coverage)
    
    return metrics
