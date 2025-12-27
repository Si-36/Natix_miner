"""
Exit metrics preserving exact logic from train_stage1_head.py

Compute exit coverage, exit accuracy, FNR_on_exited for all exit policies.
"""

import numpy as np
from typing import Optional
from .bootstrap import bootstrap_resample, compute_confidence_intervals


def compute_exit_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    exit_policy: str = "softmax",
    threshold: Optional[float] = None,
    gate_scores: Optional[np.ndarray] = None
) -> dict:
    """
    Compute exit metrics preserving logic from train_stage1_head.py.
    
    CRITICAL: Always compute and report FNR_on_exited and coverage for active policy.
    
    Args:
        probs: (N, num_classes) predicted probabilities
        labels: (N,) ground truth labels
        exit_policy: "softmax", "gate", or "scrc"
        threshold: Exit threshold (for softmax or gate)
        gate_scores: Gate scores [N] (for gate or scrc policies)
    
    Returns:
        Dictionary with exit metrics
    """
    if exit_policy == "softmax":
        # Phase 1: Softmax threshold exit
        if threshold is None:
            threshold = 0.88  # Default
        
        # Exit mask: high confidence in either direction
        exit_mask = (probs[:, 1] >= threshold) | (probs[:, 1] <= (1 - threshold))
        
    elif exit_policy == "gate":
        # Phase 3: Gate threshold exit
        if gate_scores is None or threshold is None:
            raise ValueError("gate_policy requires gate_scores and threshold")
        
        exit_mask = gate_scores >= threshold
        
    elif exit_policy == "scrc":
        # Phase 6: SCRC prediction sets
        # TODO: Implement SCRC exit logic (Phase 6)
        raise NotImplementedError("SCRC exit policy - Phase 6 only")
    
    else:
        raise ValueError(f"Unknown exit_policy: {exit_policy}")
    
    # Compute metrics
    coverage = exit_mask.mean()
    
    if exit_mask.sum() > 0:
        exit_labels = labels[exit_mask]
        exit_preds = np.argmax(probs[exit_mask], axis=1)
        exit_accuracy = (exit_preds == exit_labels).mean()
        
        # CRITICAL: FNR_on_exited (False Negative Rate on exited samples)
        # FNR = missed positives / total positives in exited samples
        positive_mask = exit_labels == 1
        if positive_mask.sum() > 0:
            fnr_on_exited = (exit_preds[positive_mask] == 0).mean()
        else:
            fnr_on_exited = 0.0
    else:
        exit_accuracy = 0.0
        fnr_on_exited = 0.0
    
    return {
        "exit_coverage": coverage,
        "exit_accuracy": exit_accuracy,
        "fnr_on_exited": fnr_on_exited
    }


def compute_exit_metrics_with_bootstrap(
    probs: np.ndarray,
    labels: np.ndarray,
    policy_dict: dict,
    active_policy: str = "softmax",
    target_fnr: float = 0.02,
    gate_scores: Optional[np.ndarray] = None,
    bootstrap_samples: int = 1000,
    confidence: float = 0.95
) -> dict:
    """
    Compute exit metrics with bootstrap confidence intervals.
    
    Args:
        probs: (N, num_classes) predicted probabilities
        labels: (N,) ground truth labels
        policy_dict: Policy parameters dictionary (from thresholds.json, gateparams.json, or scrcparams.json)
        active_policy: "softmax", "gate", or "scrc"
        target_fnr: Target FNR constraint
        gate_scores: Gate scores [N] (for gate/scrc policies)
        bootstrap_samples: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        Dictionary with metrics + CI bounds
    """
    # Extract threshold from policy_dict
    threshold = None
    if active_policy == "softmax":
        threshold = policy_dict.get('threshold', 0.88)
    elif active_policy == "gate":
        threshold = policy_dict.get('gate_threshold', 0.90)
    elif active_policy == "scrc":
        # SCRC uses lambda1 and lambda2 (Phase 6)
        threshold = policy_dict.get('lambda1', 0.90)
    
    metrics_samples = {
        "coverage": [],
        "exit_accuracy": [],
        "fnr_on_exited": []
    }
    
    for _ in range(bootstrap_samples):
        indices = bootstrap_resample(len(probs))
        metrics = compute_exit_metrics(
            probs[indices],
            labels[indices],
            active_policy,
            threshold,
            gate_scores[indices] if gate_scores is not None else None
        )
        metrics_samples["coverage"].append(metrics["exit_coverage"])
        metrics_samples["exit_accuracy"].append(metrics["exit_accuracy"])
        metrics_samples["fnr_on_exited"].append(metrics["fnr_on_exited"])
    
    result = {}
    for key, samples in metrics_samples.items():
        samples_array = np.array(samples)
        result[key] = {
            'mean': float(samples_array.mean()),
            'std': float(samples_array.std())
        }
        ci_lower, ci_upper = compute_confidence_intervals(
            samples_array.reshape(-1, 1), confidence
        )
        result[key]['ci_lower'] = float(ci_lower[0])
        result[key]['ci_upper'] = float(ci_upper[0])
    
    return result
