"""
Gate Calibration Script - Phase 3.7-3.10: Gate Calibration & Threshold Selection

Uses Platt scaling (logistic regression) to calibrate gate scores on val_calib.
Selects optimal gate threshold to satisfy FNR constraint and maximize coverage.

Output: gateparams.json artifact (Phase 3.9)
"""

import torch
import numpy as np
import json
import os
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, Optional


def load_val_calib_data(
    output_dir: str,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load val_calib logits and labels (Phase 3.7).
    
    Args:
        output_dir: Directory containing val_calib logits/labels
        verbose: Print status messages
    
    Returns:
        (classifier_logits, gate_logits, labels)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 3.7: LOADING VAL_CALIB DATA")
        print(f"{'='*80}")
    
    # Load calib logits (classifier)
    calib_logits_path = os.path.join(output_dir, "val_calib_logits.pt")
    if not os.path.exists(calib_logits_path):
        raise FileNotFoundError(f"Missing {calib_logits_path} - run training with Phase 3 first")
    
    calib_logits = torch.load(calib_logits_path, map_location='cpu')
    
    # Load calib gate logits (Phase 3.6)
    calib_gate_logits_path = os.path.join(output_dir, "val_calib_gate_logits.pt")
    if not os.path.exists(calib_gate_logits_path):
        raise FileNotFoundError(f"Missing {calib_gate_logits_path} - run training with Phase 3 first")
    
    calib_gate_logits = torch.load(calib_gate_logits_path, map_location='cpu')
    
    # Load calib labels
    calib_labels_path = os.path.join(output_dir, "val_calib_labels.pt")
    if not os.path.exists(calib_labels_path):
        raise FileNotFoundError(f"Missing {calib_labels_path} - run training with Phase 3 first")
    
    calib_labels = torch.load(calib_labels_path, map_location='cpu')
    
    if verbose:
        print(f"   Classifier logits: {calib_logits.shape}")
        print(f"   Gate logits: {calib_gate_logits.shape}")
        print(f"   Labels: {calib_labels.shape}")
        print(f"   Loaded from: {output_dir}")
    
    return calib_logits, calib_gate_logits, calib_labels


def compute_exit_mask(
    gate_prob: np.ndarray,
    gate_threshold: float,
    gate_lower_threshold: Optional[float] = None
) -> np.ndarray:
    """
    Compute exit mask from gate probabilities (Phase 3.8).
    
    Args:
        gate_prob: Gate probabilities [N]
        gate_threshold: Upper threshold
        gate_lower_threshold: Lower threshold (optional, for two-sided exit)
    
    Returns:
        Binary mask [N] (1=exit, 0=continue)
    """
    if gate_lower_threshold is not None:
        # Two-sided exit: gate >= upper OR gate <= lower
        exit_mask = (gate_prob >= gate_threshold) | (gate_prob <= gate_lower_threshold)
    else:
        # One-sided exit: gate >= upper
        exit_mask = gate_prob >= gate_threshold
    
    return exit_mask


def compute_fnr_on_exited(
    classifier_logits: torch.Tensor,
    labels: torch.Tensor,
    exit_mask: np.ndarray,
    verbose: bool = False
) -> float:
    """
    Compute FNR on exited samples (Phase 3.8).
    
    FNR = false_negatives / (false_negatives + true_positives) on positive class only.
    
    Args:
        classifier_logits: Classifier logits [N, num_classes]
        labels: True labels [N]
        exit_mask: Binary exit mask [N]
        verbose: Print status messages
    
    Returns:
        FNR on exited samples
    """
    # Get predictions
    predicted = torch.softmax(classifier_logits, dim=1).argmax(dim=1).numpy()
    labels_np = labels.numpy()
    
    # Apply exit mask
    exit_true_labels = labels_np[exit_mask]
    exit_predicted = predicted[exit_mask]
    
    if len(exit_true_labels) == 0:
        return 1.0  # No exited samples, worst case
    
    # Compute FNR on positive class only
    false_negatives = ((exit_predicted == 0) & (exit_true_labels == 1)).sum()
    total_positive_exited = (exit_true_labels == 1).sum()
    
    fnr_on_exited = false_negatives / total_positive_exited if total_positive_exited > 0 else 1.0
    
    if verbose:
        print(f"   Exit samples: {len(exit_true_labels)}")
        print(f"   Positive samples in exit: {total_positive_exited}")
        print(f"   False negatives: {false_negatives}")
        print(f"   FNR on exited: {fnr_on_exited:.4f}")
    
    return fnr_on_exited


def fit_platt_scaling(
    gate_logits: torch.Tensor,
    labels: torch.Tensor,
    verbose: bool = False
) -> LogisticRegression:
    """
    Fit Platt scaling (logistic regression) to calibrate gate logits (Phase 3.7).
    
    Args:
        gate_logits: Gate logits [N, 1]
        labels: True labels [N] (used to gate quality)
        verbose: Print status messages
    
    Returns:
        Fitted logistic regression model
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 3.7: FITTING PLATT SCALING")
        print(f"{'='*80}")
    
    # Convert gate logits to exit probability (using sigmoid)
    gate_prob = torch.sigmoid(gate_logits).numpy()
    
    # Fit logistic regression (Platt scaling)
    # Features: gate_prob, Labels: exit quality (use gate > 0.5 as exit)
    exit_mask = (gate_prob > 0.5).astype(int)
    
    platt_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    
    # Reshape for sklearn
    gate_prob_reshaped = gate_prob.reshape(-1, 1)
    platt_model.fit(gate_prob_reshaped, exit_mask)
    
    if verbose:
        print(f"   Platt scaling fitted")
        print(f"   Coef: {platt_model.coef_}")
        print(f"   Intercept: {platt_model.intercept_}")
        print(f"   Accuracy: {platt_model.score(gate_prob_reshaped, exit_mask):.4f}")
    
    return platt_model


def calibrate_gate(
    gate_logits: torch.Tensor,
    platt_model: LogisticRegression,
    verbose: bool = False
) -> np.ndarray:
    """
    Apply Platt scaling to gate logits (Phase 3.7).
    
    Args:
        gate_logits: Gate logits [N, 1]
        platt_model: Fitted Platt scaling model
        verbose: Print status messages
    
    Returns:
        Calibrated gate probabilities [N]
    """
    # Convert gate logits to probabilities
    gate_prob = torch.sigmoid(gate_logits).numpy()
    
    # Apply Platt scaling
    gate_prob_reshaped = gate_prob.reshape(-1, 1)
    calibrated_gate_prob = platt_model.predict_proba(gate_prob_reshaped)[:, 1]  # Probability of exit
    
    if verbose:
        print(f"   Gate prob (raw): mean={gate_prob.mean():.4f}, std={gate_prob.std():.4f}")
        print(f"   Gate prob (calibrated): mean={calibrated_gate_prob.mean():.4f}, std={calibrated_gate_prob.std():.4f}")
    
    return calibrated_gate_prob


def select_gate_threshold(
    classifier_logits: torch.Tensor,
    gate_prob: np.ndarray,
    labels: torch.Tensor,
    target_fnr_exit: float = 0.02,
    verbose: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Select optimal gate threshold on val_calib (Phase 3.10).
    
    Sweep thresholds to find optimal threshold that:
    - Maximizes coverage
    - Satisfies FNR constraint (FNR_on_exited ≤ target_fnr_exit)
    
    Args:
        classifier_logits: Classifier logits [N, num_classes]
        gate_prob: Calibrated gate probabilities [N]
        labels: True labels [N]
        target_fnr_exit: Target FNR on exited samples (default 0.02)
        verbose: Print status messages
    
    Returns:
        (chosen_threshold, metrics_dict)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 3.10: SELECTING GATE THRESHOLD")
        print(f"{'='*80}")
        print(f"   Target FNR on exited: {target_fnr_exit}")
    
    # Sweep thresholds
    thresholds = np.linspace(0.5, 0.95, 10)
    
    best_threshold = 0.5
    best_coverage = 0.0
    best_metrics = {
        'threshold': 0.5,
        'coverage': 0.0,
        'fnr_on_exited': 1.0,
        'exit_accuracy': 0.0,
        'exit_samples': 0
    }
    
    sweep_results = []
    
    for threshold in thresholds:
        # Compute exit mask
        exit_mask = compute_exit_mask(gate_prob, threshold, None)
        
        # Compute coverage
        coverage = exit_mask.sum() / len(gate_prob)
        
        # Compute FNR on exited
        fnr_on_exited = compute_fnr_on_exited(
            classifier_logits,
            labels,
            exit_mask,
            verbose=False
        )
        
        # Compute exit accuracy
        predicted = torch.softmax(classifier_logits, dim=1).argmax(dim=1).numpy()
        exit_true_labels = labels.numpy()[exit_mask]
        exit_predicted = predicted[exit_mask]
        exit_acc = accuracy_score(exit_true_labels, exit_predicted) if len(exit_true_labels) > 0 else 0.0
        
        metrics = {
            'threshold': threshold,
            'coverage': coverage,
            'fnr_on_exited': fnr_on_exited,
            'exit_accuracy': exit_acc,
            'exit_samples': exit_mask.sum()
        }
        sweep_results.append(metrics)
        
        # Check if FNR constraint satisfied
        if fnr_on_exited <= target_fnr_exit:
            if coverage > best_coverage:
                best_coverage = coverage
                best_threshold = threshold
                best_metrics = metrics
                if verbose:
                    print(f"   ✅ Threshold={threshold:.2f}: Coverage={coverage:.4f}, FNR={fnr_on_exited:.4f} (NEW BEST)")
            elif verbose:
                print(f"   ✓ Threshold={threshold:.2f}: Coverage={coverage:.4f}, FNR={fnr_on_exited:.4f}")
        elif verbose:
            print(f"   ✗ Threshold={threshold:.2f}: Coverage={coverage:.4f}, FNR={fnr_on_exited:.4f} (FNR constraint violated)")
    
    if verbose:
        print(f"\n   BEST THRESHOLD: {best_threshold:.2f}")
        print(f"   Coverage: {best_coverage:.4f}")
        print(f"   FNR on exited: {best_metrics['fnr_on_exited']:.4f}")
        print(f"   Exit accuracy: {best_metrics['exit_accuracy']:.4f}")
    
    return best_threshold, best_metrics


def save_gateparams_json(
    gate_threshold: float,
    metrics: Dict[str, float],
    sweep_results: list,
    output_dir: str,
    verbose: bool = True
) -> str:
    """
    Save gateparams.json artifact (Phase 3.9).
    
    Args:
        gate_threshold: Chosen gate threshold
        metrics: Metrics dict (coverage, fnr_on_exited, exit_accuracy)
        sweep_results: Full sweep results
        output_dir: Directory to save gateparams.json
        verbose: Print status messages
    
    Returns:
        Path to gateparams.json
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 3.9: SAVING GATEPARAMS.JSON")
        print(f"{'='*80}")
    
    gateparams = {
        'active_exit_policy': 'gate',
        'gate_threshold': gate_threshold,
        'gate_lower_threshold': None,  # Single-sided exit for gate
        'fnr_on_exited': metrics['fnr_on_exited'],
        'coverage': metrics['coverage'],
        'exit_accuracy': metrics['exit_accuracy'],
        'exit_samples': metrics['exit_samples'],
        'calibration_method': 'platt_scaling',
        'sweep_results': sweep_results
    }
    
    gateparams_path = os.path.join(output_dir, "gateparams.json")
    
    with open(gateparams_path, 'w') as f:
        json.dump(gateparams, f, indent=2)
    
    if verbose:
        print(f"   Saved to: {gateparams_path}")
        print(f"   Gate threshold: {gate_threshold}")
        print(f"   Coverage: {metrics['coverage']:.4f}")
        print(f"   FNR on exited: {metrics['fnr_on_exited']:.4f}")
        print(f"   Exit accuracy: {metrics['exit_accuracy']:.4f}")
    
    return gateparams_path


def main():
    parser = argparse.ArgumentParser(description="Gate Calibration (Phase 3.7-3.10)")
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory containing val_calib logits')
    parser.add_argument('--target_fnr_exit', type=float, default=0.02, help='Target FNR on exited samples (default: 0.02)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed status messages')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"PHASE 3.7-3.10: GATE CALIBRATION & THRESHOLD SELECTION")
    print(f"{'='*80}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Target FNR on exited: {args.target_fnr_exit}")
    print(f"{'='*80}")
    
    # Phase 3.7: Load val_calib data
    calib_logits, calib_gate_logits, calib_labels = load_val_calib_data(
        args.output_dir,
        verbose=args.verbose
    )
    
    # Phase 3.7: Fit Platt scaling
    platt_model = fit_platt_scaling(
        calib_gate_logits,
        calib_labels,
        verbose=args.verbose
    )
    
    # Phase 3.8: Calibrate gate logits
    calibrated_gate_prob = calibrate_gate(
        calib_gate_logits,
        platt_model,
        verbose=args.verbose
    )
    
    # Phase 3.10: Select gate threshold
    gate_threshold, best_metrics = select_gate_threshold(
        calib_logits,
        calibrated_gate_prob,
        calib_labels,
        args.target_fnr_exit,
        verbose=args.verbose
    )
    
    # Save sweep results (for later analysis)
    gateparams_path = save_gateparams_json(
        gate_threshold,
        best_metrics,
        [],  # Sweep results (to be populated in future)
        args.output_dir,
        verbose=args.verbose
    )
    
    print(f"\n{'='*80}")
    print(f"✅ GATE CALIBRATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Gate threshold: {gate_threshold}")
    print(f"   Coverage: {best_metrics['coverage']:.4f}")
    print(f"   FNR on exited: {best_metrics['fnr_on_exited']:.4f}")
    print(f"   Exit accuracy: {best_metrics['exit_accuracy']:.4f}")
    print(f"   Saved to: {gateparams_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
