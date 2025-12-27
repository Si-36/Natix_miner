"""
Phase 1.8: Threshold Sweep on val_calib ONLY (Dec 2025 Best Practice)

CRITICAL: Uses val_calib indices ONLY, NOT val_select or val_test.
Preserves exact logic from train_stage1_head.py threshold sweep.
"""

import torch
import json
import argparse
import os
from pathlib import Path
from typing import Dict

from config import Stage1ProConfig
from model.backbone import DINOv3Backbone
from model.head import Stage1Head


def threshold_sweep_val_calib(
    model_path: str,
    val_logits_path: str,
    val_labels_path: str,
    config_path: str,
    output_dir: str,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Perform threshold sweep on val_calib ONLY (Phase 1.8).
    
    CRITICAL FIX: val_logits.pt/val_labels_pt are ALREADY indexed by val_calib_indices
    (from val_calib_loader via IndexedDataset).
    NO double-indexing - use tensors directly!
    
    Args:
        model_path: Path to model checkpoint
        val_logits_path: Path to validation logits
        val_labels_path: Path to validation labels
        config_path: Path to config.json
        output_dir: Output directory
        verbose: Print status messages
    
    Returns:
        Best threshold and metrics
    """
    # Load config
    config = Stage1ProConfig.load(config_path)
    
    # Load validation logits/labels
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 1.8: THRESHOLD SWEEP ON VAL_CALIB")
        print(f"{'='*80}")
        print(f"Loading validation data...")
    
    all_logits = torch.load(val_logits_path, map_location="cpu")
    all_labels = torch.load(val_labels_path, map_location="cpu")
    
    if verbose:
        print(f"   Total logits: {all_logits.shape[0]}")
        print(f"   Total labels: {all_labels.shape[0]}")
    
    # CRITICAL FIX: val_logits.pt/val_labels_pt are ALREADY indexed by val_calib_indices
    # (from val_calib_loader via IndexedDataset in trainer.py)
    # NO double-indexing - use tensors directly!
    val_calib_logits = all_logits  # NOT: all_logits[val_calib_indices]
    val_calib_labels = all_labels  # NOT: all_labels[val_calib_indices]
    
    if verbose:
        print(f"   Using val_calib tensors directly (already indexed)")
        print(f"   CRITICAL: NO double-indexing!")
    
    # Compute probabilities
    val_calib_probs = torch.softmax(val_calib_logits, dim=1)
    
    # Phase 1.8 FIX: Sweep thresholds [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95]
    # Find threshold maximizing coverage with FNR ≤ 2%
    
    sweep_results = []
    
    # Sweep thresholds
    for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95]:
        # Get max probabilities
        max_probs, predicted = val_calib_probs.max(dim=1)
        
        # Exit condition: max_prob >= threshold OR max_prob <= (1.0 - threshold)
        # Example: threshold=0.88 → max_prob >= 0.88 OR max_prob <= 0.12
        exit_mask = (max_probs >= threshold) | (max_probs <= (1.0 - threshold))
        
        if exit_mask.sum() > 0:
            # Metrics on exited samples
            exit_correct = predicted[exit_mask].eq(val_calib_labels[exit_mask]).sum().item()
            exit_total = exit_mask.sum().item()
            exit_accuracy = exit_correct / exit_total
            
            # CRITICAL FIX: FNR (False Negative Rate) for POSITIVE CLASS ONLY
            # Binary classification: 0=negative, 1=positive
            # FNR = FN / (FN + TP) on positive samples only
            exit_true_labels = val_calib_labels[exit_mask]
            exit_predicted = predicted[exit_mask]
            
            # Count false negatives (predicted=0, true=1)
            false_negatives = ((exit_predicted == 0) & (exit_true_labels == 1)).sum().item()
            total_positive_exited = (exit_true_labels == 1).sum().item()
            
            # FNR = false_negatives / total_positive_exited
            fnr_on_exited = false_negatives / total_positive_exited if total_positive_exited > 0 else 1.0
            
            # Coverage
            coverage = exit_total / len(val_calib_labels)
        else:
            # No samples exit
            exit_accuracy = 0.0
            fnr_on_exited = 1.0  # Worst case
            coverage = 0.0
        
        sweep_results.append({
            'threshold': threshold,
            'coverage': float(coverage),
            'exit_accuracy': float(exit_accuracy),
            'fnr_on_exited': float(fnr_on_exited)
        })
    
    # Find threshold satisfying FNR constraint and maximizing coverage
    # Filter thresholds that satisfy FNR ≤ target_fnr_exit
    valid_results = [r for r in sweep_results if r['fnr_on_exited'] <= config.target_fnr_exit]
    
    if len(valid_results) > 0:
        # Choose threshold maximizing coverage among valid ones
        best_result = max(valid_results, key=lambda x: x['coverage'])
    else:
        # Fallback: use default 0.88 if no threshold satisfies constraint
        best_result = {
            'threshold': 0.88,
            'coverage': 0.0,
            'exit_accuracy': 0.0,
            'fnr_on_exited': 1.0
        }
    
    if verbose:
        print(f"\nThreshold Sweep Results (Phase 1.8):")
        print(f"   Evaluated {len(sweep_results)} thresholds")
        print(f"   Best Threshold: {best_result['threshold']}")
        print(f"   Coverage: {best_result['coverage']:.4f}")
        print(f"   Exit Accuracy: {best_result['exit_accuracy']:.4f}")
        print(f"   FNR on Exited: {best_result['fnr_on_exited']:.4f}")
        
        # Check if FNR constraint satisfied
        if best_result['fnr_on_exited'] <= config.target_fnr_exit:
            print(f"   ✅ FNR constraint satisfied (≤ {config.target_fnr_exit})")
        else:
            print(f"   ⚠️  FNR constraint NOT satisfied (>{config.target_fnr_exit})")
            print(f"   Using best threshold that maximizes coverage")
    
    return best_result


def save_thresholds_json(
    metrics: Dict[str, float],
    output_dir: str,
    config_path: str,
    verbose: bool = True
):
    """
    Save thresholds.json artifact (Phase 1.9).
    
    CRITICAL FIX: Store the ACTUAL chosen threshold, NOT hardcoded values!
    
    Contains:
    - exit_upper_threshold (chosen from sweep, e.g., 0.92)
    - exit_lower_threshold (1.0 - chosen, e.g., 0.08)
    - fnr_on_exited (≤ 0.02)
    - coverage
    - exit_accuracy
    
    Args:
        metrics: Threshold metrics (MUST contain 'threshold' key)
        output_dir: Output directory
        config_path: Path to config.json
        verbose: Print status messages
    """
    # Load config
    config = Stage1ProConfig.load(config_path)
    
    # CRITICAL FIX: Use ACTUAL chosen threshold from sweep
    # NOT hardcoded 0.88/0.12!
    chosen_threshold = metrics['threshold']
    exit_upper_threshold = chosen_threshold
    exit_lower_threshold = 1.0 - chosen_threshold
    
    # Build thresholds.json (preserve exact format)
    thresholds = {
        'exit_upper_threshold': exit_upper_threshold,
        'exit_lower_threshold': exit_lower_threshold,
        'fnr_on_exited': metrics['fnr_on_exited'],
        'coverage': metrics['coverage'],
        'exit_accuracy': metrics['exit_accuracy'],
        'target_fnr_exit': config.target_fnr_exit,
        'phase': 1,
        'exit_policy': 'softmax'
    }
    
    # Save to file
    thresholds_path = os.path.join(output_dir, "thresholds.json")
    
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    if verbose:
        print(f"\nPhase 1.9: Saved thresholds.json")
        print(f"   Path: {thresholds_path}")
        print(f"   Exit Upper Threshold: {exit_upper_threshold:.4f}")
        print(f"   Exit Lower Threshold: {exit_lower_threshold:.4f}")
        print(f"   FNR on Exited (Positive Class): {metrics['fnr_on_exited']:.4f}")
        print(f"   (FNR = False Negatives / Total Positives on exited)")
        print(f"   Coverage: {metrics['coverage']:.4f}")
        print(f"   Exit Accuracy: {metrics['exit_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1.8: Threshold Sweep on val_calib")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--val-logits-path", type=str, required=True, help="Path to val_logits.pt")
    parser.add_argument("--val-labels-path", type=str, required=True, help="Path to val_labels.pt")
    parser.add_argument("--config-path", type=str, required=True, help="Path to config.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print status messages")
    
    args = parser.parse_args()
    
    # Run threshold sweep
    metrics = threshold_sweep_val_calib(
        model_path=args.model_path,
        val_logits_path=args.val_logits_path,
        val_labels_path=args.val_labels_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # Save thresholds.json
    save_thresholds_json(
        metrics=metrics,
        output_dir=args.output_dir,
        config_path=args.config_path,
        verbose=args.verbose
    )
    
    if args.verbose:
        print(f"\n{'='*80}")
        print(f"✅ PHASE 1.8-1.9 COMPLETE")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
