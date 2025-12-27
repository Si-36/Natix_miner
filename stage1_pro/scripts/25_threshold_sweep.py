#!/usr/bin/env python3
"""
Phase 1: Threshold sweep matching cascade exit logic.
Find threshold satisfying FNR ≤ 2% and maximizing coverage.
Save thresholds.json (Phase 1 policy artifact).
"""
import sys
sys.path.insert(0, '/home/sina/projects/miner_b')

import torch
import numpy as np
import json
import argparse
from pathlib import Path

def threshold_sweep(
    val_probs: torch.Tensor,
    val_labels: torch.Tensor,
    thresholds: list = None,
    target_fnr: float = 0.02
) -> dict:
    """
    Perform threshold sweep matching cascade exit logic.
    
    Args:
        val_probs: Validation probabilities [N, 2]
        val_labels: Validation labels [N]
        thresholds: List of thresholds to evaluate. If None, use default sweep
        target_fnr: Target false negative rate (default 2%)
    
    Returns:
        Dictionary with sweep results and optimal threshold
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95]
    
    results = []
    
    for threshold in thresholds:
        max_probs, preds = val_probs.max(dim=-1)
        exit_mask = max_probs >= threshold
        
        coverage = exit_mask.float().mean().item()
        
        if exit_mask.sum() > 0:
            exit_labels = val_labels[exit_mask]
            exit_preds = preds[exit_mask]
            exit_accuracy = (exit_preds == exit_labels).float().mean().item()
            fnr_on_exits = ((exit_labels == 1) & (exit_preds == 0)).float().sum() / max((exit_labels == 1).sum(), 1)
        else:
            exit_accuracy = 0.0
            fnr_on_exits = 1.0
        
        results.append({
            'threshold': threshold,
            'coverage': coverage,
            'exit_accuracy': exit_accuracy,
            'fnr_on_exits': fnr_on_exits
        })
    
    # Find threshold satisfying FNR constraint and maximizing coverage
    valid_results = [r for r in results if r['fnr_on_exits'] <= target_fnr]
    
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['coverage'])
    else:
        print(f"WARNING: No threshold found satisfying FNR ≤ {target_fnr}")
        best_result = min(results, key=lambda x: abs(x['fnr_on_exits'] - target_fnr)
    
    return {
        'sweep_results': results,
        'best_threshold': best_result['threshold'],
        'best_fnr_on_exits': best_result['fnr_on_exits'],
        'best_coverage': best_result['coverage'],
        'best_exit_accuracy': best_result['exit_accuracy']
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_probs', type=str, required=True)
    parser.add_argument('--val_labels', type=str, required=True)
    parser.add_argument('--thresholds', type=str, default=None)
    parser.add_argument('--target_fnr', type=float, default=0.02)
    parser.add_argument('--output', type=str, default='thresholds.json')
    args = parser.parse_args()
    
    print("Loading validation data...")
    val_probs = torch.load(args.val_probs)
    val_labels = torch.load(args.val_labels)
    
    print(f"Running threshold sweep...")
    sweep_results = threshold_sweep(
        val_probs, val_labels, 
        thresholds=eval(args.thresholds) if args.thresholds else None,
        target_fnr=args.target_fnr
    )
    
    output_path = Path(args.output)
    output_dict = {
        'exit_threshold': sweep_results['best_threshold'],
        'fnr_on_exits': sweep_results['best_fnr_on_exits'],
        'coverage': sweep_results['best_coverage'],
        'exit_accuracy': sweep_results['best_exit_accuracy'],
        'sweep_results': sweep_results['sweep_results'],
        'target_fnr': args.target_fnr
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nThreshold sweep results:")
    print(f"  Best threshold: {sweep_results['best_threshold']:.4f}")
    print(f"  FNR on exits: {sweep_results['best_fnr_on_exits']:.4f}")
    print(f"  Coverage: {sweep_results['best_coverage']:.4f}")
    print(f"  Exit accuracy: {sweep_results['best_exit_accuracy']:.4f}")
    print(f"\nSaved thresholds.json to {output_path}")

if __name__ == '__main__':
    main()
