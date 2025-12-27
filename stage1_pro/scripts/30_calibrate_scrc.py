#!/usr/bin/env python3
"""
Phase 2: SCRC-I calibration script.

Applies Dirichlet calibration to improve FNR constraint satisfaction.
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

import torch
import argparse
from pathlib import Path
import json
from stage1_pro.calibration.dirichlet import DirichletCalibrator
from scripts.threshold_sweep import threshold_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_probs", type=str, required=True)
    parser.add_argument("--val_labels", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="scrc_params.json")
    parser.add_argument("--thresholds", type=str, default=None)
    parser.add_argument("--target_fnr", type=float, default=0.02)
    args = parser.parse_args()

    print("Loading validation data...")
    val_probs = torch.load(args.val_probs)
    val_labels = torch.load(args.val_labels)

    # Convert probs to logits for calibration
    val_logits = torch.log(val_probs + 1e-8)

    print("Fitting Dirichlet calibrator...")
    calibrator = DirichletCalibrator(alpha=args.alpha)
    calibrator.fit(val_logits, val_labels)

    # Calibrate probabilities
    calibrated_probs = calibrator.predict(val_logits)

    # Run threshold sweep on calibrated probs
    print("Running threshold sweep on calibrated probabilities...")
    sweep_results = threshold_sweep(
        calibrated_probs,
        val_labels,
        thresholds=eval(args.thresholds) if args.thresholds else None,
        target_fnr=args.target_fnr,
    )

    # Save calibrator params
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calibrator.save(output_path)

    # Save sweep results
    results_path = output_path.parent / "scrc_sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\nSCRC Calibration Results:")
    print(f"  Best threshold: {sweep_results['best_threshold']:.4f}")
    print(f"  FNR on exits: {sweep_results['best_fnr_on_exits']:.4f}")
    print(f"  Coverage: {sweep_results['best_coverage']:.4f}")
    print(f"  Exit accuracy: {sweep_results['best_exit_accuracy']:.4f}")
    print(f"\nSaved SCRC params to {output_path}")
    print(f"Saved sweep results to {results_path}")


if __name__ == "__main__":
    main()
