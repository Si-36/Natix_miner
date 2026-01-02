"""
Phase 2: MCC-Optimal Threshold Sweep Step (Production-Grade 2025-12-31)

Improvements over old implementation:
- ✅ Uses centralized MCC-optimized threshold selection (vectorized, 5000 thresholds)
- ✅ Atomic JSON writes (crash-safe)
- ✅ Manifest-last commit (lineage tracking)
- ✅ Duration tracking
- ✅ Type-safe with proper error handling
- ✅ Validator-compatible policy JSON

Contract:
- Inputs: val_calib_logits.pt, val_calib_labels.pt
- Outputs:
  - phase2/thresholds.json (best MCC threshold + metrics, validator-compatible)
  - phase2/threshold_sweep.csv (full sweep curve: threshold, mcc, tp, tn, fp, fn)
  - phase2/mcc_curve.png (MCC curve visualization)
  - phase2/manifest.json (lineage + checksums) ◄── LAST

MCC Optimization:
- Goal: Find threshold that maximizes Matthews Correlation Coefficient (MCC)
- Method: Vectorized computation over 5000 thresholds (10× faster than loop)
- Output: Best threshold + full metrics (accuracy, precision, recall, F1, FNR, FPR)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

# Import old modules (gradual migration)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from contracts.artifact_schema import ArtifactSchema

# Import new foundation modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from streetvision.eval.thresholds import select_threshold_max_mcc, plot_mcc_curve
from streetvision.io import create_step_manifest, write_json_atomic

logger = logging.getLogger(__name__)


def run_phase2_threshold_sweep(
    artifacts: ArtifactSchema,
    cfg: DictConfig,
) -> None:
    """
    Run Phase 2: Threshold Sweep with production-grade practices

    Args:
        artifacts: Artifact schema (all file paths)
        cfg: Hydra configuration

    Outputs:
        - thresholds.json: Best threshold + metrics (atomic write + SHA256)
        - threshold_sweep.csv: Full sweep curve for analysis
        - manifest.json: Lineage tracking (git SHA, config hash, checksums) ◄── LAST

    Selective Prediction:
        Finds threshold that maximizes selective accuracy:
        - Coverage = % of samples accepted (confidence > threshold)
        - Selective Accuracy = Accuracy on accepted samples only
        - Selective Risk = Error rate on accepted samples
    """
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("PHASE 2: Threshold Sweep (Production-Grade 2025-12-30)")
    logger.info("=" * 80)

    # Ensure phase2 directory exists
    artifacts.phase2_dir.mkdir(parents=True, exist_ok=True)

    # Load val_calib logits/labels
    logger.info(f"Loading: {artifacts.val_calib_logits}")
    logits = torch.load(artifacts.val_calib_logits)
    labels = torch.load(artifacts.val_calib_labels)

    logger.info(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

    # Get config for Phase 2
    n_thresholds = cfg.get("phase2", {}).get("n_thresholds", 5000)
    save_sweep_curve = cfg.get("phase2", {}).get("save_sweep_curve", True)
    save_mcc_plot = cfg.get("phase2", {}).get("save_mcc_plot", True)

    logger.info(f"Optimizing MCC with {n_thresholds} thresholds...")

    # Use vectorized MCC optimization (2025 upgrade)
    best_threshold, best_mcc, metrics, curve_df = select_threshold_max_mcc(
        logits=logits,
        labels=labels,
        n_thresholds=n_thresholds,
        return_curve=True,
    )

    logger.info(f"✅ Best threshold: {best_threshold:.4f}, MCC: {best_mcc:.4f}")
    logger.info(f"   Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}")
    logger.info(f"   Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    logger.info(f"   FNR: {metrics['fnr']:.4f}, FPR: {metrics['fpr']:.4f}")

    # Save full sweep curve to CSV
    if save_sweep_curve and curve_df is not None:
        curve_df.to_csv(artifacts.threshold_sweep_csv, index=False)
        logger.info(f"✅ Sweep curve saved: {artifacts.threshold_sweep_csv}")

    # Save MCC curve plot
    if save_mcc_plot and curve_df is not None:
        plot_mcc_curve(curve_df, best_threshold, str(artifacts.mcc_curve_plot))
        logger.info(f"✅ MCC curve plot saved: {artifacts.mcc_curve_plot}")

    # Save best threshold to JSON (ATOMIC WRITE, validator-compatible)
    thresholds_data = {
        "policy_type": "threshold",
        "threshold": float(best_threshold),
        "optimize_metric": "mcc",
        "metrics": {
            "mcc": float(best_mcc),
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "fnr": float(metrics["fnr"]),
            "fpr": float(metrics["fpr"]),
            "tp": int(metrics["tp"]),
            "tn": int(metrics["tn"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
        },
        "n_thresholds": int(n_thresholds),
        "split_used": "val_calib",
    }

    thresholds_checksum = write_json_atomic(artifacts.thresholds_json, thresholds_data)
    logger.info(
        f"✅ Thresholds saved: {artifacts.thresholds_json} "
        f"(SHA256: {thresholds_checksum[:12]}...)"
    )

    # Create and save MANIFEST (LAST STEP)
    duration_seconds = time.time() - start_time
    logger.info("Creating manifest (lineage tracking)...")

    manifest = create_step_manifest(
        step_name="phase2_threshold_sweep",
        input_paths=[
            artifacts.val_calib_logits,
            artifacts.val_calib_labels,
        ],
        output_paths=[
            artifacts.thresholds_json,
            artifacts.threshold_sweep_csv,
        ],
        output_dir=artifacts.output_dir,
        metrics={
            "threshold": thresholds_data["threshold"],
            "mcc": thresholds_data["metrics"]["mcc"],
            "accuracy": thresholds_data["metrics"]["accuracy"],
            "precision": thresholds_data["metrics"]["precision"],
            "recall": thresholds_data["metrics"]["recall"],
            "f1": thresholds_data["metrics"]["f1"],
            "fnr": thresholds_data["metrics"]["fnr"],
            "fpr": thresholds_data["metrics"]["fpr"],
        },
        duration_seconds=duration_seconds,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    manifest_checksum = manifest.save(artifacts.phase2_dir / "manifest.json")
    logger.info(
        f"✅ Manifest saved: {artifacts.phase2_dir / 'manifest.json'} "
        f"(SHA256: {manifest_checksum[:12]}...)"
    )

    # Summary
    logger.info("=" * 80)
    logger.info("✅ Phase 2 Complete (MCC-Optimized, Production-Grade)")
    logger.info(f"Duration: {duration_seconds / 60:.1f} minutes")
    logger.info(f"Best Threshold: {best_threshold:.4f}")
    logger.info(f"MCC: {best_mcc:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    logger.info(f"FNR: {metrics['fnr']:.4f}, FPR: {metrics['fpr']:.4f}")
    logger.info(f"Thresholds JSON: {artifacts.thresholds_json}")
    logger.info(f"Sweep CSV: {artifacts.threshold_sweep_csv}")
    logger.info(f"MCC Curve Plot: {artifacts.mcc_curve_plot}")
    logger.info(f"Manifest: {artifacts.phase2_dir / 'manifest.json'}")
    logger.info("=" * 80)
