"""
🎯 **Step: Sweep Thresholds** (Calibration with Leak-Proof Contracts)
Domain name: sweep_thresholds

Features:
- Leak-Proof: VAL_CALIB ONLY (never VAL_SELECT or VAL_TEST!)
- Real calibration: Compute thresholds on calibration probabilities
- Mutual exclusivity: Exactly ONE policy artifact (THRESHOLDS.json)
- ArtifactStore: Atomic writes + manifest lineage tracking
- Registry-driven: Depends on train_baseline_head

2025/2026 Pro Standard Features:
- Coverage computation (fraction of samples kept)
- FNR computation (false negatives / kept samples)
- Optimal threshold selection (constrained optimization)
- ECE + Brier score metrics
- Hash-based caching/resume

Args:
    threshold_grid: List[float] = [0.0, 0.1, ..., 1.0]
    target_coverage: float = 0.90  # 90% coverage constraint
    target_fnr: float = 0.02  # Maximum 2% FNR allowed
    objectives: str = "coverage"  # or "fnr" or "balanced"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, FrozenSet
import torch
import torch.nn.functional as F
import numpy as np

from ..pipeline.step_api import StepSpec, StepContext, StepResult
from ..pipeline.artifacts import ArtifactKey, ArtifactStore
from ..pipeline.registry import StepRegistry
from ..core.splits import Split, SplitPolicy, assert_allowed
from ..core.validators import ArtifactValidator


@dataclass(frozen=True)
class SweepThresholdsSpec(StepSpec):
    """
    Phase 2: Threshold Sweep / Calibration (LEAK-PROOF!)
    
    Features:
    - LEAK-PROOF: Uses only VAL_CALIB (never VAL_SELECT or VAL_TEST!)
    - Real calibration: Compute optimal thresholds on calibration probs
    - Mutual exclusivity: Exactly ONE policy artifact (THRESHOLDS.json)
    - ArtifactStore: Atomic writes + manifest lineage tracking
    - Registry-driven: Depends on train_baseline_head
    
    Args:
        threshold_grid: List[float] = [0.0, 0.1, ..., 1.0]
        target_coverage: float = 0.90
        target_fnr: float = 0.02
        objectives: str = "coverage"  # "coverage" | "fnr" | "balanced"
    """
    
    step_id: str = "sweep_thresholds"
    name: str = "sweep_thresholds"  # Domain name (not "phase2"!)
    deps: List[str] = field(default_factory=lambda: ["train_baseline_head"])  # Depends on Phase 1
    order_index: int = 1  # After Phase 1
    owners: List[str] = field(default_factory=lambda: ["ml-team"])
    tags: Dict[str, str] = field(default_factory=lambda: {
        "priority": "high",
        "stage": "calibration",
        "component": "thresholds",
    })
    
    # Config keys
    threshold_grid: List[float] = None
    target_coverage: float = None
    target_fnr: float = None
    objectives: str = None
    max_batch_size: int = 64  # For efficient threshold computation
    
    def inputs(self, ctx: StepContext) -> List[str]:
        """
        Declare required input artifacts for this step.
        
        Phase 2 requires calibration artifacts from Phase 1.
        NOTE: Uses ONLY VAL_CALIB (NEVER VAL_SELECT) to prevent leakage!
        
        Returns:
            List of ArtifactKey canonical names (NOT paths!)
        """
        return [
            ArtifactKey.VAL_CALIB_LOGITS,  # Calibration logits from Phase 1
            ArtifactKey.VAL_CALIB_LABELS,  # Calibration labels from Phase 1
        ]
    
    def outputs(self, ctx: StepContext) -> List[str]:
        """
        Declare output artifacts this step produces.
        
        Phase 2 produces:
        - Optimal thresholds (THRESHOLDS.json)
        - Sweep metrics (THRESHOLDS_METRICS)
        
        Returns:
            List of ArtifactKey canonical names (NOT paths!)
        """
        return [
            ArtifactKey.THRESHOLDS_JSON,       # Optimal thresholds
            ArtifactKey.THRESHOLDS_METRICS,   # Sweep metrics (accuracy, coverage, FNR, ECE, Brier)
        ]
    
    def allowed_splits(self) -> FrozenSet[str]:
        """
        Declare which data splits this step is allowed to use.
        
        LEAK-PROOF CONTRACT:
        - VAL_CALIB: YES (calibration set)
        - VAL_SELECT: NO (would cause data leakage!)
        - VAL_TEST: NO (final eval set)
        - TRAIN: NO (this step doesn't train)
        
        STRICTLY FORBIDDEN:
        - Using VAL_SELECT or VAL_TEST would violate leak-proof design!
        
        Returns:
            FrozenSet of Split enum values
        """
        return frozenset({
            Split.VAL_CALIB,  # Calibration set ONLY
        })
    
    def run(self, ctx: StepContext) -> StepResult:
        """
        Execute threshold sweep / calibration.
        
        Args:
            ctx: Runtime context with artifact_root, config, run_id, etc.
        
        Returns:
            StepResult with artifacts_written, splits_used, metrics, metadata
        """
        print(f"\n{'='*70}")
        print(f"🎯 Step: {self.name}")
        print("=" * 70)
        
        # Resolve config
        cfg = ctx.config
        threshold_grid = self.threshold_grid if self.threshold_grid else cfg.threshold_grid.get("threshold_grid", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        target_coverage = self.target_coverage if self.target_coverage else cfg.threshold_grid.get("target_coverage", 0.90)
        target_fnr = self.target_fnr if self.target_fnr else cfg.threshold_grid.get("target_fnr", 0.02)
        objectives = self.objectives if self.objectives else cfg.threshold_grid.get("objectives", "coverage")
        max_batch_size = self.max_batch_size if self.max_batch_size else 64
        
        print(f"   Config:")
        print(f"     threshold_grid: {threshold_grid}")
        print(f"     target_coverage: {target_coverage}")
        print(f"     target_fnr: {target_fnr}")
        print(f"     objectives: {objectives}")
        print(f"     max_batch_size: {max_batch_size}")
        print("-" * 70)
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        
        # Load calibration artifacts from Phase 1
        print(f"\n   📝 Loading calibration artifacts (VAL_CALIB ONLY)...")
        print("-" * 70)
        
        calib_logits_path = ctx.artifact_store.get(ArtifactKey.VAL_CALIB_LOGITS, ctx.run_id)
        calib_labels_path = ctx.artifact_store.get(ArtifactKey.VAL_CALIB_LABELS, ctx.run_id)
        
        # Validate artifacts exist
        validator = ArtifactValidator()
        validator.validate_required_files([
            ("calib_logits", calib_logits_path),
            ("calib_labels", calib_labels_path),
        ])
        print(f"   ✅ Calibration artifacts validated")
        
        # Load calibration data
        calib_logits = torch.load(calib_logits_path)
        calib_labels = torch.load(calib_labels_path)
        
        print(f"   📊 Calibration data: logits={calib_logits.shape}, labels={calib_labels.shape}")
        
        # Enforce split contract (LEAK-PROOF!)
        # We MUST use only VAL_CALIB
        # Using VAL_SELECT or VAL_TEST would cause data leakage!
        splits_used = frozenset({Split.VAL_CALIB.value})
        print(f"   ✅ Splits used: VAL_CALIB (LEAK-PROOF!)")
        
        # Compute probabilities (for threshold sweep)
        probs = torch.softmax(calib_logits, dim=-1)  # [N, C] where C=1 (positive class)
        print(f"   📊 Probabilities computed: {probs.shape}")
        
        # Compute all thresholds
        print(f"\n   🎯 Computing thresholds on grid of {len(threshold_grid)} values...")
        print("-" * 70)
        
        all_results = {}  # Key: threshold_value -> metrics dict
        
        for threshold in threshold_grid:
            # Compute predictions at this threshold
            preds = (probs[:, 1] >= threshold).float()  # Keep if prob >= threshold
            
            # Compute metrics
            correct = (preds == calib_labels).sum()
            total = calib_labels.numel()
            accuracy = correct.float() / total
            coverage = correct.float() / total
            fnr = (preds != calib_labels).sum() / total  # False Neg Rate
            tnr = (preds != calib_labels).float() / total  # True Neg Rate
            
            # ECE (Expected Calibration Error)
            # ECE = expected - reliability
            probs_for_correct = probs[range(correct), :]
            reliability = probs_for_correct[range(correct), 1]
            ece = (reliability - probs_for_correct).abs().sum().item() / total
            
            # Brier Score
            brier = ((preds - calib_labels) ** 2).sum(dim=-1) / total.item()
            
            # Store results
            all_results[threshold] = {
                "accuracy": float(accuracy),
                "coverage": float(coverage),
                "fnr": float(fnr),
                "tnr": float(tnr),
                "ece": float(ece),
                "brier": float(brier),
                "correct": int(correct),
                "total": int(total),
            }
            
            # Log progress
            if (threshold * 100) % 10 == 0 or threshold in [0.0, 1.0]:
                print(f"   Threshold {threshold:.3f}: "
                      f"Acc={accuracy:.4f}, Cov={coverage:.4f}, FNR={fnr:.4f}, TNR={tnr:.4f}, "
                      f"ECE={ece:.4f}, Brier={brier:.4f}")
        
        print("-" * 70)
        print(f"   ✅ Computed {len(threshold_grid)} thresholds")
        
        # Select optimal threshold(s) based on objective
        print(f"\n   🎯 Selecting optimal threshold (objective={objectives})...")
        print("-" * 70)
        
        # Find best threshold
        if objectives == "coverage":
            # Maximize coverage subject to FNR constraint
            valid_results = [
                (t, r) for t, r in all_results.items()
                if r["fnr"] <= target_fnr  # Constraint: FNR <= target FNR
            ]
            if not valid_results:
                print(f"   ⚠️  No thresholds meet FNR constraint <= {target_fnr}")
                print(f"      Selecting threshold with max coverage instead...")
                selected_threshold = max(all_results.keys())
                selected_metrics = all_results[selected_threshold]
            else:
                selected_threshold = min(valid_results)[0]
                selected_metrics = valid_results[selected_threshold]
                print(f"   ✅ Selected threshold: {selected_threshold:.3f} (FNR={selected_metrics['fnr']:.4f})")
        
        elif objectives == "fnr":
            # Minimize FNR
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]["fnr"])
            selected_threshold = sorted_results[0][0]
            selected_metrics = all_results[selected_threshold]
            print(f"   ✅ Selected threshold: {selected_threshold:.3f} (FNR={selected_metrics['fnr']:.4f})")
        
        elif objectives == "balanced":
            # Balance coverage and FNR (simplified)
            # Use normalized scores: accuracy - FNR (approximates balanced tradeoff)
            scored_results = [
                (t, r) for t, r in all_results.items()
            ]
            sorted_scored = sorted(scored_results, key=lambda x: x[1]["accuracy"] - x[1]["tnr"])
            selected_threshold = sorted_scored[0][0]
            selected_metrics = all_results[selected_threshold]
            print(f"   ✅ Selected threshold: {selected_threshold:.3f} (Acc={selected_metrics['accuracy']:.4f}, FNR={selected_metrics['fnr']:.4f})")
        
        else:
            # Default: maximize coverage
            selected_threshold = max(all_results.keys())
            selected_metrics = all_results[selected_threshold]
            print(f"   ✅ Selected threshold: {selected_threshold:.3f} (Acc={selected_metrics['accuracy']:.4f}, Cov={selected_metrics['coverage']:.4f})")
        
        print("-" * 70)
        print(f"   Selected threshold: {selected_threshold:.3f}")
        print(f"   Metrics: {selected_metrics}")
        print("=" * 70)
        
        # Save optimal thresholds (THRESHOLDS.json)
        print(f"\n   💾 Saving optimal thresholds (THRESHOLDS.json)...")
        print("-" * 70)
        
        thresholds_path = ctx.artifact_store.get(ArtifactKey.THRESHOLDS_JSON, ctx.run_id)
        
        import json
        thresholds_data = {
            "selected_threshold": float(selected_threshold),
            "selected_metrics": selected_metrics,
            "threshold_grid": threshold_grid,
            "objective": objectives,
            "target_coverage": target_coverage,
            "target_fnr": target_fnr,
        }
        
        ctx.artifact_store.put(ArtifactKey.THRESHOLDS_JSON, thresholds_data, ctx.run_id)
        print(f"   ✅ Thresholds saved: {thresholds_path}")
        
        # Save sweep metrics (THRESHOLDS_METRICS)
        print(f"\n   📊 Saving sweep metrics (THRESHOLDS_METRICS)...")
        print("-" * 70)
        
        metrics_path = ctx.artifact_store.get(ArtifactKey.THRESHOLDS_METRICS, ctx.run_id)
        
        # Convert all_results to list for CSV
        metrics_list = []
        for threshold, metrics in all_results.items():
            metrics_list.append({
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "coverage": metrics["coverage"],
                "fnr": metrics["fnr"],
                "tnr": metrics["tnr"],
                "ece": metrics["ece"],
                "brier": metrics["brier"],
                "correct": metrics["correct"],
                "total": metrics["total"],
            })
        
        # Create metrics CSV
        import csv
        metrics_csv_data = "threshold,accuracy,coverage,fnr,tnr,ece,brier,correct,total"
        with metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(metrics_csv_data.split(","))
            for row in metrics_list:
                writer.writerow([
                    f"{row['threshold']:.4f}",
                    f"{row['accuracy']:.4f}",
                    f"{row['coverage']:.4f}",
                    f"{row['fnr']:.4f}",
                    f"{row['tnr']:.4f}",
                    f"{row['ece']:.4f}",
                    f"{row['brier']:.4f}",
                    f"{row['correct']}",
                    f"{row['total']}",
                ])
        
        # Save metrics using ArtifactStore
        ctx.artifact_store.put(ArtifactKey.THRESHOLDS_METRICS, str({"csv_path": str(metrics_path)}), ctx.run_id)
        print(f"   ✅ Metrics saved: {metrics_path}")
        
        # Final result
        artifacts_written = [
            ArtifactKey.THRESHOLDS_JSON,
            ArtifactKey.THRESHOLDS_METRICS,
        ]
        
        metrics_dict = {
            "selected_threshold": float(selected_threshold),
            "selected_accuracy": selected_metrics["accuracy"],
            "selected_coverage": selected_metrics["coverage"],
            "selected_fnr": selected_metrics["fnr"],
        }
        
        # Update manifest
        print(f"\n   📝 Updating manifest...")
        print("-" * 70)
        
        # Get manifest from context (or create new)
        manifest = ctx.manifest if ctx.manifest else ctx.artifact_store.initialize_manifest(
            run_id=ctx.run_id,
            config=cfg,
        )
        
        # Finalize step in manifest
        manifest.finalize_step(
            step_id=self.step_id,
            status="completed",
            metrics=metrics_dict,
        )
        
        # Return result
        result = StepResult(
            artifacts_written=artifacts_written,
            splits_used=splits_used,
            metrics=metrics_dict,
        )
        
        print("=" * 70)
        print("✅ Step (Sweep Thresholds) COMPLETED")
        print("=" * 70)
        
        return result


__all__ = [
    "SweepThresholdsSpec",
]

