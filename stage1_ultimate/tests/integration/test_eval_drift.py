"""
Eval drift protection tests: MCC consistency

Tests:
- Same logits → same MCC (centralized eval)
- No drift across phases
- Threshold selection uses centralized eval
- Calibration uses centralized eval

CRITICAL: Prevents metric inconsistencies
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from contracts.artifact_schema import create_artifact_schema
from pipeline.dag_engine import DAGEngine
from pipeline.phase_spec import PhaseType

from streetvision.pipeline.steps import (
    run_phase1_baseline,
    run_phase2_threshold_sweep,
)

# Import centralized eval
from streetvision.eval.metrics import compute_mcc

from tests.conftest import load_manifest


def run_phase1_only(cfg) -> Path:
    """Helper: Run Phase-1 only"""
    output_dir = Path(cfg.output_dir)
    artifacts = create_artifact_schema(output_dir)
    artifacts.ensure_dirs()

    engine = DAGEngine(artifacts=artifacts)

    def phase1_executor(artifacts):
        run_phase1_baseline(artifacts=artifacts, cfg=cfg)

    engine.register_executor(PhaseType.PHASE1_BASELINE, phase1_executor)
    engine.run([PhaseType.PHASE1_BASELINE])

    return output_dir


def run_phase1_phase2(cfg) -> Path:
    """Helper: Run Phase-1 → Phase-2"""
    output_dir = Path(cfg.output_dir)
    artifacts = create_artifact_schema(output_dir)
    artifacts.ensure_dirs()

    engine = DAGEngine(artifacts=artifacts)

    def phase1_executor(artifacts):
        run_phase1_baseline(artifacts=artifacts, cfg=cfg)

    def phase2_executor(artifacts):
        run_phase2_threshold_sweep(artifacts=artifacts, cfg=cfg)

    engine.register_executor(PhaseType.PHASE1_BASELINE, phase1_executor)
    engine.register_executor(PhaseType.PHASE2_THRESHOLD, phase2_executor)

    engine.run([PhaseType.PHASE1_BASELINE, PhaseType.PHASE2_THRESHOLD])

    return output_dir


@pytest.mark.integration
@pytest.mark.slow
def test_no_mcc_drift_recomputation(tiny_config_with_output):
    """
    CRITICAL TEST: Prevents MCC drift

    Scenario:
    1. Run Phase-1 (saves val_calib logits/labels)
    2. Load logits/labels
    3. Recompute MCC using centralized eval
    4. Compare with Phase-1 reported MCC
    5. Assert: IDENTICAL (no drift)

    Why Important:
        If different code paths compute MCC differently,
        threshold selection will be wrong!
    """
    cfg = tiny_config_with_output

    # Run Phase-1
    output_dir = run_phase1_only(cfg)

    # Load Phase-1 manifest
    manifest_p1_path = output_dir / "phase1_baseline" / "manifest.json"
    manifest_p1 = load_manifest(manifest_p1_path)
    phase1_mcc = manifest_p1["metrics"]["mcc"]

    # Load saved logits/labels
    logits_path = output_dir / "phase1_baseline" / "val_calib_logits.pt"
    labels_path = output_dir / "phase1_baseline" / "val_calib_labels.pt"

    logits = torch.load(logits_path)
    labels = torch.load(labels_path)

    # Compute MCC using CENTRALIZED eval
    preds = torch.argmax(logits, dim=1)
    recomputed_mcc = compute_mcc(labels.numpy(), preds.numpy())

    # Assert: IDENTICAL (no drift)
    assert abs(recomputed_mcc - phase1_mcc) < 1e-6, \
        f"MCC drift detected!\n" \
        f"  Phase-1 reported: {phase1_mcc:.6f}\n" \
        f"  Recomputed:       {recomputed_mcc:.6f}\n" \
        f"  Difference:       {abs(recomputed_mcc - phase1_mcc):.6e}"


@pytest.mark.integration
@pytest.mark.slow
def test_no_mcc_drift_across_phases(tiny_config_with_output):
    """
    Test Phase-1 and Phase-2 use same MCC computation

    Scenario:
    1. Run Phase-1 → Phase-2
    2. Load logits/labels
    3. Recompute MCC
    4. Compare with both Phase-1 and Phase-2
    5. Assert: All IDENTICAL

    Ensures:
        - Phase-1 training uses centralized eval
        - Phase-2 threshold sweep uses centralized eval
        - No metric drift between phases
    """
    cfg = tiny_config_with_output

    # Run Phase-1 → Phase-2
    output_dir = run_phase1_phase2(cfg)

    # Load manifests
    manifest_p1 = load_manifest(output_dir / "phase1_baseline" / "manifest.json")
    manifest_p2 = load_manifest(output_dir / "phase2_threshold" / "manifest.json")

    phase1_mcc = manifest_p1["metrics"]["mcc"]
    phase2_mcc_base = manifest_p2["metrics"].get("mcc_baseline", phase1_mcc)

    # Load logits/labels
    logits = torch.load(output_dir / "phase1_baseline" / "val_calib_logits.pt")
    labels = torch.load(output_dir / "phase1_baseline" / "val_calib_labels.pt")

    # Recompute using centralized eval
    preds = torch.argmax(logits, dim=1)
    recomputed_mcc = compute_mcc(labels.numpy(), preds.numpy())

    # Assert all match
    assert abs(recomputed_mcc - phase1_mcc) < 1e-6, \
        "Phase-1 MCC drift detected"

    assert abs(recomputed_mcc - phase2_mcc_base) < 1e-6, \
        "Phase-2 MCC drift detected"


@pytest.mark.integration
@pytest.mark.slow
def test_phase2_uses_same_logits_as_phase1(tiny_config_with_output):
    """
    Test Phase-2 uses EXACT same logits as Phase-1

    Validates:
    - Input artifact checksums match
    - No recomputation of logits
    - No silent data drift
    """
    cfg = tiny_config_with_output

    # Run Phase-1 → Phase-2
    output_dir = run_phase1_phase2(cfg)

    # Load manifests
    manifest_p1 = load_manifest(output_dir / "phase1_baseline" / "manifest.json")
    manifest_p2 = load_manifest(output_dir / "phase2_threshold" / "manifest.json")

    # Phase-1 outputs = Phase-2 inputs
    p1_logits_sha = manifest_p1["output_artifacts"]["val_calib_logits.pt"]["sha256"]
    p1_labels_sha = manifest_p1["output_artifacts"]["val_calib_labels.pt"]["sha256"]

    p2_logits_sha = manifest_p2["input_artifacts"]["val_calib_logits.pt"]["sha256"]
    p2_labels_sha = manifest_p2["input_artifacts"]["val_calib_labels.pt"]["sha256"]

    # Assert exact match
    assert p1_logits_sha == p2_logits_sha, \
        "Phase-2 should use EXACT Phase-1 logits (checksum mismatch!)"

    assert p1_labels_sha == p2_labels_sha, \
        "Phase-2 should use EXACT Phase-1 labels (checksum mismatch!)"


@pytest.mark.integration
def test_centralized_eval_import():
    """
    Test centralized eval module imports correctly

    Validates:
    - streetvision.eval.metrics exists
    - compute_mcc() available
    - compute_all_metrics() available
    """
    from streetvision.eval.metrics import (
        compute_mcc,
        compute_all_metrics,
        compute_accuracy,
        compute_precision,
        compute_recall,
        compute_f1,
    )

    # Test with dummy data
    import numpy as np

    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    # Should not raise
    mcc = compute_mcc(y_true, y_pred)
    assert isinstance(mcc, float)

    all_metrics = compute_all_metrics(y_true, y_pred)
    assert "mcc" in all_metrics
    assert "accuracy" in all_metrics
    assert "precision" in all_metrics
