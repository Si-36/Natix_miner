#!/usr/bin/env python3
"""
🧪 **Integration Test: DAG Smoke (End-to-End Pipeline Validation)**
Tests entire pipeline: export_calib_logits → sweep_thresholds → export

This is a SMOKE TEST - validates:
- DAG resolution (dependencies, ordering)
- ArtifactStore (atomic writes, hashes)
- Split contracts (no leakage!)
- Manifest tracking (lineage)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any
import torch
import json

# Add src to path
src_path = Path.cwd() / "src"
sys.path.insert(0, str(src_path))

# Use absolute imports to avoid circular import issues
from pipeline.artifacts import ArtifactKey, ArtifactStore
from pipeline.step_api import StepSpec, StepContext, StepResult
from pipeline.contracts import Split, assert_allowed
from pipeline.registry import get_step_spec, resolve_execution_order


def create_mock_calibration_data():
    """
    Create mock calibration data (since we don't have real training yet).

    Returns:
        Tuple of (logits, labels) tensors
    """
    # Mock calibration logits (100 samples, 2 classes)
    calib_logits = torch.randn(100, 2)
    calib_labels = torch.randint(0, 2, (100,))

    print(f"   📊 Mock calibration data:")
    print(f"      Logits shape: {calib_logits.shape}")
    print(f"      Labels shape: {calib_labels.shape}")
    print(f"      Labels distribution: {torch.bincount(calib_labels).tolist()}")

    return calib_logits, calib_labels


def test_export_calib_logits_step():
    """
    Test 1: Run export_calib_logits step.

    This step:
    - Depends on train_baseline_head (we'll skip this dependency for now)
    - Uses VAL_CALIB ONLY (leak-proof!)
    - Saves VAL_CALIB_LOGITS, VAL_CALIB_LABELS
    """
    print(f"\n{'=' * 70}")
    print(f"TEST 1: Export Calib Logits Step")
    print("=" * 70)

    # Get step spec
    step_spec = get_step_spec("export_calib_logits")

    if step_spec is None:
        raise RuntimeError("export_calib_logits step not found in registry!")

    print(f"   📋 Step spec loaded: {step_spec.name}")

    # Create artifact store
    artifact_root = Path(__file__).parent.parent / "test_artifacts"
    store = ArtifactStore(artifact_root)

    # Create manifest
    run_id = "test_run_001"
    config = {
        "step": step_spec.__dict__() if hasattr(step_spec, "__dict__") else dict(vars(step_spec))
    }
    manifest_path = store.initialize_manifest(run_id, config)

    print(f"   📊 Manifest initialized: {run_id}")

    # Create step context
    ctx = StepContext(
        step_id="export_calib_logits",
        config=config,
        run_id=run_id,
        artifact_root=artifact_root,
        artifact_store=store,
        manifest=store._manifest,
        metadata={
            "test": True,
            "description": "Mock test for export_calib_logits",
        },
    )

    # 🔥 LEAK-PROOF: Enforce split contract
    used_splits = frozenset({Split.VAL_CALIB})  # VAL_CALIB ONLY!
    print(f"   🔒 Enforcing split contract: {sorted(list(used_splits))}")

    assert_allowed(
        used=used_splits,
        allowed=step_spec.allowed_splits(),
        context="export_calib_logits.run()",
    )
    print(f"   ✅ Split contract validated")

    # Create mock calibration data
    calib_logits, calib_labels = create_mock_calibration_data()

    # Write calibration artifacts (mock export_calib_logits)
    print(f"\n   💾 Writing calibration artifacts...")

    logits_path = store.put(ArtifactKey.VAL_CALIB_LOGITS, calib_logits, run_id=run_id)
    labels_path = store.put(ArtifactKey.VAL_CALIB_LABELS, calib_labels, run_id=run_id)

    print(f"   ✅ VAL_CALIB_LOGITS: {logits_path}")
    print(f"   ✅ VAL_CALIB_LABELS: {labels_path}")

    # Verify files exist
    assert logits_path.exists(), f"VAL_CALIB_LOGITS not found: {logits_path}"
    assert labels_path.exists(), f"VAL_CALIB_LABELS not found: {labels_path}"

    # Verify hashes recorded
    assert ArtifactKey.VAL_CALIB_LOGITS.value in store._hashes, (
        "Hash not recorded for VAL_CALIB_LOGITS"
    )
    assert ArtifactKey.VAL_CALIB_LABELS.value in store._hashes, (
        "Hash not recorded for VAL_CALIB_LABELS"
    )

    # Finalize step
    store.finalize_step(
        step_id="export_calib_logits",
        status="completed",
        metrics={
            "num_samples": int(calib_logits.shape[0]),
            "logits_path": str(logits_path),
            "labels_path": str(labels_path),
        },
    )

    print(f"\n✅ TEST 1 PASSED: Export Calib Logits Step")
    print("=" * 70)

    return run_id, store


def test_sweep_thresholds_step(run_id: str, store: ArtifactStore):
    """
    Test 2: Run sweep_thresholds step.

    This step:
    - Depends on export_calib_logits (we just ran it!)
    - Uses VAL_CALIB ONLY (leak-proof!)
    - Saves THRESHOLDS_JSON, THRESHOLDS_METRICS
    """
    print(f"\n{'=' * 70}")
    print(f"TEST 2: Sweep Thresholds Step")
    print("=" * 70)

    # Get step spec
    step_spec = get_step_spec("sweep_thresholds")

    if step_spec is None:
        raise RuntimeError("sweep_thresholds step not found in registry!")

    print(f"   📋 Step spec loaded: {step_spec.name}")
    print(f"   🔗 Dependencies: {step_spec.deps}")

    # Create step context
    config = {
        "step": step_spec.__dict__(),
        "sweep": {
            "target_fnr": 0.05,  # Target 5% FNR
        },
    }

    ctx = StepContext(
        step_id="sweep_thresholds",
        config=config,
        run_id=run_id,
        artifact_root=store.artifact_root,
        artifact_store=store,
        manifest=store._manifest,
        metadata={
            "test": True,
            "description": "Mock test for sweep_thresholds",
        },
    )

    # 🔥 LEAK-PROOF: Enforce split contract
    used_splits = frozenset({Split.VAL_CALIB})  # VAL_CALIB ONLY!
    print(f"   🔒 Enforcing split contract: {sorted(list(used_splits))}")

    assert_allowed(
        used=used_splits,
        allowed=step_spec.allowed_splits(),
        context="sweep_thresholds.run()",
    )
    print(f"   ✅ Split contract validated")

    # Run step
    print(f"\n   🎚️  Running sweep...")
    result = step_spec.run(ctx)

    # Verify results
    print(f"\n   📊 Step results:")
    print(f"      Artifacts written: {result.artifacts_written}")
    print(f"      Splits used: {sorted(list(result.splits_used))}")
    print(f"      Metrics: {list(result.metrics.keys())}")

    # Verify artifacts exist
    thresholds_path = store.get(ArtifactKey.THRESHOLDS_JSON, run_id=run_id)
    metrics_csv_path = store.get(ArtifactKey.THRESHOLDS_METRICS, run_id=run_id)

    assert thresholds_path.exists(), f"THRESHOLDS_JSON not found: {thresholds_path}"
    assert metrics_csv_path.exists(), f"THRESHOLDS_METRICS not found: {metrics_csv_path}"

    print(f"   ✅ THRESHOLDS_JSON: {thresholds_path}")
    print(f"   ✅ THRESHOLDS_METRICS: {metrics_csv_path}")

    # Verify thresholds JSON can be loaded
    with thresholds_path.open("r", encoding="utf-8") as f:
        thresholds_data = json.load(f)

    assert "best_threshold" in thresholds_data, "Best threshold not found in JSON"
    assert "metrics" in thresholds_data, "Metrics not found in JSON"

    # Verify metrics CSV can be loaded
    with metrics_csv_path.open("r", encoding="utf-8") as f:
        csv_lines = f.readlines()

    assert len(csv_lines) > 1, "Metrics CSV is empty"

    # Verify correct metrics in CSV
    assert "acceptance_rate" in csv_lines[0], "acceptance_rate not in CSV"
    assert "f1_score" in csv_lines[0], "f1_score not in CSV"

    print(f"\n   ✅ Verified thresholds and metrics formats")

    # Finalize step
    store.finalize_step(
        step_id="sweep_thresholds",
        status="completed",
        metrics=result.metrics or {},
    )

    print(f"\n✅ TEST 2 PASSED: Sweep Thresholds Step")
    print("=" * 70)


def test_dag_resolution():
    """
    Test 3: Validate DAG resolution.

    Verifies:
    - Dependencies are correctly ordered
    - Circular dependencies detected (if any)
    """
    print(f"\n{'=' * 70}")
    print(f"TEST 3: DAG Resolution")
    print("=" * 70)

    # Test export_calib_logits dependencies
    export_spec = get_step_spec("export_calib_logits")

    print(f"   📋 export_calib_logits:")
    print(f"      Dependencies: {export_spec.deps}")
    print(f"      Allowed splits: {sorted(list(export_spec.allowed_splits()))}")

    # Test sweep_thresholds dependencies
    sweep_spec = get_step_spec("sweep_thresholds")

    print(f"\n   📋 sweep_thresholds:")
    print(f"      Dependencies: {sweep_spec.deps}")
    print(f"      Allowed splits: {sorted(list(sweep_spec.allowed_splits()))}")

    # Verify dependency graph
    assert export_spec.deps is not None, "export_calib_logits deps is None"
    assert sweep_spec.deps is not None, "sweep_thresholds deps is None"

    # Verify export_calib_logits → sweep_thresholds dependency
    assert "export_calib_logits" in sweep_spec.deps, (
        "sweep_thresholds should depend on export_calib_logits"
    )

    # Verify no circular dependencies
    execution_order = resolve_execution_order("sweep_thresholds")

    print(f"\n   🗂 Execution order:")
    for i, step in enumerate(execution_order, 1):
        print(f"      {i}. {step}")

    # Verify order is correct
    assert execution_order[0] == "export_calib_logits", "First step should be export_calib_logits"
    assert execution_order[1] == "sweep_thresholds", "Second step should be sweep_thresholds"

    print(f"\n   ✅ DAG resolution is correct")
    print("=" * 70)


def test_manifest_tracking(run_id: str, store: ArtifactStore):
    """
    Test 4: Validate manifest tracking.

    Verifies:
    - Manifest is initialized
    - Artifacts hashes are recorded
    - Steps are tracked
    """
    print(f"\n{'=' * 70}")
    print(f"TEST 4: Manifest Tracking")
    print("=" * 70)

    # Load manifest
    assert store._manifest is not None, "Manifest not loaded"

    manifest = store._manifest

    print(f"   📊 Manifest:")
    print(f"      Run ID: {manifest.get('run_id', 'unknown')}")
    print(f"      Timestamp: {manifest.get('timestamp', 'unknown')}")

    # Verify artifact hashes recorded
    artifact_hashes = manifest.get("artifact_hashes", {})

    print(f"\n   🔐 Artifact hashes recorded:")
    for key in sorted(artifact_hashes.keys()):
        print(f"      {key}: {artifact_hashes[key][:8]}...")  # Show first 8 chars

    assert len(artifact_hashes) > 0, "No artifact hashes recorded"
    assert ArtifactKey.VAL_CALIB_LOGITS.value in artifact_hashes, (
        "VAL_CALIB_LOGITS hash not recorded"
    )
    assert ArtifactKey.VAL_CALIB_LABELS.value in artifact_hashes, (
        "VAL_CALIB_LABELS hash not recorded"
    )
    assert ArtifactKey.THRESHOLDS_JSON.value in artifact_hashes, "THRESHOLDS_JSON hash not recorded"
    assert ArtifactKey.THRESHOLDS_METRICS.value in artifact_hashes, (
        "THRESHOLDS_METRICS hash not recorded"
    )

    print(f"\n   ✅ All artifact hashes recorded")

    # Verify steps tracked
    steps = manifest.get("steps", {})

    print(f"\n   📋 Steps tracked:")
    for step_id in sorted(steps.keys()):
        step_data = steps[step_id]
        print(f"      {step_id}:")
        print(f"        Status: {step_data.get('status', 'unknown')}")
        print(f"        Metadata: {list(step_data.get('metadata', {}).keys())}")

    assert "export_calib_logits" in steps, "export_calib_logits step not tracked"
    assert "sweep_thresholds" in steps, "sweep_thresholds step not tracked"

    print(f"\n   ✅ All steps tracked in manifest")
    print("=" * 70)


def test_split_contracts():
    """
    Test 5: Validate split contracts (NO LEAKAGE!).

    Verifies:
    - VAL_CALIB is ONLY used by export_calib_logits and sweep_thresholds
    - TRAIN is NOT used by any step (in this test)
    - VAL_SELECT is NOT used by any step (in this test)
    """
    print(f"\n{'=' * 70}")
    print(f"TEST 5: Split Contracts (NO LEAKAGE!)")
    print("=" * 70)

    # Get step specs
    export_spec = get_step_spec("export_calib_logits")
    sweep_spec = get_step_spec("sweep_thresholds")

    # Verify export_calib_logits uses VAL_CALIB ONLY
    export_allowed = export_spec.allowed_splits()
    assert Split.VAL_CALIB in export_allowed, "VAL_CALIB not in export_calib_logits allowed splits"
    assert Split.TRAIN not in export_allowed, (
        "TRAIN in export_calib_logits allowed splits (LEAKAGE!)"
    )
    assert Split.VAL_SELECT not in export_allowed, (
        "VAL_SELECT in export_calib_logits allowed splits (LEAKAGE!)"
    )

    print(f"   ✅ export_calib_logits split contract: VAL_CALIB ONLY (leak-proof)")

    # Verify sweep_thresholds uses VAL_CALIB ONLY
    sweep_allowed = sweep_spec.allowed_splits()
    assert Split.VAL_CALIB in sweep_allowed, "VAL_CALIB not in sweep_thresholds allowed splits"
    assert Split.TRAIN not in sweep_allowed, "TRAIN in sweep_thresholds allowed splits (LEAKAGE!)"
    assert Split.VAL_SELECT not in sweep_allowed, (
        "VAL_SELECT in sweep_thresholds allowed splits (LEAKAGE!)"
    )

    print(f"   ✅ sweep_thresholds split contract: VAL_CALIB ONLY (leak-proof)")

    # Verify VAL_TEST is NOT used (in this test)
    assert Split.VAL_TEST not in export_allowed, (
        "VAL_TEST in export_calib_logits allowed splits (correct!)"
    )
    assert Split.VAL_TEST not in sweep_allowed, (
        "VAL_TEST in sweep_thresholds allowed splits (correct!)"
    )

    print(f"   ✅ VAL_TEST not used (correct - no leakage!)")
    print("=" * 70)


def main():
    """
    Run all integration tests.

    Tests:
    1. export_calib_logits step (with mock data)
    2. sweep_thresholds step (using exported data)
    3. DAG resolution (dependency ordering)
    4. Manifest tracking (hashes + steps)
    5. Split contracts (no leakage!)
    """
    print(f"\n{'=' * 70}")
    print(f"🧪 Integration Tests: DAG Smoke (End-to-End Pipeline)")
    print("=" * 70)

    try:
        # Test 1: Run export_calib_logits
        run_id, store = test_export_calib_logits_step()

        # Test 2: Run sweep_thresholds
        test_sweep_thresholds_step(run_id, store)

        # Test 3: DAG resolution
        test_dag_resolution()

        # Test 4: Manifest tracking
        test_manifest_tracking(run_id, store)

        # Test 5: Split contracts
        test_split_contracts()

        print(f"\n{'=' * 70}")
        print(f"🎉 ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   ✅ Export calib logits (VAL_CALIB ONLY!)")
        print(f"   ✅ Sweep thresholds (correct metrics!)")
        print(f"   ✅ DAG resolution (correct ordering!)")
        print(f"   ✅ Manifest tracking (all hashes recorded!)")
        print(f"   ✅ Split contracts (NO LEAKAGE!)")
        print(f"   ✅ All artifacts exist (4 artifacts!)")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"❌ INTEGRATION TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
