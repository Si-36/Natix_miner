"""
Smoke tests: End-to-end pipeline execution

Tests:
- Pipeline runs without crashes
- All manifests created
- All artifacts exist
- Bundle export works

Time: ~5-10 minutes on CPU (tiny config)
"""

import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from contracts.artifact_schema import create_artifact_schema
from pipeline.dag_engine import DAGEngine
from pipeline.phase_spec import PhaseType

from streetvision.pipeline.steps import (
    run_phase1_baseline,
    run_phase2_threshold_sweep,
    run_phase6_bundle_export,
)

from tests.conftest import load_manifest, verify_manifest


def run_pipeline_phases(
    cfg: DictConfig,
    phases: list,
) -> Path:
    """
    Run pipeline with specified phases

    Args:
        cfg: Hydra config
        phases: List of phase names (e.g., ["phase1", "phase2"])

    Returns:
        Output directory path
    """
    output_dir = Path(cfg.output_dir)
    artifacts = create_artifact_schema(output_dir)
    artifacts.ensure_dirs()

    # Create DAG engine
    engine = DAGEngine(artifacts=artifacts)

    # Register executors (production-grade v2)
    def phase1_executor(artifacts):
        run_phase1_baseline(artifacts=artifacts, cfg=cfg)

    def phase2_executor(artifacts):
        run_phase2_threshold_sweep(artifacts=artifacts, cfg=cfg)

    def phase6_executor(artifacts):
        run_phase6_bundle_export(artifacts=artifacts, cfg=cfg)

    engine.register_executor(PhaseType.PHASE1_BASELINE, phase1_executor)
    engine.register_executor(PhaseType.PHASE2_THRESHOLD, phase2_executor)
    engine.register_executor(PhaseType.PHASE6_BUNDLE, phase6_executor)

    # Resolve phases
    phase_map = {
        "phase1": PhaseType.PHASE1_BASELINE,
        "phase2": PhaseType.PHASE2_THRESHOLD,
        "phase6": PhaseType.PHASE6_BUNDLE,
    }

    phases_to_run = [phase_map[p] for p in phases if p in phase_map]

    # Execute
    engine.run(phases_to_run)

    return output_dir


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_runs_phase1_tiny(tiny_config_with_output):
    """
    Test Phase-1 runs end-to-end on tiny config

    Validates:
    - Phase-1 completes without crash
    - Manifest exists
    - All artifacts exist
    - Checksums verify

    Time: ~5 minutes on CPU
    """
    cfg = tiny_config_with_output
    cfg.pipeline.phases = ["phase1"]

    # Run pipeline
    output_dir = run_pipeline_phases(cfg, phases=["phase1"])

    # Assert manifest exists
    manifest_path = output_dir / "phase1_baseline" / "manifest.json"
    assert manifest_path.exists(), f"Manifest not found: {manifest_path}"

    # Load and verify manifest
    manifest = load_manifest(manifest_path)
    verify_manifest(manifest, output_dir)

    # Assert key artifacts exist
    assert (output_dir / "phase1_baseline" / "model_best.pth").exists()
    assert (output_dir / "phase1_baseline" / "val_calib_logits.pt").exists()
    assert (output_dir / "phase1_baseline" / "val_calib_labels.pt").exists()

    # Assert metrics recorded
    assert "mcc" in manifest["metrics"]
    assert "accuracy" in manifest["metrics"]


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_runs_phase1_phase2(tiny_config_with_output):
    """
    Test Phase-1 → Phase-2 pipeline

    Validates:
    - Both phases complete
    - Both manifests exist
    - Phase-2 uses Phase-1 outputs
    - Threshold selected

    Time: ~7 minutes on CPU
    """
    cfg = tiny_config_with_output
    cfg.pipeline.phases = ["phase1", "phase2"]

    # Run pipeline
    output_dir = run_pipeline_phases(cfg, phases=["phase1", "phase2"])

    # Assert Phase-1 manifest
    manifest_p1_path = output_dir / "phase1_baseline" / "manifest.json"
    assert manifest_p1_path.exists()

    manifest_p1 = load_manifest(manifest_p1_path)
    verify_manifest(manifest_p1, output_dir)

    # Assert Phase-2 manifest
    manifest_p2_path = output_dir / "phase2_threshold" / "manifest.json"
    assert manifest_p2_path.exists()

    manifest_p2 = load_manifest(manifest_p2_path)
    verify_manifest(manifest_p2, output_dir)

    # Assert Phase-2 input artifacts match Phase-1 outputs
    # (checksums should match)
    p1_logits_sha = manifest_p1["output_artifacts"]["val_calib_logits.pt"]["sha256"]
    p2_logits_sha = manifest_p2["input_artifacts"]["val_calib_logits.pt"]["sha256"]

    assert p1_logits_sha == p2_logits_sha, \
        "Phase-2 should use exact Phase-1 logits (no drift!)"

    # Assert threshold selected
    assert "best_threshold" in manifest_p2["metrics"]
    assert "mcc_at_threshold" in manifest_p2["metrics"]


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_full_flow_with_bundle(tiny_config_with_output):
    """
    Test Phase-1 → Phase-2 → Phase-6 (full deployable flow)

    Validates:
    - All phases complete
    - Bundle created
    - Bundle contains all required files
    - Bundle uses relative paths

    Time: ~10 minutes on CPU
    """
    cfg = tiny_config_with_output
    cfg.pipeline.phases = ["phase1", "phase2", "phase6"]

    # Run pipeline
    output_dir = run_pipeline_phases(cfg, phases=["phase1", "phase2", "phase6"])

    # Assert all manifests exist
    assert (output_dir / "phase1_baseline" / "manifest.json").exists()
    assert (output_dir / "phase2_threshold" / "manifest.json").exists()
    assert (output_dir / "phase6_bundle" / "manifest.json").exists()

    # Assert bundle files exist
    bundle_dir = output_dir / "phase6_bundle" / "export"
    assert bundle_dir.exists()

    bundle_json = bundle_dir / "bundle.json"
    assert bundle_json.exists()

    # Load bundle
    import json
    with open(bundle_json, "r") as f:
        bundle = json.load(f)

    # Assert bundle structure
    assert "metadata" in bundle
    assert "artifacts" in bundle
    assert "lineage" in bundle

    # Assert all files use relative paths
    for artifact_name, artifact_info in bundle["artifacts"].items():
        path = artifact_info["path"]
        assert not path.startswith("/"), \
            f"Bundle should use relative paths, got: {path}"

    # Assert all referenced files exist
    for artifact_name, artifact_info in bundle["artifacts"].items():
        artifact_path = bundle_dir / artifact_info["path"]
        assert artifact_path.exists(), \
            f"Bundle artifact missing: {artifact_name} at {artifact_path}"
