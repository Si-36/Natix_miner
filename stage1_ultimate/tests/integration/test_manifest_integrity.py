"""
Manifest integrity tests: Checksum verification

Tests:
- All output artifacts have checksums
- Checksums match actual files
- Corrupted files detected
- Missing files detected

Critical for production lineage tracking
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from contracts.artifact_schema import create_artifact_schema
from pipeline.dag_engine import DAGEngine
from pipeline.phase_spec import PhaseType

from streetvision.pipeline.steps import run_phase1_baseline

from tests.conftest import (
    load_manifest,
    compute_file_sha256,
    verify_manifest,
)


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


@pytest.mark.integration
@pytest.mark.slow
def test_manifest_has_all_checksums(tiny_config_with_output):
    """
    Test manifest contains checksums for all output artifacts

    Validates:
    - All output artifacts have sha256
    - All output artifacts have size_bytes
    - All output artifacts have created_at
    """
    cfg = tiny_config_with_output

    # Run Phase-1
    output_dir = run_phase1_only(cfg)

    # Load manifest
    manifest_path = output_dir / "phase1_baseline" / "manifest.json"
    manifest = load_manifest(manifest_path)

    # Assert output_artifacts exist
    assert "output_artifacts" in manifest
    assert len(manifest["output_artifacts"]) > 0

    # Check each artifact
    for artifact_name, artifact_info in manifest["output_artifacts"].items():
        # Must have checksum
        assert "sha256" in artifact_info, \
            f"Missing sha256 for {artifact_name}"

        # Must have size
        assert "size_bytes" in artifact_info, \
            f"Missing size_bytes for {artifact_name}"

        # Must have timestamp
        assert "created_at" in artifact_info, \
            f"Missing created_at for {artifact_name}"

        # Checksum must be 64 hex chars (SHA256)
        assert len(artifact_info["sha256"]) == 64, \
            f"Invalid SHA256 length for {artifact_name}"


@pytest.mark.integration
@pytest.mark.slow
def test_checksums_match_actual_files(tiny_config_with_output):
    """
    Test manifest checksums match actual file contents

    Validates:
    - Recomputed checksums match manifest
    - No silent corruption
    """
    cfg = tiny_config_with_output

    # Run Phase-1
    output_dir = run_phase1_only(cfg)

    # Load manifest
    manifest_path = output_dir / "phase1_baseline" / "manifest.json"
    manifest = load_manifest(manifest_path)

    # Verify each artifact
    for artifact_name, artifact_info in manifest["output_artifacts"].items():
        artifact_path = output_dir / artifact_info["path"]

        # Compute actual checksum
        actual_sha256 = compute_file_sha256(artifact_path)

        # Compare with manifest
        expected_sha256 = artifact_info["sha256"]

        assert actual_sha256 == expected_sha256, \
            f"Checksum mismatch for {artifact_name}:\n" \
            f"  Expected: {expected_sha256}\n" \
            f"  Actual:   {actual_sha256}"


@pytest.mark.integration
@pytest.mark.slow
def test_verify_manifest_helper(tiny_config_with_output):
    """
    Test verify_manifest() helper function

    Validates:
    - verify_manifest() passes for valid manifest
    - No exceptions raised
    """
    cfg = tiny_config_with_output

    # Run Phase-1
    output_dir = run_phase1_only(cfg)

    # Load manifest
    manifest_path = output_dir / "phase1_baseline" / "manifest.json"
    manifest = load_manifest(manifest_path)

    # Should not raise
    verify_manifest(manifest, output_dir)


@pytest.mark.integration
@pytest.mark.slow
def test_corrupted_file_detected(tiny_config_with_output):
    """
    Test corrupted file is detected by checksum verification

    Scenario:
    1. Run Phase-1
    2. Corrupt a file
    3. Verify manifest
    4. Expect: ValueError raised
    """
    cfg = tiny_config_with_output

    # Run Phase-1
    output_dir = run_phase1_only(cfg)

    # Load manifest
    manifest_path = output_dir / "phase1_baseline" / "manifest.json"
    manifest = load_manifest(manifest_path)

    # Corrupt checkpoint (append garbage)
    ckpt_path = output_dir / "phase1_baseline" / "model_best.pth"
    with open(ckpt_path, "ab") as f:
        f.write(b"CORRUPTED_12345")

    # Verify should fail
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_manifest(manifest, output_dir)


@pytest.mark.integration
@pytest.mark.slow
def test_missing_file_detected(tiny_config_with_output):
    """
    Test missing file is detected by verification

    Scenario:
    1. Run Phase-1
    2. Delete a file
    3. Verify manifest
    4. Expect: FileNotFoundError raised
    """
    cfg = tiny_config_with_output

    # Run Phase-1
    output_dir = run_phase1_only(cfg)

    # Load manifest
    manifest_path = output_dir / "phase1_baseline" / "manifest.json"
    manifest = load_manifest(manifest_path)

    # Delete checkpoint
    ckpt_path = output_dir / "phase1_baseline" / "model_best.pth"
    ckpt_path.unlink()

    # Verify should fail
    with pytest.raises(FileNotFoundError, match="Artifact missing"):
        verify_manifest(manifest, output_dir)


@pytest.mark.integration
@pytest.mark.slow
def test_manifest_has_metadata(tiny_config_with_output):
    """
    Test manifest contains required metadata fields

    Validates:
    - run_id present
    - step_name present
    - git_sha present
    - config_hash present
    - duration_seconds present
    - hostname present
    """
    cfg = tiny_config_with_output

    # Run Phase-1
    output_dir = run_phase1_only(cfg)

    # Load manifest
    manifest_path = output_dir / "phase1_baseline" / "manifest.json"
    manifest = load_manifest(manifest_path)

    # Check required fields
    required_fields = [
        "run_id",
        "step_name",
        "git_sha",
        "config_hash",
        "duration_seconds",
        "hostname",
        "python_version",
    ]

    for field in required_fields:
        assert field in manifest, f"Missing required field: {field}"

    # Check types
    assert isinstance(manifest["run_id"], str)
    assert isinstance(manifest["step_name"], str)
    assert isinstance(manifest["git_sha"], str)
    assert isinstance(manifest["config_hash"], str)
    assert isinstance(manifest["duration_seconds"], (int, float))
    assert isinstance(manifest["hostname"], str)
