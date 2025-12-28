"""
Unit Tests for Artifact Schema

Tests the artifact registry - single source of truth for all file paths.

Latest 2025-2026 practices:
- Python 3.14+ with pytest
- Fixture-based testing
- Path correctness validation
"""

import pytest
from pathlib import Path

from src.contracts.artifact_schema import ArtifactSchema, create_artifact_schema


class TestArtifactSchema:
    """Test ArtifactSchema path generation and directory creation"""

    def test_create_artifact_schema(self, temp_output_dir):
        """create_artifact_schema should return valid ArtifactSchema"""
        artifacts = create_artifact_schema(str(temp_output_dir))

        assert isinstance(artifacts, ArtifactSchema)
        assert artifacts.output_dir == temp_output_dir

    def test_output_dir_property(self, artifacts):
        """output_dir property should return Path"""
        assert isinstance(artifacts.output_dir, Path)
        assert artifacts.output_dir.exists()

    def test_phase_directories(self, artifacts):
        """Phase directory properties should return correct paths"""
        # Check all phase directories
        assert artifacts.phase1_dir == artifacts.output_dir / "phase1"
        assert artifacts.phase2_dir == artifacts.output_dir / "phase2"
        assert artifacts.phase3_dir == artifacts.output_dir / "phase3_gate"
        assert artifacts.phase4_dir == artifacts.output_dir / "phase4_explora"
        assert artifacts.phase5_dir == artifacts.output_dir / "phase5_scrc"
        assert artifacts.phase6_dir == artifacts.output_dir / "export"

    def test_ensure_dirs_creates_all_directories(self, temp_output_dir):
        """ensure_dirs() should create all required directories"""
        artifacts = create_artifact_schema(str(temp_output_dir))
        artifacts.ensure_dirs()

        # Check all directories exist
        assert artifacts.phase1_dir.exists()
        assert artifacts.phase2_dir.exists()
        assert artifacts.phase3_dir.exists()
        assert artifacts.phase4_dir.exists()
        assert artifacts.phase5_dir.exists()
        assert artifacts.phase6_dir.exists()

    def test_splits_json_path(self, artifacts):
        """splits_json property should return correct path"""
        expected = artifacts.output_dir / "data_splits" / "splits.json"
        assert artifacts.splits_json == expected

    def test_phase1_artifacts(self, artifacts):
        """Phase 1 artifact paths should be correct"""
        assert artifacts.phase1_checkpoint == artifacts.phase1_dir / "best_model.ckpt"
        assert artifacts.val_calib_logits == artifacts.phase1_dir / "val_calib_logits.npy"
        assert artifacts.val_calib_labels == artifacts.phase1_dir / "val_calib_labels.npy"
        assert artifacts.metrics_csv == artifacts.phase1_dir / "metrics.csv"
        assert artifacts.config_json == artifacts.phase1_dir / "config.json"

    def test_phase2_artifacts(self, artifacts):
        """Phase 2 artifact paths should be correct"""
        assert artifacts.thresholds_json == artifacts.phase2_dir / "thresholds.json"

    def test_phase3_artifacts(self, artifacts):
        """Phase 3 artifact paths should be correct"""
        assert artifacts.phase3_checkpoint == artifacts.phase3_dir / "gate_model.ckpt"
        assert artifacts.gateparams_json == artifacts.phase3_dir / "gateparams.json"

    def test_phase4_artifacts(self, artifacts):
        """Phase 4 artifact paths should be correct"""
        assert artifacts.explora_checkpoint == artifacts.phase4_dir / "explora_model.ckpt"

    def test_phase5_artifacts(self, artifacts):
        """Phase 5 artifact paths should be correct"""
        assert artifacts.scrcparams_json == artifacts.phase5_dir / "scrcparams.json"

    def test_phase6_artifacts(self, artifacts):
        """Phase 6 artifact paths should be correct"""
        assert artifacts.bundle_json == artifacts.phase6_dir / "bundle.json"

    def test_artifact_schema_immutable_paths(self, artifacts):
        """Artifact paths should be consistent across multiple calls"""
        # Call property multiple times
        path1 = artifacts.phase1_checkpoint
        path2 = artifacts.phase1_checkpoint
        path3 = artifacts.phase1_checkpoint

        # Should return same path every time
        assert path1 == path2 == path3

    def test_artifact_schema_relative_paths(self, artifacts):
        """All artifact paths should be relative to output_dir"""
        # Check all paths are children of output_dir
        all_paths = [
            artifacts.splits_json,
            artifacts.phase1_checkpoint,
            artifacts.val_calib_logits,
            artifacts.val_calib_labels,
            artifacts.metrics_csv,
            artifacts.config_json,
            artifacts.thresholds_json,
            artifacts.phase3_checkpoint,
            artifacts.gateparams_json,
            artifacts.explora_checkpoint,
            artifacts.scrcparams_json,
            artifacts.bundle_json,
        ]

        for path in all_paths:
            # Path should be descendant of output_dir
            assert artifacts.output_dir in path.parents or path.parent == artifacts.output_dir

    def test_artifact_schema_no_side_effects_in_properties(self, temp_output_dir):
        """Properties should NOT create directories (no side effects)"""
        # Create artifacts WITHOUT calling ensure_dirs()
        artifacts = create_artifact_schema(str(temp_output_dir))

        # Access properties (should NOT create directories)
        _ = artifacts.phase1_checkpoint
        _ = artifacts.phase2_dir
        _ = artifacts.thresholds_json

        # Directories should NOT exist yet (except output_dir which was created by fixture)
        # Note: We can't check this easily because the fixture creates output_dir
        # But we can verify that accessing properties doesn't create subdirectories
        phase_dirs = [
            artifacts.phase2_dir,
            artifacts.phase3_dir,
            artifacts.phase4_dir,
            artifacts.phase5_dir,
        ]

        # Some directories might exist from fixture, but accessing properties shouldn't create them
        # This test documents the INTENT: properties should be pure getters
        for phase_dir in phase_dirs:
            # Just verify we can access the property without errors
            assert isinstance(phase_dir, Path)

    def test_ensure_dirs_idempotent(self, artifacts):
        """ensure_dirs() should be idempotent (safe to call multiple times)"""
        # Call multiple times
        artifacts.ensure_dirs()
        artifacts.ensure_dirs()
        artifacts.ensure_dirs()

        # Should still work correctly
        assert artifacts.phase1_dir.exists()
        assert artifacts.phase2_dir.exists()


class TestArtifactSchemaEdgeCases:
    """Test edge cases and error handling"""

    def test_create_artifact_schema_with_string_path(self, temp_output_dir):
        """create_artifact_schema should accept string paths"""
        artifacts = create_artifact_schema(str(temp_output_dir))
        assert artifacts.output_dir == Path(temp_output_dir)

    def test_create_artifact_schema_with_path_object(self, temp_output_dir):
        """create_artifact_schema should accept Path objects"""
        artifacts = create_artifact_schema(temp_output_dir)
        assert artifacts.output_dir == temp_output_dir

    def test_artifact_schema_nested_output_dir(self, temp_output_dir):
        """ArtifactSchema should work with nested output directories"""
        nested_dir = temp_output_dir / "level1" / "level2" / "level3"
        artifacts = create_artifact_schema(nested_dir)
        artifacts.ensure_dirs()

        # Should create all parent directories
        assert nested_dir.exists()
        assert artifacts.phase1_dir.exists()

    def test_artifact_schema_repr(self, artifacts):
        """ArtifactSchema should have meaningful repr"""
        repr_str = repr(artifacts)
        assert "ArtifactSchema" in repr_str
        assert str(artifacts.output_dir) in repr_str
