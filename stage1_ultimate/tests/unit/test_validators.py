"""
Unit Tests for Validators

Tests validation functions for checkpoints, logits, and bundles.

Latest 2025-2026 practices:
- Python 3.14+ with pytest
- Fixture-based testing
- Clear error message validation
"""

import pytest
import json
import numpy as np
import torch
from pathlib import Path

from src.contracts.validators import (
    CheckpointValidationError,
    LogitsValidationError,
    BundleValidationError,
    validate_checkpoint,
    validate_logits,
    validate_bundle,
)


class TestCheckpointValidation:
    """Test checkpoint validation"""

    def test_validate_checkpoint_valid(self, sample_checkpoint):
        """Valid checkpoint should pass validation"""
        # Should not raise
        validate_checkpoint(sample_checkpoint)

    def test_validate_checkpoint_missing_file(self, artifacts):
        """Missing checkpoint file should raise error"""
        missing_path = artifacts.phase1_dir / "missing.ckpt"

        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint(missing_path)

        assert "does not exist" in str(exc_info.value)

    def test_validate_checkpoint_empty_file(self, artifacts):
        """Empty checkpoint file should raise error"""
        empty_path = artifacts.phase1_checkpoint
        empty_path.write_bytes(b"")

        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint(empty_path)

        assert "empty" in str(exc_info.value).lower()

    def test_validate_checkpoint_corrupted_file(self, artifacts):
        """Corrupted checkpoint file should raise error"""
        corrupted_path = artifacts.phase1_checkpoint
        corrupted_path.write_bytes(b"not a valid checkpoint")

        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint(corrupted_path)

        assert "load" in str(exc_info.value).lower()

    def test_validate_checkpoint_missing_model_state(self, artifacts):
        """Checkpoint without model_state_dict should raise error"""
        checkpoint = {
            "epoch": 10,
            "optimizer_state_dict": {},
            "loss": 0.123,
        }

        checkpoint_path = artifacts.phase1_checkpoint
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint(checkpoint_path)

        assert "model_state_dict" in str(exc_info.value)

    def test_validate_checkpoint_empty_model_state(self, artifacts):
        """Checkpoint with empty model_state_dict should raise error"""
        checkpoint = {
            "epoch": 10,
            "model_state_dict": {},  # Empty!
            "optimizer_state_dict": {},
        }

        checkpoint_path = artifacts.phase1_checkpoint
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(CheckpointValidationError) as exc_info:
            validate_checkpoint(checkpoint_path)

        assert "empty" in str(exc_info.value).lower()


class TestLogitsValidation:
    """Test logits and labels validation"""

    def test_validate_logits_valid(self, sample_logits, sample_labels):
        """Valid logits and labels should pass validation"""
        # Should not raise
        validate_logits(sample_logits, sample_labels)

    def test_validate_logits_missing_logits_file(self, artifacts, sample_labels):
        """Missing logits file should raise error"""
        missing_path = artifacts.phase1_dir / "missing_logits.npy"

        with pytest.raises(LogitsValidationError) as exc_info:
            validate_logits(missing_path, sample_labels)

        assert "does not exist" in str(exc_info.value)

    def test_validate_logits_missing_labels_file(self, sample_logits, artifacts):
        """Missing labels file should raise error"""
        missing_path = artifacts.phase1_dir / "missing_labels.npy"

        with pytest.raises(LogitsValidationError) as exc_info:
            validate_logits(sample_logits, missing_path)

        assert "does not exist" in str(exc_info.value)

    def test_validate_logits_empty_logits(self, artifacts, sample_labels):
        """Empty logits array should raise error"""
        empty_logits = np.array([])
        logits_path = artifacts.val_calib_logits
        np.save(logits_path, empty_logits)

        with pytest.raises(LogitsValidationError) as exc_info:
            validate_logits(logits_path, sample_labels)

        assert "empty" in str(exc_info.value).lower()

    def test_validate_logits_empty_labels(self, sample_logits, artifacts):
        """Empty labels array should raise error"""
        empty_labels = np.array([])
        labels_path = artifacts.val_calib_labels
        np.save(labels_path, empty_labels)

        with pytest.raises(LogitsValidationError) as exc_info:
            validate_logits(sample_logits, labels_path)

        assert "empty" in str(exc_info.value).lower()

    def test_validate_logits_wrong_logits_shape(self, artifacts, sample_labels):
        """Logits with wrong shape should raise error"""
        # 1D instead of 2D
        wrong_logits = np.random.randn(100).astype(np.float32)
        logits_path = artifacts.val_calib_logits
        np.save(logits_path, wrong_logits)

        with pytest.raises(LogitsValidationError) as exc_info:
            validate_logits(logits_path, sample_labels)

        assert "2D" in str(exc_info.value) or "shape" in str(exc_info.value).lower()

    def test_validate_logits_wrong_labels_shape(self, sample_logits, artifacts):
        """Labels with wrong shape should raise error"""
        # 2D instead of 1D
        wrong_labels = np.random.randint(0, 13, size=(100, 5)).astype(np.int64)
        labels_path = artifacts.val_calib_labels
        np.save(labels_path, wrong_labels)

        with pytest.raises(LogitsValidationError) as exc_info:
            validate_logits(sample_logits, labels_path)

        assert "1D" in str(exc_info.value) or "shape" in str(exc_info.value).lower()

    def test_validate_logits_mismatched_lengths(self, artifacts):
        """Logits and labels with mismatched lengths should raise error"""
        # Different number of samples
        logits = np.random.randn(100, 13).astype(np.float32)
        labels = np.random.randint(0, 13, size=50).astype(np.int64)  # Only 50!

        logits_path = artifacts.val_calib_logits
        labels_path = artifacts.val_calib_labels
        np.save(logits_path, logits)
        np.save(labels_path, labels)

        with pytest.raises(LogitsValidationError) as exc_info:
            validate_logits(logits_path, labels_path)

        assert "mismatch" in str(exc_info.value).lower()

    def test_validate_logits_labels_out_of_range(self, artifacts):
        """Labels with values outside valid range should raise error"""
        logits = np.random.randn(100, 13).astype(np.float32)
        labels = np.random.randint(0, 20, size=100).astype(np.int64)  # Values > 12!

        logits_path = artifacts.val_calib_logits
        labels_path = artifacts.val_calib_labels
        np.save(logits_path, logits)
        np.save(labels_path, labels)

        with pytest.raises(LogitsValidationError) as exc_info:
            validate_logits(logits_path, labels_path)

        assert "range" in str(exc_info.value).lower()


class TestBundleValidation:
    """Test bundle validation"""

    def test_validate_bundle_valid_threshold_policy(self, artifacts):
        """Valid bundle with threshold policy should pass"""
        # Create valid bundle
        bundle = {
            "model_checkpoint": "phase1/best_model.ckpt",
            "policy_type": "threshold",
            "policy_params": {"thresholds": [0.5] * 13},
            "num_classes": 13,
            "created_at": "2025-12-28T00:00:00Z",
        }

        bundle_path = artifacts.bundle_json
        bundle_path.write_text(json.dumps(bundle, indent=2))

        # Create policy file
        threshold_path = artifacts.thresholds_json
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text(json.dumps({"thresholds": [0.5] * 13}))

        # Should not raise
        validate_bundle(bundle_path)

    def test_validate_bundle_missing_file(self, artifacts):
        """Missing bundle file should raise error"""
        missing_path = artifacts.phase6_dir / "missing.json"

        with pytest.raises(BundleValidationError) as exc_info:
            validate_bundle(missing_path)

        assert "does not exist" in str(exc_info.value)

    def test_validate_bundle_empty_file(self, artifacts):
        """Empty bundle file should raise error"""
        bundle_path = artifacts.bundle_json
        bundle_path.write_text("")

        with pytest.raises(BundleValidationError) as exc_info:
            validate_bundle(bundle_path)

        assert "empty" in str(exc_info.value).lower()

    def test_validate_bundle_invalid_json(self, artifacts):
        """Invalid JSON should raise error"""
        bundle_path = artifacts.bundle_json
        bundle_path.write_text("not valid json")

        with pytest.raises(BundleValidationError) as exc_info:
            validate_bundle(bundle_path)

        assert "json" in str(exc_info.value).lower()

    def test_validate_bundle_missing_required_field(self, artifacts):
        """Bundle missing required field should raise error"""
        # Missing 'policy_type'
        bundle = {
            "model_checkpoint": "phase1/best_model.ckpt",
            "policy_params": {},
            "num_classes": 13,
        }

        bundle_path = artifacts.bundle_json
        bundle_path.write_text(json.dumps(bundle))

        with pytest.raises(BundleValidationError) as exc_info:
            validate_bundle(bundle_path)

        assert "policy_type" in str(exc_info.value).lower()

    def test_validate_bundle_invalid_policy_type(self, artifacts):
        """Bundle with invalid policy_type should raise error"""
        bundle = {
            "model_checkpoint": "phase1/best_model.ckpt",
            "policy_type": "invalid_policy",  # Not threshold/gate/scrc
            "policy_params": {},
            "num_classes": 13,
        }

        bundle_path = artifacts.bundle_json
        bundle_path.write_text(json.dumps(bundle))

        with pytest.raises(BundleValidationError) as exc_info:
            validate_bundle(bundle_path)

        assert "policy_type" in str(exc_info.value).lower()

    def test_validate_bundle_mutual_exclusivity_multiple_policies(self, artifacts):
        """Bundle with multiple policy files should raise error"""
        # Create valid bundle
        bundle = {
            "model_checkpoint": "phase1/best_model.ckpt",
            "policy_type": "threshold",
            "policy_params": {},
            "num_classes": 13,
        }

        bundle_path = artifacts.bundle_json
        bundle_path.write_text(json.dumps(bundle))

        # Create MULTIPLE policy files (violation!)
        threshold_path = artifacts.thresholds_json
        gate_path = artifacts.gateparams_json
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text(json.dumps({"thresholds": [0.5] * 13}))
        gate_path.write_text(json.dumps({"gate_params": {}}))

        with pytest.raises(BundleValidationError) as exc_info:
            validate_bundle(bundle_path)

        assert "MUTUAL EXCLUSIVITY" in str(exc_info.value)
        assert "multiple" in str(exc_info.value).lower()

    def test_validate_bundle_no_policy_file(self, artifacts):
        """Bundle with no corresponding policy file should raise error"""
        bundle = {
            "model_checkpoint": "phase1/best_model.ckpt",
            "policy_type": "threshold",
            "policy_params": {},
            "num_classes": 13,
        }

        bundle_path = artifacts.bundle_json
        bundle_path.write_text(json.dumps(bundle))

        # Don't create any policy files!

        with pytest.raises(BundleValidationError) as exc_info:
            validate_bundle(bundle_path)

        # Should fail because no policy file exists
        assert "exist" in str(exc_info.value).lower() or "found 0" in str(
            exc_info.value
        ).lower()


class TestValidatorsEdgeCases:
    """Test edge cases and error handling"""

    def test_validate_checkpoint_with_extra_fields(self, artifacts):
        """Checkpoint with extra fields should pass (backward compatibility)"""
        checkpoint = {
            "epoch": 10,
            "model_state_dict": {
                "layer1.weight": torch.randn(10, 10),
            },
            "optimizer_state_dict": {},
            "extra_field": "extra_value",  # Extra field
        }

        checkpoint_path = artifacts.phase1_checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Should not raise
        validate_checkpoint(checkpoint_path)

    def test_validate_logits_with_correct_num_classes(self, artifacts):
        """Logits with correct number of classes (13) should pass"""
        logits = np.random.randn(100, 13).astype(np.float32)  # 13 classes
        labels = np.random.randint(0, 13, size=100).astype(np.int64)

        logits_path = artifacts.val_calib_logits
        labels_path = artifacts.val_calib_labels
        np.save(logits_path, logits)
        np.save(labels_path, labels)

        # Should not raise
        validate_logits(logits_path, labels_path)
