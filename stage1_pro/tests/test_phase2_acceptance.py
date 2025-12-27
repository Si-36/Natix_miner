#!/usr/bin/env python3
"""
Phase 2: Acceptance tests that must pass before Phase 3.

5 Tests:
1. Gate head architecture (3-head: cls, gate, aux)
2. Calibration modules (gate, scrc, dirichlet)
3. Risk training mode (selective + aux losses)
4. Schemas (gateparams, scrcparams)
5. Scripts (30_calibrate_scrc, 40_eval_selective)
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

import pytest
import torch
import numpy as np
from pathlib import Path
from stage1_pro.config import Stage1ProConfig
from stage1_pro.model import Stage1Head
from stage1_pro.calibration import GateCalibrator, SCRCCalibrator, DirichletCalibrator


def test_phase2_gate_head():
    """Test 1: Gate head architecture (3-head)."""
    head = Stage1Head(hidden_size=1024, dropout=0.3, use_gate=True)

    # Test forward pass
    dummy_features = torch.randn(4, 1024)
    outputs = head(dummy_features)

    assert "cls_logits" in outputs, "cls_logits not in outputs"
    assert "gate_logits" in outputs, "gate_logits not in outputs"
    assert "aux_logits" in outputs, "aux_logits not in outputs"

    assert outputs["cls_logits"].shape == (4, 2), (
        f"Expected (4, 2), got {outputs['cls_logits'].shape}"
    )
    assert outputs["gate_logits"].shape == (4,), (
        f"Expected (4,), got {outputs['gate_logits'].shape}"
    )
    assert outputs["aux_logits"].shape == (4, 2), (
        f"Expected (4, 2), got {outputs['aux_logits'].shape}"
    )

    # Test get_probs
    probs = head.get_probs(dummy_features)
    assert "cls_probs" in probs
    assert "gate_probs" in probs
    assert "aux_probs" in probs

    print("✅ Test 1 PASSED: Phase 2 gate head architecture")


def test_phase2_calibration_modules():
    """Test 2: Calibration modules."""
    # Test GateCalibrator
    gate_cal = GateCalibrator()
    dummy_gate_scores = torch.rand(100)
    dummy_labels = torch.randint(0, 2, (100,))

    gate_cal.fit(dummy_gate_scores, dummy_labels)
    calibrated = gate_cal.predict(dummy_gate_scores)

    assert calibrated.shape == dummy_gate_scores.shape, "Calibrated shape mismatch"
    assert gate_cal.fitted, "Calibrator not fitted"

    # Test SCRCCalibrator
    scrc_cal = SCRCCalibrator(alpha=0.02, n_bins=10)
    dummy_logits = torch.randn(100, 2)

    scrc_cal.fit(dummy_logits, dummy_labels)
    calibrated_probs, threshold = scrc_cal.predict(dummy_logits, target_fnr=0.02)

    assert calibrated_probs.shape == (100, 2), "SCRC calibrated probs shape mismatch"
    assert 0 <= threshold <= 1, f"Threshold out of range: {threshold}"
    assert scrc_cal.fitted, "SCRC calibrator not fitted"

    # Test DirichletCalibrator
    dirichlet_cal = DirichletCalibrator(alpha=1.0)
    dirichlet_cal.fit(dummy_logits, dummy_labels)
    calibrated = dirichlet_cal.predict(dummy_logits)

    assert calibrated.shape == (100, 2), "Dirichlet calibrated probs shape mismatch"

    print("✅ Test 2 PASSED: Phase 2 calibration modules")


def test_phase2_risk_training():
    """Test 3: Risk training mode setup."""
    config = Stage1ProConfig(
        train_image_dir="/tmp/train",
        train_labels_file="/tmp/train_labels.csv",
        val_image_dir="/tmp/val",
        val_labels_file="/tmp/val_labels.csv",
        use_dirichlet=True,  # Enable Phase 2
        calibration_iters=300,
    )

    assert config.use_dirichlet, "use_dirichlet should be True"
    assert config.calibration_iters == 300, "calibration_iters mismatch"

    from stage1_pro.training import Stage1Trainer
    from stage1_pro.model import DINOv3Backbone, Stage1Head

    backbone = DINOv3Backbone(
        model_name="facebook/dinov2-base", freeze=True, use_peft=False
    )

    head = Stage1Head(hidden_size=768, dropout=0.1, use_gate=True)

    class DummyModel(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, features=None, pixel_values=None):
            if pixel_values is not None:
                outputs = self.backbone(pixel_values)
                features = outputs.last_hidden_state[:, 0, :]
            return self.head(features)

    model = DummyModel(backbone, head)
    config.num_train_samples = 100

    trainer = Stage1Trainer(model, config)

    assert trainer.selective_loss is not None, "Selective loss not created"
    assert trainer.risk_loss is not None, "Risk loss not created"
    assert trainer.aux_loss is not None, "Aux loss not created"

    print("✅ Test 3 PASSED: Phase 2 risk training")


def test_phase2_schemas():
    """Test 4: Schemas exist and are valid."""
    schemas_dir = Path("/home/sina/projects/miner_b/stage1_pro/schemas")

    # Test gateparams schema
    gate_schema = schemas_dir / "gateparams.schema.json"
    assert gate_schema.exists(), f"{gate_schema} should exist"

    with open(gate_schema) as f:
        import json

        gate_schema_data = json.load(f)

    assert "$schema" in gate_schema_data, "Missing $schema in gateparams"
    assert "properties" in gate_schema_data, "Missing properties in gateparams"
    assert "a" in gate_schema_data["properties"], "Missing 'a' property"
    assert "b" in gate_schema_data["properties"], "Missing 'b' property"

    # Test scrcparams schema
    scrc_schema = schemas_dir / "scrcparams.schema.json"
    assert scrc_schema.exists(), f"{scrc_schema} should exist"

    with open(scrc_schema) as f:
        scrc_schema_data = json.load(f)

    assert "$schema" in scrc_schema_data, "Missing $schema in scrcparams"
    assert "alpha" in scrc_schema_data["properties"], "Missing 'alpha' property"
    assert "n_bins" in scrc_schema_data["properties"], "Missing 'n_bins' property"

    print("✅ Test 4 PASSED: Phase 2 schemas")


def test_phase2_scripts():
    """Test 5: Scripts exist."""
    scripts_dir = Path("/home/sina/projects/miner_b/stage1_pro/scripts")

    assert (scripts_dir / "30_calibrate_scrc.py").exists(), (
        "30_calibrate_scrc.py should exist"
    )
    assert (scripts_dir / "40_eval_selective.py").exists(), (
        "40_eval_selective.py should exist"
    )
    assert (scripts_dir / "50_export_bundle.py").exists(), (
        "50_export_bundle.py should exist"
    )

    print("✅ Test 5 PASSED: Phase 2 scripts")


def run_phase2_acceptance_tests():
    """Run all Phase 2 acceptance tests."""
    print("\n" + "=" * 80)
    print("PHASE 2 ACCEPTANCE TESTS")
    print("=" * 80)

    tests = [
        ("Gate Head Architecture", test_phase2_gate_head),
        ("Calibration Modules", test_phase2_calibration_modules),
        ("Risk Training Mode", test_phase2_risk_training),
        ("Schemas", test_phase2_schemas),
        ("Scripts", test_phase2_scripts),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL PHASE 2 TESTS PASSED - Ready for Phase 3!")
    else:
        print(f"❌ {failed} test(s) failed - Fix issues before Phase 3")
    print("=" * 80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_phase2_acceptance_tests()
    sys.exit(0 if success else 1)
