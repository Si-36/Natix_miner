#!/usr/bin/env python3
"""
Phase 1: Acceptance tests that must pass before Phase 2.

5 Tests:
1. Config validation (Phase 1 constraints)
2. Data modules (splits, datasets)
3. Model modules (single-head architecture)
4. Training modules (trainer, optimizer, scheduler, EMA)
5. Scripts (00_make_splits, 20_train, 25_threshold_sweep, 50_export_bundle)
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

import pytest
import torch
import numpy as np
from pathlib import Path
import json
from stage1_pro.config import Stage1ProConfig
from stage1_pro.data import (
    create_val_splits,
    save_splits,
    load_splits,
    NATIXDataset,
    MultiRoadworkDataset,
    TimmStyleAugmentation,
)


def test_phase1_config_validation():
    """Test 1: Phase 1 config validation (constraints)."""
    config = Stage1ProConfig(
        train_image_dir="/tmp/train",
        train_labels_file="/tmp/train_labels.csv",
        val_image_dir="/tmp/val",
        val_labels_file="/tmp/val_labels.csv",
    )

    assert config.exit_policy == "softmax", "Phase 1 requires exit_policy='softmax'"
    assert config.target_fnr_exit == 0.02, "Phase 1 requires target_fnr_exit=0.02"
    assert not config.use_dirichlet, "Phase 1 requires use_dirichlet=False"
    assert config.peft_type == "none", "Phase 1 requires peft_type='none'"
    assert config.optimizer == "adamw", "Phase 1 requires optimizer='adamw'"
    assert config.gate_loss_weight == 0.0, "Phase 1 requires gate_loss_weight=0.0"

    print("✅ Test 1 PASSED: Phase 1 config validation")


def test_phase1_data_modules():
    """Test 2: Data modules (splits, datasets)."""
    # Test hash-based splits
    indices = np.arange(100)
    train_idx, val_select_idx, val_calib_idx = create_val_splits(
        indices, val_select_ratio=0.5, seed=42
    )

    assert len(val_select_idx) > 0, "val_select_idx should not be empty"
    assert len(val_calib_idx) > 0, "val_calib_idx should not be empty"
    assert len(train_idx) == len(indices), "train_idx should include all indices"
    assert len(val_select_idx) + len(val_calib_idx) == len(indices), (
        "Splits should cover all indices"
    )

    # Test deterministic splits
    train_idx2, val_select_idx2, val_calib_idx2 = create_val_splits(
        indices, val_select_ratio=0.5, seed=42
    )
    assert np.array_equal(val_select_idx, val_select_idx2), (
        "Splits should be deterministic"
    )

    print("✅ Test 2 PASSED: Phase 1 data modules")


def test_phase1_model_modules():
    """Test 3: Model modules (single-head architecture)."""
    from stage1_pro.model import Stage1Head

    # Test single-head architecture
    head = Stage1Head(hidden_size=1024, dropout=0.3)

    # Test forward pass
    dummy_features = torch.randn(4, 1024)
    logits = head(dummy_features)

    assert logits.shape == (4, 2), f"Expected logits shape (4, 2), got {logits.shape}"
    assert logits.dtype == torch.float32, "Logits should be float32"

    # Test architecture
    assert isinstance(head.head[0], torch.nn.Linear), "First layer should be Linear"
    assert isinstance(head.head[1], torch.nn.ReLU), "Second layer should be ReLU"
    assert isinstance(head.head[2], torch.nn.Dropout), "Third layer should be Dropout"
    assert isinstance(head.head[3], torch.nn.Linear), "Fourth layer should be Linear"

    print("✅ Test 3 PASSED: Phase 1 model modules")


def test_phase1_training_modules():
    """Test 4: Training modules (trainer, optimizer, scheduler, EMA)."""
    from stage1_pro.training import Stage1Trainer
    from stage1_pro.model import DINOv3Backbone, Stage1Head
    from torch.utils.data import DataLoader, TensorDataset

    # Build dummy model
    backbone = DINOv3Backbone(
        model_name="facebook/dinov2-base",  # Smaller model for testing
        freeze=True,
        use_peft=False,
    )
    head = Stage1Head(hidden_size=768, dropout=0.1)

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

    # Create config
    config = Stage1ProConfig(
        train_image_dir="/tmp/train",
        train_labels_file="/tmp/train_labels.csv",
        val_image_dir="/tmp/val",
        val_labels_file="/tmp/val_labels.csv",
        epochs=2,
        max_batch_size=4,
        lr_head=1e-4,
        use_ema=True,
        ema_decay=0.9999,
    )
    config.num_train_samples = 100

    # Build trainer
    trainer = Stage1Trainer(model, config)

    # Test optimizer
    assert trainer.optimizer is not None, "Optimizer should be created"
    assert isinstance(trainer.optimizer, torch.optim.AdamW), "Optimizer should be AdamW"

    # Test scheduler
    assert trainer.scheduler is not None, "Scheduler should be created"
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LambdaLR), (
        "Scheduler should be LambdaLR"
    )

    # Test EMA
    assert trainer.ema is not None, "EMA should be created"
    assert trainer.ema.decay == 0.9999, "EMA decay should be 0.9999"

    # Test one training step
    dummy_features = torch.randn(4, 768)
    dummy_labels = torch.randint(0, 2, (4,))
    dummy_dataset = TensorDataset(dummy_features, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)

    train_loss, train_acc = trainer.train_epoch(dummy_loader, epoch=1)
    assert isinstance(train_loss, float), "Train loss should be float"
    assert isinstance(train_acc, float), "Train acc should be float"

    print("✅ Test 4 PASSED: Phase 1 training modules")


def test_phase1_scripts():
    """Test 5: Scripts (imports and structure)."""
    import importlib.util

    scripts_dir = Path("/home/sina/projects/miner_b/stage1_pro/scripts")

    # Test script files exist
    assert (scripts_dir / "00_make_splits.py").exists(), (
        "00_make_splits.py should exist"
    )
    assert (scripts_dir / "20_train_riskaware.py").exists(), (
        "20_train_riskaware.py should exist"
    )
    assert (scripts_dir / "25_threshold_sweep.py").exists(), (
        "25_threshold_sweep.py should exist"
    )
    assert (scripts_dir / "50_export_bundle.py").exists(), (
        "50_export_bundle.py should exist"
    )

    # Test schema files exist
    schemas_dir = Path("/home/sina/projects/miner_b/stage1_pro/schemas")
    assert (schemas_dir / "thresholds.schema.json").exists(), (
        "thresholds.schema.json should exist"
    )
    assert (schemas_dir / "bundle.schema.json").exists(), (
        "bundle.schema.json should exist"
    )

    print("✅ Test 5 PASSED: Phase 1 scripts")


def run_phase1_acceptance_tests():
    """Run all Phase 1 acceptance tests."""
    print("\n" + "=" * 80)
    print("PHASE 1 ACCEPTANCE TESTS")
    print("=" * 80)

    tests = [
        ("Config Validation", test_phase1_config_validation),
        ("Data Modules", test_phase1_data_modules),
        ("Model Modules", test_phase1_model_modules),
        ("Training Modules", test_phase1_training_modules),
        ("Scripts", test_phase1_scripts),
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
        print("✅ ALL PHASE 1 TESTS PASSED - Ready for Phase 2!")
    else:
        print(f"❌ {failed} test(s) failed - Fix issues before Phase 2")
    print("=" * 80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_phase1_acceptance_tests()
    sys.exit(0 if success else 1)
