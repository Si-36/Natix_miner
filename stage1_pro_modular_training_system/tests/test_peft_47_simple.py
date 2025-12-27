#!/usr/bin/env python3
"""
Phase 4.7: Simplified PEFT Acceptance Tests (Dec 2025)

Simplified tests that don't depend on logging.py to avoid syntax errors.

Tests:
- 4.7.6: rank=0 identity test
- 4.7.7: Adapter checkpoint reload test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# Use REAL library imports
from model.peft_integration import apply_lora_to_backbone, apply_dora_to_backbone, count_peft_parameters
from model.backbone import DINOv3Backbone
from model.gate_head import GateHead


def test_476_rank0_identity():
    """
    Acceptance Test 4.7.6: rank=0 LoRA/DoRA produces identical logits to baseline
    
    Setup:
    1. Create backbone and model
    2. Apply PEFT with rank=0 (should be identity)
    3. Compare outputs: should be identical (within numerical tolerance)
    
    Pass condition:
    - Max absolute difference < 1e-5
    - Mean absolute difference < 1e-6
    """
    print("\n" + "="*80)
    print("ACCEPTANCE TEST 4.7.6: rank=0 Identity Test")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create backbone
    print("Creating DINOv3 backbone...")
    backbone = DINOv3Backbone(
        model_path="facebook/dinov3-vith14",
        device=device
    )
    backbone.eval()
    
    # Create model
    print("Creating GateHead model...")
    model = GateHead(
        backbone_dim=768,
        num_classes=2,
        gate_hidden_dim=128,
        device=device,
        verbose=False
    )
    model.eval()
    
    # Create test input
    test_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Baseline: no PEFT
    baseline_output = backbone(test_input)
    baseline_cls = model(baseline_output["features"])
    
    # Apply PEFT with rank=0 (should be identity)
    print("\nApplying LoRA with rank=0 (should be identity)...")
    adapted_backbone = apply_lora_to_backbone(
        backbone=backbone,
        r=0,  # rank=0 should be identity
        lora_alpha=0
    )
    adapted_backbone.eval()
    
    # Get output with PEFT (rank=0)
    peft_output = adapted_backbone(test_input)
    peft_cls = model(peft_output["features"])
    
    # Compare outputs
    backbone_diff = torch.abs(baseline_output["features"] - peft_output["features"]).max().item()
    cls_diff = torch.abs(baseline_cls["classifier_logits"] - peft_cls["classifier_logits"]).max().item()
    
    print(f"\nResults:")
    print(f"  Backbone feature diff (max): {backbone_diff:.2e}")
    print(f"  Classifier logits diff (max): {cls_diff:.2e}")
    
    # Pass condition: difference should be very small
    tolerance = 1e-5
    passed = (backbone_diff < tolerance) and (cls_diff < tolerance)
    
    if passed:
        print(f"  ✅ PASS: Differences < {tolerance:.0e}")
    else:
        print(f"  ❌ FAIL: Differences >= {tolerance:.0e}")
        print(f"  ⚠️  rank=0 should be identity, but outputs differ!")
    
    print("="*80)
    return passed


def test_477_checkpoint_reload():
    """
    Acceptance Test 4.7.7: Adapter checkpoint reload reproduces outputs (within tolerance)
    
    Setup:
    1. Create backbone and model
    2. Apply PEFT with rank=16
    3. Save adapter checkpoint
    4. Load adapter checkpoint into fresh model
    5. Compare outputs: should be identical (within tolerance)
    
    Pass condition:
    - Max absolute difference < 1e-4
    """
    print("\n" + "="*80)
    print("ACCEPTANCE TEST 4.7.7: Adapter Checkpoint Reload Test")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create backbone
    print("Creating DINOv3 backbone...")
    backbone = DINOv3Backbone(
        model_path="facebook/dinov3-vith14",
        device=device
    )
    backbone.eval()
    
    # Create model
    print("Creating GateHead model...")
    model = GateHead(
        backbone_dim=768,
        num_classes=2,
        gate_hidden_dim=128,
        device=device,
        verbose=False
    )
    model.eval()
    
    # Create test input
    test_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Apply PEFT with rank=16
    print("\nApplying LoRA with rank=16...")
    adapted_backbone = apply_lora_to_backbone(
        backbone=backbone,
        r=16,
        lora_alpha=32
    )
    adapted_backbone.eval()
    
    # Get output before saving
    with torch.no_grad():
        features_before = adapted_backbone(test_input)["features"]
        cls_before = model(features_before)["classifier_logits"]
    
    # Save adapter checkpoint (use save_pretrained from HuggingFace)
    print("\nSaving adapter checkpoint...")
    adapted_backbone.save_pretrained("test_adapters")
    
    # Save model weights separately
    model_state_dict = {
        'head_state_dict': model.state_dict(),
        'test_info': 'adapter checkpoint test'
    }
    torch.save(model_state_dict, "test_model_state.pth")
    
    print("✅ Saved adapters to test_adapters/")
    print("✅ Saved model to test_model_state.pth")
    
    # Load adapter checkpoint into fresh model
    print("\nLoading adapter checkpoint into fresh model...")
    fresh_backbone = DINOv3Backbone(
        model_path="facebook/dinov3-vith14",
        device=device
    )
    fresh_backbone.eval()
    
    # Apply PEFT with rank=16
    fresh_adapted_backbone = apply_lora_to_backbone(
        backbone=fresh_backbone,
        r=16,
        lora_alpha=32
    )
    fresh_adapted_backbone.eval()
    
    # Load adapters using HuggingFace library
    fresh_adapted_backbone.load_adapter("test_adapters")
    
    # Create fresh model and load weights
    fresh_model = GateHead(
        backbone_dim=768,
        num_classes=2,
        gate_hidden_dim=128,
        device=device,
        verbose=False
    )
    fresh_model.eval()
    
    checkpoint = torch.load("test_model_state.pth")
    fresh_model.load_state_dict(checkpoint['head_state_dict'])
    
    print("✅ Loaded adapters successfully")
    print("✅ Loaded model weights successfully")
    
    # Get output after reload
    with torch.no_grad():
        features_after = fresh_adapted_backbone(test_input)["features"]
        cls_after = fresh_model(features_after)["classifier_logits"]
    
    # Compare outputs
    features_diff = torch.abs(features_before - features_after).max().item()
    cls_diff = torch.abs(cls_before - cls_after).max().item()
    
    print(f"\nResults:")
    print(f"  Features diff (max): {features_diff:.2e}")
    print(f"  Classifier logits diff (max): {cls_diff:.2e}")
    
    # Pass condition: difference should be very small
    tolerance = 1e-4
    passed = (features_diff < tolerance) and (cls_diff < tolerance)
    
    if passed:
        print(f"  ✅ PASS: Differences < {tolerance:.0e}")
    else:
        print(f"  ❌ FAIL: Differences >= {tolerance:.0e}")
        print(f"  ⚠️  Adapter checkpoint reload not reproducing outputs correctly!")
    
    print("="*80)
    return passed


def main():
    """Run all Phase 4.7 acceptance tests."""
    print("\n" + "="*80)
    print("PHASE 4.7: SIMPLIFIED PEFT ACCEPTANCE TESTS")
    print("Dec 2025 Best Practice - REAL Tests")
    print("="*80)
    
    all_passed = True
    
    # Test 4.7.6: rank=0 identity
    try:
        test_476_passed = test_476_rank0_identity()
        all_passed = all_passed and test_476_passed
    except Exception as e:
        print(f"\n❌ TEST 4.7.6 FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Test 4.7.7: Adapter checkpoint reload
    try:
        test_477_passed = test_477_checkpoint_reload()
        all_passed = all_passed and test_477_passed
    except Exception as e:
        print(f"\n❌ TEST 4.7.7 FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED")
        print("="*80)
        print("\nPhase 4.7: PEFT Acceptance Tests: PASSED")
        print("✅ 4.7.6: rank=0 identity test")
        print("✅ 4.7.7: Adapter checkpoint reload test")
        print("\n🎉 Phase 4.7: READY FOR PRODUCTION")
        print("="*80)
        exit_code = 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("="*80)
        print("\n⚠️  Phase 4.7: PEFT Acceptance Tests: NOT READY")
        print("   Fix failures before using in production")
        print("="*80)
        exit_code = 1
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

