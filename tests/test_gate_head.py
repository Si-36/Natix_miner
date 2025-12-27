"""
Test: Gate Head Module - Phase 3: Gate Head Architecture (OFFLINE - no HF downloads)

Tests 3-head design and selective loss computation.
"""

import torch
import sys
import os

# Add stage1_pro_modular_training_system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage1_pro_modular_training_system'))

import numpy as np


def test_gate_head_architecture():
    """
    TEST 1: Verify 3-head architecture
    """
    print("\n" + "="*80)
    print("TEST 1: Gate Head Architecture")
    print("="*80)
    
    from model.gate_head import GateHead
    
    # Create gate head
    gate_head = GateHead(
        backbone_dim=768,
        num_classes=2,
        gate_hidden_dim=128,
        use_ema=False,
        device="cpu",
        verbose=True
    )
    
    # Verify architecture
    # Classifier: Linear(768, 768) -> ReLU -> Linear(768, 2)
    assert gate_head.classifier[0].in_features == 768, f"Classifier layer 0 input: {gate_head.classifier[0].in_features}"
    assert gate_head.classifier[0].out_features == 768, f"Classifier layer 0 output: {gate_head.classifier[0].out_features}"
    assert gate_head.classifier[2].in_features == 768, f"Classifier layer 2 input: {gate_head.classifier[2].in_features}"
    assert gate_head.classifier[2].out_features == 2, f"Classifier layer 2 output: {gate_head.classifier[2].out_features}"
    
    # Gate: Linear(768, 128) -> ReLU -> Linear(128, 1)
    assert gate_head.gate[0].in_features == gate_head.backbone_dim, f"Gate layer 0 input: {gate_head.gate[0].in_features}"
    assert gate_head.gate[0].out_features == gate_head.gate_hidden_dim, f"Gate layer 0 output: {gate_head.gate[0].out_features}"
    assert gate_head.gate[2].in_features == gate_head.gate_hidden_dim, f"Gate layer 2 input: {gate_head.gate[2].in_features}"
    assert gate_head.gate[2].out_features == 1, f"Gate layer 2 output: {gate_head.gate[2].out_features}"
    
    # Head: Linear(768, 128) -> ReLU -> Linear(128, 2)
    assert gate_head.head[0].in_features == gate_head.backbone_dim, f"Head layer 0 input: {gate_head.head[0].in_features}"
    assert gate_head.head[0].out_features == gate_head.gate_hidden_dim, f"Head layer 0 output: {gate_head.head[0].out_features}"
    assert gate_head.head[2].in_features == gate_head.gate_hidden_dim, f"Head layer 2 input: {gate_head.head[2].in_features}"
    assert gate_head.head[2].out_features == gate_head.num_classes, f"Head layer 2 output: {gate_head.head[2].out_features}"
    
    print("✅ Gate Head architecture verified")
    print(f"   Head 1: 768 -> 768 -> 2")
    print(f"   Gate: {gate_head.backbone_dim} -> {gate_head.gate_hidden_dim} -> 1")
    print(f"   Head: {gate_head.backbone_dim} -> {gate_head.gate_hidden_dim} -> {gate_head.num_classes}")
    
    return True


def test_gate_head_forward():
    """
    TEST 2: Verify forward pass
    """
    print("\n" + "="*80)
    print("TEST 2: Gate Head Forward Pass")
    print("="*80)
    
    from model.gate_head import GateHead
    
    # Create gate head
    gate_head = GateHead(
        backbone_dim=768,
        num_classes=2,
        gate_hidden_dim=128,
        use_ema=False,
        device="cpu",
        verbose=True
    )
    
    # Create dummy input
    features = torch.randn(4, 768)
    
    # Forward pass (default: return all outputs)
    outputs = gate_head(features)
    
    # Verify outputs
    assert outputs['classifier_logits'].shape == (4, 2), f"Classifier logits shape: {outputs['classifier_logits'].shape}"
    assert outputs['gate_logits'].shape == (4,), f"Gate logits shape: {outputs['gate_logits'].shape}"
    assert outputs['head_logits'].shape == (4, 2), f"Head logits shape: {outputs['head_logits'].shape}"
    
    print("✅ Gate Head forward pass verified")
    print(f"   Classifier logits: {outputs['classifier_logits'].shape}")
    print(f"   Gate logits: {outputs['gate_logits'].shape}")
    print(f"   Head logits: {outputs['head_logits'].shape}")
    
    return True


def test_selective_loss():
    """
    TEST 3: Verify selective loss computation
    """
    print("\n" + "="*80)
    print("TEST 3: Selective Loss")
    print("="*80)
    
    from model.gate_head import compute_selective_loss
    
    # Create dummy data
    classifier_logits = torch.randn(4, 2)
    gate_logits = torch.randn(4, 1)
    labels = torch.randint(0, 2, (4,))
    gate_exit_mask = torch.tensor([1, 1, 0, 1], dtype=torch.float32)
    gate_threshold = 0.5
    coverage_weight = 1.0
    
    # Compute selective loss
    loss = compute_selective_loss(
        classifier_logits,
        gate_logits,
        labels,
        gate_exit_mask,
        gate_threshold,
        coverage_weight,
        verbose=True
    )
    
    # Verify loss is scalar
    assert loss.dim() == 0, f"Loss shape: {loss.shape}"
    assert loss.numel() == 1, f"Loss elements: {loss.numel()}"
    
    print("✅ Selective loss verified")
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Type: {loss.dtype}")
    
    return True


def test_auxiliary_loss():
    """
    TEST 4: Verify auxiliary loss
    """
    print("\n" + "="*80)
    print("TEST 4: Auxiliary Loss")
    print("="*80)
    
    from model.gate_head import compute_auxiliary_loss
    
    # Create dummy data
    head_logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    
    # Compute auxiliary loss
    loss = compute_auxiliary_loss(head_logits, labels, verbose=True)
    
    # Verify loss is scalar
    assert loss.dim() == 0, f"Loss shape: {loss.shape}"
    assert loss.numel() == 1, f"Loss elements: {loss.numel()}"
    
    print("✅ Auxiliary loss verified")
    print(f"   Loss: {loss.item():.6f}")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OFFLINE TESTS: Phase 3 Gate Head")
    print("="*80)
    
    test1_pass = test_gate_head_architecture()
    test2_pass = test_gate_head_forward()
    test3_pass = test_selective_loss()
    test4_pass = test_auxiliary_loss()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"TEST 1 (Architecture): {'PASS' if test1_pass else 'FAIL'}")
    print(f"TEST 2 (Forward Pass): {'PASS' if test2_pass else 'FAIL'}")
    print(f"TEST 3 (Selective Loss): {'PASS' if test3_pass else 'FAIL'}")
    print(f"TEST 4 (Auxiliary Loss): {'PASS' if test4_pass else 'FAIL'}")
    print("="*80)
    
    if all([test1_pass, test2_pass, test3_pass, test4_pass]):
        print("\n✅ ALL TESTS PASSED")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        exit(1)
