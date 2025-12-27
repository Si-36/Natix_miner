"""
Test: Gate Calibration - Phase 3.7-3.10 (OFFLINE - no HF downloads)

Tests Platt scaling, gate calibration, and threshold selection.
"""

import torch
import sys
import os

# Add stage1_pro_modular_training_system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage1_pro_modular_training_system'))

import numpy as np
from sklearn.linear_model import LogisticRegression


def test_gate_calibration():
    """
    TEST 1: Verify gate calibration with Platt scaling
    """
    print("\n" + "="*80)
    print("TEST 1: Gate Calibration (Platt Scaling)")
    print("="*80)
    
    # Import inside test function
    from scripts import calibrate_gate
    fit_platt_scaling = calibrate_gate.fit_platt_scaling
    calibrate_gate_fn = calibrate_gate.calibrate_gate
    
    # Create dummy data
    gate_logits = torch.randn(100, 1)
    labels = torch.randint(0, 2, (100,))
    
    # Fit Platt scaling
    platt_model = fit_platt_scaling(
        gate_logits,
        labels,
        verbose=True
    )
    
    # Calibrate gate
    calibrated_gate_prob = calibrate_gate_fn(
        gate_logits,
        platt_model,
        verbose=True
    )
    
    # Verify calibrated probabilities are in [0, 1]
    assert calibrated_gate_prob.min() >= 0.0, f"Min prob: {calibrated_gate_prob.min()}"
    assert calibrated_gate_prob.max() <= 1.0, f"Max prob: {calibrated_gate_prob.max()}"
    
    # Verify shape
    assert calibrated_gate_prob.shape == (100,), f"Shape: {calibrated_gate_prob.shape}"
    
    print("✅ Gate calibration verified")
    print(f"   Calibrated prob shape: {calibrated_gate_prob.shape}")
    print(f"   Calibrated prob range: [{calibrated_gate_prob.min():.4f}, {calibrated_gate_prob.max():.4f}]")
    
    return True


def test_gate_threshold_selection():
    """
    TEST 2: Verify gate threshold selection
    """
    print("\n" + "="*80)
    print("TEST 2: Gate Threshold Selection")
    print("="*80)
    
    # Import inside test function
    from scripts import calibrate_gate
    compute_exit_mask = calibrate_gate.compute_exit_mask
    compute_fnr_on_exited = calibrate_gate.compute_fnr_on_exited
    select_gate_threshold = calibrate_gate.select_gate_threshold
    
    # Create dummy data
    classifier_logits = torch.randn(100, 2)
    gate_prob = np.random.rand(100)
    labels = torch.randint(0, 2, (100,))
    
    # Compute exit mask
    exit_mask = compute_exit_mask(gate_prob, 0.5, None)
    assert exit_mask.shape == (100,), f"Exit mask shape: {exit_mask.shape}"
    assert exit_mask.dtype == bool, f"Exit mask dtype: {exit_mask.dtype}"
    
    # Compute FNR on exited
    fnr_on_exited = compute_fnr_on_exited(
        classifier_logits,
        labels,
        exit_mask,
        verbose=True
    )
    
    assert fnr_on_exited >= 0.0 and fnr_on_exited <= 1.0, f"FNR: {fnr_on_exited}"
    
    # Select gate threshold
    gate_threshold, best_metrics = select_gate_threshold(
        classifier_logits,
        gate_prob,
        labels,
        target_fnr_exit=0.02,
        verbose=True
    )
    
    assert gate_threshold >= 0.5 and gate_threshold <= 0.95, f"Gate threshold: {gate_threshold}"
    assert 'coverage' in best_metrics, f"Missing 'coverage' in metrics"
    assert 'fnr_on_exited' in best_metrics, f"Missing 'fnr_on_exited' in metrics"
    assert 'exit_accuracy' in best_metrics, f"Missing 'exit_accuracy' in metrics"
    
    print("✅ Gate threshold selection verified")
    print(f"   Gate threshold: {gate_threshold}")
    print(f"   Coverage: {best_metrics['coverage']:.4f}")
    print(f"   FNR on exited: {best_metrics['fnr_on_exited']:.4f}")
    
    return True


def test_gateparams_json():
    """
    TEST 3: Verify gateparams.json saving
    """
    print("\n" + "="*80)
    print("TEST 3: Gateparams.json Saving")
    print("="*80)
    
    # Import inside test function
    from scripts import calibrate_gate
    save_gateparams_json = calibrate_gate.save_gateparams_json
    
    # Create dummy metrics
    gate_threshold = 0.85
    metrics = {
        'fnr_on_exited': 0.015,
        'coverage': 0.75,
        'exit_accuracy': 0.92,
        'exit_samples': 75
    }
    
    # Save gateparams.json
    import tempfile
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        gateparams_path = save_gateparams_json(
            gate_threshold,
            metrics,
            [],
            tmpdir,
            verbose=True
        )
        
        # Load and verify
        with open(gateparams_path, 'r') as f:
            gateparams = json.load(f)
        
        assert gateparams['active_exit_policy'] == 'gate', f"Active exit policy: {gateparams['active_exit_policy']}"
        assert gateparams['gate_threshold'] == gate_threshold, f"Gate threshold: {gateparams['gate_threshold']}"
        assert gateparams['fnr_on_exited'] == metrics['fnr_on_exited'], f"FNR: {gateparams['fnr_on_exited']}"
        assert gateparams['coverage'] == metrics['coverage'], f"Coverage: {gateparams['coverage']}"
        assert gateparams['exit_accuracy'] == metrics['exit_accuracy'], f"Exit accuracy: {gateparams['exit_accuracy']}"
        
        print("✅ Gateparams.json verified")
        print(f"   Active exit policy: {gateparams['active_exit_policy']}")
        print(f"   Gate threshold: {gateparams['gate_threshold']}")
        print(f"   Coverage: {gateparams['coverage']}")
        print(f"   FNR on exited: {gateparams['fnr_on_exited']}")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OFFLINE TESTS: Phase 3 Gate Calibration")
    print("="*80)
    
    test1_pass = test_gate_calibration()
    test2_pass = test_gate_threshold_selection()
    test3_pass = test_gateparams_json()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"TEST 1 (Gate Calibration): {'PASS' if test1_pass else 'FAIL'}")
    print(f"TEST 2 (Threshold Selection): {'PASS' if test2_pass else 'FAIL'}")
    print(f"TEST 3 (Gateparams.json): {'PASS' if test3_pass else 'FAIL'}")
    print("="*80)
    
    if all([test1_pass, test2_pass, test3_pass]):
        print("\n✅ ALL TESTS PASSED")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        exit(1)
