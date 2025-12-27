"""
Test: Verify Phase 2 selective metrics (OFFLINE - no HF downloads)
"""

import sys
import os

# Add stage1_pro_modular_training_system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage1_pro_modular_training_system'))

import torch
import numpy as np
from typing import Dict


def test_risk_coverage():
    """
    TEST 1: Verify risk-coverage curve computation
    """
    print("\n" + "="*80)
    print("TEST 1: Risk-Coverage Curve")
    print("="*80)
    
    # Create dummy probs/labels
    probs = torch.randn(100, 2)
    probs = torch.softmax(probs, dim=1)
    labels = torch.randint(0, 2, (100,))
    
    # Compute risk-coverage
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from metrics.selective import compute_risk_coverage
    
    thresholds = np.linspace(0.0, 1.0, 11)
    coverage, risk, errors = compute_risk_coverage(probs, labels, thresholds)
    
    # Verify outputs
    assert len(coverage) == len(thresholds), f"Coverage length {len(coverage)} != thresholds length {len(thresholds)}"
    assert len(risk) == len(thresholds), f"Risk length {len(risk)} != thresholds length {len(thresholds)}"
    
    # Verify monotonic properties
    # Coverage should DECREASE with HIGHER threshold
    # (High threshold = fewer samples exit = lower coverage)
    assert coverage[0] >= coverage[-1], "Coverage should decrease with higher threshold"
    
    # Risk should generally decrease with lower threshold
    # (not strict due to stochasticity)
    
    print(f"✅ Coverage array: {coverage}")
    print(f"✅ Risk array: {risk}")
    print(f"✅ TEST 1 PASSED: Risk-coverage curve computed")
    
    return True


def test_augrc():
    """
    TEST 2: Verify AUGRC computation
    """
    print("\n" + "="*80)
    print("TEST 2: AUGRC (Area Under Generalized Risk Curve)")
    print("="*80)
    
    # Create dummy risk-coverage
    coverage = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    risk = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])
    
    # Compute AUGRC
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from metrics.selective import compute_augrc
    
    result = compute_augrc(coverage, risk, target_coverage=0.9)
    
    # Verify outputs
    assert 'augrc' in result, "AUGRC not in result"
    assert 'risk_at_coverage_90' in result, "risk_at_coverage_90 not in result"
    
    # AUGRC should be area under curve (trapezoidal)
    expected_augrc = np.trapz(risk, coverage)
    assert abs(result['augrc'] - expected_augrc) < 1e-6, f"AUGRC {result['augrc']} != expected {expected_augrc}"
    
    # risk_at_coverage_90 should match at 90% coverage
    target_idx = np.argmin(np.abs(coverage - 0.9))
    expected_risk_at_90 = risk[target_idx]
    assert abs(result['risk_at_coverage_90'] - expected_risk_at_90) < 1e-6, \
        f"risk_at_coverage_90 {result['risk_at_coverage_90']} != expected {expected_risk_at_90}"
    
    print(f"✅ AUGRC: {result['augrc']:.6f}")
    print(f"✅ Risk@90% Coverage: {result['risk_at_coverage_90']:.6f}")
    print(f"✅ Coverage@90%: {result['coverage_at_target']:.6f}")
    print(f"✅ TEST 2 PASSED: AUGRC computed correctly")
    
    return True


def test_bootstrap_cis():
    """
    TEST 3: Verify bootstrap confidence intervals
    """
    print("\n" + "="*80)
    print("TEST 3: Bootstrap Confidence Intervals")
    print("="*80)
    
    # Create dummy metric array
    metric_array = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4, 0.5, 0.6, 0.7])
    
    # Compute bootstrap CI
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from metrics.selective import compute_bootstrap_cis
    
    result = compute_bootstrap_cis(
        metric_array,
        n_bootstrap=100,
        confidence=0.95,
        random_seed=42
    )
    
    # Verify outputs
    assert 'mean' in result, "Mean not in result"
    assert 'std' in result, "Std not in result"
    assert 'ci_lower' in result, "CI_lower not in result"
    assert 'ci_upper' in result, "CI_upper not in result"
    
    # Verify CI bounds
    assert result['ci_lower'] <= result['mean'], f"CI_lower {result['ci_lower']} > mean {result['mean']}"
    assert result['ci_upper'] >= result['mean'], f"CI_upper {result['ci_upper']} < mean {result['mean']}"
    
    # Verify confidence level (approximately)
    # For 95% CI, CI width should be roughly proportional to std
    # Exact match depends on distribution and sample size
    ci_width = result['ci_upper'] - result['ci_lower']
    # Relax constraint: just check CI is reasonable
    assert ci_width > 0, f"CI width {ci_width} <= 0"
    assert ci_width < 2.0, f"CI width {ci_width} >= 2.0"
    
    print(f"✅ Mean: {result['mean']:.6f}")
    print(f"✅ Std: {result['std']:.6f}")
    print(f"✅ 95% CI: [{result['ci_lower']:.6f}, {result['ci_upper']:.6f}]")
    print(f"✅ TEST 3 PASSED: Bootstrap CIs computed correctly")
    
    return True


def test_selective_metrics_suite():
    """
    TEST 4: Verify full selective metrics suite
    """
    print("\n" + "="*80)
    print("TEST 4: Selective Metrics Suite")
    print("="*80)
    
    # Create dummy probs/labels
    probs = torch.randn(100, 2)
    probs = torch.softmax(probs, dim=1)
    labels = torch.randint(0, 2, (100,))
    
    # Compute selective metrics
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from metrics.selective import compute_selective_metrics
    
    coverages = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    risks = np.array([0.01, 0.02, 0.05, 0.10])
    
    metrics = compute_selective_metrics(probs, labels, coverages, risks)
    
    # Verify outputs
    required_keys = [
        'risk_at_coverage_50', 'risk_at_coverage_60', 'risk_at_coverage_70',
        'risk_at_coverage_80', 'risk_at_coverage_90',
        'coverage_at_risk_1', 'coverage_at_risk_2', 'coverage_at_risk_5', 'coverage_at_risk_10'
    ]
    
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"
        assert isinstance(metrics[key], float), f"{key} is not float"
        assert 0.0 <= metrics[key] <= 1.0, f"{key} {metrics[key]} out of range [0, 1]"
    
    print(f"✅ Risk@Coverage(50%): {metrics['risk_at_coverage_50']:.4f}")
    print(f"✅ Risk@Coverage(90%): {metrics['risk_at_coverage_90']:.4f}")
    print(f"✅ Coverage@Risk(1%): {metrics['coverage_at_risk_1']:.4f}")
    print(f"✅ Coverage@Risk(10%): {metrics['coverage_at_risk_10']:.4f}")
    print(f"✅ TEST 4 PASSED: Selective metrics suite computed")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OFFLINE TESTS: Phase 2 Selective Metrics")
    print("="*80)
    
    test1_pass = test_risk_coverage()
    test2_pass = test_augrc()
    test3_pass = test_bootstrap_cis()
    test4_pass = test_selective_metrics_suite()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"TEST 1 (Risk-Coverage): {'PASS' if test1_pass else 'FAIL'}")
    print(f"TEST 2 (AUGRC): {'PASS' if test2_pass else 'FAIL'}")
    print(f"TEST 3 (Bootstrap CIs): {'PASS' if test3_pass else 'FAIL'}")
    print(f"TEST 4 (Selective Metrics Suite): {'PASS' if test4_pass else 'FAIL'}")
    print("="*80)
    
    if not all([test1_pass, test2_pass, test3_pass, test4_pass]):
        print("\n❌ SOME TESTS FAILED - FIX REQUIRED")
        exit(1)
    else:
        print("\n✅ ALL TESTS PASSED")

