#!/usr/bin/env python3
"""
Phase 6 Acceptance Tests - Bootstrap Metrics
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

import numpy as np
import torch
from stage1_pro.metrics import (
    BootstrapECE,
    BootstrapConfig,
    compute_selective_metrics,
    compute_auroc,
    compute_precision_recall,
)


def test_bootstrap_ece():
    """Test Bootstrap ECE computation."""
    probs = np.random.rand(100, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.random.randint(0, 2, (100,))

    bootstrap = BootstrapECE(BootstrapConfig(n_bootstrap=20))
    results = bootstrap.compute(probs, labels)

    assert "ece_mean" in results
    assert "ece_original" in results
    assert f"ci_0.05" in results
    assert 0 <= results["ece_mean"] <= 1

    print("✅ BootstrapECE works")


def test_selective_metrics():
    """Test selective metrics computation."""
    probs = np.random.rand(100, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.random.randint(0, 2, (100,))

    results = compute_selective_metrics(probs, labels)

    assert "accuracy" in results
    assert "ece_mean" in results
    assert "threshold_sweep" in results
    assert len(results["threshold_sweep"]) > 0

    print("✅ Selective metrics work")


def test_auroc():
    """Test AUROC computation."""
    probs = np.random.rand(100, 2)
    labels = np.random.randint(0, 2, (100,))

    auroc = compute_auroc(probs, labels)

    assert 0 <= auroc <= 1

    print("✅ AUROC computation works")


def run_phase6_tests():
    print("\nPhase 6 Tests (Bootstrap Metrics):")
    test_bootstrap_ece()
    test_selective_metrics()
    test_auroc()
    print("\n✅ All Phase 6 Tests Passed")


if __name__ == "__main__":
    run_phase6_tests()
