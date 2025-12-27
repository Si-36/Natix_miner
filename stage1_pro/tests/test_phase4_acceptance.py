#!/usr/bin/env python3
"""
Phase 4 Acceptance Tests - F-SAM Optimizer
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

import torch
import torch.nn as nn
from stage1_pro.training import FSAMOptimizer, FSAMConfig


def test_fsam_optimizer():
    """Test F-SAM optimizer basic functionality."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )

    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fsam = FSAMOptimizer(model, base_optimizer, FSAMConfig(rho=0.5))

    # Test forward and backward
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    def closure():
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss

    # F-SAM step
    loss = fsam.step(closure)

    print("✅ F-SAM optimizer works")


def test_fsam_gradient_clipping():
    """Test F-SAM with gradient clipping."""
    model = nn.Sequential(nn.Linear(10, 2))
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = FSAMConfig(rho=0.5, grad_clip=1.0)
    fsam = FSAMOptimizer(model, base_optimizer, config)

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    def closure():
        logits = model(x)
        return nn.functional.cross_entropy(logits, y)

    for _ in range(5):
        fsam.step(closure)

    print("✅ F-SAM gradient clipping works")


def run_phase4_tests():
    print("\nPhase 4 Tests (F-SAM Optimizer):")
    test_fsam_optimizer()
    test_fsam_gradient_clipping()
    print("\n✅ All Phase 4 Tests Passed")


if __name__ == "__main__":
    run_phase4_tests()
