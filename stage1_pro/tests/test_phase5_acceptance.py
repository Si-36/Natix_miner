#!/usr/bin/env python3
"""
Phase 5 Acceptance Tests - Domain Adaptation
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

from PIL import Image
import torch
from stage1_pro.domain_adaptation import ExploraAugmentation, DomainAwareTransform


def test_explora_augmentation():
    """Test Explora augmentation."""
    aug = ExploraAugmentation(img_size=224)

    img = Image.new("RGB", (256, 256), color="red")

    # Test augmentation
    result = aug(img)

    assert result.shape == (3, 224, 224), f"Expected (3, 224, 224), got {result.shape}"
    assert isinstance(result, torch.Tensor)

    print("✅ ExploraAugmentation works")


def test_domain_aware_transform():
    """Test domain-aware augmentation."""
    explora = ExploraAugmentation(img_size=224)
    domain_aug = DomainAwareTransform(explora)

    img = Image.new("RGB", (256, 256), color="blue")

    # Test with different domains
    result_natix = domain_aug(img, domain="natix")
    result_roadwork = domain_aug(img, domain="roadwork")

    assert result_natix.shape == (3, 224, 224)
    assert result_roadwork.shape == (3, 224, 224)

    print("✅ DomainAwareTransform works")


def run_phase5_tests():
    print("\nPhase 5 Tests (Domain Adaptation):")
    test_explora_augmentation()
    test_domain_aware_transform()
    print("\n✅ All Phase 5 Tests Passed")


if __name__ == "__main__":
    run_phase5_tests()
