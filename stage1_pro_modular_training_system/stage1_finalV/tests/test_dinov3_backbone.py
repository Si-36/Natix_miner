"""
🧪 **Quick Test: DINOv3 Backbone**
Verify that DINOv3 small model loads correctly and is model-swap-proof
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.backbone import DINOv3Backbone


def test_dinov3_small():
    """Test DINOv3 small (vits16) - LOCAL GPU SIZE"""
    print("\n" + "=" * 70)
    print("TEST: DINOv3 Small (facebook/dinov3-vits16-pretrain-lvd1689m)")
    print("=" * 70)
    
    # Create backbone (small, 21M params, embed_dim 384)
    backbone = DINOv3Backbone(
        model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        dtype=torch.float16,
        freeze_backbone=False,
        compile_model=False,  # Skip compile for quick test
    )
    
    # Check dimensions (model-swap-proof)
    print(f"\n✅ Model-Swap-Proof Check:")
    print(f"   backbone.embed_dim = {backbone.embed_dim} (expected: 384)")
    print(f"   backbone.hidden_size = {backbone.hidden_size} (expected: 384)")
    assert backbone.embed_dim == 384, "❌ embed_dim should be 384"
    assert backbone.hidden_size == 384, "❌ hidden_size should be 384"
    print("   ✅ PASS")
    
    # Test forward pass
    print(f"\n✅ Forward Pass Test:")
    dummy_images = torch.randn(2, 3, 224, 224, dtype=torch.float16)
    
    # Preprocess (correct normalization for DINOv3)
    pixel_values = backbone.preprocess_images(
        [dummy_images[0], dummy_images[1]]  # Simulate PIL images
    )
    print(f"   Preprocessed shape: {pixel_values.shape}")
    
    # Extract features
    features = backbone.forward_features(pixel_values)
    print(f"   Features shape: {features.shape}")
    print(f"   Features dtype: {features.dtype}")
    
    # Check output shape
    assert features.shape == (2, 384), f"❌ Features shape should be (2, 384), got {features.shape}"
    print("   ✅ PASS")
    
    # Test head construction (model-swap-proof)
    print(f"\n✅ Head Construction Test:")
    from src.models.head import Stage1Head
    
    # Use backbone.embed_dim (NOT hard-coded 768)
    head = Stage1Head(backbone_dim=backbone.embed_dim, num_classes=2)
    print(f"   Head input dim: {head.backbone_dim} (from backbone.embed_dim)")
    print(f"   Head hidden dim: {head.hidden_dim}")
    print(f"   Head output dim: {head.num_classes}")
    print("   ✅ PASS - Model-swap-proof!")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - DINOv3 Backbone Ready!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        test_dinov3_small()
        print("\n✅ DINOv3 small model (21M params) ready for local training")
        print("✅ Model-swap-proof design allows easy upgrade to vitb16/vith16plus on SSH")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

