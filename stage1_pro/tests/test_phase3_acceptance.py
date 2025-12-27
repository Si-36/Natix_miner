#!/usr/bin/env python3
"""Phase 3 Acceptance Tests"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

from stage1_pro.model.peft import DoRAAdapter, LoRAAdapter


def test_peft_adapters():
    """Test DoRA and LoRA adapters."""
    torch.manual_seed(42)

    # Test DoRA
    dora = DoRAAdapter(768, 2, rank=16, alpha=32.0)
    weight = nn.Parameter(torch.randn(2, 768))
    dora.register_weight(weight)

    x = torch.randn(4, 768)
    out = dora(x)

    assert out.shape == (4, 2)
    print("✅ DoRAAdapter works")

    # Test LoRA
    lora = LoRAAdapter(768, 2, rank=16, alpha=32.0)
    lora.register_weight(weight)

    out2 = lora(x)
    assert out2.shape == (4, 2)
    print("✅ LoRAAdapter works")


def test_backbone_peft_hooks():
    """Test backbone PEFT registration."""
    from stage1_pro.model import DINOv3Backbone

    backbone = DINOv3Backbone(use_peft=True, peft_type="dora", peft_blocks=2)

    assert backbone.use_peft
    assert len(backbone.peft_adapters) > 0
    print("✅ Backbone PEFT hooks work")


def test_phase3_integration():
    """Test Phase 3 end-to-end."""
    from stage1_pro.model import Stage1Head, DINOv3Backbone
    from stage1_pro.training import Stage1Trainer
    from stage1_pro.config import Stage1ProConfig

    config = Stage1ProConfig(
        peft_type="dora",
        train_image_dir="/tmp",
        train_labels_file="/tmp.csv",
        val_image_dir="/tmp",
        val_labels_file="/tmp.csv",
    )

    backbone = DINOv3Backbone(use_peft=True, peft_type="dora")
    head = Stage1Head(use_gate=True)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            feats = self.backbone(x)
            return self.head(feats)

    model = DummyModel()

    trainer = Stage1Trainer(model, config)
    assert trainer.optimizer is not None
    print("✅ Phase 3 integration works")


def run_phase3_tests():
    print("\nPhase 3 Tests:")
    test_peft_adapters()
    test_backbone_peft_hooks()
    test_phase3_integration()
    print("\n✅ All Phase 3 Tests Passed")


if __name__ == "__main__":
    run_phase3_tests()
