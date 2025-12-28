"""
Unit Tests for Multi-View Inference

Tests all multi-view components:
- MultiViewGenerator: Crop generation with overlap
- TopKMeanAggregator: Top-K mean aggregation
- AttentionAggregator: Learnable attention aggregation
- MultiViewDINOv3: End-to-end multi-view pipeline

Latest 2025-2026 practices:
- Python 3.14+ with pytest
- Clear test names
- Comprehensive edge case coverage
"""

import pytest
import torch
import torch.nn as nn

from src.models.multi_view import (
    MultiViewGenerator,
    TopKMeanAggregator,
    AttentionAggregator,
    MultiViewDINOv3,
    create_multiview_model,
)


class TestMultiViewGenerator:
    """Test multi-view crop generation"""

    def test_basic_generation(self):
        """Test basic crop generation with default settings"""
        generator = MultiViewGenerator(crop_size=224, grid_size=(3, 3), overlap=0.15)

        # Single image
        image = torch.randn(3, 518, 518)
        crops = generator(image)

        # Check shape
        assert crops.shape == (10, 3, 224, 224), f"Expected [10, 3, 224, 224], got {crops.shape}"

        # Check all crops are valid (no NaN/Inf)
        assert torch.isfinite(crops).all(), "Crops contain NaN or Inf"

    def test_different_image_sizes(self):
        """Test crop generation with different input sizes"""
        generator = MultiViewGenerator(crop_size=224, grid_size=(3, 3), overlap=0.15)

        sizes = [(518, 518), (640, 480), (1024, 768), (800, 600)]

        for H, W in sizes:
            image = torch.randn(3, H, W)
            crops = generator(image)

            assert crops.shape == (10, 3, 224, 224), \
                f"Failed for size ({H}, {W}): expected [10, 3, 224, 224], got {crops.shape}"

    def test_different_grid_sizes(self):
        """Test different grid configurations"""
        test_cases = [
            ((2, 2), 5),   # 2×2 grid = 4 tiles + 1 global = 5 crops
            ((3, 3), 10),  # 3×3 grid = 9 tiles + 1 global = 10 crops
            ((4, 4), 17),  # 4×4 grid = 16 tiles + 1 global = 17 crops
        ]

        for grid_size, expected_crops in test_cases:
            generator = MultiViewGenerator(crop_size=224, grid_size=grid_size, overlap=0.15)
            image = torch.randn(3, 518, 518)
            crops = generator(image)

            assert crops.shape[0] == expected_crops, \
                f"Grid {grid_size}: expected {expected_crops} crops, got {crops.shape[0]}"

    def test_num_crops_property(self):
        """Test num_crops property"""
        generator = MultiViewGenerator(crop_size=224, grid_size=(3, 3), overlap=0.15)
        assert generator.num_crops == 10, f"Expected 10 crops, got {generator.num_crops}"

        generator = MultiViewGenerator(crop_size=224, grid_size=(2, 2), overlap=0.15)
        assert generator.num_crops == 5, f"Expected 5 crops, got {generator.num_crops}"

    def test_overlap_validation(self):
        """Test overlap parameter validation"""
        # Valid overlaps
        valid_overlaps = [0.0, 0.1, 0.15, 0.2, 0.3, 0.49]
        for overlap in valid_overlaps:
            generator = MultiViewGenerator(crop_size=224, grid_size=(3, 3), overlap=overlap)
            assert generator.overlap == overlap

        # Invalid overlaps
        invalid_overlaps = [-0.1, 0.5, 1.0, 2.0]
        for overlap in invalid_overlaps:
            with pytest.raises(ValueError):
                MultiViewGenerator(crop_size=224, grid_size=(3, 3), overlap=overlap)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        generator = MultiViewGenerator(crop_size=224, grid_size=(3, 3), overlap=0.15)

        # Wrong number of dimensions
        with pytest.raises(ValueError):
            generator(torch.randn(518, 518))  # Missing channel dimension

        # 4D input (batch)
        with pytest.raises(ValueError):
            generator(torch.randn(2, 3, 518, 518))


class TestTopKMeanAggregator:
    """Test Top-K mean aggregation"""

    def test_basic_aggregation(self):
        """Test basic Top-K aggregation"""
        aggregator = TopKMeanAggregator(topk=2)

        # [B=2, num_crops=10, num_classes=13]
        predictions = torch.randn(2, 10, 13)
        aggregated = aggregator(predictions)

        # Check shape
        assert aggregated.shape == (2, 13), f"Expected [2, 13], got {aggregated.shape}"

        # Check output is valid probabilities (after softmax)
        assert (aggregated >= 0).all(), "Negative probabilities"
        assert torch.allclose(aggregated.sum(dim=-1), torch.ones(2), atol=1e-5), \
            "Probabilities don't sum to 1"

    def test_different_topk_values(self):
        """Test different K values"""
        predictions = torch.randn(4, 10, 13)

        for k in [1, 2, 3, 5, 10]:
            aggregator = TopKMeanAggregator(topk=k)
            aggregated = aggregator(predictions)

            assert aggregated.shape == (4, 13), \
                f"K={k}: expected [4, 13], got {aggregated.shape}"

    def test_topk_exceeds_num_crops(self):
        """Test when K > num_crops (should use all crops)"""
        aggregator = TopKMeanAggregator(topk=20)  # More than 10 crops
        predictions = torch.randn(2, 10, 13)

        # Should not crash, uses all 10 crops
        aggregated = aggregator(predictions)
        assert aggregated.shape == (2, 13)

    def test_deterministic_output(self):
        """Test that same input produces same output"""
        aggregator = TopKMeanAggregator(topk=2)
        predictions = torch.randn(2, 10, 13)

        output1 = aggregator(predictions)
        output2 = aggregator(predictions)

        assert torch.allclose(output1, output2), "Non-deterministic output"

    def test_invalid_inputs(self):
        """Test error handling"""
        aggregator = TopKMeanAggregator(topk=2)

        # Wrong shape
        with pytest.raises(ValueError):
            aggregator(torch.randn(2, 13))  # Missing num_crops dimension


class TestAttentionAggregator:
    """Test attention-based aggregation"""

    def test_basic_aggregation(self):
        """Test basic attention aggregation"""
        aggregator = AttentionAggregator(num_classes=13, hidden_dim=64)

        predictions = torch.randn(2, 10, 13)
        aggregated = aggregator(predictions)

        # Check shape
        assert aggregated.shape == (2, 13), f"Expected [2, 13], got {aggregated.shape}"

        # Check output is valid probabilities
        assert (aggregated >= 0).all(), "Negative probabilities"
        assert torch.allclose(aggregated.sum(dim=-1), torch.ones(2), atol=1e-5), \
            "Probabilities don't sum to 1"

    def test_learnable_parameters(self):
        """Test that attention has learnable parameters"""
        aggregator = AttentionAggregator(num_classes=13, hidden_dim=64)

        # Count parameters
        num_params = sum(p.numel() for p in aggregator.parameters())
        assert num_params > 0, "Attention aggregator has no learnable parameters"

        # Check all parameters require gradients
        for param in aggregator.parameters():
            assert param.requires_grad, "Some parameters don't require grad"

    def test_different_hidden_dims(self):
        """Test different hidden dimensions"""
        predictions = torch.randn(2, 10, 13)

        for hidden_dim in [32, 64, 128, 256]:
            aggregator = AttentionAggregator(num_classes=13, hidden_dim=hidden_dim)
            aggregated = aggregator(predictions)

            assert aggregated.shape == (2, 13), \
                f"hidden_dim={hidden_dim}: expected [2, 13], got {aggregated.shape}"

    def test_gradient_flow(self):
        """Test that gradients flow through attention"""
        aggregator = AttentionAggregator(num_classes=13, hidden_dim=64)
        predictions = torch.randn(2, 10, 13, requires_grad=True)

        aggregated = aggregator(predictions)
        loss = aggregated.sum()
        loss.backward()

        # Check gradients exist
        assert predictions.grad is not None, "No gradient for predictions"
        for param in aggregator.parameters():
            assert param.grad is not None, f"No gradient for parameter {param.shape}"


class TestMultiViewDINOv3:
    """Test end-to-end multi-view pipeline"""

    @pytest.fixture
    def dummy_backbone(self):
        """Create dummy backbone for testing"""
        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1280, kernel_size=1)
                self.pool = nn.AdaptiveAvgPool2d(1)

            def forward(self, x):
                # [B, 3, 224, 224] → [B, 1280]
                x = self.conv(x)
                x = self.pool(x)
                return x.view(x.size(0), -1)

        return DummyBackbone()

    @pytest.fixture
    def dummy_head(self):
        """Create dummy classification head"""
        class DummyHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1280, 13)

            def forward(self, x):
                # [B, 1280] → [B, 13]
                return self.fc(x)

        return DummyHead()

    def test_basic_forward(self, dummy_backbone, dummy_head):
        """Test basic multi-view forward pass"""
        aggregator = TopKMeanAggregator(topk=2)
        multiview = MultiViewDINOv3(
            backbone=dummy_backbone,
            head=dummy_head,
            aggregator=aggregator,
        )

        # [B=2, 3, 518, 518]
        images = torch.randn(2, 3, 518, 518)
        output = multiview(images)

        # Check shape
        assert output.shape == (2, 13), f"Expected [2, 13], got {output.shape}"

    def test_batched_processing(self, dummy_backbone, dummy_head):
        """Test that batched processing works correctly"""
        aggregator = TopKMeanAggregator(topk=2)
        multiview = MultiViewDINOv3(
            backbone=dummy_backbone,
            head=dummy_head,
            aggregator=aggregator,
        )

        batch_sizes = [1, 2, 4, 8]
        for B in batch_sizes:
            images = torch.randn(B, 3, 518, 518)
            output = multiview(images)

            assert output.shape == (B, 13), \
                f"Batch size {B}: expected [{B}, 13], got {output.shape}"

    def test_gradient_flow(self, dummy_backbone, dummy_head):
        """Test gradient flow through entire pipeline"""
        aggregator = TopKMeanAggregator(topk=2)
        multiview = MultiViewDINOv3(
            backbone=dummy_backbone,
            head=dummy_head,
            aggregator=aggregator,
        )

        images = torch.randn(2, 3, 518, 518, requires_grad=True)
        output = multiview(images)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert images.grad is not None, "No gradient for input images"

        # Check model parameters have gradients
        for name, param in multiview.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_invalid_inputs(self, dummy_backbone, dummy_head):
        """Test error handling"""
        aggregator = TopKMeanAggregator(topk=2)
        multiview = MultiViewDINOv3(
            backbone=dummy_backbone,
            head=dummy_head,
            aggregator=aggregator,
        )

        # Wrong shape (3D instead of 4D)
        with pytest.raises(ValueError):
            multiview(torch.randn(3, 518, 518))


class TestCreateMultiViewModel:
    """Test factory function"""

    @pytest.fixture
    def dummy_backbone(self):
        """Create dummy backbone"""
        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1280, kernel_size=1)
                self.pool = nn.AdaptiveAvgPool2d(1)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                return x.view(x.size(0), -1)

        return DummyBackbone()

    @pytest.fixture
    def dummy_head(self):
        """Create dummy head"""
        class DummyHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1280, 13)

            def forward(self, x):
                return self.fc(x)

        return DummyHead()

    def test_topk_mean_creation(self, dummy_backbone, dummy_head):
        """Test creating model with Top-K mean aggregation"""
        model = create_multiview_model(
            backbone=dummy_backbone,
            head=dummy_head,
            aggregation="topk_mean",
            topk=2,
        )

        assert isinstance(model, MultiViewDINOv3)
        assert isinstance(model.aggregator, TopKMeanAggregator)

    def test_attention_creation(self, dummy_backbone, dummy_head):
        """Test creating model with attention aggregation"""
        model = create_multiview_model(
            backbone=dummy_backbone,
            head=dummy_head,
            aggregation="attention",
            num_classes=13,
        )

        assert isinstance(model, MultiViewDINOv3)
        assert isinstance(model.aggregator, AttentionAggregator)

    def test_invalid_aggregation(self, dummy_backbone, dummy_head):
        """Test error handling for invalid aggregation"""
        with pytest.raises(ValueError):
            create_multiview_model(
                backbone=dummy_backbone,
                head=dummy_head,
                aggregation="invalid_method",
            )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
