"""Unit tests for production compression stack"""
import pytest
from src.compression_2026.production_stack import (
    ProductionCompressionStack,
    CompressionTechnique,
)

def test_compression_stack_init():
    """Test stack initialization"""
    stack = ProductionCompressionStack("Qwen/Qwen3-VL-72B")
    assert stack.model_name == "Qwen/Qwen3-VL-72B"
    assert len(stack.techniques) == 0

def test_add_nvidia_kvpress():
    """Test NVIDIA KVPress addition"""
    stack = ProductionCompressionStack("test-model")
    stack.add_nvidia_kvpress("expected_attention", 0.5)

    assert len(stack.techniques) == 1
    assert stack.techniques[0].name == "NVIDIA KVPress"
    assert stack.techniques[0].config["method"] == "expected_attention"
    assert stack.techniques[0].memory_reduction == 0.60

def test_add_all_techniques():
    """Test adding all 7 compression techniques"""
    stack = ProductionCompressionStack("Qwen/Qwen3-VL-72B")

    stack.add_nvidia_kvpress()
    stack.add_lmcache()
    stack.add_awq_quantization()
    stack.add_kvcache_factory()
    stack.add_gear_compression()
    stack.add_spark_compression()
    stack.add_evicpress()

    assert len(stack.techniques) == 7

def test_memory_reduction_calculation():
    """Test cumulative memory reduction"""
    stack = ProductionCompressionStack("test-model")

    stack.add_awq_quantization()  # 75% reduction
    stack.add_nvidia_kvpress()     # 60% reduction

    total = stack.get_total_memory_reduction()
    # 1 - (1-0.75) * (1-0.60) = 1 - 0.25 * 0.40 = 0.90 (90%)
    assert 0.85 <= total <= 0.95

def test_summary_generation():
    """Test summary generation"""
    stack = ProductionCompressionStack("Qwen/Qwen3-VL-72B")
    stack.add_nvidia_kvpress()
    stack.add_awq_quantization()

    summary = stack.summary()

    assert "model_name" in summary
    assert "num_techniques" in summary
    assert summary["num_techniques"] == 2
    assert "total_memory_reduction" in summary

@pytest.mark.parametrize("model,expected_size", [
    ("Qwen/Qwen3-VL-4B", 9.0),
    ("Qwen/Qwen3-VL-72B", 144.0),
    ("InternVL3.5-78B", 156.0),
])
def test_model_size_estimation(model, expected_size):
    """Test model size estimation"""
    stack = ProductionCompressionStack(model)
    estimated = stack._estimate_model_size()
    assert estimated == expected_size

def test_accuracy_loss_calculation():
    """Test cumulative accuracy loss"""
    stack = ProductionCompressionStack("test-model")

    stack.add_nvidia_kvpress()  # 0% loss
    stack.add_awq_quantization()  # 1% loss
    stack.add_kvcache_factory()  # 2% loss

    total_loss = stack.get_total_accuracy_loss()
    assert 0.025 <= total_loss <= 0.035  # ~3% total

def test_gear_compression():
    """Test GEAR compression technique"""
    stack = ProductionCompressionStack("test-model")
    stack.add_gear_compression()

    assert len(stack.techniques) == 1
    assert stack.techniques[0].name == "GEAR 4-bit KV"
    assert stack.techniques[0].memory_reduction == 0.75
    assert stack.techniques[0].accuracy_loss == 0.001  # <0.1%

def test_spark_compression():
    """Test SparK compression technique"""
    stack = ProductionCompressionStack("test-model")
    stack.add_spark_compression(sparsity_ratio=0.85)

    assert len(stack.techniques) == 1
    assert stack.techniques[0].name == "SparK"
    assert stack.techniques[0].memory_reduction == 0.85
    assert stack.techniques[0].accuracy_loss == 0.0  # Training-free

def test_evicpress():
    """Test EVICPRESS technique"""
    stack = ProductionCompressionStack("test-model")
    stack.add_evicpress()

    assert len(stack.techniques) == 1
    assert stack.techniques[0].name == "EVICPRESS"
    assert stack.techniques[0].memory_reduction == 0.0  # Manages, not compresses
