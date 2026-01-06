"""Mock vLLM engine for local CPU testing - validates logic without GPU"""
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from loguru import logger
import time

@dataclass
class MockModelConfig:
    """Mock model configuration"""
    model_name: str
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.95

@dataclass
class MockOutput:
    """Mock vLLM output"""
    text: str
    tokens: List[int]
    logprobs: Optional[List[float]] = None
    finish_reason: str = "stop"

@dataclass
class MockRequestOutput:
    """Mock request output matching vLLM API"""
    request_id: str
    outputs: List[MockOutput]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float

class MockAsyncLLMEngine:
    """
    Mock vLLM AsyncLLMEngine for local testing

    This validates:
    - Request routing logic
    - Confidence-based tier selection
    - Batch processing
    - Error handling
    - Memory tracking

    WITHOUT requiring GPU or actual model weights.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.95,
        **kwargs
    ):
        self.config = MockModelConfig(
            model_name=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self.request_count = 0
        self.total_tokens_generated = 0
        self.total_latency_ms = 0.0

        logger.info(f"✅ [MOCK] Initialized {model_name}")
        logger.debug(f"   Config: {self.config}")

    async def generate(
        self,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[MockRequestOutput]:
        """
        Mock generation with realistic latency simulation

        Args:
            prompt: Input prompt text
            sampling_params: Sampling parameters (max_tokens, temperature, etc.)
            request_id: Unique request identifier

        Yields:
            MockRequestOutput objects simulating streaming output
        """
        request_id = request_id or f"request_{self.request_count}"
        self.request_count += 1

        # Extract sampling params
        sampling_params = sampling_params or {}
        max_tokens = sampling_params.get("max_tokens", 256)
        temperature = sampling_params.get("temperature", 0.7)

        # Simulate processing latency (scaled by model size)
        model_size_gb = self._estimate_model_size(self.config.model_name)
        base_latency_ms = 5 + (model_size_gb * 2)  # Larger models = slower

        # Simulate time-to-first-token (TTFT)
        await asyncio.sleep(base_latency_ms / 1000)

        # Generate mock response
        mock_text = self._generate_mock_response(prompt, max_tokens)
        tokens = list(range(len(mock_text.split())))  # Mock token IDs

        # Calculate mock metrics
        prompt_tokens = len(prompt.split())
        completion_tokens = len(tokens)
        total_tokens = prompt_tokens + completion_tokens
        latency_ms = base_latency_ms + (completion_tokens * 0.5)

        # Update stats
        self.total_tokens_generated += total_tokens
        self.total_latency_ms += latency_ms

        # Yield final output (simulating streaming end)
        output = MockOutput(
            text=mock_text,
            tokens=tokens,
            finish_reason="stop"
        )

        result = MockRequestOutput(
            request_id=request_id,
            outputs=[output],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

        logger.debug(f"[MOCK] Generated {completion_tokens} tokens in {latency_ms:.1f}ms")

        yield result

    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in GB based on name"""
        size_map = {
            "4b": 4.5,
            "7b": 7.5,
            "8b": 8.5,
            "13b": 13.5,
            "30b": 30.0,
            "32b": 32.0,
            "70b": 70.0,
            "72b": 72.0,
            "78b": 78.0,
            "235b": 235.0,
        }

        model_lower = model_name.lower()
        for key, size in size_map.items():
            if key in model_lower:
                return size

        return 10.0  # Default

    def _generate_mock_response(self, prompt: str, max_tokens: int) -> str:
        """Generate realistic mock response based on prompt"""
        if "roadwork" in prompt.lower():
            return "[MOCK RESPONSE] Roadwork detected with high confidence. Analysis complete."
        elif "cascade" in prompt.lower():
            return "[MOCK RESPONSE] Cascade routing: confidence=0.85, tier=fast"
        else:
            return f"[MOCK RESPONSE] Processed prompt with {max_tokens} max tokens."

    def get_memory_usage(self) -> float:
        """Return mock memory usage in GB"""
        return self._estimate_model_size(self.config.model_name)

    def get_stats(self) -> Dict[str, Any]:
        """Return mock statistics"""
        avg_latency = (
            self.total_latency_ms / self.request_count
            if self.request_count > 0
            else 0.0
        )

        return {
            "model_name": self.config.model_name,
            "request_count": self.request_count,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_latency_ms": avg_latency,
            "memory_usage_gb": self.get_memory_usage(),
        }

# Example usage
async def test_mock_engine():
    """Test mock engine"""
    engine = MockAsyncLLMEngine("Qwen/Qwen3-VL-4B-Instruct")

    async for output in engine.generate(
        "Is there roadwork in this image?",
        sampling_params={"max_tokens": 50}
    ):
        print(f"✅ Generated: {output.outputs[0].text}")
        print(f"   Latency: {output.latency_ms:.1f}ms")

    stats = engine.get_stats()
    print(f"\n📊 Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_mock_engine())
