"""Manages multiple vLLM servers for cascade routing"""
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

from src.infrastructure.vllm.mock_engine import MockAsyncLLMEngine

@dataclass
class ServerConfig:
    """Configuration for a single vLLM server"""
    model_name: str
    port: int
    tier: str  # "fast", "medium", "power", "precision"
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    tensor_parallel_size: int = 1
    max_seqs: int = 64
    gpu_util: float = 0.85

class VLLMServerManager:
    """
    Manages multiple vLLM servers for cascade inference

    Supports:
    - Confidence-based routing
    - Batch processing
    - Health monitoring
    - Load balancing
    """

    def __init__(self):
        self.servers: Dict[str, MockAsyncLLMEngine] = {}
        self.configs: Dict[str, ServerConfig] = {}
        logger.info("Initialized VLLMServerManager")

    def add_server(self, config: ServerConfig):
        """Add a vLLM server to the manager"""
        server = MockAsyncLLMEngine(
            model_name=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=4096,
            gpu_memory_utilization=config.gpu_util,
        )

        self.servers[config.tier] = server
        self.configs[config.tier] = config

        logger.info(f"✅ Added {config.tier} server: {config.model_name} on port {config.port}")

    async def route_request(
        self,
        prompt: str,
        confidence: float,
        sampling_params: Optional[Dict] = None
    ):
        """
        Route request to appropriate tier based on confidence

        Args:
            prompt: Input prompt
            confidence: Detection confidence (0.0-1.0)
            sampling_params: Sampling parameters

        Returns:
            Generated text from appropriate tier
        """
        # Determine tier based on confidence
        if confidence >= 0.95:
            # Skip VLM entirely (high confidence)
            logger.debug(f"High confidence ({confidence:.2f}) - skipping VLM")
            return None

        tier = self._select_tier(confidence)
        logger.debug(f"Routing to {tier} tier (confidence={confidence:.2f})")

        # Get server for this tier
        server = self.servers.get(tier)
        if not server:
            logger.error(f"No server found for tier: {tier}")
            return None

        # Generate response
        async for output in server.generate(prompt, sampling_params):
            return output.outputs[0].text

    def _select_tier(self, confidence: float) -> str:
        """Select appropriate tier based on confidence"""
        if confidence >= 0.85:
            return "fast"
        elif confidence >= 0.70:
            return "medium"
        elif confidence >= 0.55:
            return "power"
        else:
            return "precision"

    async def batch_generate(
        self,
        requests: List[Dict]
    ) -> List[str]:
        """
        Process multiple requests in batch

        Args:
            requests: List of {prompt, confidence, sampling_params}

        Returns:
            List of generated texts
        """
        tasks = [
            self.route_request(
                req["prompt"],
                req["confidence"],
                req.get("sampling_params")
            )
            for req in requests
        ]

        results = await asyncio.gather(*tasks)
        return results

    def get_stats(self) -> Dict:
        """Get statistics from all servers"""
        stats = {}
        for tier, server in self.servers.items():
            stats[tier] = server.get_stats()
        return stats

    def get_tier_distribution(self, requests: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of requests across tiers"""
        distribution = {"fast": 0, "medium": 0, "power": 0, "precision": 0, "skipped": 0}

        for req in requests:
            confidence = req["confidence"]
            if confidence >= 0.95:
                distribution["skipped"] += 1
            else:
                tier = self._select_tier(confidence)
                distribution[tier] += 1

        return distribution

# Example usage
async def test_server_manager():
    """Test server manager with cascade routing"""
    manager = VLLMServerManager()

    # Add servers for each tier
    manager.add_server(ServerConfig(
        model_name="Qwen/Qwen3-VL-4B",
        port=8000,
        tier="fast",
        min_confidence=0.85,
        max_confidence=0.95
    ))

    manager.add_server(ServerConfig(
        model_name="Qwen/Qwen3-VL-32B",
        port=8001,
        tier="medium",
        min_confidence=0.70,
        max_confidence=0.85
    ))

    manager.add_server(ServerConfig(
        model_name="Qwen/Qwen3-VL-72B",
        port=8002,
        tier="precision",
        min_confidence=0.0,
        max_confidence=0.70
    ))

    # Test routing
    requests = [
        {"prompt": "Is roadwork present?", "confidence": 0.92},  # → fast
        {"prompt": "Analyze this scene", "confidence": 0.78},     # → medium
        {"prompt": "Complex analysis", "confidence": 0.45},       # → precision
    ]

    results = await manager.batch_generate(requests)

    for req, result in zip(requests, results):
        print(f"Confidence {req['confidence']:.2f}: {result}")

    print(f"\n📊 Stats:")
    import json
    print(json.dumps(manager.get_stats(), indent=2))

    # Test tier distribution
    distribution = manager.get_tier_distribution(requests)
    print(f"\n📊 Tier Distribution:")
    print(json.dumps(distribution, indent=2))

if __name__ == "__main__":
    asyncio.run(test_server_manager())
