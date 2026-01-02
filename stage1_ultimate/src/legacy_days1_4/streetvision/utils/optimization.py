"""
Model optimization utilities (Day 7)

Features:
- torch.compile support (PyTorch 2.0+)
- Automatic optimization detection
- Performance benchmarking

2025-12-30 best practices:
- Automatic PyTorch 2.0+ detection
- Fallback to non-compiled for older versions
- Clear logging of optimization status
"""

import logging
from typing import Optional, Any
import warnings

logger = logging.getLogger(__name__)

# Check PyTorch availability and version
try:
    import torch
    HAS_TORCH = True
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
    HAS_COMPILE = TORCH_VERSION >= (2, 0) and hasattr(torch, "compile")
except ImportError:
    HAS_TORCH = False
    HAS_COMPILE = False
    TORCH_VERSION = (0, 0)


def compile_model(
    model: Any,
    mode: str = "default",
    dynamic: bool = False,
    fullgraph: bool = False,
    backend: str = "inductor",
    disable: bool = False,
) -> Any:
    """
    Compile model using torch.compile (PyTorch 2.0+)

    Args:
        model: PyTorch model to compile
        mode: Optimization mode
            - "default": Balanced speed/compilation time
            - "reduce-overhead": Reduce Python overhead (best for small models)
            - "max-autotune": Maximum performance (slow compile)
        dynamic: Enable dynamic shapes (slower but more flexible)
        fullgraph: Require full graph capture (may fail on complex models)
        backend: Compiler backend ("inductor", "aot_eager", etc.)
        disable: Disable compilation (useful for debugging)

    Returns:
        Compiled model (or original if compilation not available/disabled)

    Benefits:
        - 1.5-2× speedup for inference
        - 1.2-1.5× speedup for training
        - Zero code changes required
        - Automatic kernel fusion
    """
    if disable:
        logger.info("⚠️  torch.compile DISABLED (disable=True)")
        return model

    if not HAS_TORCH:
        logger.warning("⚠️  PyTorch not available - returning original model")
        return model

    if not HAS_COMPILE:
        logger.warning(
            f"⚠️  torch.compile not available (PyTorch {torch.__version__})\n"
            f"   Requires PyTorch 2.0+\n"
            f"   Returning original model"
        )
        return model

    # Compile model
    try:
        logger.info(f"🔥 Compiling model with torch.compile (mode={mode}, backend={backend})...")

        compiled_model = torch.compile(
            model,
            mode=mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
            backend=backend,
        )

        logger.info("✅ Model compiled successfully")
        logger.info(f"   Expected speedup: 1.2-2× (depends on model/hardware)")

        return compiled_model

    except Exception as e:
        logger.warning(
            f"⚠️  torch.compile failed: {e}\n"
            f"   Returning original model\n"
            f"   This is expected for some model architectures"
        )
        return model


def compile_if_available(
    model: Any,
    compile_config: Optional[dict] = None,
) -> Any:
    """
    Compile model if torch.compile is available (safe wrapper)

    Args:
        model: PyTorch model
        compile_config: Optional compile configuration (passed to compile_model)

    Returns:
        Compiled model (or original if compilation not available)

    Usage:
        # Safely compile model (no error if compile not available)
        model = compile_if_available(model, {"mode": "max-autotune"})
    """
    if compile_config is None:
        compile_config = {"mode": "default"}

    return compile_model(model, **compile_config)


def benchmark_model(
    model: Any,
    input_tensor: Any,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> dict:
    """
    Benchmark model inference speed

    Args:
        model: PyTorch model
        input_tensor: Example input tensor
        num_iterations: Number of iterations to benchmark
        warmup_iterations: Number of warmup iterations

    Returns:
        Benchmark results:
        {
            "mean_time_ms": ...,
            "std_time_ms": ...,
            "throughput_samples_per_sec": ...,
        }
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available")

    import time
    import numpy as np

    model.eval()

    # Warmup
    logger.info(f"🔥 Warming up ({warmup_iterations} iterations)...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_tensor)

    # Benchmark
    logger.info(f"📊 Benchmarking ({num_iterations} iterations)...")
    times = []

    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(input_tensor)

            # Synchronize if CUDA
            if torch.cuda.is_available() and input_tensor.is_cuda:
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)
    mean_time_ms = float(np.mean(times))
    std_time_ms = float(np.std(times))

    # Compute throughput (samples/sec)
    batch_size = input_tensor.shape[0] if hasattr(input_tensor, "shape") else 1
    throughput = (batch_size / mean_time_ms) * 1000

    logger.info(f"✅ Benchmark complete:")
    logger.info(f"   Mean: {mean_time_ms:.2f} ms")
    logger.info(f"   Std:  {std_time_ms:.2f} ms")
    logger.info(f"   Throughput: {throughput:.1f} samples/sec")

    return {
        "mean_time_ms": mean_time_ms,
        "std_time_ms": std_time_ms,
        "throughput_samples_per_sec": float(throughput),
        "num_iterations": num_iterations,
    }


def compare_compiled_vs_uncompiled(
    model: Any,
    input_tensor: Any,
    compile_config: Optional[dict] = None,
    num_iterations: int = 100,
) -> dict:
    """
    Compare compiled vs uncompiled model performance

    Args:
        model: PyTorch model
        input_tensor: Example input
        compile_config: Compile configuration
        num_iterations: Number of benchmark iterations

    Returns:
        Comparison results:
        {
            "uncompiled": {...},
            "compiled": {...},
            "speedup": 1.5,
        }
    """
    if not HAS_COMPILE:
        logger.warning("torch.compile not available - skipping comparison")
        return {}

    logger.info("📊 Benchmarking UNCOMPILED model...")
    uncompiled_results = benchmark_model(model, input_tensor, num_iterations)

    logger.info("📊 Compiling model...")
    compiled_model = compile_model(model, **(compile_config or {}))

    logger.info("📊 Benchmarking COMPILED model...")
    compiled_results = benchmark_model(compiled_model, input_tensor, num_iterations)

    # Compute speedup
    speedup = uncompiled_results["mean_time_ms"] / compiled_results["mean_time_ms"]

    logger.info(f"\n{'='*80}")
    logger.info("COMPILATION SPEEDUP RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Uncompiled: {uncompiled_results['mean_time_ms']:.2f} ms")
    logger.info(f"Compiled:   {compiled_results['mean_time_ms']:.2f} ms")
    logger.info(f"Speedup:    {speedup:.2f}×")
    logger.info(f"{'='*80}\n")

    return {
        "uncompiled": uncompiled_results,
        "compiled": compiled_results,
        "speedup": float(speedup),
    }


__all__ = [
    "compile_model",
    "compile_if_available",
    "benchmark_model",
    "compare_compiled_vs_uncompiled",
    "HAS_COMPILE",
    "TORCH_VERSION",
]
