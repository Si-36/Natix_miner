"""
Utility modules (Day 7)

Features:
- Profiling (cProfile + PyTorch profiler)
- Model optimization (torch.compile)
- Performance benchmarking
"""

from .profiling import (
    profile_context,
    profile_function,
    torch_profile_context,
    create_profile_summary,
    analyze_bottlenecks,
)
from .optimization import (
    compile_model,
    compile_if_available,
    benchmark_model,
    compare_compiled_vs_uncompiled,
    HAS_COMPILE,
    TORCH_VERSION,
)

__all__ = [
    # Profiling
    "profile_context",
    "profile_function",
    "torch_profile_context",
    "create_profile_summary",
    "analyze_bottlenecks",
    # Optimization
    "compile_model",
    "compile_if_available",
    "benchmark_model",
    "compare_compiled_vs_uncompiled",
    "HAS_COMPILE",
    "TORCH_VERSION",
]
