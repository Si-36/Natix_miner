"""
Profiling utilities (Day 7)

Features:
- cProfile integration
- Torch profiler support
- Performance bottleneck identification
- Profile report generation

2025-12-30 best practices:
- Optional profiling (no overhead when disabled)
- Atomic report writes
- Clear profiling output
"""

import cProfile
import pstats
import io
import time
from pathlib import Path
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@contextmanager
def profile_context(
    enabled: bool = True,
    output_path: Optional[Path] = None,
    sort_by: str = "cumulative",
    top_n: int = 50,
):
    """
    Context manager for cProfile profiling

    Usage:
        with profile_context(enabled=True, output_path="profile.txt"):
            # Your code here
            train_model()

    Args:
        enabled: Enable profiling (set False for no overhead)
        output_path: Optional path to save profile report
        sort_by: Sort criterion ("cumulative", "time", "calls")
        top_n: Number of top functions to show

    Yields:
        cProfile.Profile instance (or None if disabled)
    """
    if not enabled:
        yield None
        return

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    logger.info("🔍 Profiling started...")
    start_time = time.time()

    try:
        yield profiler
    finally:
        # Stop profiling
        profiler.disable()
        duration = time.time() - start_time

        logger.info(f"✅ Profiling completed ({duration:.1f}s)")

        # Generate report
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(top_n)

        profile_output = s.getvalue()

        # Print to console
        logger.info("\n" + "="*80)
        logger.info("PROFILE REPORT (Top Functions)")
        logger.info("="*80)
        logger.info(profile_output)
        logger.info("="*80 + "\n")

        # Save to file if requested
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write stats in text format
            with open(output_path, "w") as f:
                f.write(f"Profile Report (sorted by {sort_by})\n")
                f.write(f"Total runtime: {duration:.2f}s\n")
                f.write("="*80 + "\n")
                f.write(profile_output)

            # Also save binary stats for later analysis
            binary_path = output_path.with_suffix(".pstats")
            ps.dump_stats(str(binary_path))

            logger.info(f"📊 Profile saved to: {output_path}")
            logger.info(f"📊 Binary stats saved to: {binary_path}")


def profile_function(
    func: Callable,
    *args,
    output_path: Optional[Path] = None,
    sort_by: str = "cumulative",
    **kwargs,
) -> Any:
    """
    Profile a single function call

    Usage:
        result = profile_function(
            train_model,
            config=cfg,
            output_path="profile.txt"
        )

    Args:
        func: Function to profile
        *args: Positional arguments to func
        output_path: Optional path to save profile
        sort_by: Sort criterion
        **kwargs: Keyword arguments to func

    Returns:
        Function return value
    """
    with profile_context(enabled=True, output_path=output_path, sort_by=sort_by):
        return func(*args, **kwargs)


@contextmanager
def torch_profile_context(
    enabled: bool = True,
    output_path: Optional[Path] = None,
    with_stack: bool = True,
):
    """
    Context manager for PyTorch profiler

    Usage:
        with torch_profile_context(enabled=True, output_path="torch_profile"):
            for batch in dataloader:
                model(batch)

    Args:
        enabled: Enable profiling
        output_path: Optional path to save trace (Chrome trace format)
        with_stack: Include Python stack trace

    Yields:
        torch.profiler.profile instance (or None if disabled)
    """
    if not enabled:
        yield None
        return

    try:
        import torch
        from torch.profiler import profile, ProfilerActivity
    except ImportError:
        logger.warning("PyTorch profiler not available (torch not installed)")
        yield None
        return

    # Create profiler
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    profiler = profile(
        activities=activities,
        with_stack=with_stack,
        record_shapes=True,
        profile_memory=True,
    )

    profiler.start()
    logger.info("🔍 PyTorch profiling started...")

    try:
        yield profiler
    finally:
        profiler.stop()
        logger.info("✅ PyTorch profiling completed")

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("PYTORCH PROFILE REPORT")
        logger.info("="*80)
        logger.info(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        logger.info("="*80 + "\n")

        # Save trace if requested
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export Chrome trace (view in chrome://tracing)
            trace_path = output_path.with_suffix(".json")
            profiler.export_chrome_trace(str(trace_path))
            logger.info(f"📊 PyTorch trace saved to: {trace_path}")
            logger.info(f"   View in Chrome: chrome://tracing")


def create_profile_summary(
    profile_path: Path,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Create JSON summary from profile stats

    Args:
        profile_path: Path to .pstats file
        output_json: Optional path to save JSON summary

    Returns:
        Profile summary dictionary
    """
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    # Load stats
    ps = pstats.Stats(str(profile_path))

    # Get top functions
    top_functions = []
    for func, stats in sorted(
        ps.stats.items(),
        key=lambda x: x[1][3],  # Sort by cumulative time
        reverse=True,
    )[:50]:
        filename, line, func_name = func
        cc, nc, tt, ct, callers = stats

        top_functions.append({
            "function": func_name,
            "file": filename,
            "line": line,
            "calls": nc,
            "total_time": float(tt),
            "cumulative_time": float(ct),
            "time_per_call": float(tt / nc) if nc > 0 else 0.0,
        })

    # Build summary
    summary = {
        "total_functions": len(ps.stats),
        "total_calls": sum(stats[0] for stats in ps.stats.values()),
        "top_functions": top_functions,
    }

    # Save JSON if requested
    if output_json:
        import json
        from streetvision.io.atomic import write_json_atomic
        write_json_atomic(output_json, summary)
        logger.info(f"📊 Profile summary saved to: {output_json}")

    return summary


def analyze_bottlenecks(
    profile_path: Path,
    threshold_percent: float = 5.0,
) -> Dict[str, Any]:
    """
    Analyze performance bottlenecks from profile

    Args:
        profile_path: Path to .pstats file
        threshold_percent: Report functions taking > this % of total time

    Returns:
        Bottleneck analysis dictionary
    """
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    # Load stats
    ps = pstats.Stats(str(profile_path))

    # Get total time
    total_time = sum(stats[3] for stats in ps.stats.values())

    # Find bottlenecks
    bottlenecks = []
    for func, stats in ps.stats.items():
        filename, line, func_name = func
        cc, nc, tt, ct, callers = stats

        # Calculate percentage
        percent = (ct / total_time * 100) if total_time > 0 else 0.0

        if percent >= threshold_percent:
            bottlenecks.append({
                "function": func_name,
                "file": filename,
                "line": line,
                "cumulative_time": float(ct),
                "percent_total": float(percent),
                "calls": nc,
            })

    # Sort by percentage
    bottlenecks.sort(key=lambda x: x["percent_total"], reverse=True)

    return {
        "total_time": float(total_time),
        "threshold_percent": threshold_percent,
        "bottlenecks": bottlenecks,
        "num_bottlenecks": len(bottlenecks),
    }


__all__ = [
    "profile_context",
    "profile_function",
    "torch_profile_context",
    "create_profile_summary",
    "analyze_bottlenecks",
]
