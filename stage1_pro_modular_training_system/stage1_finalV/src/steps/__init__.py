"""
🎯 **Steps Package** (Pipeline Steps - Domain-Stable Names)

All pipeline steps are registered here with domain-stable names (not "phase" numbers).

Steps:
- train_baseline_head: Phase 1 baseline training
- export_calib_logits: Export calibration artifacts
- sweep_thresholds: Phase 2 threshold sweep / calibration
- train_gate: Phase 3 gate training
- export_bundle: Phase 6 bundle export

2025/2026 Pro Standard Features:
- Leak-proof split contracts
- ArtifactStore integration
- Manifest lineage tracking
- Registry-driven execution
"""

from .sweep_thresholds import SweepThresholdsSpec

__all__ = [
    "SweepThresholdsSpec",
]

