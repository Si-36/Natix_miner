"""
🎯 **Steps Package** (Pipeline Steps - Domain-Stable Names)

All pipeline steps are registered in src/pipeline/registry.py.

2025/2026 Pro Standard Features:
- Leak-proof split contracts
- ArtifactStore integration
- Manifest lineage tracking
- Registry-driven execution

Steps:
- export_calib_logits: Export calibration artifacts
- sweep_thresholds: Phase 2 threshold sweep / calibration
- train_gate: Phase 3 gate training
- export_bundle: Phase 6 bundle export
"""

# Steps are lazy-loaded by registry, imported on demand
# This prevents circular dependencies

__all__ = []
