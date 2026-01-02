"""
Pipeline step implementations (2025 production-grade)

Each step:
- Uses atomic writes (crash-safe)
- Generates manifest-last (lineage tracking)
- Uses centralized metrics (no MCC drift)
- Fully Hydra-driven (zero hardcoding)

Steps:
- train_baseline: Phase-1 baseline training
- sweep_thresholds: Phase-2 threshold selection
- train_explora: Phase-4 ExPLoRA domain adaptation
- calibrate_scrc: Phase-5 SCRC calibration (NEW)
- export_bundle: Phase-6 deployment bundle
"""

from .train_baseline import run_phase1_baseline
from .sweep_thresholds import run_phase2_threshold_sweep
from .train_explora import run_phase4_explora, validate_peft_merge
from .calibrate_scrc import run_phase5_scrc_calibration
from .export_bundle import run_phase6_bundle_export, load_bundle

__all__ = [
    "run_phase1_baseline",
    "run_phase2_threshold_sweep",
    "run_phase4_explora",
    "validate_peft_merge",
    "run_phase5_scrc_calibration",
    "run_phase6_bundle_export",
    "load_bundle",
]
