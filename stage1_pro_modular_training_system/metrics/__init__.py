"""
Metrics module for Stage-1 Pro Modular Training System

Provides selective metrics, calibration metrics, exit metrics, and bootstrap utilities.
"""

from .selective import compute_risk_coverage, compute_augrc, compute_selective_metrics
from .calibration import compute_calibration_metrics, compute_nll, compute_brier_score
from .exit import compute_exit_metrics
from .bootstrap import bootstrap_resample, compute_confidence_intervals

__all__ = [
    "compute_risk_coverage",
    "compute_augrc",
    "compute_selective_metrics",
    "compute_calibration_metrics",
    "compute_nll",
    "compute_brier_score",
    "compute_exit_metrics",
    "bootstrap_resample",
    "compute_confidence_intervals",
]
