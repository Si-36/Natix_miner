"""
Calibration module for Stage-1 Pro Modular Training System

Phase 3+: Gate calibration (Platt/Isotonic)
Phase 6+: Dirichlet calibration + SCRC
"""

from .gate_calib import PlattCalibrator, IsotonicCalibrator
from .dirichlet import DirichletCalibrator, ODIRRegularizer
from .scrc import SCRCCalibrator

__all__ = [
    "PlattCalibrator",
    "IsotonicCalibrator",
    "DirichletCalibrator",
    "ODIRRegularizer",
    "SCRCCalibrator",
]
