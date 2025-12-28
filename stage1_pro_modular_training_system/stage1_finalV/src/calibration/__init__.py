"""
Calibration module for Stage-1 Pro System (2026 Best Practices)
Library-backed conformal prediction + ensemble calibration
"""

from .temperature import TemperatureScaling
from .beta import BetaCalibration
from .isotonic import IsotonicRegression
from .ensemble import EnsembleCalibration, EnsembleTemperatureCalibrator, MCDropoutCalibration
from .conformal import (
    ConformalPolicy,
    TemperatureConformalWrapper,
    TORCHCP_AVAILABLE,
)

__all__ = [
    # Calibration methods (probability scaling)
    "TemperatureScaling",
    "BetaCalibration",
    "IsotonicRegression",
    
    # Ensemble methods
    "EnsembleCalibration",
    "EnsembleTemperatureCalibrator",
    "MCDropoutCalibration",
    
    # 2026 Conformal methods (library-backed)
    "ConformalPolicy",
    "TemperatureConformalWrapper",
    "TORCHCP_AVAILABLE",
]
