"""
Configs module for Stage-1 Pro System (2025 Best Practices)
"""

from .artifacts import ArtifactPaths, get_artifact_registry
from .split_contracts import (
    Split,
    SplitRules,
    Phase1SplitRules,
    Phase2SplitRules,
    Phase3SplitRules,
    validate_splits_for_phase,
    get_allowed_splits,
)

__all__ = [
    "ArtifactPaths",
    "get_artifact_registry",
    "Split",
    "SplitRules",
    "Phase1SplitRules",
    "Phase2SplitRules",
    "Phase3SplitRules",
    "validate_splits_for_phase",
    "get_allowed_splits",
]
