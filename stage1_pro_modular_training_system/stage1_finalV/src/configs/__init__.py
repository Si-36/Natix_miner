"""
Configs module for Stage-1 Pro System (2025 Best Practices)
"""

from .artifacts import ArtifactPaths, get_artifact_registry
from pipeline.contracts import Split, assert_allowed

__all__ = [
    "ArtifactPaths",
    "get_artifact_registry",
    "Split",
    "assert_allowed",
]
