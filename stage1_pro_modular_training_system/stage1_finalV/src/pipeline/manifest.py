"""
🔒️ **Run Manifest** - Lineage Tracking for Reproducible Runs
Implements run_manifest.json with:
- Config snapshot
- Step graph
- Git commit tracking
- Artifact hashes
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import json
import platform
import subprocess


@dataclass(frozen=True)
class RunManifest:
    """
    Run manifest - captures all lineage for a single run.

    Contains:
    - run_id: Unique identifier
    - timestamp: Extracted from run_id
    - resolved_config: Full Hydra config snapshot
    - git_commit: Git SHA (optional)
    - environment: System info (Python, CUDA, OS)
    - artifact_hashes: Per-artifact SHA256 hashes
    - steps: Step execution graph
    - metadata: Run-level metadata (duration, status, etc.)
    """

    run_id: str  # e.g., YYYYMMDD-HHMMSS
    resolved_config: Dict[str, Any]

    # Timestamp (auto-extracted from run_id)
    timestamp: str = ""  # ISO timestamp extracted from run_id

    # Git info (optional)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None

    # Environment info
    environment: Dict[str, Any] = field(default_factory=dict)

    # Artifact hashes (key: artifact_key → value)
    artifact_hashes: Dict[str, str] = field(default_factory=dict)

    # Step graph (key: step_id → step info)
    steps: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metadata (run-level)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Extract timestamp from run_id"""
        # run_id format: YYYYMMDD-HHMMSS
        if len(self.run_id) >= 12 and self.run_id[8] == "T":
            # Extract timestamp
            date_str = self.run_id[:8] + "-" + self.run_id[8:]
            time_str = self.run_id[9:] + ":" + self.run_id[10:]
            object.__setattr__(self, "timestamp", f"{date_str}T{time_str}")
        else:
            object.__setattr__(self, "timestamp", self.run_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization"""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "resolved_config": self.resolved_config,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "environment": self.environment,
            "artifact_hashes": self.artifact_hashes,
            "steps": self.steps,
            "metadata": self.metadata,
        }

    def to_json(self, path: Path) -> None:
        """Save manifest to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_environment(self) -> Dict[str, Any]:
        """
        Get system environment information.

        Returns:
            Dict with python, cuda, os versions
        """
        env = {
            "python": self._get_python_version(),
            "cuda": self._get_cuda_version(),
            "os": f"{platform.system()} {platform.release()}",
        }
        self.environment = env
        return env

    @staticmethod
    def _get_python_version() -> str:
        """Get Python version"""
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    @staticmethod
    def _get_cuda_version() -> str:
        """Get CUDA version"""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.version.cuda
            return "N/A"
        except Exception:
            return "N/A"


__all__ = [
    "RunManifest",
]
