"""
Manifest-last commit pattern for lineage tracking

2025 best practices:
- Manifest is written LAST after all artifacts
- Contains SHA256 checksums of all inputs/outputs
- Git SHA for code provenance
- Config hash for reproducibility
- Metrics for quick filtering without loading artifacts

Manifest format:
{
  "run_id": "20251230T123456",
  "step_name": "phase1_baseline",
  "git_sha": "a1b2c3d",
  "config_hash": "e4f5g6h",
  "input_artifacts": {
    "splits.json": {"path": "splits.json", "sha256": "...", "size_bytes": 1234}
  },
  "output_artifacts": {
    "model_best.pth": {"path": "phase1/model_best.pth", "sha256": "...", "size_bytes": 567890}
  },
  "metrics": {"mcc": 0.856, "acc": 0.912, "fnr": 0.089},
  "duration_seconds": 3600.5,
  "hostname": "gpu-server-01",
  "python_version": "3.11.7"
}

Why manifest-last:
- If manifest exists, all artifacts are guaranteed to exist
- No need for transaction rollback
- Simple to implement, impossible to get wrong
- Works with Hydra run isolation
"""

import hashlib
import json
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .atomic import compute_file_sha256, get_file_size, write_json_atomic


@dataclass
class ArtifactInfo:
    """
    Metadata for a single artifact (file)

    Attributes:
        path: Relative path from output_dir (for portability)
        sha256: SHA256 checksum (integrity verification)
        size_bytes: File size in bytes
        created_at: ISO 8601 timestamp
    """

    path: str
    sha256: str
    size_bytes: int
    created_at: str


@dataclass
class StepManifest:
    """
    Lineage tracking manifest for a pipeline step

    Written LAST after all artifacts are saved.
    If manifest exists, all artifacts are guaranteed to exist.

    Attributes:
        run_id: Unique run identifier (YYYYMMDDTHHMMSS)
        step_name: Step name (e.g., "phase1_baseline")
        git_sha: Git commit SHA (code provenance)
        config_hash: Hash of Hydra config (reproducibility)
        input_artifacts: Dict of input file metadata
        output_artifacts: Dict of output file metadata
        metrics: Key metrics (MCC, accuracy, FNR)
        duration_seconds: Execution time
        hostname: Machine hostname
        python_version: Python version string
    """

    run_id: str
    step_name: str
    git_sha: str
    config_hash: str
    input_artifacts: Dict[str, ArtifactInfo]
    output_artifacts: Dict[str, ArtifactInfo]
    metrics: Dict[str, float]
    duration_seconds: float
    hostname: str
    python_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert ArtifactInfo dataclasses to dicts
        data["input_artifacts"] = {
            k: asdict(v) for k, v in self.input_artifacts.items()
        }
        data["output_artifacts"] = {
            k: asdict(v) for k, v in self.output_artifacts.items()
        }
        return data

    def save(self, path: Path) -> str:
        """
        Save manifest to JSON file (atomic write)

        Args:
            path: Target manifest path

        Returns:
            SHA256 checksum of manifest file
        """
        return write_json_atomic(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "StepManifest":
        """
        Load manifest from JSON file

        Args:
            path: Manifest path

        Returns:
            StepManifest instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Convert dicts back to ArtifactInfo
        data["input_artifacts"] = {
            k: ArtifactInfo(**v) for k, v in data["input_artifacts"].items()
        }
        data["output_artifacts"] = {
            k: ArtifactInfo(**v) for k, v in data["output_artifacts"].items()
        }

        return cls(**data)


def get_git_sha() -> str:
    """
    Get current Git commit SHA (short form)

    Returns:
        Short SHA (7 chars) or 'unknown' if not in Git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def get_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of Hydra config

    Args:
        config: OmegaConf config dict

    Returns:
        SHA256 hash (first 12 chars)

    Implementation:
        - Sort keys for deterministic serialization
        - Hash JSON representation
        - Return short hash for readability
    """
    config_json = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    return config_hash[:12]


def create_artifact_info(
    file_path: Path,
    output_dir: Path,
) -> ArtifactInfo:
    """
    Create ArtifactInfo for a file

    Args:
        file_path: Absolute path to file
        output_dir: Output directory (for computing relative path)

    Returns:
        ArtifactInfo with checksum and metadata
    """
    relative_path = str(file_path.relative_to(output_dir))
    sha256 = compute_file_sha256(file_path)
    size_bytes = get_file_size(file_path)
    created_at = datetime.now().isoformat()

    return ArtifactInfo(
        path=relative_path,
        sha256=sha256,
        size_bytes=size_bytes,
        created_at=created_at,
    )


def create_step_manifest(
    step_name: str,
    input_paths: List[Path],
    output_paths: List[Path],
    output_dir: Path,
    metrics: Dict[str, float],
    duration_seconds: float,
    config: Dict[str, Any],
) -> StepManifest:
    """
    Create a complete step manifest

    Args:
        step_name: Step name (e.g., "phase1_baseline")
        input_paths: List of input file paths (absolute)
        output_paths: List of output file paths (absolute)
        output_dir: Output directory root
        metrics: Dict of metrics (MCC, accuracy, etc.)
        duration_seconds: Execution time
        config: Hydra config dict

    Returns:
        StepManifest ready to save

    Example:
        >>> manifest = create_step_manifest(
        ...     step_name="phase1_baseline",
        ...     input_paths=[Path("outputs/splits.json")],
        ...     output_paths=[Path("outputs/phase1/model_best.pth")],
        ...     output_dir=Path("outputs"),
        ...     metrics={"mcc": 0.856, "acc": 0.912},
        ...     duration_seconds=3600.5,
        ...     config=cfg,
        ... )
        >>> manifest.save(Path("outputs/phase1/manifest.json"))
    """
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Create input artifact metadata
    input_artifacts = {
        path.name: create_artifact_info(path, output_dir)
        for path in input_paths
        if path.exists()
    }

    # Create output artifact metadata
    output_artifacts = {
        path.name: create_artifact_info(path, output_dir)
        for path in output_paths
        if path.exists()
    }

    return StepManifest(
        run_id=run_id,
        step_name=step_name,
        git_sha=get_git_sha(),
        config_hash=get_config_hash(config),
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        metrics=metrics,
        duration_seconds=duration_seconds,
        hostname=socket.gethostname(),
        python_version=sys.version.split()[0],
    )
