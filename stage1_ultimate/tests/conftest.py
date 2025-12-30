"""
Pytest fixtures for integration tests

Provides:
- Tiny config for fast tests (~5 min on CPU)
- Temporary output directories
- Helper functions for manifest loading
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def tiny_config() -> DictConfig:
    """
    Tiny configuration for integration tests

    Features:
    - 10 samples max (fast execution)
    - 1 epoch only
    - CPU-safe (no GPU required)
    - All production features enabled

    Time: ~5 minutes on CPU
    """
    config_dict = {
        "data": {
            "data_root": "${oc.env:HOME}/data/natix_subset",
            "max_samples": 10,  # Tiny dataset
            "dataloader": {
                "batch_size": 2,
                "val_batch_size": 2,
                "num_workers": 0,  # Single-threaded for tests
            },
        },
        "training": {
            "epochs": 1,  # Minimal training
            "val_check_interval": 1.0,
            "log_every_n_steps": 1,
        },
        "model": {
            "backbone": "dinov2_vits14",  # Smallest backbone
            "lr": 0.001,
            "weight_decay": 0.0001,
            "explora": {
                "use_labeled_data": False,  # Unsupervised (no splits.json)
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
            },
        },
        "hardware": {
            "num_gpus": 0,  # CPU-only for tests
            "precision": "32",
        },
        "pipeline": {
            "phases": ["phase1"],  # Override per test
            "save_state": True,
        },
        "output_dir": None,  # Set by temp_output_dir fixture
    }

    return OmegaConf.create(config_dict)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """
    Temporary output directory for test runs

    Automatically cleaned up after test
    """
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir

    # Cleanup (pytest handles tmp_path cleanup automatically)


@pytest.fixture
def tiny_config_with_output(tiny_config: DictConfig, temp_output_dir: Path) -> DictConfig:
    """
    Tiny config with temporary output directory set
    """
    cfg = OmegaConf.create(OmegaConf.to_container(tiny_config))
    cfg.output_dir = str(temp_output_dir)
    return cfg


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    Load manifest JSON file

    Args:
        manifest_path: Path to manifest.json

    Returns:
        Manifest dictionary

    Raises:
        FileNotFoundError: If manifest doesn't exist
        json.JSONDecodeError: If manifest is malformed
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        return json.load(f)


def compute_file_sha256(file_path: Path) -> str:
    """
    Compute SHA256 checksum of file

    Args:
        file_path: Path to file

    Returns:
        Hex-encoded SHA256 checksum
    """
    import hashlib

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_manifest(manifest: Dict[str, Any], base_dir: Path) -> None:
    """
    Verify all artifacts in manifest exist and checksums match

    Args:
        manifest: Manifest dictionary
        base_dir: Base directory for resolving paths

    Raises:
        FileNotFoundError: If artifact missing
        ValueError: If checksum mismatch
    """
    for artifact_name, artifact_info in manifest.get("output_artifacts", {}).items():
        artifact_path = base_dir / artifact_info["path"]

        # Check exists
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Artifact missing: {artifact_name} at {artifact_path}"
            )

        # Check checksum
        actual_sha256 = compute_file_sha256(artifact_path)
        expected_sha256 = artifact_info["sha256"]

        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {artifact_name}:\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual:   {actual_sha256}"
            )


# Export helper functions for tests
__all__ = [
    "tiny_config",
    "temp_output_dir",
    "tiny_config_with_output",
    "load_manifest",
    "compute_file_sha256",
    "verify_manifest",
]
