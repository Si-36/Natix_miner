"""
Pytest Configuration and Shared Fixtures

Latest 2025-2026 practices:
- Python 3.14+ with modern fixtures
- Clear fixture scope management
- Reusable test utilities
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from typing import Generator

from src.contracts.artifact_schema import ArtifactSchema, create_artifact_schema
from src.contracts.split_contracts import Split


@pytest.fixture(scope="session")
def temp_output_dir() -> Generator[Path, None, None]:
    """
    Create a temporary output directory for tests

    Scope: session (one directory for all tests)
    Cleanup: Automatic after all tests complete
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="streetvision_test_"))
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def artifacts(temp_output_dir: Path) -> ArtifactSchema:
    """
    Create an ArtifactSchema instance for testing

    Scope: function (new instance for each test)
    """
    artifacts = create_artifact_schema(str(temp_output_dir))
    artifacts.ensure_dirs()  # Create all directories
    return artifacts


@pytest.fixture
def sample_splits_json(artifacts: ArtifactSchema) -> Path:
    """
    Create a sample splits.json file for testing

    Returns:
        Path to the created splits.json file
    """
    import json

    splits_data = {
        "train": ["image_001.jpg", "image_002.jpg", "image_003.jpg"],
        "val_select": ["image_004.jpg", "image_005.jpg"],
        "val_calib": ["image_006.jpg", "image_007.jpg"],
        "val_test": ["image_008.jpg", "image_009.jpg"],
        "metadata": {
            "total_images": 9,
            "split_strategy": "balanced",
            "created_at": "2025-12-28T00:00:00Z",
        },
    }

    splits_path = artifacts.splits_json
    splits_path.write_text(json.dumps(splits_data, indent=2))
    return splits_path


@pytest.fixture
def sample_checkpoint(artifacts: ArtifactSchema) -> Path:
    """
    Create a sample checkpoint file for testing

    Returns:
        Path to the created checkpoint file
    """
    import torch

    checkpoint = {
        "epoch": 10,
        "model_state_dict": {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        },
        "optimizer_state_dict": {},
        "loss": 0.123,
        "accuracy": 0.892,
    }

    checkpoint_path = artifacts.phase1_checkpoint
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def sample_logits(artifacts: ArtifactSchema) -> Path:
    """
    Create sample logits.npy file for testing

    Returns:
        Path to the created logits file
    """
    import numpy as np

    # Shape: (num_samples, num_classes)
    logits = np.random.randn(100, 13).astype(np.float32)

    logits_path = artifacts.val_calib_logits
    np.save(logits_path, logits)
    return logits_path


@pytest.fixture
def sample_labels(artifacts: ArtifactSchema) -> Path:
    """
    Create sample labels.npy file for testing

    Returns:
        Path to the created labels file
    """
    import numpy as np

    # Shape: (num_samples,)
    labels = np.random.randint(0, 13, size=100).astype(np.int64)

    labels_path = artifacts.val_calib_labels
    np.save(labels_path, labels)
    return labels_path


# Test utilities
def create_dummy_file(path: Path, content: str = "dummy") -> Path:
    """
    Create a dummy file for testing

    Args:
        path: Path to create
        content: Content to write

    Returns:
        Path to the created file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
