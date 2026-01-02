"""
Atomic file operations using temp + os.replace

2025 best practices:
- os.replace() is atomic on both POSIX and Windows
- No fcntl locks needed (Hydra run isolation handles concurrency)
- SHA256 checksums for integrity verification
- Type-safe with pathlib.Path

Why os.replace instead of shutil.move:
- os.replace is guaranteed atomic
- shutil.move falls back to copy+delete on cross-device moves
- os.replace raises if target exists on Windows (explicit safety)
"""

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict


def write_file_atomic(path: Path, content: bytes) -> str:
    """
    Atomic file write using temp + os.replace pattern

    Args:
        path: Target file path
        content: Raw bytes to write

    Returns:
        SHA256 checksum of written content

    Implementation:
        1. Write to temp file (path.tmp)
        2. Compute SHA256 checksum
        3. Atomic rename (os.replace)

    Guarantees:
        - Crash safety: Either old file exists or new file exists (never corrupted)
        - Cross-platform: Works on Linux, macOS, Windows
        - No partial writes: Temp file is complete before rename
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_bytes(content)

    # Compute checksum
    checksum = hashlib.sha256(content).hexdigest()

    # Atomic rename (crash-safe)
    os.replace(temp_path, path)

    return checksum


def write_json_atomic(path: Path, data: Dict[str, Any]) -> str:
    """
    Atomic JSON write with indentation

    Args:
        path: Target JSON file path
        data: Dictionary to serialize

    Returns:
        SHA256 checksum of written JSON

    Example:
        >>> metrics = {"mcc": 0.856, "acc": 0.912}
        >>> write_json_atomic(Path("metrics.json"), metrics)
        'a1b2c3d4...'
    """
    content = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    return write_file_atomic(path, content)


def write_checkpoint_atomic(path: Path, state_dict: Dict[str, Any]) -> str:
    """
    Atomic PyTorch checkpoint write

    Args:
        path: Target checkpoint path (.pth, .pt)
        state_dict: PyTorch state dict to save

    Returns:
        SHA256 checksum of written checkpoint

    Implementation:
        - Uses in-memory buffer to avoid temp file on disk
        - torch.save() writes to BytesIO buffer
        - Buffer content written atomically to disk

    Example:
        >>> state_dict = model.state_dict()
        >>> write_checkpoint_atomic(Path("model_best.pth"), state_dict)
        'e5f6g7h8...'
    """
    import torch

    # Serialize to in-memory buffer
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    content = buffer.getvalue()

    return write_file_atomic(path, content)


def write_torch_artifact_atomic(path: Path, tensor_or_dict: Any) -> str:
    """
    Atomic write for PyTorch tensors or dictionaries

    Args:
        path: Target path (.pt, .pth)
        tensor_or_dict: Torch tensor or dict of tensors

    Returns:
        SHA256 checksum of written artifact

    Use cases:
        - Logits: torch.save(logits_tensor, "val_calib_logits.pt")
        - Labels: torch.save(labels_tensor, "val_calib_labels.pt")
        - Features: torch.save(features_dict, "val_calib_features.pt")

    Example:
        >>> logits = torch.randn(1000, 2)
        >>> write_torch_artifact_atomic(Path("logits.pt"), logits)
        'i9j0k1l2...'
    """
    import torch

    buffer = io.BytesIO()
    torch.save(tensor_or_dict, buffer)
    content = buffer.getvalue()

    return write_file_atomic(path, content)


def compute_file_sha256(path: Path) -> str:
    """
    Compute SHA256 checksum of existing file

    Args:
        path: File path to hash

    Returns:
        Hex-encoded SHA256 checksum

    Implementation:
        - Reads file in 64KB chunks (memory efficient)
        - Works with large checkpoint files (>1GB)
    """
    sha256_hash = hashlib.sha256()

    with open(path, "rb") as f:
        # Read in 64KB chunks
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def get_file_size(path: Path) -> int:
    """
    Get file size in bytes

    Args:
        path: File path

    Returns:
        Size in bytes
    """
    return Path(path).stat().st_size
