"""
🔒️ **Artifacts** - Canonical Artifact Keys (No Raw Paths)
Implements: ArtifactKey enum + ArtifactStore for path resolution
This prevents "forgot to save X" bugs and provides atomic writes

🔥 2026 PRO: Real atomic writes with os.fsync()!
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union
import hashlib
import os
import tempfile
import json
import torch


@dataclass(frozen=True)
class ArtifactKey(str, Enum):
    """
    Canonical artifact namespace - no raw paths allowed.

    All artifact accesses must go through ArtifactKey enum.
    Example usage:
        store.get(ArtifactKey.MODEL_CHECKPOINT)  # Returns Path
        store.put(ArtifactKey.MODEL_CHECKPOINT, ...)   # Atomic write
    """

    # Run-level artifacts
    RUN_MANIFEST = "run_manifest"

    # Phase 1: Training
    MODEL_CHECKPOINT = "model_checkpoint"
    VAL_SELECT_LOGITS = "val_select_logits"
    VAL_SELECT_LABELS = "val_select_labels"
    VAL_SELECT_METRICS = "val_select_metrics"

    # 🔥 2026 PRO: Calibration artifacts (from Phase 1)
    # These are exported from Phase 1 and consumed by Phase 2
    VAL_CALIB_LOGITS = "val_calib_logits"
    VAL_CALIB_LABELS = "val_calib_labels"

    # Phase 2: Thresholds / Policy
    THRESHOLDS_JSON = "thresholds_json"
    THRESHOLDS_METRICS = "thresholds_metrics"
    POLICY_JSON = "policy_json"  # Alternative policy artifact

    # Phase 3: Gate
    GATE_CHECKPOINT = "gate_checkpoint"
    GATE_PARAMS_JSON = "gate_params_json"
    GATE_EVALUATION_METRICS = "gate_evaluation_metrics"

    # Phase 6: Export Bundle
    BUNDLE_JSON = "bundle_json"
    BUNDLE_README = "bundle_readme"


class ArtifactStore:
    """
    Artifact store with atomic writes and path resolution.

    🔥 2026 PRO: Real atomic writes with os.fsync()!

    Features:
    - get(key): Resolve canonical key to absolute path
    - put(key, data): Atomic write with streaming hash
    - hash_exists(key, expected_hash): Check if hash matches
    - get_loader(split, run_id): Get data loader for split (step-safe)
    - initialize_manifest(): Create run manifest with Git SHA + config snapshot
    - finalize_step(step_id, status, metrics): Finalize step in manifest
    - save_manifest(): Save manifest to disk
    - load_manifest(manifest_path): Load manifest from disk
    - update_step(step_id, status, metadata): Update step status in manifest
    - get_run_dir(run_id): Get run directory path

    Prevents:
    - Partial artifacts (crashes)
    - Path drift (renames not reflected in store)
    - "Forgot to save X" bugs (no central source of truth)
    """

    def __init__(self, artifact_root: Path):
        """
        Initialize artifact store.

        Args:
            artifact_root: Root directory for all artifacts
        """
        self.artifact_root = Path(artifact_root)
        self._hashes: Dict[str, str] = {}  # key -> hash cache
        self._manifest: Optional[Dict[str, Any]] = None  # Loaded manifest
        self._manifest_path: Optional[Path] = None  # Path to loaded manifest

    def _get_key_path(self, key: ArtifactKey, run_id: str) -> Path:
        """
        Resolve artifact key to absolute path.

        Args:
            key: Canonical artifact key
            run_id: Run identifier

        Returns:
            Absolute path to artifact
        """
        # Key-specific path resolution
        run_dir = self.artifact_root / "runs" / run_id

        key_paths = {
            # 🔥 CRITICAL: Use .value (string) not enum object as key!
            # Python 3.14+ enum hash collision bug - use string keys
            # Run-level
            ArtifactKey.RUN_MANIFEST.value: run_dir / "run_manifest.json",
            # Phase 1
            ArtifactKey.MODEL_CHECKPOINT.value: run_dir / "phase1" / "model_best.pth",
            ArtifactKey.VAL_SELECT_LOGITS.value: run_dir / "phase1" / "val_select_logits.pt",
            ArtifactKey.VAL_SELECT_LABELS.value: run_dir / "phase1" / "val_select_labels.pt",
            ArtifactKey.VAL_SELECT_METRICS.value: run_dir / "phase1" / "metrics.csv",
            # 🔥 2026 PRO: Calibration artifacts (exported from Phase 1)
            ArtifactKey.VAL_CALIB_LOGITS.value: run_dir / "phase1" / "val_calib_logits.pt",
            ArtifactKey.VAL_CALIB_LABELS.value: run_dir / "phase1" / "val_calib_labels.pt",
            # Phase 2
            ArtifactKey.THRESHOLDS_JSON.value: run_dir / "phase2" / "thresholds.json",
            ArtifactKey.THRESHOLDS_METRICS.value: run_dir / "phase2" / "thresholds_metrics.csv",
            # Phase 3
            ArtifactKey.GATE_CHECKPOINT.value: run_dir / "phase3" / "gate_best.pth",
            ArtifactKey.GATE_PARAMS_JSON.value: run_dir / "phase3" / "gate_params.json",
            ArtifactKey.GATE_EVALUATION_METRICS.value: run_dir / "phase3" / "gate_metrics.csv",
            # Phase 6
            ArtifactKey.BUNDLE_JSON.value: run_dir / "phase6" / "export" / "bundle.json",
            ArtifactKey.BUNDLE_README.value: run_dir / "phase6" / "export" / "README.md",
        }

        # 🔥 CRITICAL: Use key.value (string) for lookup, not enum object
        if key.value not in key_paths:
            raise ValueError(f"Unknown artifact key: {key}")

        return key_paths[key.value]

    def _contains_tensors(self, data: Union[dict, list]) -> bool:
        """
        Check if dict or list contains torch.Tensor objects.

        This is used to detect checkpoint state_dicts which must be saved
        via torch.save() instead of JSON serialization.

        Args:
            data: Dict or list to check

        Returns:
            True if any tensor found, False otherwise
        """
        if isinstance(data, dict):
            return any(isinstance(v, torch.Tensor) for v in data.values())
        elif isinstance(data, list):
            return any(isinstance(item, torch.Tensor) for item in data)
        return False

    def get(self, key: ArtifactKey, run_id: str = "current") -> Path:
        """
        Get artifact path by key.

        Args:
            key: Canonical artifact key
            run_id: Run identifier (default to "current")

        Returns:
            Absolute path to artifact
        """
        if run_id == "current":
            # Use current run's manifest to resolve run_id
            if self._manifest is None:
                raise RuntimeError("No manifest loaded - call load_manifest() first")
            run_id = self._manifest.get("run_id", "unknown")

        return self._get_key_path(key, run_id)

    def put(
        self,
        key: ArtifactKey,
        data: Union[torch.Tensor, dict, list, str, bytes],
        run_id: str = "current",
    ) -> Path:
        """
        Atomically write artifact data.

        🔥 2026 PRO: Real atomic writes with os.fsync()!

        Atomic write pattern:
        1. Write to temp file in same directory
        2. Flush to disk (fsync on file descriptor)
        3. Atomic replace (rename temp → target)
        4. Clean up temp

        Prevents:
        - Partial artifacts (crashes mid-write)
        - Data corruption (rename before complete write)

        Args:
            key: Canonical artifact key
            data: Data to write (torch.Tensor, dict, list, str, bytes)
            run_id: Run identifier (default to "current")

        Returns:
            Path to written artifact
        """
        if run_id == "current":
            if self._manifest is None:
                raise RuntimeError("No manifest loaded - call load_manifest() first")
            run_id = self._manifest.get("run_id", "unknown")

        # Resolve target path
        target_path = self._get_key_path(key, run_id)

        # Create parent directory
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle different data types with proper atomic writes
        if isinstance(data, torch.Tensor):
            # 🔥 Binary data (tensors)
            self._put_binary_data(data, target_path, key, run_id)

        elif isinstance(data, dict):
            # 🔥 Check if dict contains tensors (checkpoint state_dict)
            if self._contains_tensors(data):
                # Dict with tensors → checkpoint/state_dict (save via torch.save)
                self._put_binary_data(data, target_path, key, run_id)
            else:
                # Regular dict → JSON
                self._put_json_data(data, target_path, key, run_id)

        elif isinstance(data, list):
            # 🔥 Check if list contains tensors
            if self._contains_tensors(data):
                # List with tensors → save via torch.save
                self._put_binary_data(data, target_path, key, run_id)
            else:
                # Regular list → JSON
                self._put_json_data(data, target_path, key, run_id)

        elif isinstance(data, str):
            # 🔥 String data (text, CSV, etc.)
            self._put_string_data(data, target_path, key, run_id)

        elif isinstance(data, bytes):
            # 🔥 Raw bytes data
            self._put_bytes_data(data, target_path, key, run_id)

        else:
            raise TypeError(f"Unsupported data type for artifact write: {type(data)}")

        return target_path

    def _put_binary_data(
        self, data: torch.Tensor, target_path: Path, key: ArtifactKey, run_id: str
    ) -> None:
        """
        Atomically write binary data (tensors) with fsync.

        Args:
            data: PyTorch tensor
            target_path: Target file path
            key: Artifact key
            run_id: Run identifier

        🔥 2026 PRO: Uses real fsync for crash safety!
        """
        # Create temp file in same directory
        temp_dir = tempfile.mkdtemp(dir=str(target_path.parent))
        temp_file_path = os.path.join(temp_dir, f"tmp_{target_path.name}.bin")

        try:
            # Save tensor to temp file
            torch.save(data, temp_file_path)

            # 🔥 CRITICAL: Flush to disk (fsync on file descriptor!)
            temp_fd = os.open(temp_file_path, os.O_RDONLY)
            os.fsync(temp_fd)
            os.close(temp_fd)

            # Atomic replace
            os.replace(temp_file_path, target_path)
            print(f"✅ Atomic write (tensor with fsync): {target_path}")

            # Clean up
            os.removedirs(temp_dir)

            # Record hash
            file_hash = self._hash_file(target_path)
            self._hashes[key.value] = file_hash
            self._update_manifest_hashes()

        except Exception as e:
            # On failure, clean up temp
            try:
                os.remove(temp_file_path)
            except:
                pass
            try:
                os.removedirs(temp_dir)
            except:
                pass
            raise e

    def _put_json_data(
        self, data: Union[dict, list], target_path: Path, key: ArtifactKey, run_id: str
    ) -> None:
        """
        Atomically write JSON/text data.

        🔥 2026 PRO: Uses real fsync for crash safety!
        """
        # Create temp file in same directory
        temp_dir = tempfile.mkdtemp(dir=str(target_path.parent))
        temp_file_path = os.path.join(temp_dir, f"tmp_{target_path.name}.json")

        try:
            # Serialize JSON
            if isinstance(data, (dict, list)):
                json_data = json.dumps(data, indent=2)
            else:
                json_data = str(data)

            # Write to temp
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(json_data)

            # 🔥 CRITICAL: Flush to disk (fsync on file descriptor!)
            temp_fd = os.open(temp_file_path, os.O_RDONLY)
            os.fsync(temp_fd)
            os.close(temp_fd)

            # Atomic replace
            os.replace(temp_file_path, target_path)
            print(f"✅ Atomic write (json with fsync): {target_path}")

            # Clean up
            os.removedirs(temp_dir)

            # Record hash
            file_hash = self._hash_file(target_path)
            self._hashes[key.value] = file_hash
            self._update_manifest_hashes()

        except Exception as e:
            # On failure, clean up temp
            try:
                os.remove(temp_file_path)
            except:
                pass
            try:
                os.removedirs(temp_dir)
            except:
                pass
            raise e

    def _put_string_data(self, data: str, target_path: Path, key: ArtifactKey, run_id: str) -> None:
        """
        Atomically write string data (text, CSV, etc.).

        🔥 2026 PRO: Uses real fsync for crash safety!
        """
        # Create temp file in same directory
        temp_dir = tempfile.mkdtemp(dir=str(target_path.parent))
        temp_file_path = os.path.join(temp_dir, f"tmp_{target_path.name}.txt")

        try:
            # Write to temp
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(data)

            # 🔥 CRITICAL: Flush to disk (fsync on file descriptor!)
            temp_fd = os.open(temp_file_path, os.O_RDONLY)
            os.fsync(temp_fd)
            os.close(temp_fd)

            # Atomic replace
            os.replace(temp_file_path, target_path)
            print(f"✅ Atomic write (string with fsync): {target_path}")

            # Clean up
            os.removedirs(temp_dir)

            # Record hash
            file_hash = self._hash_file(target_path)
            self._hashes[key.value] = file_hash
            self._update_manifest_hashes()

        except Exception as e:
            # On failure, clean up temp
            try:
                os.remove(temp_file_path)
            except:
                pass
            try:
                os.removedirs(temp_dir)
            except:
                pass
            raise e

    def _put_bytes_data(
        self, data: bytes, target_path: Path, key: ArtifactKey, run_id: str
    ) -> None:
        """
        Atomically write raw bytes data.

        🔥 2026 PRO: Uses real fsync for crash safety!
        """
        # Create temp file in same directory
        temp_dir = tempfile.mkdtemp(dir=str(target_path.parent))
        temp_file_path = os.path.join(temp_dir, f"tmp_{target_path.name}.bin")

        try:
            # Write bytes to temp
            with open(temp_file_path, "wb") as f:
                f.write(data)

            # 🔥 CRITICAL: Flush to disk (fsync on file descriptor!)
            temp_fd = os.open(temp_file_path, os.O_RDONLY)
            os.fsync(temp_fd)
            os.close(temp_fd)

            # Atomic replace
            os.replace(temp_file_path, target_path)
            print(f"✅ Atomic write (bytes with fsync): {target_path}")

            # Clean up
            os.removedirs(temp_dir)

            # Record hash
            file_hash = self._hash_file(target_path)
            self._hashes[key.value] = file_hash
            self._update_manifest_hashes()

        except Exception as e:
            # On failure, clean up temp
            try:
                os.remove(temp_file_path)
            except:
                pass
            try:
                os.removedirs(temp_dir)
            except:
                pass
            raise e

    def put_from_file(self, key: ArtifactKey, source_path: Path, run_id: str = "current") -> Path:
        """
        Copy file to artifact location with atomic write.

        🔥 2026 PRO: Real atomic writes with fsync!
        """
        if run_id == "current":
            if self._manifest is None:
                raise RuntimeError("No manifest loaded - call load_manifest() first")
            run_id = self._manifest.get("run_id", "unknown")

        target_path = self._get_key_path(key, run_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic copy: copy to temp → fsync → replace
        temp_dir = tempfile.mkdtemp(dir=str(target_path.parent))
        temp_file_path = os.path.join(temp_dir, f"tmp_{target_path.name}")

        try:
            # Copy
            import shutil

            shutil.copy2(source_path, temp_file_path)

            # 🔥 CRITICAL: Flush to disk (fsync on file descriptor!)
            temp_fd = os.open(temp_file_path, os.O_RDONLY)
            os.fsync(temp_fd)
            os.close(temp_fd)

            # Atomic replace
            shutil.replace(temp_file_path, target_path)
            print(f"✅ Atomic write (copy with fsync): {target_path}")

            # Clean up
            os.removedirs(temp_dir)

            # Record hash
            file_hash = self._hash_file(target_path)
            self._hashes[key.value] = file_hash
            self._update_manifest_hashes()

        except Exception as e:
            # On failure, clean up temp
            try:
                os.remove(temp_file_path)
            except:
                pass
            try:
                os.removedirs(temp_dir)
            except:
                pass
            raise e

        return target_path

    def exists(self, key: ArtifactKey, run_id: str = "current") -> bool:
        """
        Check if artifact exists.

        Args:
            key: Canonical artifact key
            run_id: Run identifier

        Returns:
            True if artifact exists and is non-empty
        """
        if run_id == "current":
            if self._manifest is None:
                raise RuntimeError("No manifest loaded - call load_manifest() first")
            run_id = self._manifest.get("run_id", "unknown")

        target_path = self._get_key_path(key, run_id)

        if not target_path.exists():
            return False

        if target_path.is_dir():
            return False  # Directories don't count

        if target_path.stat().st_size == 0:
            return False  # Empty files don't count

        return True

    def hash_exists(self, key: ArtifactKey, expected_hash: str, run_id: str = "current") -> bool:
        """
        Check if artifact hash matches expected.

        Used for:
        - Caching: Skip step if hash matches
        - Resume: Only resume if hash matches expected

        Args:
            key: Canonical artifact key
            expected_hash: Expected hash value
            run_id: Run identifier

        Returns:
            True if hash matches, False otherwise
        """
        if run_id == "current":
            if self._manifest is None:
                raise RuntimeError("No manifest loaded - call load_manifest() first")
            run_id = self._manifest.get("run_id", "unknown")

        if key not in self._hashes:
            return False  # No hash recorded

        return self._hashes.get(key.value, "") == expected_hash

    def _hash_file(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Compute SHA256 hash of file.

        🔥 2026 PRO: Streaming hash for large files (to avoid memory issues).

        Args:
            file_path: Path to file
            chunk_size: Chunk size in bytes (default: 8KB)

        Returns:
            Hex SHA256 hash
        """
        sha256 = hashlib.sha256()

        with file_path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)

        return sha256.hexdigest()

    def _update_manifest_hashes(self):
        """Update manifest with new hashes"""
        if self._manifest is None:
            return
        self._manifest["artifact_hashes"] = self._hashes.copy()

    def load_manifest(self, manifest_path: Path) -> None:
        """
        Load run manifest from disk.

        Args:
            manifest_path: Path to manifest.json

        Returns:
            None
        """
        if not manifest_path.exists():
            self._manifest = None
            return

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        self._manifest = manifest
        return manifest_path

    def save_manifest(self, manifest_path: Path):
        """
        Save run manifest to disk.

        Args:
            manifest_path: Path to save manifest

        Returns:
            None
        """
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

        return manifest_path

    def initialize_manifest(
        self,
        run_id: str,
        config: Dict[str, Any],
        git_commit: str = None,
        env: Dict[str, Any] = None,
    ) -> Path:
        """
        Initialize a new run manifest.

        🔥 2026 PRO: Includes config snapshot + environment info!

        Args:
            run_id: Unique run identifier (e.g., YYYYMMDD-HHMMSS)
            config: Resolved Hydra config (snapshot)
            git_commit: Git commit SHA
            env: Environment info (CUDA, Python, etc.)

        Returns:
            Path to created manifest
        """
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.artifact_root / "runs" / run_id

        # Create run manifest
        manifest = {
            "run_id": run_id,
            "timestamp": run_id.split("_")[0],  # Extract timestamp from run_id
            "resolved_config": config,
            "git_commit": git_commit,
            "environment": {
                "python": self._get_python_version(),
                "cuda": self._get_cuda_version(),
            },
            "artifact_hashes": {},  # Will be populated during run
            "steps": {},  # Will be populated during run
            "metadata": {},  # Will be populated during run
        }

        # Save manifest
        manifest_path = run_dir / "run_manifest.json"
        self.save_manifest(manifest_path)

        # Update internal manifest reference
        self._manifest = manifest
        self._manifest_path = manifest_path

        return manifest_path

    def update_step(
        self, step_id: str, status: str = "running", metadata: Dict[str, Any] = None
    ) -> None:
        """
        Update step status in manifest.

        Args:
            step_id: Step identifier
            status: Status (running, completed, failed)
            metadata: Additional metadata (timing, owners, tags)

        Returns:
            None
        """
        if self._manifest is None:
            raise RuntimeError("No manifest loaded - call load_manifest() first")

        if step_id not in self._manifest.get("steps", {}):
            self._manifest["steps"][step_id] = {
                "status": status,
                "metadata": metadata or {},
            }

        manifest_path = (
            self.artifact_root
            / "runs"
            / self._manifest.get("run_id", "unknown")
            / "run_manifest.json"
        )
        self.save_manifest(manifest_path)
        return None

    def finalize_step(
        self, step_id: str, status: str = "completed", metrics: Dict[str, Any] = None
    ) -> None:
        """
        Finalize a step as completed with metrics.

        Args:
            step_id: Step identifier
            status: Status (running → completed)
            metrics: Step metrics (timing, artifact hashes)

        Returns:
            None
        """
        if self._manifest is None:
            raise RuntimeError("No manifest loaded - call load_manifest() first")

        # 🔥 CRITICAL: Allow creating new steps (auto-add if not exists)
        if "steps" not in self._manifest:
            self._manifest["steps"] = {}

        # Update step status (create if not exists)
        self._manifest["steps"][step_id] = {
            "status": status,
            "metadata": metrics or {},
        }

        manifest_path = (
            self.artifact_root
            / "runs"
            / self._manifest.get("run_id", "unknown")
            / "run_manifest.json"
        )
        self.save_manifest(manifest_path)
        return None

    def get_run_dir(self, run_id: str = "current") -> Path:
        """
        Get run directory path.

        Args:
            run_id: Run identifier

        Returns:
            Path to run directory
        """
        if run_id == "current":
            if self._manifest is None:
                raise RuntimeError("No manifest loaded - call load_manifest() first")
            run_id = self._manifest.get("run_id", "unknown")

        return self.artifact_root / "runs" / run_id

    def _get_python_version(self) -> str:
        """Get Python version"""
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_cuda_version(self) -> str:
        """Get CUDA version"""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.version.cuda
            return "N/A"
        except Exception:
            return "N/A"


__all__ = [
    "ArtifactKey",
    "ArtifactStore",
]
