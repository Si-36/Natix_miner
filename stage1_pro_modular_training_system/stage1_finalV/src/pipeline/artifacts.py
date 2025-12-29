"""
🔒️ **Artifacts** - Canonical Artifact Keys (No Raw Paths)
Implements: ArtifactKey enum + ArtifactStore for path resolution
This prevents "forgot to save X" bugs and provides atomic writes
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
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
        key_paths = {
            ArtifactKey.RUN_MANIFEST: self.artifact_root / "runs" / run_id / "run_manifest.json",
            
            # Phase 1
            ArtifactKey.MODEL_CHECKPOINT: self.artifact_root / "runs" / run_id / "phase1" / "model_best.pth",
            ArtifactKey.VAL_SELECT_LOGITS: self.artifact_root / "runs" / run_id / "phase1" / "val_select_logits.pt",
            ArtifactKey.VAL_SELECT_LABELS: self.artifact_root / "runs" / run_id / "phase1" / "val_select_labels.pt",
            ArtifactKey.VAL_SELECT_METRICS: self.artifact_root / "runs" / run_id / "phase1" / "metrics.csv",
            
            # 🔥 2026 PRO: Calibration artifacts (exported from Phase 1)
            ArtifactKey.VAL_CALIB_LOGITS: self.artifact_root / "runs" / run_id / "phase1" / "val_calib_logits.pt",
            ArtifactKey.VAL_CALIB_LABELS: self.artifact_root / "runs" / run_id / "phase1" / "val_calib_labels.pt",
            
            # Phase 2
            ArtifactKey.THRESHOLDS_JSON: self.artifact_root / "runs" / run_id / "phase2" / "thresholds.json",
            ArtifactKey.THRESHOLDS_METRICS: self.artifact_root / "runs" / run_id / "phase2" / "thresholds_metrics.csv",
            
            # Phase 3
            ArtifactKey.GATE_CHECKPOINT: self.artifact_root / "runs" / run_id / "phase3" / "gate_best.pth",
            ArtifactKey.GATE_PARAMS_JSON: self.artifact_root / "runs" / run_id / "phase3" / "gate_params.json",
            ArtifactKey.GATE_EVALUATION_METRICS: self.artifact_root / "runs" / run_id / "phase3" / "gate_metrics.csv",
            
            # Phase 6
            ArtifactKey.BUNDLE_JSON: self.artifact_root / "runs" / run_id / "phase6" / "export" / "bundle.json",
            ArtifactKey.BUNDLE_README: self.artifact_root / "runs" / run_id / "phase6" / "export" / "README.md",
        }
        
        if key not in key_paths:
            raise ValueError(f"Unknown artifact key: {key}")
        
        return key_paths[key]
    
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
    
    def put(self, key: ArtifactKey, data: Any, run_id: str = "current") -> Path:
        """
        Atomically write artifact data.
        
        🔥 2026 PRO: Real atomic writes with fsync!
        
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
            data: Data to write (bytes, str, tensor, dict, list, json-serializable)
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
        if isinstance(data, (torch.Tensor,)):
            # 🔥 Binary data (tensors, bytes)
            self._put_binary_data(data, target_path, key, run_id)
        
        elif isinstance(data, (dict, list)):
            # 🔥 Text/JSON data
            self._put_json_data(data, target_path, key, run_id)
        
        else:
            raise TypeError(f"Unsupported data type for artifact write: {type(data)}")
        
        return target_path
    
    def _put_binary_data(self, data: torch.Tensor | bytes, target_path: Path, key: ArtifactKey, run_id: str) -> None:
        """
        Atomically write binary data (tensors, bytes) with fsync.
        
        Args:
            data: Binary data to write
            target_path: Target file path
            key: Artifact key
            run_id: Run identifier
        
        🔥 2026 PRO: Uses real fsync for crash safety!
        """
        # Create temp file in same directory
        import tempfile
        temp_dir = tempfile.mkdtemp(dir=str(target_path.parent))
        temp_file = temp_dir / f"tmp_{target_path.name}.bin"
        
        try:
            # Write data
            if isinstance(data, torch.Tensor):
                # Save tensor
                torch.save(data, temp_file)
                data_size = temp_file.stat().st_size
            else:
                # Save bytes
                temp_file.write_bytes(data)
                data_size = temp_file.stat().st_size
            
            # 🔥 CRITICAL: Flush to disk (fsync on file descriptor!)
            temp_fd = temp_file.open("r")
            os.fsync(temp_fd)
            os.close(temp_fd)
            
            # Atomic replace
            os.replace(temp_file, target_path)
            print(f"✅ Atomic write (binary with fsync): {target_path}")
            
            # Clean up
            os.remove(temp_file)
            os.removedirs(temp_dir, exist_ok=True)
            
            # Record hash
            file_hash = self._hash_file(target_path)
            self._hashes[key.value] = file_hash
            self._update_manifest_hashes()
            
        except Exception as e:
            # On failure, clean up temp
            os.remove(temp_file)
            os.removedirs(temp_dir, exist_ok=True)
            raise e
    
    def _put_json_data(self, data: Any, target_path: Path, key: ArtifactKey, run_id: str) -> None:
        """
        Atomically write JSON/text data.
        
        🔥 2026 PRO: Uses real fsync for crash safety!
        """
        # Create temp file in same directory
        import tempfile
        temp_dir = tempfile.mkdtemp(dir=str(target_path.parent))
        temp_file = temp_dir / f"tmp_{target_path.name}.json"
        
        try:
            # Serialize JSON
            if isinstance(data, (dict, list)):
                json_data = json.dumps(data, indent=2)
            else:
                json_data = str(data)
            
            # Write to temp
            with temp_file.open("w", encoding="utf-8") as f:
                f.write(json_data)
            
            # 🔥 CRITICAL: Flush to disk (fsync on file descriptor!)
            temp_fd = temp_file.open("r")
            os.fsync(temp_fd)
            os.close(temp_fd)
            
            # Atomic replace
            os.replace(temp_file, target_path)
            print(f"✅ Atomic write (json with fsync): {target_path}")
            
            # Clean up
            os.remove(temp_file)
            os.removedirs(temp_dir, exist_ok=True)
            
            # Record hash
            file_hash = self._hash_file(target_path)
            self._hashes[key.value] = file_hash
            self._update_manifest_hashes()
            
        except Exception as e:
            # On failure, clean up temp
            os.remove(temp_file)
            os.removedirs(temp_dir, exist_ok=True)
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
        import tempfile
        temp_dir = tempfile.mkdtemp(dir=str(target_path.parent))
        temp_file = temp_dir / f"tmp_{target_path.name}"
        
        # Copy
        import shutil
        shutil.copy2(source_path, temp_file)
        
        # 🔥 CRITICAL: Flush to disk (fsync on file descriptor!)
        temp_fd = temp_file.open("r")
        os.fsync(temp_fd)
        os.close(temp_fd)
        
        # Atomic replace
        shutil.replace(temp_file, target_path)
        print(f"✅ Atomic write (copy with fsync): {target_path}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        # Record hash
        file_hash = self._hash_file(target_path)
        self._hashes[key.value] = file_hash
        self._update_manifest_hashes()
        
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
    
    def hash_exists(self, key: ArtifactKey, run_id: str = "current", expected_hash: str) -> bool:
        """
        Check if artifact hash matches expected.
        
        Used for:
        - Caching: Skip step if hash matches
        - Resume: Only resume if hash matches expected
        
        Args:
            key: Canonical artifact key
            run_id: Run identifier
            expected_hash: Expected hash value
        
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
        return manifest
    
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
        
        return manifest
    
    def initialize_manifest(self, run_id: str, config: Dict[str, Any], git_commit: str = None, env: Dict[str, Any] = None) -> Path:
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
    
    def update_step(self, step_id: str, status: str = "running", metadata: Dict[str, Any] = None) -> None:
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
        
        self.save_manifest(self.artifact_root / "runs" / self._manifest.get("run_id", "unknown") / "run_manifest.json")
        return None
    
    def finalize_step(self, step_id: str, status: str = "completed", metrics: Dict[str, Any] = None) -> None:
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
        
        if step_id not in self._manifest.get("steps", {}):
            raise ValueError(f"Cannot finalize unknown step: {step_id}")
        
        # Update step status
        self._manifest["steps"][step_id] = {
            "status": status,
            "metadata": metrics or {},
        }
        
        self.save_manifest(self.artifact_root / "runs" / self._manifest.get("run_id", "unknown") / "run_manifest.json")
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
