"""
🔒️ **Artifact Validators** - Fail-Fast Validation
Implements TODO 123: Hard validators (missing/corrupt/shape/policy mutual exclusivity)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import json


@dataclass(frozen=True)
class ArtifactValidator:
    """
    Fail-fast validators for artifacts.
    
    Enforces:
    - Files exist and are non-empty
    - JSON files are valid
    - Policy files are mutually exclusive (exactly one exists)
    """
    
    def _require_exists_nonempty(self, p: Path, name: str) -> None:
        """
        Require file exists and is non-empty.
        
        Args:
            p: File path to validate
            name: Artifact name (for error messages)
        
        Raises:
            FileNotFoundError: If file missing
            ValueError: If file is empty/corrupt
        """
        if not p.exists():
            raise FileNotFoundError(f"[Validator] Missing {name}: {p}")
        if p.stat().st_size == 0:
            raise ValueError(f"[Validator] Empty/corrupt {name}: {p}")
    
    def validate_json(self, p: Path, name: str) -> dict:
        """
        Validate JSON file exists and is valid.
        
        Args:
            p: JSON file path
            name: Artifact name
        
        Returns:
            Parsed JSON dict
        
        Raises:
            FileNotFoundError, ValueError: If missing or invalid JSON
        """
        self._require_exists_nonempty(p, name)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    def validate_policy_mutual_exclusive(self, policy_files: Sequence[Path]) -> Path:
        """
        Ensure exactly one policy file exists (mutual exclusivity).
        
        This prevents silent bugs where multiple policy methods
        are present, causing ambiguity about which was used.
        
        Args:
            policy_files: List of policy file paths (e.g., thresholds.json, gate_params.json)
        
        Returns:
            The existing policy file path
        
        Raises:
            ValueError: If zero or multiple policy files exist
        """
        existing = [p for p in policy_files if p.exists() and p.stat().st_size > 0]
        if len(existing) == 0:
            raise ValueError(
                "[Validator] Policy mutual exclusivity violated: "
                f"Found 0 policy files. Expected exactly 1 of: {[p.name for p in policy_files]}"
            )
        if len(existing) > 1:
            raise ValueError(
                "[Validator] Policy mutual exclusivity violated: "
                f"Found {len(existing)} policy files ({[p.name for p in existing]}). "
                "Expected exactly 1!"
            )
        return existing[0]
    
    def validate_required_files(self, files: Iterable[tuple[str, Path]]) -> None:
        """
        Validate all required files exist and are non-empty.
        
        Args:
            files: Iterable of (name, Path) tuples
        
        Raises:
            FileNotFoundError, ValueError: If any file missing or empty
        """
        for name, p in files:
            self._require_exists_nonempty(p, name)
    
    def validate_tensor_shape(self, tensor, expected_shape, name: str) -> None:
        """
        Validate tensor has expected shape.
        
        Args:
            tensor: PyTorch tensor to validate
            expected_shape: Expected shape (tuple)
            name: Tensor name (for error messages)
        
        Raises:
            ValueError: If shape mismatch
        """
        import torch
        if tensor.shape != expected_shape:
            raise ValueError(
                f"[Validator] Shape mismatch {name}: expected {expected_shape}, got {tensor.shape}"
            )
    
    def validate_probabilities_range(self, probs, name: str) -> None:
        """
        Validate probabilities are in [0, 1] and sum to 1.
        
        Args:
            probs: Probability tensor [B, C]
            name: Tensor name (for error messages)
        
        Raises:
            ValueError: If probabilities out of range or don't sum to 1
        """
        import torch
        if torch.any(probs < 0) or torch.any(probs > 1):
            raise ValueError(f"[Validator] Probabilities out of range {name}: min={probs.min():.4f}, max={probs.max():.4f}")
        
        # Check sum is close to 1 (allow floating point tolerance)
        sums = probs.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-3):
            violators = ~torch.allclose(sums, torch.ones_like(sums), atol=1e-3)
            raise ValueError(
                f"[Validator] Probabilities don't sum to 1 {name}: "
                f"found {torch.where(violators, 'sum', sums)[0].item():.6f}"
            )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "ArtifactValidator",
]

