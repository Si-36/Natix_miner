"""
🔒️ **Step API** - Typed Interfaces for Step/Asset Pipeline (2026 Pro Standard)
StepSpec, StepContext, StepResult - TYPED for production-grade contracts
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar, Dict, List, Optional, FrozenSet, Any
from enum import Enum
import json
from datetime import datetime
import hashlib


# ============================================================================
# StepResult - Typed Step Execution Result
# ============================================================================

@dataclass(frozen=True)
class StepResult:
    """
    Result of step execution.
    
    Provides:
    - artifacts_written: List of ArtifactKey (canonical names)
    - splits_used: Set of Split used (enforced by contracts)
    - metrics: Optional runtime metrics (timing, sizes, etc.)
    - metadata: Rich metadata (owners, tags, runtime stats)
    """
    artifacts_written: List[str]  # Canonical artifact keys produced
    splits_used: FrozenSet[str]  # Splits actually used
    metrics: Optional[Dict[str, Any]]  # Runtime metrics (timings, sizes)
    metadata: Dict[str, Any]  # Owners, tags, runtime stats
    
    def __post_init__(self):
        """Validate results after initialization"""
        # Ensure artifacts_written is a list (not empty if failed)
        if not isinstance(self.artifacts_written, list):
            raise TypeError("artifacts_written must be a list of ArtifactKey strings")
        
        # Ensure splits_used is a frozenset
        if not isinstance(self.splits_used, frozenset):
            raise TypeError("splits_used must be a frozenset of Split enums")


# ============================================================================
# StepContext - Runtime State (Metadata + State Tracking)
# ============================================================================

@dataclass
class StepContext:
    """
    Runtime context for step execution.
    
    Provides:
    - step_id: Unique identifier
    - config: Resolved Hydra config snapshot
    - run_id: Current run identifier
    - artifact_root: Artifact store root (for path resolution)
    - metadata: Step-level metadata (owners, tags)
    - state: Optional runtime state (for step chaining)
    """
    step_id: str  # Step unique identifier (e.g., train_baseline_head)
    config: Dict[str, Any]  # Resolved Hydra config (snapshot)
    run_id: str  # Current run ID (YYYYMMDD-HHMMSS)
    artifact_root: Path  # Root for artifact path resolution
    metadata: Dict[str, str] = field(default_factory=dict)  # Owners, tags
    state: Optional[Dict[str, Any]] = None  # Runtime state (for step chaining)
    
    def __post_init__(self):
        """Validate context after initialization"""
        # Ensure artifact_root is a Path
        if not isinstance(self.artifact_root, Path):
            raise TypeError("artifact_root must be a Path")


# ============================================================================
# StepSpec - Abstract Base for All Steps
# ============================================================================

class StepSpec(ABC):
    """
    Abstract base class for all step specifications.
    
    Each step must declare:
    - name: Domain name (e.g., train_baseline_head)
    - deps: List of step names this step depends on
    - order_index: Optional ordering hint
    - owners: List of owner identifiers
    - tags: Dict of tag key-values
    - inputs(): List of ArtifactKey required for this step
    - outputs(): List of ArtifactKey this step produces
    - allowed_splits(): Set of Split this step is allowed to use
    - run(ctx): Execute step logic
    
    This design enforces "hidden contracts" so steps can't:
    - Use unauthorized splits (leak-proof)
    - Access missing artifacts (fail-fast dependency)
    - Write artifacts outside declared outputs (forbidden)
    """
    
    step_id: str
    name: str
    deps: List[str] = field(default_factory=list)
    order_index: Optional[int] = None  # Optional ordering hint
    owners: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    
    @abstractmethod
    def inputs(self, ctx: StepContext) -> List[str]:
        """
        Declare required input artifacts for this step.
        
        Returns:
            List of ArtifactKey canonical names (NOT paths!)
        
        Example:
            return [ArtifactKey.MODEL_CHECKPOINT, ArtifactKey.CALIB_LOGITS]
        """
        pass
    
    @abstractmethod
    def outputs(self, ctx: StepContext) -> List[str]:
        """
        Declare output artifacts this step produces.
        
        Returns:
            List of ArtifactKey canonical names (NOT paths!)
        
        Example:
            return [ArtifactKey.THRESHOLDS_JSON, ArtifactKey.METRICS_CSV]
        """
        pass
    
    @abstractmethod
    def allowed_splits(self) -> FrozenSet[str]:
        """
        Declare which data splits this step is allowed to use.
        
        Returns:
            FrozenSet of Split enum values (e.g., TRAIN, VAL_SELECT)
        
        Example (training step):
            return frozenset([Split.TRAIN, Split.VAL_SELECT])
        
        Example (calibration step):
            return frozenset([Split.VAL_CALIB])
        """
        pass
    
    @abstractmethod
    def run(self, ctx: StepContext) -> StepResult:
        """
        Execute step logic.
        
        Args:
            ctx: Runtime context with artifact_root, config, etc.
        
        Returns:
            StepResult with artifacts_written, splits_used, metrics, metadata
        """
        pass
    
    def __post_init__(self):
        """Validate spec after initialization"""
        # Ensure deps is a list
        if not isinstance(self.deps, list):
            raise TypeError("deps must be a list")
        
        # Ensure name is a non-empty string
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("name must be a non-empty string")


__all__ = [
    "StepSpec",
    "StepContext",
    "StepResult",
]

