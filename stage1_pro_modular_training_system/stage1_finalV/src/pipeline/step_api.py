"""
🔒️ **Step API** - Typed Interfaces for Step/Asset Pipeline (2026 Pro Standard)
StepSpec, StepContext, StepResult - TYPED for production-grade contracts
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar, Dict, List, Optional, FrozenSet, Any, Protocol, runtime_checkable
from enum import Enum
import json
from datetime import datetime

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
    metrics: Optional[Dict[str, Any]] = None  # Runtime metrics (timings, sizes)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Owners, tags, runtime stats
    
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
    - artifact_store: ArtifactStore instance (for artifact I/O)
    - manifest: RunManifest instance (for lineage tracking)
    - metadata: Step-level metadata (owners, tags)
    - state: Optional runtime state (for step chaining)
    """
    step_id: str  # Step unique identifier (e.g., train_baseline_head)
    config: Dict[str, Any]  # Resolved Hydra config (snapshot)
    run_id: str  # Current run ID (YYYYMMDD-HHMMSS)
    artifact_root: Path  # Root for artifact path resolution
    artifact_store: Any  # ArtifactStore instance (for artifact I/O)
    manifest: Any  # RunManifest instance (for lineage tracking)
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

class StepSpec(Protocol):
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
    - Use unauthorized splits (leak-proof!)
    - Access missing artifacts (fail-fast dependency)
    - Write artifacts outside declared outputs (forbidden)
    """
    
    @property
    def step_id(self) -> str:
        """Unique step identifier (e.g., train_baseline_head)"""
        ...
    
    @property
    def name(self) -> str:
        """Human-readable step name (e.g., Train Baseline Head)"""
        ...
    
    @property
    def deps(self) -> List[str]:
        """List of step names this step depends on"""
        ...
    
    @property
    def order_index(self) -> Optional[int]:
        """Optional ordering hint for execution"""
        ...
    
    @property
    def owners(self) -> List[str]:
        """List of owner identifiers"""
        ...
    
    @property
    def tags(self) -> Dict[str, str]:
        """Dict of tag key-values"""
        ...
    
    def inputs(self, ctx: StepContext) -> List[str]:
        """
        Declare required input artifacts for this step.
        
        Returns:
            List of ArtifactKey canonical names (NOT paths!)
        
        Example:
            return [ArtifactKey.MODEL_CHECKPOINT, ArtifactKey.CALIB_LOGITS]
        """
        ...
    
    def outputs(self, ctx: StepContext) -> List[str]:
        """
        Declare output artifacts this step produces.
        
        Returns:
            List of ArtifactKey canonical names (NOT paths!)
        
        Example:
            return [ArtifactKey.THRESHOLDS_JSON, ArtifactKey.METRICS_CSV]
        """
        ...
    
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
        ...
    
    def run(self, ctx: StepContext) -> StepResult:
        """
        Execute step logic.
        
        Args:
            ctx: Runtime context with artifact_root, config, run_id, etc.
        
        Returns:
            StepResult with artifacts_written, splits_used, metrics, metadata
        """
        ...


__all__ = [
    "StepResult",
    "StepContext",
    "StepSpec",
]

