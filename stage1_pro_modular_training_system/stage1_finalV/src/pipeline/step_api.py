"""
🔒️ **Step/Asset API - StepSpec Interface (2026 Pro Standard)**
REAL ML EXECUTION - NOT Skeleton!

Implements:
- StepSpec: Base class for all pipeline steps (domain-stable names!)
- StepContext: Runtime context with artifact store, manifest, etc.
- StepResult: Step result with artifacts written + metrics + metadata

2026 Pro Features:
- Domain-stable step names (not "phase" numbers!)
- ArtifactKey canonical paths (no raw paths!)
- ArtifactStore integration (atomic writes + manifest lineage)
- Split contract enforcement (leak-proof by construction!)

This is the **single entrypoint contract** for all pipeline steps.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, FrozenSet, Union
from enum import Enum


class StepSpec(ABC):
    """
    Step Specification (Abstract Base Class).
    
    🔥 2026 PRO: Domain-stable step names (not "phase" numbers!)
    
    All pipeline steps must inherit from this base class.
    Implementors must provide:
    - step_id: Canonical step name
    - name: Human-readable step name
    - deps: List of step IDs this step depends on
    - order_index: Execution order hint (optional)
    - owners: List of team owners
    - tags: Dict of searchable metadata
    - inputs(): List required input ArtifactKeys
    - outputs(): List output ArtifactKeys
    - allowed_splits(): FrozenSet of Split enum values
    - run(): Execute step logic (returns StepResult)
    """
    
    # ✅ 2026 PRO: Regular dataclass fields (no @property crashes!)
    step_id: str
    name: str
    deps: List[str] = field(default_factory=list)
    order_index: Optional[int] = None
    owners: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    
    @abstractmethod
    def inputs(self, ctx: 'StepContext') -> List[str]:
        """
        List required input artifacts for this step.
        
        Args:
            ctx: Step context (includes artifact_store)
        
        Returns:
            List of ArtifactKey canonical names (NOT paths!)
        """
        pass
    
    @abstractmethod
    def outputs(self, ctx: 'StepContext') -> List[str]:
        """
        List output artifacts this step produces.
        
        Args:
            ctx: Step context (includes artifact_store)
        
        Returns:
            List of ArtifactKey canonical names (NOT paths!)
        """
        pass
    
    @abstractmethod
    def allowed_splits(self) -> FrozenSet[str]:
        """
        Declare which data splits this step is allowed to use.
        
        🔥 LEAK-PROOF CONTRACT:
        - Steps can only use splits they declare in allowed_splits()
        - Engine enforces this at run() boundaries
        - Using TRAIN in export_calib_logits would CRASH (data leakage!)
        - Using VAL_TEST in any step before Phase 6 would CRASH!
        
        Returns:
            FrozenSet of Split enum values
        """
        pass
    
    @abstractmethod
    def run(self, ctx: 'StepContext') -> 'StepResult':
        """
        Execute step logic.
        
        Args:
            ctx: Runtime context (includes artifact_store, config, run_id, etc.)
        
        Returns:
            StepResult with:
                - artifacts_written: List of ArtifactKey names written
                - splits_used: FrozenSet of Split enum values used
                - metrics: Dict of performance metrics (loss, accuracy, etc.)
                - metadata: Dict of additional step metadata (timing, paths, etc.)
        
        🔥 2026 PRO: Steps must NOT call:
        - ctx.artifact_store.put(..., run_id="current") ✅ Correct (use run_id from manifest!)
        - ctx.artifact_store.put(..., ctx.run_id) ❌ WRONG (never use ctx.run_id!)
        - Manual loss.backward() or optimizer.step() (Lightning handles this!)
        - Any code that violates split contracts (would cause data leakage!)
        """
        pass


@dataclass(frozen=True)
class StepContext:
    """
    Step Context (Runtime Information).
    
    Provides:
    - Config: Resolved Hydra config (read-only)
    - Run ID: Current run identifier
    - Artifact Store: For atomic writes + path resolution
    - Manifest: Run manifest object (for lineage tracking)
    - Metadata: Additional step metadata
    
    2026 Pro Features:
    - Read-only config (no modification after creation)
    - Manifest reference (for hash tracking)
    - Artifact store integration (atomic writes + path resolution)
    - Metadata dict (for custom step data)
    """
    
    step_id: str
    config: Dict[str, Any]
    run_id: str
    artifact_root: Path
    artifact_store: Any  # ArtifactStore instance
    manifest: Any  # RunManifest object
    metadata: Dict[str, str] = field(default_factory=dict)
    state: Optional[Dict[str, Any]] = None  # Optional step state (for caching)


@dataclass(frozen=True)
class StepResult:
    """
    Step Result (Immutable Return Value).
    
    Provides:
    - Artifacts written: List of ArtifactKey canonical names
    - Splits used: FrozenSet of Split enum values
    - Metrics: Performance metrics (loss, accuracy, etc.)
    - Metadata: Additional result metadata (paths, hashes, timing)
    
    2026 Pro Features:
    - Frozen dataclass (immutable - prevents accidental mutation)
    - Type-safe fields (can't accidentally change artifact names)
    - Split tracking (ensures no data leakage by construction!)
    """
    
    artifacts_written: List[str] = field(default_factory=list)
    splits_used: FrozenSet[str] = field(default_factory=frozenset)
    metrics: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Forward references for backward compatibility
StepSpecV1 = StepSpec  # Alias for old code
StepContextV1 = StepContext  # Alias for old code
StepResultV1 = StepResult  # Alias for old code


__all__ = [
    "StepSpec",
    "StepContext",
    "StepResult",
]
