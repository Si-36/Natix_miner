"""
🔒️ **Phase Specification** - Phase I/O Declarations
Implements TODO 124: PhaseSpec ABC with inputs/outputs/allowed splits
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Iterable, List

from .split_contracts import Split


@dataclass(frozen=True)
class PhaseResult:
    """
    Result of a phase execution.
    
    Attributes:
        outputs: List of output file paths produced
        splits_used: Set of splits actually used
    """
    outputs: List[Path]
    splits_used: FrozenSet[Split]


class PhaseSpec(ABC):
    """
    Abstract base class for all phase specifications.
    
    Each phase must declare:
    - inputs(): Required input artifacts (from previous phases)
    - outputs(): Output artifacts this phase produces
    - allowed_splits(): Which splits this phase is allowed to use
    - run(): Execute the phase logic
    
    This design enforces "hidden contracts" so phases can't accidentally
    use wrong splits or access missing artifacts.
    """
    phase_id: int
    name: str
    
    @abstractmethod
    def inputs(self, artifacts) -> List[Path]:
        """
        Declare required input artifacts for this phase.
        
        Args:
            artifacts: ArtifactSchema with paths to previous phases' outputs
        
        Returns:
            List of required file paths
        
        Example:
            return [
                artifacts.phase1_checkpoint,  # Phase 2 needs trained model
                artifacts.phase1_val_select_logits_pt,  # Phase 2 needs logits
            ]
        """
        pass
    
    @abstractmethod
    def outputs(self, artifacts) -> List[Path]:
        """
        Declare output artifacts this phase produces.
        
        Args:
            artifacts: ArtifactSchema for path resolution
        
        Returns:
            List of output file paths
        
        Example:
            return [
                artifacts.phase2_thresholds_json,  # Phase 2 produces thresholds
                artifacts.phase2_policy_metrics_csv,  # Phase 2 produces metrics
            ]
        """
        pass
    
    @abstractmethod
    def allowed_splits(self) -> FrozenSet[Split]:
        """
        Declare which data splits this phase is allowed to use.
        
        Returns:
            FrozenSet of Split enum values
        
        Example (Phase 1 - Training):
            return frozenset({Split.TRAIN, Split.VAL_SELECT})
        
        Example (Phase 2 - Calibration):
            return frozenset({Split.VAL_CALIB})  # NEVER train or select!
        """
        pass
    
    @abstractmethod
    def run(self, cfg, artifacts) -> PhaseResult:
        """
        Execute the phase logic.
        
        Args:
            cfg: Hydra config (validated)
            artifacts: ArtifactSchema for path resolution
        
        Returns:
            PhaseResult with outputs and splits used
        
        This method should:
        1. Load inputs from ArtifactSchema
        2. Execute phase-specific logic
        3. Save outputs to ArtifactSchema paths
        4. Return which splits were used
        5. NOT use unauthorized splits (enforced by SplitPolicy)
        """
        pass


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "PhaseSpec",
    "PhaseResult",
]

