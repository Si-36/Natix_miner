"""
🔒️ **Step Registry** - Single Source of Truth for All Steps
Implements: Step names, deps, order_index - no hardcoded dependencies
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StepRegistry:
    """
    Single source of truth for all step specifications.
    
    Defines:
    - Step names (canonical domain names)
    - Dependencies (no hardcoded _deps in engine)
    - Order index (explicit ordering hint)
    - Owners (team responsible for step)
    - Tags (searchable metadata)
    
    This replaces hardcoded _deps dictionaries and prevents
    "forgotten dependency" bugs.
    """
    
    # Step catalog (step_id: (name, deps, order_index, owners))
    steps: Dict[int, tuple] = {}
    
    def __post_init__(self):
        """Initialize step catalog from known steps"""
        # Phase 1 steps
        self.steps[1] = (
            "train_baseline_head",
            frozenset([]),  # No deps
            0,  # Order index
            ["ml-team"],  # Owners
        )
        
        # Phase 2 steps
        self.steps[2] = (
            "sweep_thresholds",
            frozenset(["train_baseline_head"]),  # Depends on trained model
            1,  # Order: after baseline
            ["ml-team"],
        )
        
        # Phase 3 steps
        self.steps[3] = (
            "train_gate",
            frozenset(["train_baseline_head"]),  # Depends on trained model
            2,  # Order: after sweep
            ["ml-team"],
        )
        
        # Phase 6 steps
        self.steps[6] = (
            "export_bundle",
            frozenset([
                "train_baseline_head",  # Needs checkpoint
                "sweep_thresholds",  # Needs thresholds
            ]),
            3,  # Order: after gate
            ["ml-team"],
        )
        
        # Validate: no duplicate step IDs
        if len(self.steps) != len(set(self.steps.keys())):
            raise ValueError(f"Duplicate step IDs detected: {self.steps.keys()}")
    
    def get_step_info(self, step_id: int) -> Dict:
        """
        Get step information by ID.
        
        Args:
            step_id: Step ID (1-6)
        
        Returns:
            Dict with name, deps, order_index, owners
        """
        if step_id not in self.steps:
            raise ValueError(f"Unknown step ID: {step_id}")
        
        name, deps, order_index, owners = self.steps[step_id]
        return {
            "name": name,
            "deps": list(deps),
            "order_index": order_index,
            "owners": list(owners),
        }
    
    def get_dependencies(self, step_id: int) -> List[str]:
        """
        Get dependency names for a step.
        
        Args:
            step_id: Step ID
        
        Returns:
            List of step names this step depends on
        """
        return self.get_step_info(step_id)["deps"]


__all__ = [
    "StepRegistry",
]

