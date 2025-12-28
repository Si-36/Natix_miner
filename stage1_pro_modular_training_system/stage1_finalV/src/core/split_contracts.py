"""
🔒️ **Split Contracts** - Leak-Proof Split Usage Rules
Implements TODO 122: Enforce split usage rules (no leakage by construction)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class Split(str, Enum):
    """
    Data splits with strict usage rules.
    
    Leak-Proof Design:
    - TRAIN: Only for training
    - VAL_SELECT: Only for model selection (checkpointing/early stopping)
    - VAL_CALIB: Only for calibration/policy fitting (NEVER for selection!)
    - VAL_TEST: Only for final evaluation (NEVER for training/calibration!)
    
    Violation Example (BLOCKED):
    - Using VAL_CALIB for hyperparameter tuning (data leakage!)
    - Using VAL_TEST for checkpointing (data leakage!)
    - Using TRAIN for evaluation (data leakage!)
    """
    TRAIN = "train"
    VAL_SELECT = "val_select"   # model selection ONLY
    VAL_CALIB = "val_calib"     # policy fitting ONLY (never for selection!)
    VAL_TEST = "val_test"       # final eval ONLY


@dataclass(frozen=True)
class SplitPolicy:
    """
    Declares allowed splits for each phase type.
    
    This is enforced at runtime to prevent accidental data leakage.
    """
    model_selection: FrozenSet[Split] = frozenset({Split.VAL_SELECT})
    policy_fitting: FrozenSet[Split] = frozenset({Split.VAL_CALIB})
    final_eval: FrozenSet[Split] = frozenset({Split.VAL_TEST})
    
    # Training can use TRAIN + VAL_SELECT
    training: FrozenSet[Split] = frozenset({Split.TRAIN, Split.VAL_SELECT})


def assert_allowed(used: FrozenSet[Split], allowed: FrozenSet[Split], context: str) -> None:
    """
    Assert that only allowed splits were used.
    
    Raises ValueError if illegal splits are found.
    
    Args:
        used: Splits actually used in this phase
        allowed: Splits allowed for this phase
        context: Phase name (for error message)
    
    Raises:
        ValueError: If illegal splits detected
    """
    illegal = set(used) - set(allowed)
    if illegal:
        raise ValueError(
            f"[SplitContracts] {context}: illegal splits detected: {sorted([s.value for s in illegal])}. "
            f"Allowed: {sorted([s.value for s in allowed])}. "
            f"This is DATA LEAKAGE - violates leak-proof design!"
        )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "Split",
    "SplitPolicy",
    "assert_allowed",
]

