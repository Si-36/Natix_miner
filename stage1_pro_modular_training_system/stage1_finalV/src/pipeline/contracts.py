"""
🔒️ **Pipeline Contracts** - Split Enforcement (Single Source of Truth)
Implements split contracts (leak-proof design) for the step/asset pipeline.
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
    model_selection: FrozenSet[str] = frozenset({Split.VAL_SELECT})
    policy_fitting: FrozenSet[str] = frozenset({Split.VAL_CALIB})
    final_eval: FrozenSet[str] = frozenset({Split.VAL_TEST})
    
    # Training can use TRAIN + VAL_SELECT
    training: FrozenSet[str] = frozenset({Split.TRAIN, Split.VAL_SELECT})


def assert_allowed(used: FrozenSet[str], allowed: FrozenSet[str], context: str) -> None:
    """
    Assert that only allowed splits were used.
    
    Args:
        used: Splits actually used (as strings like "val_calib")
        allowed: Splits allowed for this phase (as strings like ["val_calib"])
        context: Phase name (for error message)
    
    Raises:
        ValueError: If illegal splits detected
    
    Note:
        Split enum values are strings, so we use FrozenSet[str].
        This validates that used splits are subset of allowed splits.
    """
    # Convert allowed to set of strings for easier comparison
    allowed_set = set(allowed)
    
    # Check for illegal splits
    illegal = set(used) - allowed_set
    if illegal:
        raise ValueError(
            f"[SplitContracts] {context}: illegal splits detected: {sorted(list(illegal))}. "
            f"Allowed: {sorted(list(allowed_set))}. "
            f"This is DATA LEAKAGE - violates leak-proof design!"
        )


__all__ = [
    "Split",
    "SplitPolicy",
    "assert_allowed",
]
