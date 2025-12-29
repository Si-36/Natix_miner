"""
 🔥 **Split Contracts (2025 Best Practices)**
Enforce zero data leakage with Pydantic validation
"""

from enum import Enum
from pydantic import BaseModel, model_validator, field_validator, validator
from typing import Set, Optional


class Split(Enum):
    """Data split enumeration with strict rules"""

    TRAIN = "train"  # Training data ONLY
    VAL_SELECT = "val_select"  # Model selection ONLY - NEVER for calibration
    VAL_CALIB = "val_calib"  # Policy fitting ONLY - NEVER for selection
    VAL_TEST = "val_test"  # Final evaluation ONLY


class SplitRules(BaseModel):
    """
    Hard rules for split usage to prevent leakage

    CRITICAL: These rules MUST be enforced in code, not just documentation!
    """

    # Allowed splits per phase
    MODEL_SELECTION_SPLITS: Set[Split] = {Split.VAL_SELECT}
    POLICY_FITTING_SPLITS: Set[Split] = {Split.VAL_CALIB}
    FINAL_EVAL_SPLITS: Set[Split] = {Split.VAL_TEST}
    TRAINING_SPLITS: Set[Split] = {Split.TRAIN, Split.VAL_SELECT}

    # Prohibited combinations
    @field_validator("splits")
    @classmethod
    def validate_no_leakage(cls, v):
        """Validate NO data leakage"""
        if not isinstance(v, set):
            raise ValueError("Splits must be a set")

        # CRITICAL: Cannot use VAL_CALIB for model selection
        if Split.VAL_CALIB in v and Split.VAL_SELECT in v:
            raise ValueError(
                "DATA LEAKAGE! Cannot use val_calib and val_select together. "
                "val_calib is for policy fitting ONLY, not model selection."
            )

        # CRITICAL: Cannot use VAL_TEST for training
        if Split.VAL_TEST in v and len(v) > 1:
            raise ValueError(
                "DATA LEAKAGE! Cannot use val_test for training. "
                "val_test is for final evaluation ONLY."
            )

        return v


class Phase1SplitRules(SplitRules):
    """Phase 1: Baseline training rules"""

    allowed: Set[Split] = {Split.TRAIN, Split.VAL_SELECT}
    prohibited: Set[Split] = {Split.VAL_CALIB, Split.VAL_TEST}

    @validator("splits")
    @classmethod
    def validate_phase1(cls, v):
        """Validate Phase 1 split usage"""
        if not isinstance(v, set):
            raise ValueError("Splits must be a set")

        # Must NOT use val_calib or val_test
        prohibited = cls.prohibited
        if any(s in v for s in prohibited):
            raise ValueError(
                f"DATA LEAKAGE in Phase 1! Cannot use {prohibited}. "
                f"Phase 1 only allows: {cls.allowed}"
            )

        return v


class Phase2SplitRules(SplitRules):
    """Phase 2: Threshold sweep rules"""

    allowed: Set[Split] = {Split.VAL_CALIB}
    prohibited: Set[Split] = {Split.TRAIN, Split.VAL_TEST, Split.VAL_SELECT}

    @validator("splits")
    @classmethod
    def validate_phase2(cls, v):
        """Validate Phase 2 split usage"""
        if not isinstance(v, set):
            raise ValueError("Splits must be a set")

        # Must use ONLY val_calib
        if v != {Split.VAL_CALIB}:
            raise ValueError(f"DATA LEAKAGE in Phase 2! Must use ONLY val_calib. Got: {v}")

        return v


class Phase3SplitRules(SplitRules):
    """Phase 3: Gate training rules"""

    allowed: Set[Split] = {Split.TRAIN, Split.VAL_CALIB}
    prohibited: Set[Split] = {Split.VAL_TEST, Split.VAL_SELECT}

    @validator("splits")
    @classmethod
    def validate_phase3(cls, v):
        """Validate Phase 3 split usage"""
        if not isinstance(v, set):
            raise ValueError("Splits must be a set")

        # Must use train + val_calib
        if v != {Split.TRAIN, Split.VAL_CALIB}:
            raise ValueError(f"DATA LEAKAGE in Phase 3! Must use ONLY train + val_calib. Got: {v}")

        return v


def validate_splits_for_phase(phase: int, splits: Set[Split]) -> bool:
    """
    Validate splits for a specific phase

    Args:
        phase: Phase number (1-6)
        splits: Set of splits being used

    Returns:
        True if valid, raises ValueError if invalid
    """
    if phase == 1:
        Phase1SplitRules(splits=splits)
    elif phase == 2:
        Phase2SplitRules(splits=splits)
    elif phase == 3:
        Phase3SplitRules(splits=splits)
    else:
        raise ValueError(f"Phase {phase} not yet implemented")

    return True


def get_allowed_splits(phase: int) -> Set[Split]:
    """Get allowed splits for a phase"""
    if phase == 1:
        return Phase1SplitRules.allowed
    elif phase == 2:
        return Phase2SplitRules.allowed
    elif phase == 3:
        return Phase3SplitRules.allowed
    else:
        raise ValueError(f"Phase {phase} not yet implemented")


# Export for easy imports
__all__ = [
    "Split",
    "SplitRules",
    "Phase1SplitRules",
    "Phase2SplitRules",
    "Phase3SplitRules",
    "validate_splits_for_phase",
    "get_allowed_splits",
]
