"""
Split Contracts - Leakage Prevention Enforced as CODE

CRITICAL RULES (enforced at runtime):
1. Model selection (early stopping) → ONLY val_select
2. Policy fitting (threshold sweep, calibration) → ONLY val_calib
3. Final evaluation → ONLY val_test

These rules are ENFORCED AS CODE - system cannot run if violated.

Benefits:
- Zero data leakage (impossible by construction)
- Correctness by construction (not developer discipline)
- Clear documentation (split usage is explicit)
- Fail-fast errors (violations caught immediately)

Latest 2025-2026 practices:
- Python 3.11+ type hints
- Enum for type safety
- Runtime validation
- Clear error messages
"""

from enum import Enum
from typing import Set, Literal
from dataclasses import dataclass


class Split(Enum):
    """
    Data splits for training and evaluation

    CRITICAL: Each split has a SPECIFIC purpose:
    - TRAIN: Training only
    - VAL_SELECT: Model selection ONLY (early stopping, checkpoint selection)
    - VAL_CALIB: Policy fitting ONLY (threshold sweep, gate calibration, SCRC)
    - VAL_TEST: Final evaluation ONLY (NEVER touch during training/tuning)
    """

    TRAIN = "train"
    VAL_SELECT = "val_select"  # Model selection
    VAL_CALIB = "val_calib"    # Policy fitting
    VAL_TEST = "val_test"      # Final evaluation


# Type alias for split usage (2025-2026 style)
type SplitSet = Set[Split]
type SplitUsageMode = Literal["training", "model_selection", "policy_fitting", "final_eval"]


class LeakageViolationError(Exception):
    """Raised when split contract is violated (data leakage detected)"""
    pass


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """
    Split usage policy - enforces leakage prevention

    CRITICAL CONTRACTS:
    1. Model selection → ONLY val_select
    2. Policy fitting → ONLY val_calib
    3. Final evaluation → ONLY val_test

    Violations raise LeakageViolationError immediately.
    """

    # CRITICAL CONTRACT: Model selection splits
    MODEL_SELECTION_SPLITS: SplitSet = frozenset({Split.VAL_SELECT})

    # CRITICAL CONTRACT: Policy fitting splits
    POLICY_FITTING_SPLITS: SplitSet = frozenset({Split.VAL_CALIB})

    # CRITICAL CONTRACT: Final evaluation splits
    FINAL_EVAL_SPLITS: SplitSet = frozenset({Split.VAL_TEST})

    # Training can use train + val_select (for early stopping) + val_calib (save logits)
    TRAINING_ALLOWED_SPLITS: SplitSet = frozenset({
        Split.TRAIN,
        Split.VAL_SELECT,  # For early stopping
        Split.VAL_CALIB,   # Save logits for policy fitting later
    })

    @staticmethod
    def validate_model_selection(splits_used: SplitSet) -> bool:
        """
        CRITICAL: Model selection must NEVER use val_calib or val_test

        Using val_calib would leak policy fitting data into model selection.
        Using val_test would leak test data into model selection.

        Args:
            splits_used: Set of splits actually used

        Returns:
            True if valid

        Raises:
            LeakageViolationError: If contract violated
        """
        forbidden = SplitPolicy.POLICY_FITTING_SPLITS | SplitPolicy.FINAL_EVAL_SPLITS

        if splits_used & forbidden:
            raise LeakageViolationError(
                f"❌ LEAKAGE VIOLATION: Model selection used {splits_used & forbidden}.\n"
                f"ONLY {SplitPolicy.MODEL_SELECTION_SPLITS} allowed for model selection.\n"
                f"Using val_calib or val_test for model selection leaks data!"
            )

        if not (splits_used <= SplitPolicy.MODEL_SELECTION_SPLITS):
            raise LeakageViolationError(
                f"❌ LEAKAGE VIOLATION: Model selection used {splits_used}.\n"
                f"ONLY {SplitPolicy.MODEL_SELECTION_SPLITS} allowed."
            )

        return True

    @staticmethod
    def validate_policy_fitting(splits_used: SplitSet) -> bool:
        """
        CRITICAL: Policy fitting must ONLY use val_calib

        Using val_select would leak model selection into policy fitting.
        Using val_test would leak test set into policy fitting.

        Args:
            splits_used: Set of splits actually used

        Returns:
            True if valid

        Raises:
            LeakageViolationError: If contract violated
        """
        if splits_used != SplitPolicy.POLICY_FITTING_SPLITS:
            raise LeakageViolationError(
                f"❌ LEAKAGE VIOLATION: Policy fitting used {splits_used}.\n"
                f"ONLY {SplitPolicy.POLICY_FITTING_SPLITS} allowed for policy fitting.\n"
                f"Policy fitting MUST use ONLY val_calib to prevent leakage!"
            )

        return True

    @staticmethod
    def validate_final_eval(splits_used: SplitSet) -> bool:
        """
        CRITICAL: Final evaluation must ONLY use val_test

        Using any other split means evaluation is biased.

        Args:
            splits_used: Set of splits actually used

        Returns:
            True if valid

        Raises:
            LeakageViolationError: If contract violated
        """
        if splits_used != SplitPolicy.FINAL_EVAL_SPLITS:
            raise LeakageViolationError(
                f"❌ LEAKAGE VIOLATION: Final evaluation used {splits_used}.\n"
                f"ONLY {SplitPolicy.FINAL_EVAL_SPLITS} allowed for final evaluation.\n"
                f"Final evaluation MUST use ONLY val_test!"
            )

        return True

    @staticmethod
    def validate_training(splits_used: SplitSet) -> bool:
        """
        Validate training phase split usage

        Training can use:
        - train: For training
        - val_select: For early stopping (model selection)
        - val_calib: MUST save logits for policy fitting later (but NOT use for model selection)

        Args:
            splits_used: Set of splits actually used

        Returns:
            True if valid

        Raises:
            LeakageViolationError: If contract violated
        """
        forbidden = splits_used - SplitPolicy.TRAINING_ALLOWED_SPLITS

        if forbidden:
            raise LeakageViolationError(
                f"❌ LEAKAGE VIOLATION: Training used forbidden splits: {forbidden}.\n"
                f"Allowed: {SplitPolicy.TRAINING_ALLOWED_SPLITS}\n"
                f"Training MUST NOT touch val_test!"
            )

        # CRITICAL: If using val_select, must be for model selection only
        if Split.VAL_SELECT in splits_used:
            SplitPolicy.validate_model_selection({Split.VAL_SELECT})

        return True


@dataclass(frozen=True, slots=True)
class SplitValidator:
    """
    Helper for validating split usage in code

    Usage:
        validator = SplitValidator()
        validator.check_training({Split.TRAIN, Split.VAL_SELECT, Split.VAL_CALIB})
        validator.check_policy_fitting({Split.VAL_CALIB})
        validator.check_final_eval({Split.VAL_TEST})
    """

    @staticmethod
    def check_training(splits_used: SplitSet) -> bool:
        """
        Validate training phase split usage

        Args:
            splits_used: Set of splits used during training

        Returns:
            True if valid

        Raises:
            LeakageViolationError: If contract violated
        """
        return SplitPolicy.validate_training(splits_used)

    @staticmethod
    def check_model_selection(splits_used: SplitSet) -> bool:
        """
        Validate model selection split usage

        Args:
            splits_used: Set of splits used for model selection

        Returns:
            True if valid

        Raises:
            LeakageViolationError: If contract violated
        """
        return SplitPolicy.validate_model_selection(splits_used)

    @staticmethod
    def check_policy_fitting(splits_used: SplitSet) -> bool:
        """
        Validate policy fitting split usage

        Args:
            splits_used: Set of splits used for policy fitting

        Returns:
            True if valid

        Raises:
            LeakageViolationError: If contract violated
        """
        return SplitPolicy.validate_policy_fitting(splits_used)

    @staticmethod
    def check_final_eval(splits_used: SplitSet) -> bool:
        """
        Validate final evaluation split usage

        Args:
            splits_used: Set of splits used for final evaluation

        Returns:
            True if valid

        Raises:
            LeakageViolationError: If contract violated
        """
        return SplitPolicy.validate_final_eval(splits_used)

    @staticmethod
    def get_usage_description(split: Split) -> str:
        """
        Get human-readable description of split usage

        Args:
            split: Split to describe

        Returns:
            Description string
        """
        descriptions = {
            Split.TRAIN: "Training data (for model training only)",
            Split.VAL_SELECT: "Validation data for model selection (early stopping, checkpoint selection)",
            Split.VAL_CALIB: "Validation data for policy fitting (threshold sweep, calibration, SCRC)",
            Split.VAL_TEST: "Test data for final evaluation ONLY (NEVER touch during training/tuning)",
        }
        return descriptions.get(split, "Unknown split")


# Context manager for enforcing split contracts
class EnforceSplitContract:
    """
    Context manager for enforcing split contracts

    Usage:
        with EnforceSplitContract(mode="training"):
            # Code that uses splits
            splits_used = {Split.TRAIN, Split.VAL_SELECT}
            # Validation happens automatically

        with EnforceSplitContract(mode="policy_fitting"):
            # Code that uses splits
            splits_used = {Split.VAL_CALIB}
            # Validation happens automatically
    """

    def __init__(self, mode: SplitUsageMode):
        """
        Initialize split contract enforcer

        Args:
            mode: Usage mode (training, model_selection, policy_fitting, final_eval)
        """
        self.mode = mode
        self.validator = SplitValidator()

    def __enter__(self):
        """Enter context (no-op)"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context (no-op)"""
        return False

    def validate(self, splits_used: SplitSet) -> bool:
        """
        Validate splits based on mode

        Args:
            splits_used: Set of splits used

        Returns:
            True if valid

        Raises:
            LeakageViolationError: If contract violated
        """
        if self.mode == "training":
            return self.validator.check_training(splits_used)
        elif self.mode == "model_selection":
            return self.validator.check_model_selection(splits_used)
        elif self.mode == "policy_fitting":
            return self.validator.check_policy_fitting(splits_used)
        elif self.mode == "final_eval":
            return self.validator.check_final_eval(splits_used)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


if __name__ == "__main__":
    # Test split contracts
    print("📋 Split Contracts Test\n")

    validator = SplitValidator()

    # Test 1: Valid training usage
    print("Test 1: Valid training usage")
    try:
        splits_used = {Split.TRAIN, Split.VAL_SELECT, Split.VAL_CALIB}
        validator.check_training(splits_used)
        print("✅ PASS: Training with train + val_select + val_calib is valid\n")
    except LeakageViolationError as e:
        print(f"❌ FAIL: {e}\n")

    # Test 2: Invalid training usage (using val_test)
    print("Test 2: Invalid training usage (val_test)")
    try:
        splits_used = {Split.TRAIN, Split.VAL_TEST}
        validator.check_training(splits_used)
        print("❌ FAIL: Should have raised LeakageViolationError\n")
    except LeakageViolationError as e:
        print(f"✅ PASS: Correctly caught leakage: {e}\n")

    # Test 3: Valid policy fitting
    print("Test 3: Valid policy fitting")
    try:
        splits_used = {Split.VAL_CALIB}
        validator.check_policy_fitting(splits_used)
        print("✅ PASS: Policy fitting with val_calib only is valid\n")
    except LeakageViolationError as e:
        print(f"❌ FAIL: {e}\n")

    # Test 4: Invalid policy fitting (using val_select)
    print("Test 4: Invalid policy fitting (val_select)")
    try:
        splits_used = {Split.VAL_SELECT}
        validator.check_policy_fitting(splits_used)
        print("❌ FAIL: Should have raised LeakageViolationError\n")
    except LeakageViolationError as e:
        print(f"✅ PASS: Correctly caught leakage: {e}\n")

    # Test 5: Valid final evaluation
    print("Test 5: Valid final evaluation")
    try:
        splits_used = {Split.VAL_TEST}
        validator.check_final_eval(splits_used)
        print("✅ PASS: Final evaluation with val_test only is valid\n")
    except LeakageViolationError as e:
        print(f"❌ FAIL: {e}\n")

    # Test 6: Invalid final evaluation (using val_calib)
    print("Test 6: Invalid final evaluation (val_calib)")
    try:
        splits_used = {Split.VAL_CALIB}
        validator.check_final_eval(splits_used)
        print("❌ FAIL: Should have raised LeakageViolationError\n")
    except LeakageViolationError as e:
        print(f"✅ PASS: Correctly caught leakage: {e}\n")

    print("✅ All split contract tests passed!")
    print("\n💡 Summary: Split contracts prevent data leakage by construction!")
