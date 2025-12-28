"""
Unit Tests for Split Contracts

Tests the CRITICAL leakage prevention contracts:
1. Model selection → ONLY val_select
2. Policy fitting → ONLY val_calib
3. Final evaluation → ONLY val_test

Latest 2025-2026 practices:
- Python 3.14+ with pytest
- Clear test names (test_<what>_<expected>)
- Explicit assertions
"""

import pytest
from src.contracts.split_contracts import (
    Split,
    SplitPolicy,
    SplitValidator,
    LeakageViolationError,
    EnforceSplitContract,
)


class TestSplitPolicy:
    """Test SplitPolicy validation methods"""

    def test_model_selection_valid(self):
        """Model selection should accept ONLY val_select"""
        # This should pass without error
        assert SplitPolicy.validate_model_selection({Split.VAL_SELECT})

    def test_model_selection_invalid_val_calib(self):
        """Model selection should REJECT val_calib (policy fitting data)"""
        with pytest.raises(LeakageViolationError) as exc_info:
            SplitPolicy.validate_model_selection({Split.VAL_CALIB})

        # Check error message mentions leakage
        assert "LEAKAGE VIOLATION" in str(exc_info.value)
        assert "val_calib" in str(exc_info.value).lower()

    def test_model_selection_invalid_val_test(self):
        """Model selection should REJECT val_test (test set)"""
        with pytest.raises(LeakageViolationError) as exc_info:
            SplitPolicy.validate_model_selection({Split.VAL_TEST})

        assert "LEAKAGE VIOLATION" in str(exc_info.value)

    def test_model_selection_invalid_train(self):
        """Model selection should REJECT train split"""
        with pytest.raises(LeakageViolationError) as exc_info:
            SplitPolicy.validate_model_selection({Split.TRAIN})

        assert "LEAKAGE VIOLATION" in str(exc_info.value)

    def test_policy_fitting_valid(self):
        """Policy fitting should accept ONLY val_calib"""
        assert SplitPolicy.validate_policy_fitting({Split.VAL_CALIB})

    def test_policy_fitting_invalid_val_select(self):
        """Policy fitting should REJECT val_select (model selection data)"""
        with pytest.raises(LeakageViolationError) as exc_info:
            SplitPolicy.validate_policy_fitting({Split.VAL_SELECT})

        assert "LEAKAGE VIOLATION" in str(exc_info.value)
        assert "val_calib" in str(exc_info.value).lower()

    def test_policy_fitting_invalid_val_test(self):
        """Policy fitting should REJECT val_test (test set)"""
        with pytest.raises(LeakageViolationError) as exc_info:
            SplitPolicy.validate_policy_fitting({Split.VAL_TEST})

        assert "LEAKAGE VIOLATION" in str(exc_info.value)

    def test_policy_fitting_invalid_multiple_splits(self):
        """Policy fitting should REJECT multiple splits"""
        with pytest.raises(LeakageViolationError) as exc_info:
            SplitPolicy.validate_policy_fitting({Split.VAL_CALIB, Split.VAL_SELECT})

        assert "LEAKAGE VIOLATION" in str(exc_info.value)

    def test_final_eval_valid(self):
        """Final evaluation should accept ONLY val_test"""
        assert SplitPolicy.validate_final_eval({Split.VAL_TEST})

    def test_final_eval_invalid_val_select(self):
        """Final evaluation should REJECT val_select"""
        with pytest.raises(LeakageViolationError) as exc_info:
            SplitPolicy.validate_final_eval({Split.VAL_SELECT})

        assert "LEAKAGE VIOLATION" in str(exc_info.value)

    def test_final_eval_invalid_val_calib(self):
        """Final evaluation should REJECT val_calib"""
        with pytest.raises(LeakageViolationError) as exc_info:
            SplitPolicy.validate_final_eval({Split.VAL_CALIB})

        assert "LEAKAGE VIOLATION" in str(exc_info.value)

    def test_training_valid_all_allowed(self):
        """Training should accept train + val_select + val_calib"""
        assert SplitPolicy.validate_training(
            {Split.TRAIN, Split.VAL_SELECT, Split.VAL_CALIB}
        )

    def test_training_valid_train_only(self):
        """Training should accept train only"""
        assert SplitPolicy.validate_training({Split.TRAIN})

    def test_training_valid_train_val_select(self):
        """Training should accept train + val_select (for early stopping)"""
        assert SplitPolicy.validate_training({Split.TRAIN, Split.VAL_SELECT})

    def test_training_invalid_val_test(self):
        """Training should REJECT val_test"""
        with pytest.raises(LeakageViolationError) as exc_info:
            SplitPolicy.validate_training({Split.TRAIN, Split.VAL_TEST})

        assert "LEAKAGE VIOLATION" in str(exc_info.value)
        assert "val_test" in str(exc_info.value).lower()


class TestSplitValidator:
    """Test SplitValidator helper class"""

    def test_check_training_valid(self):
        """SplitValidator.check_training should work correctly"""
        validator = SplitValidator()
        assert validator.check_training({Split.TRAIN, Split.VAL_SELECT})

    def test_check_training_invalid(self):
        """SplitValidator.check_training should catch violations"""
        validator = SplitValidator()
        with pytest.raises(LeakageViolationError):
            validator.check_training({Split.VAL_TEST})

    def test_check_model_selection_valid(self):
        """SplitValidator.check_model_selection should work correctly"""
        validator = SplitValidator()
        assert validator.check_model_selection({Split.VAL_SELECT})

    def test_check_model_selection_invalid(self):
        """SplitValidator.check_model_selection should catch violations"""
        validator = SplitValidator()
        with pytest.raises(LeakageViolationError):
            validator.check_model_selection({Split.VAL_CALIB})

    def test_check_policy_fitting_valid(self):
        """SplitValidator.check_policy_fitting should work correctly"""
        validator = SplitValidator()
        assert validator.check_policy_fitting({Split.VAL_CALIB})

    def test_check_policy_fitting_invalid(self):
        """SplitValidator.check_policy_fitting should catch violations"""
        validator = SplitValidator()
        with pytest.raises(LeakageViolationError):
            validator.check_policy_fitting({Split.VAL_SELECT})

    def test_check_final_eval_valid(self):
        """SplitValidator.check_final_eval should work correctly"""
        validator = SplitValidator()
        assert validator.check_final_eval({Split.VAL_TEST})

    def test_check_final_eval_invalid(self):
        """SplitValidator.check_final_eval should catch violations"""
        validator = SplitValidator()
        with pytest.raises(LeakageViolationError):
            validator.check_final_eval({Split.VAL_CALIB})

    def test_get_usage_description(self):
        """SplitValidator should return correct descriptions"""
        validator = SplitValidator()

        desc = validator.get_usage_description(Split.TRAIN)
        assert "training" in desc.lower()

        desc = validator.get_usage_description(Split.VAL_SELECT)
        assert "model selection" in desc.lower()

        desc = validator.get_usage_description(Split.VAL_CALIB)
        assert "policy fitting" in desc.lower()

        desc = validator.get_usage_description(Split.VAL_TEST)
        assert "final evaluation" in desc.lower()


class TestEnforceSplitContract:
    """Test EnforceSplitContract context manager"""

    def test_context_manager_training(self):
        """EnforceSplitContract should work as context manager for training"""
        with EnforceSplitContract(mode="training") as enforcer:
            # Valid training usage
            assert enforcer.validate({Split.TRAIN, Split.VAL_SELECT})

            # Invalid training usage
            with pytest.raises(LeakageViolationError):
                enforcer.validate({Split.VAL_TEST})

    def test_context_manager_policy_fitting(self):
        """EnforceSplitContract should work as context manager for policy fitting"""
        with EnforceSplitContract(mode="policy_fitting") as enforcer:
            # Valid policy fitting usage
            assert enforcer.validate({Split.VAL_CALIB})

            # Invalid policy fitting usage
            with pytest.raises(LeakageViolationError):
                enforcer.validate({Split.VAL_SELECT})

    def test_context_manager_final_eval(self):
        """EnforceSplitContract should work as context manager for final eval"""
        with EnforceSplitContract(mode="final_eval") as enforcer:
            # Valid final eval usage
            assert enforcer.validate({Split.VAL_TEST})

            # Invalid final eval usage
            with pytest.raises(LeakageViolationError):
                enforcer.validate({Split.VAL_CALIB})

    def test_context_manager_model_selection(self):
        """EnforceSplitContract should work as context manager for model selection"""
        with EnforceSplitContract(mode="model_selection") as enforcer:
            # Valid model selection usage
            assert enforcer.validate({Split.VAL_SELECT})

            # Invalid model selection usage
            with pytest.raises(LeakageViolationError):
                enforcer.validate({Split.VAL_CALIB})

    def test_context_manager_invalid_mode(self):
        """EnforceSplitContract should reject invalid modes"""
        with EnforceSplitContract(mode="invalid_mode") as enforcer:
            with pytest.raises(ValueError) as exc_info:
                enforcer.validate({Split.TRAIN})

            assert "Unknown mode" in str(exc_info.value)


class TestSplitEnum:
    """Test Split enum"""

    def test_split_values(self):
        """Split enum should have correct values"""
        assert Split.TRAIN.value == "train"
        assert Split.VAL_SELECT.value == "val_select"
        assert Split.VAL_CALIB.value == "val_calib"
        assert Split.VAL_TEST.value == "val_test"

    def test_split_membership(self):
        """Split enum should contain all expected members"""
        expected_members = {"TRAIN", "VAL_SELECT", "VAL_CALIB", "VAL_TEST"}
        actual_members = {member.name for member in Split}
        assert actual_members == expected_members
