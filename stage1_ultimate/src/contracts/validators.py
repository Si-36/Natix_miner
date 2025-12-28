"""
Artifact Validators - Fail-Fast Checking for ALL Artifacts

CRITICAL RULES (enforced at runtime):
1. All artifacts must exist before use
2. All artifacts must have correct format/structure
3. Fail IMMEDIATELY with clear error messages
4. NEVER proceed with corrupted/missing data

Benefits:
- Catch errors early (fail-fast)
- Clear error messages (no cryptic torch errors)
- Prevent silent failures (no NaN/inf in logits)
- Type-safe validation (no "forgot to check X")

Latest 2025-2026 practices:
- Python 3.11+ type hints
- Pydantic v2 patterns
- Pathlib for type-safe paths
- Rich error messages with context
"""

from pathlib import Path
from typing import Any, Dict, List, Set, Optional
from dataclasses import dataclass
import json

# These imports will be available at runtime
# import torch
# import numpy as np


class ArtifactValidationError(Exception):
    """Raised when artifact validation fails"""

    pass


class CheckpointValidationError(ArtifactValidationError):
    """Raised when checkpoint validation fails"""

    pass


class LogitsValidationError(ArtifactValidationError):
    """Raised when logits validation fails"""

    pass


class LabelsValidationError(ArtifactValidationError):
    """Raised when labels validation fails"""

    pass


class PolicyValidationError(ArtifactValidationError):
    """Raised when policy JSON validation fails"""

    pass


class BundleValidationError(ArtifactValidationError):
    """Raised when bundle validation fails"""

    pass


@dataclass(frozen=True, slots=True)
class CheckpointValidator:
    """
    Checkpoint file validator

    Validates:
    - File exists
    - File is loadable by torch
    - Required keys are present
    - No NaN/inf in weights
    """

    @staticmethod
    def validate(
        path: Path,
        required_keys: Optional[Set[str]] = None,
        check_nan: bool = True,
    ) -> bool:
        """
        Validate checkpoint file

        Args:
            path: Path to checkpoint file
            required_keys: Set of required keys in checkpoint dict
            check_nan: Whether to check for NaN/inf in weights

        Returns:
            True if valid

        Raises:
            CheckpointValidationError: If validation fails
        """
        import torch

        # Check existence
        if not path.exists():
            raise CheckpointValidationError(
                f"❌ Checkpoint not found: {path}\nExpected checkpoint file does not exist."
            )

        # Check loadability
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as e:
            raise CheckpointValidationError(f"❌ Failed to load checkpoint: {path}\nError: {e}")

        # Check type
        if not isinstance(checkpoint, dict):
            raise CheckpointValidationError(
                f"❌ Checkpoint is not a dict: {path}\nGot type: {type(checkpoint)}"
            )

        # Check required keys
        if required_keys is not None:
            missing_keys = required_keys - set(checkpoint.keys())
            if missing_keys:
                raise CheckpointValidationError(
                    f"❌ Checkpoint missing required keys: {path}\n"
                    f"Missing: {missing_keys}\n"
                    f"Available: {set(checkpoint.keys())}"
                )

        # Check for NaN/inf in weights
        if check_nan and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            for key, tensor in state_dict.items():
                if not isinstance(tensor, torch.Tensor):
                    continue

                if torch.isnan(tensor).any():
                    raise CheckpointValidationError(
                        f"❌ Checkpoint contains NaN values: {path}\n"
                        f"Parameter: {key}\n"
                        f"This indicates training failure or corruption."
                    )

                if torch.isinf(tensor).any():
                    raise CheckpointValidationError(
                        f"❌ Checkpoint contains inf values: {path}\n"
                        f"Parameter: {key}\n"
                        f"This indicates training instability or corruption."
                    )

        return True


@dataclass(frozen=True, slots=True)
class LogitsValidator:
    """
    Logits tensor validator

    Validates:
    - File exists
    - Tensor is loadable
    - Shape matches expected (N, num_classes)
    - Range is reasonable (no extreme values)
    - No NaN/inf
    """

    @staticmethod
    def validate(
        path: Path,
        expected_shape: Optional[tuple] = None,
        expected_range: tuple = (-100.0, 100.0),
        num_classes: Optional[int] = None,
    ) -> bool:
        """
        Validate logits tensor

        Args:
            path: Path to logits file (.pt or .npy)
            expected_shape: Expected shape (N, num_classes) or None
            expected_range: Expected value range (min, max)
            num_classes: Expected number of classes

        Returns:
            True if valid

        Raises:
            LogitsValidationError: If validation fails
        """
        import torch
        import numpy as np

        # Check existence
        if not path.exists():
            raise LogitsValidationError(
                f"❌ Logits file not found: {path}\nExpected logits file does not exist."
            )

        # Load based on extension
        try:
            if path.suffix == ".pt":
                logits = torch.load(path, map_location="cpu")
                if isinstance(logits, dict):
                    logits = logits.get("logits", logits.get("predictions"))
                if isinstance(logits, torch.Tensor):
                    logits = logits.numpy()
            elif path.suffix == ".npy":
                logits = np.load(path)
            else:
                raise LogitsValidationError(
                    f"❌ Unsupported logits file format: {path.suffix}\nSupported: .pt, .npy"
                )
        except Exception as e:
            raise LogitsValidationError(f"❌ Failed to load logits: {path}\nError: {e}")

        # Check type
        if not isinstance(logits, np.ndarray):
            raise LogitsValidationError(
                f"❌ Logits is not a numpy array: {path}\nGot type: {type(logits)}"
            )

        # Check shape
        if len(logits.shape) != 2:
            raise LogitsValidationError(
                f"❌ Logits has wrong number of dimensions: {path}\n"
                f"Expected: (N, num_classes)\n"
                f"Got: {logits.shape}"
            )

        if expected_shape is not None:
            if logits.shape != expected_shape:
                raise LogitsValidationError(
                    f"❌ Logits shape mismatch: {path}\n"
                    f"Expected: {expected_shape}\n"
                    f"Got: {logits.shape}"
                )

        if num_classes is not None:
            if logits.shape[1] != num_classes:
                raise LogitsValidationError(
                    f"❌ Logits has wrong number of classes: {path}\n"
                    f"Expected: {num_classes}\n"
                    f"Got: {logits.shape[1]}"
                )

        # Check for NaN/inf
        if np.isnan(logits).any():
            raise LogitsValidationError(
                f"❌ Logits contains NaN values: {path}\nThis indicates model prediction failure."
            )

        if np.isinf(logits).any():
            raise LogitsValidationError(
                f"❌ Logits contains inf values: {path}\nThis indicates numerical instability."
            )

        # Check range
        min_val, max_val = expected_range
        if logits.min() < min_val or logits.max() > max_val:
            raise LogitsValidationError(
                f"❌ Logits values out of expected range: {path}\n"
                f"Expected range: [{min_val}, {max_val}]\n"
                f"Got range: [{logits.min():.2f}, {logits.max():.2f}]\n"
                f"This may indicate incorrect pre-processing or model issues."
            )

        return True


@dataclass(frozen=True, slots=True)
class LabelsValidator:
    """
    Labels tensor validator

    Validates:
    - File exists
    - Tensor is loadable
    - Shape matches expected (N,)
    - Values are in valid class range
    - No negative values
    """

    @staticmethod
    def validate(
        path: Path,
        expected_classes: Optional[int] = None,
        expected_length: Optional[int] = None,
    ) -> bool:
        """
        Validate labels tensor

        Args:
            path: Path to labels file (.pt or .npy)
            expected_classes: Expected number of classes (labels in [0, num_classes-1])
            expected_length: Expected number of samples

        Returns:
            True if valid

        Raises:
            LabelsValidationError: If validation fails
        """
        import torch
        import numpy as np

        # Check existence
        if not path.exists():
            raise LabelsValidationError(
                f"❌ Labels file not found: {path}\nExpected labels file does not exist."
            )

        # Load based on extension
        try:
            if path.suffix == ".pt":
                labels = torch.load(path, map_location="cpu")
                if isinstance(labels, dict):
                    labels = labels.get("labels", labels.get("targets"))
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
            elif path.suffix == ".npy":
                labels = np.load(path)
            else:
                raise LabelsValidationError(
                    f"❌ Unsupported labels file format: {path.suffix}\nSupported: .pt, .npy"
                )
        except Exception as e:
            raise LabelsValidationError(f"❌ Failed to load labels: {path}\nError: {e}")

        # Check type
        if not isinstance(labels, np.ndarray):
            raise LabelsValidationError(
                f"❌ Labels is not a numpy array: {path}\nGot type: {type(labels)}"
            )

        # Check shape
        if len(labels.shape) != 1:
            raise LabelsValidationError(
                f"❌ Labels has wrong number of dimensions: {path}\n"
                f"Expected: (N,)\n"
                f"Got: {labels.shape}"
            )

        if expected_length is not None:
            if len(labels) != expected_length:
                raise LabelsValidationError(
                    f"❌ Labels length mismatch: {path}\n"
                    f"Expected: {expected_length}\n"
                    f"Got: {len(labels)}"
                )

        # Check dtype (should be integer)
        if not np.issubdtype(labels.dtype, np.integer):
            raise LabelsValidationError(
                f"❌ Labels has wrong dtype: {path}\nExpected: integer type\nGot: {labels.dtype}"
            )

        # Check for negative values
        if (labels < 0).any():
            raise LabelsValidationError(
                f"❌ Labels contains negative values: {path}\nLabels must be >= 0"
            )

        # Check class range
        if expected_classes is not None:
            if labels.max() >= expected_classes:
                raise LabelsValidationError(
                    f"❌ Labels out of class range: {path}\n"
                    f"Expected classes: [0, {expected_classes - 1}]\n"
                    f"Got max label: {labels.max()}\n"
                    f"This indicates a mismatch between data and model."
                )

        return True


@dataclass(frozen=True, slots=True)
class PolicyValidator:
    """
    Policy JSON validator

    Validates:
    - File exists
    - JSON is valid
    - Required fields are present
    - Policy type is valid
    """

    VALID_POLICY_TYPES = {
        "threshold",  # Softmax + threshold
        "gate",  # Learned gate head
        "scrc",  # SCRC calibration
        "temperature",  # Temperature scaling
        "dirichlet",  # Dirichlet calibration
        "beta",  # Beta calibration
        "platt",  # Platt scaling
        "isotonic",  # Isotonic regression
        "ensemble",  # Ensemble of methods
    }

    @staticmethod
    def validate(
        path: Path,
        policy_type: Optional[str] = None,
        required_fields: Optional[Set[str]] = None,
    ) -> bool:
        """
        Validate policy JSON file

        Args:
            path: Path to policy JSON file
            policy_type: Expected policy type (or None to skip check)
            required_fields: Set of required fields in JSON

        Returns:
            True if valid

        Raises:
            PolicyValidationError: If validation fails
        """
        # Check existence
        if not path.exists():
            raise PolicyValidationError(
                f"❌ Policy file not found: {path}\nExpected policy file does not exist."
            )

        # Load JSON
        try:
            with open(path, "r") as f:
                policy = json.load(f)
        except json.JSONDecodeError as e:
            raise PolicyValidationError(f"❌ Invalid JSON in policy file: {path}\nError: {e}")
        except Exception as e:
            raise PolicyValidationError(f"❌ Failed to load policy file: {path}\nError: {e}")

        # Check type
        if not isinstance(policy, dict):
            raise PolicyValidationError(
                f"❌ Policy is not a dict: {path}\nGot type: {type(policy)}"
            )

        # Check policy type
        if "policy_type" not in policy:
            raise PolicyValidationError(
                f"❌ Policy missing 'policy_type' field: {path}\n"
                f"Available fields: {set(policy.keys())}"
            )

        if policy["policy_type"] not in PolicyValidator.VALID_POLICY_TYPES:
            raise PolicyValidationError(
                f"❌ Invalid policy type: {path}\n"
                f"Got: {policy['policy_type']}\n"
                f"Valid types: {PolicyValidator.VALID_POLICY_TYPES}"
            )

        if policy_type is not None:
            if policy["policy_type"] != policy_type:
                raise PolicyValidationError(
                    f"❌ Policy type mismatch: {path}\n"
                    f"Expected: {policy_type}\n"
                    f"Got: {policy['policy_type']}"
                )

        # Check required fields
        if required_fields is not None:
            missing_fields = required_fields - set(policy.keys())
            if missing_fields:
                raise PolicyValidationError(
                    f"❌ Policy missing required fields: {path}\n"
                    f"Missing: {missing_fields}\n"
                    f"Available: {set(policy.keys())}"
                )

        return True


@dataclass(frozen=True, slots=True)
class BundleValidator:
    """
    Bundle validator - ensures mutual exclusivity

    CRITICAL: A bundle can have EXACTLY ONE policy:
    - threshold policy (Phase 2)
    - OR gate policy (Phase 3)
    - OR scrc policy (Phase 5)
    - OR calibration policy (temperature, etc.)

    Multiple policies = ambiguous deployment = ERROR
    """

    @staticmethod
    def validate(path: Path) -> bool:
        """
        Validate bundle JSON file

        Args:
            path: Path to bundle JSON file

        Returns:
            True if valid

        Raises:
            BundleValidationError: If validation fails
        """
        # Check existence
        if not path.exists():
            raise BundleValidationError(
                f"❌ Bundle file not found: {path}\nExpected bundle file does not exist."
            )

        # Load JSON
        try:
            with open(path, "r") as f:
                bundle = json.load(f)
        except json.JSONDecodeError as e:
            raise BundleValidationError(f"❌ Invalid JSON in bundle file: {path}\nError: {e}")
        except Exception as e:
            raise BundleValidationError(f"❌ Failed to load bundle file: {path}\nError: {e}")

        # Check type
        if not isinstance(bundle, dict):
            raise BundleValidationError(
                f"❌ Bundle is not a dict: {path}\nGot type: {type(bundle)}"
            )

        # Check required fields
        required_fields = {"model_checkpoint", "policy"}
        missing_fields = required_fields - set(bundle.keys())
        if missing_fields:
            raise BundleValidationError(
                f"❌ Bundle missing required fields: {path}\n"
                f"Missing: {missing_fields}\n"
                f"Required: {required_fields}"
            )

        # CRITICAL: Check policy mutual exclusivity
        policy = bundle["policy"]
        if not isinstance(policy, dict):
            raise BundleValidationError(
                f"❌ Bundle policy is not a dict: {path}\nGot type: {type(policy)}"
            )

        if "policy_type" not in policy:
            raise BundleValidationError(f"❌ Bundle policy missing 'policy_type': {path}")

        # Validate policy using PolicyValidator
        # We can't validate the policy file directly, but we can validate the policy dict
        if policy["policy_type"] not in PolicyValidator.VALID_POLICY_TYPES:
            raise BundleValidationError(
                f"❌ Bundle has invalid policy type: {path}\n"
                f"Got: {policy['policy_type']}\n"
                f"Valid types: {PolicyValidator.VALID_POLICY_TYPES}"
            )

        # CRITICAL: Enforce EXACTLY ONE policy file exists (mutual exclusivity)
        # Check parent directory for policy files
        bundle_dir = path.parent.parent  # outputs/export -> outputs/
        policy_files = {
            "thresholds_json": bundle_dir / "phase2" / "thresholds.json",
            "gateparams_json": bundle_dir / "phase3" / "gateparams.json",
            "scrcparams_json": bundle_dir / "phase5_scrc" / "scrcparams.json",
        }

        existing_policies = [name for name, fpath in policy_files.items() if fpath.exists()]

        if len(existing_policies) > 1:
            raise BundleValidationError(
                f"❌ BUNDLE MUTUAL EXCLUSIVITY VIOLATION: {path}\n"
                f"Found {len(existing_policies)} policy files, expected EXACTLY ONE!\n"
                f"Existing policies: {existing_policies}\n"
                f"Bundle can only have ONE policy: threshold OR gate OR scrc\n"
                f"Multiple policies create ambiguous deployment!"
            )

        if len(existing_policies) == 0:
            raise BundleValidationError(
                f"❌ BUNDLE VALIDATION ERROR: {path}\n"
                f"No policy files found!\n"
                f"Bundle must have EXACTLY ONE policy (threshold, gate, or scrc)"
            )

        return True


@dataclass(frozen=True, slots=True)
class ArtifactValidator:
    """
    Main artifact validator - validates phase outputs

    Usage:
        from contracts.artifact_schema import create_artifact_schema
        from contracts.validators import ArtifactValidator

        artifacts = create_artifact_schema("outputs")
        validator = ArtifactValidator()

        # Validate Phase 1 outputs
        validator.validate_phase_outputs(phase=1, artifacts=artifacts)
    """

    @staticmethod
    def validate_phase_outputs(phase: int, artifacts: Any) -> bool:
        """
        Validate all outputs for a phase

        Args:
            phase: Phase number (1-6)
            artifacts: ArtifactSchema instance

        Returns:
            True if all outputs are valid

        Raises:
            ArtifactValidationError: If any validation fails
        """
        from .artifact_schema import ArtifactSchema

        if not isinstance(artifacts, ArtifactSchema):
            raise ArtifactValidationError(
                f"❌ Invalid artifacts type\nExpected: ArtifactSchema\nGot: {type(artifacts)}"
            )

        # Get expected outputs for this phase
        expected_outputs = artifacts.get_expected_outputs(phase)

        # Check all outputs exist
        missing_outputs = [p for p in expected_outputs if not p.exists()]
        if missing_outputs:
            raise ArtifactValidationError(
                f"❌ Phase {phase} missing expected outputs:\n"
                + "\n".join([f"  - {p}" for p in missing_outputs])
            )

        # Phase-specific validation
        if phase == 1:
            # Validate Phase 1 outputs
            CheckpointValidator.validate(
                artifacts.phase1_checkpoint,
                required_keys={"model_state_dict", "epoch", "best_val_acc"},
            )
            # Note: Can't validate logits/labels shape without knowing dataset size
            # That validation happens at runtime

        elif phase == 2:
            # Validate Phase 2 outputs
            PolicyValidator.validate(
                artifacts.thresholds_json,
                policy_type="threshold",
                required_fields={"policy_type", "thresholds", "class_names"},
            )

        elif phase == 3:
            # Validate Phase 3 outputs
            CheckpointValidator.validate(
                artifacts.phase3_checkpoint,
                required_keys={"model_state_dict", "epoch"},
            )
            PolicyValidator.validate(
                artifacts.gateparams_json,
                policy_type="gate",
                required_fields={"policy_type", "gate_threshold"},
            )

        elif phase == 5:
            # Validate Phase 5 outputs
            PolicyValidator.validate(
                artifacts.scrcparams_json,
                policy_type="scrc",
                required_fields={"policy_type", "scrc_params"},
            )

        elif phase == 6:
            # Validate Phase 6 outputs
            BundleValidator.validate(artifacts.bundle_json)

        return True


# Convenience function for quick validation
def validate_checkpoint(path: Path, required_keys: Optional[Set[str]] = None) -> bool:
    """Validate checkpoint file - convenience wrapper"""
    return CheckpointValidator.validate(path, required_keys=required_keys)


def validate_logits(
    path: Path,
    expected_shape: Optional[tuple] = None,
    num_classes: Optional[int] = None,
) -> bool:
    """Validate logits tensor - convenience wrapper"""
    return LogitsValidator.validate(path, expected_shape=expected_shape, num_classes=num_classes)


def validate_labels(
    path: Path,
    expected_classes: Optional[int] = None,
    expected_length: Optional[int] = None,
) -> bool:
    """Validate labels tensor - convenience wrapper"""
    return LabelsValidator.validate(
        path, expected_classes=expected_classes, expected_length=expected_length
    )


def validate_policy(
    path: Path,
    policy_type: Optional[str] = None,
    required_fields: Optional[Set[str]] = None,
) -> bool:
    """Validate policy JSON - convenience wrapper"""
    return PolicyValidator.validate(path, policy_type=policy_type, required_fields=required_fields)


def validate_bundle(path: Path) -> bool:
    """Validate bundle JSON - convenience wrapper"""
    return BundleValidator.validate(path)


if __name__ == "__main__":
    # Test validators
    print("📋 Artifact Validators Test\n")

    # Note: These tests would require actual files to exist
    # This is just a demonstration of the API

    print("✅ Validator classes loaded successfully!")
    print("\n💡 Summary: Validators provide fail-fast checking for all artifacts!")
    print("   - CheckpointValidator: Validates model checkpoints")
    print("   - LogitsValidator: Validates logits tensors")
    print("   - LabelsValidator: Validates label tensors")
    print("   - PolicyValidator: Validates policy JSON files")
    print("   - BundleValidator: Validates deployment bundles (mutual exclusivity)")
    print("   - ArtifactValidator: Validates complete phase outputs")
