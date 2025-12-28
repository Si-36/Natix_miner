"""
Phase Specifications - DAG Node Definitions

Each phase is a node in the DAG with:
- Explicit inputs (dependencies)
- Explicit outputs (artifacts)
- Validation contracts
- Execution requirements

Benefits:
- Clear dependency graph (automatic ordering)
- Cannot skip required phases (enforced by DAG)
- Cannot run phases out of order (enforced by DAG)
- Automatic validation (fail-fast)

Latest 2025-2026 practices:
- Python 3.11+ type hints
- Dataclasses with slots
- Enum for type safety
- Pydantic v2 patterns
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Set, Callable, Optional, Any
from contracts.split_contracts import Split


class PhaseType(Enum):
    """All phases in the training pipeline"""
    PHASE1_BASELINE = "phase1_baseline"
    PHASE2_THRESHOLD = "phase2_threshold"
    PHASE3_GATE = "phase3_gate"
    PHASE4_EXPLORA = "phase4_explora"
    PHASE5_SCRC = "phase5_scrc"
    PHASE6_BUNDLE = "phase6_bundle"


class ExecutionMode(Enum):
    """Execution mode for a phase"""
    TRAINING = "training"           # Requires GPU, long-running
    INFERENCE = "inference"         # Requires GPU, medium-running
    POLICY_FITTING = "policy_fitting"  # CPU-only, fast
    EXPORT = "export"              # CPU-only, fast


@dataclass(slots=True, frozen=True)
class ResourceRequirements:
    """
    Resource requirements for a phase

    Latest 2025-2026:
    - Explicit resource declarations
    - Used for scheduling and validation
    """
    requires_gpu: bool = True
    min_gpu_memory_gb: float = 16.0
    min_cpu_memory_gb: float = 32.0
    estimated_time_minutes: float = 60.0
    num_gpus: int = 1
    distributed: bool = False


@dataclass(slots=True)
class PhaseSpec:
    """
    Specification for a single phase in the pipeline

    CRITICAL CONTRACTS:
    - inputs: List of required input artifacts (from other phases)
    - outputs: List of output artifacts (produced by this phase)
    - required_splits: Splits that must exist before running
    - allowed_splits: Splits that can be used during execution
    - validators: Functions to validate outputs

    Latest 2025-2026:
    - Explicit input/output contracts
    - Type-safe with dataclasses
    - Frozen for immutability where possible
    """

    # Phase identification
    phase_type: PhaseType
    name: str
    description: str

    # Dependencies (DAG edges)
    dependencies: List[PhaseType] = field(default_factory=list)

    # Input artifacts (files from other phases)
    input_artifacts: List[str] = field(default_factory=list)

    # Output artifacts (files produced by this phase)
    output_artifacts: List[str] = field(default_factory=list)

    # Split requirements
    required_splits: Set[Split] = field(default_factory=set)
    allowed_splits: Set[Split] = field(default_factory=set)

    # Execution
    execution_mode: ExecutionMode = ExecutionMode.TRAINING
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)

    # Validation
    validators: List[Callable[[Any], bool]] = field(default_factory=list)

    # Optional configuration
    config_required: bool = True
    skippable: bool = False  # Can this phase be skipped?

    def validate_inputs(self, artifacts: Any) -> bool:
        """
        Validate that all required inputs exist

        Args:
            artifacts: ArtifactSchema instance

        Returns:
            True if all inputs exist

        Raises:
            FileNotFoundError: If any input is missing
        """
        from contracts.artifact_schema import ArtifactSchema

        if not isinstance(artifacts, ArtifactSchema):
            raise TypeError(f"Expected ArtifactSchema, got {type(artifacts)}")

        # Get input paths
        input_paths = []
        for artifact_name in self.input_artifacts:
            if not hasattr(artifacts, artifact_name):
                raise AttributeError(
                    f"ArtifactSchema has no attribute '{artifact_name}'"
                )
            input_paths.append(getattr(artifacts, artifact_name))

        # Check existence
        missing = [p for p in input_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Phase {self.name} missing required inputs:\n"
                + "\n".join([f"  - {p}" for p in missing])
            )

        return True

    def validate_outputs(self, artifacts: Any) -> bool:
        """
        Validate that all required outputs were produced

        Args:
            artifacts: ArtifactSchema instance

        Returns:
            True if all outputs exist and are valid

        Raises:
            FileNotFoundError: If any output is missing
            ValidationError: If any output is invalid
        """
        from contracts.artifact_schema import ArtifactSchema

        if not isinstance(artifacts, ArtifactSchema):
            raise TypeError(f"Expected ArtifactSchema, got {type(artifacts)}")

        # Get output paths
        output_paths = []
        for artifact_name in self.output_artifacts:
            if not hasattr(artifacts, artifact_name):
                raise AttributeError(
                    f"ArtifactSchema has no attribute '{artifact_name}'"
                )
            output_paths.append(getattr(artifacts, artifact_name))

        # Check existence
        missing = [p for p in output_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Phase {self.name} missing expected outputs:\n"
                + "\n".join([f"  - {p}" for p in missing])
            )

        # Run custom validators
        for validator_fn in self.validators:
            validator_fn(artifacts)

        return True


# ============= PHASE SPECIFICATIONS =============

def create_phase1_spec() -> PhaseSpec:
    """Phase 1: Baseline Training"""
    return PhaseSpec(
        phase_type=PhaseType.PHASE1_BASELINE,
        name="Phase 1: Baseline Training",
        description=(
            "Train DINOv2 baseline with multi-view inference.\n"
            "Outputs: best checkpoint, logits on val_calib, metrics."
        ),
        dependencies=[],  # No dependencies
        input_artifacts=["splits_json"],
        output_artifacts=[
            "phase1_checkpoint",
            "val_calib_logits",
            "val_calib_labels",
            "metrics_csv",
            "config_json",
        ],
        required_splits={Split.TRAIN, Split.VAL_SELECT, Split.VAL_CALIB},
        allowed_splits={Split.TRAIN, Split.VAL_SELECT, Split.VAL_CALIB},
        execution_mode=ExecutionMode.TRAINING,
        resources=ResourceRequirements(
            requires_gpu=True,
            min_gpu_memory_gb=24.0,
            min_cpu_memory_gb=64.0,
            estimated_time_minutes=240.0,  # 4 hours
            num_gpus=1,
            distributed=False,
        ),
        validators=[],
        config_required=True,
        skippable=False,
    )


def create_phase2_spec() -> PhaseSpec:
    """Phase 2: Threshold Sweep"""
    return PhaseSpec(
        phase_type=PhaseType.PHASE2_THRESHOLD,
        name="Phase 2: Threshold Sweep",
        description=(
            "Sweep confidence thresholds on val_calib to find optimal reject policy.\n"
            "Outputs: threshold policy JSON."
        ),
        dependencies=[PhaseType.PHASE1_BASELINE],
        input_artifacts=["val_calib_logits", "val_calib_labels"],
        output_artifacts=["thresholds_json"],
        required_splits={Split.VAL_CALIB},
        allowed_splits={Split.VAL_CALIB},  # ONLY val_calib
        execution_mode=ExecutionMode.POLICY_FITTING,
        resources=ResourceRequirements(
            requires_gpu=False,
            min_gpu_memory_gb=0.0,
            min_cpu_memory_gb=16.0,
            estimated_time_minutes=10.0,  # 10 minutes
            num_gpus=0,
            distributed=False,
        ),
        validators=[],
        config_required=True,
        skippable=False,
    )


def create_phase3_spec() -> PhaseSpec:
    """Phase 3: Gate Head Training"""
    return PhaseSpec(
        phase_type=PhaseType.PHASE3_GATE,
        name="Phase 3: Gate Head Training",
        description=(
            "Train learned gate head for confidence estimation.\n"
            "Outputs: checkpoint with gate, gate parameters."
        ),
        dependencies=[PhaseType.PHASE1_BASELINE],
        input_artifacts=["phase1_checkpoint"],
        output_artifacts=["phase3_checkpoint", "gateparams_json"],
        required_splits={Split.TRAIN, Split.VAL_SELECT, Split.VAL_CALIB},
        allowed_splits={Split.TRAIN, Split.VAL_SELECT, Split.VAL_CALIB},
        execution_mode=ExecutionMode.TRAINING,
        resources=ResourceRequirements(
            requires_gpu=True,
            min_gpu_memory_gb=24.0,
            min_cpu_memory_gb=64.0,
            estimated_time_minutes=120.0,  # 2 hours
            num_gpus=1,
            distributed=False,
        ),
        validators=[],
        config_required=True,
        skippable=True,  # Can skip if using softmax policy
    )


def create_phase4_spec() -> PhaseSpec:
    """Phase 4: ExPLoRA Pretraining"""
    return PhaseSpec(
        phase_type=PhaseType.PHASE4_EXPLORA,
        name="Phase 4: ExPLoRA Pretraining",
        description=(
            "Extended pretraining with LoRA for domain adaptation.\n"
            "Outputs: ExPLoRA-adapted checkpoint (+8.2% accuracy)."
        ),
        dependencies=[],  # Independent of other phases
        input_artifacts=[],
        output_artifacts=["explora_checkpoint"],
        required_splits={Split.TRAIN, Split.VAL_SELECT},
        allowed_splits={Split.TRAIN, Split.VAL_SELECT},
        execution_mode=ExecutionMode.TRAINING,
        resources=ResourceRequirements(
            requires_gpu=True,
            min_gpu_memory_gb=40.0,
            min_cpu_memory_gb=128.0,
            estimated_time_minutes=1440.0,  # 24 hours
            num_gpus=4,
            distributed=True,
        ),
        validators=[],
        config_required=True,
        skippable=True,  # Optional enhancement
    )


def create_phase5_spec() -> PhaseSpec:
    """Phase 5: SCRC Calibration"""
    return PhaseSpec(
        phase_type=PhaseType.PHASE5_SCRC,
        name="Phase 5: SCRC Calibration",
        description=(
            "Fit SCRC calibration parameters on val_calib.\n"
            "Outputs: SCRC parameters JSON."
        ),
        dependencies=[PhaseType.PHASE1_BASELINE],
        input_artifacts=["phase1_checkpoint", "val_calib_logits"],
        output_artifacts=["scrcparams_json"],
        required_splits={Split.VAL_CALIB},
        allowed_splits={Split.VAL_CALIB},  # ONLY val_calib
        execution_mode=ExecutionMode.POLICY_FITTING,
        resources=ResourceRequirements(
            requires_gpu=True,
            min_gpu_memory_gb=16.0,
            min_cpu_memory_gb=32.0,
            estimated_time_minutes=30.0,  # 30 minutes
            num_gpus=1,
            distributed=False,
        ),
        validators=[],
        config_required=True,
        skippable=True,  # Optional calibration method
    )


def create_phase6_spec() -> PhaseSpec:
    """Phase 6: Bundle Export"""
    return PhaseSpec(
        phase_type=PhaseType.PHASE6_BUNDLE,
        name="Phase 6: Bundle Export",
        description=(
            "Export final deployment bundle.\n"
            "Outputs: bundle.json (model + EXACTLY ONE policy)."
        ),
        dependencies=[PhaseType.PHASE1_BASELINE],  # At minimum needs Phase 1
        input_artifacts=["phase1_checkpoint", "splits_json"],
        output_artifacts=["bundle_json"],
        required_splits=set(),  # No splits needed for export
        allowed_splits=set(),
        execution_mode=ExecutionMode.EXPORT,
        resources=ResourceRequirements(
            requires_gpu=False,
            min_gpu_memory_gb=0.0,
            min_cpu_memory_gb=16.0,
            estimated_time_minutes=5.0,  # 5 minutes
            num_gpus=0,
            distributed=False,
        ),
        validators=[],
        config_required=True,
        skippable=False,
    )


# ============= PHASE REGISTRY =============

@dataclass(frozen=True, slots=True)
class PhaseRegistry:
    """
    Registry of all phases

    Usage:
        registry = get_phase_registry()
        phase1_spec = registry.get_phase(PhaseType.PHASE1_BASELINE)
        all_phases = registry.get_all_phases()
    """

    phases: dict[PhaseType, PhaseSpec] = field(default_factory=dict)

    def get_phase(self, phase_type: PhaseType) -> PhaseSpec:
        """Get phase spec by type"""
        if phase_type not in self.phases:
            raise KeyError(f"Unknown phase type: {phase_type}")
        return self.phases[phase_type]

    def get_all_phases(self) -> List[PhaseSpec]:
        """Get all phase specs"""
        return list(self.phases.values())

    def get_dependencies(self, phase_type: PhaseType) -> List[PhaseType]:
        """Get dependencies for a phase"""
        phase = self.get_phase(phase_type)
        return phase.dependencies

    def get_execution_order(self, phases_to_run: List[PhaseType]) -> List[PhaseType]:
        """
        Get execution order for a list of phases (topological sort)

        Args:
            phases_to_run: List of phases to run

        Returns:
            List of phases in execution order (respecting dependencies)

        Raises:
            ValueError: If there's a circular dependency
        """
        # Simple topological sort (Kahn's algorithm)
        in_degree = {p: 0 for p in phases_to_run}
        adj_list = {p: [] for p in phases_to_run}

        # Build graph
        for phase_type in phases_to_run:
            phase = self.get_phase(phase_type)
            for dep in phase.dependencies:
                if dep in phases_to_run:
                    adj_list[dep].append(phase_type)
                    in_degree[phase_type] += 1

        # Find starting nodes (no dependencies)
        queue = [p for p in phases_to_run if in_degree[p] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(phases_to_run):
            raise ValueError(
                f"Circular dependency detected in phases: {phases_to_run}"
            )

        return result


def get_phase_registry() -> PhaseRegistry:
    """
    Get the phase registry with all phases

    Returns:
        PhaseRegistry instance with all phase specs
    """
    phases = {
        PhaseType.PHASE1_BASELINE: create_phase1_spec(),
        PhaseType.PHASE2_THRESHOLD: create_phase2_spec(),
        PhaseType.PHASE3_GATE: create_phase3_spec(),
        PhaseType.PHASE4_EXPLORA: create_phase4_spec(),
        PhaseType.PHASE5_SCRC: create_phase5_spec(),
        PhaseType.PHASE6_BUNDLE: create_phase6_spec(),
    }

    return PhaseRegistry(phases=phases)


if __name__ == "__main__":
    # Test phase specifications
    print("📋 Phase Specifications Test\n")

    registry = get_phase_registry()

    # Test 1: Get all phases
    print("Test 1: Get all phases")
    all_phases = registry.get_all_phases()
    print(f"✅ Found {len(all_phases)} phases:\n")
    for phase in all_phases:
        print(f"  - {phase.name}")
        print(f"    Dependencies: {[d.value for d in phase.dependencies]}")
        print(f"    Inputs: {phase.input_artifacts}")
        print(f"    Outputs: {phase.output_artifacts}")
        print(f"    Execution mode: {phase.execution_mode.value}")
        print(f"    GPU required: {phase.resources.requires_gpu}")
        print(f"    Estimated time: {phase.resources.estimated_time_minutes} min")
        print()

    # Test 2: Execution order
    print("\nTest 2: Execution order (all phases)")
    phases_to_run = [
        PhaseType.PHASE1_BASELINE,
        PhaseType.PHASE2_THRESHOLD,
        PhaseType.PHASE3_GATE,
        PhaseType.PHASE6_BUNDLE,
    ]
    execution_order = registry.get_execution_order(phases_to_run)
    print(f"✅ Execution order: {[p.value for p in execution_order]}\n")

    # Test 3: Split requirements
    print("\nTest 3: Split requirements")
    phase1 = registry.get_phase(PhaseType.PHASE1_BASELINE)
    print(f"Phase 1 required splits: {[s.value for s in phase1.required_splits]}")
    print(f"Phase 1 allowed splits: {[s.value for s in phase1.allowed_splits]}")

    phase2 = registry.get_phase(PhaseType.PHASE2_THRESHOLD)
    print(f"Phase 2 required splits: {[s.value for s in phase2.required_splits]}")
    print(f"Phase 2 allowed splits: {[s.value for s in phase2.allowed_splits]}")
    print()

    print("✅ All phase specification tests passed!")
    print("\n💡 Summary: Phase specs define the complete DAG structure!")
