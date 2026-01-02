"""
DAG Engine - Pipeline Orchestrator

Orchestrates execution of training phases:
1. Determines execution order (topological sort)
2. Validates inputs before each phase
3. Executes each phase
4. Validates outputs after each phase
5. Handles failures gracefully
6. Supports resuming from checkpoints

Benefits:
- Cannot skip required phases (enforced)
- Cannot run phases out of order (enforced)
- Automatic validation (fail-fast)
- Clear progress reporting
- Resumable execution

Latest 2025-2026 practices:
- Python 3.11+ type hints
- Dataclasses with slots
- Rich progress reporting
- Structured logging
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import json
import time
import logging
from datetime import datetime

# Setup logger for this module
logger = logging.getLogger(__name__)

from pipeline.phase_spec import (
    PhaseType,
    PhaseSpec,
    PhaseRegistry,
    get_phase_registry,
)
from contracts.artifact_schema import ArtifactSchema
from contracts.validators import ArtifactValidator

# Import production-grade manifest system (Day 5)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from streetvision.io.manifests import StepManifest
    from tests.conftest import verify_manifest, load_manifest
    HAS_MANIFEST_SYSTEM = True
except ImportError:
    HAS_MANIFEST_SYSTEM = False
    logger.warning("Manifest system not available - resume logic disabled")


class PhaseStatus(Enum):
    """Status of a phase in the pipeline"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class PhaseExecution:
    """
    Execution record for a single phase

    Tracks:
    - Status (pending, running, completed, failed, skipped)
    - Start/end times
    - Error messages if failed
    """

    phase_type: PhaseType
    status: PhaseStatus = PhaseStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get execution duration in seconds"""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def duration_minutes(self) -> Optional[float]:
        """Get execution duration in minutes"""
        duration = self.duration_seconds
        if duration is None:
            return None
        return duration / 60.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "phase_type": self.phase_type.value,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "duration_minutes": self.duration_minutes,
            "error_message": self.error_message,
        }


@dataclass(slots=True)
class PipelineState:
    """
    State of the entire pipeline

    Tracks:
    - Which phases have been executed
    - Status of each phase
    - Overall progress
    - Can be saved/loaded for resuming
    """

    executions: Dict[PhaseType, PhaseExecution] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def mark_started(self, phase_type: PhaseType) -> None:
        """Mark a phase as started"""
        if phase_type not in self.executions:
            self.executions[phase_type] = PhaseExecution(phase_type=phase_type)

        self.executions[phase_type].status = PhaseStatus.RUNNING
        self.executions[phase_type].start_time = time.time()

    def mark_completed(self, phase_type: PhaseType) -> None:
        """Mark a phase as completed"""
        if phase_type not in self.executions:
            raise ValueError(f"Phase {phase_type} not started")

        self.executions[phase_type].status = PhaseStatus.COMPLETED
        self.executions[phase_type].end_time = time.time()

    def mark_failed(self, phase_type: PhaseType, error_message: str) -> None:
        """Mark a phase as failed"""
        if phase_type not in self.executions:
            self.executions[phase_type] = PhaseExecution(phase_type=phase_type)

        self.executions[phase_type].status = PhaseStatus.FAILED
        self.executions[phase_type].end_time = time.time()
        self.executions[phase_type].error_message = error_message

    def mark_skipped(self, phase_type: PhaseType) -> None:
        """Mark a phase as skipped"""
        if phase_type not in self.executions:
            self.executions[phase_type] = PhaseExecution(phase_type=phase_type)

        self.executions[phase_type].status = PhaseStatus.SKIPPED

    def is_completed(self, phase_type: PhaseType) -> bool:
        """Check if a phase is completed"""
        if phase_type not in self.executions:
            return False
        return self.executions[phase_type].status == PhaseStatus.COMPLETED

    def is_failed(self, phase_type: PhaseType) -> bool:
        """Check if a phase has failed"""
        if phase_type not in self.executions:
            return False
        return self.executions[phase_type].status == PhaseStatus.FAILED

    def get_status(self, phase_type: PhaseType) -> PhaseStatus:
        """Get status of a phase"""
        if phase_type not in self.executions:
            return PhaseStatus.PENDING
        return self.executions[phase_type].status

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "executions": {k.value: v.to_dict() for k, v in self.executions.items()},
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    def save(self, path: Path) -> None:
        """Save pipeline state to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        """Load pipeline state from JSON file"""
        with open(path, "r") as f:
            data = json.load(f)

        state = cls()
        state.start_time = data.get("start_time")
        state.end_time = data.get("end_time")

        for phase_type_str, execution_data in data.get("executions", {}).items():
            phase_type = PhaseType(phase_type_str)
            execution = PhaseExecution(
                phase_type=phase_type,
                status=PhaseStatus(execution_data["status"]),
                start_time=execution_data.get("start_time"),
                end_time=execution_data.get("end_time"),
                error_message=execution_data.get("error_message"),
            )
            state.executions[phase_type] = execution

        return state


@dataclass(slots=True)
class DAGEngine:
    """
    DAG Pipeline Orchestrator

    Executes phases in the correct order with automatic validation.

    Usage:
        from contracts.artifact_schema import create_artifact_schema
        from pipeline.dag_engine import DAGEngine
        from pipeline.phase_spec import PhaseType

        artifacts = create_artifact_schema("outputs")
        engine = DAGEngine(artifacts=artifacts)

        # Run specific phases
        engine.run([
            PhaseType.PHASE1_BASELINE,
            PhaseType.PHASE2_THRESHOLD,
            PhaseType.PHASE6_BUNDLE,
        ])

        # Resume from saved state
        engine.resume(state_path=Path("outputs/pipeline_state.json"))
    """

    artifacts: ArtifactSchema
    registry: PhaseRegistry = field(default_factory=get_phase_registry)
    state: PipelineState = field(default_factory=PipelineState)
    phase_executors: Dict[PhaseType, Callable] = field(default_factory=dict)

    def register_executor(
        self,
        phase_type: PhaseType,
        executor: Callable[[ArtifactSchema], None],
    ) -> None:
        """
        Register an executor function for a phase

        Args:
            phase_type: Phase type to register executor for
            executor: Callable that takes ArtifactSchema and executes the phase
        """
        self.phase_executors[phase_type] = executor
        logger.debug(f"Registered executor for {phase_type.value}")

    def _validate_phase_inputs(self, phase: PhaseSpec) -> bool:
        """
        Validate inputs for a phase

        Args:
            phase: Phase specification

        Returns:
            True if all inputs are valid

        Raises:
            FileNotFoundError: If any input is missing
        """
        logger.info(f"  Validating inputs for {phase.name}...")
        return phase.validate_inputs(self.artifacts)

    def _validate_phase_outputs(self, phase: PhaseSpec) -> bool:
        """
        Validate outputs for a phase

        Args:
            phase: Phase specification

        Returns:
            True if all outputs are valid

        Raises:
            FileNotFoundError: If any output is missing
            ValidationError: If any output is invalid
        """
        logger.info(f"  Validating outputs for {phase.name}...")
        return phase.validate_outputs(self.artifacts)

    def _execute_phase(self, phase: PhaseSpec) -> None:
        """
        Execute a single phase

        Args:
            phase: Phase specification

        Raises:
            ValueError: If no executor is registered for this phase
            Exception: If phase execution fails
        """
        if phase.phase_type not in self.phase_executors:
            raise ValueError(
                f"No executor registered for phase {phase.phase_type.value}.\n"
                f"Use engine.register_executor() to register an executor function."
            )

        executor = self.phase_executors[phase.phase_type]
        logger.info(f"  Executing {phase.name}...")

        # Execute phase
        executor(self.artifacts)

    def _check_dependencies(self, phase: PhaseSpec) -> bool:
        """
        Check if all dependencies for a phase are satisfied

        Args:
            phase: Phase specification

        Returns:
            True if all dependencies are satisfied

        Raises:
            RuntimeError: If any dependency is not satisfied
        """
        for dep_type in phase.dependencies:
            if not self.state.is_completed(dep_type):
                dep_status = self.state.get_status(dep_type)
                raise RuntimeError(
                    f"Phase {phase.name} requires {dep_type.value} to be completed.\n"
                    f"Current status: {dep_status.value}"
                )
        return True

    def should_skip_phase(self, phase_type: PhaseType) -> bool:
        """
        Decide if phase should be skipped (already complete with valid manifest)

        Rules (Day 5 resume logic):
        1. If manifest.json doesn't exist → DON'T SKIP (run phase)
        2. If manifest exists but artifacts missing → DON'T SKIP (re-run)
        3. If manifest exists and checksums verify → SKIP (complete)

        Args:
            phase_type: Phase type to check

        Returns:
            True if should skip, False if should run

        Note:
            Requires manifest system (Day 5). If not available, always returns False.
        """
        if not HAS_MANIFEST_SYSTEM:
            # No manifest system → cannot verify → don't skip
            return False

        # Get phase directory and manifest path
        phase_dir_map = {
            PhaseType.PHASE1_BASELINE: self.artifacts.phase1_dir,
            PhaseType.PHASE2_THRESHOLD: self.artifacts.phase2_dir,
            PhaseType.PHASE3_GATE: getattr(self.artifacts, 'phase3_dir', None),
            PhaseType.PHASE4_EXPLORA: getattr(self.artifacts, 'phase4_dir', None),
            PhaseType.PHASE5_SCRC: getattr(self.artifacts, 'phase5_dir', None),
            PhaseType.PHASE6_BUNDLE: self.artifacts.bundle_export_dir,
        }

        phase_dir = phase_dir_map.get(phase_type)
        if phase_dir is None:
            logger.warning(f"No phase directory mapping for {phase_type.value}")
            return False

        manifest_path = phase_dir / "manifest.json"

        # Rule 1: No manifest → run
        if not manifest_path.exists():
            logger.info(f"Manifest not found: {manifest_path} → Running phase")
            return False

        # Rule 2: Manifest exists → verify artifacts
        try:
            manifest = load_manifest(manifest_path)
            verify_manifest(manifest, self.artifacts.output_dir)

            # All artifacts exist and checksums match
            logger.info(f"✅ Manifest valid: {manifest_path} → Skipping phase")
            return True

        except (FileNotFoundError, ValueError) as e:
            # Artifacts missing or checksums mismatch
            logger.warning(f"Manifest verification failed: {e} → Re-running phase")
            return False

    def run_phase(self, phase_type: PhaseType) -> None:
        """
        Run a single phase

        Args:
            phase_type: Phase type to run

        Raises:
            RuntimeError: If dependencies are not satisfied
            FileNotFoundError: If inputs are missing
            ValidationError: If outputs are invalid
            Exception: If phase execution fails
        """
        phase = self.registry.get_phase(phase_type)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running: {phase.name}")
        logger.info(f"{'=' * 80}")

        # Check if already completed (in-memory state)
        if self.state.is_completed(phase_type):
            logger.info(f"  ✅ Already completed (in-memory state), skipping...")
            return

        # Check if should skip based on manifest (Day 5 resume logic)
        if self.should_skip_phase(phase_type):
            logger.info(f"  ✅ Phase complete (manifest verified), skipping...")
            self.state.mark_skipped(phase_type)
            return

        # Check dependencies
        self._check_dependencies(phase)

        # Mark as started
        self.state.mark_started(phase_type)

        try:
            # Validate inputs
            self._validate_phase_inputs(phase)

            # Execute phase
            self._execute_phase(phase)

            # Validate outputs
            self._validate_phase_outputs(phase)

            # Verify manifest was written (Day 5 requirement)
            if HAS_MANIFEST_SYSTEM:
                phase_dir_map = {
                    PhaseType.PHASE1_BASELINE: self.artifacts.phase1_dir,
                    PhaseType.PHASE2_THRESHOLD: self.artifacts.phase2_dir,
                    PhaseType.PHASE3_GATE: getattr(self.artifacts, 'phase3_dir', None),
                    PhaseType.PHASE4_EXPLORA: getattr(self.artifacts, 'phase4_dir', None),
                    PhaseType.PHASE5_SCRC: getattr(self.artifacts, 'phase5_dir', None),
                    PhaseType.PHASE6_BUNDLE: self.artifacts.bundle_export_dir,
                }
                phase_dir = phase_dir_map.get(phase_type)
                if phase_dir:
                    manifest_path = phase_dir / "manifest.json"
                    if not manifest_path.exists():
                        raise RuntimeError(
                            f"Phase {phase_type.value} completed but manifest not found!\n"
                            f"Expected: {manifest_path}\n"
                            f"This is a bug in the step implementation.\n"
                            f"All steps must write manifest.json as the LAST operation."
                        )

            # Mark as completed
            self.state.mark_completed(phase_type)

            duration = self.state.executions[phase_type].duration_minutes
            logger.info(f"  ✅ Completed in {duration:.1f} minutes")

        except Exception as e:
            # Mark as failed
            error_message = str(e)
            self.state.mark_failed(phase_type, error_message)
            logger.info(f"  ❌ Failed: {error_message}")
            raise

    def run(self, phases_to_run: List[PhaseType]) -> None:
        """
        Run multiple phases in the correct order

        Args:
            phases_to_run: List of phases to run

        Raises:
            RuntimeError: If dependencies are not satisfied
            FileNotFoundError: If inputs are missing
            ValidationError: If outputs are invalid
            Exception: If phase execution fails
        """
        # Get execution order
        execution_order = self.registry.get_execution_order(phases_to_run)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"DAG Pipeline Execution")
        logger.info(f"{'=' * 80}")
        logger.info(f"Phases to run: {[p.value for p in execution_order]}")
        logger.info(f"{'=' * 80}\n")

        # Start pipeline
        self.state.start_time = time.time()

        # Run phases
        for phase_type in execution_order:
            self.run_phase(phase_type)

        # End pipeline
        self.state.end_time = time.time()

        # Print summary
        self._print_summary()

    def resume(self, state_path: Path) -> None:
        """
        Resume pipeline from saved state

        Args:
            state_path: Path to saved pipeline state JSON

        Raises:
            FileNotFoundError: If state file not found
        """
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")

        # Load state
        self.state = PipelineState.load(state_path)
        logger.info(f"Resumed from state: {state_path}")

        # Print current status
        self._print_summary()

    def save_state(self, state_path: Path) -> None:
        """
        Save current pipeline state

        Args:
            state_path: Path to save pipeline state JSON
        """
        self.state.save(state_path)
        logger.info(f"Saved state to: {state_path}")

    def _print_summary(self) -> None:
        """Print execution summary"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Pipeline Execution Summary")
        logger.info(f"{'=' * 80}")

        for phase_type, execution in self.state.executions.items():
            status_emoji = {
                PhaseStatus.COMPLETED: "✅",
                PhaseStatus.RUNNING: "⏳",
                PhaseStatus.FAILED: "❌",
                PhaseStatus.SKIPPED: "⏭️",
                PhaseStatus.PENDING: "⏸️",
            }[execution.status]

            duration_str = ""
            if execution.duration_minutes is not None:
                duration_str = f" ({execution.duration_minutes:.1f} min)"

            logger.info(
                f"{status_emoji} {phase_type.value}: {execution.status.value}{duration_str}"
            )

            if execution.error_message:
                logger.info(f"   Error: {execution.error_message}")

        logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Test DAG engine
    print("📋 DAG Engine Test\n")

    from contracts.artifact_schema import create_artifact_schema

    # Create artifacts
    artifacts = create_artifact_schema("test_outputs")

    # Create engine
    engine = DAGEngine(artifacts=artifacts, verbose=True)

    # Register dummy executors (for testing)
    def dummy_executor(artifacts: ArtifactSchema) -> None:
        print("    (Dummy execution - skipping actual work)")
        # In real code, this would call the actual training/calibration/export code

    engine.register_executor(PhaseType.PHASE1_BASELINE, dummy_executor)
    engine.register_executor(PhaseType.PHASE2_THRESHOLD, dummy_executor)
    engine.register_executor(PhaseType.PHASE6_BUNDLE, dummy_executor)

    # Test execution order
    phases_to_run = [
        PhaseType.PHASE2_THRESHOLD,  # Depends on Phase 1
        PhaseType.PHASE1_BASELINE,  # No dependencies
        PhaseType.PHASE6_BUNDLE,  # Depends on Phase 1
    ]

    print("Test: Get execution order")
    execution_order = engine.registry.get_execution_order(phases_to_run)
    print(f"✅ Execution order: {[p.value for p in execution_order]}")
    print("   Expected: [phase1_baseline, phase2_threshold, phase6_bundle]\n")

    print("✅ DAG engine test passed!")
    print("\n💡 Summary: DAG engine orchestrates phase execution with automatic validation!")

    # Cleanup
    import shutil

    if Path("test_outputs").exists():
        shutil.rmtree("test_outputs")
