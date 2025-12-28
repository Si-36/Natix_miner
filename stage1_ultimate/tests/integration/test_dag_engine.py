"""
Integration Tests for DAG Engine

Tests the complete DAG execution pipeline including:
- Phase registration
- Dependency resolution
- Execution order
- State management

Latest 2025-2026 practices:
- Python 3.14+ with pytest
- Integration testing patterns
- Mock executor functions
"""

import pytest
from pathlib import Path
import json

from src.pipeline.dag_engine import DAGEngine
from src.pipeline.phase_spec import (
    PhaseType,
    get_phase_registry,
)
from src.contracts.artifact_schema import create_artifact_schema


class TestDAGEngineBasics:
    """Test basic DAG engine functionality"""

    def test_dag_engine_creation(self, artifacts):
        """DAGEngine should be created successfully"""
        engine = DAGEngine(artifacts=artifacts)

        assert engine is not None
        assert engine.artifacts == artifacts
        assert engine.registry is not None

    def test_dag_engine_register_executor(self, artifacts):
        """Should be able to register executor functions"""
        engine = DAGEngine(artifacts=artifacts)

        def dummy_executor(artifacts):
            return True

        # Register executor
        engine.register_executor(PhaseType.PHASE1_BASELINE, dummy_executor)

        # Verify registration
        assert PhaseType.PHASE1_BASELINE in engine.executors
        assert engine.executors[PhaseType.PHASE1_BASELINE] == dummy_executor

    def test_dag_engine_register_multiple_executors(self, artifacts):
        """Should be able to register multiple executors"""
        engine = DAGEngine(artifacts=artifacts)

        def executor1(artifacts):
            return True

        def executor2(artifacts):
            return True

        engine.register_executor(PhaseType.PHASE1_BASELINE, executor1)
        engine.register_executor(PhaseType.PHASE2_THRESHOLD, executor2)

        assert len(engine.executors) == 2


class TestDAGEngineExecutionOrder:
    """Test execution order determination"""

    def test_execution_order_single_phase(self, artifacts):
        """Execution order for single phase should be correct"""
        engine = DAGEngine(artifacts=artifacts)

        # Phase 1 has no dependencies
        order = engine.registry.get_execution_order([PhaseType.PHASE1_BASELINE])

        assert order == [PhaseType.PHASE1_BASELINE]

    def test_execution_order_with_dependencies(self, artifacts):
        """Execution order should respect dependencies"""
        engine = DAGEngine(artifacts=artifacts)

        # Phase 2 depends on Phase 1
        order = engine.registry.get_execution_order(
            [PhaseType.PHASE2_THRESHOLD, PhaseType.PHASE1_BASELINE]
        )

        # Phase 1 should come before Phase 2
        assert order.index(PhaseType.PHASE1_BASELINE) < order.index(
            PhaseType.PHASE2_THRESHOLD
        )

    def test_execution_order_multiple_dependencies(self, artifacts):
        """Execution order with multiple dependencies should be correct"""
        engine = DAGEngine(artifacts=artifacts)

        # Phase 6 depends on Phase 1
        # Phase 2 depends on Phase 1
        # Phase 3 depends on Phase 1
        order = engine.registry.get_execution_order(
            [
                PhaseType.PHASE6_BUNDLE,
                PhaseType.PHASE2_THRESHOLD,
                PhaseType.PHASE1_BASELINE,
                PhaseType.PHASE3_GATE,
            ]
        )

        # Phase 1 should come first
        assert order[0] == PhaseType.PHASE1_BASELINE

        # Phase 2, 3, 6 should all come after Phase 1
        phase1_idx = order.index(PhaseType.PHASE1_BASELINE)
        assert order.index(PhaseType.PHASE2_THRESHOLD) > phase1_idx
        assert order.index(PhaseType.PHASE3_GATE) > phase1_idx
        assert order.index(PhaseType.PHASE6_BUNDLE) > phase1_idx

    def test_execution_order_independent_phases(self, artifacts):
        """Independent phases can be in any order"""
        engine = DAGEngine(artifacts=artifacts)

        # Phase 1 and Phase 4 are independent
        order = engine.registry.get_execution_order(
            [PhaseType.PHASE1_BASELINE, PhaseType.PHASE4_EXPLORA]
        )

        # Both should be in order (order doesn't matter for independent phases)
        assert len(order) == 2
        assert PhaseType.PHASE1_BASELINE in order
        assert PhaseType.PHASE4_EXPLORA in order


class TestDAGEngineExecution:
    """Test actual phase execution"""

    def test_execute_single_phase(self, artifacts, sample_splits_json):
        """Should be able to execute a single phase"""
        engine = DAGEngine(artifacts=artifacts)

        # Track execution
        executed = []

        def phase1_executor(artifacts):
            executed.append(PhaseType.PHASE1_BASELINE)
            # Create required outputs
            artifacts.phase1_checkpoint.write_text("dummy checkpoint")
            artifacts.val_calib_logits.write_text("dummy logits")
            artifacts.val_calib_labels.write_text("dummy labels")
            artifacts.metrics_csv.write_text("dummy metrics")
            artifacts.config_json.write_text("{}")

        engine.register_executor(PhaseType.PHASE1_BASELINE, phase1_executor)

        # Execute
        engine.run([PhaseType.PHASE1_BASELINE])

        # Verify execution
        assert PhaseType.PHASE1_BASELINE in executed

    def test_execute_multiple_phases_in_order(self, artifacts, sample_splits_json):
        """Should execute multiple phases in correct order"""
        engine = DAGEngine(artifacts=artifacts)

        # Track execution order
        execution_order = []

        def phase1_executor(artifacts):
            execution_order.append(PhaseType.PHASE1_BASELINE)
            # Create required outputs
            artifacts.phase1_checkpoint.write_text("dummy checkpoint")
            artifacts.val_calib_logits.write_text("dummy logits")
            artifacts.val_calib_labels.write_text("dummy labels")
            artifacts.metrics_csv.write_text("dummy metrics")
            artifacts.config_json.write_text("{}")

        def phase2_executor(artifacts):
            execution_order.append(PhaseType.PHASE2_THRESHOLD)
            # Create required outputs
            artifacts.thresholds_json.write_text("{}")

        engine.register_executor(PhaseType.PHASE1_BASELINE, phase1_executor)
        engine.register_executor(PhaseType.PHASE2_THRESHOLD, phase2_executor)

        # Execute Phase 2 and Phase 1 (should run Phase 1 first due to dependency)
        engine.run([PhaseType.PHASE2_THRESHOLD, PhaseType.PHASE1_BASELINE])

        # Verify order: Phase 1 before Phase 2
        assert execution_order == [PhaseType.PHASE1_BASELINE, PhaseType.PHASE2_THRESHOLD]

    def test_execute_missing_executor_raises_error(self, artifacts, sample_splits_json):
        """Executing phase without registered executor should raise error"""
        engine = DAGEngine(artifacts=artifacts)

        # Don't register any executors!

        with pytest.raises(ValueError) as exc_info:
            engine.run([PhaseType.PHASE1_BASELINE])

        assert "No executor registered" in str(exc_info.value)


class TestDAGEngineStateManagement:
    """Test state saving and loading"""

    def test_save_state(self, artifacts, sample_splits_json):
        """Should be able to save pipeline state"""
        engine = DAGEngine(artifacts=artifacts)

        # Execute a phase
        def phase1_executor(artifacts):
            artifacts.phase1_checkpoint.write_text("dummy")
            artifacts.val_calib_logits.write_text("dummy")
            artifacts.val_calib_labels.write_text("dummy")
            artifacts.metrics_csv.write_text("dummy")
            artifacts.config_json.write_text("{}")

        engine.register_executor(PhaseType.PHASE1_BASELINE, phase1_executor)
        engine.run([PhaseType.PHASE1_BASELINE])

        # Save state
        state_path = artifacts.output_dir / "test_state.json"
        engine.save_state(state_path)

        # Verify state file exists
        assert state_path.exists()

        # Verify state content
        with open(state_path) as f:
            state = json.load(f)

        assert "completed_phases" in state
        assert PhaseType.PHASE1_BASELINE.value in state["completed_phases"]

    def test_load_state(self, artifacts):
        """Should be able to load pipeline state"""
        engine = DAGEngine(artifacts=artifacts)

        # Create a state file
        state_path = artifacts.output_dir / "test_state.json"
        state = {
            "completed_phases": [PhaseType.PHASE1_BASELINE.value],
            "timestamp": "2025-12-28T00:00:00Z",
        }
        state_path.write_text(json.dumps(state))

        # Load state
        engine.load_state(state_path)

        # Verify state was loaded
        assert PhaseType.PHASE1_BASELINE in engine.completed_phases


class TestDAGEngineValidation:
    """Test input/output validation"""

    def test_validate_inputs_missing_file(self, artifacts):
        """Phase with missing inputs should raise error"""
        engine = DAGEngine(artifacts=artifacts)

        # Phase 2 requires val_calib_logits and val_calib_labels
        # Don't create these files!

        def phase2_executor(artifacts):
            pass

        engine.register_executor(PhaseType.PHASE2_THRESHOLD, phase2_executor)

        with pytest.raises(FileNotFoundError) as exc_info:
            engine.run([PhaseType.PHASE2_THRESHOLD])

        assert "missing required inputs" in str(exc_info.value).lower()

    def test_validate_outputs_missing_file(self, artifacts, sample_splits_json):
        """Phase that doesn't create outputs should raise error"""
        engine = DAGEngine(artifacts=artifacts)

        # Executor that doesn't create required outputs
        def phase1_executor(artifacts):
            # Don't create any outputs!
            pass

        engine.register_executor(PhaseType.PHASE1_BASELINE, phase1_executor)

        with pytest.raises(FileNotFoundError) as exc_info:
            engine.run([PhaseType.PHASE1_BASELINE])

        assert "missing expected outputs" in str(exc_info.value).lower()


class TestPhaseRegistry:
    """Test phase registry functionality"""

    def test_get_phase_registry(self):
        """Should be able to get phase registry"""
        registry = get_phase_registry()

        assert registry is not None
        assert len(registry.phases) == 6  # 6 phases total

    def test_get_phase(self):
        """Should be able to get individual phase specs"""
        registry = get_phase_registry()

        phase1 = registry.get_phase(PhaseType.PHASE1_BASELINE)

        assert phase1 is not None
        assert phase1.name == "Phase 1: Baseline Training"

    def test_get_all_phases(self):
        """Should be able to get all phase specs"""
        registry = get_phase_registry()

        all_phases = registry.get_all_phases()

        assert len(all_phases) == 6
        assert all([phase.phase_type in PhaseType for phase in all_phases])

    def test_get_dependencies(self):
        """Should be able to get phase dependencies"""
        registry = get_phase_registry()

        # Phase 1 has no dependencies
        phase1_deps = registry.get_dependencies(PhaseType.PHASE1_BASELINE)
        assert phase1_deps == []

        # Phase 2 depends on Phase 1
        phase2_deps = registry.get_dependencies(PhaseType.PHASE2_THRESHOLD)
        assert PhaseType.PHASE1_BASELINE in phase2_deps
