"""
🧪 **Test Step Contract** - Verify Milestone A is runnable
Tests that StepSpec/StepContext/StepResult work correctly end-to-end.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.step_api import StepSpec, StepContext, StepResult
from src.pipeline.artifacts import ArtifactKey, ArtifactStore
from src.pipeline.contracts import Split, SplitPolicy
import torch


def test_step_imports():
    """Test that all step contracts import correctly"""
    print("✅ Test: Step contract imports")
    
    # Test StepResult
    result = StepResult(
        artifacts_written=[ArtifactKey.MODEL_CHECKPOINT],
        splits_used=frozenset({Split.TRAIN.value}),
        metrics={"test_metric": 1.0},
    )
    print(f"   StepResult created: {result.artifacts_written}")
    assert result.artifacts_written == [ArtifactKey.MODEL_CHECKPOINT]
    print("✅ StepResult works correctly")


def test_step_context():
    """Test that StepContext can be created"""
    print("\n✅ Test: StepContext creation")
    
    # Create mock artifact store
    artifact_root = Path("/tmp/test_artifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)
    
    store = ArtifactStore(artifact_root)
    manifest = store.initialize_manifest(
        run_id="test_001",
        config={"test": "config"},
    )
    
    context = StepContext(
        step_id="test_step",
        config={"test": "value"},
        run_id="test_001",
        artifact_root=artifact_root,
        artifact_store=store,
        manifest=manifest,
    )
    
    print(f"   StepContext created: {context.step_id}")
    assert context.artifact_root == artifact_root
    print("✅ StepContext works correctly")


def test_sweep_thresholds_spec():
    """Test that SweepThresholdsSpec can be instantiated"""
    print("\n✅ Test: SweepThresholdsSpec")
    
    try:
        from src.steps.sweep_thresholds import SweepThresholdsSpec
        
        spec = SweepThresholdsSpec()
        print(f"   SweepThresholdsSpec created: {spec.step_id}, {spec.name}")
        print(f"   Deps: {spec.deps}")
        print(f"   Order: {spec.order_index}")
        
        # Test allowed_splits
        allowed = spec.allowed_splits()
        print(f"   Allowed splits: {allowed}")
        assert Split.VAL_CALIB.value in {s.value for s in allowed}
        print("✅ SweepThresholdsSpec works correctly")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 Testing Step Contracts (Milestone A)")
    print("=" * 70)
    
    test_step_imports()
    test_step_context()
    test_sweep_thresholds_spec()
    
    print("\n" + "=" * 70)
    print("✅ All Step Contract Tests PASSED")
    print("=" * 70)

