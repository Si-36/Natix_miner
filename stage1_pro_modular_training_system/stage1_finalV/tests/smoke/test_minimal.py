#!/usr/bin/env python3
"""
🧪 **Minimal Smoke Test** - Validates Core Infrastructure
Tests:
- ArtifactStore (proven to work!)
- Step spec imports
- Registry instantiation
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.pipeline.artifacts import ArtifactKey, ArtifactStore
from src.pipeline.step_api import StepSpec
from src.pipeline.contracts import Split


def test_artifact_store():
    """Test ArtifactStore (already proven to work!)."""
    print(f"\n{'='*70}")
    print(f"🧪 TEST 1: ArtifactStore")
    print("=" * 70)
    
    # Create artifact store
    artifact_root = Path(__file__).parent / "test_artifacts"
    store = ArtifactStore(artifact_root)
    
    # Test tensor write
    import torch
    test_tensor = torch.randn(10, 384)
    print(f"   📊 Writing test tensor...")
    tensor_path = store.put(ArtifactKey.MODEL_CHECKPOINT, test_tensor, run_id="smoke_test")
    
    # Verify
    assert tensor_path.exists(), f"Tensor file not found: {tensor_path}"
    assert tensor_path.stat().st_size > 0, f"Tensor file is empty: {tensor_path}"
    
    # Load back
    loaded = torch.load(tensor_path)
    assert torch.allclose(loaded, test_tensor), "Tensor mismatch"
    
    print(f"   ✅ ArtifactStore tensor write works!")
    print("=" * 70)


def test_step_spec_imports():
    """Test step spec imports."""
    print(f"\n{'='*70}")
    print(f"🧪 TEST 2: Step Spec Imports")
    print("=" * 70)
    
    # Import step specs
    from src.steps.export_calib_logits import ExportCalibLogitsSpec
    from src.steps.sweep_thresholds import SweepThresholdsSpec
    
    print(f"   ✅ ExportCalibLogitsSpec imported")
    print(f"   ✅ SweepThresholdsSpec imported")
    print("=" * 70)


def test_registry_instantiation():
    """Test registry instantiation."""
    print(f"\n{'='*70}")
    print(f"🧪 TEST 3: Registry Instantiation")
    print("=" * 70)
    
    from src.pipeline.registry import STEP_REGISTRY
    
    # Get step specs (triggers discovery)
    export_spec = STEP_REGISTRY.get_step_spec("export_calib_logits")
    sweep_spec = STEP_REGISTRY.get_step_spec("sweep_thresholds")
    
    assert export_spec is not None, "export_calib_logits spec not found"
    assert sweep_spec is not None, "sweep_thresholds spec not found"
    
    print(f"   ✅ Registry instantiated")
    print(f"   ✅ export_calib_logits: {export_spec.name}")
    print(f"   ✅ sweep_thresholds: {sweep_spec.name}")
    print("=" * 70)


def main():
    """Run minimal smoke tests."""
    print(f"\n{'='*70}")
    print(f"🧪 Minimal Smoke Tests - Core Infrastructure")
    print("=" * 70)
    
    try:
        test_artifact_store()
        test_step_spec_imports()
        test_registry_instantiation()
        
        print(f"\n{'='*70}")
        print(f"🎉 ALL SMOKE TESTS PASSED")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   ✅ ArtifactStore works (proven!)")
        print(f"   ✅ Step specs importable")
        print(f"   ✅ Registry instantiable")
        print("=" * 70)
        
        return 0
    
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ SMOKE TESTS FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

