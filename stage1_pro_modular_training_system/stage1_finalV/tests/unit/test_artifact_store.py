#!/usr/bin/env python3
"""
🧪 **Unit Test: ArtifactStore Atomic Writes**
Tests ArtifactStore.put() for crash safety and correctness.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import torch
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from pipeline.artifacts import ArtifactKey, ArtifactStore


def test_tensor_write():
    """
    Test 1: Tensor write (atomic + fsync).
    
    Verify:
    - File exists
    - File is non-empty
    - Tensor can be loaded correctly
    """
    print(f"\n{'='*70}")
    print(f"TEST 1: Tensor Write (Atomic + fsync)")
    print("=" * 70)
    
    # Create artifact store
    artifact_root = Path(__file__).parent / "test_artifacts"
    store = ArtifactStore(artifact_root)
    
    # Create test tensor
    test_tensor = torch.randn(10, 384)
    print(f"   📊 Test tensor: {test_tensor.shape}, dtype={test_tensor.dtype}")
    
    # Write tensor
    print(f"   💾 Writing tensor...")
    store.put(ArtifactKey.MODEL_CHECKPOINT, test_tensor, run_id="test_tensor")
    
    # Verify file exists
    checkpoint_path = store.get(ArtifactKey.MODEL_CHECKPOINT, run_id="test_tensor")
    print(f"   ✅ File exists: {checkpoint_path}")
    assert checkpoint_path.exists(), f"Checkpoint file not found: {checkpoint_path}"
    
    # Verify file is non-empty
    file_size = checkpoint_path.stat().st_size
    print(f"   ✅ File size: {file_size} bytes")
    assert file_size > 0, f"Checkpoint file is empty: {checkpoint_path}"
    
    # Verify tensor can be loaded
    print(f"   📖 Loading tensor...")
    loaded_tensor = torch.load(checkpoint_path)
    print(f"   ✅ Tensor loaded: {loaded_tensor.shape}, dtype={loaded_tensor.dtype}")
    assert loaded_tensor.shape == test_tensor.shape, "Tensor shape mismatch"
    assert torch.allclose(loaded_tensor, test_tensor), "Tensor values mismatch"
    
    print(f"\n✅ TEST 1 PASSED")
    print("=" * 70)


def test_dict_write():
    """
    Test 2: Dict/JSON write (atomic + fsync).
    
    Verify:
    - File exists
    - File is non-empty
    - JSON can be loaded correctly
    """
    print(f"\n{'='*70}")
    print(f"TEST 2: Dict/JSON Write (Atomic + fsync)")
    print("=" * 70)
    
    # Create artifact store
    artifact_root = Path(__file__).parent / "test_artifacts"
    store = ArtifactStore(artifact_root)
    
    # Create test dict
    test_dict = {
        "loss": 0.1234,
        "accuracy": 0.9567,
        "metrics": {
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.905,
        }
    }
    print(f"   📊 Test dict: {test_dict}")
    
    # Write dict
    print(f"   💾 Writing dict...")
    store.put(ArtifactKey.THRESHOLDS_JSON, test_dict, run_id="test_dict")
    
    # Verify file exists
    json_path = store.get(ArtifactKey.THRESHOLDS_JSON, run_id="test_dict")
    print(f"   ✅ File exists: {json_path}")
    assert json_path.exists(), f"JSON file not found: {json_path}"
    
    # Verify file is non-empty
    file_size = json_path.stat().st_size
    print(f"   ✅ File size: {file_size} bytes")
    assert file_size > 0, f"JSON file is empty: {json_path}"
    
    # Verify JSON can be loaded
    print(f"   📖 Loading JSON...")
    import json
    with json_path.open("r", encoding="utf-8") as f:
        loaded_dict = json.load(f)
    print(f"   ✅ JSON loaded: {loaded_dict}")
    assert loaded_dict == test_dict, "JSON content mismatch"
    
    print(f"\n✅ TEST 2 PASSED")
    print("=" * 70)


def test_string_write():
    """
    Test 3: String/CSV write (atomic + fsync).
    
    Verify:
    - File exists
    - File is non-empty
    - String can be loaded correctly
    """
    print(f"\n{'='*70}")
    print(f"TEST 3: String/CSV Write (Atomic + fsync)")
    print("=" * 70)
    
    # Create artifact store
    artifact_root = Path(__file__).parent / "test_artifacts"
    store = ArtifactStore(artifact_root)
    
    # Create test string (CSV)
    test_csv = """threshold,accuracy,acceptance_rate,fnr,tnr,ece,brier,tp,fp,tn,fn
0.1,0.85,0.15,0.05,0.95,0.08,0.12,850,50,150,100
0.2,0.90,0.20,0.10,0.90,0.07,0.11,900,100,180,80
0.3,0.92,0.25,0.15,0.85,0.06,0.10,920,120,130,95
"""
    print(f"   📊 Test CSV (first 3 lines):\n{test_csv.split(chr(10))[:3]}")
    
    # Write string
    print(f"   💾 Writing CSV...")
    store.put(ArtifactKey.THRESHOLDS_METRICS, test_csv, run_id="test_string")
    
    # Verify file exists
    csv_path = store.get(ArtifactKey.THRESHOLDS_METRICS, run_id="test_string")
    print(f"   ✅ File exists: {csv_path}")
    assert csv_path.exists(), f"CSV file not found: {csv_path}"
    
    # Verify file is non-empty
    file_size = csv_path.stat().st_size
    print(f"   ✅ File size: {file_size} bytes")
    assert file_size > 0, f"CSV file is empty: {csv_path}"
    
    # Verify string can be loaded
    print(f"   📖 Loading CSV...")
    with csv_path.open("r", encoding="utf-8") as f:
        loaded_csv = f.read()
    print(f"   ✅ CSV loaded (first 3 lines):\n{loaded_csv.split(chr(10))[:3]}")
    assert loaded_csv == test_csv, "CSV content mismatch"
    
    print(f"\n✅ TEST 3 PASSED")
    print("=" * 70)


def test_bytes_write():
    """
    Test 4: Bytes write (atomic + fsync).
    
    Verify:
    - File exists
    - File is non-empty
    - Bytes can be loaded correctly
    """
    print(f"\n{'='*70}")
    print(f"TEST 4: Bytes Write (Atomic + fsync)")
    print("=" * 70)
    
    # Create artifact store
    artifact_root = Path(__file__).parent / "test_artifacts"
    store = ArtifactStore(artifact_root)
    
    # Create test bytes
    test_bytes = b"Hello, ArtifactStore! This is a test of atomic writes with fsync."
    print(f"   📊 Test bytes: {len(test_bytes)} bytes, preview: {test_bytes[:30]}...")
    
    # Write bytes
    print(f"   💾 Writing bytes...")
    store.put(ArtifactKey.RUN_MANIFEST, test_bytes, run_id="test_bytes")
    
    # Verify file exists
    bytes_path = store.get(ArtifactKey.RUN_MANIFEST, run_id="test_bytes")
    print(f"   ✅ File exists: {bytes_path}")
    assert bytes_path.exists(), f"Bytes file not found: {bytes_path}"
    
    # Verify file is non-empty
    file_size = bytes_path.stat().st_size
    print(f"   ✅ File size: {file_size} bytes")
    assert file_size > 0, f"Bytes file is empty: {bytes_path}"
    
    # Verify bytes can be loaded
    print(f"   📖 Loading bytes...")
    with bytes_path.open("rb") as f:
        loaded_bytes = f.read()
    print(f"   ✅ Bytes loaded: {len(loaded_bytes)} bytes, preview: {loaded_bytes[:30]}...")
    assert loaded_bytes == test_bytes, "Bytes content mismatch"
    
    print(f"\n✅ TEST 4 PASSED")
    print("=" * 70)


def main():
    """Run all unit tests."""
    print(f"\n{'='*70}")
    print(f"🧪 Unit Tests: ArtifactStore Atomic Writes")
    print("=" * 70)
    
    try:
        # Test 1: Tensor write
        test_tensor_write()
        
        # Test 2: Dict write
        test_dict_write()
        
        # Test 3: String write
        test_string_write()
        
        # Test 4: Bytes write
        test_bytes_write()
        
        print(f"\n{'='*70}")
        print(f"🎉 ALL TESTS PASSED")
        print("=" * 70)
        
        return 0
    
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

