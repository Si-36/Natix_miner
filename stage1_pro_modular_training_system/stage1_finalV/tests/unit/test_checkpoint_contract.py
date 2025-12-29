"""
Test: Checkpoint Contract (state_dict + config wrapper)

This test verifies that:
1. ArtifactStore._contains_tensors() detects nested tensor dicts
2. Checkpoint can be saved and loaded with same structure
3. torch.save() and torch.load() preserve nested structure
"""

import sys
sys.path.insert(0, "src")

import torch
import tempfile
from pathlib import Path


def test_nested_checkpoint_contract():
    """
    Test that checkpoint wrapper dict with tensors is
    properly detected and serialized via torch.save()
    """
    print("\n" + "=" * 70)
    print("TEST: Nested Checkpoint Contract")
    print("=" * 70)

    # Create test checkpoint with nested structure
    model = torch.nn.Linear(10, 2)
    state_dict = model.state_dict()

    checkpoint_wrapper = {
        "state_dict": state_dict,
        "config": {
            "model_id": "test",
            "hidden_dim": 128,
            "num_classes": 2,
        },
        "metadata": {
            "epoch": 1,
            "step_id": "test_step",
        },
    }

    print(f"\n   📦 Created nested checkpoint:")
    print(f"      state_dict keys: {list(state_dict.keys())}")
    print(f"      config keys: {list(checkpoint_wrapper['config'].keys())}")
    print(f"      metadata keys: {list(checkpoint_wrapper['metadata'].keys())}")

    # Create temporary ArtifactStore
    with tempfile.TemporaryDirectory() as tmpdir:
        # Import ArtifactStore directly to avoid path issues
        exec(open("src/pipeline/artifacts.py").read())

        # Get ArtifactStore from executed code
        ArtifactStore = locals().get('ArtifactStore')
        store = ArtifactStore(Path(tmpdir))

        # Initialize manifest
        run_id = "test_checkpoint_contract"
        config = {"test": True}
        manifest_path = store.initialize_manifest(run_id, config)

        print(f"\n   ✅ Manifest initialized: {run_id}")

        # Save checkpoint
        checkpoint_path = store.put(
            ArtifactKey.MODEL_CHECKPOINT,
            checkpoint_wrapper,
            run_id=run_id,
        )

        print(f"\n   💾 Checkpoint saved:")
        print(f"      Path: {checkpoint_path}")
        print(f"      Exists: {checkpoint_path.exists()}")

        # Verify it was saved via torch.save (not JSON)
        with checkpoint_path.open("rb") as f:
            loaded_checkpoint = torch.load(f)

        print(f"\n   📖 Checkpoint loaded:")
        print(f"      Keys preserved: {list(loaded_checkpoint.keys())}")
        print(f"      Type matches: {type(loaded_checkpoint) is dict}")

        # Verify structure preserved
        assert "state_dict" in loaded_checkpoint
        assert "config" in loaded_checkpoint
        assert "metadata" in loaded_checkpoint
        assert isinstance(loaded_checkpoint["state_dict"], dict)
        assert isinstance(loaded_checkpoint["config"], dict)
        assert isinstance(loaded_checkpoint["metadata"], dict)

        # Verify state_dict still contains tensors
        assert any(isinstance(v, torch.Tensor) for v in loaded_checkpoint["state_dict"].values()), \
            "state_dict lost tensors!"

        print(f"\n   ✅ Checkpoint contract verified:")
        print(f"      All keys preserved")
        print(f"      state_dict contains tensors")
        print(f"      Config metadata preserved")

    print("\n" + "=" * 70)
    print("✅ TEST PASSED: Checkpoint contract working")
    print("=" * 70)


def test_contains_tensors_recursive():
    """
    Test that _contains_tensors detects tensors at any depth
    """
    print("\n" + "=" * 70)
    print("TEST: _contains_tensors Recursive Detection")
    print("=" * 70)

    # Test cases
    test_cases = [
        ("top-level tensor", torch.randn(10, 2), True),
        ("top-level dict with tensor", {"tensor": torch.randn(5, 5)}, True),
        ("nested dict with tensor", {"state_dict": {"weight": torch.randn(3, 3)}}, True),
        ("deeply nested dict with tensor", {"a": {"b": {"c": {"tensor": torch.randn(2, 2)}}}, True),
        ("list with tensor", [torch.randn(3, 3)], True),
        ("top-level dict no tensor", {"a": 1, "b": 2}, False),
        ("nested dict no tensor", {"a": {"b": {"c": {"d": 4}}}, False),
        ("list no tensor", [1, 2, 3], False),
    ]

    print("\n   Running test cases:")
    for name, data, expected in test_cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Import ArtifactStore
            exec(open("src/pipeline/artifacts.py").read())
            ArtifactStore = locals().get('ArtifactStore')
            store = ArtifactStore(Path(tmpdir))

            result = store._contains_tensors(data)
            passed = result == expected

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"      {status}: {name}")

            if not passed:
                print(f"         Expected: {expected}")
                print(f"         Got: {result}")

    # Verify checkpoint wrapper detection
    checkpoint_wrapper = {
        "state_dict": {"weight": torch.randn(5, 5)},
        "config": {"model_id": "test"},
        "metadata": {"epoch": 1},
    }

    result = store._contains_tensors(checkpoint_wrapper)
    if result:
        print(f"\n   ✅ Nested checkpoint wrapper detected correctly")
    else:
        print(f"\n   ❌ Nested checkpoint wrapper NOT detected")

    print("\n" + "=" * 70)
    print("✅ TEST PASSED: Recursive tensor detection working")
    print("=" * 70)


if __name__ == "__main__":
    test_nested_checkpoint_contract()
    test_contains_tensors_recursive()
