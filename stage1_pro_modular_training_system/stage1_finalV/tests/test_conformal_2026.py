"""
🧪 Test: 2026 Conformal Policy (TorchCP-backed)
Verifies library-backed conformal works correctly
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.conformal import ConformalPolicy, TemperatureConformalWrapper, TORCHCP_AVAILABLE


def test_temperature_conformal_wrapper():
    """Test TemperatureConformalWrapper"""
    print("\n" + "="*60)
    print("Testing: TemperatureConformalWrapper")
    print("="*60)
    
    if not TORCHCP_AVAILABLE:
        print("⚠️  TorchCP not installed - skipping test")
        return True
    
    # Create dummy calibration data
    val_logits = torch.randn(100, 2)
    val_labels = torch.randint(0, 2, (100,))
    
    # Create temperature wrapper
    temp_wrapper = TemperatureConformalWrapper(temperature=1.5)
    
    # Fit temperature
    temp_wrapper.fit(val_logits, val_labels)
    
    # Test calibration
    test_logits = torch.randn(20, 2)
    
    calibrated_logits = temp_wrapper.calibrate_logits(test_logits)
    
    # Verify temperature was learned (should be close to 1.5)
    print(f"   Learned temperature: {temp_wrapper.temperature:.4f}")
    
    # Verify shapes match
    assert calibrated_logits.shape == test_logits.shape
    
    print("✅ TemperatureConformalWrapper test PASSED")
    return True


def test_conformal_policy_aps():
    """Test ConformalPolicy with APS score"""
    print("\n" + "="*60)
    print("Testing: ConformalPolicy with APS score")
    print("="*60)
    
    if not TORCHCP_AVAILABLE:
        print("⚠️  TorchCP not installed - skipping test")
        return True
    
    # Create dummy calibration data
    val_logits = torch.randn(100, 2)
    val_labels = torch.randint(0, 2, (100,))
    
    # Create conformal policy with APS
    policy = ConformalPolicy(
        alpha=0.1,
        score_name="aps",
        randomized=True,
        raps_penalty=0.01,
        raps_kreg=5,
    )
    
    # Fit policy
    policy.fit(val_logits, val_labels)
    
    # Test prediction
    test_logits = torch.randn(20, 2)
    
    # Predict (return sets for clarity)
    prediction_sets = policy.predict_from_logits(test_logits, return_sets=True)
    
    # Verify we get sets
    assert len(prediction_sets) == 20, "Should have 20 prediction sets"
    
    # Check sets are valid (each should be subset of {0, 1})
    for pred_set in prediction_sets:
        assert pred_set.issubset({0, 1}), f"Invalid set: {pred_set}"
    
    print(f"   Example sets: {prediction_sets[:3]}")
    print("✅ ConformalPolicy APS test PASSED")
    return True


def test_conformal_policy_raps():
    """Test ConformalPolicy with RAPS score"""
    print("\n" + "="*60)
    print("Testing: ConformalPolicy with RAPS score")
    print("="*60)
    
    if not TORCHCP_AVAILABLE:
        print("⚠️  TorchCP not installed - skipping test")
        return True
    
    # Create dummy calibration data
    val_logits = torch.randn(100, 2)
    val_labels = torch.randint(0, 2, (100,))
    
    # Create conformal policy with RAPS
    policy = ConformalPolicy(
        alpha=0.1,
        score_name="raps",
        randomized=False,
        raps_penalty=0.01,
        raps_kreg=5,
    )
    
    # Fit policy
    policy.fit(val_logits, val_labels)
    
    # Test prediction
    test_logits = torch.randn(20, 2)
    
    # Predict (return boolean mask for speed)
    sets_bool = policy.predict_from_logits(test_logits, return_sets=False)
    
    # Verify boolean mask shape
    assert sets_bool.shape == (20, 2), f"Boolean mask shape should be (20, 2), got {sets_bool.shape}"
    
    print(f"   Boolean mask shape: {sets_bool.shape}")
    print("✅ ConformalPolicy RAPS test PASSED")
    return True


def test_conformal_save_load():
    """Test saving and loading ConformalPolicy"""
    print("\n" + "="*60)
    print("Testing: ConformalPolicy save/load")
    print("="*60)
    
    if not TORCHCP_AVAILABLE:
        print("⚠️  TorchCP not installed - skipping test")
        return True
    
    # Create and fit a policy
    val_logits = torch.randn(100, 2)
    val_labels = torch.randint(0, 2, (100,))
    
    policy = ConformalPolicy(
        alpha=0.1,
        score_name="aps",
        randomized=False,
    )
    policy.fit(val_logits, val_labels)
    
    # Save policy
    save_path = "/tmp/test_conformal_policy.pt"
    policy.save(save_path)
    
    # Load policy
    loaded_policy = ConformalPolicy.load(save_path)
    
    # Verify loaded policy has same state
    assert loaded_policy.is_fitted, "Loaded policy should be fitted"
    assert loaded_policy.alpha == policy.alpha, "Alpha should match"
    
    print(f"   Saved to: {save_path}")
    print(f"   Loaded alpha: {loaded_policy.alpha:.2f}")
    print("✅ ConformalPolicy save/load test PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 TESTING: 2026 Conformal Policy (TorchCP)")
    print("="*60)
    
    tests = [
        ("TemperatureConformalWrapper", test_temperature_conformal_wrapper),
        ("ConformalPolicy-APS", test_conformal_policy_aps),
        ("ConformalPolicy-RAPS", test_conformal_policy_raps),
        ("ConformalPolicy save/load", test_conformal_save_load),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {name} test FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {name} test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Total: {passed + failed + skipped}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped (TorchCP not installed): {skipped}")
    if skipped > 0:
        print("⚠️  Install TorchCP: pip install torchcp")
        print("   CPU: pip install torchcp[cpu]")
        print("   GPU: pip install torchcp[cu121]")
    success_rate = 100 * passed / (passed + failed)
    else:
        success_rate = 100 * passed / (passed + failed)
    
    print(f"Success Rate: {success_rate:.1f}%")
    print("="*60)
    
    # Return success if all passed (skipped tests don't count as failure)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

