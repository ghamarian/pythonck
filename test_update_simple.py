#!/usr/bin/env python3

"""
Simple test to verify update methods work correctly.
"""

import sys
sys.path.append('.')

from pytensor.tensor_coordinate import MultiIndex
from pytensor.tensor_descriptor import (
    PassThroughTransform, EmbedTransform, UnmergeTransform, OffsetTransform,
    PadTransform, MergeTransform, XorTransform, ReplicateTransform
)


def test_transform_update_method(transform, test_name, upper_sequences):
    """Test that update method matches calculate method."""
    print(f"\n=== Testing {test_name} ===")
    
    if not hasattr(transform, 'update_lower_index'):
        print(f"  ❌ No update_lower_index method!")
        return False
    
    all_passed = True
    for i in range(len(upper_sequences) - 1):
        current_upper = MultiIndex(len(upper_sequences[i]), upper_sequences[i])
        next_upper = MultiIndex(len(upper_sequences[i+1]), upper_sequences[i+1])
        
        # Calculate using standard method
        current_lower_calc = transform.calculate_lower_index(current_upper)
        next_lower_calc = transform.calculate_lower_index(next_upper)
        
        # Calculate using update method
        upper_diff = MultiIndex(len(current_upper), 
                              [next_upper[j] - current_upper[j] for j in range(len(current_upper))])
        
        try:
            lower_diff, updated_lower = transform.update_lower_index(upper_diff, current_lower_calc, next_upper)
            
            # Compare results
            if list(next_lower_calc) == list(updated_lower):
                print(f"  ✅ Step {i}: {upper_sequences[i]} -> {upper_sequences[i+1]}")
                print(f"     Calc: {list(next_lower_calc)} == Update: {list(updated_lower)}")
            else:
                print(f"  ❌ Step {i}: MISMATCH!")
                print(f"     Calc: {list(next_lower_calc)} != Update: {list(updated_lower)}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ Step {i}: ERROR - {e}")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("🧪 Simple Transform Update Method Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: PassThroughTransform
    total_tests += 1
    if test_transform_update_method(
        PassThroughTransform(16),
        "PassThroughTransform",
        [[0], [1], [5], [10], [15]]
    ):
        tests_passed += 1
    
    # Test 2: OffsetTransform  
    total_tests += 1
    if test_transform_update_method(
        OffsetTransform(16, 5),
        "OffsetTransform",
        [[0], [1], [5], [10], [15]]
    ):
        tests_passed += 1
    
    # Test 3: EmbedTransform
    total_tests += 1
    if test_transform_update_method(
        EmbedTransform([4, 8, 16], [128, 16, 1]),
        "EmbedTransform",
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 2, 5]]
    ):
        tests_passed += 1
    
    # Test 4: UnmergeTransform
    total_tests += 1
    if test_transform_update_method(
        UnmergeTransform([4, 8, 16]),
        "UnmergeTransform", 
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [2, 3, 5]]
    ):
        tests_passed += 1
    
    # Test 5: PadTransform
    total_tests += 1
    if test_transform_update_method(
        PadTransform(8, 2, 3),
        "PadTransform",
        [[0], [2], [5], [9], [12]]
    ):
        tests_passed += 1
    
    # Test 6: MergeTransform
    total_tests += 1
    if test_transform_update_method(
        MergeTransform([4, 8]),
        "MergeTransform",
        [[0], [1], [8], [15], [31]]
    ):
        tests_passed += 1
    
    # Test 7: XorTransform
    total_tests += 1
    if test_transform_update_method(
        XorTransform([16, 8]),
        "XorTransform", 
        [[0, 0], [0, 1], [1, 0], [1, 1], [8, 4], [15, 7]]
    ):
        tests_passed += 1
    
    # Test 8: ReplicateTransform
    total_tests += 1
    if test_transform_update_method(
        ReplicateTransform([4, 8]),
        "ReplicateTransform",
        [[0, 0], [1, 2], [3, 7], [2, 5]]
    ):
        tests_passed += 1
    
    print(f"\n🏁 Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All update methods working correctly!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 