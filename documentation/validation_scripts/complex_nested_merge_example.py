#!/usr/bin/env python3
"""
Complex nested merge example validation script.
This demonstrates the flattening of nested transforms.
"""

import sys
import os
# Handle both direct execution and exec() from documentation
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
except NameError:
    sys.path.insert(0, os.path.join(os.getcwd(), '..'))  # For exec() from documentation

from pytensor.tensor_descriptor import (
    make_naive_tensor_descriptor_packed,
    make_merge_transform,
    make_pass_through_transform,
    transform_tensor_descriptor,
    make_tuple,
    number,
    sequence,
    MergeTransform,
    PassThroughTransform,
    UnmergeTransform
)

def complex_nested_merge_example():
    """Test: Complex Nested Merge example - demonstrates transform flattening"""
    print("🔗 Complex Nested Merge Example")
    print("=" * 50)
    
    # Parameters from examples.py
    A, B, C, D = 3, 2, 4, 5
    
    print(f"Transform dimensions: A={A}, B={B}, C={C}, D={D}")
    print("Goal: Flatten nested merge structure")
    
    # Create input descriptor
    input_desc = make_naive_tensor_descriptor_packed([A, B*C, D, 1])
    print(f"Input descriptor: {input_desc.get_lengths()}")
    
    # Create nested transforms - this is the key test case
    print("\n🔧 Creating nested merge transforms:")
    print("  Inner merge: merge(B, C)")
    print("  Outer merge: merge(pass_through(A), inner_merge)")
    print("  Pass-through: pass_through(D)")
    
    transforms = make_tuple(
        make_merge_transform(
            make_tuple(
                make_pass_through_transform(number(A)),
                make_merge_transform(make_tuple(number(B), number(C)))
            )
        ),
        make_pass_through_transform(number(D))
    )
    
    # Apply transformation
    result = transform_tensor_descriptor(
        input_desc,
        transforms,
        make_tuple(sequence(0, 1, 2), sequence(3)),
        make_tuple(sequence(0), sequence(1))
    )
    
    print(f"\n✅ Result:")
    print(f"  Input: {input_desc.get_lengths()}")
    print(f"  Output: {result.get_lengths()}")
    print(f"  Expected: [{A * B * C}, {D}] = [{A * B * C}, {D}]")
    
    # Verify the result has the expected dimensions
    expected_lengths = [A * B * C, D]
    actual_lengths = result.get_lengths()
    
    print(f"\n🎯 Verification:")
    print(f"  Expected lengths: {expected_lengths}")
    print(f"  Actual lengths: {actual_lengths}")
    print(f"  Match: {expected_lengths == actual_lengths}")
    
    # Verify we have the expected number of transforms (original + new)
    print(f"\n🔍 Transform Analysis:")
    print(f"  Number of transforms: {len(result.transforms)}")
    print(f"  Transform types: {[type(t).__name__ for t in result.transforms]}")
    
    # Check that the nested merge was flattened properly
    merge_found = False
    for i, transform in enumerate(result.transforms):
        if isinstance(transform, MergeTransform):
            print(f"  Transform {i}: MergeTransform with lengths {transform.lengths}")
            # Check if this merge has the expected flattened structure
            if transform.lengths == [A, B * C]:
                merge_found = True
                print(f"    ✅ Found flattened merge: A={A}, B*C={B*C}")
    
    if not merge_found:
        print(f"    ⚠️  Nested structure may not have been flattened as expected")
    
    return result, expected_lengths == actual_lengths

def realistic_multi_descriptor_example():
    """Test: Realistic Multi-Descriptor Example with XOR transforms"""
    print("\n🔗 Realistic Multi-Descriptor Example")
    print("=" * 50)
    
    # Parameters from examples.py
    KThreadWrite = 8
    kfold = 2
    KThreadReadPerm = 4
    K0PerThreadWrite = 2
    N1 = 2
    N0 = 8
    npair = 2
    BK1 = 1
    
    print(f"Parameters:")
    print(f"  KThreadWrite={KThreadWrite}, kfold={kfold}, KThreadReadPerm={KThreadReadPerm}")
    print(f"  K0PerThreadWrite={K0PerThreadWrite}, N1={N1}, N0={N0}, npair={npair}, BK1={BK1}")
    
    # Create the base descriptor
    base_desc = make_naive_tensor_descriptor_packed([
        KThreadWrite // kfold // KThreadReadPerm,
        K0PerThreadWrite,
        KThreadReadPerm * N1,
        kfold * N0 // npair,
        npair,
        BK1
    ])
    
    print(f"\nBase descriptor: {base_desc.get_lengths()}")
    
    # Create complex transforms
    print("\n🔧 Creating complex transformation pipeline:")
    print("  Multiple pass-through transforms")
    print("  One merge transform combining dimensions")
    
    transforms = make_tuple(
        make_pass_through_transform(number(KThreadWrite // kfold // KThreadReadPerm)),
        make_pass_through_transform(number(K0PerThreadWrite)),
        make_merge_transform(make_tuple(
            number(KThreadReadPerm * N1),
            number(kfold * N0 // npair)
        )),
        make_pass_through_transform(number(npair)),
        make_pass_through_transform(number(BK1))
    )
    
    # Apply transformation
    result = transform_tensor_descriptor(
        base_desc,
        transforms,
        make_tuple(
            sequence(0), sequence(1), sequence(2, 3), sequence(4), sequence(5)
        ),
        make_tuple(
            sequence(0), sequence(1), sequence(2), sequence(3), sequence(4)
        )
    )
    
    print(f"\n✅ Result:")
    print(f"  Input: {base_desc.get_lengths()}")
    print(f"  Output: {result.get_lengths()}")
    
    # Verify dimensions
    expected_lengths = [
        KThreadWrite // kfold // KThreadReadPerm,
        K0PerThreadWrite,
        (KThreadReadPerm * N1) * (kfold * N0 // npair),
        npair,
        BK1
    ]
    actual_lengths = result.get_lengths()
    
    print(f"\n🎯 Verification:")
    print(f"  Expected lengths: {expected_lengths}")
    print(f"  Actual lengths: {actual_lengths}")
    print(f"  Match: {expected_lengths == actual_lengths}")
    
    # Verify we have the expected number of transforms
    print(f"\n🔍 Transform Analysis:")
    print(f"  Number of transforms: {len(result.transforms)}")
    print(f"  Transform types: {[type(t).__name__ for t in result.transforms]}")
    
    return result, expected_lengths == actual_lengths

def mixed_transform_types_example():
    """Test: Mixed Transform Types - combining pass-through and merge"""
    print("\n🔗 Mixed Transform Types Example")
    print("=" * 50)
    
    A, B, C, D = 2, 3, 4, 5
    print(f"Dimensions: A={A}, B={B}, C={C}, D={D}")
    
    # Create input descriptor
    input_desc = make_naive_tensor_descriptor_packed([A, B, C, D])
    print(f"Input descriptor: {input_desc.get_lengths()}")
    
    # Create mixed transforms
    print("\n🔧 Creating mixed transforms:")
    print("  Pass-through A")
    print("  Merge B and C")
    print("  Pass-through D")
    
    transforms = make_tuple(
        make_pass_through_transform(number(A)),
        make_merge_transform(make_tuple(number(B), number(C))),
        make_pass_through_transform(number(D))
    )
    
    # Apply transformation
    result = transform_tensor_descriptor(
        input_desc,
        transforms,
        make_tuple(sequence(0), sequence(1, 2), sequence(3)),
        make_tuple(sequence(0), sequence(1), sequence(2))
    )
    
    print(f"\n✅ Result:")
    print(f"  Input: {input_desc.get_lengths()}")
    print(f"  Output: {result.get_lengths()}")
    
    # Verify dimensions
    expected_lengths = [A, B * C, D]
    actual_lengths = result.get_lengths()
    
    print(f"\n🎯 Verification:")
    print(f"  Expected lengths: {expected_lengths}")
    print(f"  Actual lengths: {actual_lengths}")
    print(f"  Match: {expected_lengths == actual_lengths}")
    
    # Verify transform structure
    print(f"\n🔍 Transform Analysis:")
    print(f"  Number of transforms: {len(result.transforms)}")
    print(f"  Transform types: {[type(t).__name__ for t in result.transforms]}")
    
    # Check specific transforms
    expected_transforms = [UnmergeTransform, PassThroughTransform, MergeTransform, PassThroughTransform]
    actual_transforms = [type(t) for t in result.transforms]
    
    print(f"  Expected transform types: {[t.__name__ for t in expected_transforms]}")
    print(f"  Actual transform types: {[t.__name__ for t in actual_transforms]}")
    
    return result, expected_lengths == actual_lengths

def main():
    """Run all complex examples"""
    print("🎊 COMPLEX NESTED MERGE EXAMPLES")
    print("=" * 80)
    
    # Run all examples
    results = []
    
    try:
        result1, success1 = complex_nested_merge_example()
        results.append(("Complex Nested Merge", success1))
    except Exception as e:
        print(f"❌ Complex Nested Merge failed: {e}")
        results.append(("Complex Nested Merge", False))
    
    try:
        result2, success2 = realistic_multi_descriptor_example()
        results.append(("Realistic Multi-Descriptor", success2))
    except Exception as e:
        print(f"❌ Realistic Multi-Descriptor failed: {e}")
        results.append(("Realistic Multi-Descriptor", False))
    
    try:
        result3, success3 = mixed_transform_types_example()
        results.append(("Mixed Transform Types", success3))
    except Exception as e:
        print(f"❌ Mixed Transform Types failed: {e}")
        results.append(("Mixed Transform Types", False))
    
    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 50)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    print(f"\n🎊 All tests passed: {all_passed}")
    
    return all_passed

if __name__ == "__main__":
    main() 