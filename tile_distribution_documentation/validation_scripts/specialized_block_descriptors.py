#!/usr/bin/env python3
"""
Specialized block descriptor examples validation script.
This demonstrates real-world GPU memory layout patterns.
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
    make_unmerge_transform,
    transform_tensor_descriptor,
    make_tuple,
    number,
    sequence,
    MergeTransform,
    PassThroughTransform,
    UnmergeTransform
)

def k_lds_block_desc_example():
    """Test: K LDS Block Desc - Local Data Store block for K dimension"""
    print("🔗 K LDS Block Descriptor Example")
    print("=" * 50)
    
    # Parameters from examples.py - GPU memory layout parameters
    NumKLdsBuffers = 2
    kNPerBlock = 128
    kKPerBlock = 32
    kKVector = 8
    kKPack = 4
    
    print(f"GPU Memory Layout Parameters:")
    print(f"  NumKLdsBuffers={NumKLdsBuffers} (double buffering)")
    print(f"  kNPerBlock={kNPerBlock} (N dimension per block)")
    print(f"  kKPerBlock={kKPerBlock} (K dimension per block)")
    print(f"  kKVector={kKVector} (vectorization factor)")
    print(f"  kKPack={kKPack} (packing factor)")
    
    # Create input descriptor - represents the raw memory layout
    input_desc = make_naive_tensor_descriptor_packed([
        NumKLdsBuffers, kNPerBlock, kKPerBlock // kKVector, kKVector, kKPack
    ])
    
    print(f"\nInput descriptor (raw memory layout): {input_desc.get_lengths()}")
    print(f"  Dimensions: [buffers, N, K_blocks, K_vector, K_pack]")
    
    # Create transforms - merge related dimensions
    print("\n🔧 Creating specialized transforms:")
    print("  Merge buffers and N dimension (for block-level access)")
    print("  Merge K-related dimensions (K_blocks, K_vector, K_pack)")
    
    transforms = make_tuple(
        make_merge_transform(make_tuple(number(NumKLdsBuffers), number(kNPerBlock))),
        make_merge_transform(make_tuple(
            number(kKPerBlock // kKVector),
            number(kKVector // kKPack),
            number(kKPack)
        ))
    )
    
    # Apply transformation
    result = transform_tensor_descriptor(
        input_desc,
        transforms,
        make_tuple(sequence(0, 1), sequence(2, 3, 4)),
        make_tuple(sequence(0), sequence(1))
    )
    
    print(f"\n✅ Result:")
    print(f"  Input: {input_desc.get_lengths()}")
    print(f"  Output: {result.get_lengths()}")
    
    # Verify dimensions
    expected_lengths = [
        NumKLdsBuffers * kNPerBlock,
        (kKPerBlock // kKVector) * (kKVector // kKPack) * kKPack
    ]
    actual_lengths = result.get_lengths()
    
    print(f"\n🎯 Verification:")
    print(f"  Expected lengths: {expected_lengths}")
    print(f"  Actual lengths: {actual_lengths}")
    print(f"  Match: {expected_lengths == actual_lengths}")
    
    # Verify calculation
    print(f"\n🔍 Calculation Verification:")
    print(f"  Merged N dimension: {NumKLdsBuffers} * {kNPerBlock} = {NumKLdsBuffers * kNPerBlock}")
    print(f"  Merged K dimension: ({kKPerBlock}//{kKVector}) * ({kKVector}//{kKPack}) * {kKPack}")
    print(f"                    = {kKPerBlock // kKVector} * {kKVector // kKPack} * {kKPack}")
    print(f"                    = {(kKPerBlock // kKVector) * (kKVector // kKPack) * kKPack}")
    
    return result, expected_lengths == actual_lengths

def v_lds_block_desc_example():
    """Test: V LDS Block Desc - Local Data Store block for V dimension"""
    print("\n🔗 V LDS Block Descriptor Example")
    print("=" * 50)
    
    # Parameters from examples.py - GPU memory layout parameters
    NumVLdsBuffers = 2
    kNPerBlock = 128
    NPerRow = 4
    kKPerBlock = 32
    kKPack = 4
    
    print(f"GPU Memory Layout Parameters:")
    print(f"  NumVLdsBuffers={NumVLdsBuffers} (double buffering)")
    print(f"  kNPerBlock={kNPerBlock} (N dimension per block)")
    print(f"  NPerRow={NPerRow} (N elements per row)")
    print(f"  kKPerBlock={kKPerBlock} (K dimension per block)")
    print(f"  kKPack={kKPack} (packing factor)")
    
    # Create input descriptor - represents the raw memory layout
    input_desc = make_naive_tensor_descriptor_packed([
        NumVLdsBuffers, kNPerBlock // NPerRow, NPerRow, kKPerBlock, kKPack
    ])
    
    print(f"\nInput descriptor (raw memory layout): {input_desc.get_lengths()}")
    print(f"  Dimensions: [buffers, N_rows, N_per_row, K_blocks, K_pack]")
    
    # Create transforms - merge related dimensions
    print("\n🔧 Creating specialized transforms:")
    print("  Merge all N-related dimensions (buffers, N_rows, N_per_row)")
    print("  Merge K-related dimensions (K_blocks, K_pack)")
    
    transforms = make_tuple(
        make_merge_transform(make_tuple(
            number(NumVLdsBuffers),
            number(kNPerBlock // NPerRow),
            number(NPerRow)
        )),
        make_merge_transform(make_tuple(
            number(kKPerBlock // kKPack),
            number(kKPack)
        ))
    )
    
    # Apply transformation
    result = transform_tensor_descriptor(
        input_desc,
        transforms,
        make_tuple(sequence(0, 1, 2), sequence(3, 4)),
        make_tuple(sequence(0), sequence(1))
    )
    
    print(f"\n✅ Result:")
    print(f"  Input: {input_desc.get_lengths()}")
    print(f"  Output: {result.get_lengths()}")
    
    # Verify dimensions
    expected_lengths = [
        NumVLdsBuffers * (kNPerBlock // NPerRow) * NPerRow,
        (kKPerBlock // kKPack) * kKPack
    ]
    actual_lengths = result.get_lengths()
    
    print(f"\n🎯 Verification:")
    print(f"  Expected lengths: {expected_lengths}")
    print(f"  Actual lengths: {actual_lengths}")
    print(f"  Match: {expected_lengths == actual_lengths}")
    
    # Verify calculation
    print(f"\n🔍 Calculation Verification:")
    print(f"  Merged N dimension: {NumVLdsBuffers} * ({kNPerBlock}//{NPerRow}) * {NPerRow}")
    print(f"                    = {NumVLdsBuffers} * {kNPerBlock // NPerRow} * {NPerRow}")
    print(f"                    = {NumVLdsBuffers * (kNPerBlock // NPerRow) * NPerRow}")
    print(f"  Merged K dimension: ({kKPerBlock}//{kKPack}) * {kKPack}")
    print(f"                    = {kKPerBlock // kKPack} * {kKPack}")
    print(f"                    = {(kKPerBlock // kKPack) * kKPack}")
    
    return result, expected_lengths == actual_lengths

def arithmetic_sequence_transform_example():
    """Test: Arithmetic Sequence Transform - unmerging to multiple dimensions"""
    print("\n🔗 Arithmetic Sequence Transform Example")
    print("=" * 50)
    
    # Parameters from examples.py
    lengths = [2, 4, 8]
    total_length = lengths[0] * lengths[1] * lengths[2]
    
    print(f"Sequence parameters:")
    print(f"  Target lengths: {lengths}")
    print(f"  Total elements: {total_length}")
    
    # Create input descriptor - single flat dimension
    input_desc = make_naive_tensor_descriptor_packed([total_length])
    print(f"\nInput descriptor (flat): {input_desc.get_lengths()}")
    
    # Create unmerge transform - split into multiple dimensions
    print("\n🔧 Creating unmerge transform:")
    print(f"  Split {total_length} elements into {lengths}")
    
    transforms = make_tuple(
        make_unmerge_transform(make_tuple(number(lengths[0]), number(lengths[1]), number(lengths[2])))
    )
    
    # Apply transformation
    result = transform_tensor_descriptor(
        input_desc,
        transforms,
        make_tuple(sequence(0)),
        make_tuple(sequence(0, 1, 2))
    )
    
    print(f"\n✅ Result:")
    print(f"  Input: {input_desc.get_lengths()}")
    print(f"  Output: {result.get_lengths()}")
    
    # Verify dimensions
    expected_lengths = lengths
    actual_lengths = result.get_lengths()
    
    print(f"\n🎯 Verification:")
    print(f"  Expected lengths: {expected_lengths}")
    print(f"  Actual lengths: {actual_lengths}")
    print(f"  Match: {expected_lengths == actual_lengths}")
    
    # Verify calculation
    print(f"\n🔍 Calculation Verification:")
    print(f"  Original: {total_length} elements")
    print(f"  Split into: {lengths[0]} × {lengths[1]} × {lengths[2]} = {lengths[0] * lengths[1] * lengths[2]}")
    print(f"  Conservation: {total_length} = {lengths[0] * lengths[1] * lengths[2]} ✅")
    
    return result, expected_lengths == actual_lengths

def complex_memory_layout_example():
    """Test: Complex Memory Layout - realistic GPU memory access pattern"""
    print("\n🔗 Complex Memory Layout Example")
    print("=" * 50)
    
    # Parameters representing a realistic GPU kernel memory layout
    ThreadsPerBlock = 64
    ElementsPerThread = 4
    VectorSize = 2
    BlocksPerSM = 4
    
    print(f"GPU Memory Layout:")
    print(f"  ThreadsPerBlock={ThreadsPerBlock}")
    print(f"  ElementsPerThread={ElementsPerThread}")
    print(f"  VectorSize={VectorSize}")
    print(f"  BlocksPerSM={BlocksPerSM}")
    
    # Create input descriptor representing memory hierarchy
    input_desc = make_naive_tensor_descriptor_packed([
        BlocksPerSM,
        ThreadsPerBlock,
        ElementsPerThread // VectorSize,
        VectorSize
    ])
    
    print(f"\nInput descriptor (memory hierarchy): {input_desc.get_lengths()}")
    print(f"  Dimensions: [blocks_per_SM, threads_per_block, elements_per_thread, vector_size]")
    
    # Create transforms for different access patterns
    print("\n🔧 Creating memory access transforms:")
    print("  Merge blocks and threads (for global addressing)")
    print("  Merge elements and vectors (for vectorized access)")
    
    transforms = make_tuple(
        make_merge_transform(make_tuple(number(BlocksPerSM), number(ThreadsPerBlock))),
        make_merge_transform(make_tuple(number(ElementsPerThread // VectorSize), number(VectorSize)))
    )
    
    # Apply transformation
    result = transform_tensor_descriptor(
        input_desc,
        transforms,
        make_tuple(sequence(0, 1), sequence(2, 3)),
        make_tuple(sequence(0), sequence(1))
    )
    
    print(f"\n✅ Result:")
    print(f"  Input: {input_desc.get_lengths()}")
    print(f"  Output: {result.get_lengths()}")
    
    # Verify dimensions
    expected_lengths = [
        BlocksPerSM * ThreadsPerBlock,
        (ElementsPerThread // VectorSize) * VectorSize
    ]
    actual_lengths = result.get_lengths()
    
    print(f"\n🎯 Verification:")
    print(f"  Expected lengths: {expected_lengths}")
    print(f"  Actual lengths: {actual_lengths}")
    print(f"  Match: {expected_lengths == actual_lengths}")
    
    # Verify calculation
    print(f"\n🔍 Memory Access Pattern Analysis:")
    print(f"  Global threads: {BlocksPerSM} × {ThreadsPerBlock} = {BlocksPerSM * ThreadsPerBlock}")
    print(f"  Elements per thread: ({ElementsPerThread}//{VectorSize}) × {VectorSize} = {ElementsPerThread}")
    print(f"  Total elements: {BlocksPerSM * ThreadsPerBlock} × {ElementsPerThread} = {BlocksPerSM * ThreadsPerBlock * ElementsPerThread}")
    
    return result, expected_lengths == actual_lengths

def main():
    """Run all specialized block descriptor examples"""
    print("🎊 SPECIALIZED BLOCK DESCRIPTOR EXAMPLES")
    print("=" * 80)
    
    # Run all examples
    results = []
    
    try:
        result1, success1 = k_lds_block_desc_example()
        results.append(("K LDS Block Descriptor", success1))
    except Exception as e:
        print(f"❌ K LDS Block Descriptor failed: {e}")
        results.append(("K LDS Block Descriptor", False))
    
    try:
        result2, success2 = v_lds_block_desc_example()
        results.append(("V LDS Block Descriptor", success2))
    except Exception as e:
        print(f"❌ V LDS Block Descriptor failed: {e}")
        results.append(("V LDS Block Descriptor", False))
    
    try:
        result3, success3 = arithmetic_sequence_transform_example()
        results.append(("Arithmetic Sequence Transform", success3))
    except Exception as e:
        print(f"❌ Arithmetic Sequence Transform failed: {e}")
        results.append(("Arithmetic Sequence Transform", False))
    
    try:
        result4, success4 = complex_memory_layout_example()
        results.append(("Complex Memory Layout", success4))
    except Exception as e:
        print(f"❌ Complex Memory Layout failed: {e}")
        results.append(("Complex Memory Layout", False))
    
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