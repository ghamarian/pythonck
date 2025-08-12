#!/usr/bin/env python3
"""
Purpose: Demonstrate pytensor adaptor concepts - the transformation engine that chains transforms.

Shows how TensorAdaptor chains individual transforms together to create complex
coordinate transformations, and how to use built-in adaptors like transpose.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code_examples'))

from common_utils import *
from pytensor.tensor_adaptor import (
    TensorAdaptor,
    make_transpose_adaptor,
    make_identity_adaptor
)
from pytensor.tensor_descriptor import (
    EmbedTransform,
    UnmergeTransform,
    MergeTransform,
    PassThroughTransform
)
from pytensor.tensor_coordinate import MultiIndex
import numpy as np

def demonstrate_identity_adaptor():
    """Show the simplest adaptor: identity transformation."""
    print_step(1, "Identity Adaptor: No transformation")
    
    # Create identity adaptor for 2D tensor
    adaptor = make_identity_adaptor(2)
    show_adaptor_info(adaptor, "Identity adaptor")
    
    # Test coordinate transformation (should be unchanged)
    test_coords = [[0, 0], [1, 2], [2, 3]]
    print("  Coordinate transformations (identity):")
    for coord_list in test_coords:
        upper_coord = MultiIndex(2, coord_list)
        lower_coord = adaptor.calculate_bottom_index(upper_coord)
        print(f"    {coord_list} → {lower_coord.to_list()} (no change)")
    
    return adaptor

def demonstrate_transpose_adaptor():
    """Show transpose adaptor for matrix operations."""
    print_step(2, "Transpose Adaptor: Matrix transposition")
    
    # Create transpose adaptor for 2D tensor (swap dimensions)
    adaptor = make_transpose_adaptor(2, [1, 0])  # Swap axes 0 and 1
    show_adaptor_info(adaptor, "Transpose adaptor")
    
    # Test coordinate transformation (dimensions should be swapped)
    test_coords = [[0, 0], [1, 2], [2, 3]]
    print("  Coordinate transformations (transpose):")
    for coord_list in test_coords:
        upper_coord = MultiIndex(2, coord_list)
        lower_coord = adaptor.calculate_bottom_index(upper_coord)
        expected = [coord_list[1], coord_list[0]]  # Swapped
        print(f"    {coord_list} → {lower_coord.to_list()} (expected {expected})")
    
    return adaptor

def demonstrate_custom_adaptor():
    """Show how to create custom adaptors with multiple transforms."""
    print_step(3, "Custom Adaptor: Simple 2D passthrough")
    
    # Create simple 2D passthrough adaptor using separate transforms for each dimension
    pass_dim0 = PassThroughTransform(3)  # First dimension, length 3
    pass_dim1 = PassThroughTransform(4)  # Second dimension, length 4
    
    # Chain them together in an adaptor
    adaptor = TensorAdaptor(
        transforms=[pass_dim0, pass_dim1],
        lower_dimension_hidden_idss=[[0], [1]],     # Each transform outputs to its own dimension
        upper_dimension_hidden_idss=[[2], [3]],     # Each transform takes input from top dimensions
        bottom_dimension_hidden_ids=[0, 1],         # Final output combines both dimensions
        top_dimension_hidden_ids=[2, 3]             # Top input is 2D
    )
    
    show_adaptor_info(adaptor, "Custom 2D passthrough adaptor")
    
    # Test the transformation chain
    test_coords = [[0, 0], [1, 2], [2, 3]]
    print("  Transformation chain: 2D → 2D (passthrough):")
    for coord_list in test_coords:
        upper_coord = MultiIndex(2, coord_list)
        lower_coord = adaptor.calculate_bottom_index(upper_coord)
        print(f"    {coord_list} → {lower_coord.to_list()} (no change expected)")
    
    return adaptor

def demonstrate_adaptor_properties():
    """Show adaptor properties and inspection methods."""
    print_step(4, "Adaptor properties and inspection")
    
    # Create a transpose adaptor to inspect
    adaptor = make_transpose_adaptor(3, [2, 0, 1])  # 3D rotation: x→y, y→z, z→x
    
    print(f"  Adaptor details:")
    print(f"    Number of transforms: {len(adaptor.transforms)}")
    print(f"    Top dimensions: {adaptor.get_num_of_top_dimension()}")
    print(f"    Bottom dimensions: {adaptor.get_num_of_bottom_dimension()}")
    print(f"    Hidden dimensions: {adaptor.get_num_of_hidden_dimension()}")
    print(f"    Is static: {adaptor.is_static()}")
    
    # Show transform details
    print(f"  Transform components:")
    for i, transform in enumerate(adaptor.transforms):
        print(f"    Transform {i}: {type(transform).__name__}")
        print(f"      Upper dims: {transform.get_num_of_upper_dimension()}")
        print(f"      Lower dims: {transform.get_num_of_lower_dimension()}")
    
    # Show dimension mapping
    print(f"  Dimension mappings:")
    print(f"    Lower dimension hidden IDs: {adaptor.lower_dimension_hidden_idss}")
    print(f"    Upper dimension hidden IDs: {adaptor.upper_dimension_hidden_idss}")
    print(f"    Bottom dimension hidden IDs: {adaptor.bottom_dimension_hidden_ids}")
    print(f"    Top dimension hidden IDs: {adaptor.top_dimension_hidden_ids}")
    
    return adaptor

def demonstrate_adaptor_composition():
    """Show how adaptors can be composed for complex transformations."""
    print_step(5, "Adaptor composition patterns")
    
    # Create multiple simple adaptors
    identity_2d = make_identity_adaptor(2)
    transpose_2d = make_transpose_adaptor(2, [1, 0])
    
    print(f"  Available adaptors:")
    show_adaptor_info(identity_2d, "Identity 4x4")
    show_adaptor_info(transpose_2d, "Transpose 2D")
    
    # Show how they could be conceptually composed
    print(f"  Composition concept:")
    print("    Identity(4x4) → data unchanged")  
    print("    Transpose(2D) → dimensions swapped")
    print("    Composed: Apply identity first, then transpose")
    print("    Result: Transpose without changing the data layout")
    
    # Test coordinates through both
    test_coord = [1, 2]
    print(f"  Coordinate through composition:")
    
    # Through identity (no change)
    upper_coord = MultiIndex(2, test_coord)
    identity_result = identity_2d.calculate_bottom_index(upper_coord)
    print(f"    {test_coord} → Identity → {identity_result.to_list()}")
    
    # Through transpose (swap dimensions)
    transpose_result = transpose_2d.calculate_bottom_index(upper_coord)
    print(f"    {test_coord} → Transpose → {transpose_result.to_list()}")
    
    return identity_2d, transpose_2d

def demonstrate_adaptor_use_cases():
    """Show common use cases for tensor adaptors."""
    print_step(6, "Common adaptor use cases")
    
    print("  🎯 Use Case 1: Matrix Transpose")
    print("    Input: Row-major 3x4 matrix")
    print("    Goal: Access as column-major 4x3 matrix")
    print("    Solution: Transpose adaptor with permutation [1, 0]")
    
    transpose = make_transpose_adaptor(2, [1, 0])
    test_coord = [1, 2]
    result = transpose.calculate_bottom_index(MultiIndex(2, test_coord))
    print(f"    Example: {test_coord} → {result.to_list()}")
    
    print(f"\n  🎯 Use Case 2: Dimension Permutation")
    print("    Input: 3D tensor with shape [D, H, W]")
    print("    Goal: Reorder to [W, D, H] for different processing")
    print("    Solution: Transpose adaptor with permutation [2, 0, 1]")
    
    perm_3d = make_transpose_adaptor(3, [2, 0, 1])
    test_coord_3d = [1, 2, 3]
    result_3d = perm_3d.calculate_bottom_index(MultiIndex(3, test_coord_3d))
    print(f"    Example: {test_coord_3d} → {result_3d.to_list()}")
    
    print(f"\n  🎯 Use Case 3: Identity for Passthrough")
    print("    Input: Any tensor shape")
    print("    Goal: No transformation, but need adaptor interface")
    print("    Solution: Identity adaptor")
    
    identity = make_identity_adaptor(2)
    test_coord_id = [0, 1]
    result_id = identity.calculate_bottom_index(MultiIndex(2, test_coord_id))
    print(f"    Example: {test_coord_id} → {result_id.to_list()}")
    
    return transpose, perm_3d, identity

def test_adaptor_operations():
    """Test adaptor operations work correctly."""
    print_step(7, "Testing adaptor operations")
    
    def test_identity_preserves_coordinates():
        adaptor = make_identity_adaptor(2)
        coord = MultiIndex(2, [1, 2])
        result = adaptor.calculate_bottom_index(coord)
        return result.to_list() == [1, 2]
    
    def test_transpose_swaps_dimensions():
        adaptor = make_transpose_adaptor(2, [1, 0])
        coord = MultiIndex(2, [1, 2])
        result = adaptor.calculate_bottom_index(coord)
        return result.to_list() == [2, 1]  # Swapped
    
    def test_3d_permutation():
        adaptor = make_transpose_adaptor(3, [2, 0, 1])  # [x,y,z] → [z,x,y]
        coord = MultiIndex(3, [1, 2, 3])
        result = adaptor.calculate_bottom_index(coord)
        return result.to_list() == [2, 3, 1]  # Permuted: [1,2,3] with [2,0,1] → [2,3,1]
    
    def test_adaptor_properties():
        adaptor = make_transpose_adaptor(2, [1, 0])
        return (adaptor.get_num_of_top_dimension() == 2 and 
                adaptor.get_num_of_bottom_dimension() == 2 and
                not adaptor.is_static())  # Adaptors are not static by default
    
    def test_custom_adaptor():
        # Simple two-transform adaptor (one for each dimension)
        pass_transform1 = PassThroughTransform(4)  # First dimension length 4
        pass_transform2 = PassThroughTransform(4)  # Second dimension length 4
        adaptor = TensorAdaptor(
            transforms=[pass_transform1, pass_transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]
        )
        coord = MultiIndex(2, [1, 1])
        result = adaptor.calculate_bottom_index(coord)
        return result.to_list() == [1, 1]  # Should be unchanged
    
    tests = [
        ("Identity preserves coordinates", test_identity_preserves_coordinates),
        ("Transpose swaps dimensions", test_transpose_swaps_dimensions),
        ("3D permutation", test_3d_permutation),
        ("Adaptor properties", test_adaptor_properties),
        ("Custom adaptor", test_custom_adaptor)
    ]
    
    results = []
    for test_name, test_func in tests:
        results.append(validate_example(test_name, test_func))
    
    return all(results)

def main():
    """Main function to run all adaptor demonstrations."""
    if not check_imports():
        return False
    
    print_section("Tensor Adaptors")
    
    # Run demonstrations
    identity_adaptor = demonstrate_identity_adaptor()
    transpose_adaptor = demonstrate_transpose_adaptor()
    custom_adaptor = demonstrate_custom_adaptor()
    inspected_adaptor = demonstrate_adaptor_properties()
    identity_2d, transpose_2d = demonstrate_adaptor_composition()
    transpose, perm_3d, identity = demonstrate_adaptor_use_cases()
    
    # Run tests
    all_tests_passed = test_adaptor_operations()
    
    print_section("Summary")
    print(f"✅ Tensor adaptor demonstrations completed")
    print(f"✅ All tests passed: {all_tests_passed}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = run_script_safely(main, "Tensor Adaptors")
    sys.exit(0 if success else 1) 