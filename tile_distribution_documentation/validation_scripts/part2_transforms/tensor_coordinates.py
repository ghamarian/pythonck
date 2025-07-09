#!/usr/bin/env python3
"""
Purpose: Demonstrate pytensor coordinate concepts - the foundation of tensor operations.

Shows how MultiIndex and TensorCoordinate work, including creation, manipulation,
and coordinate transformations that form the basis for all tensor operations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code_examples'))

from common_utils import *
from pytensor.tensor_coordinate import (
    MultiIndex,
    TensorCoordinate,
    make_tensor_coordinate
)
from pytensor.tensor_descriptor import make_naive_tensor_descriptor
import numpy as np

def demonstrate_multiindex_basics():
    """Show MultiIndex creation and operations."""
    print_step(1, "MultiIndex: Multi-dimensional coordinates")
    
    # Create different dimensional indices
    coord_1d = MultiIndex(1, [5])
    coord_2d = MultiIndex(2, [2, 3])
    coord_3d = MultiIndex(3, [1, 2, 3])
    
    show_coordinate_info(coord_1d, "1D coordinate")
    show_coordinate_info(coord_2d, "2D coordinate")
    show_coordinate_info(coord_3d, "3D coordinate")
    
    # Show coordinate operations
    print("  Coordinate operations:")
    print(f"    1D[0] = {coord_1d[0]}")
    print(f"    2D as list = {coord_2d.to_list()}")
    print(f"    3D size = {coord_3d.size}")
    
    return coord_1d, coord_2d, coord_3d

def demonstrate_tensor_coordinate_creation():
    """Show TensorCoordinate creation from descriptors."""
    print_step(2, "TensorCoordinate: Index + Offset storage")
    
    # Create a 4x6 tensor descriptor (row-major layout)
    descriptor = make_naive_tensor_descriptor([4, 6], [6, 1])
    show_tensor_info(descriptor, "4x6 tensor descriptor")
    
    # Create tensor coordinates for logical indices
    logical_indices = [[0, 0], [2, 3], [3, 5]]
    
    print("  Creating tensor coordinates:")
    for idx in logical_indices:
        coord = make_tensor_coordinate(descriptor, idx)
        offset = coord.get_offset()
        calculation = idx[0] * 6 + idx[1] * 1
        print(f"    Index {idx} → Offset {offset} (calculation: {idx[0]}*6 + {idx[1]}*1 = {calculation})")
    
    return descriptor, logical_indices

def demonstrate_coordinate_properties():
    """Show TensorCoordinate properties and methods."""
    print_step(3, "TensorCoordinate properties")
    
    # Create descriptor and coordinate
    descriptor = make_naive_tensor_descriptor([3, 4], [4, 1])
    coord = make_tensor_coordinate(descriptor, [1, 2])
    
    print(f"  TensorCoordinate properties:")
    print(f"    Index: {coord.get_index().to_list()}")
    print(f"    Offset: {coord.get_offset()}")
    print(f"    Hidden dimensions: {coord.ndim_hidden}")
    
    # Show hidden index details
    hidden_idx = coord.get_hidden_index()
    print(f"    Hidden index: {hidden_idx.to_list()}")
    print(f"    Hidden index size: {hidden_idx.size}")
    
    return coord

def demonstrate_coordinate_arithmetic():
    """Show coordinate arithmetic and movement."""
    print_step(4, "Coordinate arithmetic")
    
    # Create base coordinate
    descriptor = make_naive_tensor_descriptor([4, 4], [4, 1])
    base_coord = make_tensor_coordinate(descriptor, [1, 1])
    
    print(f"  Base coordinate:")
    print(f"    Index: {base_coord.get_index().to_list()}")
    print(f"    Offset: {base_coord.get_offset()}")
    
    # Create offsets to add
    offsets = [
        MultiIndex(2, [0, 1]),  # Move right
        MultiIndex(2, [1, 0]),  # Move down
        MultiIndex(2, [1, 1])   # Move diagonally
    ]
    
    print(f"  Coordinate movements:")
    for i, offset in enumerate(offsets):
        new_coord = make_tensor_coordinate(descriptor, [
            base_coord.get_index()[0] + offset[0],
            base_coord.get_index()[1] + offset[1]
        ])
        print(f"    Move by {offset.to_list()}: Index {new_coord.get_index().to_list()} → Offset {new_coord.get_offset()}")
    
    return base_coord, offsets

def demonstrate_different_layouts():
    """Show coordinates with different tensor layouts."""
    print_step(5, "Coordinates with different layouts")
    
    # Same logical shape, different strides
    layouts = [
        ("Row-major", [3, 4], [4, 1]),
        ("Column-major", [3, 4], [1, 3]),
        ("Custom stride", [3, 4], [8, 2])
    ]
    
    test_index = [1, 2]
    
    print(f"  Same logical index {test_index} in different layouts:")
    for name, shape, strides in layouts:
        descriptor = make_naive_tensor_descriptor(shape, strides)
        coord = make_tensor_coordinate(descriptor, test_index)
        calculation = test_index[0] * strides[0] + test_index[1] * strides[1]
        print(f"    {name}: Offset {coord.get_offset()} (calculation: {test_index[0]}*{strides[0]} + {test_index[1]}*{strides[1]} = {calculation})")
    
    return layouts

def demonstrate_coordinate_validation():
    """Show coordinate validation and bounds checking."""
    print_step(6, "Coordinate validation")
    
    descriptor = make_naive_tensor_descriptor([3, 3], [3, 1])
    
    # Test valid and invalid coordinates
    test_coordinates = [
        ([0, 0], "Valid - top-left"),
        ([2, 2], "Valid - bottom-right"),
        ([1, 1], "Valid - center"),
        ([3, 0], "Invalid - out of bounds row"),
        ([0, 3], "Invalid - out of bounds column"),
        ([-1, 0], "Invalid - negative index")
    ]
    
    print(f"  Coordinate validation for 3x3 tensor:")
    for coord_list, description in test_coordinates:
        try:
            if coord_list[0] >= 0 and coord_list[1] >= 0 and coord_list[0] < 3 and coord_list[1] < 3:
                coord = make_tensor_coordinate(descriptor, coord_list)
                print(f"    {coord_list} - {description}: ✅ Valid (offset {coord.get_offset()})")
            else:
                print(f"    {coord_list} - {description}: ❌ Out of bounds")
        except Exception as e:
            print(f"    {coord_list} - {description}: ❌ Error ({str(e)[:50]}...)")
    
    return test_coordinates

def test_coordinate_operations():
    """Test coordinate operations work correctly."""
    print_step(7, "Testing coordinate operations")
    
    def test_multiindex_creation():
        coord = MultiIndex(3, [1, 2, 3])
        return coord.size == 3 and coord.to_list() == [1, 2, 3]
    
    def test_tensor_coordinate_creation():
        descriptor = make_naive_tensor_descriptor([2, 3], [3, 1])
        coord = make_tensor_coordinate(descriptor, [1, 2])
        expected_offset = 1 * 3 + 2 * 1  # 5
        return coord.get_offset() == expected_offset
    
    def test_coordinate_indexing():
        coord = MultiIndex(2, [4, 5])
        return coord[0] == 4 and coord[1] == 5
    
    def test_different_strides():
        # Row-major vs column-major should give different offsets
        row_major = make_naive_tensor_descriptor([2, 3], [3, 1])
        col_major = make_naive_tensor_descriptor([2, 3], [1, 2])
        
        coord_rm = make_tensor_coordinate(row_major, [1, 1])
        coord_cm = make_tensor_coordinate(col_major, [1, 1])
        
        # Should be different offsets for same logical position
        return coord_rm.get_offset() != coord_cm.get_offset()
    
    def test_coordinate_bounds():
        descriptor = make_naive_tensor_descriptor([2, 2], [2, 1])
        # Valid coordinate
        valid_coord = make_tensor_coordinate(descriptor, [1, 1])
        # Should not throw error and should have correct offset
        return valid_coord.get_offset() == 3  # 1*2 + 1*1 = 3
    
    tests = [
        ("MultiIndex creation", test_multiindex_creation),
        ("TensorCoordinate creation", test_tensor_coordinate_creation),
        ("Coordinate indexing", test_coordinate_indexing),
        ("Different strides", test_different_strides),
        ("Coordinate bounds", test_coordinate_bounds)
    ]
    
    results = []
    for test_name, test_func in tests:
        results.append(validate_example(test_name, test_func))
    
    return all(results)

def main():
    """Main function to run all coordinate demonstrations."""
    if not check_imports():
        return False
    
    print_section("Tensor Coordinates")
    
    # Run demonstrations
    coord_1d, coord_2d, coord_3d = demonstrate_multiindex_basics()
    descriptor, logical_indices = demonstrate_tensor_coordinate_creation()
    coord = demonstrate_coordinate_properties()
    base_coord, offsets = demonstrate_coordinate_arithmetic()
    layouts = demonstrate_different_layouts()
    test_coordinates = demonstrate_coordinate_validation()
    
    # Run tests
    all_tests_passed = test_coordinate_operations()
    
    print_section("Summary")
    print(f"✅ Tensor coordinate demonstrations completed")
    print(f"✅ All tests passed: {all_tests_passed}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = run_script_safely(main, "Tensor Coordinates")
    sys.exit(0 if success else 1) 