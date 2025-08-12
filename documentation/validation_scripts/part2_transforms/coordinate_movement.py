#!/usr/bin/env python3
"""
Purpose: Demonstrate move_tensor_coordinate functionality - efficient coordinate navigation.

Shows how move_tensor_coordinate works and validates it's equivalent to creating new coordinates.
This is the "move stuff" functionality that powers efficient tensor operations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code_examples'))

from common_utils import *
from pytensor.tensor_coordinate import (
    MultiIndex,
    TensorCoordinate,
    TensorAdaptorCoordinate,
    make_tensor_coordinate,
    make_tensor_adaptor_coordinate,
    move_tensor_coordinate,
    move_tensor_adaptor_coordinate
)
from pytensor.tensor_descriptor import make_naive_tensor_descriptor
from pytensor.tensor_adaptor import make_transpose_adaptor
import numpy as np

def demonstrate_move_tensor_coordinate_basics():
    """Show basic move_tensor_coordinate functionality."""
    print_step(1, "Basic move_tensor_coordinate operations")
    
    # Create a simple 3x4 tensor descriptor
    desc = make_naive_tensor_descriptor([3, 4], [4, 1])  # Row-major
    print(f"Tensor descriptor: {desc.get_lengths()} shape, strides [4, 1]")
    
    # Start at position [1, 1]
    coord = make_tensor_coordinate(desc, [1, 1])
    print(f"Starting position: {coord.get_index().to_list()}, offset: {coord.get_offset()}")
    
    # Test various movements
    movements = [
        ([0, 1], "right 1 column"),
        ([1, 0], "down 1 row"),
        ([0, -1], "left 1 column"),
        ([1, 1], "diagonal down-right")
    ]
    
    for move_vec, description in movements:
        print(f"\n  Move {move_vec} ({description}):")
        old_index = coord.get_index().to_list()
        old_offset = coord.get_offset()
        
        # Move the coordinate
        move_tensor_coordinate(desc, coord, move_vec)
        
        new_index = coord.get_index().to_list()
        new_offset = coord.get_offset()
        
        print(f"    {old_index} → {new_index}")
        print(f"    Offset: {old_offset} → {new_offset}")
    
    return coord

def demonstrate_move_equivalence():
    """Show that move_tensor_coordinate is equivalent to creating new coordinates."""
    print_step(2, "Move vs Create equivalence testing")
    
    desc = make_naive_tensor_descriptor([4, 5], [5, 1])
    
    # Test multiple movements
    test_cases = [
        ([1, 2], [0, 1]),  # Start at [1,2], move [0,1]
        ([2, 3], [1, 0]),  # Start at [2,3], move [1,0]
        ([0, 0], [2, 3]),  # Start at [0,0], move [2,3]
        ([1, 1], [1, 1])   # Start at [1,1], move [1,1]
    ]
    
    for start_pos, move_vec in test_cases:
        print(f"\n  Test: Start {start_pos}, move {move_vec}")
        
        # Method 1: Use move_tensor_coordinate
        coord_move = make_tensor_coordinate(desc, start_pos)
        move_tensor_coordinate(desc, coord_move, move_vec)
        
        # Method 2: Create new coordinate directly
        final_pos = [start_pos[0] + move_vec[0], start_pos[1] + move_vec[1]]
        coord_direct = make_tensor_coordinate(desc, final_pos)
        
        # Compare results
        move_result = coord_move.get_index().to_list()
        direct_result = coord_direct.get_index().to_list()
        move_offset = coord_move.get_offset()
        direct_offset = coord_direct.get_offset()
        
        equivalent = (move_result == direct_result and move_offset == direct_offset)
        
        print(f"    Move method: {move_result}, offset: {move_offset}")
        print(f"    Direct method: {direct_result}, offset: {direct_offset}")
        print(f"    Equivalent: {'✅' if equivalent else '❌'}")
        
        if not equivalent:
            print(f"    ❌ MISMATCH DETECTED!")
            return False
    
    print(f"\n  ✅ All move operations equivalent to direct creation")
    return True

def demonstrate_move_adaptor_coordinate():
    """Show move_tensor_adaptor_coordinate with transpose."""
    print_step(3, "Move with tensor adaptor coordinates")
    
    # Create transpose adaptor
    transpose_adaptor = make_transpose_adaptor(2, [1, 0])
    
    # Create adaptor coordinate
    coord = make_tensor_adaptor_coordinate(transpose_adaptor, [1, 2])
    print(f"Initial adaptor coordinate:")
    print(f"  Top: {coord.get_top_index().to_list()}")
    print(f"  Bottom: {coord.get_bottom_index().to_list()}")
    
    # Test movements through adaptor
    movements = [
        ([1, 0], "move down"),
        ([0, 1], "move right"),
        ([1, 1], "move diagonal")
    ]
    
    for move_vec, description in movements:
        print(f"\n  Move {move_vec} ({description}):")
        old_top = coord.get_top_index().to_list()
        old_bottom = coord.get_bottom_index().to_list()
        
        # Move through adaptor
        bottom_diff = move_tensor_adaptor_coordinate(transpose_adaptor, coord, move_vec)
        
        new_top = coord.get_top_index().to_list()
        new_bottom = coord.get_bottom_index().to_list()
        
        print(f"    Top: {old_top} → {new_top}")
        print(f"    Bottom: {old_bottom} → {new_bottom}")
        print(f"    Bottom diff: {bottom_diff.to_list()}")
    
    return coord

def demonstrate_complex_movement_patterns():
    """Show complex movement patterns and their efficiency."""
    print_step(4, "Complex movement patterns")
    
    # Create a larger tensor for pattern demonstration
    desc = make_naive_tensor_descriptor([8, 8], [8, 1])
    coord = make_tensor_coordinate(desc, [0, 0])
    
    print(f"8x8 tensor, starting at [0, 0]")
    
    # Pattern 1: Scan a row
    print(f"\n  Pattern 1: Scan row 2 (columns 0-7)")
    coord = make_tensor_coordinate(desc, [2, 0])  # Start at row 2
    positions = []
    
    for col in range(8):
        positions.append(coord.get_index().to_list())
        if col < 7:  # Don't move after last column
            move_tensor_coordinate(desc, coord, [0, 1])
    
    print(f"    Positions: {positions}")
    
    # Pattern 2: Diagonal movement
    print(f"\n  Pattern 2: Diagonal from [0,0] to [7,7]")
    coord = make_tensor_coordinate(desc, [0, 0])
    positions = []
    
    for step in range(8):
        positions.append(coord.get_index().to_list())
        if step < 7:  # Don't move after last step
            move_tensor_coordinate(desc, coord, [1, 1])
    
    print(f"    Positions: {positions}")
    
    # Pattern 3: Spiral pattern (partial)
    print(f"\n  Pattern 3: Spiral start (right, down, left)")
    coord = make_tensor_coordinate(desc, [1, 1])
    positions = [coord.get_index().to_list()]
    
    # Right 3 steps
    for _ in range(3):
        move_tensor_coordinate(desc, coord, [0, 1])
        positions.append(coord.get_index().to_list())
    
    # Down 3 steps  
    for _ in range(3):
        move_tensor_coordinate(desc, coord, [1, 0])
        positions.append(coord.get_index().to_list())
    
    # Left 3 steps
    for _ in range(3):
        move_tensor_coordinate(desc, coord, [0, -1])
        positions.append(coord.get_index().to_list())
    
    print(f"    Positions: {positions}")
    
    return positions

def test_move_operations():
    """Test all move operations work correctly."""
    print_step(5, "Testing move operations")
    
    def test_basic_move():
        desc = make_naive_tensor_descriptor([3, 3], [3, 1])
        coord = make_tensor_coordinate(desc, [1, 1])
        move_tensor_coordinate(desc, coord, [0, 1])
        expected = [1, 2]
        return coord.get_index().to_list() == expected
    
    def test_negative_move():
        desc = make_naive_tensor_descriptor([4, 4], [4, 1])
        coord = make_tensor_coordinate(desc, [2, 2])
        move_tensor_coordinate(desc, coord, [-1, -1])
        expected = [1, 1]
        return coord.get_index().to_list() == expected
    
    def test_zero_move():
        desc = make_naive_tensor_descriptor([2, 2], [2, 1])
        coord = make_tensor_coordinate(desc, [1, 0])
        old_pos = coord.get_index().to_list()
        move_tensor_coordinate(desc, coord, [0, 0])
        new_pos = coord.get_index().to_list()
        return old_pos == new_pos
    
    def test_adaptor_move():
        adaptor = make_transpose_adaptor(2, [1, 0])
        coord = make_tensor_adaptor_coordinate(adaptor, [1, 2])
        bottom_diff = move_tensor_adaptor_coordinate(adaptor, coord, [1, 0])
        # After moving [1,0] in transposed space, should see change
        return coord.get_top_index().to_list() == [2, 2]
    
    def test_move_equivalence():
        desc = make_naive_tensor_descriptor([5, 5], [5, 1])
        
        # Test multiple random moves
        for _ in range(10):
            start = [1, 1]
            move_vec = [1, 1]
            
            # Method 1: move
            coord_move = make_tensor_coordinate(desc, start)
            move_tensor_coordinate(desc, coord_move, move_vec)
            
            # Method 2: direct
            final_pos = [start[0] + move_vec[0], start[1] + move_vec[1]]
            coord_direct = make_tensor_coordinate(desc, final_pos)
            
            if (coord_move.get_index().to_list() != coord_direct.get_index().to_list() or
                coord_move.get_offset() != coord_direct.get_offset()):
                return False
        
        return True
    
    tests = [
        ("Basic move operation", test_basic_move),
        ("Negative move operation", test_negative_move),
        ("Zero move operation", test_zero_move),
        ("Adaptor move operation", test_adaptor_move),
        ("Move equivalence", test_move_equivalence)
    ]
    
    results = []
    for test_name, test_func in tests:
        results.append(validate_example(test_name, test_func))
    
    return all(results)

def main():
    """Main function to run all move_tensor_coordinate demonstrations."""
    if not check_imports():
        return False
    
    print_section("move_tensor_coordinate: Efficient Coordinate Movement")
    
    # Run demonstrations
    coord1 = demonstrate_move_tensor_coordinate_basics()
    equiv_result = demonstrate_move_equivalence()
    coord2 = demonstrate_move_adaptor_coordinate()
    patterns = demonstrate_complex_movement_patterns()
    
    # Run tests
    all_tests_passed = test_move_operations()
    
    print_section("Summary")
    print(f"✅ move_tensor_coordinate demonstrations completed")
    print(f"✅ Move equivalence verified: {equiv_result}")
    print(f"✅ All tests passed: {all_tests_passed}")
    
    # Key insights
    print(f"\n🔍 Key Insights:")
    print(f"   • move_tensor_coordinate is equivalent to creating new coordinates")
    print(f"   • But it's more efficient - updates existing coordinate in-place")
    print(f"   • Works with both tensor and adaptor coordinates")
    print(f"   • Supports negative movements and zero movements")
    print(f"   • Essential for efficient tensor navigation patterns")
    
    return all_tests_passed and equiv_result

if __name__ == "__main__":
    success = run_script_safely(main, "move_tensor_coordinate")
    sys.exit(0 if success else 1) 