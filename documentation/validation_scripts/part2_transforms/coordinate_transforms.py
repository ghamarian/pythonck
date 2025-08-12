#!/usr/bin/env python3
"""
Purpose: Demonstrate pytensor coordinate transforms - the core transformation engine.

Shows how individual transforms (EmbedTransform, UnmergeTransform, MergeTransform, etc.)
work to convert coordinates between different spaces. These are the building blocks
of all tensor operations in Composable Kernels.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code_examples'))

from common_utils import *
from pytensor.tensor_descriptor import (
    EmbedTransform,
    UnmergeTransform,
    MergeTransform,
    ReplicateTransform,
    OffsetTransform,
    PassThroughTransform,
    PadTransform
)
from pytensor.tensor_coordinate import MultiIndex
import numpy as np

def demonstrate_embed_transform():
    """Show EmbedTransform - maps multi-dimensional coordinates to linear memory."""
    print_step(1, "EmbedTransform: Multi-dimensional → Linear memory")
    
    # Create 2D → 1D embedding with strides [6, 1]
    embed = EmbedTransform([2, 3], [6, 1])
    show_transform_info(embed, "EmbedTransform")
    
    # Forward transformation: 2D → 1D
    test_coords = [[0, 0], [0, 2], [1, 0], [1, 2]]
    print("  Forward (upper → lower):")
    for coord_list in test_coords:
        upper_coord = MultiIndex(2, coord_list)
        lower_coord = embed.calculate_lower_index(upper_coord)
        calculation = coord_list[0] * 6 + coord_list[1] * 1
        show_result(f"    {coord_list} → {lower_coord.to_list()}", f"({coord_list[0]}*6 + {coord_list[1]}*1 = {calculation})")
    
    # Backward transformation: 1D → 2D
    test_indices = [0, 2, 6, 8]
    print("  Backward (lower → upper):")
    for idx in test_indices:
        lower_coord = MultiIndex(1, [idx])
        upper_coord = embed.calculate_upper_index(lower_coord)
        show_result(f"    [{idx}] → {upper_coord.to_list()}", "")
    
    return embed

def demonstrate_unmerge_transform():
    """Show UnmergeTransform - unpacks linear index to multi-dimensional coordinates."""
    print_step(2, "UnmergeTransform: Linear → Multi-dimensional (packed)")
    
    # Create 1D → 2D unmerge (2x3 matrix)
    unmerge = UnmergeTransform([2, 3])
    show_transform_info(unmerge, "UnmergeTransform")
    
    # Forward transformation: 2D → 1D (packing)
    test_coords = [[0, 0], [0, 2], [1, 0], [1, 2]]
    print("  Forward (upper → lower):")
    for coord_list in test_coords:
        upper_coord = MultiIndex(2, coord_list)
        lower_coord = unmerge.calculate_lower_index(upper_coord)
        calculation = coord_list[0] * 3 + coord_list[1]
        show_result(f"    {coord_list} → {lower_coord.to_list()}", f"({coord_list[0]}*3 + {coord_list[1]} = {calculation})")
    
    # Backward transformation: 1D → 2D (unpacking)
    test_indices = [0, 2, 3, 5]
    print("  Backward (lower → upper):")
    for idx in test_indices:
        lower_coord = MultiIndex(1, [idx])
        upper_coord = unmerge.calculate_upper_index(lower_coord)
        row = idx // 3
        col = idx % 3
        show_result(f"    [{idx}] → {upper_coord.to_list()}", f"(row={row}, col={col})")
    
    return unmerge

def demonstrate_merge_transform():
    """Show MergeTransform - merges multiple dimensions into single dimension."""
    print_step(3, "MergeTransform: Multi-dimensional → Single dimension")
    
    # Create 2D → 1D merge (collapse 2x3 into single dimension)
    merge = MergeTransform([2, 3])
    show_transform_info(merge, "MergeTransform")
    
    # Forward transformation: 1D → 2D (splitting)
    test_indices = [0, 2, 3, 5]
    print("  Forward (upper → lower):")
    for idx in test_indices:
        upper_coord = MultiIndex(1, [idx])
        lower_coord = merge.calculate_lower_index(upper_coord)
        row = idx // 3
        col = idx % 3
        show_result(f"    [{idx}] → {lower_coord.to_list()}", f"(split to row={row}, col={col})")
    
    # Backward transformation: 2D → 1D (merging)
    test_coords = [[0, 0], [0, 2], [1, 0], [1, 2]]
    print("  Backward (lower → upper):")
    for coord_list in test_coords:
        lower_coord = MultiIndex(2, coord_list)
        upper_coord = merge.calculate_upper_index(lower_coord)
        calculation = coord_list[0] * 3 + coord_list[1]
        show_result(f"    {coord_list} → {upper_coord.to_list()}", f"({coord_list[0]}*3 + {coord_list[1]} = {calculation})")
    
    return merge

def demonstrate_replicate_transform():
    """Show ReplicateTransform - replicates data across multiple processing elements."""
    print_step(4, "ReplicateTransform: Data replication")
    
    # Create replication transform (4 replicas)
    replicate = ReplicateTransform([4])
    show_transform_info(replicate, "ReplicateTransform")
    
    # Forward transformation: replica index → empty
    print("  Forward (upper → lower):")
    test_replicas = [0, 1, 2, 3]
    for replica in test_replicas:
        upper_coord = MultiIndex(1, [replica])  # 1D upper coordinate
        lower_coord = replicate.calculate_lower_index(upper_coord)
        show_result(f"    [replica {replica}] → {lower_coord.to_list()}", "Replica maps to empty coord")
    
    # Backward transformation: empty → replica index
    print("  Backward (lower → upper):")
    lower_coord = MultiIndex(0, [])  # Empty coordinate
    upper_coord = replicate.calculate_upper_index(lower_coord)
    show_result(f"    [] → {upper_coord.to_list()}", "Empty coord maps to zero coordinates")
    
    return replicate

def demonstrate_offset_transform():
    """Show OffsetTransform - adds constant offset to coordinates."""
    print_step(5, "OffsetTransform: Coordinate offsetting")
    
    # Create offset transform (element_space_size=4, offset=2)
    offset = OffsetTransform(4, 2)
    show_transform_info(offset, "OffsetTransform")
    
    # Forward transformation: add offset (1D → 1D)
    test_coords = [0, 1, 2, 3]
    print("  Forward (upper → lower):")
    for coord in test_coords:
        upper_coord = MultiIndex(1, [coord])
        lower_coord = offset.calculate_lower_index(upper_coord)
        result = coord + 2
        show_result(f"    [{coord}] → {lower_coord.to_list()}", f"Add offset 2 = [{result}]")
    
    # Backward transformation: subtract offset (1D → 1D)
    test_coords = [2, 3, 4, 5]
    print("  Backward (lower → upper):")
    for coord in test_coords:
        lower_coord = MultiIndex(1, [coord])
        upper_coord = offset.calculate_upper_index(lower_coord)
        result = coord - 2
        show_result(f"    [{coord}] → {upper_coord.to_list()}", f"Subtract offset 2 = [{result}]")
    
    return offset

def demonstrate_passthrough_transform():
    """Show PassThroughTransform - identity transformation."""
    print_step(6, "PassThroughTransform: Identity transformation")
    
    # Create passthrough transform (single dimension)
    passthrough = PassThroughTransform(5)
    show_transform_info(passthrough, "PassThroughTransform")
    
    # Forward transformation: no change (1D coordinates)
    test_coords = [0, 1, 2, 4]
    print("  Forward (upper → lower):")
    for coord in test_coords:
        upper_coord = MultiIndex(1, [coord])
        lower_coord = passthrough.calculate_lower_index(upper_coord)
        show_result(f"    [{coord}] → {lower_coord.to_list()}", "No change")
    
    # Backward transformation: no change (1D coordinates)
    print("  Backward (lower → upper):")
    for coord in test_coords:
        lower_coord = MultiIndex(1, [coord])
        upper_coord = passthrough.calculate_upper_index(lower_coord)
        show_result(f"    [{coord}] → {upper_coord.to_list()}", "No change")
    
    return passthrough

def demonstrate_pad_transform():
    """Show PadTransform - adds padding to tensor dimensions."""
    print_step(7, "PadTransform: Tensor padding")
    
    # Create pad transform (lower_length=5, left_pad=1, right_pad=1)
    pad = PadTransform(lower_length=5, left_pad=1, right_pad=1)
    show_transform_info(pad, "PadTransform")
    
    # Forward transformation: add padding offset
    test_coords = [0, 1, 2, 3, 4]
    print("  Forward (upper → lower):")
    for coord in test_coords:
        upper_coord = MultiIndex(1, [coord])
        lower_coord = pad.calculate_lower_index(upper_coord)
        result = coord - 1  # Subtract left padding
        show_result(f"    [{coord}] → {lower_coord.to_list()}", f"Subtract left_pad 1 = [{result}]")
    
    # Backward transformation: subtract padding offset
    test_coords = [0, 1, 2, 3]
    print("  Backward (lower → upper):")
    for coord in test_coords:
        lower_coord = MultiIndex(1, [coord])
        upper_coord = pad.calculate_upper_index(lower_coord)
        result = coord + 1  # Add left padding
        show_result(f"    [{coord}] → {upper_coord.to_list()}", f"Add left_pad 1 = [{result}]")
    
    return pad

def test_transform_operations():
    """Test all transform operations."""
    print_step(8, "Testing transform operations")
    
    def test_embed():
        embed = EmbedTransform([2, 3], [6, 1])
        upper = MultiIndex(2, [1, 2])
        lower = embed.calculate_lower_index(upper)
        return lower.to_list() == [8]  # 1*6 + 2*1 = 8
    
    def test_unmerge():
        unmerge = UnmergeTransform([2, 3])
        upper = MultiIndex(2, [1, 2])
        lower = unmerge.calculate_lower_index(upper)
        return lower.to_list() == [5]  # 1*3 + 2 = 5
    
    def test_merge():
        merge = MergeTransform([2, 3])
        upper = MultiIndex(1, [5])
        lower = merge.calculate_lower_index(upper)
        return lower.to_list() == [1, 2]  # 5//3=1, 5%3=2
    
    def test_replicate():
        replicate = ReplicateTransform([4])
        upper = MultiIndex(1, [2])  # 1D upper coordinate (replica index)
        lower = replicate.calculate_lower_index(upper)
        return len(lower.to_list()) == 0  # Should produce empty coordinate
    
    def test_offset():
        offset = OffsetTransform(10, 3)
        upper = MultiIndex(1, [1])
        lower = offset.calculate_lower_index(upper)
        return lower.to_list() == [4]  # 1 + 3 = 4
    
    tests = [
        ("EmbedTransform", test_embed),
        ("UnmergeTransform", test_unmerge),
        ("MergeTransform", test_merge),
        ("ReplicateTransform", test_replicate),
        ("OffsetTransform", test_offset)
    ]
    
    results = []
    for test_name, test_func in tests:
        results.append(validate_example(test_name, test_func))
    
    return all(results)

def main():
    """Main function to run all coordinate transform demonstrations."""
    if not check_imports():
        return False
    
    print_section("Coordinate Transforms")
    
    # Run demonstrations
    embed = demonstrate_embed_transform()
    unmerge = demonstrate_unmerge_transform()
    merge = demonstrate_merge_transform()
    replicate = demonstrate_replicate_transform()
    offset = demonstrate_offset_transform()
    passthrough = demonstrate_passthrough_transform()
    pad = demonstrate_pad_transform()
    
    # Run tests
    all_tests_passed = test_transform_operations()
    
    print_section("Summary")
    print(f"✅ Coordinate transform demonstrations completed")
    print(f"✅ All tests passed: {all_tests_passed}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = run_script_safely(main, "Coordinate Transforms")
    sys.exit(0 if success else 1) 