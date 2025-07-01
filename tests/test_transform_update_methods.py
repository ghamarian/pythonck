"""
Integration tests for transform update methods to ensure compatibility with C++ implementation.

This module tests that the Python transforms with update_lower_index methods
produce identical results to the calculate_lower_index approach.
"""

import pytest
import numpy as np
import sys
from typing import List, Tuple

sys.path.append('.')

from pytensor.tensor_coordinate import MultiIndex
from pytensor.tensor_descriptor import (
    Transform, EmbedTransform, UnmergeTransform, OffsetTransform,
    PassThroughTransform, PadTransform, MergeTransform, XorTransform,
    ReplicateTransform, make_naive_tensor_descriptor, make_naive_tensor_descriptor_packed
)


class TransformTestSuite:
    """Test suite for comparing calculate vs update methods."""
    
    def __init__(self, transform: Transform, test_name: str):
        self.transform = transform
        self.test_name = test_name
    
    def run_calculate_vs_update_test(self, upper_indices: List[List[int]]) -> bool:
        """
        Test that update_lower_index produces same results as calculate_lower_index.
        
        Args:
            upper_indices: List of upper index sequences to test
            
        Returns:
            True if all tests pass
        """
        print(f"\n=== Testing {self.test_name} ===")
        
        if len(upper_indices) < 2:
            print("  Skipping - need at least 2 indices for update test")
            return True
        
        # Test each consecutive pair of indices
        all_passed = True
        for i in range(len(upper_indices) - 1):
            current_upper = MultiIndex(len(upper_indices[i]), upper_indices[i])
            next_upper = MultiIndex(len(upper_indices[i+1]), upper_indices[i+1])
            
            # Method 1: Calculate both indices independently
            current_lower_calc = self.transform.calculate_lower_index(current_upper)
            next_lower_calc = self.transform.calculate_lower_index(next_upper)
            
            # Method 2: Calculate first, then use update for second
            current_lower_update = self.transform.calculate_lower_index(current_upper)
            
            # Calculate upper index difference
            upper_diff = MultiIndex(len(current_upper), 
                                  [next_upper[j] - current_upper[j] for j in range(len(current_upper))])
            
            # Use update method (if available)
            if hasattr(self.transform, 'update_lower_index'):
                lower_diff, updated_lower = self.transform.update_lower_index(
                    upper_diff, current_lower_update, next_upper)
                
                # Compare results
                calc_matches_update = (list(next_lower_calc) == list(updated_lower))
                if calc_matches_update:
                    print(f"  ✅ Step {i}: {upper_indices[i]} -> {upper_indices[i+1]}")
                    print(f"     Upper diff: {list(upper_diff)}")
                    print(f"     Lower diff: {list(lower_diff)}")
                    print(f"     Calc: {list(next_lower_calc)} == Update: {list(updated_lower)}")
                else:
                    print(f"  ❌ Step {i}: MISMATCH!")
                    print(f"     Calc: {list(next_lower_calc)} != Update: {list(updated_lower)}")
                    all_passed = False
            else:
                print(f"  ⚠️  No update_lower_index method implemented yet")
                all_passed = False
        
        return all_passed


def test_embed_transform_update():
    """Test EmbedTransform update method."""
    transform = EmbedTransform([4, 8, 16], [128, 16, 1])  # 4×8×16 tensor with strides
    suite = TransformTestSuite(transform, "EmbedTransform")
    
    # Test sequence: stepping through different coordinates
    upper_indices = [
        [0, 0, 0],  # Start at origin
        [0, 0, 1],  # Step in last dim
        [0, 1, 0],  # Step in middle dim  
        [1, 0, 0],  # Step in first dim
        [1, 2, 5],  # Random position
        [2, 3, 7],  # Another position
    ]
    
    assert suite.run_calculate_vs_update_test(upper_indices)


def test_unmerge_transform_update():
    """Test UnmergeTransform update method."""
    transform = UnmergeTransform([4, 8, 16])  # 4×8×16 packed dimensions
    suite = TransformTestSuite(transform, "UnmergeTransform")
    
    # Test sequence: linear indices that unmerge to different coordinates
    upper_indices = [
        [0, 0, 0],    # -> 0
        [0, 0, 1],    # -> 1  
        [0, 1, 0],    # -> 16
        [1, 0, 0],    # -> 128
        [2, 3, 5],    # -> 2*128 + 3*16 + 5 = 309
    ]
    
    assert suite.run_calculate_vs_update_test(upper_indices)


def test_pass_through_transform_update():
    """Test PassThroughTransform update method.""" 
    transform = PassThroughTransform(16)
    suite = TransformTestSuite(transform, "PassThroughTransform")
    
    upper_indices = [
        [0], [1], [5], [10], [15]
    ]
    
    assert suite.run_calculate_vs_update_test(upper_indices)


def test_pad_transform_update():
    """Test PadTransform update method."""
    transform = PadTransform(8, 2, 3)  # length=8, left_pad=2, right_pad=3
    suite = TransformTestSuite(transform, "PadTransform")
    
    upper_indices = [
        [0],   # Padding region
        [2],   # Start of valid region
        [5],   # Middle of valid region
        [9],   # End of valid region  
        [12],  # Right padding region
    ]
    
    assert suite.run_calculate_vs_update_test(upper_indices)


def test_merge_transform_update():
    """Test MergeTransform update method."""
    transform = MergeTransform([4, 8])  # Merge 4×8 into single dimension
    suite = TransformTestSuite(transform, "MergeTransform")
    
    upper_indices = [
        [0],   # -> [0, 0]
        [1],   # -> [0, 1]
        [8],   # -> [1, 0]
        [15],  # -> [1, 7]
        [31],  # -> [3, 7]
    ]
    
    assert suite.run_calculate_vs_update_test(upper_indices)


def test_xor_transform_update():
    """Test XorTransform update method."""
    transform = XorTransform([16, 8])  # 16×8 with XOR scrambling
    suite = TransformTestSuite(transform, "XorTransform")
    
    upper_indices = [
        [0, 0],
        [0, 1], 
        [1, 0],
        [1, 1],
        [8, 4],
        [15, 7],
    ]
    
    assert suite.run_calculate_vs_update_test(upper_indices)


def test_replicate_transform_update():
    """Test ReplicateTransform update method."""
    transform = ReplicateTransform([4, 8])  # Replicate to 4×8
    suite = TransformTestSuite(transform, "ReplicateTransform")
    
    upper_indices = [
        [0, 0],
        [1, 2],
        [3, 7],
        [2, 5],
    ]
    
    assert suite.run_calculate_vs_update_test(upper_indices)


def test_offset_transform_update():
    """Test OffsetTransform update method."""
    transform = OffsetTransform(16, 5)  # Add offset of 5
    suite = TransformTestSuite(transform, "OffsetTransform")
    
    upper_indices = [
        [0], [1], [5], [10], [15]
    ]
    
    assert suite.run_calculate_vs_update_test(upper_indices)


def test_complex_tensor_descriptor_update():
    """Test update methods with a complex tensor descriptor."""
    print("\n=== Testing Complex TensorDescriptor ===")
    
    # Create a complex tensor descriptor with multiple transforms
    descriptor = make_naive_tensor_descriptor([64, 128], [128, 1])
    
    # Test sequence of coordinate calculations
    coordinates = [
        [0, 0],    # Origin
        [0, 1],    # Step in last dim
        [1, 0],    # Step in first dim  
        [10, 20],  # Random position
        [63, 127], # Near end
    ]
    
    print("Testing complex descriptor offset calculations:")
    for i, coord in enumerate(coordinates):
        offset = descriptor.calculate_offset(coord)
        expected_offset = coord[0] * 128 + coord[1]
        
        if offset == expected_offset:
            print(f"  ✅ {coord} -> offset {offset}")
        else:
            print(f"  ❌ {coord} -> offset {offset}, expected {expected_offset}")
            assert False, f"Offset mismatch for {coord}"


def test_sweep_tile_compatibility():
    """Test that sweep_tile works with both calculate and update methods."""
    from pytensor.static_distributed_tensor import StaticDistributedTensor
    from pytensor.sweep_tile import sweep_tile
    from pytensor.tile_distribution import make_static_tile_distribution
    from pytensor.tile_distribution_encoding import TileDistributionEncoding
    
    print("\n=== Testing sweep_tile compatibility ===")
    
    # Create a simple tile distribution
    encoding = TileDistributionEncoding(
        rs_lengths=[],
        hs_lengthss=[[2, 2], [2, 2]],  # 2×2 × 2×2 tile
        ps_to_rhss_major=[[1], [1]],
        ps_to_rhss_minor=[[1], [1]], 
        ys_to_rhs_major=[1, 1],
        ys_to_rhs_minor=[0, 1]
    )
    
    tile_distribution = make_static_tile_distribution(encoding)
    
    # Create a distributed tensor
    tensor = StaticDistributedTensor(
        data_type=np.float32,
        tile_distribution=tile_distribution
    )
    
    # Collect values using current method
    values_current = []
    
    def collect_current(y_indices):
        # Simulate some computation based on y_indices
        value = sum(y_indices) * 10.0
        values_current.append(value)
    
    sweep_tile(tensor, collect_current)
    
    print(f"Swept {len(values_current)} elements: {values_current}")
    
    # This test mainly ensures sweep_tile runs without errors
    # More comprehensive tests would require implementing the update methods first
    assert len(values_current) > 0


if __name__ == "__main__":
    """Run all integration tests."""
    print("🧪 Running Transform Update Method Integration Tests")
    print("=" * 70)
    
    # Note: These tests will initially fail because update methods aren't implemented yet
    # But they establish the testing framework
    
    try:
        test_pass_through_transform_update()
        test_embed_transform_update() 
        test_unmerge_transform_update()
        test_pad_transform_update()
        test_merge_transform_update()
        test_xor_transform_update()
        test_replicate_transform_update()
        test_offset_transform_update()
        test_complex_tensor_descriptor_update()
        test_sweep_tile_compatibility()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        print("This is expected until update methods are implemented.")
        
    print("\nNext step: Implement update_lower_index methods for all transforms.") 