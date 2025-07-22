"""
Tests for sequence utility functions.
"""

import pytest
import sys
sys.path.append('..')

from pytensor.sequence_utils import (
    reduce_on_sequence, slice_sequence, reverse_slice_sequence,
    multiplies, plus, gcd, Multiplies, Plus, get_y_unpacks_from_x_unpacks
)


class TestGCD:
    """Test greatest common divisor function."""
    
    def test_gcd_basic(self):
        """Test basic GCD cases."""
        assert gcd(12, 8) == 4
        assert gcd(8, 12) == 4
        assert gcd(15, 25) == 5
        assert gcd(7, 13) == 1
        assert gcd(0, 5) == 5
        assert gcd(5, 0) == 5


class TestFunctionObjects:
    """Test function objects."""
    
    def test_multiplies(self):
        """Test multiplication function object."""
        mult = Multiplies()
        assert mult(3, 4) == 12
        assert mult(0, 5) == 0
        assert mult(-2, 3) == -6
        
        # Test the global instance
        assert multiplies(3, 4) == 12
    
    def test_plus(self):
        """Test addition function object."""
        add = Plus()
        assert add(3, 4) == 7
        assert add(0, 5) == 5
        assert add(-2, 3) == 1
        
        # Test the global instance
        assert plus(3, 4) == 7


class TestReduceOnSequence:
    """Test reduce_on_sequence function."""
    
    def test_reduce_multiply(self):
        """Test reduction with multiplication."""
        result = reduce_on_sequence([2, 3, 4], multiplies, 1)
        assert result == 24
        
        result = reduce_on_sequence([1, 2, 3, 4], multiplies, 1)
        assert result == 24
        
        result = reduce_on_sequence([5], multiplies, 2)
        assert result == 10
        
        result = reduce_on_sequence([], multiplies, 5)
        assert result == 5
    
    def test_reduce_add(self):
        """Test reduction with addition."""
        result = reduce_on_sequence([1, 2, 3], plus, 0)
        assert result == 6
        
        result = reduce_on_sequence([10, 20, 30], plus, 5)
        assert result == 65
        
        result = reduce_on_sequence([], plus, 10)
        assert result == 10


class TestReverseSliceSequence:
    """Test reverse_slice_sequence function."""
    
    def test_reverse_slice_basic(self):
        """Test basic reverse slicing."""
        # Simple case
        lengths, nums, split_idx = reverse_slice_sequence([4, 2], 8)
        assert lengths == [4, 2]
        assert nums == [1, 1]
        assert split_idx == 0
    
    def test_reverse_slice_with_splitting(self):
        """Test reverse slicing that requires splitting."""
        # This should split the last dimension
        lengths, nums, split_idx = reverse_slice_sequence([4, 8], 4)
        assert lengths == [1, 4]
        assert nums == [4, 2]
        assert split_idx == 1
    
    def test_reverse_slice_invalid(self):
        """Test invalid reverse slicing."""
        with pytest.raises(ValueError, match="Cannot evenly divide"):
            reverse_slice_sequence([3, 5], 7)


class TestSliceSequence:
    """Test slice_sequence function based on C++ examples."""
    
    def test_slice_examples_from_cpp(self):
        """Test the exact examples from C++ comments."""
        
        # <2, 1, 4, 2>, 8 -> lengths:<1, 1, 4, 2>, nums: <2, 1, 1, 1>: 2 slices, slice_idx: 0
        lengths, nums, split_idx = slice_sequence([2, 1, 4, 2], 8)
        assert lengths == [1, 1, 4, 2]
        assert nums == [2, 1, 1, 1]
        assert split_idx == 0
        
        # <4, 2, 4, 1, 2>, 4 -> lengths:<1, 1, 2, 1, 2>, nums: <4, 2, 2, 1, 1>: 16 slices, slice_idx: 2
        lengths, nums, split_idx = slice_sequence([4, 2, 4, 1, 2], 4)
        assert lengths == [1, 1, 2, 1, 2]
        assert nums == [4, 2, 2, 1, 1]
        assert split_idx == 2
        
        # <4, 2, 8>, 64 -> lengths:<4, 2, 8>, nums: <1, 1, 1>: 1 slices, slice_idx: 0
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 64)
        assert lengths == [4, 2, 8]
        assert nums == [1, 1, 1]
        assert split_idx == 0
        
        # <4, 2, 8>, 32 -> lengths:<2, 2, 8>, nums: <2, 1, 1>: 2 slices, slice_idx: 0
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 32)
        assert lengths == [2, 2, 8]
        assert nums == [2, 1, 1]
        assert split_idx == 0
        
        # <4, 2, 8>, 16 -> lengths:<1, 2, 8>, nums: <4, 1, 1>: 4 slices, slice_idx: 0
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 16)
        assert lengths == [1, 2, 8]
        assert nums == [4, 1, 1]
        assert split_idx == 0
        
        # <4, 2, 8>, 8 -> lengths:<1, 1, 8>, nums: <4, 2, 1>: 8 slices, slice_idx: 1
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 8)
        assert lengths == [1, 1, 8]
        assert nums == [4, 2, 1]
        assert split_idx == 1
        
        # <4, 2, 8>, 4 -> lengths:<1, 1, 4>, nums: <4, 2, 2>: 16 slices, slice_idx: 2
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 4)
        assert lengths == [1, 1, 4]
        assert nums == [4, 2, 2]
        assert split_idx == 2
        
        # <4, 2, 8>, 2 -> lengths:<1, 1, 2>, nums: <4, 2, 4>: 32 slices, slice_idx: 2
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 2)
        assert lengths == [1, 1, 2]
        assert nums == [4, 2, 4]
        assert split_idx == 2
        
        # <4, 2, 8>, 1 -> lengths:<1, 1, 1>, nums: <4, 2, 8>: 64 slices, slice_idx: 2
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 1)
        assert lengths == [1, 1, 1]
        assert nums == [4, 2, 8]
        assert split_idx == 2
    
    def test_slice_with_mask(self):
        """Test slicing with mask."""
        # mask:<1, 1, 1, 0, 1> -> lengths:<1, 2, 1, 4, 2>, nums: <4, 1, 1, 1, 1>: 8 slices, slice_idx: 0
        lengths, nums, split_idx = slice_sequence([4, 2, 1, 4, 2], 4, [1, 1, 1, 0, 1])
        assert lengths == [1, 2, 1, 4, 2]
        assert nums == [4, 1, 1, 1, 1]
        assert split_idx == 0
    
    def test_slice_verify_total_slices(self):
        """Verify that the total number of slices is correct."""
        
        # For [4, 2, 8] with slice_size 4, we should get 16 slices total
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 4)
        total_slices = 1
        for num in nums:
            total_slices *= num
        assert total_slices == 16  # 4 * 2 * 2 = 16
        
        # For [4, 2, 8] with slice_size 8, we should get 8 slices total
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 8)
        total_slices = 1
        for num in nums:
            total_slices *= num
        assert total_slices == 8  # 4 * 2 * 1 = 8
    
    def test_slice_verify_slice_size(self):
        """Verify that each slice has the correct size."""
        
        # For [4, 2, 8] with slice_size 4
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 4)
        slice_size = 1
        for length in lengths:
            slice_size *= length
        assert slice_size == 4  # 1 * 1 * 4 = 4
        
        # For [4, 2, 8] with slice_size 8
        lengths, nums, split_idx = slice_sequence([4, 2, 8], 8)
        slice_size = 1
        for length in lengths:
            slice_size *= length
        assert slice_size == 8  # 1 * 1 * 8 = 8


class TestGetYUnpacksFromXUnpacks:
    """Test get_y_unpacks_from_x_unpacks function."""
    
    def test_basic_unpacking(self):
        """Test basic Y unpacking from X unpacking."""
        
        # Simple case: [2, 4] with x_unpacks=2
        # y_size = 2*4 = 8, y_slice_size = 8/2 = 4
        # slice_sequence([2, 4], 4) should give nums=[2, 1]
        result = get_y_unpacks_from_x_unpacks([2, 4], 2)
        assert result == [2, 1]
        
        # Another case: [4, 2] with x_unpacks=2
        # y_size = 4*2 = 8, y_slice_size = 8/2 = 4  
        # slice_sequence([4, 2], 4) should give nums=[2, 1]
        result = get_y_unpacks_from_x_unpacks([4, 2], 2)
        assert result == [2, 1]
    
    def test_no_unpacking(self):
        """Test case with no unpacking (x_unpacks=1)."""
        
        # With x_unpacks=1, we should get all 1s
        result = get_y_unpacks_from_x_unpacks([2, 3, 4], 1)
        assert result == [1, 1, 1]
        
        result = get_y_unpacks_from_x_unpacks([8], 1)
        assert result == [1]
    
    def test_full_unpacking(self):
        """Test case with full unpacking."""
        
        # [2, 3, 4] has total size 24
        # With x_unpacks=24, each slice should be size 1
        result = get_y_unpacks_from_x_unpacks([2, 3, 4], 24)
        assert result == [2, 3, 4]
        
        # [4, 2] has total size 8
        # With x_unpacks=8, each slice should be size 1
        result = get_y_unpacks_from_x_unpacks([4, 2], 8)
        assert result == [4, 2]
    
    def test_intermediate_unpacking(self):
        """Test intermediate unpacking cases."""
        
        # [4, 2, 8] has total size 64
        # With x_unpacks=4, slice_size=16, should give [1, 2, 8] -> nums=[4, 1, 1]
        result = get_y_unpacks_from_x_unpacks([4, 2, 8], 4)
        assert result == [4, 1, 1]
        
        # With x_unpacks=8, slice_size=8, should give [1, 1, 8] -> nums=[4, 2, 1]
        result = get_y_unpacks_from_x_unpacks([4, 2, 8], 8)
        assert result == [4, 2, 1]
        
        # With x_unpacks=16, slice_size=4, should give [1, 1, 4] -> nums=[4, 2, 2]
        result = get_y_unpacks_from_x_unpacks([4, 2, 8], 16)
        assert result == [4, 2, 2]
    
    def test_invalid_unpacking(self):
        """Test invalid unpacking cases."""
        
        # y_size = 6, x_unpacks = 4, 6 is not divisible by 4
        with pytest.raises(ValueError, match="Y size 6 is not divisible by Y packs 4"):
            get_y_unpacks_from_x_unpacks([2, 3], 4)
        
        # y_size = 8, x_unpacks = 3, 8 is not divisible by 3
        with pytest.raises(ValueError, match="Y size 8 is not divisible by Y packs 3"):
            get_y_unpacks_from_x_unpacks([4, 2], 3)
    
    def test_edge_cases(self):
        """Test edge cases."""
        
        # Empty sequence
        result = get_y_unpacks_from_x_unpacks([], 1)
        assert result == []
        
        # Single dimension
        result = get_y_unpacks_from_x_unpacks([8], 2)
        assert result == [2]
        
        result = get_y_unpacks_from_x_unpacks([8], 4)
        assert result == [4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 