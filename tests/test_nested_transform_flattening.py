#!/usr/bin/env python3
"""
Test nested transform flattening to match C++ behavior.

This test verifies that the Python implementation correctly handles nested transforms
like the C++ template system does by automatically flattening them.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytensor.tensor_descriptor import (
    transform_tensor_descriptor,
    make_merge_transform,
    make_pass_through_transform,
    make_tuple,
    number,
    sequence,
    make_naive_tensor_descriptor_packed,
    MergeTransform
)
from pytensor.tensor_adaptor import make_single_stage_tensor_adaptor
from pytensor.tensor_coordinate import MultiIndex


class TestNestedTransformFlattening:
    """Test suite for nested transform flattening functionality."""
    
    def test_nested_merge_transform_creation(self):
        """Test that nested merge transforms are correctly flattened during creation."""
        # Test dimensions
        B, C = 2, 4
        
        # Create nested merge transform
        inner_merge = make_merge_transform(make_tuple(number(B), number(C)))
        assert isinstance(inner_merge, MergeTransform)
        assert inner_merge.lengths == [B, C]
        
        # Create outer merge with nested transform
        A = 3
        outer_merge = make_merge_transform(make_tuple(
            make_pass_through_transform(number(A)),
            inner_merge
        ))
        
        # Should be flattened to [A, B*C]
        expected_lengths = [A, B * C]
        assert isinstance(outer_merge, MergeTransform)
        assert outer_merge.lengths == expected_lengths, f"Expected {expected_lengths}, got {outer_merge.lengths}"
    
    def test_cpp_equivalent_nested_case(self):
        """Test the user's specific nested case that should work like C++."""
        # Test dimensions from user's example
        A, B, C, D = 3, 2, 4, 5
        
        # Create input descriptor 
        input_desc = make_naive_tensor_descriptor_packed([A, B*C, D, 1])
        
        # Create the nested transforms from user's C++ example:
        # make_merge_transform(
        #     make_tuple(
        #         make_pass_through_transform(number<A>{}),
        #         make_merge_transform(make_tuple(number<B>{}, number<C>{}))
        #     )
        # )
        
        # Inner merge: B*C
        inner_merge = make_merge_transform(make_tuple(number(B), number(C)))
        
        # Outer merge: A, (B*C) - should be automatically flattened
        outer_merge = make_merge_transform(make_tuple(
            make_pass_through_transform(number(A)),
            inner_merge
        ))
        
        # Pass-through for D
        pass_through_d = make_pass_through_transform(number(D))
        
        # Apply transforms using the user's dimension mappings
        result_desc = transform_tensor_descriptor(
            input_desc,
            make_tuple(outer_merge, pass_through_d),
            make_tuple(sequence(0, 1, 2), sequence(3)),
            make_tuple(sequence(0), sequence(1))
        )
        
        # Verify the result
        expected_lengths = [A * B * C, D]  # [3*2*4, 5] = [24, 5]
        actual_lengths = result_desc.get_lengths()
        
        assert actual_lengths == expected_lengths, f"Expected {expected_lengths}, got {actual_lengths}"
        assert result_desc.get_num_of_dimension() == 2, f"Expected 2D output, got {result_desc.get_num_of_dimension()}D"
    
    def test_flattening_equivalence(self):
        """Test that nested approach gives same result as manual flattening."""
        A, B, C, D = 3, 2, 4, 5
        
        # Create input descriptor
        input_desc = make_naive_tensor_descriptor_packed([A, B*C, D, 1])
        
        # Approach 1: Nested transforms (should be auto-flattened)
        inner_merge = make_merge_transform(make_tuple(number(B), number(C)))
        outer_merge = make_merge_transform(make_tuple(
            make_pass_through_transform(number(A)),
            inner_merge
        ))
        
        nested_result = transform_tensor_descriptor(
            input_desc,
            make_tuple(outer_merge, make_pass_through_transform(D)),
            make_tuple(sequence(0, 1, 2), sequence(3)),
            make_tuple(sequence(0), sequence(1))
        )
        
        # Approach 2: Manual flattening
        bc_merged = B * C
        manual_result = make_single_stage_tensor_adaptor(
            transforms=[
                make_merge_transform([A, bc_merged]),
                make_pass_through_transform(D)
            ],
            lower_dimension_old_top_idss=[
                [0, 1, 2],  # First transform takes dims 0,1,2
                [3]         # Second transform takes dim 3
            ],
            upper_dimension_new_top_idss=[
                [0],        # First transform outputs to dim 0
                [1]         # Second transform outputs to dim 1
            ]
        )
        
        # Compare results
        assert nested_result.get_lengths() == [A * B * C, D]
        assert manual_result.get_num_of_top_dimension() == 2
        
        # Both should produce equivalent dimensional structure
        assert nested_result.get_num_of_dimension() == manual_result.get_num_of_top_dimension()
    
    def test_pass_through_in_nested_transform(self):
        """Test that PassThroughTransform works correctly in nested contexts."""
        length = 5
        
        # Create a merge with pass-through
        merge_with_passthrough = make_merge_transform(make_tuple(
            make_pass_through_transform(number(length)),
            number(3)
        ))
        
        # Should flatten to [5, 3]
        expected = [length, 3]
        assert merge_with_passthrough.lengths == expected
    
    def test_unsupported_nested_transform_error(self):
        """Test that unsupported nested transforms raise appropriate errors."""
        from pytensor.tensor_descriptor import UnmergeTransform
        
        # This should raise an error since UnmergeTransform isn't supported in nested context
        with pytest.raises(ValueError, match="Unsupported nested transform"):
            make_merge_transform(make_tuple(
                make_pass_through_transform(number(3)),
                UnmergeTransform([2, 2])  # This should cause error
            ))
    
    def test_input_validation(self):
        """Test input validation for the tensor descriptor functions."""
        # Test with invalid dimensions
        input_desc = make_naive_tensor_descriptor_packed([2, 3])
        
        # This should work fine
        result = transform_tensor_descriptor(
            input_desc,
            [make_pass_through_transform(5)],
            [[0]],
            [[0]]
        )
        
        assert result.get_lengths() == [5]


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 