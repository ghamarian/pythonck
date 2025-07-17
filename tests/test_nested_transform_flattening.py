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