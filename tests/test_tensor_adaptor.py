"""
Tests for tensor_adaptor module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.tensor_adaptor import (
    make_single_stage_tensor_adaptor, transform_tensor_adaptor,
    chain_tensor_adaptors, chain_tensor_adaptors_multi,
    make_identity_adaptor, make_transpose_adaptor
)
from pytensor.tensor_descriptor import (
    PassThroughTransform, PadTransform, MergeTransform,
    ReplicateTransform, EmbedTransform, UnmergeTransform
)
from pytensor.tensor_coordinate import MultiIndex


class TestMakeSingleStageTensorAdaptor:
    """Test cases for make_single_stage_tensor_adaptor function."""
    
    def test_simple_passthrough(self):
        """Test single stage adaptor with pass-through transforms."""
        transforms = [PassThroughTransform(4), PassThroughTransform(3)]
        lower_idss = [[0], [1]]
        upper_idss = [[0], [1]]
        
        adaptor = make_single_stage_tensor_adaptor(transforms, lower_idss, upper_idss)
        
        assert adaptor.get_num_of_transform() == 2
        assert adaptor.get_num_of_bottom_dimension() == 2
        assert adaptor.get_num_of_top_dimension() == 2
        assert adaptor.bottom_dimension_hidden_ids == [0, 1]
        assert adaptor.top_dimension_hidden_ids == [2, 3]
    
    def test_merge_transform(self):
        """Test single stage adaptor with merge transform."""
        transforms = [MergeTransform([2, 3])]
        lower_idss = [[0, 1]]
        upper_idss = [[0]]
        
        adaptor = make_single_stage_tensor_adaptor(transforms, lower_idss, upper_idss)
        
        assert adaptor.get_num_of_bottom_dimension() == 2
        assert adaptor.get_num_of_top_dimension() == 1
        
        # Test calculate_bottom_index
        result = adaptor.calculate_bottom_index(MultiIndex(1, [5]))
        assert list(result) == [1, 2]  # 5 = 1*3 + 2
    
    def test_validation(self):
        """Test validation of input parameters."""
        transforms = [PassThroughTransform(4)]
        
        # Mismatched lower dimensions
        with pytest.raises(ValueError, match="lower dimension mappings"):
            make_single_stage_tensor_adaptor(transforms, [[0], [1]], [[0]])
        
        # Mismatched upper dimensions
        with pytest.raises(ValueError, match="upper dimension mappings"):
            make_single_stage_tensor_adaptor(transforms, [[0]], [[0], [1]])


class TestTransformTensorAdaptor:
    """Test cases for transform_tensor_adaptor function."""
    
    def test_add_transform_to_identity(self):
        """Test adding transform to identity adaptor."""
        # Start with 2D identity
        old_adaptor = make_identity_adaptor(2)
        
        # Add merge transform
        new_transforms = [MergeTransform([2, 2])]
        new_lower_idss = [[0, 1]]
        new_upper_idss = [[0]]
        
        new_adaptor = transform_tensor_adaptor(
            old_adaptor, new_transforms, new_lower_idss, new_upper_idss
        )
        
        assert new_adaptor.get_num_of_transform() == 3  # 2 passthrough + 1 merge
        assert new_adaptor.get_num_of_bottom_dimension() == 2
        assert new_adaptor.get_num_of_top_dimension() == 1
    
    def test_chain_multiple_transforms(self):
        """Test chaining multiple transforms."""
        # Start with passthrough
        adaptor = make_single_stage_tensor_adaptor(
            [PassThroughTransform(4)], [[0]], [[0]]
        )
        
        # Add padding
        adaptor = transform_tensor_adaptor(
            adaptor,
            [PadTransform(4, 1, 1)],
            [[0]],
            [[0]]
        )
        
        assert adaptor.get_num_of_transform() == 2
        assert adaptor.transforms[1].get_upper_lengths() == [6]  # 4 + 1 + 1


class TestChainTensorAdaptors:
    """Test cases for chain_tensor_adaptors function."""
    
    def test_chain_two_adaptors(self):
        """Test chaining two compatible adaptors."""
        # First adaptor: 2D -> 1D (merge)
        adaptor0 = make_single_stage_tensor_adaptor(
            [MergeTransform([2, 3])],
            [[0, 1]],
            [[0]]
        )
        
        # Second adaptor: 1D -> 2D (unmerge)
        adaptor1 = make_single_stage_tensor_adaptor(
            [UnmergeTransform([2, 3])],
            [[0]],
            [[0, 1]]
        )
        
        # Chain them
        chained = chain_tensor_adaptors(adaptor0, adaptor1)
        
        assert chained.get_num_of_transform() == 2
        assert chained.get_num_of_bottom_dimension() == 2
        assert chained.get_num_of_top_dimension() == 2
        
        # Should be identity overall
        result = chained.calculate_bottom_index(MultiIndex(2, [1, 2]))
        assert list(result) == [1, 2]
    
    def test_incompatible_dimensions(self):
        """Test that incompatible dimensions raise error."""
        adaptor0 = make_identity_adaptor(2)
        adaptor1 = make_identity_adaptor(3)
        
        with pytest.raises(ValueError, match="must match bottom dimensions"):
            chain_tensor_adaptors(adaptor0, adaptor1)
    
    def test_chain_multiple(self):
        """Test chaining multiple adaptors."""
        adaptor0 = make_identity_adaptor(2)
        adaptor1 = make_single_stage_tensor_adaptor(
            [MergeTransform([2, 2])],
            [[0, 1]],
            [[0]]
        )
        adaptor2 = make_identity_adaptor(1)
        
        chained = chain_tensor_adaptors_multi(adaptor0, adaptor1, adaptor2)
        
        assert chained.get_num_of_transform() == 4  # 2 + 1 + 1
        assert chained.get_num_of_bottom_dimension() == 2
        assert chained.get_num_of_top_dimension() == 1


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_make_identity_adaptor(self):
        """Test identity adaptor creation."""
        adaptor = make_identity_adaptor(3)
        
        assert adaptor.get_num_of_top_dimension() == 3
        assert adaptor.get_num_of_transform() == 3
        assert all(isinstance(t, PassThroughTransform) for t in adaptor.transforms)
        
        # Should be identity mapping
        result = adaptor.calculate_bottom_index(MultiIndex(3, [1, 2, 0]))
        assert list(result) == [1, 2, 0]
    
    def test_make_transpose_adaptor(self):
        """Test transpose adaptor creation."""
        # Transpose [0, 1, 2] -> [2, 0, 1]
        adaptor = make_transpose_adaptor(3, [2, 0, 1])
        
        assert adaptor.get_num_of_top_dimension() == 3
        assert adaptor.get_num_of_transform() == 3
        
        # Note: This is a simplified test - actual transpose logic
        # would need proper coordinate transformation
    
    def test_transpose_validation(self):
        """Test transpose adaptor validation."""
        # Wrong length
        with pytest.raises(ValueError, match="same length"):
            make_transpose_adaptor(3, [0, 1])
        
        # Invalid permutation
        with pytest.raises(ValueError, match="valid permutation"):
            make_transpose_adaptor(3, [0, 1, 3])


class TestTransformClasses:
    """Test cases for new transform classes."""
    
    def test_passthrough_transform(self):
        """Test PassThroughTransform."""
        transform = PassThroughTransform(5)
        
        assert transform.get_num_of_lower_dimension() == 1
        assert transform.get_num_of_upper_dimension() == 1
        assert transform.get_upper_lengths() == [5]
        
        # Test with MultiIndex
        idx_in = MultiIndex(1, [3])
        idx_out = transform.calculate_lower_index(idx_in)
        assert list(idx_out) == [3]
        
        # Vector properties
        vec_lens, vec_strides = transform.calculate_upper_dimension_safe_vector_length_strides(
            [4], [1]
        )
        assert vec_lens == [4]
        assert vec_strides == [1]
    
    def test_pad_transform(self):
        """Test PadTransform."""
        transform = PadTransform(4, 1, 2)  # [1, 4, 2]
        
        assert transform.get_upper_lengths() == [7]  # 4 + 1 + 2
        
        # Test index mapping with MultiIndex (matches C++ - no clamping)
        assert list(transform.calculate_lower_index(MultiIndex(1, [0]))) == [-1]  # In left pad
        assert list(transform.calculate_lower_index(MultiIndex(1, [1]))) == [0]   # First real element
        assert list(transform.calculate_lower_index(MultiIndex(1, [4]))) == [3]   # Last real element
        assert list(transform.calculate_lower_index(MultiIndex(1, [6]))) == [5]   # In right pad
        
        # Padding breaks vectorization
        vec_lens, vec_strides = transform.calculate_upper_dimension_safe_vector_length_strides(
            [4], [1]
        )
        assert vec_lens == [1]
        assert vec_strides == [1]
    
    def test_merge_transform(self):
        """Test MergeTransform."""
        transform = MergeTransform([2, 3, 4])
        
        assert transform.get_num_of_lower_dimension() == 3
        assert transform.get_num_of_upper_dimension() == 1
        assert transform.get_upper_lengths() == [24]  # 2 * 3 * 4
        
        # Test index mapping with MultiIndex
        assert list(transform.calculate_lower_index(MultiIndex(1, [0]))) == [0, 0, 0]
        assert list(transform.calculate_lower_index(MultiIndex(1, [1]))) == [0, 0, 1]
        assert list(transform.calculate_lower_index(MultiIndex(1, [4]))) == [0, 1, 0]
        assert list(transform.calculate_lower_index(MultiIndex(1, [12]))) == [1, 0, 0]
        assert list(transform.calculate_lower_index(MultiIndex(1, [23]))) == [1, 2, 3]
    
    def test_merge_transform_upper_index(self):
        """Test MergeTransform's calculate_upper_index."""
        transform = MergeTransform([2, 3, 4])
        
        # Test upper index calculation
        assert list(transform.calculate_upper_index(MultiIndex(3, [0, 0, 0]))) == [0]
        assert list(transform.calculate_upper_index(MultiIndex(3, [0, 0, 1]))) == [1]
        assert list(transform.calculate_upper_index(MultiIndex(3, [0, 1, 0]))) == [4]
        assert list(transform.calculate_upper_index(MultiIndex(3, [1, 0, 0]))) == [12]
        assert list(transform.calculate_upper_index(MultiIndex(3, [1, 2, 3]))) == [23]
        
        # Test invalid input - wrong dimensionality
        with pytest.raises(ValueError, match="Index dimension 2 doesn't match transform dimension 3"):
            transform.calculate_upper_index(MultiIndex(2, [0, 0]))  # 2D instead of 3D
        
        # Test invalid input - out of bounds indices
        with pytest.raises(ValueError, match="Index out of bounds"):
            transform.calculate_upper_index(MultiIndex(3, [2, 0, 0]))  # First index too large
        with pytest.raises(ValueError, match="Index out of bounds"):
            transform.calculate_upper_index(MultiIndex(3, [0, 3, 0]))  # Second index too large
        with pytest.raises(ValueError, match="Index out of bounds"):
            transform.calculate_upper_index(MultiIndex(3, [0, 0, 4]))  # Third index too large
    
    def test_replicate_transform(self):
        """Test ReplicateTransform."""
        transform = ReplicateTransform([2, 3])
        
        assert transform.get_num_of_lower_dimension() == 0
        assert transform.get_num_of_upper_dimension() == 2
        assert transform.get_upper_lengths() == [2, 3]
        
        # Test with MultiIndex
        idx_in = MultiIndex(2, [1, 2])
        idx_out = transform.calculate_lower_index(idx_in)
        assert len(idx_out) == 0  # Empty lower index
        
        # Replicated dimensions have stride 0
        vec_lens, vec_strides = transform.calculate_upper_dimension_safe_vector_length_strides(
            [], []
        )
        assert vec_lens == [2, 3]
        assert vec_strides == [0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 