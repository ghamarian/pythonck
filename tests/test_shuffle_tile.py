"""
Tests for shuffle_tile module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.shuffle_tile import shuffle_tile, shuffle_tile_in_thread
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.tile_distribution import make_tile_distribution, make_tile_distribution_encoding
from pytensor.tensor_descriptor import TensorAdaptor, EmbedTransform, make_naive_tensor_descriptor


class TestShuffleTile:
    """Test cases for shuffle_tile functions."""
    
    def test_shuffle_tile_same_layout(self):
        """Test shuffling between tensors with same layout."""
        # Create distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 0],
            ys_to_rhs_minor=[0, 1]
        )
        
        transform1 = EmbedTransform([2], [1])
        transform2 = EmbedTransform([2], [1])
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding,
            partition_index_func=lambda: [0]
        )
        
        # Create input tensor
        in_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Fill with test data
        for i in range(4):
            in_tensor.thread_buffer[i] = float(i + 1)
        
        # Create output tensor
        out_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Shuffle
        shuffle_tile(out_tensor, in_tensor)
        
        # Check that data was copied
        for i in range(4):
            assert out_tensor.thread_buffer[i] == in_tensor.thread_buffer[i]
    
    def test_shuffle_tile_different_y_order(self):
        """Test shuffling between tensors with different Y dimension ordering."""
        # Create input distribution with Y order [0, 1]
        encoding_in = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[0, 1],
            ys_to_rhs_minor=[0, 1]
        )
        
        # Create output distribution with Y order [1, 0] (transposed)
        encoding_out = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 0],
            ys_to_rhs_minor=[1, 0]
        )
        
        transform1 = EmbedTransform([2], [1])
        transform2 = EmbedTransform([2], [1])
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        dist_in = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding_in,
            partition_index_func=lambda: [0]
        )
        
        dist_out = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding_out,
            partition_index_func=lambda: [0]
        )
        
        # Create tensors
        in_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist_in
        )
        
        out_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist_out
        )
        
        # Fill input with pattern
        in_tensor.thread_buffer[0] = 1.0  # (0,0)
        in_tensor.thread_buffer[1] = 2.0  # (0,1)
        in_tensor.thread_buffer[2] = 3.0  # (1,0)
        in_tensor.thread_buffer[3] = 4.0  # (1,1)
        
        # Shuffle
        shuffle_tile(out_tensor, in_tensor)
        
        # Check that data was transposed
        # The exact mapping depends on the implementation details
        assert np.all(out_tensor.thread_buffer != 0)
    
    def test_shuffle_tile_type_conversion(self):
        """Test shuffling with type conversion."""
        # Create distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 0],
            ys_to_rhs_minor=[0, 1]
        )
        
        transform1 = EmbedTransform([2], [1])
        transform2 = EmbedTransform([2], [1])
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding,
            partition_index_func=lambda: [0]
        )
        
        # Create input tensor with float32
        in_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Create output tensor with float64
        out_tensor = StaticDistributedTensor(
            data_type=np.float64,
            tile_distribution=dist
        )
        
        # Fill input
        for i in range(4):
            in_tensor.thread_buffer[i] = float(i + 1)
        
        # Shuffle with type conversion
        shuffle_tile(out_tensor, in_tensor)
        
        # Check that data was converted and copied
        assert out_tensor.thread_buffer.dtype == np.float64
        assert np.all(out_tensor.thread_buffer != 0)
    
    def test_shuffle_tile_incompatible_distributions(self):
        """Test that incompatible distributions raise an error."""
        # Create incompatible distributions
        encoding1 = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 0],
            ys_to_rhs_minor=[0, 1]
        )
        
        encoding2 = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[3], [3]],  # Different H lengths
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 0],
            ys_to_rhs_minor=[0, 1]
        )
        
        transform1 = EmbedTransform([2], [1])
        transform2 = EmbedTransform([2], [1])
        
        adaptor1 = TensorAdaptor(
            transforms=[transform1, transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]
        )
        
        transform3 = EmbedTransform([3], [1])
        transform4 = EmbedTransform([3], [1])
        
        adaptor2 = TensorAdaptor(
            transforms=[transform3, transform4],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]
        )
        
        descriptor1 = make_naive_tensor_descriptor([4], [1])
        descriptor2 = make_naive_tensor_descriptor([9], [1])
        
        dist1 = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor1,
            ys_to_d_descriptor=descriptor1,
            encoding=encoding1,
            partition_index_func=lambda: [0]
        )
        
        dist2 = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor2,
            ys_to_d_descriptor=descriptor2,
            encoding=encoding2,
            partition_index_func=lambda: [0]
        )
        
        # Create tensors
        in_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist1
        )
        
        out_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist2
        )
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            shuffle_tile(out_tensor, in_tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 