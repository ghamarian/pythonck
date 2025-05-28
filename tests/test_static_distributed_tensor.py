"""
Tests for static_distributed_tensor module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.static_distributed_tensor import (
    StaticDistributedTensor, make_static_distributed_tensor
)
from pytensor.tile_distribution import (
    TileDistribution, TileDistributionEncoding,
    make_tile_distribution_encoding, make_tile_distribution
)
from pytensor.tensor_descriptor import (
    TensorAdaptor, UnmergeTransform, make_naive_tensor_descriptor
)


class TestStaticDistributedTensor:
    """Test cases for StaticDistributedTensor class."""
    
    def test_basic_creation(self):
        """Test basic tensor creation."""
        # Create simple distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = UnmergeTransform([4])
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding,
            partition_index_func=lambda: [0]
        )
        
        # Create distributed tensor
        tensor = make_static_distributed_tensor(np.float32, dist)
        
        assert tensor.data_type == np.float32
        assert len(tensor.get_thread_buffer()) == 4
        assert np.all(tensor.get_thread_buffer() == 0)
    
    def test_get_set_thread_data(self):
        """Test getting and setting thread data."""
        # Create simple distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[8]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = UnmergeTransform([8])
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        descriptor = make_naive_tensor_descriptor([8], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding,
            partition_index_func=lambda: [0]
        )
        
        tensor = StaticDistributedTensor(np.float32, dist)
        
        # Set and get data
        tensor.set_thread_data(0, 1.0)
        tensor.set_thread_data(3, 4.5)
        tensor.set_thread_data(7, -2.0)
        
        assert tensor.get_thread_data(0) == 1.0
        assert tensor.get_thread_data(3) == 4.5
        assert tensor.get_thread_data(7) == -2.0
    
    def test_get_set_element(self):
        """Test getting and setting elements using Y indices."""
        # Create 2D distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2, 3]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 1],
            ys_to_rhs_minor=[0, 1]
        )
        
        transform = UnmergeTransform([2, 3])
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1, 2]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1, 2]
        )
        
        descriptor = make_naive_tensor_descriptor([2, 3], [3, 1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding,
            partition_index_func=lambda: [0]
        )
        
        tensor = StaticDistributedTensor(np.int32, dist)
        
        # Set elements using Y indices
        tensor.set_element([0, 0], 10)
        tensor.set_element([0, 1], 20)
        tensor.set_element([1, 0], 30)
        tensor.set_element([1, 2], 40)
        
        # Get elements
        assert tensor.get_element([0, 0]) == 10
        assert tensor.get_element([0, 1]) == 20
        assert tensor.get_element([1, 0]) == 30
        assert tensor.get_element([1, 2]) == 40
    
    def test_clear_and_fill(self):
        """Test clear and fill operations."""
        # Create simple distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = UnmergeTransform([4])
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding,
            partition_index_func=lambda: [0]
        )
        
        tensor = StaticDistributedTensor(np.float32, dist)
        
        # Fill with value
        tensor.fill(3.14)
        assert np.all(tensor.get_thread_buffer() == 3.14)
        
        # Clear
        tensor.clear()
        assert np.all(tensor.get_thread_buffer() == 0)
    
    def test_copy_from(self):
        """Test copying from another tensor."""
        # Create distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[6]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = UnmergeTransform([6])
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        descriptor = make_naive_tensor_descriptor([6], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding,
            partition_index_func=lambda: [0]
        )
        
        # Create two tensors
        tensor1 = StaticDistributedTensor(np.float32, dist)
        tensor2 = StaticDistributedTensor(np.float32, dist)
        
        # Fill tensor1 with data
        for i in range(6):
            tensor1.set_thread_data(i, i * 1.5)
        
        # Copy to tensor2
        tensor2.copy_from(tensor1)
        
        # Verify
        for i in range(6):
            assert tensor2.get_thread_data(i) == i * 1.5
    
    def test_copy_from_incompatible(self):
        """Test that copying between incompatible tensors raises error."""
        # Create two different distributions
        encoding1 = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        encoding2 = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[6]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform1 = UnmergeTransform([4])
        adaptor1 = TensorAdaptor(
            transforms=[transform1],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        transform2 = UnmergeTransform([6])
        adaptor2 = TensorAdaptor(
            transforms=[transform2],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        descriptor1 = make_naive_tensor_descriptor([4], [1])
        descriptor2 = make_naive_tensor_descriptor([6], [1])
        
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
        
        tensor1 = StaticDistributedTensor(np.float32, dist1)
        tensor2 = StaticDistributedTensor(np.float32, dist2)
        
        with pytest.raises(ValueError, match="different shapes"):
            tensor2.copy_from(tensor1)
    
    def test_repr(self):
        """Test string representation."""
        # Create simple distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = UnmergeTransform([4])
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding,
            partition_index_func=lambda: [0]
        )
        
        tensor = StaticDistributedTensor(np.float64, dist)
        
        repr_str = repr(tensor)
        assert "StaticDistributedTensor" in repr_str
        assert "dtype=float64" in repr_str
        assert "size=4" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 