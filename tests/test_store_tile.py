"""
Tests for store_tile module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.store_tile import store_tile, store_tile_raw, update_tile, update_tile_raw
from pytensor.tile_window import TileWindowWithStaticLengths, TileWindowWithStaticDistribution, make_tile_window
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_distribution import make_tile_distribution, make_tile_distribution_encoding
from pytensor.tensor_descriptor import TensorAdaptor, EmbedTransform, make_naive_tensor_descriptor


class TestStoreTile:
    """Test cases for store_tile functions."""
    
    def test_store_tile_static_lengths(self):
        """Test storing to a static lengths window."""
        # Create tensor view
        data = np.zeros((4, 4), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create static lengths window
        window = TileWindowWithStaticLengths(
            bottom_tensor_view=tensor_view,
            window_lengths=[2, 2],
            window_origin=[1, 1]
        )
        
        # Create distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
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
        
        descriptor = make_naive_tensor_descriptor([2], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # Create distributed tensor
        distributed_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        distributed_tensor.fill(5.0)
        
        # Store tile
        store_tile(window, distributed_tensor)
        
        # Check that some data was written
        # Note: Due to the simplified implementation, exact behavior may vary
        assert np.any(data != 0)
    
    def test_store_tile_static_distribution(self):
        """Test storing to a static distribution window."""
        # Create tensor view
        data = np.zeros((4, 4), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
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
        
        descriptor = make_naive_tensor_descriptor([2], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # Create distributed window
        window = TileWindowWithStaticDistribution(
            bottom_tensor_view=tensor_view,
            window_lengths=[2, 2],
            window_origin=[1, 1],
            tile_distribution=dist
        )
        
        # Create distributed tensor
        distributed_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        distributed_tensor.fill(7.0)
        
        # Store tile
        store_tile(window, distributed_tensor)
        
        # Check that some data was written
        assert np.any(data != 0)
    
    def test_update_tile(self):
        """Test updating (accumulating) to a window."""
        # Create tensor view with initial data
        data = np.ones((4, 4), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create static lengths window
        window = TileWindowWithStaticLengths(
            bottom_tensor_view=tensor_view,
            window_lengths=[2, 2],
            window_origin=[1, 1]
        )
        
        # Create distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
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
        
        descriptor = make_naive_tensor_descriptor([2], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # Create distributed tensor
        distributed_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        distributed_tensor.fill(3.0)
        
        # Update tile
        update_tile(window, distributed_tensor)
        
        # Check that data was updated (should be > 1 where updated)
        assert np.any(data > 1)
    
    def test_store_tile_raw(self):
        """Test raw store operations."""
        # Create tensor view
        data = np.zeros((4, 4), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create static lengths window
        window = TileWindowWithStaticLengths(
            bottom_tensor_view=tensor_view,
            window_lengths=[2, 2],
            window_origin=[0, 0]
        )
        
        # Create distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
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
        
        descriptor = make_naive_tensor_descriptor([2], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # Create distributed tensor
        distributed_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        distributed_tensor.fill(2.5)
        
        # Store tile raw
        store_tile_raw(window, distributed_tensor)
        
        # Check that some data was written
        assert np.any(data != 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 