"""
Tests for tile_scatter_gather module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.tile_scatter_gather import TileScatterGather, make_tile_scatter_gather
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_distribution import make_tile_distribution, make_tile_distribution_encoding
from pytensor.tensor_descriptor import TensorAdaptor, EmbedTransform, make_naive_tensor_descriptor
from pytensor.static_distributed_tensor import StaticDistributedTensor


class TestTileScatterGather:
    """Test cases for tile scatter/gather operations."""
    
    def test_scatter_gather_creation(self):
        """Test creating a scatter/gather accessor."""
        # Create tensor view
        data = np.arange(16).reshape(4, 4).astype(np.float32)
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
        
        # Create page index array
        page_idx = np.array([0, 2], dtype=np.int32)
        
        # Create scatter/gather
        sg = make_tile_scatter_gather(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist,
            page_idx=page_idx,
            hs_gather_dim=0
        )
        
        assert sg is not None
        assert sg.window_lengths == [2, 2]
        assert sg.window_origin == [0, 0]
        assert np.array_equal(sg.page_idx_array, page_idx)
    
    def test_scatter_gather_load(self):
        """Test loading data with scatter/gather pattern."""
        # Create tensor with known pattern
        data = np.arange(16).reshape(4, 4).astype(np.float32)
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
        
        # Page indices select rows 0 and 2
        page_idx = np.array([0, 2], dtype=np.int32)
        
        # Create scatter/gather
        sg = make_tile_scatter_gather(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist,
            page_idx=page_idx,
            hs_gather_dim=0,
            ys_gather_dim=0
        )
        
        # Load data
        loaded_tensor = sg.load()
        
        # Check that data was loaded
        assert loaded_tensor is not None
        assert loaded_tensor.thread_buffer is not None
    
    def test_scatter_gather_store(self):
        """Test storing data with scatter pattern."""
        # Create empty tensor
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
        
        # Page indices select rows 1 and 3
        page_idx = np.array([1, 3], dtype=np.int32)
        
        # Create scatter/gather
        sg = make_tile_scatter_gather(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist,
            page_idx=page_idx,
            hs_gather_dim=0,
            ys_gather_dim=0
        )
        
        # Create source tensor
        src_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        src_tensor.fill(5.0)
        
        # Store data
        sg.store(src_tensor)
        
        # Check that some data was written
        assert np.any(data != 0)
    
    def test_scatter_gather_move_window(self):
        """Test moving the scatter/gather window."""
        # Create tensor
        data = np.arange(16).reshape(4, 4).astype(np.float32)
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
        
        page_idx = np.array([0, 1], dtype=np.int32)
        
        # Create scatter/gather
        sg = make_tile_scatter_gather(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist,
            page_idx=page_idx,
            hs_gather_dim=0
        )
        
        # Move window
        sg.move([0, 1])
        
        # Check that origin was updated
        assert sg.window_origin[1] == 1
        assert sg.window_origin[0] == 0  # hs_gather_dim not moved
    
    def test_scatter_gather_update_page_idx(self):
        """Test updating page indices."""
        # Create tensor
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
        
        page_idx = np.array([0, 1], dtype=np.int32)
        
        # Create scatter/gather
        sg = make_tile_scatter_gather(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist,
            page_idx=page_idx,
            hs_gather_dim=0
        )
        
        # Update page indices
        new_page_idx = np.array([2, 3], dtype=np.int32)
        sg.update_page_idx(new_page_idx)
        
        # Check update
        assert np.array_equal(sg.page_idx_array, new_page_idx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 