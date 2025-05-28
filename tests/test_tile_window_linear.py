"""
Tests for tile_window_linear module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.tile_window_linear import (
    TileWindowLinear, SpaceFillingCurve, make_tile_window_linear, is_tile_window_linear
)
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_distribution import make_tile_distribution, make_tile_distribution_encoding
from pytensor.tensor_descriptor import TensorAdaptor, EmbedTransform, make_naive_tensor_descriptor
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.buffer_view import AddressSpaceEnum


class TestSpaceFillingCurve:
    """Test cases for SpaceFillingCurve class."""
    
    def test_space_filling_curve_creation(self):
        """Test creating a space-filling curve."""
        tensor_lengths = [4, 4]
        dim_access_order = [0, 1]
        scalars_per_access = [1, 1]
        
        sfc = SpaceFillingCurve(tensor_lengths, dim_access_order, scalars_per_access)
        
        assert sfc.get_num_of_access() == 16
        assert sfc.access_lengths == [4, 4]
    
    def test_space_filling_curve_indexing(self):
        """Test index calculation."""
        tensor_lengths = [2, 3]
        dim_access_order = [0, 1]
        scalars_per_access = [1, 1]
        
        sfc = SpaceFillingCurve(tensor_lengths, dim_access_order, scalars_per_access)
        
        # Test some indices
        assert sfc.get_index(0) == [0, 0]
        assert sfc.get_index(1) == [0, 1]
        assert sfc.get_index(2) == [0, 2]
        assert sfc.get_index(3) == [1, 0]
        assert sfc.get_index(4) == [1, 1]
        assert sfc.get_index(5) == [1, 2]
    
    def test_space_filling_curve_forward_step(self):
        """Test forward step calculation."""
        tensor_lengths = [2, 2]
        dim_access_order = [0, 1]
        scalars_per_access = [1, 1]
        
        sfc = SpaceFillingCurve(tensor_lengths, dim_access_order, scalars_per_access)
        
        # Test steps
        assert sfc.get_forward_step(0) == [0, 1]  # (0,0) -> (0,1)
        assert sfc.get_forward_step(1) == [1, -1]  # (0,1) -> (1,0)
        assert sfc.get_forward_step(2) == [0, 1]  # (1,0) -> (1,1)
        assert sfc.get_forward_step(3) == [0, 0]  # Last position


class TestTileWindowLinear:
    """Test cases for TileWindowLinear class."""
    
    def test_tile_window_linear_creation(self):
        """Test creating a linear tile window."""
        # Create tensor view
        data = np.zeros((8, 8), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [8, 8], [8, 1])
        
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
        
        # Create linear window
        window = make_tile_window_linear(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist
        )
        
        assert window is not None
        assert window.get_num_of_dimension() == 2
        assert window.get_num_of_access() > 0
    
    def test_tile_window_linear_default_dims(self):
        """Test default linear dimensions based on address space."""
        # Global memory tensor
        data = np.zeros((4, 4), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        tensor_view.buffer_view.address_space = AddressSpaceEnum.GLOBAL
        
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
        
        # Create window without specifying linear dims
        window = TileWindowLinear(tensor_view, [2, 2], [0, 0], dist)
        
        # For global memory, last dim should be linear
        assert window.linear_bottom_dims == [0, 1]
    
    def test_tile_window_linear_load(self):
        """Test loading data with linear window."""
        # Create tensor with pattern
        data = np.arange(16).reshape(4, 4).astype(np.float32)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
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
        
        # Create window
        window = make_tile_window_linear(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist,
            linear_bottom_dims=[0, 0]  # All non-linear for testing
        )
        
        # Load data
        loaded = window.load()
        
        assert loaded is not None
        assert len(loaded.thread_buffer) == 4
    
    def test_tile_window_linear_store(self):
        """Test storing data with linear window."""
        # Create empty tensor
        data = np.zeros((4, 4), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
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
        
        # Create window
        window = make_tile_window_linear(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist
        )
        
        # Create source tensor
        src_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        src_tensor.fill(7.0)
        
        # Store data
        window.store(src_tensor)
        
        # Check that data was written
        assert np.any(data != 0)
    
    def test_tile_window_linear_move(self):
        """Test moving linear window."""
        # Create tensor
        data = np.zeros((8, 8), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [8, 8], [8, 1])
        
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
        
        # Create window
        window = make_tile_window_linear(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist
        )
        
        # Move window
        window.move([2, 2])
        
        # Check that origin was updated
        assert window.window_origin == [2, 2]
    
    def test_is_tile_window_linear(self):
        """Test type checking function."""
        # Create tensor
        data = np.zeros((4, 4), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
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
        
        # Create linear window
        window = make_tile_window_linear(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist
        )
        
        assert is_tile_window_linear(window) == True
        assert is_tile_window_linear("not a window") == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 