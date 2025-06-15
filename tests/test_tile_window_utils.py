"""
Tests for tile_window_utils module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.tile_window_utils import (
    move_tile_window, get_async_store_smem_info,
    get_warp_id, get_lane_id, m0_set_with_memory, m0_inc_with_memory
)
from pytensor.tile_window import TileWindowWithStaticDistribution, make_tile_window
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_distribution import make_tile_distribution, make_tile_distribution_encoding
from pytensor.tensor_descriptor import TensorAdaptor, EmbedTransform, make_naive_tensor_descriptor


class TestTileWindowUtils:
    """Test cases for tile window utility functions."""
    
    def test_move_tile_window(self):
        """Test moving a tile window."""
        # Create tensor view
        data = np.zeros((8, 8), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [8, 8], [8, 1])
        
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
        
        # Create window
        window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist
        )
        
        # Move window
        move_tile_window(window, [1, 1])
        
        # Check that origin was updated
        assert window.window_origin == [1, 1]
    
    def test_get_async_store_smem_info(self):
        """Test extracting LDS store information."""
        # Create 3D tensor view for LDS
        data = np.zeros((4, 2, 32), dtype=np.float32)  # issues, warps, lanes
        tensor_view = make_naive_tensor_view(data, [4, 2, 32], [64, 32, 1])
        
        # Create 3D distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4], [2], [32]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 2, 3],
            ys_to_rhs_minor=[0, 0, 0]
        )
        
        transform1 = EmbedTransform([4], [64])
        transform2 = EmbedTransform([2], [32])
        transform3 = EmbedTransform([32], [1])
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2, transform3],
            lower_dimension_hidden_idss=[[0], [1], [2]],
            upper_dimension_hidden_idss=[[3], [4], [5]],
            bottom_dimension_hidden_ids=[0, 1, 2],
            top_dimension_hidden_ids=[3, 4, 5]
        )
        
        descriptor = make_naive_tensor_descriptor([256], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # Create LDS window
        lds_window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=[4, 2, 32],
            origin=[0, 0, 0],
            tile_distribution=dist
        )
        
        # Get async store info
        m0_init, size_per_issue = get_async_store_smem_info(lds_window)
        
        # Check results
        assert m0_init >= 0
        assert size_per_issue > 0
        
        # Size per issue should be 64 * 4 = 256 bytes (64 floats * 4 bytes)
        assert size_per_issue == 256
    
    def test_get_async_store_smem_info_wrong_dims(self):
        """Test that wrong dimensions raise an error."""
        # Create 2D tensor view (wrong dimensions)
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
        
        # Create window
        window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=dist
        )
        
        # Should raise error
        with pytest.raises(ValueError):
            get_async_store_smem_info(window)
    
    def test_hardware_simulation_functions(self):
        """Test hardware simulation functions."""
        # These should always return 0 in Python simulation
        assert get_warp_id() == 0
        assert get_lane_id() == 0
        
        # These should be no-ops
        m0_set_with_memory(123)  # Should not raise
        m0_inc_with_memory(456)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 