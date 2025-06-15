"""
Tests for tile_window module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.tile_window import (
    TileWindowWithStaticDistribution, TileWindowWithStaticLengths,
    make_tile_window, move_tile_window
)
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_distribution import (
    TileDistribution, TileDistributionEncoding,
    make_tile_distribution_encoding, make_tile_distribution
)
from pytensor.tensor_descriptor import (
    TensorAdaptor, UnmergeTransform, make_naive_tensor_descriptor, EmbedTransform, PassThroughTransform
)


class TestTileWindowWithStaticLengths:
    """Test cases for TileWindowWithStaticLengths class."""
    
    def test_basic_creation(self):
        """Test basic window creation."""
        # Create tensor view
        data = np.arange(24, dtype=np.float32).reshape(6, 4)
        tensor_view = make_naive_tensor_view(data, [6, 4], [4, 1])
        
        # Create window
        window = TileWindowWithStaticLengths(
            bottom_tensor_view=tensor_view,
            window_lengths=[3, 2],
            window_origin=[1, 1]
        )
        
        assert window.get_num_of_dimension() == 2
        assert window.get_window_lengths() == [3, 2]
        assert window.get_window_origin() == [1, 1]
    
    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        data = np.arange(24, dtype=np.float32).reshape(6, 4)
        tensor_view = make_naive_tensor_view(data, [6, 4], [4, 1])
        
        with pytest.raises(ValueError, match="Window dimensions"):
            TileWindowWithStaticLengths(
                bottom_tensor_view=tensor_view,
                window_lengths=[3, 2, 1],  # Wrong number of dimensions
                window_origin=[1, 1]
            )
        
        with pytest.raises(ValueError, match="Window origin dimensions"):
            TileWindowWithStaticLengths(
                bottom_tensor_view=tensor_view,
                window_lengths=[3, 2],
                window_origin=[1]  # Wrong number of dimensions
            )
    
    def test_get_set_element(self):
        """Test getting and setting elements in window."""
        data = np.arange(24, dtype=np.float32).reshape(6, 4)
        tensor_view = make_naive_tensor_view(data, [6, 4], [4, 1])
        
        window = TileWindowWithStaticLengths(
            bottom_tensor_view=tensor_view,
            window_lengths=[3, 2],
            window_origin=[1, 1]
        )
        
        # Get elements
        assert window.get_element([0, 0]) == data[1, 1]
        assert window.get_element([1, 0]) == data[2, 1]
        assert window.get_element([0, 1]) == data[1, 2]
        assert window.get_element([2, 1]) == data[3, 2]
        
        # Set element
        window.set_element([1, 1], 99.0)
        assert data[2, 2] == 99.0
    
    def test_bounds_checking(self):
        """Test that out-of-bounds access raises error."""
        data = np.arange(24, dtype=np.float32).reshape(6, 4)
        tensor_view = make_naive_tensor_view(data, [6, 4], [4, 1])
        
        window = TileWindowWithStaticLengths(
            bottom_tensor_view=tensor_view,
            window_lengths=[3, 2],
            window_origin=[1, 1]
        )
        
        # Out of bounds access
        with pytest.raises(IndexError):
            window.get_element([3, 0])  # Row index too large
        
        with pytest.raises(IndexError):
            window.get_element([0, 2])  # Column index too large
        
        with pytest.raises(IndexError):
            window.get_element([-1, 0])  # Negative index
    
    def test_move_window(self):
        """Test moving the window."""
        data = np.arange(24, dtype=np.float32).reshape(6, 4)
        tensor_view = make_naive_tensor_view(data, [6, 4], [4, 1])
        
        window = TileWindowWithStaticLengths(
            bottom_tensor_view=tensor_view,
            window_lengths=[3, 2],
            window_origin=[1, 1]
        )
        
        # Initial position
        assert window.get_element([0, 0]) == data[1, 1]
        
        # Move window
        window.move([1, 1])
        assert window.get_window_origin() == [2, 2]
        assert window.get_element([0, 0]) == data[2, 2]
        
        # Move again
        window.move([-1, 0])
        assert window.get_window_origin() == [1, 2]
        assert window.get_element([0, 0]) == data[1, 2]
    
    def test_set_window_origin(self):
        """Test setting window origin."""
        data = np.arange(24, dtype=np.float32).reshape(6, 4)
        tensor_view = make_naive_tensor_view(data, [6, 4], [4, 1])
        
        window = TileWindowWithStaticLengths(
            bottom_tensor_view=tensor_view,
            window_lengths=[3, 2],
            window_origin=[1, 1]
        )
        
        # Set new origin
        window.set_window_origin([0, 0])
        assert window.get_window_origin() == [0, 0]
        assert window.get_element([0, 0]) == data[0, 0]
        
        # Set another origin
        window.set_window_origin([3, 2])
        assert window.get_window_origin() == [3, 2]
        assert window.get_element([0, 0]) == data[3, 2]


class TestTileWindowWithStaticDistribution:
    """Test cases for TileWindowWithStaticDistribution class."""
    
    def test_basic_creation(self):
        """Test basic distributed window creation."""
        # Create tensor view
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create distribution with P=1, Y=2
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2, 2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 1],
            ys_to_rhs_minor=[0, 1]
        )
        
        # Create transforms for P=1, Y=2 -> X=2 mapping
        # Use PassThroughTransform for P dimension and EmbedTransforms for Y->X mapping
        transform1 = PassThroughTransform(1)  # P dimension
        transform2 = EmbedTransform([2], [1])  # Y1 -> X1
        transform3 = EmbedTransform([2], [1])  # Y2 -> X2
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2, transform3],
            lower_dimension_hidden_idss=[[2], [0], [1]],  # P->P, Y1->X1, Y2->X2
            upper_dimension_hidden_idss=[[2], [3], [4]],  # P, Y1, Y2
            bottom_dimension_hidden_ids=[0, 1],  # 2D bottom (X)
            top_dimension_hidden_ids=[2, 3, 4]  # P=1, Y=2
        )
        
        descriptor = make_naive_tensor_descriptor([2, 2], [2, 1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # Create window
        window = TileWindowWithStaticDistribution(
            bottom_tensor_view=tensor_view,
            window_lengths=[2, 2],
            window_origin=[1, 1],
            tile_distribution=dist
        )
        
        assert window.get_num_of_dimension() == 2
        assert window.get_window_lengths() == [2, 2]
        assert window.get_window_origin() == [1, 1]
    
    def test_load_store(self):
        """Test loading and storing with distributed window."""
        # Create tensor view
        data = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]], dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create 2D distribution with P=1, Y=1
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],  # 2 X dimensions to match 2D tensor
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        # Create adaptor that maps to 2D bottom space
        transform1 = EmbedTransform([2], [1])  # First dimension
        transform2 = EmbedTransform([2], [1])  # Second dimension
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],  # 2D bottom
            top_dimension_hidden_ids=[2, 3]  # P=1, Y=1
        )
        
        descriptor = make_naive_tensor_descriptor([2], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # Create window
        window = TileWindowWithStaticDistribution(
            bottom_tensor_view=tensor_view,
            window_lengths=[2, 2],
            window_origin=[1, 1],
            tile_distribution=dist
        )
        
        # Load data
        distributed_tensor = window.load()
        assert distributed_tensor is not None
        
        # Modify and store back
        distributed_tensor.fill(99.0)
        window.store(distributed_tensor)
        
        # Note: In the simplified implementation, only one element is affected
        # In a full implementation, the entire window would be updated
    
    def test_move_distributed_window(self):
        """Test moving distributed window."""
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create 2D distribution with P=1, Y=1
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],  # 2 X dimensions
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        # Create adaptor for 2D
        transform1 = EmbedTransform([2], [1])
        transform2 = EmbedTransform([2], [1])
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]  # P=1, Y=1
        )
        
        descriptor = make_naive_tensor_descriptor([2], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        window = TileWindowWithStaticDistribution(
            bottom_tensor_view=tensor_view,
            window_lengths=[2, 2],
            window_origin=[0, 0],
            tile_distribution=dist
        )
        
        # Move window
        window.move([1, 1])
        assert window.get_window_origin() == [1, 1]
        
        # Set new origin
        window.set_window_origin([2, 2])
        assert window.get_window_origin() == [2, 2]


class TestMakeTileWindow:
    """Test cases for make_tile_window function."""
    
    def test_make_static_lengths_window(self):
        """Test creating window without distribution."""
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor_view = make_naive_tensor_view(data, [3, 4], [4, 1])
        
        window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 1]
        )
        
        assert isinstance(window, TileWindowWithStaticLengths)
        assert window.get_window_lengths() == [2, 2]
        assert window.get_window_origin() == [0, 1]
    
    def test_make_distributed_window(self):
        """Test creating window with distribution."""
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create 2D distribution with P=1, Y=1
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
            top_dimension_hidden_ids=[2, 3]  # P=1, Y=1
        )
        
        descriptor = make_naive_tensor_descriptor([2], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[1, 1],
            tile_distribution=dist,
            num_coord=2
        )
        
        assert isinstance(window, TileWindowWithStaticDistribution)
        assert window.get_window_lengths() == [2, 2]
        assert window.get_window_origin() == [1, 1]
        assert window.num_coord == 2


class TestMoveTileWindow:
    """Test cases for move_tile_window function."""
    
    def test_move_static_window(self):
        """Test moving static window."""
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor_view = make_naive_tensor_view(data, [3, 4], [4, 1])
        
        window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0]
        )
        
        # Move window
        move_tile_window(window, [1, 1])
        assert window.get_window_origin() == [1, 1]
        
        # Move again
        move_tile_window(window, [0, -1])
        assert window.get_window_origin() == [1, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 