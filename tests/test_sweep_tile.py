"""
Tests for sweep_tile module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.sweep_tile import (
    sweep_tile_span, sweep_tile_uspan, sweep_tile, 
    TileSweeper, make_tile_sweeper
)
from pytensor.tile_distribution import (
    TileDistributedSpan, TileDistributedIndex,
    make_tile_distributed_span, make_tile_distributed_index
)
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.tile_distribution import make_tile_distribution, make_tile_distribution_encoding
from pytensor.tensor_descriptor import TensorAdaptor, EmbedTransform, make_naive_tensor_descriptor


class TestSweepTileSpan:
    """Test cases for sweep_tile_span function."""
    
    def test_sweep_simple_span(self):
        """Test sweeping over a simple span."""
        # Create a span
        span = make_tile_distributed_span([2, 3])
        
        # Track visited indices
        visited = []
        
        def visitor(idx: TileDistributedIndex):
            visited.append(idx.partial_indices)
        
        # Sweep the span
        sweep_tile_span(span, visitor)
        
        # Check all indices were visited
        expected = [
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2]
        ]
        assert len(visited) == 6
        for exp in expected:
            assert exp in visited
    
    def test_sweep_1d_span(self):
        """Test sweeping over a 1D span."""
        span = make_tile_distributed_span([4])
        
        visited = []
        
        def visitor(idx: TileDistributedIndex):
            visited.append(idx.partial_indices[0])
        
        sweep_tile_span(span, visitor)
        
        assert visited == [0, 1, 2, 3]


class TestSweepTileUspan:
    """Test cases for sweep_tile_uspan function."""
    
    def test_sweep_with_unpacking(self):
        """Test sweeping with unpacking."""
        span = make_tile_distributed_span([4, 2])
        
        # Track calls
        calls = []
        
        def visitor(*indices):
            call = []
            for idx in indices:
                call.append(idx.partial_indices)
            calls.append(call)
        
        # Sweep with unpacking [2, 1] - process 2 from first dim at a time
        sweep_tile_uspan(span, visitor, [2, 1])
        
        # Should have 2 * 2 = 4 calls
        assert len(calls) == 4
    
    def test_sweep_no_unpacking(self):
        """Test sweeping without unpacking (default)."""
        span = make_tile_distributed_span([2, 2])
        
        calls = []
        
        def visitor(idx):
            calls.append(idx.partial_indices)
        
        sweep_tile_uspan(span, visitor)
        
        # Should visit each index individually
        assert len(calls) == 4


class TestSweepTile:
    """Test cases for sweep_tile function."""
    
    def test_sweep_tile_basic(self):
        """Test basic sweep_tile functionality."""
        # Create a simple distributed tensor type
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
            encoding=encoding,
            partition_index_func=lambda: [0]
        )
        
        # Create distributed tensor
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Track visits
        visits = []
        
        def process(idx):
            visits.append(idx)
        
        # Sweep tile
        sweep_tile(tensor, process)
        
        # Should have visited some indices
        assert len(visits) > 0
    
    def test_sweep_tile_with_unpacking(self):
        """Test sweep_tile with unpacking."""
        # Create distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4]],  # Single dimension with 4 elements
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = EmbedTransform([4], [1])
        
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
            partition_index_func=lambda: []
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Track calls with pairs
        pairs = []
        
        def process_pair(idx0, idx1):
            pairs.append((idx0, idx1))
        
        # Sweep with unpacking of 2
        sweep_tile(tensor, process_pair, [2])
        
        # Should have processed pairs
        assert len(pairs) > 0


class TestTileSweeper:
    """Test cases for TileSweeper class."""
    
    def test_tile_sweeper_creation(self):
        """Test creating a tile sweeper."""
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
            encoding=encoding,
            partition_index_func=lambda: [0]
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        visits = []
        
        def visitor(idx):
            visits.append(idx)
        
        # Create sweeper
        sweeper = make_tile_sweeper(tensor, visitor)
        
        # Get number of accesses
        num_access = sweeper.get_num_of_access()
        assert num_access > 0
        
        # Execute sweep
        sweeper()
        
        # Check visits
        assert len(visits) > 0
    
    def test_tile_sweeper_specific_access(self):
        """Test executing specific access in tile sweeper."""
        # Create simple 1D distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = EmbedTransform([4], [1])
        
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
            partition_index_func=lambda: []
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        visits = []
        
        def visitor(idx):
            visits.append(idx)
        
        sweeper = TileSweeper(
            distributed_tensor_type=tensor,
            func=visitor,
            unpacks_per_x_dim=[1]
        )
        
        # Execute specific access
        sweeper(0)  # First access
        assert len(visits) == 1
        
        visits.clear()
        sweeper(1)  # Second access
        assert len(visits) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 