"""
Tests for tile_distribution module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.tile_distribution import (
    TileDistributedSpan, TileDistributedIndex, TileDistributionEncoding,
    TileDistribution, make_tile_distributed_span, make_tile_distributed_index,
    make_tile_distribution_encoding, make_tile_distribution,
    make_static_tile_distribution, slice_distribution_from_x,
    StaticDistributedTensor
)
from pytensor.tensor_descriptor import (
    TensorAdaptor, TensorDescriptor, UnmergeTransform, make_naive_tensor_descriptor
)
from pytensor.tensor_coordinate import MultiIndex
from pytensor.tensor_adaptor import make_single_stage_tensor_adaptor


class TestTileDistributedSpan:
    """Test cases for TileDistributedSpan class."""
    
    def test_creation(self):
        """Test basic span creation."""
        span = TileDistributedSpan([2, 4, 8])
        
        assert span.partial_lengths == [2, 4, 8]
        assert span.is_static()
    
    def test_factory_function(self):
        """Test factory function."""
        span = make_tile_distributed_span([3, 5, 7])
        
        assert isinstance(span, TileDistributedSpan)
        assert span.partial_lengths == [3, 5, 7]
    
    def test_repr(self):
        """Test string representation."""
        span = TileDistributedSpan([1, 2, 3])
        assert repr(span) == "TileDistributedSpan([1, 2, 3])"


class TestTileDistributedIndex:
    """Test cases for TileDistributedIndex class."""
    
    def test_creation(self):
        """Test basic index creation."""
        index = TileDistributedIndex([0, 1, 2])
        
        assert index.partial_indices == [0, 1, 2]
        assert index.is_static()
    
    def test_factory_function(self):
        """Test factory function."""
        index = make_tile_distributed_index([1, 3, 5])
        
        assert isinstance(index, TileDistributedIndex)
        assert index.partial_indices == [1, 3, 5]
    
    def test_repr(self):
        """Test string representation."""
        index = TileDistributedIndex([2, 4, 6])
        assert repr(index) == "TileDistributedIndex([2, 4, 6])"


class TestTileDistributionEncoding:
    """Test cases for TileDistributionEncoding class."""
    
    def test_simple_encoding(self):
        """Test simple distribution encoding."""
        # Simple 2D tensor with no replication
        encoding = TileDistributionEncoding(
            rs_lengths=[],  # No replication
            hs_lengthss=[[4], [8]],  # 4x8 tensor
            ps_to_rhss_major=[[1]],  # Single partition maps to first H
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 2],  # Y0->H0, Y1->H1
            ys_to_rhs_minor=[0, 0]
        )
        
        assert encoding.ndim_x == 2
        assert encoding.ndim_p == 1
        assert encoding.ndim_y == 2
        assert encoding.ndim_r == 0
    
    def test_encoding_with_replication(self):
        """Test encoding with replication dimensions."""
        encoding = TileDistributionEncoding(
            rs_lengths=[2, 3],  # 2 replication dimensions
            hs_lengthss=[[4, 2], [8]],  # Hierarchical dimensions
            ps_to_rhss_major=[[0, 1], [0]],  # P0 maps to R and H0, P1 maps to R
            ps_to_rhss_minor=[[0, 0], [1]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[1, 0]
        )
        
        assert encoding.ndim_x == 2
        assert encoding.ndim_p == 2
        assert encoding.ndim_y == 2
        assert encoding.ndim_r == 2
    
    def test_invalid_encoding(self):
        """Test invalid encoding validation."""
        # Mismatched P dimension mappings
        with pytest.raises(ValueError, match="P dimension mappings"):
            TileDistributionEncoding(
                rs_lengths=[],
                hs_lengthss=[[4]],
                ps_to_rhss_major=[[1], [2]],  # Length 2
                ps_to_rhss_minor=[[0]],       # Length 1 - mismatch!
                ys_to_rhs_major=[1],
                ys_to_rhs_minor=[0]
            )
        
        # Mismatched Y dimension mappings
        with pytest.raises(ValueError, match="Y dimension mappings"):
            TileDistributionEncoding(
                rs_lengths=[],
                hs_lengthss=[[4]],
                ps_to_rhss_major=[[1]],
                ps_to_rhss_minor=[[0]],
                ys_to_rhs_major=[1, 2],  # Length 2
                ys_to_rhs_minor=[0]      # Length 1 - mismatch!
            )
    
    def test_get_distributed_spans(self):
        """Test getting distributed spans."""
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[4, 2], [8, 4]],  # 2 X dimensions with hierarchical splits
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 2],  # Y0 maps to X0, Y1 maps to X1
            ys_to_rhs_minor=[0, 1]   # Different minor indices
        )
        
        spans = encoding.get_distributed_spans()
        
        assert len(spans) == 2
        assert spans[0].partial_lengths == [4]  # X0: H[0] is distributed
        assert spans[1].partial_lengths == [4]  # X1: H[1] is distributed
    
    def test_factory_function(self):
        """Test factory function."""
        encoding = make_tile_distribution_encoding(
            rs_lengths=[2],
            hs_lengthss=[[4, 2]],
            ps_to_rhss_major=[[0, 1]],
            ps_to_rhss_minor=[[0, 0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[1]
        )
        
        assert isinstance(encoding, TileDistributionEncoding)
        assert encoding.ndim_r == 1


class TestTileDistribution:
    """Test cases for TileDistribution class."""
    
    def test_simple_distribution(self):
        """Test simple tile distribution."""
        # Create simple adaptor and descriptor
        from pytensor.tensor_descriptor import EmbedTransform
        
        # Simple 1D to 1D mapping using UnmergeTransform
        transform = UnmergeTransform([4])
        
        # Create adaptor using factory function
        adaptor = make_single_stage_tensor_adaptor(
            transforms=[transform],
            lower_dimension_old_top_idss=[[0]],  # Transform outputs to hidden dim 0
            upper_dimension_new_top_idss=[[1]]   # Transform takes hidden dim 1
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[4]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        dist = TileDistribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        assert dist.ndim_x == 1
        assert dist.ndim_y == 1  # Y dimension from descriptor
        assert dist.ndim_p == 0  # P = top_dims - Y_dims = 1 - 1 = 0
        assert dist.ndim_r == 0
    
    def test_partition_index(self):
        """Test partition index handling."""
        # Create mock adaptor and descriptor
        transform = UnmergeTransform([8])
        
        adaptor = make_single_stage_tensor_adaptor(
            transforms=[transform],
            lower_dimension_old_top_idss=[[0]],
            upper_dimension_new_top_idss=[[1, 2]]  # Takes 2 upper dims
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[8]],
            ps_to_rhss_major=[[1], [1]],
            ps_to_rhss_minor=[[0], [0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        # Custom partition function
        def my_partition_func():
            return [1, 3]
        
        dist = TileDistribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding,
            partition_index_func=my_partition_func
        )
        
        assert dist.get_partition_index() == [1, 3]
    
    def test_default_partition_index(self):
        """Test default partition index."""
        transform = UnmergeTransform([8])
        
        adaptor = make_single_stage_tensor_adaptor(
            transforms=[transform],
            lower_dimension_old_top_idss=[[0]],
            upper_dimension_new_top_idss=[[1, 2]]
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[8]],
            ps_to_rhss_major=[[1], [1]],
            ps_to_rhss_minor=[[0], [0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        dist = TileDistribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # Should default to zeros
        assert dist.get_partition_index() == [0, 0]
    
    def test_calculate_index(self):
        """Test calculating X index from partition index."""
        # Create a simple transform
        transform = UnmergeTransform([2, 4])  # 2D unmerge transform
        
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1, 2]],  # Takes 2 dims
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1, 2]  # P=1, Y=1
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[2, 4]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[1]
        )
        
        dist = TileDistribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # Calculate index for partition [1]
        x_index = dist.calculate_index([1])
        assert isinstance(x_index, MultiIndex)
    
    def test_get_lengths(self):
        """Test getting X dimension lengths."""
        transform = UnmergeTransform([96])  # 12 * 8 = 96
        
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1, 2, 3]],
            bottom_dimension_hidden_ids=[0, 1],  # 2 bottom dims
            top_dimension_hidden_ids=[2, 3]      # 2 top dims
        )
        
        descriptor = make_naive_tensor_descriptor([2, 2], [2, 1])
        
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[3, 4], [2, 4]],  # 3*4=12, 2*4=8
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[1, 1]
        )
        
        dist = TileDistribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        lengths = dist.get_lengths()
        assert lengths == [12, 8]
    
    def test_get_y_indices_from_distributed_indices(self):
        """Test mapping distributed indices to Y indices."""
        transform = UnmergeTransform([64])
        
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1, 2, 3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3, 4]  # P=1, Y=2
        )
        
        descriptor = make_naive_tensor_descriptor([2, 2], [2, 1])
        
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[4, 2], [4, 2]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[0, 0]
        )
        
        dist = TileDistribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # Create distributed indices
        dstr_indices = [
            TileDistributedIndex([1]),  # X0
            TileDistributedIndex([2])   # X1
        ]
        
        y_indices = dist.get_y_indices_from_distributed_indices(dstr_indices)
        assert y_indices == [1, 2]
    
    def test_is_static(self):
        """Test static check."""
        transform = UnmergeTransform([8])
        
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[8]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        dist = TileDistribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        # In Python implementation, is_static returns False
        assert not dist.is_static()
    
    def test_repr(self):
        """Test string representation."""
        transform = UnmergeTransform([8])
        
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1, 2]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1, 2]
        )
        
        descriptor = make_naive_tensor_descriptor([2], [1])
        
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[8]],
            ps_to_rhss_major=[[1], [1]],
            ps_to_rhss_minor=[[0], [0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        dist = TileDistribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        repr_str = repr(dist)
        assert "TileDistribution" in repr_str
        assert "ndim_x=1" in repr_str
        assert "ndim_y=1" in repr_str  # Y dimension from descriptor
        assert "ndim_p=1" in repr_str
    
    def test_factory_function(self):
        """Test factory function."""
        transform = UnmergeTransform([8])
        
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        descriptor = make_naive_tensor_descriptor([4], [1])
        
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[8]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        assert isinstance(dist, TileDistribution)


def test_make_static_tile_distribution():
    # Create a simple encoding
    encoding = make_tile_distribution_encoding(
        rs_lengths=[2],  # One replication dimension
        hs_lengthss=[[4, 4]],  # One X dimension with two H dimensions
        ps_to_rhss_major=[[0, 1]],  # One P dimension mapping to R and first H
        ps_to_rhss_minor=[[0, 0]],  # First R and first H
        ys_to_rhs_major=[1],  # One Y dimension mapping to first H
        ys_to_rhs_minor=[1]  # Second H
    )
    
    # Create static distribution
    distribution = make_static_tile_distribution(encoding)
    
    # Test basic properties
    assert distribution.ndim_x == 1
    assert distribution.ndim_y == 1
    assert distribution.ndim_p == 1
    assert distribution.ndim_r == 1
    
    # Test adaptor properties
    assert distribution.ps_ys_to_xs_adaptor.is_static()
    assert distribution.ys_to_d_descriptor.is_static()
    
    # Test lengths
    lengths = distribution.get_lengths()
    assert len(lengths) == 1
    assert lengths[0] == 16  # 4 * 4
    
    # Test distributed spans
    spans = distribution.get_distributed_spans()
    assert len(spans) == 1
    assert len(spans[0].partial_lengths) == 1
    assert spans[0].partial_lengths[0] == 4

def test_make_static_tile_distribution_complex():
    # Create a more complex encoding
    encoding = make_tile_distribution_encoding(
        rs_lengths=[2, 2],  # Two replication dimensions
        hs_lengthss=[[4, 4], [2, 2]],  # Two X dimensions with two H dimensions each
        ps_to_rhss_major=[[0, 1], [0, 2]],  # Two P dimensions
        ps_to_rhss_minor=[[0, 0], [1, 0]],  # Mapping to different R and H dimensions
        ys_to_rhs_major=[1, 2],  # Two Y dimensions
        ys_to_rhs_minor=[1, 1]  # Mapping to second H of each X
    )
    
    # Create static distribution
    distribution = make_static_tile_distribution(encoding)
    
    # Test basic properties
    assert distribution.ndim_x == 2
    assert distribution.ndim_y == 2
    assert distribution.ndim_p == 2
    assert distribution.ndim_r == 2
    
    # Test adaptor properties
    assert distribution.ps_ys_to_xs_adaptor.is_static()
    assert distribution.ys_to_d_descriptor.is_static()
    
    # Test lengths
    lengths = distribution.get_lengths()
    assert len(lengths) == 2
    assert lengths[0] == 16  # 4 * 4
    assert lengths[1] == 4   # 2 * 2
    
    # Test distributed spans
    spans = distribution.get_distributed_spans()
    assert len(spans) == 2
    assert len(spans[0].partial_lengths) == 1
    assert len(spans[1].partial_lengths) == 1
    assert spans[0].partial_lengths[0] == 4
    assert spans[1].partial_lengths[0] == 2

def test_make_static_tile_distribution_invalid():
    # Test with invalid encoding (invalid major_id: 2 for only 1 X dimension)
    with pytest.raises(ValueError):
        encoding = make_tile_distribution_encoding(
            rs_lengths=[2],
            hs_lengthss=[[4, 4]],
            ps_to_rhss_major=[[2, 0]],  # 2 is invalid: only 0 (R) and 1 (X0) are valid
            ps_to_rhss_minor=[[0, 0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[1]
        )
        make_static_tile_distribution(encoding)


class TestStaticDistributedTensor:
    """Test cases for StaticDistributedTensor class."""

    def test_creation(self):
        """Test basic creation of StaticDistributedTensor."""
        # Create a simple tile distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[2],
            hs_lengthss=[[4, 4]],
            ps_to_rhss_major=[[0, 1]],
            ps_to_rhss_minor=[[0, 0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[1]
        )
        distribution = make_static_tile_distribution(encoding)
        # Create a thread buffer (simplified for testing)
        thread_buffer = [0] * 16  # 4 * 4 = 16
        tensor = StaticDistributedTensor(thread_buffer, distribution)
        assert tensor.distribution == distribution
        assert tensor.thread_buffer == thread_buffer

    def test_get_y_sliced_thread_data(self):
        """Test getting a slice of the thread buffer."""
        encoding = make_tile_distribution_encoding(
            rs_lengths=[2],
            hs_lengthss=[[4, 4]],
            ps_to_rhss_major=[[0, 1]],
            ps_to_rhss_minor=[[0, 0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[1]
        )
        distribution = make_static_tile_distribution(encoding)
        thread_buffer = list(range(16))  # [0, 1, 2, ..., 15]
        tensor = StaticDistributedTensor(thread_buffer, distribution)
        # Slice from index 4 to 8
        y_slice_origins = [4]
        y_slice_lengths = [4]
        sliced_data = tensor.get_y_sliced_thread_data(y_slice_origins, y_slice_lengths)
        assert sliced_data == [4, 5, 6, 7]

    def test_set_y_sliced_thread_data(self):
        """Test setting a slice of the thread buffer."""
        encoding = make_tile_distribution_encoding(
            rs_lengths=[2],
            hs_lengthss=[[4, 4]],
            ps_to_rhss_major=[[0, 1]],
            ps_to_rhss_minor=[[0, 0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[1]
        )
        distribution = make_static_tile_distribution(encoding)
        thread_buffer = [0] * 16
        tensor = StaticDistributedTensor(thread_buffer, distribution)
        # Set slice from index 4 to 8
        y_slice_origins = [4]
        y_slice_lengths = [4]
        new_data = [10, 11, 12, 13]
        tensor.set_y_sliced_thread_data(y_slice_origins, y_slice_lengths, new_data)
        assert tensor.thread_buffer[4:8] == new_data


class TestSliceDistributionFromX:
    """Test cases for slice_distribution_from_x function."""

    def test_slice_distribution_from_x(self):
        # Create a simple tile distribution encoding
        encoding = make_tile_distribution_encoding(
            rs_lengths=[2, 2],
            hs_lengthss=[[4, 4], [4, 4]],
            ps_to_rhss_major=[[0, 1], [0, 1]],
            ps_to_rhss_minor=[[0, 0], [1, 1]],
            ys_to_rhs_major=[0, 1],
            ys_to_rhs_minor=[0, 1]
        )
        distribution = make_static_tile_distribution(encoding)

        # Slice the distribution along X dimensions
        x_slice_begins = [1, 1]
        x_slice_ends = [3, 3]
        new_distribution, y_slice_origins, y_slice_lengths = slice_distribution_from_x(distribution, x_slice_begins, x_slice_ends)

        # Verify the new distribution
        assert new_distribution.ndim_x == distribution.ndim_x
        assert len(new_distribution.encoding.hs_lengthss) == len(distribution.encoding.hs_lengthss)
        assert new_distribution.ndim_p == distribution.ndim_p
        assert new_distribution.ndim_y == distribution.ndim_y

        # Verify Y slice origins and lengths
        assert y_slice_origins == x_slice_begins
        assert y_slice_lengths == [2, 2]  # 3 - 1 = 2 for each dimension

    def test_slice_distribution_from_x_invalid(self):
        # Create a simple tile distribution encoding
        encoding = make_tile_distribution_encoding(
            rs_lengths=[2, 2],
            hs_lengthss=[[4, 4], [4, 4]],
            ps_to_rhss_major=[[0, 1], [0, 1]],
            ps_to_rhss_minor=[[0, 0], [1, 1]],
            ys_to_rhs_major=[0, 1],
            ys_to_rhs_minor=[0, 1]
        )
        distribution = make_static_tile_distribution(encoding)

        # Test invalid slice parameters
        with pytest.raises(ValueError):
            slice_distribution_from_x(distribution, [1], [3, 3])  # Mismatched lengths

        with pytest.raises(ValueError):
            slice_distribution_from_x(distribution, [1, 1, 1], [3, 3, 3])  # Too many dimensions


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 