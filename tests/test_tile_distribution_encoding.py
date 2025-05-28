"""
Tests for tile_distribution_encoding module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.tile_distribution_encoding import (
    TileDistributionEncodingDetail, TileDistributionEncoding,
    make_embed_tile_distribution_encoding, make_reduce_tile_distribution_encoding
)


class TestTileDistributionEncodingDetail:
    """Test cases for TileDistributionEncodingDetail class."""
    
    def test_simple_detail(self):
        """Test detail computation for simple encoding."""
        # Simple 2D case with no replication
        detail = TileDistributionEncodingDetail.from_encoding(
            rs_lengths=[],
            hs_lengthss=[[4], [8]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[0, 0]
        )
        
        assert detail.ndim_rh_major == 3  # R + 2 H dimensions
        assert detail.ndim_span_major == 2  # 2 X dimensions
        assert detail.ndims_rhs_minor == [0, 1, 1]  # No R, 1 H each
        assert detail.max_ndim_rh_minor == 1
        
        # Check Y lengths
        assert detail.ys_lengths == [4, 8]
        
        # Check distributed spans
        assert detail.ndims_distributed_spans_minor == [1, 1]
    
    def test_detail_with_replication(self):
        """Test detail computation with replication dimensions."""
        detail = TileDistributionEncodingDetail.from_encoding(
            rs_lengths=[2, 3],
            hs_lengthss=[[4, 2], [8]],
            ps_to_rhss_major=[[0, 1], [0]],
            ps_to_rhss_minor=[[0, 0], [1]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[1, 0]
        )
        
        assert detail.ndim_rh_major == 3
        assert detail.ndims_rhs_minor == [2, 2, 1]  # 2 R, 2 H, 1 H
        assert detail.max_ndim_rh_minor == 2
        
        # Check RHS lengths
        assert detail.rhs_lengthss[0][:2] == [2, 3]  # R lengths
        assert detail.rhs_lengthss[1][:2] == [4, 2]  # First H lengths
        assert detail.rhs_lengthss[2][0] == 8  # Second H length
        
        # Check P ownership of R
        assert detail.does_p_own_r[0][0] == True   # P0 owns R0
        assert detail.does_p_own_r[0][1] == False  # P0 doesn't own R1
        assert detail.does_p_own_r[1][0] == False  # P1 doesn't own R0
        assert detail.does_p_own_r[1][1] == True   # P1 owns R1
    
    def test_y_to_span_mappings(self):
        """Test Y to span dimension mappings."""
        detail = TileDistributionEncodingDetail.from_encoding(
            rs_lengths=[],
            hs_lengthss=[[2, 3, 4], [5, 6]],
            ps_to_rhss_major=[[1, 2]],
            ps_to_rhss_minor=[[0, 1]],
            ys_to_rhs_major=[1, 1, 2],
            ys_to_rhs_minor=[1, 2, 0]
        )
        
        # Y0 maps to H0[1], Y1 maps to H0[2], Y2 maps to H1[0]
        assert detail.ys_to_span_major == [0, 0, 1]
        assert detail.ys_lengths == [3, 4, 5]
        
        # Check distributed span lengths
        assert detail.distributed_spans_lengthss[0][0] == 3  # Y0 -> H0[1]
        assert detail.distributed_spans_lengthss[0][1] == 4  # Y1 -> H0[2]
        assert detail.distributed_spans_lengthss[1][0] == 5  # Y2 -> H1[0]
    
    def test_ps_over_rs_derivative(self):
        """Test P over R derivative calculation."""
        detail = TileDistributionEncodingDetail.from_encoding(
            rs_lengths=[2, 3],
            hs_lengthss=[[4]],
            ps_to_rhss_major=[[0, 0, 1]],
            ps_to_rhss_minor=[[0, 1, 0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        # P0 maps to R0, R1, H0[0] with lengths 2, 3, 4
        # Derivative is calculated in reverse: 4*3=12 for R0, 4 for R1
        assert detail.ps_over_rs_derivative[0][0] == 12  # 4 * 3
        assert detail.ps_over_rs_derivative[0][1] == 4   # 4


class TestTileDistributionEncoding:
    """Test cases for TileDistributionEncoding class."""
    
    def test_basic_encoding(self):
        """Test basic encoding creation."""
        encoding = TileDistributionEncoding(
            rs_lengths=[2],
            hs_lengthss=[[4, 2], [3]],
            ps_to_rhss_major=[[0, 1]],
            ps_to_rhss_minor=[[0, 0]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[1, 0]
        )
        
        assert encoding.ndim_x == 2
        assert encoding.ndim_p == 1
        assert encoding.ndim_y == 2
        assert encoding.ndim_r == 1
        
        # Check that detail was computed
        assert encoding.detail is not None
        assert encoding.detail.ndim_rh_major == 3
    
    def test_h_dim_lengths_prefix_sum(self):
        """Test H dimension lengths prefix sum calculation."""
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[1, 4, 32], [4, 1, 4, 2, 4]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        prefix_sum = encoding.get_h_dim_lengths_prefix_sum()
        assert prefix_sum == [0, 3, 8]  # [0, 3, 3+5]
    
    def test_uniformed_idx_y_to_h(self):
        """Test uniformed Y to H index mapping."""
        encoding = TileDistributionEncoding(
            rs_lengths=[1],
            hs_lengthss=[[2, 3], [4, 5]],
            ps_to_rhss_major=[[0]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 2, 1],
            ys_to_rhs_minor=[0, 1, 1]
        )
        
        # Y0 -> H0[0], Y1 -> H1[1], Y2 -> H0[1]
        # With R dimension: h_dim_lengths = [1, 2, 2], prefix_sum = [0, 1, 3, 5]
        # Y0 -> major=1, minor=0 -> 1 + 0 = 1
        # Y1 -> major=2, minor=1 -> 3 + 1 = 4
        # Y2 -> major=1, minor=1 -> 1 + 1 = 2
        uniformed = encoding.get_uniformed_idx_y_to_h()
        assert uniformed == [1, 4, 2]
    
    def test_empty_encoding(self):
        """Test encoding with minimal dimensions."""
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[],
            ps_to_rhss_major=[],
            ps_to_rhss_minor=[],
            ys_to_rhs_major=[],
            ys_to_rhs_minor=[]
        )
        
        assert encoding.ndim_x == 0
        assert encoding.ndim_p == 0
        assert encoding.ndim_y == 0
        assert encoding.ndim_r == 0
    
    def test_repr(self):
        """Test string representation."""
        encoding = TileDistributionEncoding(
            rs_lengths=[2],
            hs_lengthss=[[4]],
            ps_to_rhss_major=[[0]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        repr_str = repr(encoding)
        assert "TileDistributionEncoding" in repr_str
        assert "ndim_x=1" in repr_str
        assert "ndim_p=1" in repr_str
        assert "ndim_y=1" in repr_str
        assert "ndim_r=1" in repr_str


class TestEmbedEncoding:
    """Test cases for make_embed_tile_distribution_encoding."""
    
    def test_simple_embed(self):
        """Test embedding two simple encodings."""
        outer = TileDistributionEncoding(
            rs_lengths=[1],
            hs_lengthss=[[2], [3]],
            ps_to_rhss_major=[[0]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        inner = TileDistributionEncoding(
            rs_lengths=[2],
            hs_lengthss=[[4], [5]],
            ps_to_rhss_major=[[0]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[2],
            ys_to_rhs_minor=[0]
        )
        
        embedded = make_embed_tile_distribution_encoding(outer, inner)
        
        # Check merged dimensions
        assert embedded.ndim_r == 2  # Unique R values: [1, 2]
        assert embedded.rs_lengths == [1, 2]
        assert embedded.hs_lengthss == [[2, 4], [3, 5]]
        
        # Check P mappings
        assert embedded.ndim_p == 2  # 1 + 1
        assert embedded.ps_to_rhss_major == [[0], [0]]
        assert embedded.ps_to_rhss_minor == [[0], [1]]  # Inner P shifted by 1
        
        # Check Y mappings
        assert embedded.ndim_y == 2  # 1 + 1
        assert embedded.ys_to_rhs_major == [1, 2]
        assert embedded.ys_to_rhs_minor == [0, 1]  # Inner Y shifted by 1
    
    def test_embed_with_multiple_h_dims(self):
        """Test embedding with multiple H dimensions."""
        outer = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[2, 3], [4]],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[1, 0]
        )
        
        inner = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[5, 6], [7, 8]],
            ps_to_rhss_major=[[1, 2]],
            ps_to_rhss_minor=[[0, 1]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[0, 0]
        )
        
        embedded = make_embed_tile_distribution_encoding(outer, inner)
        
        # Check merged H dimensions
        assert embedded.hs_lengthss == [[2, 3, 5, 6], [4, 7, 8]]
        
        # Check inner Y mappings are shifted correctly
        assert embedded.ys_to_rhs_minor == [1, 0, 2, 1]  # Outer: [1,0], Inner shifted: [0+2, 0+1]
    
    def test_embed_dimension_mismatch(self):
        """Test that embedding fails with mismatched X dimensions."""
        outer = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[2]],  # 1 X dimension
            ps_to_rhss_major=[],
            ps_to_rhss_minor=[],
            ys_to_rhs_major=[],
            ys_to_rhs_minor=[]
        )
        
        inner = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[3], [4]],  # 2 X dimensions
            ps_to_rhss_major=[],
            ps_to_rhss_minor=[],
            ys_to_rhs_major=[],
            ys_to_rhs_minor=[]
        )
        
        with pytest.raises(ValueError, match="same number of X dimensions"):
            make_embed_tile_distribution_encoding(outer, inner)


class TestReduceEncoding:
    """Test cases for make_reduce_tile_distribution_encoding."""
    
    def test_simple_reduce(self):
        """Test reducing a single X dimension."""
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[2, 3], [4, 5]],
            ps_to_rhss_major=[[1, 2]],
            ps_to_rhss_minor=[[0, 1]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[1, 0]
        )
        
        # Reduce X dimension 0 (first H dimension)
        reduced = make_reduce_tile_distribution_encoding(encoding, [0])
        
        # Check dimensions
        assert reduced.ndim_x == 1  # 2 - 1
        assert reduced.hs_lengthss == [[4, 5]]  # Only second H remains
        
        # H0 dimensions that weren't mapped by Y move to R
        assert reduced.ndim_r == 1  # H0[0] moves to R
        assert reduced.rs_lengths == [2]  # H0[0] had length 2
        
        # Y0 was mapped to reduced dimension, so it's removed
        assert reduced.ndim_y == 1  # Only Y1 remains
        assert reduced.ys_to_rhs_major == [1]  # Y1 still maps to H1 (now H0)
        assert reduced.ys_to_rhs_minor == [0]
    
    def test_reduce_with_replication(self):
        """Test reducing with existing replication dimensions."""
        encoding = TileDistributionEncoding(
            rs_lengths=[2, 3],
            hs_lengthss=[[4], [5, 6]],
            ps_to_rhss_major=[[0, 1], [0, 2]],
            ps_to_rhss_minor=[[0, 0], [1, 0]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[0, 1]
        )
        
        # Reduce X dimension 1 (second H dimension)
        reduced = make_reduce_tile_distribution_encoding(encoding, [1])
        
        # Original R dimensions preserved, H1[0] moves to R
        assert reduced.ndim_r == 3  # 2 + 1
        assert reduced.rs_lengths == [2, 3, 5]  # Original Rs + H1[0]
        
        # Only first H dimension remains
        assert reduced.ndim_x == 1
        assert reduced.hs_lengthss == [[4]]
        
        # Y1 was mapped to reduced dimension, so it's removed
        assert reduced.ndim_y == 1
        assert reduced.ys_to_rhs_major == [1]
        assert reduced.ys_to_rhs_minor == [0]
    
    def test_reduce_multiple_dimensions(self):
        """Test reducing multiple X dimensions."""
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[2], [3], [4]],
            ps_to_rhss_major=[[1], [2], [3]],
            ps_to_rhss_minor=[[0], [0], [0]],
            ys_to_rhs_major=[1, 2, 3],
            ys_to_rhs_minor=[0, 0, 0]
        )
        
        # Reduce X dimensions 0 and 2
        reduced = make_reduce_tile_distribution_encoding(encoding, [0, 2])
        
        # Only middle H dimension remains
        assert reduced.ndim_x == 1
        assert reduced.hs_lengthss == [[3]]
        
        # No H dimensions move to R (all were mapped by Y)
        assert reduced.ndim_r == 0
        assert reduced.rs_lengths == []
        
        # Only Y1 remains (mapped to the surviving dimension)
        assert reduced.ndim_y == 1
        assert reduced.ys_to_rhs_major == [1]
        assert reduced.ys_to_rhs_minor == [0]
    
    def test_reduce_empty(self):
        """Test reducing with no dimensions specified."""
        encoding = TileDistributionEncoding(
            rs_lengths=[2],
            hs_lengthss=[[3, 4]],
            ps_to_rhss_major=[[0, 1]],
            ps_to_rhss_minor=[[0, 0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[1]
        )
        
        # No reduction
        reduced = make_reduce_tile_distribution_encoding(encoding, [])
        
        # Everything should remain the same
        assert reduced.ndim_x == encoding.ndim_x
        assert reduced.ndim_r == encoding.ndim_r
        assert reduced.ndim_y == encoding.ndim_y
        assert reduced.rs_lengths == encoding.rs_lengths
        assert reduced.hs_lengthss == encoding.hs_lengthss


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 