"""
Tests for LoadStoreTraits class from tile_window module.
Updated to verify C++ alignment behavior.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.tile_window import LoadStoreTraits
from pytensor.tile_distribution import (
    TileDistribution, TileDistributionEncoding,
    make_tile_distribution_encoding, make_tile_distribution
)
from pytensor.tensor_descriptor import (
    TensorAdaptor, UnmergeTransform, make_naive_tensor_descriptor, 
    EmbedTransform, PassThroughTransform
)
from pytensor.space_filling_curve import SpaceFillingCurve


class TestLoadStoreTraits:
    """Test cases for LoadStoreTraits class with C++ alignment behavior."""
    
    def _create_simple_tile_distribution(self, y_lengths, y_strides):
        """Helper to create a simple tile distribution for testing."""
        # Create encoding for P=1, Y=len(y_lengths)
        ndim_y = len(y_lengths)
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[y_lengths],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1] * ndim_y,
            ys_to_rhs_minor=list(range(ndim_y))
        )
        
        # Create transforms for P=1, Y=ndim_y -> X=ndim_y mapping
        transforms = [PassThroughTransform(1)]  # P dimension
        for i in range(ndim_y):
            transforms.append(EmbedTransform([y_lengths[i]], [1]))  # Yi -> Xi
        
        lower_dimension_hidden_idss = [[ndim_y]] + [[i] for i in range(ndim_y)]  # P->P, Yi->Xi
        upper_dimension_hidden_idss = [[ndim_y + 1 + i] for i in range(ndim_y + 1)]  # P, Y1, Y2, ...
        bottom_dimension_hidden_ids = list(range(ndim_y))  # X dimensions
        top_dimension_hidden_ids = [ndim_y + 1 + i for i in range(ndim_y + 1)]  # P, Y1, Y2, ...
        
        adaptor = TensorAdaptor(
            transforms=transforms,
            lower_dimension_hidden_idss=lower_dimension_hidden_idss,
            upper_dimension_hidden_idss=upper_dimension_hidden_idss,
            bottom_dimension_hidden_ids=bottom_dimension_hidden_ids,
            top_dimension_hidden_ids=top_dimension_hidden_ids
        )
        
        descriptor = make_naive_tensor_descriptor(y_lengths, y_strides)
        
        return make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
    
    def test_basic_creation(self):
        """Test basic LoadStoreTraits creation."""
        # Create simple 2D distribution
        tile_dist = self._create_simple_tile_distribution([4, 4], [4, 1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        assert traits.tile_distribution == tile_dist
        assert traits.data_type == np.float32
        assert traits.ndim_y == 2
        assert traits.packed_size == 1  # Always 1 in Python
        assert hasattr(traits, 'vector_dim_y')
        assert hasattr(traits, 'scalar_per_vector')
        assert hasattr(traits, 'scalars_per_access')
        assert hasattr(traits, 'sfc_ys')
        assert hasattr(traits, 'num_access')
    
    def test_single_y_dimension(self):
        """Test LoadStoreTraits with single Y dimension - C++ aligned behavior."""
        # Create 1D distribution
        tile_dist = self._create_simple_tile_distribution([8], [1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        assert traits.ndim_y == 1
        assert traits.vector_dim_y == 0  # Only dimension available
        assert traits.scalar_per_vector == 1  # C++ alignment: single element access
        assert traits.scalars_per_access == [1]  # C++ alignment: [1] for single dimension
        assert traits.num_access == 8  # C++ alignment: 8 individual accesses
    
    def test_multiple_y_dimensions(self):
        """Test LoadStoreTraits with multiple Y dimensions - C++ aligned behavior."""
        # Create 2D distribution with different strides
        tile_dist = self._create_simple_tile_distribution([4, 6], [4, 1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        assert traits.ndim_y == 2
        # Should choose dimension 1 (index 1) since it has stride 1 and length 6
        assert traits.vector_dim_y == 1
        assert traits.scalar_per_vector == 1  # C++ alignment: single element access
        assert traits.scalars_per_access == [1, 1]  # C++ alignment: [1, 1] for both dimensions
        assert traits.num_access == 24  # C++ alignment: 4 * 6 = 24 individual accesses
    
    def test_vector_dim_selection(self):
        """Test that vector dimension is correctly selected based on stride and length."""
        # Test case 1: First dimension has stride 1, longer length
        tile_dist1 = self._create_simple_tile_distribution([8, 4], [1, 4])
        traits1 = LoadStoreTraits(tile_dist1, np.float32)
        assert traits1.vector_dim_y == 0
        assert traits1.scalar_per_vector == 1  # C++ alignment
        
        # Test case 2: Second dimension has stride 1, longer length
        tile_dist2 = self._create_simple_tile_distribution([4, 8], [4, 1])
        traits2 = LoadStoreTraits(tile_dist2, np.float32)
        assert traits2.vector_dim_y == 1
        assert traits2.scalar_per_vector == 1  # C++ alignment
        
        # Test case 3: Both have stride 1, second has longer length
        tile_dist3 = self._create_simple_tile_distribution([4, 6], [1, 1])
        traits3 = LoadStoreTraits(tile_dist3, np.float32)
        assert traits3.vector_dim_y == 1
        assert traits3.scalar_per_vector == 1  # C++ alignment
    
    def test_no_stride_one_dimension(self):
        """Test behavior when no dimension has stride 1."""
        # Create distribution where no dimension has stride 1
        tile_dist = self._create_simple_tile_distribution([4, 4], [4, 4])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        # Should default to first dimension with length 1
        assert traits.vector_dim_y == 0
        assert traits.scalar_per_vector == 1
        assert traits.scalars_per_access == [1, 1]  # C++ alignment
    
    def test_get_y_indices(self):
        """Test get_y_indices method with C++ aligned behavior."""
        # Create 2D distribution
        tile_dist = self._create_simple_tile_distribution([2, 3], [2, 1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        # Test first access
        indices0 = traits.get_y_indices(0)
        assert indices0 == [0, 0]  # First access should be at origin
        
        # Test second access - C++ follows space-filling curve pattern
        indices1 = traits.get_y_indices(1)
        assert indices1 == [0, 1]  # C++ alignment: next in space-filling curve
        
        # Test third access
        indices2 = traits.get_y_indices(2)
        assert indices2 == [0, 2]  # Continue in space-filling curve
    
    def test_get_vectorized_access_info_single_dim(self):
        """Test get_vectorized_access_info with single Y dimension - C++ aligned."""
        # Create 1D distribution
        tile_dist = self._create_simple_tile_distribution([4], [1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        # Test first access - C++ alignment: single element access
        access_info = traits.get_vectorized_access_info(0)
        
        assert access_info['base_indices'] == [0]
        assert access_info['vector_indices'] == [[0]]  # C++ alignment: single element
        assert access_info['vector_dim'] == 0
        assert access_info['vector_size'] == 1  # C++ alignment: single element
    
    def test_get_vectorized_access_info_multi_dim(self):
        """Test get_vectorized_access_info with multiple Y dimensions - C++ aligned."""
        # Create 2D distribution
        tile_dist = self._create_simple_tile_distribution([2, 3], [2, 1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        # Test first access - C++ alignment: single element access
        access_info = traits.get_vectorized_access_info(0)
        
        assert access_info['base_indices'] == [0, 0]
        assert access_info['vector_indices'] == [[0, 0]]  # C++ alignment: single element
        assert access_info['vector_dim'] == 1  # Second dimension is vector dimension
        assert access_info['vector_size'] == 1  # C++ alignment: single element
    
    def test_space_filling_curve_creation(self):
        """Test that space-filling curve is created correctly for C++ alignment."""
        # Create 2D distribution
        tile_dist = self._create_simple_tile_distribution([2, 3], [2, 1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        # Check space-filling curve properties
        sfc = traits.sfc_ys
        assert isinstance(sfc, SpaceFillingCurve)
        assert sfc.tensor_lengths == [2, 3]
        assert sfc.scalars_per_access == [1, 1]  # C++ alignment: [1, 1] for both dimensions
        assert sfc.snake_curved == True
    
    def test_num_access_calculation(self):
        """Test that number of accesses is calculated correctly for C++ alignment."""
        # Test 1D case
        tile_dist1 = self._create_simple_tile_distribution([8], [1])
        traits1 = LoadStoreTraits(tile_dist1, np.float32)
        assert traits1.num_access == 8  # C++ alignment: 8 individual accesses
        
        # Test 2D case
        tile_dist2 = self._create_simple_tile_distribution([4, 6], [4, 1])
        traits2 = LoadStoreTraits(tile_dist2, np.float32)
        assert traits2.num_access == 24  # C++ alignment: 4 * 6 = 24 individual accesses
        
        # Test 3D case
        tile_dist3 = self._create_simple_tile_distribution([2, 3, 4], [2, 3, 1])
        traits3 = LoadStoreTraits(tile_dist3, np.float32)
        assert traits3.num_access == 24  # C++ alignment: 2 * 3 * 4 = 24 individual accesses
    
    def test_scalars_per_access_calculation(self):
        """Test scalars_per_access calculation for C++ alignment."""
        # Single dimension
        tile_dist1 = self._create_simple_tile_distribution([4], [1])
        traits1 = LoadStoreTraits(tile_dist1, np.float32)
        assert traits1.scalars_per_access == [1]  # C++ alignment: [1]
        
        # Multiple dimensions, first dimension vectorized
        tile_dist2 = self._create_simple_tile_distribution([4, 3], [1, 3])
        traits2 = LoadStoreTraits(tile_dist2, np.float32)
        assert traits2.scalars_per_access == [1, 1]  # C++ alignment: [1, 1]
        
        # Multiple dimensions, second dimension vectorized
        tile_dist3 = self._create_simple_tile_distribution([3, 4], [3, 1])
        traits3 = LoadStoreTraits(tile_dist3, np.float32)
        assert traits3.scalars_per_access == [1, 1]  # C++ alignment: [1, 1]
    
    def test_access_pattern_consistency(self):
        """Test that access patterns are consistent across multiple accesses."""
        # Create 2D distribution
        tile_dist = self._create_simple_tile_distribution([2, 3], [2, 1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        # Test multiple accesses
        for i in range(traits.num_access):
            access_info = traits.get_vectorized_access_info(i)
            
            # Check that vector indices follow C++ alignment (single element)
            vector_indices = access_info['vector_indices']
            assert len(vector_indices) == 1  # C++ alignment: single element access
            
            # Check that base indices are valid
            base_idx = access_info['base_indices']
            assert len(base_idx) == 2  # 2D distribution
            y_lengths = traits.tile_distribution.ys_to_d_descriptor.get_lengths()
            assert all(0 <= idx < y_lengths[dim] 
                      for dim, idx in enumerate(base_idx))
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small dimensions
        tile_dist1 = self._create_simple_tile_distribution([1, 1], [1, 1])
        traits1 = LoadStoreTraits(tile_dist1, np.float32)
        assert traits1.num_access == 1  # C++ alignment: 1 * 1 = 1 access
        assert traits1.scalar_per_vector == 1
        
        # Test with large dimensions
        tile_dist2 = self._create_simple_tile_distribution([16, 8], [16, 1])
        traits2 = LoadStoreTraits(tile_dist2, np.float32)
        assert traits2.vector_dim_y == 1
        assert traits2.scalar_per_vector == 1  # C++ alignment
        assert traits2.num_access == 128  # C++ alignment: 16 * 8 = 128 accesses
        
        # Test with different data types
        tile_dist3 = self._create_simple_tile_distribution([4, 4], [4, 1])
        traits3 = LoadStoreTraits(tile_dist3, np.int32)
        assert traits3.data_type == np.int32
    
    def test_three_dimensional_access(self):
        """Test LoadStoreTraits with three Y dimensions - C++ aligned."""
        # Create 3D distribution
        tile_dist = self._create_simple_tile_distribution([2, 3, 4], [2, 3, 1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        assert traits.ndim_y == 3
        assert traits.vector_dim_y == 2  # Third dimension has stride 1
        assert traits.scalar_per_vector == 1  # C++ alignment: single element
        assert traits.scalars_per_access == [1, 1, 1]  # C++ alignment: [1, 1, 1]
        assert traits.num_access == 24  # C++ alignment: 2 * 3 * 4 = 24 accesses
        
        # Test access info
        access_info = traits.get_vectorized_access_info(0)
        assert access_info['base_indices'] == [0, 0, 0]
        assert access_info['vector_dim'] == 2
        assert access_info['vector_size'] == 1  # C++ alignment: single element
        assert len(access_info['vector_indices']) == 1  # C++ alignment: single element
        
        # Check vector indices
        expected_indices = [[0, 0, 0]]  # C++ alignment: single element access
        assert access_info['vector_indices'] == expected_indices
    
    def test_invalid_access_index(self):
        """Test behavior with invalid access indices."""
        # Create simple distribution
        tile_dist = self._create_simple_tile_distribution([2, 2], [2, 1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        # Test access beyond num_access
        # This should still work but return indices that may be out of bounds
        access_info = traits.get_vectorized_access_info(traits.num_access)
        assert access_info is not None  # Should not raise exception
        
        # Test negative access index
        access_info_neg = traits.get_vectorized_access_info(-1)
        assert access_info_neg is not None  # Should not raise exception


class TestLoadStoreTraitsIntegration:
    """Integration tests for LoadStoreTraits with more complex scenarios."""
    
    def test_complex_tile_distribution(self):
        """Test LoadStoreTraits with a more complex tile distribution."""
        # Create a complex distribution with multiple transforms
        encoding = make_tile_distribution_encoding(
            rs_lengths=[2],
            hs_lengthss=[[4, 3], [2, 2]],
            ps_to_rhss_major=[[1, 1]],
            ps_to_rhss_minor=[[0, 1]],
            ys_to_rhs_major=[1, 1, 1, 1],
            ys_to_rhs_minor=[0, 1, 2, 3]
        )
        
        # Create complex adaptor
        transform1 = PassThroughTransform(1)  # P dimension
        transform2 = EmbedTransform([4], [1])  # Y1 -> X1
        transform3 = EmbedTransform([3], [1])  # Y2 -> X2
        transform4 = EmbedTransform([2], [1])  # Y3 -> X3
        transform5 = EmbedTransform([2], [1])  # Y4 -> X4
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2, transform3, transform4, transform5],
            lower_dimension_hidden_idss=[[4], [0], [1], [2], [3]],
            upper_dimension_hidden_idss=[[4], [5], [6], [7], [8]],
            bottom_dimension_hidden_ids=[0, 1, 2, 3],
            top_dimension_hidden_ids=[4, 5, 6, 7, 8]
        )
        
        descriptor = make_naive_tensor_descriptor([4, 3, 2, 2], [6, 2, 1, 1])
        
        tile_dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        assert traits.ndim_y == 4
        # Should choose dimension with stride 1 and maximum length
        assert traits.vector_dim_y == 2  # Third dimension has stride 1 and length 2
        assert traits.scalar_per_vector == 1  # C++ alignment: single element
        assert traits.num_access == 48  # C++ alignment: 4 * 3 * 2 * 2 = 48 accesses
    
    def test_memory_access_pattern(self):
        """Test that memory access patterns follow space-filling curve."""
        # Create 2D distribution
        tile_dist = self._create_simple_tile_distribution([3, 4], [3, 1])
        
        traits = LoadStoreTraits(tile_dist, np.float32)
        
        # Collect all access patterns
        all_accesses = []
        for i in range(traits.num_access):
            access_info = traits.get_vectorized_access_info(i)
            all_accesses.append(access_info['base_indices'])
        
        # Check that accesses follow a reasonable pattern
        # First access should be [0, 0]
        assert all_accesses[0] == [0, 0]
        
        # Should have 12 accesses total (3 * 4 = 12) for C++ alignment
        assert len(all_accesses) == 12
        
        # Check that we don't have duplicate accesses
        unique_accesses = set(tuple(access) for access in all_accesses)
        assert len(unique_accesses) == len(all_accesses)
    
    def _create_simple_tile_distribution(self, y_lengths, y_strides):
        """Helper method for integration tests."""
        ndim_y = len(y_lengths)
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[y_lengths],
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1] * ndim_y,
            ys_to_rhs_minor=list(range(ndim_y))
        )
        
        transforms = [PassThroughTransform(1)]
        for i in range(ndim_y):
            transforms.append(EmbedTransform([y_lengths[i]], [1]))
        
        adaptor = TensorAdaptor(
            transforms=transforms,
            lower_dimension_hidden_idss=[[ndim_y]] + [[i] for i in range(ndim_y)],
            upper_dimension_hidden_idss=[[ndim_y + 1 + i] for i in range(ndim_y + 1)],
            bottom_dimension_hidden_ids=list(range(ndim_y)),
            top_dimension_hidden_ids=[ndim_y + 1 + i for i in range(ndim_y + 1)]
        )
        
        descriptor = make_naive_tensor_descriptor(y_lengths, y_strides)
        
        return make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        ) 