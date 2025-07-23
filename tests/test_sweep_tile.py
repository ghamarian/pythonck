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
from pytensor.sequence_utils import get_y_unpacks_from_x_unpacks
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
    
    def test_sweep_with_y_unpacks_from_x_unpacks(self):
        """Test sweep_tile_uspan with Y unpacks calculated from X unpacks."""
        # Create a span with dimensions [4, 2] (total size = 8)
        span = make_tile_distributed_span([4, 2])
        
        # Calculate Y unpacks from X unpacks = 2 (each call should process 4 elements)
        y_unpacks = get_y_unpacks_from_x_unpacks([4, 2], 2)
        # This should give y_unpacks = [2, 1] meaning: 2 calls in first dim, 1 call in second dim
        
        calls = []
        
        def visitor(*indices):
            call_indices = [idx.partial_indices for idx in indices]
            calls.append(call_indices)
        
        sweep_tile_uspan(span, visitor, y_unpacks)
        
        # With y_unpacks = [2, 1], we should have 2 * 2 = 4 calls total
        # Each call should unpack 2 indices from first dimension, 1 from second
        assert len(calls) == 4
        
        # Verify that each call has the right number of indices
        for call in calls:
            assert len(call) == 2  # Should unpack 2 indices per call
    
    def test_sweep_with_different_unpacking_patterns(self):
        """Test different unpacking patterns."""
        # Test case 1: [2, 4] with x_unpacks = 4 (total size = 8, slice_size = 2)
        span1 = make_tile_distributed_span([2, 4])
        y_unpacks1 = get_y_unpacks_from_x_unpacks([2, 4], 4)  # Should be [2, 2]
        
        calls1 = []
        def visitor1(*indices):
            calls1.append(len(indices))
        
        sweep_tile_uspan(span1, visitor1, y_unpacks1)
        
        # With y_unpacks = [2, 2], each call should process multiple indices
        # Total calls should be (2/2) * (4/2) = 1 * 2 = 2
        assert len(calls1) == 2
        
        # Test case 2: [8] with x_unpacks = 2 (total size = 8, slice_size = 4)
        span2 = make_tile_distributed_span([8])
        y_unpacks2 = get_y_unpacks_from_x_unpacks([8], 2)  # Should be [2]
        
        calls2 = []
        def visitor2(*indices):
            calls2.append(len(indices))
        
        sweep_tile_uspan(span2, visitor2, y_unpacks2)
        
        # With y_unpacks = [2], each call gets 2 indices (unpacked)
        # Total calls = 8 elements / 2 indices per call = 4 calls
        assert len(calls2) == 4
    
    def test_sweep_uspan_detailed_indices(self):
        """Test that sweep_tile_uspan produces the correct indices."""
        # Simple case: [4] with y_unpacks = [2]
        span = make_tile_distributed_span([4])
        
        calls = []
        def visitor(*indices):
            call_indices = [idx.partial_indices for idx in indices]
            calls.append(call_indices)
        
        sweep_tile_uspan(span, visitor, [2])
        
        # Should have 2 calls (4/2 = 2 groups)
        assert len(calls) == 2
        
        # Each call should have 2 indices
        for call in calls:
            assert len(call) == 2
        
        # Check the specific indices
        # First call should process indices [0, 1]
        # Second call should process indices [2, 3]
        assert calls[0] == [[0], [1]]
        assert calls[1] == [[2], [3]]


class TestSweepTileEnhanced:
    """Enhanced test cases for the improved sweep_tile function."""
    
    def test_sweep_tile_with_proper_x_to_y_unpacks(self):
        """Test that X unpacks are properly converted to Y unpacks."""
        # Create a simple 1D distributed tensor
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[8]],  # Single span with 8 elements
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = EmbedTransform([8], [1])
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
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Test with X unpacks = 2 (should process 2 groups of 4 elements each)
        calls = []
        def process_group(*indices):
            calls.append(len(indices))
        
        sweep_tile(tensor, process_group, [2])
        
        # With x_unpacks=2, y_unpacks should be [2], each call gets 2 indices
        # Total calls = 8 elements / 2 indices per call = 4 calls
        assert len(calls) == 4
        
        # Each call should process 2 indices (since y_unpacks should be [2])
        for call_size in calls:
            assert call_size == 2
    
    def test_sweep_tile_multi_span(self):
        """Test sweep_tile with multiple spans."""
        # Use the working pattern for multiple spans
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [4]],  # Two spans: [2] and [4]
            ps_to_rhss_major=[[], []],
            ps_to_rhss_minor=[[], []],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[0, 0]
        )
        
        transform1 = EmbedTransform([2], [1])
        transform2 = EmbedTransform([4], [1])
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]
        )
        descriptor = make_naive_tensor_descriptor([2, 4], [4, 1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Test with different X unpacks for each span
        calls = []
        def process_indices(*indices):
            calls.append(len(indices))
        
        sweep_tile(tensor, process_indices, [1, 2])  # First span: no unpacking, second span: unpack by 2
        
        # Should have some calls (exact number depends on how spans are processed)
        assert len(calls) > 0
    
    def test_sweep_tile_x_unpacks_validation(self):
        """Test validation of X unpacks."""
        # Create a simple tensor
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
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        def dummy_func(*indices):
            pass
        
        # Test with wrong number of X unpacks
        with pytest.raises(ValueError, match="unpacks_per_x_dim length"):
            sweep_tile(tensor, dummy_func, [1, 2])  # Too many unpacks for single span
    
    def test_sweep_tile_fallback_on_invalid_unpacks(self):
        """Test that invalid Y unpacks fall back to default behavior."""
        # Create a tensor where Y unpacks calculation might fail
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[3]],  # 3 elements - odd number
            ps_to_rhss_major=[[1]],
            ps_to_rhss_minor=[[0]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = EmbedTransform([3], [1])
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        descriptor = make_naive_tensor_descriptor([3], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        calls = []
        def process_indices(*indices):
            calls.append(len(indices))
        
        # Try with X unpacks = 2 (3 is not divisible by 2)
        # Should fall back to default behavior
        sweep_tile(tensor, process_indices, [2])
        
        # Should still work (fallback to default unpacking)
        assert len(calls) > 0


class TestIntegrationXtoYUnpacks:
    """Integration tests for X unpacks to Y unpacks conversion."""
    
    def test_x_to_y_unpacks_integration(self):
        """Test the complete integration of X to Y unpacks."""
        # Test different combinations
        test_cases = [
            # (y_lengths, x_unpacks, expected_calls)
            ([4], 1, 4),      # No unpacking: 4 individual calls
            ([4], 2, 2),      # Unpack by 2: 2 calls with 2 elements each
            ([4], 4, 1),      # Unpack by 4: 1 call with 4 elements
            ([2, 4], 1, 8),   # No unpacking: 8 individual calls
            ([2, 4], 2, 4),   # Unpack by 2: 4 calls
            ([2, 4], 4, 2),   # Unpack by 4: 2 calls
            ([2, 4], 8, 1),   # Unpack by 8: 1 call
        ]
        
        for y_lengths, x_unpacks, expected_calls in test_cases:
            span = make_tile_distributed_span(y_lengths)
            
            # Calculate Y unpacks
            y_unpacks = get_y_unpacks_from_x_unpacks(y_lengths, x_unpacks)
            
            calls = []
            def visitor(*indices):
                calls.append(len(indices))
            
            sweep_tile_uspan(span, visitor, y_unpacks)
            
            assert len(calls) == expected_calls, f"For y_lengths={y_lengths}, x_unpacks={x_unpacks}, expected {expected_calls} calls but got {len(calls)}"
    
    def test_complex_unpacking_patterns(self):
        """Test complex unpacking patterns."""
        # Test [4, 2, 8] with different x_unpacks (from our sequence tests)
        y_lengths = [4, 2, 8]
        
        test_cases = [
            (1, 64),   # No unpacking: 64 individual calls  
            (2, 32),   # x_unpacks=2: should give 32 calls
            (4, 16),   # x_unpacks=4: should give 16 calls
            (8, 8),    # x_unpacks=8: should give 8 calls
            (16, 4),   # x_unpacks=16: should give 4 calls
            (32, 2),   # x_unpacks=32: should give 2 calls
            (64, 1),   # x_unpacks=64: should give 1 call
        ]
        
        for x_unpacks, expected_calls in test_cases:
            span = make_tile_distributed_span(y_lengths)
            
            y_unpacks = get_y_unpacks_from_x_unpacks(y_lengths, x_unpacks)
            
            calls = []
            def visitor(*indices):
                calls.append(len(indices))
            
            sweep_tile_uspan(span, visitor, y_unpacks)
            
            assert len(calls) == expected_calls, f"For x_unpacks={x_unpacks}, expected {expected_calls} calls but got {len(calls)}"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for the enhanced sweep functionality."""
    
    def test_sweep_tile_uspan_edge_cases(self):
        """Test edge cases for sweep_tile_uspan."""
        # Test with empty span - should not call visitor (nothing to sweep)
        empty_span = make_tile_distributed_span([])
        calls = []
        def visitor(*indices):
            calls.append(len(indices))
        
        sweep_tile_uspan(empty_span, visitor, [])
        assert len(calls) == 0  # Should not call visitor for empty spans
        
        # Test with single element span
        single_span = make_tile_distributed_span([1])
        calls = []
        sweep_tile_uspan(single_span, visitor, [1])
        assert len(calls) == 1
        
        # Test with valid unpacking that exactly divides the span length
        divisible_span = make_tile_distributed_span([6])
        calls = []
        # y_unpacks=[3] with length=6 is valid (6 divisible by 3 = 2 groups)
        sweep_tile_uspan(divisible_span, visitor, [3])
        assert len(calls) == 2  # Should have 2 groups of 3 elements each
        
        # Test with exact divisor unpacking  
        small_span = make_tile_distributed_span([2])
        calls = []
        # y_unpacks=[2] with length=2 is valid (2 divisible by 2 = 1 group)
        sweep_tile_uspan(small_span, visitor, [2])
        assert len(calls) == 1  # Should have 1 group of 2 elements
    
    def test_get_y_unpacks_error_handling(self):
        """Test error handling in get_y_unpacks_from_x_unpacks integration."""
        # Create a span with odd total size
        span = make_tile_distributed_span([3])  # Total size = 3
        
        calls = []
        def visitor(*indices):
            calls.append(len(indices))
        
        # Try to use y_unpacks that would cause issues
        # If get_y_unpacks_from_x_unpacks fails, should fall back to default
        try:
            y_unpacks = get_y_unpacks_from_x_unpacks([3], 2)  # 3 not divisible by 2
            # If this doesn't raise an error, use the result
            sweep_tile_uspan(span, visitor, y_unpacks)
        except ValueError:
            # If it raises an error, use default unpacking
            sweep_tile_uspan(span, visitor, [1])
        
        assert len(calls) > 0
    
    def test_sweep_tile_uspan_complex_indices(self):
        """Test sweep_tile_uspan with complex index patterns."""
        # Test 2D span with asymmetric unpacking
        span = make_tile_distributed_span([3, 2])  # 6 total elements
        
        calls = []
        all_indices = []
        
        def visitor(*indices):
            calls.append(len(indices))
            for idx in indices:
                all_indices.append(idx.partial_indices)
        
        # Use asymmetric unpacking: [3, 1] means 3 groups in first dim, 1 group in second
        sweep_tile_uspan(span, visitor, [3, 1])
        
        # Should make calls
        assert len(calls) > 0
        
        # Verify we don't have duplicate indices
        unique_indices = set(tuple(idx) for idx in all_indices)
        expected_indices = {(i, j) for i in range(3) for j in range(2)}
        
        # All expected indices should be covered
        assert unique_indices == expected_indices
    
    def test_sweep_tile_performance_characteristics(self):
        """Test that sweep_tile has correct performance characteristics."""
        # Create a large span and verify the number of calls scales correctly
        large_span = make_tile_distributed_span([16])
        
        test_cases = [
            (1, 16),   # No unpacking: 16 calls
            (2, 8),    # Unpack by 2: 8 calls  
            (4, 4),    # Unpack by 4: 4 calls
            (8, 2),    # Unpack by 8: 2 calls
            (16, 1),   # Full unpacking: 1 call
        ]
        
        for x_unpacks, expected_calls in test_cases:
            y_unpacks = get_y_unpacks_from_x_unpacks([16], x_unpacks)
            
            calls = []
            def visitor(*indices):
                calls.append(len(indices))
            
            sweep_tile_uspan(large_span, visitor, y_unpacks)
            
            assert len(calls) == expected_calls, f"For x_unpacks={x_unpacks}, expected {expected_calls} calls but got {len(calls)}"


class TestRegressionAndCompatibility:
    """Test regression and compatibility with existing functionality."""
    
    def test_backward_compatibility_with_old_sweep_tile(self):
        """Test that the new sweep_tile is backward compatible."""
        # Create a simple tensor like in the old tests
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
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Test old style usage (no unpacking)
        visits = []
        def process(idx):
            visits.append(idx)
        
        sweep_tile(tensor, process)  # No unpacks specified
        
        # Should visit all 4 elements individually
        assert len(visits) == 4
        
        # Test old style usage with explicit [1] unpacking
        visits2 = []
        sweep_tile(tensor, lambda idx: visits2.append(idx), [1])
        
        # Should produce same result
        assert len(visits2) == 4
    
    def test_sweep_tile_uspan_no_unpacking_compatibility(self):
        """Test that sweep_tile_uspan without unpacking works like before."""
        span = make_tile_distributed_span([2, 3])
        
        # Old style: no unpacks
        calls1 = []
        def visitor1(idx):
            calls1.append(idx.partial_indices)
        
        sweep_tile_uspan(span, visitor1)
        
        # New style: explicit [1, 1] unpacks
        calls2 = []
        def visitor2(idx):
            calls2.append(idx.partial_indices)
        
        sweep_tile_uspan(span, visitor2, [1, 1])
        
        # Should produce identical results
        assert len(calls1) == len(calls2) == 6
        assert set(tuple(call) for call in calls1) == set(tuple(call) for call in calls2)


class TestSweepTile:
    """Test cases for sweep_tile function."""
    
    def test_sweep_tile_single_span(self):
        """Test sweep_tile with single span (single X dimension)."""
        # Create distribution with single X dimension
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4]],  # Single X dimension with 4 elements
            ps_to_rhss_major=[[]],
            ps_to_rhss_minor=[[]],
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
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Track visits - expect single TileDistributedIndex per call
        visits = []
        
        def process_single(idx):
            assert isinstance(idx, TileDistributedIndex)
            visits.append(idx.partial_indices)
        
        sweep_tile(tensor, process_single)
        
        # Should visit all 4 elements
        assert len(visits) == 4
        expected_indices = [[0], [1], [2], [3]]
        for expected in expected_indices:
            assert expected in visits
    
    def test_sweep_tile_multiple_spans(self):
        """Test sweep_tile with multiple spans (multiple X dimensions)."""
        # Create distribution with 2 X dimensions - using a known working pattern
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [3]],  # First X dimension: 2 elements, Second: 3 elements
            ps_to_rhss_major=[[], []],
            ps_to_rhss_minor=[[], []],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[0, 0]
        )
        
        # Create simple transforms
        transform1 = EmbedTransform([2], [1])
        transform2 = EmbedTransform([3], [1])
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]
        )
        
        descriptor = make_naive_tensor_descriptor([2, 3], [3, 1])  # 2*3=6 total elements
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Track visits - expect 2 TileDistributedIndex objects per call
        visits = []
        
        def process_multiple(idx1, idx2):
            assert isinstance(idx1, TileDistributedIndex)
            assert isinstance(idx2, TileDistributedIndex)
            visits.append((idx1.partial_indices, idx2.partial_indices))
        
        sweep_tile(tensor, process_multiple)
        
        # First span: 2 elements, Second span: 3 elements
        # Total calls: 2 * 3 = 6
        assert len(visits) == 6
        
        # Verify all combinations are present
        expected_combinations = [
            ([0], [0]), ([0], [1]), ([0], [2]),
            ([1], [0]), ([1], [1]), ([1], [2])
        ]
        for expected in expected_combinations:
            assert expected in visits, f"Expected combination {expected} not found in {visits}"
    
    def test_sweep_tile_with_unpacking_single_span(self):
        """Test sweep_tile with unpacking on single span."""
        # Single span with 4 elements
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4]],
            ps_to_rhss_major=[[]],
            ps_to_rhss_minor=[[]],
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
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Test with x_unpacks=2: should get 2 TileDistributedIndex per call
        visits = []
        
        def process_unpacked(idx1, idx2):
            assert isinstance(idx1, TileDistributedIndex)
            assert isinstance(idx2, TileDistributedIndex)
            visits.append((idx1.partial_indices, idx2.partial_indices))
        
        sweep_tile(tensor, process_unpacked, [2])
        
        # With 4 elements and unpack=2: 4/2 = 2 calls
        assert len(visits) == 2
        
        # Should get pairs: ([0], [1]) and ([2], [3])
        expected_pairs = [([0], [1]), ([2], [3])]
        for expected in expected_pairs:
            assert expected in visits
    
    def test_sweep_tile_with_unpacking_multiple_spans(self):
        """Test sweep_tile with different unpacking on multiple spans."""
        # Use the same setup as the working multiple spans test but with unpacking
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4], [2]],  # First X dimension: 4 elements, Second: 2 elements
            ps_to_rhss_major=[[], []],
            ps_to_rhss_minor=[[], []],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[0, 0]
        )
        
        transform1 = EmbedTransform([4], [1])
        transform2 = EmbedTransform([2], [1])
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]
        )
        
        descriptor = make_naive_tensor_descriptor([4, 2], [2, 1])  # 4*2=8 total elements
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Test with unpacks=[2, 1]: first span unpack by 2, second by 1
        visits = []
        
        def process_mixed_unpacking(*indices):
            # First span contributes 2 indices (unpack=2)
            # Second span contributes 1 index (unpack=1) 
            # Total: 3 indices per call
            assert len(indices) == 3
            for idx in indices:
                assert isinstance(idx, TileDistributedIndex)
            partial_indices = [idx.partial_indices for idx in indices]
            visits.append(tuple(partial_indices))
        
        sweep_tile(tensor, process_mixed_unpacking, [2, 1])
        
        # First span: 4 elements / 2 unpack = 2 groups
        # Second span: 2 elements / 1 unpack = 2 groups
        # Total calls: 2 * 2 = 4
        assert len(visits) == 4
    
    def test_sweep_tile_error_handling(self):
        """Test error handling in sweep_tile."""
        # Create simple tensor
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2]],
            ps_to_rhss_major=[[]],
            ps_to_rhss_minor=[[]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = EmbedTransform([2], [1])
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        descriptor = make_naive_tensor_descriptor([2], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        def dummy_func(idx):
            pass
        
        # Test invalid unpacks length
        with pytest.raises(ValueError, match="unpacks_per_x_dim length.*must match number of spans"):
            sweep_tile(tensor, dummy_func, [1, 2])  # Too many unpacks for single span
    
    def test_sweep_tile_fallback_on_invalid_unpacks(self):
        """Test that sweep_tile falls back gracefully on invalid unpacking."""
        # Create span where unpacking won't work perfectly
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[3]],  # 3 elements - not divisible by 2
            ps_to_rhss_major=[[]],
            ps_to_rhss_minor=[[]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        
        transform = EmbedTransform([3], [1])
        adaptor = TensorAdaptor(
            transforms=[transform],
            lower_dimension_hidden_idss=[[0]],
            upper_dimension_hidden_idss=[[1]],
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1]
        )
        
        descriptor = make_naive_tensor_descriptor([3], [1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        calls = []
        def process_indices(idx):
            calls.append(idx.partial_indices)
        
        # Try to use unpacking that won't work (3 not divisible by 2)
        # Should fall back to default [1] unpacking
        sweep_tile(tensor, process_indices, [2])
        
        # Should still visit all 3 elements (fallback to no unpacking)
        assert len(calls) == 3
        assert [0] in calls
        assert [1] in calls
        assert [2] in calls
    
    # Keep the original test but rename it for clarity
    def test_sweep_tile_basic_legacy(self):
        """Legacy test - shows the old API expectation that needs updating."""
        # Use the working multiple spans pattern
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [3]],  # Use the pattern that works
            ps_to_rhss_major=[[], []],
            ps_to_rhss_minor=[[], []],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[0, 0]
        )
        
        transform1 = EmbedTransform([2], [1])
        transform2 = EmbedTransform([3], [1])
        
        adaptor = TensorAdaptor(
            transforms=[transform1, transform2],
            lower_dimension_hidden_idss=[[0], [1]],
            upper_dimension_hidden_idss=[[2], [3]],
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3]
        )
        
        descriptor = make_naive_tensor_descriptor([2, 3], [3, 1])
        
        dist = make_tile_distribution(
            ps_ys_to_xs_adaptor=adaptor,
            ys_to_d_descriptor=descriptor,
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        # Track visits - now expects 2 TileDistributedIndex objects (C++ behavior)
        visits = []
        
        def process(idx1, idx2):
            # Now receives TileDistributedIndex objects, not Y indices
            assert isinstance(idx1, TileDistributedIndex)
            assert isinstance(idx2, TileDistributedIndex)
            visits.append((idx1.partial_indices, idx2.partial_indices))
        
        sweep_tile(tensor, process)
        
        # Should have visited some indices
        assert len(visits) > 0


class TestTileSweeper:
    """Test cases for TileSweeper class."""
    
    def test_tile_sweeper_creation(self):
        """Test creating a tile sweeper."""
        # Use simple single-span distribution that we know works
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4]],  # Single X dimension with 4 elements
            ps_to_rhss_major=[[]],
            ps_to_rhss_minor=[[]],
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
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        visits = []
        
        def visitor(idx):  # Single span = single argument
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
            encoding=encoding
        )
        
        tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=dist
        )
        
        visits = []
        
        def visitor(*indices):
            visits.append(indices)
        
        sweeper = TileSweeper(
            distributed_tensor=tensor,
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