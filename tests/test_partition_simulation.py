"""
Tests for partition simulation and coordinate calculation.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.partition_simulation import (
    create_partition_index_func, 
    PartitionSimulator, 
    PartitionConfig,
    with_thread_position
)
from pytensor.tile_distribution import make_tile_distribution, make_tile_distribution_encoding
from pytensor.tensor_descriptor import TensorAdaptor, EmbedTransform, make_naive_tensor_descriptor
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_window import TileWindowWithStaticDistribution


class TestPartitionSimulation:
    """Test cases for partition simulation."""
    
    def test_basic_partition_functions(self):
        """Test basic partition index function creation."""
        # Test 1D partition (NDimP = 1)
        func_1d = create_partition_index_func(ndim_p=1, warp_id=2, lane_id=5)
        assert func_1d() == [5]  # Should return [lane_id]
        
        # Test 2D partition (NDimP = 2)
        func_2d = create_partition_index_func(ndim_p=2, warp_id=2, lane_id=5)
        assert func_2d() == [2, 5]  # Should return [warp_id, lane_id]
    
    def test_partition_simulator(self):
        """Test the PartitionSimulator class."""
        config = PartitionConfig(warp_size=64, num_warps=4)
        simulator = PartitionSimulator(config)
        
        # Test setting thread position
        simulator.set_thread_position(2, 10)
        assert simulator.get_warp_id() == 2
        assert simulator.get_lane_id() == 10
        
        # Test setting from global ID
        simulator.set_thread_position_from_global_id(130)  # 130 = 2*64 + 2
        assert simulator.get_warp_id() == 2
        assert simulator.get_lane_id() == 2
        
        # Test partition index functions
        func_1d = simulator.create_partition_index_func(1)
        func_2d = simulator.create_partition_index_func(2)
        
        assert func_1d() == [2]
        assert func_2d() == [2, 2]
    
    def test_coordinate_calculation_with_different_partitions(self):
        """Test how different partition indices affect coordinate calculation."""
        # Create a simple distribution setup
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
        
        # Test different partition indices
        partition_indices = [0, 1, 2, 3]
        calculated_indices = []
        
        for lane_id in partition_indices:
            # Set the global thread position to simulate this thread
            from pytensor.partition_simulation import set_global_thread_position
            set_global_thread_position(warp_id=0, lane_id=lane_id)
            
            dist = make_tile_distribution(
                ps_ys_to_xs_adaptor=adaptor,
                ys_to_d_descriptor=descriptor,
                encoding=encoding
            )
            
            # Calculate the index that this partition would map to
            x_index = dist.calculate_index()
            calculated_indices.append(x_index.to_list())
            
            print(f"Partition [0, {lane_id}] -> X index: {x_index.to_list()}")
        
        # Check that different partition indices produce different X indices
        unique_indices = set(tuple(idx) for idx in calculated_indices)
        print(f"Unique X indices: {unique_indices}")
        
        # We should get different indices for different partition values
        assert len(unique_indices) > 1, f"All partition indices mapped to the same X index: {calculated_indices}"
    
    def test_tile_window_coordinate_precomputation(self):
        """Test how partition indices affect tile window coordinate precomputation."""
        # Create tensor view
        data = np.zeros((4, 4), dtype=np.float32)
        tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create distribution setup
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
        
        # Test different partition indices and their effect on precomputed coordinates
        for lane_id in [0, 1, 2, 3]:
            # Set the global thread position to simulate this thread
            from pytensor.partition_simulation import set_global_thread_position
            set_global_thread_position(warp_id=0, lane_id=lane_id)
            
            dist = make_tile_distribution(
                ps_ys_to_xs_adaptor=adaptor,
                ys_to_d_descriptor=descriptor,
                encoding=encoding
            )
            
            # Create tile window
            window = TileWindowWithStaticDistribution(
                bottom_tensor_view=tensor_view,
                window_lengths=[2, 2],
                window_origin=[0, 0],
                tile_distribution=dist
            )
            
            # Check the precomputed coordinates
            print(f"\nPartition [0, {lane_id}]:")
            for i, (adaptor_coord, bottom_coord) in enumerate(window.pre_computed_coords):
                print(f"  Coord bundle {i}:")
                print(f"    Adaptor coord bottom index: {adaptor_coord.get_bottom_index().to_list()}")
                print(f"    Bottom tensor coord offset: {bottom_coord.get_offset()}")
                print(f"    Bottom tensor coord indices: {bottom_coord.get_index().to_list()}")
    
    def test_context_manager(self):
        """Test the thread position context manager."""
        from pytensor.partition_simulation import get_simulated_warp_id, get_simulated_lane_id, set_global_thread_position
        
        # Reset to initial position first
        set_global_thread_position(0, 0)
        
        # Initial position should be (0, 0)
        assert get_simulated_warp_id() == 0
        assert get_simulated_lane_id() == 0
        
        # Use context manager to temporarily change position
        with with_thread_position(2, 5):
            assert get_simulated_warp_id() == 2
            assert get_simulated_lane_id() == 5
        
        # Should restore to original position
        assert get_simulated_warp_id() == 0
        assert get_simulated_lane_id() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 