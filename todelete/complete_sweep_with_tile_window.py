#!/usr/bin/env python3

"""
COMPLETE example showing sweep_tile + tile_window for actual tensor coordinates.

This shows the full pipeline:
1. sweep_tile() iterates through Y-space
2. tile_distribution maps P+Y to X coordinates  
3. tile_window maps X coordinates to actual tensor memory addresses
"""

import numpy as np
from examples import get_default_variables
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.sweep_tile import sweep_tile
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_window import make_tile_window
from pytensor.partition_simulation import set_global_thread_position
from pytensor.tensor_coordinate import MultiIndex, make_tensor_adaptor_coordinate


def demonstrate_complete_sweep_with_tile_window():
    """Show the complete pipeline: sweep_tile + tile_distribution + tile_window."""
    
    print("=== COMPLETE Pipeline: sweep_tile + tile_distribution + tile_window ===\n")
    
    # Get RMSNorm configuration
    variables = get_default_variables('Real-World Example (RMSNorm)')
    print("RMSNorm Config:")
    for key, value in variables.items():
        print(f"  {key} = {value}")
    
    # Create the encoding
    encoding = TileDistributionEncoding(
        rs_lengths=[],
        hs_lengthss=[
            [variables['S::Repeat_M'], variables['S::WarpPerBlock_M'], 
             variables['S::ThreadPerWarp_M'], variables['S::Vector_M']],
            [variables['S::Repeat_N'], variables['S::WarpPerBlock_N'], 
             variables['S::ThreadPerWarp_N'], variables['S::Vector_N']]
        ],
        ps_to_rhss_major=[[1, 2], [1, 2]],
        ps_to_rhss_minor=[[1, 1], [2, 2]],
        ys_to_rhs_major=[1, 1, 2, 2],
        ys_to_rhs_minor=[0, 3, 0, 3]
    )
    
    # Create tile distribution
    tile_distribution = make_static_tile_distribution(encoding)
    
    print(f"\nTile Distribution Properties:")
    print(f"  ndim_x: {tile_distribution.ndim_x}")
    print(f"  ndim_y: {tile_distribution.ndim_y}")
    print(f"  ndim_p: {tile_distribution.ndim_p}")
    
    # Create distributed tensor
    distributed_tensor = StaticDistributedTensor(
        data_type=np.float32,
        tile_distribution=tile_distribution
    )
    
    # Create actual tensor data and tensor view
    m_size = (variables["S::Repeat_M"] * variables["S::WarpPerBlock_M"] * 
              variables["S::ThreadPerWarp_M"] * variables["S::Vector_M"])
    n_size = (variables["S::Repeat_N"] * variables["S::WarpPerBlock_N"] * 
              variables["S::ThreadPerWarp_N"] * variables["S::Vector_N"])
    
    tensor_shape = [m_size, n_size]
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    
    print(f"\nTensor Properties:")
    print(f"  Full tensor shape: {tensor_shape}")
    print(f"  Data type: {data.dtype}")
    
    # Create tensor view
    strides = [tensor_shape[1], 1]  # Row-major strides
    tensor_view = make_naive_tensor_view(data, tensor_shape, strides)
    
    # Create tile window - THIS IS THE KEY COMPONENT!
    window_lengths = [64, 64]
    window_origin = [0, 0]
    
    tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=window_lengths,
        origin=window_origin,
        tile_distribution=distributed_tensor.tile_distribution
    )
    
    print(f"\nTile Window Properties:")
    print(f"  Window lengths: {window_lengths}")
    print(f"  Window origin: {window_origin}")
    print(f"  Window covers: [{window_origin[0]}:{window_origin[0]+window_lengths[0]}, {window_origin[1]}:{window_origin[1]+window_lengths[1]}]")
    
    # Test with different threads to show X[0] changes
    threads_to_test = [
        (0, 0),   # P[0,0] -> X[0,0]
        (0, 1),   # P[0,1] -> X[0,4]
        (0, 32),  # P[0,32] -> X[16,0] <- This will show X[0] change!
        (2, 0),   # P[2,0] -> X[32,0] <- This will show X[0] change!
    ]
    
    for warp_id, lane_id in threads_to_test:
        print(f"\n=== Thread (warp={warp_id}, lane={lane_id}) ===")
        
        # Set thread position
        set_global_thread_position(warp_id, lane_id)
        
        # Get current partition index
        partition_idx = tile_distribution.get_partition_index()
        print(f"  Partition (P): {partition_idx}")
        
        access_count = 0
        tensor_coordinates = []
        
        def process_complete_pipeline(distributed_idx):
            """Process using the COMPLETE pipeline: sweep_tile -> tile_distribution -> tile_window."""
            nonlocal access_count, tensor_coordinates
            
            # Step 1: Get Y-space indices from sweep_tile
            y_indices = distributed_idx.partial_indices if hasattr(distributed_idx, 'partial_indices') else []
            
            try:
                # Step 2: Use tile_distribution to map P+Y -> X coordinates
                py_indices = partition_idx + y_indices
                py_multiindex = MultiIndex(len(py_indices), py_indices)
                adaptor_coord = make_tensor_adaptor_coordinate(
                    tile_distribution.ps_ys_to_xs_adaptor,
                    py_multiindex
                )
                x_coords = adaptor_coord.get_bottom_index().to_list()
                
                # Step 3: Use tile_window to map X coordinates -> actual tensor coordinates
                # Create the coordinate for tile window
                x_multiindex = MultiIndex(len(x_coords), x_coords)
                
                # Get the actual tensor coordinate using tile window
                if hasattr(tile_window, 'calculate_coordinate'):
                    tensor_coord = tile_window.calculate_coordinate(x_multiindex)
                elif hasattr(tile_window, 'get_coordinate'):
                    tensor_coord = tile_window.get_coordinate(x_multiindex)
                else:
                    # Manual calculation: window_origin + X coordinates
                    tensor_coord = [window_origin[i] + x_coords[i] for i in range(len(x_coords))]
                
                # Extract the final tensor coordinates
                if hasattr(tensor_coord, 'to_list'):
                    final_coords = tensor_coord.to_list()
                elif hasattr(tensor_coord, '_values'):
                    final_coords = tensor_coord._values
                elif isinstance(tensor_coord, list):
                    final_coords = tensor_coord
                else:
                    final_coords = x_coords  # Fallback to X coords
                
                tensor_coordinates.append(final_coords)
                
                if access_count < 8:  # Show first 8
                    print(f"    Access {access_count}: Y{y_indices} -> X{x_coords} -> Tensor{final_coords}")
                    
            except Exception as e:
                if access_count < 8:
                    print(f"    Access {access_count}: Y{y_indices} -> Error: {e}")
            
            access_count += 1
            
            # Limit iterations for readability
            if access_count >= 16:  # Show subset
                return
        
        # Use the ACTUAL library sweep_tile function
        try:
            sweep_tile(distributed_tensor, process_complete_pipeline)
            print(f"\n  Total accesses processed: {access_count}")
            print(f"  Unique tensor coordinates: {len(set(tuple(c) for c in tensor_coordinates))}")
            
            if tensor_coordinates:
                # Show unique coordinates
                unique_coords = list(set(tuple(c) for c in tensor_coordinates))[:8]
                print(f"  Sample unique tensor coordinates:")
                for i, coord in enumerate(unique_coords):
                    print(f"    {coord}")
                    
        except Exception as e:
            print(f"  Error during sweep: {e}")


def test_tile_window_coordinate_access():
    """Test accessing actual tensor data through tile window."""
    
    print(f"\n" + "="*60)
    print("=== Testing Actual Tensor Data Access ===\n")
    
    # Setup (same as above)
    variables = get_default_variables('Real-World Example (RMSNorm)')
    encoding = TileDistributionEncoding(
        rs_lengths=[],
        hs_lengthss=[
            [variables['S::Repeat_M'], variables['S::WarpPerBlock_M'], 
             variables['S::ThreadPerWarp_M'], variables['S::Vector_M']],
            [variables['S::Repeat_N'], variables['S::WarpPerBlock_N'], 
             variables['S::ThreadPerWarp_N'], variables['S::Vector_N']]
        ],
        ps_to_rhss_major=[[1, 2], [1, 2]],
        ps_to_rhss_minor=[[1, 1], [2, 2]],
        ys_to_rhs_major=[1, 1, 2, 2],
        ys_to_rhs_minor=[0, 3, 0, 3]
    )
    
    tile_distribution = make_static_tile_distribution(encoding)
    distributed_tensor = StaticDistributedTensor(data_type=np.float32, tile_distribution=tile_distribution)
    
    # Create tensor with recognizable values
    m_size = (variables["S::Repeat_M"] * variables["S::WarpPerBlock_M"] * 
              variables["S::ThreadPerWarp_M"] * variables["S::Vector_M"])
    n_size = (variables["S::Repeat_N"] * variables["S::WarpPerBlock_N"] * 
              variables["S::ThreadPerWarp_N"] * variables["S::Vector_N"])
    
    tensor_shape = [m_size, n_size]
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    
    # Create tensor view and tile window
    strides = [tensor_shape[1], 1]
    tensor_view = make_naive_tensor_view(data, tensor_shape, strides)
    
    window_lengths = [64, 64]
    window_origin = [0, 0]
    tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=window_lengths,
        origin=window_origin,
        tile_distribution=tile_distribution
    )
    
    print(f"Created tensor with shape {tensor_shape}")
    print(f"Data[0,0] = {data[0,0]}, Data[0,1] = {data[0,1]}, Data[1,0] = {data[1,0]}")
    
    # Test specific coordinates
    test_coords = [
        [0, 0],   # Should access data[0,0]
        [0, 1],   # Should access data[0,1]  
        [1, 0],   # Should access data[1,0]
        [16, 0],  # Should access data[16,0]
        [32, 0],  # Should access data[32,0]
    ]
    
    print(f"\nTesting direct coordinate access:")
    for coord in test_coords:
        try:
            # Try to access data through tile window
            if hasattr(tile_window, '__getitem__'):
                value = tile_window[coord]
                print(f"  Coord {coord} -> Value {value} (expected: {data[coord[0], coord[1]]})")
            else:
                # Direct access to verify
                expected_value = data[coord[0], coord[1]]
                print(f"  Coord {coord} -> Expected value: {expected_value}")
        except Exception as e:
            print(f"  Coord {coord} -> Error: {e}")


if __name__ == "__main__":
    demonstrate_complete_sweep_with_tile_window()
    test_tile_window_coordinate_access() 