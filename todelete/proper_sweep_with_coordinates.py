#!/usr/bin/env python3

"""
Proper example showing how to use sweep_tile with the actual library 
to get real tensor coordinates (not simulation).
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


def demonstrate_proper_sweep_with_coordinates():
    """Show how to properly use sweep_tile to get actual tensor coordinates."""
    
    print("=== Proper sweep_tile with Real Tensor Coordinates ===\n")
    
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
    
    print(f"\nTensor shape: {tensor_shape}")
    
    # Create tensor view
    strides = [tensor_shape[1], 1]
    tensor_view = make_naive_tensor_view(data, tensor_shape, strides)
    
    # Create tile window
    window_lengths = [64, 64]
    window_origin = [0, 0]
    
    tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=window_lengths,
        origin=window_origin,
        tile_distribution=distributed_tensor.tile_distribution
    )
    
    print(f"Window: lengths={window_lengths}, origin={window_origin}")
    
    # Test with different threads
    threads_to_test = [
        (0, 0),  # Thread 0
        (0, 1),  # Thread 1  
        (1, 0),  # Thread 64 (different warp)
    ]
    
    for warp_id, lane_id in threads_to_test:
        print(f"\n=== Thread (warp={warp_id}, lane={lane_id}) ===")
        
        # Set thread position
        set_global_thread_position(warp_id, lane_id)
        
        access_count = 0
        coordinates = []
        
        def process_distributed_index(distributed_idx):
            """Process each distributed index and get actual tensor coordinates."""
            nonlocal access_count, coordinates
            
            # Get Y-space indices from sweep_tile
            y_indices = distributed_idx.partial_indices if hasattr(distributed_idx, 'partial_indices') else []
            
            # Get current partition (P-space) from thread position
            try:
                partition_idx = distributed_tensor.tile_distribution.get_partition_index()
                p_indices = partition_idx if isinstance(partition_idx, list) else []
                
                # Attempt to get actual tensor coordinates
                # The tile distribution should map P+Y to X coordinates
                try:
                    # Try to get the actual X coordinates using the tile distribution
                    if hasattr(distributed_tensor.tile_distribution, 'calculate_index'):
                        x_coords = distributed_tensor.tile_distribution.calculate_index(
                            p_indices + y_indices
                        )
                    elif hasattr(distributed_tensor.tile_distribution, 'map_py_to_x'):
                        x_coords = distributed_tensor.tile_distribution.map_py_to_x(
                            p_indices, y_indices
                        )
                    else:
                        # Alternative: access through tile window
                        if hasattr(tile_window, 'get_coordinate'):
                            x_coords = tile_window.get_coordinate(distributed_idx)
                        else:
                            x_coords = None
                    
                    if x_coords is not None:
                        coordinates.append(x_coords)
                        
                    if access_count < 10:  # Show first 10
                        print(f"  Access {access_count}:")
                        print(f"    Y: {y_indices}")
                        print(f"    P: {p_indices}")
                        if x_coords is not None:
                            print(f"    X: {x_coords}")
                        else:
                            print(f"    X: (unable to calculate)")
                    
                except Exception as coord_e:
                    if access_count < 10:
                        print(f"  Access {access_count}:")
                        print(f"    Y: {y_indices}")
                        print(f"    P: {p_indices}")
                        print(f"    X: Error calculating coordinates: {coord_e}")
                
            except Exception as p_e:
                if access_count < 10:
                    print(f"  Access {access_count}:")
                    print(f"    Y: {y_indices}")
                    print(f"    P: Error getting partition: {p_e}")
            
            access_count += 1
            
            # Limit to avoid too much output
            if access_count >= 256:  # Full sweep
                return
        
        # Use the ACTUAL library sweep_tile function
        try:
            sweep_tile(distributed_tensor, process_distributed_index)
            print(f"\n  Total accesses: {access_count}")
            print(f"  Unique coordinates found: {len(coordinates)}")
            
            if coordinates:
                # Show sample coordinates
                print(f"  Sample coordinates:")
                for i, coord in enumerate(coordinates[:5]):
                    print(f"    {i}: {coord}")
                if len(coordinates) > 5:
                    print(f"    ... and {len(coordinates) - 5} more")
                    
        except Exception as e:
            print(f"  Error during sweep: {e}")


def try_alternative_coordinate_access():
    """Try alternative ways to get coordinates from the library."""
    
    print("\n=== Alternative Coordinate Access Methods ===\n")
    
    # Get configuration
    variables = get_default_variables('Real-World Example (RMSNorm)')
    
    # Create encoding and distribution
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
    
    print("Available methods on tile_distribution:")
    for attr in dir(tile_distribution):
        if not attr.startswith('_'):
            print(f"  {attr}")
    
    # Test direct coordinate calculation if available
    print(f"\nTesting direct coordinate methods:")
    
    # Set thread position
    set_global_thread_position(warp_id=0, lane_id=0)
    
    try:
        # Try to get partition
        if hasattr(tile_distribution, 'get_partition_index'):
            partition = tile_distribution.get_partition_index()
            print(f"  Partition index: {partition}")
    except Exception as e:
        print(f"  get_partition_index error: {e}")
    
    # Try other coordinate methods
    test_y_coords = [0, 0, 0, 0]  # Sample Y coordinates
    
    if hasattr(tile_distribution, 'calculate_x_from_py'):
        try:
            result = tile_distribution.calculate_x_from_py([0, 0], test_y_coords)
            print(f"  calculate_x_from_py([0,0], {test_y_coords}): {result}")
        except Exception as e:
            print(f"  calculate_x_from_py error: {e}")
    
    if hasattr(tile_distribution, 'get_x_coord'):
        try:
            result = tile_distribution.get_x_coord(test_y_coords)
            print(f"  get_x_coord({test_y_coords}): {result}")
        except Exception as e:
            print(f"  get_x_coord error: {e}")


if __name__ == "__main__":
    demonstrate_proper_sweep_with_coordinates()
    try_alternative_coordinate_access() 