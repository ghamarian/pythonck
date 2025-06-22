#!/usr/bin/env python3

"""
PROPER usage of sweep_tile with the ACTUAL library coordinate calculation.

This shows how to use sweep_tile to iterate through Y-space and 
calculate actual tensor coordinates using the library's built-in methods.
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
from pytensor.tensor_coordinate import MultiIndex, TensorAdaptorCoordinate, make_tensor_adaptor_coordinate


def demonstrate_proper_library_usage():
    """Show how to properly use sweep_tile with real coordinate calculation."""
    
    print("=== PROPER Library Usage: sweep_tile + coordinate calculation ===\n")
    
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
    
    # Test with different threads
    threads_to_test = [
        (0, 0),  # Thread 0
        (0, 1),  # Thread 1
        (1, 0),  # Different warp
    ]
    
    for warp_id, lane_id in threads_to_test:
        print(f"\n=== Thread (warp={warp_id}, lane={lane_id}) ===")
        
        # Set thread position
        set_global_thread_position(warp_id, lane_id)
        
        # Get current partition index
        partition_idx = tile_distribution.get_partition_index()
        print(f"  Partition (P): {partition_idx}")
        
        access_count = 0
        coordinates = []
        
        def process_sweep_coordinate(distributed_idx):
            """Process each Y coordinate from sweep_tile and calculate X coordinates."""
            nonlocal access_count, coordinates
            
            # Get Y-space indices from sweep_tile
            y_indices = distributed_idx.partial_indices if hasattr(distributed_idx, 'partial_indices') else []
            
            try:
                # Method 1: Use the adaptor directly (this is the PROPER way)
                # Combine P + Y indices 
                py_indices = partition_idx + y_indices
                
                # Create adaptor coordinate
                py_multiindex = MultiIndex(len(py_indices), py_indices)
                adaptor_coord = make_tensor_adaptor_coordinate(
                    tile_distribution.ps_ys_to_xs_adaptor,
                    py_multiindex
                )
                
                # Get X coordinates
                x_coords = adaptor_coord.get_bottom_index()
                coordinates.append(x_coords.to_list())
                
                if access_count < 8:  # Show first 8
                    print(f"    Access {access_count}: Y{y_indices} → X{x_coords.to_list()}")
                    
            except Exception as e:
                if access_count < 8:
                    print(f"    Access {access_count}: Y{y_indices} → Error: {e}")
            
            access_count += 1
            
            # Limit iterations for readability
            if access_count >= 64:  # Show subset of full sweep
                return
        
        # Use the ACTUAL library sweep_tile function
        try:
            sweep_tile(distributed_tensor, process_sweep_coordinate)
            print(f"\n  Total accesses processed: {access_count}")
            print(f"  Unique X coordinates: {len(set(tuple(c) for c in coordinates))}")
            
            if coordinates:
                # Show unique coordinates
                unique_coords = list(set(tuple(c) for c in coordinates))[:10]
                print(f"  Sample unique coordinates:")
                for i, coord in enumerate(unique_coords):
                    print(f"    {coord}")
                if len(unique_coords) == 10:
                    print(f"    ... (showing first 10)")
                    
        except Exception as e:
            print(f"  Error during sweep: {e}")


def compare_with_manual_simulation():
    """Compare the library sweep_tile with our manual simulation."""
    
    print("\n" + "="*60)
    print("=== Comparison: Library vs Manual Simulation ===\n")
    
    # Get configuration
    variables = get_default_variables('Real-World Example (RMSNorm)')
    
    # Create tile distribution
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
    distributed_tensor = StaticDistributedTensor(
        data_type=np.float32,
        tile_distribution=tile_distribution
    )
    
    # Test Thread 0
    set_global_thread_position(warp_id=0, lane_id=0)
    partition_idx = tile_distribution.get_partition_index()
    
    print("Library approach (sweep_tile):")
    print(f"  Partition: {partition_idx}")
    
    library_coords = []
    access_count = 0
    
    def collect_library_coords(distributed_idx):
        nonlocal access_count, library_coords
        
        y_indices = distributed_idx.partial_indices if hasattr(distributed_idx, 'partial_indices') else []
        
        try:
            py_indices = partition_idx + y_indices
            py_multiindex = MultiIndex(len(py_indices), py_indices)
            adaptor_coord = make_tensor_adaptor_coordinate(
                tile_distribution.ps_ys_to_xs_adaptor,
                py_multiindex
            )
            x_coords = adaptor_coord.get_bottom_index()
            library_coords.append((tuple(y_indices), tuple(x_coords.to_list())))
            
        except Exception as e:
            pass
        
        access_count += 1
        if access_count >= 16:  # Limit for comparison
            return
    
    sweep_tile(distributed_tensor, collect_library_coords)
    
    print(f"  First 8 library results (Y → X):")
    for i, (y, x) in enumerate(library_coords[:8]):
        print(f"    Y{list(y)} → X{list(x)}")
    
    # Manual simulation approach (like thread_coordinate_mapper.py was doing)
    print(f"\nManual simulation approach:")
    print(f"  Would manually iterate Y=[0,0,0,0] to [3,3,3,3]")
    print(f"  Would manually calculate coordinates using custom logic")
    print(f"  → This is what we were doing WRONG in thread_coordinate_mapper.py")
    
    print("\n=== KEY INSIGHT ===")
    print("The LIBRARY way:")
    print("  1. Use sweep_tile() to iterate through Y-space")
    print("  2. Use tile_distribution.ps_ys_to_xs_adaptor for coordinate mapping")
    print("  3. Use make_tensor_adaptor_coordinate() to get X coordinates")
    print("  4. This is the ACTUAL library functionality!")
    
    print("\nThe WRONG way (manual simulation):")
    print("  1. Manually iterate Y coordinates") 
    print("  2. Manually implement coordinate mapping logic")
    print("  3. Duplicate library functionality instead of using it")
    print("  4. → This is what thread_coordinate_mapper.py was doing")


if __name__ == "__main__":
    demonstrate_proper_library_usage()
    compare_with_manual_simulation() 