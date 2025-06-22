#!/usr/bin/env python3

"""
Example showing how to use the actual sweep_tile library functionality.

This demonstrates the REAL library usage, not a simulation.
"""

import numpy as np
from examples import get_default_variables
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.sweep_tile import sweep_tile, make_tile_sweeper
from pytensor.tensor_view import TensorView, make_naive_tensor_view
from pytensor.tile_window import make_tile_window
from pytensor.partition_simulation import set_global_thread_position


def create_rmsnorm_distributed_tensor():
    """Create a distributed tensor using RMSNorm configuration."""
    
    # Get RMSNorm configuration
    variables = get_default_variables('Real-World Example (RMSNorm)')
    
    print("=== RMSNorm Configuration ===")
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
    
    print(f"\n=== Tile Distribution Properties ===")
    print(f"  ndim_x: {tile_distribution.ndim_x}")
    print(f"  ndim_y: {tile_distribution.ndim_y}")
    print(f"  ndim_p: {tile_distribution.ndim_p}")
    
    # Create distributed tensor
    distributed_tensor = StaticDistributedTensor(
        data_type=np.float32,
        tile_distribution=tile_distribution
    )
    
    return distributed_tensor, variables


def demonstrate_actual_sweep_tile():
    """Demonstrate the actual sweep_tile library functionality."""
    
    print("=== Demonstrating Actual sweep_tile Library ===\n")
    
    # Create distributed tensor
    distributed_tensor, variables = create_rmsnorm_distributed_tensor()
    
    # Method 1: Use sweep_tile directly
    print("=== Method 1: Direct sweep_tile usage ===")
    
    access_count = 0
    coordinate_map = {}
    
    def process_coordinate(distributed_idx):
        """Process each coordinate during sweep."""
        nonlocal access_count
        
        # Get the distributed index information
        indices = distributed_idx.partial_indices if hasattr(distributed_idx, 'partial_indices') else []
        
        print(f"  Access {access_count}: DistributedIndex{indices}")
        coordinate_map[access_count] = indices
        access_count += 1
        
        # Limit output for readability
        if access_count >= 10:
            return
    
    # Set a specific thread position for this sweep
    set_global_thread_position(warp_id=0, lane_id=0)
    
    try:
        # This is the ACTUAL library call!
        sweep_tile(distributed_tensor, process_coordinate)
        print(f"    Total accesses: {access_count}")
    except Exception as e:
        print(f"    Error with sweep_tile: {e}")
    
    print()
    
    # Method 2: Use TileSweeper
    print("=== Method 2: TileSweeper usage ===")
    
    access_count = 0
    
    def process_with_sweeper(distributed_idx):
        """Process coordinate with sweeper."""
        nonlocal access_count
        indices = distributed_idx.partial_indices if hasattr(distributed_idx, 'partial_indices') else []
        print(f"  Sweeper Access {access_count}: DistributedIndex{indices}")
        access_count += 1
        
        if access_count >= 10:
            return
    
    try:
        # Create sweeper
        sweeper = make_tile_sweeper(distributed_tensor, process_with_sweeper)
        
        # Get number of accesses
        num_accesses = sweeper.get_num_of_access()
        print(f"    Sweeper reports {num_accesses} total accesses")
        
        # Execute the sweep
        sweeper()
        print(f"    Completed {access_count} accesses")
        
    except Exception as e:
        print(f"    Error with TileSweeper: {e}")
    
    print()
    
    # Method 3: Using sweep_tile with different threads
    print("=== Method 3: sweep_tile with different thread positions ===")
    
    threads_to_test = [
        (0, 0),  # warp=0, lane=0
        (0, 1),  # warp=0, lane=1
        (1, 0),  # warp=1, lane=0
    ]
    
    for warp_id, lane_id in threads_to_test:
        print(f"  Thread (warp={warp_id}, lane={lane_id}):")
        
        # Set thread position
        set_global_thread_position(warp_id, lane_id)
        
        access_count = 0
        
        def thread_process(distributed_idx):
            nonlocal access_count
            indices = distributed_idx.partial_indices if hasattr(distributed_idx, 'partial_indices') else []
            if access_count < 5:  # Show first 5 only
                print(f"    Access {access_count}: DistributedIndex{indices}")
            access_count += 1
        
        try:
            sweep_tile(distributed_tensor, thread_process)
            print(f"    Total accesses for this thread: {access_count}")
        except Exception as e:
            print(f"    Error: {e}")
        
        print()


def demonstrate_sweep_with_tensor_window():
    """Show how sweep_tile relates to tile windows."""
    
    print("=== Demonstrating sweep_tile with Tile Window Context ===\n")
    
    # Create the full setup including tensor view and tile window
    distributed_tensor, variables = create_rmsnorm_distributed_tensor()
    
    # Create actual tensor data
    m_size = (variables["S::Repeat_M"] * variables["S::WarpPerBlock_M"] * 
              variables["S::ThreadPerWarp_M"] * variables["S::Vector_M"])
    n_size = (variables["S::Repeat_N"] * variables["S::WarpPerBlock_N"] * 
              variables["S::ThreadPerWarp_N"] * variables["S::Vector_N"])
    
    tensor_shape = [m_size, n_size]
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    
    print(f"Tensor shape: {tensor_shape}")
    
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
    print()
    
    # Now use sweep_tile to process the distributed tensor
    # This shows the RELATIONSHIP between sweep_tile and tensor coordinates
    
    access_count = 0
    
    def process_with_context(distributed_idx):
        """Process distributed index and show tensor coordinate relationship."""
        nonlocal access_count
        
        # Get distributed indices
        indices = distributed_idx.partial_indices if hasattr(distributed_idx, 'partial_indices') else []
        
        # The key insight: distributed_idx relates to the Y-space of the tile distribution
        # The actual tensor coordinates come from the tile distribution's coordinate mapping
        
        print(f"  Access {access_count}:")
        print(f"    DistributedIndex: {indices}")
        
        # Show how this relates to the distributed tensor's coordinate system
        # Note: The distributed tensor coordinates are in the Y-space
        # To get actual tensor (X-space) coordinates, we need the tile distribution
        
        if hasattr(distributed_tensor.tile_distribution, 'calculate_index'):
            try:
                # This would require knowing the partition index (P dimensions)
                # which comes from the current thread position
                partition_idx = distributed_tensor.tile_distribution.get_partition_index()
                print(f"    Current partition (P): {partition_idx}")
                
                # The actual mapping from Y-space to X-space happens inside the tile distribution
                # sweep_tile iterates through the Y-space (distributed indices)
                # while the tile distribution maps P+Y to X coordinates
                
            except Exception as e:
                print(f"    Partition calculation error: {e}")
        
        access_count += 1
        
        # Limit output
        if access_count >= 8:
            return
    
    # Set thread position and sweep
    set_global_thread_position(warp_id=0, lane_id=0)
    
    print("=== sweep_tile output (this is what the library actually does) ===")
    try:
        sweep_tile(distributed_tensor, process_with_context)
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"\nTotal accesses: {access_count}")
    print("\n=== Key Insight ===")
    print("sweep_tile iterates through the Y-space (distributed indices)")
    print("The tile distribution maps P+Y coordinates to actual tensor X coordinates")
    print("P comes from thread position, Y comes from sweep_tile iteration")


if __name__ == "__main__":
    demonstrate_actual_sweep_tile()
    print("\n" + "="*50 + "\n")
    demonstrate_sweep_with_tensor_window() 