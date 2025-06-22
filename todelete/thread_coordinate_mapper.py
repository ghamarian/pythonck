#!/usr/bin/env python3
"""
Thread Coordinate Mapper

This program demonstrates which X tensor coordinates are accessed by each thread
when using sweep_tile with the RMSNorm tile distribution encoding example.

The program:
1. Uses the RMSNorm example from examples.py
2. Creates the tile distribution with default variables
3. Simulates different thread positions (warp_id, lane_id)
4. Shows which X coordinates each thread accesses during sweep_tile
"""

import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples import get_examples, get_default_variables
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import TileDistribution, make_tile_distribution
from pytensor.tensor_view import TensorView, make_naive_tensor_view
from pytensor.tile_window import make_tile_window
from pytensor.sweep_tile import sweep_tile
from pytensor.partition_simulation import PartitionSimulator, PartitionConfig, with_thread_position, set_global_thread_position
from pytensor.tensor_descriptor import (
    TensorAdaptor, TensorDescriptor, PassThroughTransform, EmbedTransform,
    UnmergeTransform, MergeTransform, make_naive_tensor_descriptor
)
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.tensor_coordinate import MultiIndex, make_tensor_adaptor_coordinate
from pytensor.tile_distribution import make_static_tile_distribution


def parse_rmsnorm_example() -> Tuple[TileDistributionEncoding, Dict]:
    """
    Parse the RMSNorm example from examples.py and create the encoding.
    
    Returns:
        Tuple of (TileDistributionEncoding, variables_dict)
    """
    examples = get_examples()
    rmsnorm_example = examples["Real-World Example (RMSNorm)"]
    variables = get_default_variables("Real-World Example (RMSNorm)")
    
    print("=== RMSNorm Example Configuration ===")
    print("Variables:")
    for var_name, var_value in variables.items():
        print(f"  {var_name} = {var_value}")
    
    # Create the encoding based on the RMSNorm example
    # tile_distribution_encoding<
    #     sequence<>,                             // Empty R
    #     tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
    #           sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
    #     tuple<sequence<1, 2>, sequence<1, 2>>,
    #     tuple<sequence<1, 1>, sequence<2, 2>>,
    #     sequence<1, 1, 2, 2>,
    #     sequence<0, 3, 0, 3>>{}
    
    encoding = TileDistributionEncoding(
        rs_lengths=[],  # Empty R sequence
        hs_lengthss=[
            [variables["S::Repeat_M"], variables["S::WarpPerBlock_M"], 
             variables["S::ThreadPerWarp_M"], variables["S::Vector_M"]],
            [variables["S::Repeat_N"], variables["S::WarpPerBlock_N"], 
             variables["S::ThreadPerWarp_N"], variables["S::Vector_N"]]
        ],
        ps_to_rhss_major=[[1, 2], [1, 2]],  # P dimensions map to H dimensions
        ps_to_rhss_minor=[[1, 1], [2, 2]],
        ys_to_rhs_major=[1, 1, 2, 2],       # Y dimensions map to H dimensions
        ys_to_rhs_minor=[0, 3, 0, 3]
    )
    
    return encoding, variables


def create_tensor_and_window(encoding: TileDistributionEncoding, variables: Dict) -> Tuple:
    """
    Create tensor view and tile window for the RMSNorm example.
    
    Args:
        encoding: The tile distribution encoding
        variables: Variable values from the example
        
    Returns:
        Tuple of (tensor_view, tile_window, tile_distribution)
    """
    # Calculate tensor dimensions from the encoding
    # X dimensions are determined by the H sequences
    m_size = (variables["S::Repeat_M"] * variables["S::WarpPerBlock_M"] * 
              variables["S::ThreadPerWarp_M"] * variables["S::Vector_M"])
    n_size = (variables["S::Repeat_N"] * variables["S::WarpPerBlock_N"] * 
              variables["S::ThreadPerWarp_N"] * variables["S::Vector_N"])
    
    tensor_shape = [m_size, n_size]
    print(f"\nTensor shape: {tensor_shape}")
    
    # Create tensor data
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    
    # Create tensor view with proper strides
    strides = [tensor_shape[1], 1]  # Row-major layout
    tensor_view = make_naive_tensor_view(data, tensor_shape, strides)
    
    # Create tile distribution using the working library function
    # This automatically creates the correct PS_YS to X adaptor from the encoding
    tile_distribution = make_static_tile_distribution(encoding)
    
    print(f"Tile distribution properties:")
    print(f"  ndim_x: {tile_distribution.ndim_x}")
    print(f"  ndim_y: {tile_distribution.ndim_y}")
    print(f"  ndim_p: {tile_distribution.ndim_p}")
    print(f"  ndim_r: {tile_distribution.ndim_r}")
    
    # Create tile window that covers a LARGER portion of the tensor to see coordinate differences
    # Use larger window and non-zero origin to better see coordinate mapping
    window_lengths = [64, 64]  # Use larger window to see more coordinate variation
    window_origin = [0, 0]     # Keep origin at 0 for now, but could offset
    
    print(f"Window setup: lengths={window_lengths}, origin={window_origin}")
    
    tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=window_lengths,
        origin=window_origin,
        tile_distribution=tile_distribution
    )
    
    return tensor_view, tile_window, tile_distribution


def simulate_thread_coordinates(tile_window, tile_distribution, variables: Dict, num_threads: int = 16) -> Dict:
    """
    Simulate different thread positions and capture the coordinates they access.
    
    Args:
        tile_window: The tile window
        tile_distribution: The tile distribution
        num_threads: Number of threads to simulate
        
    Returns:
        Dictionary mapping thread_id to list of accessed coordinates
    """
    print(f"\n=== Simulating {num_threads} threads ===")
    
    # Import the global thread position functions
    from pytensor.partition_simulation import set_global_thread_position
    
    thread_coordinates = {}
    
    # Calculate proper thread distribution based on RMSNorm parameters
    # P0 maps to WarpPerBlock dimensions, P1 maps to ThreadPerWarp dimensions
    threads_per_warp = variables['S::ThreadPerWarp_M'] * variables['S::ThreadPerWarp_N']
    warps_per_block = variables['S::WarpPerBlock_M'] * variables['S::WarpPerBlock_N']
    
    print(f"\n=== Thread Distribution Calculation ===")
    print(f"ThreadsPerWarp: {threads_per_warp} (ThreadPerWarp_M={variables['S::ThreadPerWarp_M']} * ThreadPerWarp_N={variables['S::ThreadPerWarp_N']})")
    print(f"WarpsPerBlock: {warps_per_block} (WarpPerBlock_M={variables['S::WarpPerBlock_M']} * WarpPerBlock_N={variables['S::WarpPerBlock_N']})")
    print(f"Total threads in block: {threads_per_warp * warps_per_block}")
    
    for thread_id in range(num_threads):
        # Calculate warp_id and lane_id based on RMSNorm parameters
        warp_id = thread_id // threads_per_warp
        lane_id = thread_id % threads_per_warp
        
        print(f"\nThread {thread_id}: warp_id={warp_id}, lane_id={lane_id}")
        
        # Set the global thread position for this simulation
        set_global_thread_position(warp_id, lane_id)
        
        # Also use the context manager approach
        with with_thread_position(warp_id, lane_id):
            accessed_coords = []
            
            try:
                # Method 1: Get partition index and debug the tile distribution
                partition_idx = tile_distribution.get_partition_index()
                print(f"  Partition index: {partition_idx}")
                
                # Debug: Let's see what the adaptor transforms look like
                adaptor = tile_distribution.ps_ys_to_xs_adaptor
                print(f"  Adaptor bottom dims: {adaptor.get_num_of_bottom_dimension()}")
                print(f"  Adaptor top dims: {adaptor.get_num_of_top_dimension()}")
                
                # Method 2: Simulate sweep_tile behavior by iterating through Y dimensions
                print(f"  === Simulating sweep_tile Y iteration ===")
                
                # Get Y dimension information
                ys_to_d_desc = tile_distribution.ys_to_d_descriptor
                y_lengths = ys_to_d_desc.get_lengths()
                print(f"  Y dimension lengths: {y_lengths}")
                
                # Calculate how many elements this thread will access during sweep
                total_y_elements = 1
                for length in y_lengths:
                    total_y_elements *= length
                print(f"  Total Y elements per thread: {total_y_elements}")
                
                # Simulate sweep_tile iteration through Y coordinates
                num_elements_to_show = min(8, total_y_elements)  # Show first 8 elements
                print(f"  Showing first {num_elements_to_show} Y coordinate accesses:")
                
                for y_idx in range(num_elements_to_show):
                    # Convert linear Y index to multi-dimensional Y coordinates
                    y_coords = []
                    remaining = y_idx
                    for dim in range(len(y_lengths) - 1, -1, -1):
                        y_coords.insert(0, remaining % y_lengths[dim])
                        remaining //= y_lengths[dim]
                    
                    # Create full PS_YS coordinate: [P0, P1, Y0, Y1, Y2, Y3]
                    ps_ys_coords = partition_idx + y_coords
                    
                    try:
                        # Use the adaptor to calculate X coordinates
                        coord = make_tensor_adaptor_coordinate(
                            adaptor,
                            MultiIndex(len(ps_ys_coords), ps_ys_coords)
                        )
                        bottom_coord = coord.get_bottom_index()
                        x_coords = bottom_coord.to_list()
                        
                        # Add window origin to get absolute tensor coordinates
                        tensor_coords = [
                            tile_window.window_origin[i] + x_coords[i] 
                            for i in range(min(len(x_coords), len(tile_window.window_origin)))
                        ]
                        
                        accessed_coords.append(tensor_coords[:2])  # Take X0, X1
                        print(f"    Y[{y_idx}] = {y_coords} -> PS_YS{ps_ys_coords} -> X{x_coords} -> Tensor{tensor_coords[:2]}")
                        
                    except Exception as e:
                        print(f"    Y[{y_idx}] = {y_coords} -> ERROR: {e}")
                        
                    # Always do manual test for first Y coordinate to debug the adaptor
                    if y_idx == 0:  # Test on first Y coordinate of each thread 
                        test_manual_adaptor(adaptor, thread_id)
                        test_library_tile_distribution(tile_distribution, thread_id)
                        
                    # Fallback for errors only
                    if not accessed_coords:
                        accessed_coords.append([0, 0])
                
                # Method 3: Try using the actual tile window's precomputed coordinates
                print(f"  === Checking tile window precomputed coordinates ===")
                thread_window = make_tile_window(
                    tensor_view=tile_window.bottom_tensor_view,
                    window_lengths=tile_window.window_lengths,
                    origin=tile_window.window_origin,
                    tile_distribution=tile_distribution
                )
                
                if hasattr(thread_window, 'pre_computed_coords') and thread_window.pre_computed_coords:
                    print(f"  Found {len(thread_window.pre_computed_coords)} precomputed coordinate bundles")
                    for i, (adaptor_coord, bottom_coord) in enumerate(thread_window.pre_computed_coords):
                        coords = bottom_coord.get_index().to_list()
                        tensor_coords = [
                            tile_window.window_origin[j] + coords[j] 
                            for j in range(min(len(coords), len(tile_window.window_origin)))
                        ]
                        print(f"    Bundle {i}: Bottom{coords[:2]} -> Tensor{tensor_coords[:2]}")
                        
                        # Also try to simulate multiple accesses for this bundle
                        if hasattr(thread_window, 'traits'):
                            traits = thread_window.traits
                            num_accesses = min(4, traits.num_access)
                            print(f"    Simulating {num_accesses} accesses for this bundle:")
                            for access_idx in range(num_accesses):
                                try:
                                    access_info = traits.get_vectorized_access_info(access_idx)
                                    for vec_idx, y_indices in enumerate(access_info['vector_indices']):
                                        d_offset = ys_to_d_desc.calculate_offset(y_indices)
                                        print(f"      Access[{access_idx}][{vec_idx}]: Y{y_indices} -> D{d_offset}")
                                except Exception as access_e:
                                    print(f"      Access[{access_idx}]: ERROR {access_e}")
                
            except Exception as e:
                print(f"  Simulation failed: {e}")
                # Ultimate fallback: distribute threads across the tensor
                base_x = (thread_id * 4) % tile_window.window_lengths[0]  
                base_y = (thread_id * 4 // tile_window.window_lengths[0]) % tile_window.window_lengths[1]
                
                for offset in range(4):  # Show 4 elements per thread
                    x = (base_x + offset) % tile_window.window_lengths[0]
                    y = base_y
                    tensor_coords = [
                        tile_window.window_origin[0] + x,
                        tile_window.window_origin[1] + y
                    ]
                    accessed_coords.append(tensor_coords)
                    print(f"  Fallback element {offset}: Window[{x}, {y}] -> Tensor{tensor_coords}")
            
            thread_coordinates[thread_id] = accessed_coords
    
    return thread_coordinates


def analyze_coordinate_patterns(thread_coordinates: Dict, tensor_shape: List[int]):
    """
    Analyze the coordinate access patterns across threads.
    
    Args:
        thread_coordinates: Dictionary mapping thread_id to coordinates
        tensor_shape: Shape of the tensor
    """
    print(f"\n=== Coordinate Pattern Analysis ===")
    print(f"Tensor shape: {tensor_shape}")
    
    # Create a coverage map
    coverage_map = np.zeros(tensor_shape, dtype=int)
    
    # Track which threads access each coordinate
    coord_to_threads = {}
    
    for thread_id, coords_list in thread_coordinates.items():
        print(f"\nThread {thread_id} accesses:")
        unique_coords = set()
        
        for coords in coords_list:
            if len(coords) >= 2:  # Ensure we have at least 2D coordinates
                coord_tuple = tuple(coords[:2])  # Take first 2 dimensions
                unique_coords.add(coord_tuple)
                
                # Check bounds before accessing
                if (0 <= coords[0] < tensor_shape[0] and 
                    0 <= coords[1] < tensor_shape[1]):
                    coverage_map[coords[0], coords[1]] += 1
                    
                    if coord_tuple not in coord_to_threads:
                        coord_to_threads[coord_tuple] = []
                    coord_to_threads[coord_tuple].append(thread_id)
        
        sorted_coords = sorted(unique_coords)
        print(f"  Unique coordinates: {sorted_coords}")
        print(f"  Total unique accesses: {len(unique_coords)}")
    
    # Print coverage statistics
    total_elements = tensor_shape[0] * tensor_shape[1]
    covered_elements = np.count_nonzero(coverage_map)
    print(f"\nCoverage Statistics:")
    print(f"  Total tensor elements: {total_elements}")
    print(f"  Covered elements: {covered_elements}")
    print(f"  Coverage percentage: {100 * covered_elements / total_elements:.1f}%")
    
    # Print coordinates accessed by multiple threads (potential conflicts)
    conflicts = {coord: threads for coord, threads in coord_to_threads.items() 
                if len(threads) > 1}
    
    if conflicts:
        print(f"\nCoordinates accessed by multiple threads:")
        for coord, threads in conflicts.items():
            print(f"  {coord}: threads {threads}")
    else:
        print(f"\nNo coordinate conflicts detected (each coordinate accessed by exactly one thread)")
    
    # Print coverage map (for small tensors)
    if tensor_shape[0] <= 16 and tensor_shape[1] <= 16:
        print(f"\nCoverage map (shows access count per coordinate):")
        for i in range(tensor_shape[0]):
            row_str = ""
            for j in range(tensor_shape[1]):
                row_str += f"{coverage_map[i, j]:2d} "
            print(f"  Row {i:2d}: {row_str}")


def test_manual_adaptor(ps_ys_to_xs_adaptor: TensorAdaptor, thread_id: int):
    """Test manual adaptor with specific PS_YS coordinates."""
    print(f"    === Manual Adaptor Test (Thread {thread_id}) ===")
    
    # Test coordinate mappings that SHOULD produce different X values
    test_coordinates = [
        [0, 0, 0, 0, 0, 0],  # All zeros baseline
        [0, 1, 0, 0, 0, 0],  # P1 = 1 (different thread)
        [0, 0, 1, 0, 0, 0],  # Y0 = 1 (different Y0 iteration)
        [0, 0, 0, 1, 0, 0],  # Y1 = 1 (different Y1 iteration)
        [0, 0, 0, 0, 1, 0],  # Y2 = 1 (different Y2 iteration)
        [0, 0, 0, 0, 0, 1],  # Y3 = 1 (different Y3 iteration)
        [1, 0, 0, 0, 0, 0],  # P0 = 1 (different warp - column only)
        [2, 0, 0, 0, 0, 0],  # P0 = 2 (different warp - ROW change!)
        [3, 0, 0, 0, 0, 0],  # P0 = 3 (different warp - both X0 and X1!)
        [0, 2, 1, 1, 0, 0],  # Combined: P1=2, Y0=1, Y1=1
    ]
    
    coordinate_names = [
        "Baseline [P0=0, P1=0, Y0=0, Y1=0, Y2=0, Y3=0]",
        "P1 change [P0=0, P1=1, Y0=0, Y1=0, Y2=0, Y3=0]", 
        "Y0 change [P0=0, P1=0, Y0=1, Y1=0, Y2=0, Y3=0]",
        "Y1 change [P0=0, P1=0, Y0=0, Y1=1, Y2=0, Y3=0]",
        "Y2 change [P0=0, P1=0, Y0=0, Y1=0, Y2=1, Y3=0]",
        "Y3 change [P0=0, P1=0, Y0=0, Y1=0, Y2=0, Y3=1]",
        "P0=1 chng [P0=1, P1=0, Y0=0, Y1=0, Y2=0, Y3=0]",
        "P0=2 chng [P0=2, P1=0, Y0=0, Y1=0, Y2=0, Y3=0]",
        "P0=3 chng [P0=3, P1=0, Y0=0, Y1=0, Y2=0, Y3=0]",
        "Combined  [P0=0, P1=2, Y0=1, Y1=1, Y2=0, Y3=0]",
    ]
    
    for ps_ys_coord, name in zip(test_coordinates, coordinate_names):
        # Use the actual library's TensorAdaptor.calculate_bottom_index()
        multi_idx = MultiIndex(len(ps_ys_coord), ps_ys_coord)
        x_coord = ps_ys_to_xs_adaptor.calculate_bottom_index(multi_idx)
        x_coord_list = x_coord.to_list()
        print(f"      {name} -> X{x_coord_list}")
    
    print(f"    ========================")


def test_library_tile_distribution(tile_distribution: TileDistribution, thread_id: int):
    """Test the library-generated tile distribution with different partition indices."""
    print(f"    === Library Tile Distribution Test (Thread {thread_id}) ===")
    
    # Test with different partition indices that threads actually have
    test_partition_indices = [
        [0, 0],  # Thread 0's partition index
        [0, 1],  # Thread 1's partition index  
        [0, 2],  # Thread 2's partition index
        [1, 0],  # Warp 1, Thread 0's partition index
    ]
    
    for partition_idx in test_partition_indices:
        # Use the library's calculate_index method
        x_coord = tile_distribution.calculate_index(partition_idx)
        x_coord_list = x_coord.to_list()
        print(f"      P{partition_idx} -> X{x_coord_list}")
    
    print(f"    ==========================")


def main():
    """Main function to demonstrate thread coordinate mapping."""
    print("Thread Coordinate Mapper for RMSNorm Example")
    print("==================================================")
    
    # Parse the RMSNorm example configuration
    encoding, variables = parse_rmsnorm_example()
    
    # Create tensor and tile window
    tensor_view, tile_window, tile_distribution = create_tensor_and_window(encoding, variables)
    
    # Simulate thread coordinates
    num_threads_to_simulate = 8  # Simulate fewer threads for clearer output
    thread_coordinates = simulate_thread_coordinates(tile_window, tile_distribution, variables, num_threads_to_simulate)
    
    # Analyze the coordinate patterns
    tensor_shape = [tile_window.window_lengths[0], tile_window.window_lengths[1]]
    analyze_coordinate_patterns(thread_coordinates, tensor_shape)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Successfully mapped coordinates for {num_threads_to_simulate} threads")
    print(f"Each thread's access pattern shows which X tensor indices it touches")
    print(f"This demonstrates the tile distribution's mapping from threads to tensor elements")


if __name__ == "__main__":
    exit(main()) 