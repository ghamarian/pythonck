#!/usr/bin/env python3

"""
Test to find which P values cause X[0] to change in the RMSNorm configuration.
Based on the real library sweep_tile functionality.
"""

import numpy as np
from examples import get_default_variables
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.sweep_tile import sweep_tile
from pytensor.partition_simulation import set_global_thread_position
from pytensor.tensor_coordinate import MultiIndex, make_tensor_adaptor_coordinate


def test_p_values_effect_on_x0():
    """Test different P values to see which ones affect X[0]."""
    
    print("=== Testing P values effect on X[0] (RMSNorm config) ===\n")
    
    # Get RMSNorm configuration
    variables = get_default_variables('Real-World Example (RMSNorm)')
    print("Configuration:")
    print(f"  WarpPerBlock: {variables['S::WarpPerBlock_M']}×{variables['S::WarpPerBlock_N']} = {variables['S::WarpPerBlock_M'] * variables['S::WarpPerBlock_N']} warps")
    print(f"  ThreadPerWarp: {variables['S::ThreadPerWarp_M']}×{variables['S::ThreadPerWarp_N']} = {variables['S::ThreadPerWarp_M'] * variables['S::ThreadPerWarp_N']} threads/warp")
    
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
    print(f"  ndim_p: {tile_distribution.ndim_p}")
    print(f"  ndim_y: {tile_distribution.ndim_y}")
    print(f"  ndim_x: {tile_distribution.ndim_x}")
    
    # Test different P values systematically
    # P dimensions: [warp_id, lane_id] for ndim_p=2
    
    # Calculate max P values
    max_warp_id = variables['S::WarpPerBlock_M'] * variables['S::WarpPerBlock_N'] - 1  # 2*2-1 = 3
    max_lane_id = variables['S::ThreadPerWarp_M'] * variables['S::ThreadPerWarp_N'] - 1  # 8*8-1 = 63
    
    print(f"\nP value ranges:")
    print(f"  P[0] (warp_id): 0 to {max_warp_id}")
    print(f"  P[1] (lane_id): 0 to {max_lane_id}")
    
    # Test specific P combinations to find X[0] changes
    test_cases = [
        (0, 0),   # warp=0, lane=0
        (0, 1),   # warp=0, lane=1  
        (0, 32),  # warp=0, lane=32 (middle of warp)
        (1, 0),   # warp=1, lane=0
        (2, 0),   # warp=2, lane=0
        (3, 0),   # warp=3, lane=0
        (0, 63),  # warp=0, last lane
        (1, 32),  # warp=1, middle lane
        (2, 16),  # warp=2, different lane
        (3, 48),  # warp=3, different lane
    ]
    
    results = []
    
    print(f"\n=== Testing P combinations for X[0] changes ===")
    
    for warp_id, lane_id in test_cases:
        if warp_id > max_warp_id or lane_id > max_lane_id:
            continue
            
        # Set thread position
        set_global_thread_position(warp_id, lane_id)
        
        # Get partition index
        partition_idx = tile_distribution.get_partition_index()
        
        # Test first Y coordinate to see X mapping
        y_indices = [0, 0, 0, 0]  # First Y coordinate
        py_indices = partition_idx + y_indices
        
        try:
            py_multiindex = MultiIndex(len(py_indices), py_indices)
            adaptor_coord = make_tensor_adaptor_coordinate(
                tile_distribution.ps_ys_to_xs_adaptor,
                py_multiindex
            )
            x_coords = adaptor_coord.get_bottom_index().to_list()
            
            results.append({
                'warp_id': warp_id,
                'lane_id': lane_id,
                'P': partition_idx,
                'X': x_coords
            })
            
            print(f"  P[{partition_idx[0]}, {partition_idx[1]}] → X[{x_coords[0]}, {x_coords[1]}]  (warp={warp_id}, lane={lane_id})")
            
        except Exception as e:
            print(f"  Error for warp={warp_id}, lane={lane_id}: {e}")
    
    # Analyze results
    print(f"\n=== Analysis ===")
    
    # Group by X[0] values
    x0_groups = {}
    for result in results:
        x0 = result['X'][0]
        if x0 not in x0_groups:
            x0_groups[x0] = []
        x0_groups[x0].append(result)
    
    print(f"X[0] value groups:")
    for x0_val in sorted(x0_groups.keys()):
        group = x0_groups[x0_val]
        print(f"  X[0] = {x0_val}:")
        for result in group:
            print(f"    P[{result['P'][0]}, {result['P'][1]}] → X[{result['X'][0]}, {result['X'][1]}]")
    
    # Check for X[0] changes
    unique_x0_values = sorted(set(result['X'][0] for result in results))
    print(f"\nUnique X[0] values found: {unique_x0_values}")
    
    if len(unique_x0_values) > 1:
        print("✓ SUCCESS: P changes DO affect X[0]!")
        
        # Find which P values cause X[0] changes
        baseline_x0 = results[0]['X'][0]
        print(f"\nBaseline (P[0,0]): X[0] = {baseline_x0}")
        
        for result in results[1:]:
            if result['X'][0] != baseline_x0:
                print(f"X[0] change: P[{result['P'][0]}, {result['P'][1]}] → X[0] = {result['X'][0]} (Δ = {result['X'][0] - baseline_x0})")
    else:
        print("✗ ISSUE: All P values produce same X[0] value")
        print("This suggests we need to test different P value ranges")


def test_manual_p_construction():
    """Test by manually constructing P values beyond hardware limits."""
    
    print(f"\n" + "="*60)
    print("=== Manual P Construction Test ===\n")
    
    # Get configuration and tile distribution
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
    
    # Test P values beyond hardware limits to explore the full mapping
    test_p_values = [
        [0, 0],   # Baseline
        [0, 1],   # Lane change
        [1, 0],   # Warp change  
        [0, 8],   # Larger lane change
        [0, 16],  # Even larger lane change
        [2, 0],   # Different warp
        [0, 32],  # Large lane change
        [4, 0],   # Beyond hardware warp limit (theoretical)
        [8, 0],   # Much beyond warp limit
        [0, 64],  # Beyond hardware lane limit (theoretical)
        [16, 0],  # Way beyond warp limit
    ]
    
    print("Testing manual P values (beyond hardware limits):")
    print("Format: P[p0, p1] → X[x0, x1]")
    
    y_indices = [0, 0, 0, 0]  # Test with first Y coordinate
    
    for p_val in test_p_values:
        try:
            # Manually construct P+Y coordinates
            py_indices = p_val + y_indices
            
            py_multiindex = MultiIndex(len(py_indices), py_indices)
            adaptor_coord = make_tensor_adaptor_coordinate(
                tile_distribution.ps_ys_to_xs_adaptor,
                py_multiindex
            )
            x_coords = adaptor_coord.get_bottom_index().to_list()
            
            print(f"  P[{p_val[0]:2d}, {p_val[1]:2d}] → X[{x_coords[0]:3d}, {x_coords[1]:3d}]")
            
        except Exception as e:
            print(f"  P[{p_val[0]:2d}, {p_val[1]:2d}] → Error: {e}")
    
    print(f"\nThis shows the THEORETICAL P→X mapping beyond hardware limits")
    print(f"In real GPU execution, P values are constrained by warp/thread counts")


if __name__ == "__main__":
    test_p_values_effect_on_x0()
    test_manual_p_construction() 