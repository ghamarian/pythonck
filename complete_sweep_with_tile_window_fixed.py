#!/usr/bin/env python3

"""
FIXED example showing sweep_tile + tile_window for actual tensor coordinates.

This shows the CORRECT way to use tile_window:
1. Set thread position FIRST
2. Create tile_window INSIDE thread loop with correct partition context  
3. Use tile_window.load() to get distributed tensor with pre-computed coordinate mappings
4. Use sweep_tile to iterate and access the loaded distributed tensor directly
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
from pytensor.tile_distribution import make_tile_distributed_index


def proper_tile_window_usage():
    """
    Demonstrates the CORRECT way to use tile_window with sweep_tile.
    
    Key points:
    - tile_window.load() handles ALL coordinate transformations internally
    - No manual P+Y->X or X->tensor coordinate calculations needed
    - Each thread gets its own tile_window with correct partition context
    """
    
    print("=== Proper tile_window Usage ===\n")
    
    # Get RMSNorm configuration
    variables = get_default_variables('Real-World Example (RMSNorm)')
    print("Configuration:")
    for key, value in variables.items():
        print(f"  {key} = {value}")
    
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
    
    # Create distributed tensor for sweep_tile iteration
    distributed_tensor = StaticDistributedTensor(
        data_type=np.float32,
        tile_distribution=tile_distribution
    )
    
    # Create tensor data and view
    m_size = (variables["S::Repeat_M"] * variables["S::WarpPerBlock_M"] * 
              variables["S::ThreadPerWarp_M"] * variables["S::Vector_M"])
    n_size = (variables["S::Repeat_N"] * variables["S::WarpPerBlock_N"] * 
              variables["S::ThreadPerWarp_N"] * variables["S::Vector_N"])
    
    tensor_shape = [m_size, n_size]
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    tensor_view = make_naive_tensor_view(data, tensor_shape, [tensor_shape[1], 1])
    
    # Define tile window properties
    window_lengths = [64, 64]
    window_origin = [0, 0]
    
    print(f"\nTensor shape: {tensor_shape}")
    print(f"Window: {window_lengths} at origin {window_origin}")
    
    # Test different threads
    test_threads = [(0, 0), (0, 1), (0, 32), (2, 0)]
    
    for warp_id, lane_id in test_threads:
        print(f"\n--- Thread (warp={warp_id}, lane={lane_id}) ---")
        
        # STEP 1: Set thread position BEFORE creating tile_window
        set_global_thread_position(warp_id, lane_id)
        partition_idx = tile_distribution.get_partition_index()
        print(f"Partition: {partition_idx}")
        
        # STEP 2: Create tile_window with correct thread context
        tile_window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=window_lengths,
            origin=window_origin,
            tile_distribution=tile_distribution
        )
        
        # STEP 3: Load data - this handles ALL coordinate transformations
        loaded_tensor = tile_window.load()
        print(f"Loaded {loaded_tensor.get_num_of_elements()} elements")
        
        # STEP 4: Use sweep_tile to iterate and access loaded data
        access_count = 0
        values = []
        
        def process_access(distributed_idx):
            nonlocal access_count, values
            
            # STEP 1: Convert distributed index to Y indices (matches C++ pattern)
            # This is what C++ static_distributed_tensor::operator[] does:
            # constexpr auto y_idx = get_tile_distribution().get_y_indices_from_distributed_indices(TileDistributedIndices{});
            
            try:
                # FIXED: Convert flat distributed index correctly using spans
                if hasattr(distributed_idx, 'partial_indices'):
                    flat_indices = distributed_idx.partial_indices
                    
                    # Use spans to correctly group distributed indices by X dimensions
                    spans = loaded_tensor.tile_distribution.get_distributed_spans()
                    distributed_indices_list = []
                    idx_offset = 0
                    
                    for span in spans:
                        span_size = len(span.partial_lengths)
                        x_indices = flat_indices[idx_offset:idx_offset + span_size]
                        distributed_indices_list.append(make_tile_distributed_index(x_indices))
                        idx_offset += span_size
                else:
                    distributed_indices_list = distributed_idx
                
                # Get Y indices from distributed indices (now using fixed function)
                y_indices = loaded_tensor.tile_distribution.get_y_indices_from_distributed_indices(distributed_indices_list)
                
                # STEP 2: Access using Y indices (matches C++ implementation)
                value = loaded_tensor.get_element(y_indices)
                values.append(value)
                
                # Show first few accesses
                if access_count < 6:
                    print(f"  distributed_idx {distributed_idx.partial_indices} -> Y{y_indices} -> value {value}")
                
            except Exception as e:
                if access_count < 6:
                    print(f"  distributed_idx -> Error: {e}")
            
            access_count += 1
            if access_count >= 16:  # Limit for demo
                return
        
        # Execute the sweep
        sweep_tile(distributed_tensor, process_access)
        
        print(f"  Processed {access_count} accesses")
        print(f"  Value range: [{min(values):.1f}, {max(values):.1f}]")


def educational_comparison():
    """
    Educational comparison showing the WRONG vs RIGHT approach.
    This is for learning purposes only.
    """
    
    print(f"\n" + "="*60)
    print("=== Educational: Wrong vs Right Approach ===\n")
    
    # Setup
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
    
    m_size = (variables["S::Repeat_M"] * variables["S::WarpPerBlock_M"] * 
              variables["S::ThreadPerWarp_M"] * variables["S::Vector_M"])
    n_size = (variables["S::Repeat_N"] * variables["S::WarpPerBlock_N"] * 
              variables["S::ThreadPerWarp_N"] * variables["S::Vector_N"])
    
    tensor_shape = [m_size, n_size]
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    tensor_view = make_naive_tensor_view(data, tensor_shape, [tensor_shape[1], 1])
    
    window_lengths = [64, 64]
    window_origin = [0, 0]
    
    print("❌ WRONG APPROACH:")
    print("1. Create tile_window ONCE outside thread loop")
    print("2. All threads use SAME tile_window (wrong partition context)")
    print("3. Results in incorrect data access patterns")
    
    # Simulate wrong approach
    set_global_thread_position(0, 0)  # Initial thread
    shared_tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=window_lengths,
        origin=window_origin,
        tile_distribution=tile_distribution
    )
    shared_loaded_tensor = shared_tile_window.load()
    print(f"Shared tile_window created for partition {tile_distribution.get_partition_index()}")
    
    # Test with different thread - but using SAME tile_window
    set_global_thread_position(2, 0)  # Different thread
    current_partition = tile_distribution.get_partition_index()
    print(f"Thread (2,0) has partition {current_partition}, but uses shared tile_window!")
    print("-> This gives WRONG data mapping!")
    
    print(f"\n✅ RIGHT APPROACH:")
    print("1. Set thread position FIRST")
    print("2. Create tile_window INSIDE thread loop")  
    print("3. Each thread gets correct partition context")
    print("4. tile_window.load() handles coordinate transformations automatically")
    
    # Demonstrate right approach
    set_global_thread_position(2, 0)
    correct_tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=window_lengths,
        origin=window_origin,
        tile_distribution=tile_distribution
    )
    correct_loaded_tensor = correct_tile_window.load()
    correct_partition = tile_distribution.get_partition_index()
    print(f"Thread (2,0): tile_window created with correct partition {correct_partition}")
    print("-> This gives CORRECT data mapping!")


def debug_coordinate_transformations():
    """
    Debug the coordinate transformations to understand the actual issue.
    """
    
    print("=== Debugging Coordinate Transformations ===\n")
    
    # Get RMSNorm configuration
    variables = get_default_variables('Real-World Example (RMSNorm)')
    print("Configuration:")
    for key, value in variables.items():
        print(f"  {key} = {value}")
    
    # Create tile distribution with original encoding
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
    
    print(f"\nEncoding structure:")
    print(f"  hs_lengthss: {encoding.hs_lengthss}")
    print(f"  ys_to_rhs_major: {encoding.ys_to_rhs_major}")
    print(f"  ys_to_rhs_minor: {encoding.ys_to_rhs_minor}")
    
    # Debug the detail structure
    detail = encoding.detail
    print(f"\nEncoding detail structure:")
    print(f"  ys_to_span_major: {detail.ys_to_span_major}")
    print(f"  ys_to_span_minor: {detail.ys_to_span_minor}")
    print(f"  rhs_major_minor_to_span_minor: {detail.rhs_major_minor_to_span_minor}")
    
    print(f"\nDistribution properties:")
    print(f"  ndim_x: {tile_distribution.ndim_x}")
    print(f"  ndim_y: {tile_distribution.ndim_y}")
    print(f"  ndim_p: {tile_distribution.ndim_p}")
    
    # Check what the actual H lengths are
    h0_lengths = encoding.hs_lengthss[0]  # [4, 2, 8, 4]
    h1_lengths = encoding.hs_lengthss[1]  # [4, 2, 8, 4]
    
    print(f"\nH dimension lengths:")
    print(f"  H0: {h0_lengths}")
    print(f"  H1: {h1_lengths}")
    
    print(f"\nY mapping analysis (RH vs Span):")
    for i, (rh_major, rh_minor) in enumerate(zip(encoding.ys_to_rhs_major, encoding.ys_to_rhs_minor)):
        span_major = detail.ys_to_span_major[i]
        span_minor = detail.ys_to_span_minor[i]
        
        if rh_major == 1:  # H0
            h_length = h0_lengths[rh_minor] if rh_minor < len(h0_lengths) else "OUT_OF_BOUNDS"
        elif rh_major == 2:  # H1
            h_length = h1_lengths[rh_minor] if rh_minor < len(h1_lengths) else "OUT_OF_BOUNDS"
        else:
            h_length = "INVALID_MAJOR"
        
        print(f"  Y{i}: RH({rh_major},{rh_minor}) -> Span({span_major},{span_minor}) -> length {h_length}")
    
    # Debug spans structure issue
    print(f"\n=== DEBUG: Fixed get_y_indices_from_distributed_indices ===")
    
    # Test the fixed function - use CORRECT approach with spans
    test_flat_indices = [0, 1, 2, 3]  # Example: (0,1,2,3)
    
    # Get spans to split correctly
    spans = distributed_tensor.tile_distribution.get_distributed_spans()
    test_distributed_indices = []
    idx_offset = 0
    
    for span in spans:
        span_size = len(span.partial_lengths)
        x_indices = test_flat_indices[idx_offset:idx_offset + span_size]
        test_distributed_indices.append(make_tile_distributed_index(x_indices))
        idx_offset += span_size
    
    print(f"Test input flat: {test_flat_indices}")
    print(f"Test input distributed: {[idx.partial_indices for idx in test_distributed_indices]}")
    
    # Manual fixed calculation using span mappings
    y_indices_fixed = [0] * tile_distribution.ndim_y
    for y_idx in range(tile_distribution.ndim_y):
        span_major = detail.ys_to_span_major[y_idx]  # Use span_major instead of rh_major-1
        span_minor = detail.ys_to_span_minor[y_idx]  # Use span_minor from detail
        
        print(f"  Y{y_idx}: span_major={span_major}, span_minor={span_minor}")
        
        # Get the distributed index for this span
        if 0 <= span_major < len(test_distributed_indices):
            dstr_index = test_distributed_indices[span_major]
            # Get the specific component's index
            if span_minor < len(dstr_index.partial_indices):
                y_indices_fixed[y_idx] = dstr_index.partial_indices[span_minor]
                print(f"    -> y_indices_fixed[{y_idx}] = {y_indices_fixed[y_idx]}")
            else:
                print(f"    -> span_minor {span_minor} out of bounds for distributed index {dstr_index.partial_indices}")
        else:
            print(f"    -> span_major {span_major} out of bounds for distributed_indices length {len(test_distributed_indices)}")
    
    print(f"Fixed Y indices: {y_indices_fixed}")
    
    # Compare with library function
    library_y_indices = tile_distribution.get_y_indices_from_distributed_indices(test_distributed_indices)
    print(f"Library Y indices: {library_y_indices}")
    
    # Verify they match
    if y_indices_fixed == library_y_indices:
        print("✅ SUCCESS: Manual calculation matches library function!")
    else:
        print("❌ MISMATCH: Manual calculation differs from library function!")


if __name__ == "__main__":
    proper_tile_window_usage()
    educational_comparison()
    debug_coordinate_transformations() 