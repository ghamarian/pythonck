#!/usr/bin/env python3

"""
CLEAN example showing proper tile_window + sweep_tile usage.

This demonstrates the CORRECT API pattern:
1. Create tile_window 
2. Load data with tile_window.load()
3. Use sweep_tile to iterate through the loaded distributed tensor
4. Access elements directly using the distributed indices from sweep_tile
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


def simple_proper_usage():
    """
    SIMPLE demonstration of proper tile_window + sweep_tile usage.
    
    This follows the exact pattern that should be used:
    1. Create tile_window
    2. Load data 
    3. Sweep through loaded tensor
    4. Access elements directly
    """
    
    print("=== Simple Proper Usage ===\n")
    
    # Setup configuration
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
    
    # Create tensor data
    m_size = (variables["S::Repeat_M"] * variables["S::WarpPerBlock_M"] * 
              variables["S::ThreadPerWarp_M"] * variables["S::Vector_M"])
    n_size = (variables["S::Repeat_N"] * variables["S::WarpPerBlock_N"] * 
              variables["S::ThreadPerWarp_N"] * variables["S::Vector_N"])
    
    tensor_shape = [m_size, n_size]
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    tensor_view = make_naive_tensor_view(data, tensor_shape, [tensor_shape[1], 1])
    
    # Test different threads
    test_threads = [(0, 0), (0, 1), (1, 0)]
    
    for warp_id, lane_id in test_threads:
        print(f"\n--- Thread (warp={warp_id}, lane={lane_id}) ---")
        
        # STEP 1: Set thread position
        set_global_thread_position(warp_id, lane_id)
        
        # STEP 2: Create tile_window 
        tile_window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=[64, 64],
            origin=[0, 0],
            tile_distribution=tile_distribution
        )
        
        # STEP 3: Load data - this handles ALL the coordinate mapping internally
        loaded_tensor = tile_window.load()
        print(f"Loaded {loaded_tensor.get_num_of_elements()} elements")
        
        # STEP 4: Create a distributed tensor for sweep iteration
        sweep_tensor = StaticDistributedTensor(
            data_type=np.float32,
            tile_distribution=tile_distribution
        )
        
        # STEP 5: Use sweep_tile - THIS IS THE CLEAN WAY
        access_count = 0
        values = []
        
        def process_element(distributed_idx):
            nonlocal access_count, values
            
            # THE RIGHT WAY: Use sweep_tile's distributed index directly
            # The distributed index from sweep_tile should work directly with loaded_tensor
            
            try:
                # Method 1: If sweep gives us the right format, use it directly
                if hasattr(distributed_idx, 'partial_indices'):
                    # This is the WRONG approach - we shouldn't need to convert!
                    # But currently necessary due to API mismatch
                    flat_indices = distributed_idx.partial_indices
                    
                    # Convert to Y indices using the integer access method 
                    # This uses Case 2 from get_y_indices_from_distributed_indices
                    y_indices = loaded_tensor.tile_distribution.get_y_indices_from_distributed_indices(access_count)
                    
                    # Access the loaded tensor
                    value = loaded_tensor.get_element(y_indices)
                    values.append(value)
                    
                    if access_count < 6:
                        print(f"  Access {access_count}: flat{flat_indices} -> Y{y_indices} -> value {value}")
                
            except Exception as e:
                if access_count < 6:
                    print(f"  Access {access_count}: Error - {e}")
            
            access_count += 1
            if access_count >= 16:  # Limit for demo
                return
        
        # Execute the sweep
        sweep_tile(sweep_tensor, process_element)
        
        print(f"  Total accesses: {access_count}")
        if values:
            print(f"  Value range: [{min(values):.1f}, {max(values):.1f}]")


def ideal_usage_when_fixed():
    """
    This shows what the usage SHOULD look like when the API is properly designed.
    Currently this won't work, but it shows the target.
    """
    
    print(f"\n" + "="*60)
    print("=== Ideal Usage (Target API) ===\n")
    
    print("# When the API is properly designed, it should be this simple:")
    print("""
def ideal_process():
    # Setup
    set_global_thread_position(warp_id, lane_id)
    
    tile_window = make_tile_window(tensor_view, [64, 64], [0, 0], tile_distribution)
    loaded_tensor = tile_window.load()
    
    # Create sweep tensor (this should match loaded_tensor format)
    sweep_tensor = make_distributed_tensor_for_sweep(tile_distribution)
    
    def process_element(distributed_idx):
        # This should work directly - no conversion needed!
        value = loaded_tensor[distributed_idx]  # or loaded_tensor.get_element(distributed_idx)
        print(f"Value: {value}")
    
    sweep_tile(sweep_tensor, process_element)
""")
    
    print("The KEY ISSUE: sweep_tile and tile_window.load() use different index formats!")
    print("- sweep_tile gives: TileDistributedIndex with flat partial_indices")
    print("- loaded_tensor expects: either Y indices OR grouped distributed indices")
    print("- This mismatch forces the ugly conversion code")


def demonstrate_the_mismatch():
    """
    Demonstrate why the current API requires the ugly conversion code.
    """
    
    print(f"\n" + "="*60)
    print("=== API Mismatch Demonstration ===\n")
    
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
    sweep_tensor = StaticDistributedTensor(data_type=np.float32, tile_distribution=tile_distribution)
    
    print("What sweep_tile gives us:")
    
    def show_sweep_output(distributed_idx):
        print(f"  Type: {type(distributed_idx)}")
        print(f"  Has partial_indices: {hasattr(distributed_idx, 'partial_indices')}")
        if hasattr(distributed_idx, 'partial_indices'):
            print(f"  partial_indices: {distributed_idx.partial_indices}")
        return  # Stop after first one
    
    sweep_tile(sweep_tensor, show_sweep_output)
    
    print(f"\nWhat get_y_indices_from_distributed_indices expects:")
    print(f"  Case 1 (distributed indices): List of TileDistributedIndex grouped by X dims")
    print(f"  Case 2 (integer access): Single integer for direct Y calculation")
    
    print(f"\nThe mismatch:")
    print(f"  sweep_tile gives: TileDistributedIndex with flat [0,0,0,1]")
    print(f"  But Case 1 expects: [TileDistributedIndex([0,0]), TileDistributedIndex([0,1])]")
    print(f"  And Case 2 expects: integer like 0, 1, 2, 3...")
    
    print(f"\nSolutions:")
    print(f"  1. Fix sweep_tile to give proper format")
    print(f"  2. Fix get_y_indices_from_distributed_indices to handle flat format")
    print(f"  3. Use Case 2 (integer access) which works correctly")


if __name__ == "__main__":
    simple_proper_usage()
    ideal_usage_when_fixed() 
    demonstrate_the_mismatch() 