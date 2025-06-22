#!/usr/bin/env python3

"""
Test the FIXED API that eliminates the conversion code.

This demonstrates that we've successfully fixed the Python implementation
to match the C++ pattern exactly.
"""

import numpy as np
from examples import get_default_variables
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.sweep_tile import sweep_tile, sweep_tensor_direct
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_window import make_tile_window
from pytensor.partition_simulation import set_global_thread_position


def test_fixed_api():
    """Test that the fixed API works correctly."""
    
    print("=== Testing FIXED API ===\n")
    
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
    
    # Create tensor data
    m_size = (variables["S::Repeat_M"] * variables["S::WarpPerBlock_M"] * 
              variables["S::ThreadPerWarp_M"] * variables["S::Vector_M"])
    n_size = (variables["S::Repeat_N"] * variables["S::WarpPerBlock_N"] * 
              variables["S::ThreadPerWarp_N"] * variables["S::Vector_N"])
    
    tensor_shape = [m_size, n_size]
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    tensor_view = make_naive_tensor_view(data, tensor_shape, [tensor_shape[1], 1])
    
    # Test thread (0, 1) 
    set_global_thread_position(0, 1)
    
    # Create tile window and load data
    tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=[64, 64],
        origin=[0, 0],
        tile_distribution=tile_distribution
    )
    
    loaded_tensor = tile_window.load()
    print(f"Loaded tensor with {loaded_tensor.get_num_of_elements()} elements")
    
    # Test Method 1: Using the new sweep_tensor_direct function
    print("\n--- Method 1: sweep_tensor_direct (direct API) ---")
    
    values_direct = []
    access_count_direct = 0
    
    def process_direct(y_indices):
        nonlocal access_count_direct, values_direct
        value = loaded_tensor.get_element(y_indices)
        values_direct.append(value)
        
        if access_count_direct < 5:
            print(f"  Y{y_indices} -> value {value}")
        access_count_direct += 1
    
    sweep_tensor_direct(loaded_tensor, process_direct)
    
    print(f"  Total accesses: {access_count_direct}")
    print(f"  Value range: [{min(values_direct):.1f}, {max(values_direct):.1f}]")
    
    # Test Method 2: Using the enhanced sweep_tile function
    print("\n--- Method 2: Enhanced sweep_tile (automatic detection) ---")
    
    values_enhanced = []
    access_count_enhanced = 0
    
    def process_enhanced(y_indices):
        nonlocal access_count_enhanced, values_enhanced
        value = loaded_tensor.get_element(y_indices)
        values_enhanced.append(value)
        
        if access_count_enhanced < 5:
            print(f"  Y{y_indices} -> value {value}")
        access_count_enhanced += 1
    
    # This should automatically detect that we passed a StaticDistributedTensor instance
    # and use the clean API
    sweep_tile(loaded_tensor, process_enhanced)
    
    print(f"  Total accesses: {access_count_enhanced}")
    print(f"  Value range: [{min(values_enhanced):.1f}, {max(values_enhanced):.1f}]")
    
    # Verify both methods give the same results
    assert access_count_direct == access_count_enhanced
    assert values_direct == values_enhanced
    print(f"\n✅ Both methods give identical results!")
    
    return loaded_tensor, values_direct


def compare_old_vs_new_api():
    """Compare the old ugly API with the new clean API."""
    
    print("\n" + "="*70)
    print("API COMPARISON: Old vs New")
    print("="*70)
    
    print("\n🔴 OLD API (ugly conversions needed):")
    print("```python")
    print("loaded = tile_window.load()")
    print("template = StaticDistributedTensor(...)  # Empty template!")
    print("sweep_tile(template, lambda distributed_idx:")
    print("    # UGLY conversion code:")
    print("    flat_indices = distributed_idx.partial_indices")
    print("    spans = loaded.tile_distribution.get_distributed_spans()")
    print("    distributed_indices_list = []")
    print("    idx_offset = 0")
    print("    for span in spans:")
    print("        span_size = len(span.partial_lengths)")
    print("        x_indices = flat_indices[idx_offset:idx_offset + span_size]")
    print("        distributed_indices_list.append(make_tile_distributed_index(x_indices))")
    print("        idx_offset += span_size")
    print("    y_indices = loaded.tile_distribution.get_y_indices_from_distributed_indices(distributed_indices_list)")
    print("    value = loaded.get_element(y_indices)")
    print(")")
    print("```")
    
    print("\n🟢 NEW API (clean, matches C++):")
    print("```python")
    print("loaded = tile_window.load()")
    print("sweep_tile(loaded, lambda y_indices:  # Use loaded tensor directly!")
    print("    value = loaded.get_element(y_indices)  # Direct access!")
    print(")")
    print("```")
    
    print("\n📊 IMPROVEMENT:")
    print("- Lines of code: 12+ lines → 2 lines")
    print("- Conversion logic: Complex → None")
    print("- API consistency: Different → Matches C++")
    print("- Maintainability: Error-prone → Simple")


def test_edge_cases():
    """Test edge cases to ensure the fix is robust."""
    
    print("\n" + "="*70)
    print("EDGE CASE TESTING")
    print("="*70)
    
    # Test with different threads
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
    
    # Create tensor
    m_size = 256  # Full size
    n_size = 256
    tensor_shape = [m_size, n_size]
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    tensor_view = make_naive_tensor_view(data, tensor_shape, [tensor_shape[1], 1])
    
    test_threads = [(0, 0), (0, 1), (0, 32), (1, 0), (2, 0)]
    
    for warp_id, lane_id in test_threads:
        print(f"\n--- Thread (warp={warp_id}, lane={lane_id}) ---")
        
        set_global_thread_position(warp_id, lane_id)
        
        tile_window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=[64, 64],
            origin=[0, 0],
            tile_distribution=tile_distribution
        )
        
        loaded_tensor = tile_window.load()
        
        # Test that we can sweep and get reasonable values
        values = []
        def collect_values(y_indices):
            value = loaded_tensor.get_element(y_indices)
            values.append(value)
        
        sweep_tensor_direct(loaded_tensor, collect_values)
        
        print(f"  Elements accessed: {len(values)}")
        print(f"  Value range: [{min(values):.1f}, {max(values):.1f}]")
        print(f"  Unique values: {len(set(values))}")
        
        # Ensure we got the expected number of elements
        assert len(values) == 256, f"Expected 256 elements, got {len(values)}"


if __name__ == "__main__":
    # Test the fixed API
    loaded_tensor, values = test_fixed_api()
    
    # Compare old vs new
    compare_old_vs_new_api()
    
    # Test edge cases
    test_edge_cases()
    
    print("\n" + "="*70)
    print("🎉 SUCCESS: FIXED API WORKS PERFECTLY!")
    print("="*70)
    print("The Python implementation now matches the C++ pattern exactly:")
    print("✅ No conversion code needed")
    print("✅ Direct tensor access")
    print("✅ Clean, intuitive API")
    print("✅ Robust across different threads")
    print("\nThe ugly conversion code is now ELIMINATED! 🚀") 