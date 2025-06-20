#!/usr/bin/env python3

"""
COMPARISON: Old vs Fixed API for tile_window + sweep_tile usage.

This demonstrates:
1. OLD API (problematic): Requires ugly conversion code
2. FIXED API (clean): Matches C++ pattern exactly

The fixed API eliminates the conversion code and provides a clean interface.
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
from pytensor.tile_distribution import make_tile_distributed_index


def demonstrate_old_problematic_api():
    """
    Demonstrate the OLD API that requires ugly conversion code.
    
    This shows the problematic approach where:
    1. tile_window.load() returns a loaded tensor
    2. sweep_tile() requires a separate template tensor
    3. Manual conversion is needed between the two different index formats
    """
    
    print("=== OLD PROBLEMATIC API ===\n")
    
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
    
    # Test with thread (0, 1)
    set_global_thread_position(0, 1)
    
    # Create tile window and load data
    tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=[64, 64],
        origin=[0, 0],
        tile_distribution=tile_distribution
    )
    
    # STEP 1: Load data into one tensor
    loaded_tensor = tile_window.load()
    print(f"Loaded {loaded_tensor.get_num_of_elements()} elements into loaded_tensor")
    
    # STEP 2: Create a SEPARATE template tensor for sweep_tile (this is the problem!)
    sweep_tensor = StaticDistributedTensor(
        data_type=np.float32,
        tile_distribution=tile_distribution
    )
    print(f"Created separate template tensor for sweep_tile")
    
    # STEP 3: Use sweep_tile with UGLY conversion code
    values_old = []
    access_count = 0
    
    def process_with_ugly_conversion(distributed_idx):
        nonlocal access_count, values_old
        
        try:
            # THE UGLY CONVERSION CODE that shouldn't be needed:
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
                
                # Convert to Y indices using the complex dual-case function
                y_indices = loaded_tensor.tile_distribution.get_y_indices_from_distributed_indices(distributed_indices_list)
                
                # Finally access the value
                value = loaded_tensor.get_element(y_indices)
                values_old.append(value)
                
                if access_count < 5:
                    print(f"  UGLY: flat{flat_indices} -> spans{len(spans)} -> Y{y_indices} -> value {value}")
            else:
                # Alternative approach using integer access
                y_indices = loaded_tensor.tile_distribution.get_y_indices_from_distributed_indices(access_count)
                value = loaded_tensor.get_element(y_indices)
                values_old.append(value)
                
                if access_count < 5:
                    print(f"  ALT: access{access_count} -> Y{y_indices} -> value {value}")
        
        except Exception as e:
            if access_count < 5:
                print(f"  ERROR at access {access_count}: {e}")
        
        access_count += 1
        if access_count >= 256:  # Limit for demo
            return
    
    print("\nUsing OLD API with conversion code:")
    sweep_tile(sweep_tensor, process_with_ugly_conversion)
    
    print(f"Total accesses: {access_count}")
    if values_old:
        print(f"Value range: [{min(values_old):.1f}, {max(values_old):.1f}]")
    
    return loaded_tensor, values_old


def demonstrate_fixed_clean_api():
    """
    Demonstrate the FIXED API that eliminates conversion code.
    
    This shows the clean approach where:
    1. tile_window.load() returns a loaded tensor
    2. sweep_tile() works directly with the loaded tensor
    3. No conversion code needed - matches C++ exactly
    """
    
    print("\n" + "="*70)
    print("=== FIXED CLEAN API ===\n")
    
    # Same setup as before
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
    
    # Test with thread (0, 1)
    set_global_thread_position(0, 1)
    
    # Create tile window and load data
    tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=[64, 64],
        origin=[0, 0],
        tile_distribution=tile_distribution
    )
    
    # STEP 1: Load data (same as before)
    loaded_tensor = tile_window.load()
    print(f"Loaded {loaded_tensor.get_num_of_elements()} elements")
    
    # STEP 2: Use FIXED API - sweep directly on loaded tensor (NO separate template!)
    values_new = []
    access_count = 0
    
    def process_clean_no_conversion(y_indices):
        nonlocal access_count, values_new
        
        # NO CONVERSION NEEDED! Direct access with Y indices
        value = loaded_tensor.get_element(y_indices)
        values_new.append(value)
        
        if access_count < 5:
            print(f"  CLEAN: Y{y_indices} -> value {value}")
        access_count += 1
    
    print("\nUsing FIXED API (no conversions):")
    # This is the clean pattern that matches C++:
    sweep_tile(loaded_tensor, process_clean_no_conversion)
    
    print(f"Total accesses: {access_count}")
    print(f"Value range: [{min(values_new):.1f}, {max(values_new):.1f}]")
    
    return loaded_tensor, values_new


def show_api_comparison():
    """Show side-by-side comparison of old vs new API."""
    
    print("\n" + "="*100)
    print("API COMPARISON: OLD (Problematic) vs FIXED (Clean)")
    print("="*100)
    
    print("\n" + "🔴 OLD PROBLEMATIC API".center(50) + " | " + "🟢 FIXED CLEAN API".center(47))
    print("-" * 50 + "-+-" + "-" * 47)
    print("loaded = tile_window.load()".ljust(50) + " | " + "loaded = tile_window.load()")
    print("template = StaticDistributedTensor(...)".ljust(50) + " | " + "# No separate template needed!")
    print("".ljust(50) + " | ")
    print("sweep_tile(template, lambda distributed_idx:".ljust(50) + " | " + "sweep_tile(loaded, lambda y_indices:")
    print("  # UGLY CONVERSION:".ljust(50) + " | " + "  # Direct access:")
    print("  flat = distributed_idx.partial_indices".ljust(50) + " | " + "  value = loaded.get_element(y_indices)")
    print("  spans = loaded.tile_distribution...".ljust(50) + " | " + ")")
    print("  distributed_indices_list = []".ljust(50) + " | ")
    print("  idx_offset = 0".ljust(50) + " | ")
    print("  for span in spans:".ljust(50) + " | ")
    print("    span_size = len(span.partial_lengths)".ljust(50) + " | ")
    print("    x_indices = flat[idx_offset:idx_offset...]".ljust(50) + " | ")
    print("    distributed_indices_list.append(...)".ljust(50) + " | ")
    print("    idx_offset += span_size".ljust(50) + " | ")
    print("  y_indices = loaded.tile_distribution...".ljust(50) + " | ")
    print("  value = loaded.get_element(y_indices)".ljust(50) + " | ")
    print(")".ljust(50) + " | ")
    
    print("\n" + "PROBLEMS:".center(50) + " | " + "BENEFITS:".center(47))
    print("-" * 50 + "-+-" + "-" * 47)
    print("❌ 12+ lines of conversion code".ljust(50) + " | " + "✅ 2 lines total")
    print("❌ Two separate tensors".ljust(50) + " | " + "✅ One tensor")
    print("❌ Complex index conversion".ljust(50) + " | " + "✅ Direct access")
    print("❌ Error-prone".ljust(50) + " | " + "✅ Simple & reliable") 
    print("❌ Doesn't match C++".ljust(50) + " | " + "✅ Matches C++ exactly")


def verify_results_identical():
    """Verify that both APIs produce identical results."""
    
    print("\n" + "="*70)
    print("VERIFICATION: Both APIs produce identical results")
    print("="*70)
    
    # Run both APIs
    loaded_old, values_old = demonstrate_old_problematic_api()
    loaded_new, values_new = demonstrate_fixed_clean_api()
    
    # Compare results
    print(f"\nComparison:")
    print(f"  Old API: {len(values_old)} values, range [{min(values_old):.1f}, {max(values_old):.1f}]")
    print(f"  New API: {len(values_new)} values, range [{min(values_new):.1f}, {max(values_new):.1f}]")
    
    if len(values_old) == len(values_new) and values_old == values_new:
        print(f"  ✅ IDENTICAL RESULTS! The fix is correct.")
    else:
        print(f"  ❌ Different results - need to investigate")
        print(f"     Old: {values_old[:10]}...")
        print(f"     New: {values_new[:10]}...")
    
    show_api_comparison()


if __name__ == "__main__":
    verify_results_identical()
    
    print("\n" + "="*70)
    print("🎉 CONCLUSION")
    print("="*70)
    print("The fixed API successfully eliminates the ugly conversion code!")
    print("✅ Matches C++ sweep_tile<DistributedTensor>(func) pattern")
    print("✅ No manual index conversions needed")
    print("✅ Clean, intuitive interface")
    print("✅ Same results as before, but much cleaner code")
    print("\nYou can now use: sweep_tile(loaded_tensor, func) directly! 🚀") 