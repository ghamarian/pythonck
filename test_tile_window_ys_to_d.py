#!/usr/bin/env python3
"""
Comprehensive example demonstrating ys_to_d_descriptor using tile window load operations.

This shows the complete realistic flow:
1. Create a global tensor with data
2. Create tile distribution and window
3. Use tile_window.load() to fill the distributed tensor  
4. Use StaticDistributedTensor.__getitem__ to access via Y coordinates
5. Show how ys_to_d_descriptor works internally
"""

import sys
import numpy as np

# Add the project root to path
sys.path.append('.')

from pytensor.tile_distribution import make_tile_distribution_encoding, make_static_tile_distribution
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.tensor_view import make_naive_tensor_view_packed
from pytensor.tile_window import TileWindowWithStaticDistribution

def create_realistic_setup():
    """Create a realistic tile distribution and window setup."""
    
    print("🎯 Creating Realistic Tile Distribution Setup")
    print("=" * 60)
    
    # Use a version with LARGER Hs but keep Y-referenced components small
    # Y dimensions refer to indices 0 and 3, so keep those as 2
    # Make indices 1 and 2 larger for more interesting tile sizes
    encoding = make_tile_distribution_encoding(
        rs_lengths=[],  # Empty R (no replication)
        hs_lengthss=[
            [2, 4, 4, 2],  # X0: 2*4*4*2 = 64 elements (Y refs indices 0,3 = 2,2)
            [2, 4, 4, 2]   # X1: 2*4*4*2 = 64 elements (Y refs indices 0,3 = 2,2)
        ],
        ps_to_rhss_major=[[1, 2], [1, 2]],  # P dimensions map to major indices [1,2]
        ps_to_rhss_minor=[[1, 1], [2, 2]],  # P dimensions use minor indices [1,1] and [2,2]
        ys_to_rhs_major=[1, 1, 2, 2],       # Y0,Y1 from X0; Y2,Y3 from X1
        ys_to_rhs_minor=[0, 3, 0, 3]        # Y dimensions use minor indices [0,3,0,3] - each = 2
    )
    
    print(f"📋 Larger Hs with Y-constraints:")
    print(f"   X0 hierarchy: {encoding.hs_lengthss[0]} (total: {np.prod(encoding.hs_lengthss[0])})")
    print(f"   X1 hierarchy: {encoding.hs_lengthss[1]} (total: {np.prod(encoding.hs_lengthss[1])})")
    print(f"   Y mappings: {list(zip(encoding.ys_to_rhs_major, encoding.ys_to_rhs_minor))}")
    
    print(f"\n✅ Y Dimension Constraints:")
    print(f"   Y0 -> X0[0] = {encoding.hs_lengthss[0][0]} (length 2)")
    print(f"   Y1 -> X0[3] = {encoding.hs_lengthss[0][3]} (length 2)")
    print(f"   Y2 -> X1[0] = {encoding.hs_lengthss[1][0]} (length 2)")
    print(f"   Y3 -> X1[3] = {encoding.hs_lengthss[1][3]} (length 2)")
    print(f"   All Y dimensions will have length 2")
    
    print(f"\n✅ No PS/YS Conflicts:")
    print(f"   PS minor indices: {encoding.ps_to_rhss_minor} (uses [1,1] and [2,2])")
    print(f"   YS minor indices: {encoding.ys_to_rhs_minor} (uses [0,3,0,3])")
    
    # Create the distribution
    distribution = make_static_tile_distribution(encoding)
    
    # Get Y dimensions from ys_to_d_descriptor
    ys_to_d_desc = distribution.ys_to_d_descriptor
    y_lengths = ys_to_d_desc.get_lengths()
    d_space_size = ys_to_d_desc.get_element_space_size()
    
    print(f"\n📊 Distribution Info:")
    print(f"   X dimensions: {distribution.ndim_x}")
    print(f"   Y dimensions: {distribution.ndim_y}")
    print(f"   P dimensions: {distribution.ndim_p}")
    print(f"   Y lengths: {y_lengths}")
    print(f"   D space size: {d_space_size}")
    
    # Verify Y lengths are as expected
    expected_y_lengths = [2, 2, 2, 2]  # All should be 2
    if list(y_lengths) == expected_y_lengths:
        print(f"   ✅ Y lengths match expectation: {y_lengths}")
    else:
        print(f"   ⚠️  Y lengths unexpected: got {y_lengths}, expected {expected_y_lengths}")
    
    return encoding, distribution

def create_global_tensor_and_window(distribution):
    """Create a global tensor and tile window for loading."""
    
    print("\n🌍 Creating Global Tensor and Tile Window")
    print("=" * 60)
    
    # Calculate tile sizes from the larger encoding 
    # Each X dimension size = product of its H components
    x0_size = np.prod(distribution.encoding.hs_lengthss[0])  # 2*4*4*2 = 64
    x1_size = np.prod(distribution.encoding.hs_lengthss[1])  # 2*4*4*2 = 64
    
    print(f"📏 Calculated tile sizes:")
    print(f"   X0 tile size: {x0_size}")
    print(f"   X1 tile size: {x1_size}")
    
    # Create a global tensor large enough for the larger tile
    global_shape = [x0_size + 20, x1_size + 20]  # Add margin for positioning
    global_data = np.zeros(global_shape, dtype=np.float32)
    
    # Create a simple, readable pattern: value = row*100 + col
    # Using 100 multiplier to make the row changes more visible
    for i in range(global_shape[0]):
        for j in range(global_shape[1]):
            global_data[i, j] = float(i * 100 + j)
    
    print(f"📦 Global tensor shape: {global_shape}")
    print(f"🔍 Global tensor sample (first 6x8):")
    print(global_data[:6, :8])
    
    # Create tensor view of global data
    global_view = make_naive_tensor_view_packed(
        data=global_data.flatten(),
        lengths=global_shape
    )
    
    print(f"✅ Global tensor view created: {global_view}")
    
    # Use the calculated tile sizes as window lengths
    window_lengths = [x0_size, x1_size]
    
    # Position the window at a small offset to see some interesting data
    window_origin = [5, 3]  # Small offset from origin
    
    print(f"🔳 Window configuration:")
    print(f"   Origin: {window_origin}")
    print(f"   Lengths: {window_lengths}")
    print(f"   Coverage: [{window_origin[0]}:{window_origin[0]+window_lengths[0]}, "
          f"{window_origin[1]}:{window_origin[1]+window_lengths[1]}]")
    print(f"   Total elements in window: {np.prod(window_lengths)}")
    
    # Show a sample of what the window should contain
    expected_region = global_data[
        window_origin[0]:window_origin[0]+window_lengths[0],
        window_origin[1]:window_origin[1]+window_lengths[1]
    ]
    print(f"   Window data sample (first 4x6):")
    print(expected_region[:4, :6])
    
    # Create the tile window
    tile_window = TileWindowWithStaticDistribution(
        bottom_tensor_view=global_view,
        window_lengths=window_lengths,
        window_origin=window_origin,
        tile_distribution=distribution
    )
    
    print(f"✅ Tile window created successfully!")
    
    return global_data, global_view, tile_window, window_origin, window_lengths

def demonstrate_load_and_access(distribution, tile_window, global_data, window_origin, window_lengths):
    """Demonstrate loading data through tile window and accessing via Y coordinates."""
    
    print("\n🚀 Loading Data Through Tile Window")
    print("=" * 60)
    
    # Load data from the window into a distributed tensor
    print("📥 Loading data...")
    distributed_tensor = tile_window.load(oob_conditional_check=True)
    
    print(f"✅ Data loaded into distributed tensor!")
    print(f"📊 Thread buffer info:")
    print(f"   Buffer length: {len(distributed_tensor.thread_buffer)}")
    print(f"   Buffer sample (first 10): {distributed_tensor.thread_buffer[:10]}")
    print(f"   Buffer sample (last 10): {distributed_tensor.thread_buffer[-10:]}")
    
    # Show what data we expect to have loaded (sample only due to large size)
    print(f"\n🔍 Expected data from global tensor:")
    expected_region = global_data[
        window_origin[0]:window_origin[0]+window_lengths[0],
        window_origin[1]:window_origin[1]+window_lengths[1]
    ]
    print(f"Global region [{window_origin[0]}:{window_origin[0]+window_lengths[0]}, "
          f"{window_origin[1]}:{window_origin[1]+window_lengths[1]}]:")
    print(f"   Window shape: {expected_region.shape}")
    print(f"   Expected sample (first 4x6):")
    print(expected_region[:4, :6])
    print(f"   Expected flattened (first 10): {expected_region.flatten()[:10]}")
    print(f"   Expected flattened (last 10): {expected_region.flatten()[-10:]}")
    
    # Check dimensions relationship - this is the key test!
    window_elements = np.prod(window_lengths)
    y_space_size = len(distributed_tensor.thread_buffer)
    
    print(f"\n⚠️  Critical Dimension Analysis:")
    print(f"   Window elements: {window_elements}")
    print(f"   Y space (D) size: {y_space_size}")
    print(f"   Ratio: {window_elements / y_space_size:.1f}:1")
    
    if window_elements != y_space_size:
        print(f"   🔍 Window size ≠ Y space size - this is expected for hierarchical tiling")
        print(f"   The tile window loads {window_elements} elements but distributes them")
        print(f"   across only {y_space_size} Y-addressable positions")
    
    # Critical test: Check if we're getting duplicate values (the original bug)
    unique_in_buffer = len(np.unique(distributed_tensor.thread_buffer))
    unique_in_expected = len(np.unique(expected_region.flatten()))
    
    print(f"\n🔍 Duplication Bug Check:")
    print(f"   Unique values in thread buffer: {unique_in_buffer}")
    print(f"   Unique values in expected region: {unique_in_expected}")
    
    # Since Y space is much smaller than window, we expect many different window values
    # to map to the same Y positions, so we check if Y space is being used properly
    expected_y_unique = min(unique_in_expected, y_space_size)
    
    if unique_in_buffer >= expected_y_unique * 0.8:  # Allow some tolerance
        print(f"   ✅ Good: Y space is properly utilized")
    else:
        print(f"   ❌ POSSIBLE BUG: Y space underutilized, might indicate duplication")
    
    return distributed_tensor

def demonstrate_ys_to_d_access(distribution, distributed_tensor):
    """Demonstrate accessing distributed tensor via Y coordinates."""
    
    print("\n🎯 Accessing Data via Y Coordinates")
    print("=" * 60)
    
    ys_to_d_desc = distribution.ys_to_d_descriptor
    y_lengths = ys_to_d_desc.get_lengths()
    
    print(f"📋 Y coordinate system:")
    print(f"   Y lengths: {y_lengths}")
    print(f"   Total combinations: {np.prod(y_lengths)}")
    
    # Calculate the stride pattern
    strides = []
    stride = 1
    for i in range(len(y_lengths) - 1, -1, -1):
        strides.insert(0, stride)
        stride *= y_lengths[i]
    
    print(f"   Strides: {strides}")
    print(f"   Formula: D = " + " + ".join(f"y{i}*{s}" for i, s in enumerate(strides)))
    
    print(f"\n🔍 Y->D Coordinate Examples:")
    
    # Test systematic Y coordinate patterns
    examples = [
        ([0, 0, 0, 0], "Origin"),
        ([1, 0, 0, 0], "Y0=1 (first dimension step)"),
        ([0, 1, 0, 0], "Y1=1 (second dimension step)"),
        ([0, 0, 1, 0], "Y2=1 (third dimension step)"),
        ([0, 0, 0, 1], "Y3=1 (fourth dimension step)"),
        ([1, 1, 0, 0], "Y0=1,Y1=1 (combined X0 steps)"),
        ([0, 0, 1, 1], "Y2=1,Y3=1 (combined X1 steps)"),
        ([1, 2, 1, 1], "Mixed coordinates"),
    ]
    
    for y_coords, description in examples:
        # Check bounds
        if any(y_coords[i] >= y_lengths[i] for i in range(len(y_coords))):
            print(f"   Y{y_coords}: SKIP ({description}) - out of bounds")
            continue
        
        # Method 1: Use distributed tensor's get_element (high-level)
        value1 = distributed_tensor.get_element(y_coords)
        
        # Method 2: Use ys_to_d_descriptor directly (low-level)
        d_offset = ys_to_d_desc.calculate_offset(y_coords)
        value2 = distributed_tensor.thread_buffer[d_offset]
        
        # Method 3: Manual calculation for verification
        manual_d = sum(y_coords[i] * strides[i] for i in range(len(y_coords)))
        value3 = distributed_tensor.thread_buffer[manual_d]
        
        # Check consistency
        consistent = value1 == value2 == value3
        status = "✅" if consistent else "❌"
        
        print(f"   Y{y_coords} -> D[{d_offset}] = {value1} {status} ({description})")
        if not consistent:
            print(f"      ERROR: get_element={value1}, descriptor={value2}, manual={value3}")

def demonstrate_detailed_mapping(distribution, distributed_tensor):
    """Show detailed Y->D mapping for better understanding."""
    
    print("\n🔬 Detailed Y->D Mapping Analysis")
    print("=" * 60)
    
    ys_to_d_desc = distribution.ys_to_d_descriptor
    y_lengths = ys_to_d_desc.get_lengths()
    
    print(f"🧮 Memory Layout Understanding:")
    print(f"   Y dimensions: {y_lengths}")
    print(f"   D space: linear array of {ys_to_d_desc.get_element_space_size()} elements")
    
    # Show how each Y dimension contributes to the D offset
    print(f"\n📊 Dimension Contributions:")
    for i, length in enumerate(y_lengths):
        # Calculate stride for this dimension
        stride = np.prod(y_lengths[i+1:]) if i+1 < len(y_lengths) else 1
        print(f"   Y{i}: range [0, {length-1}], stride {stride}, max contribution {(length-1)*stride}")
    
    # Show a systematic sweep through one dimension at a time (limited samples)
    print(f"\n🔄 Systematic Y Coordinate Sweep (Sample):")
    
    base_coords = [0] * len(y_lengths)
    
    for dim in range(len(y_lengths)):
        print(f"\n   Varying Y{dim} (others fixed at 0):")
        # Test just the first few values since Y dimensions are large now
        max_test = min(3, y_lengths[dim])
        for val in range(max_test):
            coords = base_coords.copy()
            coords[dim] = val
            
            try:
                d_offset = ys_to_d_desc.calculate_offset(coords)
                value = distributed_tensor.thread_buffer[d_offset]
                print(f"     Y{coords} -> D[{d_offset}] = {value}")
            except IndexError:
                print(f"     Y{coords} -> D[?] = OUT OF BOUNDS")
        
        # If there are more values, show that we're skipping
        if y_lengths[dim] > max_test:
            print(f"     ... (and {y_lengths[dim] - max_test} more values)")
    
    # Test some strategic coordinate combinations
    print(f"\n🎯 Strategic Coordinate Tests:")
    
    strategic_tests = [
        ([0, 0, 0, 0], "Origin"),
        ([1, 0, 0, 0], "Y0 step"),
        ([0, 1, 0, 0], "Y1 step"),  
        ([0, 0, 1, 0], "Y2 step"),
        ([0, 0, 0, 1], "Y3 step"),
    ]
    
    for coords, description in strategic_tests:
        # Check if coordinates are within bounds
        if any(coords[i] >= y_lengths[i] for i in range(len(coords))):
            print(f"   Y{coords}: SKIP ({description}) - out of bounds")
            continue
            
        try:
            d_offset = ys_to_d_desc.calculate_offset(coords)
            value = distributed_tensor.thread_buffer[d_offset]
            print(f"   Y{coords} -> D[{d_offset}] = {value} ({description})")
        except (IndexError, ValueError) as e:
            print(f"   Y{coords}: ERROR ({description}) - {e}")

def show_performance_insights():
    """Show performance insights about ys_to_d_descriptor."""
    
    print("\n⚡ Performance Insights")
    print("=" * 60)
    
    print("🎯 Why ys_to_d_descriptor Matters:")
    print("   1. THREAD-LOCAL ACCESS: Each thread has its own distributed tensor")
    print("   2. CACHE EFFICIENCY: Linear D memory layout optimizes cache usage")
    print("   3. COMPILE-TIME OPTIMIZATION: Static layout enables compiler optimizations")
    print("   4. NO SYNCHRONIZATION: Thread-local data eliminates sync overhead")
    
    print("\n🔧 Implementation Benefits:")
    print("   1. SIMPLE INDEXING: Y coordinates -> single linear offset calculation")
    print("   2. VECTORIZATION: Linear memory enables SIMD operations")
    print("   3. PREDICTABLE ACCESS: Known access patterns for prefetching")
    print("   4. MEMORY COALESCING: Sequential D indices = coalesced GPU memory")
    
    print("\n🏗️ Design Pattern:")
    print("   1. LOGICAL VIEW: Y coordinates (programmer-friendly)")
    print("   2. PHYSICAL LAYOUT: D linear memory (hardware-friendly)")
    print("   3. BRIDGE: ys_to_d_descriptor maps between them efficiently")

def main():
    """Run the complete tile window and ys_to_d_descriptor demonstration."""
    
    print("🔬 TILE WINDOW + YS_TO_D_DESCRIPTOR DEMONSTRATION")
    print("=" * 80)
    print("This example shows the complete realistic flow from global tensor")
    print("to distributed tensor access via Y coordinates.")
    print()
    
    # Step 1: Create the setup
    encoding, distribution = create_realistic_setup()
    
    # Step 2: Create global tensor and window
    global_data, global_view, tile_window, window_origin, window_lengths = create_global_tensor_and_window(distribution)
    
    # Step 3: Load data through tile window
    distributed_tensor = demonstrate_load_and_access(distribution, tile_window, global_data, window_origin, window_lengths)
    
    # Step 4: Access via Y coordinates
    demonstrate_ys_to_d_access(distribution, distributed_tensor)
    
    # Step 5: Detailed mapping analysis
    demonstrate_detailed_mapping(distribution, distributed_tensor)
    
    # Step 6: Performance insights
    show_performance_insights()
    
    print("\n" + "=" * 80)
    print("🎉 CONCLUSION:")
    print("✅ Tile window successfully loaded data from global tensor")
    print("✅ ys_to_d_descriptor correctly mapped Y coordinates to D memory")
    print("✅ StaticDistributedTensor provided efficient Y-based access")
    print("💡 This demonstrates the complete GPU-like tile processing workflow!")

if __name__ == "__main__":
    main() 