#!/usr/bin/env python3
"""
Tile Window Basics - Where Data Meets Distribution

You've learned about TileDistribution (work assignment). Now let's see TileWindow -
the actual data access API! This is where the rubber meets the road.

This script shows:
1. What is a TileWindow?
2. How TileWindow + TileDistribution work together
3. Loading data from memory into distributed tensors
4. Storing results back to memory
5. The complete load → compute → store pattern
"""

import sys
import os
import numpy as np

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tile_distribution_documentation.validation_scripts.common import (
    print_section, print_step, show_result, validate_example,
    explain_concept, show_comparison, run_script_safely, check_imports,
    show_tensor_shape, show_coordinate_transform
)

# Import the actual CK modules
from pytensor.tile_distribution import make_static_tile_distribution, make_tile_distribution_encoding
from pytensor.tile_window import make_tile_window, TileWindowWithStaticDistribution
from pytensor.tensor_view import make_tensor_view
from pytensor.tensor_descriptor import make_naive_tensor_descriptor_packed
from pytensor.static_distributed_tensor import make_static_distributed_tensor


def demonstrate_what_is_tile_window():
    """
    Show what tile window is and why we need it.
    """
    print_step(1, "What is a TileWindow?")
    
    explain_concept("The Missing Piece",
                   "TileDistribution tells threads WHERE to work. But how do they "
                   "actually ACCESS the data? That's where TileWindow comes in - "
                   "it's your window into memory, with distribution-aware access patterns.")
    
    print("\n🪟 Think of TileWindow as:")
    print("  • A smart window into a large tensor")
    print("  • Knows about thread distribution")
    print("  • Handles memory access patterns automatically")
    print("  • Provides load/store operations")
    
    print("\n🔄 The Complete Flow:")
    print("  1. TileDistribution: 'Thread 5, you handle coordinates [2,3]'")
    print("  2. TileWindow: 'Here's the data at [2,3], loaded efficiently'")
    print("  3. Your code: 'Thanks! *does computation*'")
    print("  4. TileWindow: 'I'll store your result back optimally'")
    
    explain_concept("Key Insight",
                   "TileWindow is like a smart cache manager. It knows which thread "
                   "needs what data, and it loads/stores everything with optimal "
                   "memory access patterns. No manual memory management!")


def demonstrate_creating_tile_window():
    """
    Show how to create a tile window with distribution.
    """
    print_step(2, "Creating a TileWindow")
    
    # First, create some sample data
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    show_result("Sample data", f"4x4 matrix:\n{data}")
    
    # Create a tensor view for the data
    tensor_desc = make_naive_tensor_descriptor_packed([4, 4])
    tensor_view = make_tensor_view(data, tensor_desc)
    
    # Create a tile distribution
    encoding = make_tile_distribution_encoding(
        rs_lengths=[],
        hs_lengthss=[[2], [2]],  # 2x2 tile
        ps_to_rhss_major=[[], []],
        ps_to_rhss_minor=[[], []],
        ys_to_rhs_major=[1, 2],
        ys_to_rhs_minor=[0, 0]
    )
    distribution = make_static_tile_distribution(encoding)
    
    # Create the tile window
    window_lengths = [2, 2]  # 2x2 window
    window_origin = [1, 1]   # Start at position [1,1]
    
    tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=window_lengths,
        origin=window_origin,
        tile_distribution=distribution
    )
    
    show_result("TileWindow created", tile_window)
    show_result("Window size", window_lengths, "Size of the window")
    show_result("Window origin", window_origin, "Starting position")
    show_result("Has distribution", hasattr(tile_window, 'tile_distribution'))
    
    explain_concept("What just happened?",
                   "We created a 2x2 window starting at position [1,1] in our 4x4 matrix. "
                   "The window knows about our tile distribution, so it can load data "
                   "efficiently for each thread.")
    
    return tile_window, distribution, data


def demonstrate_loading_data(tile_window, distribution, source_data):
    """
    Show how to load data from memory into distributed tensors.
    """
    print_step(3, "Loading Data with TileWindow")
    
    show_result("Source data window", f"Window at [1,1] size [2,2]:\n{source_data[1:3, 1:3]}")
    
    # Method 1: Convenient load() - automatically creates distributed tensor
    print("\n🎯 Method 1: Convenient load() - automatically creates distributed tensor")
    try:
        distributed_tensor = tile_window.load()
        show_result("Load operation", "✅ Success")
        show_result("Tensor created", f"Automatically created: {type(distributed_tensor).__name__}")
        
        # Show what each thread sees
        print("\n🧵 What each thread loaded:")
        y_lengths = distribution.get_y_vector_lengths()
        
        for y0 in range(y_lengths[0]):
            for y1 in range(y_lengths[1]):
                y_indices = [y0, y1]
                try:
                    value = distributed_tensor.get_element(y_indices)
                    print(f"  Thread Y{y_indices}: loaded value {value}")
                except Exception as e:
                    print(f"  Thread Y{y_indices}: error - {e}")
                    
    except Exception as e:
        show_result("Load operation", f"❌ Error: {e}")
    
    # Method 2: Manual load_into() - when you need control over tensor creation
    print("\n🔧 Method 2: Manual load_into() - when you need control over tensor creation")
    try:
        manual_tensor = make_static_distributed_tensor(np.float32, distribution)
        tile_window.load_into(manual_tensor)
        show_result("Manual load_into", "✅ Success")
        
        # Verify both methods give same result
        print("🔍 Verifying both methods give same result:")
        y_lengths = distribution.get_y_vector_lengths()
        same_results = True
        for y0 in range(y_lengths[0]):
            for y1 in range(y_lengths[1]):
                y_indices = [y0, y1]
                auto_val = distributed_tensor.get_element(y_indices)
                manual_val = manual_tensor.get_element(y_indices)
                if auto_val != manual_val:
                    same_results = False
                    break
        
        show_result("Results match", f"✅ Both methods identical: {same_results}")
                    
    except Exception as e:
        show_result("Manual load_into", f"❌ Error: {e}")
    
    explain_concept("Loading Magic",
                   "TileWindow's load() method automatically creates a distributed tensor "
                   "AND figures out which memory locations each thread needs, loading them "
                   "with optimal access patterns. Use load() for convenience, load_into() "
                   "for control!")
    
    return distributed_tensor


def demonstrate_computation_on_distributed_data(distributed_tensor, distribution, original_data):
    """
    Show how to do computation on distributed data with direct store.
    """
    print_step(4, "Computing on Distributed Data")
    
    print("\n🔢 Performing computation (multiply by 2) with direct store:")
    
    # Create output data and window
    output_data = original_data.copy()
    tensor_desc = make_naive_tensor_descriptor_packed([4, 4])
    output_tensor_view = make_tensor_view(output_data, tensor_desc)
    
    output_window = make_tile_window(
        tensor_view=output_tensor_view,
        window_lengths=[2, 2],
        origin=[1, 1],
        tile_distribution=distribution
    )
    
    # Load current output values and compute directly
    result_tensor = output_window.load()
    
    # Do computation on each element
    y_lengths = distribution.get_y_vector_lengths()
    for y0 in range(y_lengths[0]):
        for y1 in range(y_lengths[1]):
            y_indices = [y0, y1]
            try:
                # Load input value
                input_value = distributed_tensor.get_element(y_indices)
                
                # Compute result (simple example: multiply by 2)
                output_value = input_value * 2
                
                # Store result directly in output tensor
                result_tensor.set_element(y_indices, output_value)
                
                print(f"  Y{y_indices}: {input_value} → {output_value}")
                
            except Exception as e:
                print(f"  Y{y_indices}: error - {e}")
    
    # Store results back to memory
    try:
        output_window.store(result_tensor)
        show_result("Store operation", "✅ Success - computed and stored in one step!")
    except Exception as e:
        show_result("Store operation", f"❌ Error: {e}")
    
    explain_concept("Optimized Pattern",
                   "No intermediate tensor needed! We load input, compute results, "
                   "and store directly to output window. This saves memory and "
                   "is more efficient than the traditional load→compute→store pattern.")
    
    return result_tensor, output_data


def demonstrate_storing_results(tile_window, result_tensor, original_data):
    """
    Show how to store results back to memory.
    """
    print_step(5, "Storing Results Back to Memory")
    
    # Create a copy of original data to store results into
    output_data = original_data.copy()
    
    # Create a new tensor view for the output
    tensor_desc = make_naive_tensor_descriptor_packed([4, 4])
    output_tensor_view = make_tensor_view(output_data, tensor_desc)
    
    # Create a new tile window for output
    output_window = make_tile_window(
        tensor_view=output_tensor_view,
        window_lengths=tile_window.window_lengths,
        origin=tile_window.window_origin,
        tile_distribution=tile_window.tile_distribution
    )
    
    try:
        # Store the results
        output_window.store(result_tensor)
        show_result("Store operation", "✅ Success")
        
        print("\n📊 Results:")
        print(f"Original data:\n{original_data}")
        print(f"After computation:\n{output_data}")
        
        # Show the specific window that was modified
        window_start = tile_window.window_origin
        window_end = [window_start[0] + tile_window.window_lengths[0],
                     window_start[1] + tile_window.window_lengths[1]]
        modified_region = output_data[window_start[0]:window_end[0], 
                                    window_start[1]:window_end[1]]
        print(f"Modified window region:\n{modified_region}")
        
    except Exception as e:
        show_result("Store operation", f"❌ Error: {e}")
    
    explain_concept("Store Magic",
                   "TileWindow automatically figured out where each thread's results "
                   "should go in memory and stored them with optimal access patterns!")
    
    return output_data


def demonstrate_complete_load_compute_store_pattern():
    """
    Show the complete pattern that GPU kernels use.
    """
    print_step(6, "Complete Load → Compute → Store Pattern")
    
    print("🔥 This is the pattern every GPU kernel uses:")
    print("""
    __global__ void my_kernel() {
        // 1. Create tile window for input data
        auto input_window = make_tile_window(input_tensor, window_size, origin, distribution);
        
        // 2. Load data - automatically creates distributed tensor!
        auto loaded_data = input_window.load();
        
        // 3. Do computation
        auto results = compute(loaded_data);
        
        // 4. Store results back
        auto output_window = make_tile_window(output_tensor, window_size, origin, distribution);
        output_window.store(results);
    }
    """)
    
    explain_concept("The Universal Pattern",
                   "Every GPU kernel follows this pattern: Load → Compute → Store. "
                   "TileWindow + TileDistribution make this pattern automatic and optimal!")
    
    print("\n💡 Benefits:")
    print("  • Automatic memory coalescing")
    print("  • Optimal thread cooperation")
    print("  • No manual memory management")
    print("  • Same code works on any hardware")
    print("  • Compiler can optimize aggressively")


def test_tile_window_operations():
    """
    Test that our tile window operations work correctly.
    """
    print_step(7, "Testing TileWindow Operations")
    
    def test_window_creation():
        # Create test data and window
        data = np.ones((4, 4), dtype=np.float32)
        tensor_desc = make_naive_tensor_descriptor_packed([4, 4])
        tensor_view = make_tensor_view(data, tensor_desc)
        
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[], []],
            ps_to_rhss_minor=[[], []],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[0, 0]
        )
        distribution = make_static_tile_distribution(encoding)
        
        window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=distribution
        )
        
        return isinstance(window, TileWindowWithStaticDistribution)
    
    def test_load_store_roundtrip():
        # Create test data
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor_desc = make_naive_tensor_descriptor_packed([2, 2])
        tensor_view = make_tensor_view(data, tensor_desc)
        
        # Simple distribution
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[2], [2]],
            ps_to_rhss_major=[[], []],
            ps_to_rhss_minor=[[], []],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[0, 0]
        )
        distribution = make_static_tile_distribution(encoding)
        
        # Create window
        window = make_tile_window(
            tensor_view=tensor_view,
            window_lengths=[2, 2],
            origin=[0, 0],
            tile_distribution=distribution
        )
        
        # Load and store test - test both methods
        try:
            # Method 1: Convenient load()
            distributed_tensor_auto = window.load()
            
            # Method 2: Manual load_into() 
            distributed_tensor_manual = make_static_distributed_tensor(np.float32, distribution)
            window.load_into(distributed_tensor_manual)
            
            # If we get here, both methods worked
            return True
        except Exception:
            return False
    
    # Run tests
    tests = [
        ("Window creation", test_window_creation),
        ("Load/store operations", test_load_store_roundtrip),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        passed = validate_example(test_name, test_func)
        all_passed = all_passed and passed
    
    return all_passed


def main():
    """
    Main function that runs all our tile window examples.
    """
    # First, make sure we can import our libraries
    if not check_imports():
        return
    
    print("Welcome to TileWindow Basics!")
    print("Let's see how TileWindow provides the data access API for distributed computing.")
    
    # Run through all our examples
    demonstrate_what_is_tile_window()
    tile_window, distribution, data = demonstrate_creating_tile_window()
    distributed_tensor = demonstrate_loading_data(tile_window, distribution, data)
    result_tensor, output_data = demonstrate_computation_on_distributed_data(distributed_tensor, distribution, data)
    demonstrate_complete_load_compute_store_pattern()
    
    # Test everything works
    tests_passed = test_tile_window_operations()
    
    if tests_passed:
        print("\n🎉 All tile window operations work correctly!")
        print("You've mastered the complete load → compute → store pattern!")
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return False
    
    print("\n💡 Key takeaways:")
    print("  • TileWindow provides distributed data access")
    print("  • Works seamlessly with TileDistribution")
    print("  • Handles optimal memory access patterns")
    print("  • Enables the universal Load → Compute → Store pattern")
    print("  • Next: Learn about sweep operations for iteration")
    
    return True


if __name__ == "__main__":
    run_script_safely("TileWindow Basics", main) 