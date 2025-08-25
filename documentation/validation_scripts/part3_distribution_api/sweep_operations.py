#!/usr/bin/env python3
"""
Sweep Operations - Iterating Over Distributed Data

You've learned TileDistribution (work assignment) and TileWindow (data access).
Now let's see sweep operations - the elegant way to iterate over distributed data!

This script shows:
1. What are sweep operations?
2. How sweep_tile works with distributed tensors
3. Different iteration patterns
4. The complete workflow: load → sweep → compute → store
"""

import sys
import os
import numpy as np

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from documentation.validation_scripts.common import (
    print_section, print_step, show_result, validate_example,
    explain_concept, show_comparison, run_script_safely, check_imports
)

# Import the actual CK modules
from pytensor.tile_distribution import make_static_tile_distribution, make_tile_distribution_encoding
from pytensor.tile_window import make_tile_window
from pytensor.tensor_view import make_tensor_view
from pytensor.tensor_descriptor import make_naive_tensor_descriptor_packed
from pytensor.static_distributed_tensor import make_static_distributed_tensor
from pytensor.sweep_tile import sweep_tile, make_tile_sweeper


def demonstrate_what_are_sweep_operations():
    """
    Show what sweep operations are and why we need them.
    """
    print_step(1, "What are Sweep Operations?")
    
    explain_concept("The Final Piece",
                   "You have distributed data loaded from TileWindow. Now you need to "
                   "process every element. How do you iterate elegantly? That's where "
                   "sweep operations come in - they provide clean iteration patterns!")
    
    print("\n🔄 The Complete GPU Workflow:")
    print("  1. TileDistribution: 'Here's how to divide work'")
    print("  2. TileWindow: 'Here's your data, loaded efficiently'")
    print("  3. Sweep Operations: 'Here's how to process every element'")
    print("  4. Your code: 'Thanks! *does computation*'")
    
    print("\n🎯 Without Sweep Operations:")
    print("  • Manual nested loops over Y dimensions")
    print("  • Complex index calculations")
    print("  • Easy to miss elements or double-process")
    print("  • Different code for different access patterns")
    
    print("\n🎯 With Sweep Operations:")
    print("  • Elegant lambda-based iteration")
    print("  • Automatic handling of all elements")
    print("  • Same pattern for any distribution")
    print("  • Compiler-optimizable")
    
    explain_concept("Key Insight",
                   "Sweep operations are like forEach() for distributed tensors. "
                   "Give them a function, and they'll call it for every element "
                   "in the optimal order. No manual loops needed!")


def demonstrate_basic_sweep():
    """
    Show basic sweep operations on distributed tensors.
    """
    print_step(2, "Basic Sweep Operations")
    
    # Create a simple distribution and distributed tensor
    encoding = make_tile_distribution_encoding(
        rs_lengths=[],
        hs_lengthss=[[2], [2]],  # 2x2 distribution
        ps_to_rhss_major=[[], []],
        ps_to_rhss_minor=[[], []],
        ys_to_rhs_major=[1, 2],
        ys_to_rhs_minor=[0, 0]
    )
    distribution = make_static_tile_distribution(encoding)
    
    # Create distributed tensor with some data
    distributed_tensor = make_static_distributed_tensor(np.float32, distribution)
    
    # Populate with test data
    test_values = [[1.0, 2.0], [3.0, 4.0]]
    y_lengths = distribution.get_y_vector_lengths()
    
    for y0 in range(y_lengths[0]):
        for y1 in range(y_lengths[1]):
            distributed_tensor.set_element([y0, y1], test_values[y0][y1])
    
    show_result("Test data loaded", "2x2 tensor with values [[1,2], [3,4]]")
    
    # Demonstrate sweep operation
    print("\n🔄 Sweeping over distributed tensor:")
    
    collected_values = []
    
    def collect_value(*dstr_indices):
        """Function to collect values during sweep."""
        # Get Y indices from distributed indices using the distribution
        y_indices = distribution.get_y_indices_from_distributed_indices(list(dstr_indices))
        value = distributed_tensor.get_element(y_indices)
        collected_values.append((y_indices.copy(), value))
        print(f"  Visited Y{y_indices}: value = {value}")
    
    # Perform the sweep
    sweep_tile(distributed_tensor, collect_value)
    
    show_result("Sweep completed", f"Visited {len(collected_values)} elements")
    
    explain_concept("What happened?",
                   "sweep_tile automatically iterated over all Y indices "
                   "in the distributed tensor and called our function for each element. "
                   "No manual loops, no missed elements!")
    
    return distributed_tensor, collected_values


def demonstrate_sweep_with_computation(distributed_tensor):
    """
    Show sweep operations with actual computation.
    """
    print_step(3, "Sweep with Computation")
    
    # Create result tensor
    result_tensor = make_static_distributed_tensor(np.float32, distributed_tensor.tile_distribution)
    
    print("\n🔢 Computing squares using sweep:")
    
    def compute_square(*dstr_indices):
        """Compute square of each element."""
        # Get Y indices from distributed indices
        y_indices = distributed_tensor.tile_distribution.get_y_indices_from_distributed_indices(list(dstr_indices))
        input_value = distributed_tensor.get_element(y_indices)
        output_value = input_value ** 2
        result_tensor.set_element(y_indices, output_value)
        print(f"  Y{y_indices}: {input_value}² = {output_value}")
    
    # Perform computation sweep
    sweep_tile(distributed_tensor, compute_square)
    
    # Verify results
    print("\n📊 Results verification:")
    def verify_result(*dstr_indices):
        y_indices = distributed_tensor.tile_distribution.get_y_indices_from_distributed_indices(list(dstr_indices))
        original = distributed_tensor.get_element(y_indices)
        computed = result_tensor.get_element(y_indices)
        expected = original ** 2
        print(f"  Y{y_indices}: {original} → {computed} (expected {expected})")
    
    sweep_tile(result_tensor, verify_result)
    
    explain_concept("Computation Pattern",
                   "This is the classic pattern: sweep over input tensor, compute "
                   "something, store in result tensor. The sweep handles all the "
                   "iteration complexity automatically!")
    
    return result_tensor


def demonstrate_advanced_sweep_patterns():
    """
    Show advanced sweep patterns and tile sweeper.
    """
    print_step(4, "Advanced Sweep Patterns")
    
    # Create a larger distribution for more interesting patterns
    encoding = make_tile_distribution_encoding(
        rs_lengths=[],
        hs_lengthss=[[4], [4]],  # 4x4 distribution
        ps_to_rhss_major=[[], []],
        ps_to_rhss_minor=[[], []],
        ys_to_rhs_major=[1, 2],
        ys_to_rhs_minor=[0, 0]
    )
    distribution = make_static_tile_distribution(encoding)
    distributed_tensor = make_static_distributed_tensor(np.float32, distribution)
    
    # Fill with test data
    y_lengths = distribution.get_y_vector_lengths()
    for y0 in range(y_lengths[0]):
        for y1 in range(y_lengths[1]):
            distributed_tensor.set_element([y0, y1], y0 * 4 + y1)
    
    print("\n🎛️ Using TileSweeper for controlled iteration:")
    
    # Create a tile sweeper
    def process_element(*dstr_indices):
        y_indices = distributed_tensor.tile_distribution.get_y_indices_from_distributed_indices(list(dstr_indices))
        value = distributed_tensor.get_element(y_indices)
        print(f"  Processing Y{y_indices}: {value}")
    
    sweeper = make_tile_sweeper(distributed_tensor, process_element)
    
    show_result("Sweeper created", sweeper)
    show_result("Number of accesses", sweeper.get_num_of_access())
    
    # Execute the sweep
    print("\n🔄 Executing sweep:")
    sweeper()
    
    explain_concept("TileSweeper Benefits",
                   "TileSweeper gives you more control - you can query the number "
                   "of accesses, execute specific access indices, or run the full "
                   "sweep. Perfect for debugging and optimization!")


def demonstrate_real_world_sweep_pattern():
    """
    Show how sweep operations fit in real GPU kernels.
    """
    print_step(5, "Real-World Sweep Pattern")
    
    print("🔥 Complete GPU kernel with sweep operations:")
    print("""
    __global__ void my_kernel() {
        // 1. Load data using tile window
        auto input_window = make_tile_window(input_tensor, window_size, origin, distribution);
        auto loaded_data = input_window.load();
        
        // 2. Create result tensor
        auto results = make_static_distributed_tensor(distribution);
        
        // 3. Sweep over loaded data and compute
        sweep_tile(loaded_data, [&](auto y_indices) {
            auto input_value = loaded_data.get_element(y_indices);
            auto output_value = compute_function(input_value);
            results.set_element(y_indices, output_value);
        });
        
        // 4. Store results back
        auto output_window = make_tile_window(output_tensor, window_size, origin, distribution);
        output_window.store(results);
    }
    """)
    
    explain_concept("The Universal GPU Pattern",
                   "This is the pattern every GPU kernel uses: "
                   "Load → Sweep & Compute → Store. Sweep operations make the "
                   "middle step clean and efficient!")
    
    print("\n💡 Benefits of sweep operations:")
    print("  • Clean, readable code")
    print("  • Automatic optimization by compiler")
    print("  • Same pattern for any distribution")
    print("  • No manual index management")
    print("  • Easy to debug and test")


def test_sweep_operations():
    """
    Test that sweep operations work correctly.
    """
    print_step(6, "Testing Sweep Operations")
    
    def test_basic_sweep():
        """Test basic sweep functionality."""
        try:
            # Create simple distribution
            encoding = make_tile_distribution_encoding(
                rs_lengths=[],
                hs_lengthss=[[2], [2]],
                ps_to_rhss_major=[[], []],
                ps_to_rhss_minor=[[], []],
                ys_to_rhs_major=[1, 2],
                ys_to_rhs_minor=[0, 0]
            )
            distribution = make_static_tile_distribution(encoding)
            distributed_tensor = make_static_distributed_tensor(np.float32, distribution)
            
            # Test sweep
            count = 0
            def count_elements(*dstr_indices):
                nonlocal count
                count += 1
            
            sweep_tile(distributed_tensor, count_elements)
            
            # Should visit 4 elements (2x2)
            return count == 4
        except Exception as e:
            print(f"Basic sweep error: {e}")
            return False
    
    def test_tile_sweeper():
        """Test tile sweeper functionality."""
        try:
            # Create distribution
            encoding = make_tile_distribution_encoding(
                rs_lengths=[],
                hs_lengthss=[[2], [2]],
                ps_to_rhss_major=[[], []],
                ps_to_rhss_minor=[[], []],
                ys_to_rhs_major=[1, 2],
                ys_to_rhs_minor=[0, 0]
            )
            distribution = make_static_tile_distribution(encoding)
            distributed_tensor = make_static_distributed_tensor(np.float32, distribution)
            
            # Create sweeper
            def dummy_func(*dstr_indices):
                pass
            
            sweeper = make_tile_sweeper(distributed_tensor, dummy_func)
            
            # Test that sweeper was created and has expected number of accesses
            return sweeper.get_num_of_access() > 0
        except Exception as e:
            print(f"Tile sweeper error: {e}")
            return False
    
    # Run tests
    tests = [
        ("Basic sweep functionality", test_basic_sweep),
        ("Tile sweeper functionality", test_tile_sweeper),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        passed = validate_example(test_name, test_func)
        all_passed = all_passed and passed
    
    return all_passed


def main():
    """
    Main function that runs all our sweep operation examples.
    """
    # First, make sure we can import our libraries
    if not check_imports():
        return
    
    print("Welcome to Sweep Operations!")
    print("Let's see how to elegantly iterate over distributed data.")
    
    # Run through all our examples
    demonstrate_what_are_sweep_operations()
    distributed_tensor, collected_values = demonstrate_basic_sweep()
    result_tensor = demonstrate_sweep_with_computation(distributed_tensor)
    demonstrate_advanced_sweep_patterns()
    demonstrate_real_world_sweep_pattern()
    
    # Test everything works
    tests_passed = test_sweep_operations()
    
    if tests_passed:
        print("\n🎉 All sweep operations work correctly!")
        print("You've mastered the complete distributed computing workflow!")
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return False
    
    print("\n💡 Key takeaways:")
    print("  • Sweep operations provide elegant iteration over distributed data")
    print("  • No manual loops or index management needed")
    print("  • Same pattern works for any distribution")
    print("  • Completes the Load → Sweep & Compute → Store workflow")
    print("  • Next: Learn about the internal encoding that makes it all work")
    
    return True


if __name__ == "__main__":
    run_script_safely("Sweep Operations", main)
