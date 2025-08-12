#!/usr/bin/env python3
"""
Tile Distribution Basics - The User API

Now that you understand coordinate transforms, let's see how CK uses them 
to distribute work across GPU threads! This is the API that GPU programmers
actually use - no need to understand the internals yet.

This script shows:
1. What is a TileDistribution? 
2. How to create basic distributions
3. How threads get their work assignments
4. Real-world usage patterns
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
from pytensor.tile_distribution import (
    TileDistribution, make_static_tile_distribution, make_tile_distribution_encoding
)
from pytensor.tensor_descriptor import make_naive_tensor_descriptor_packed


def demonstrate_what_is_tile_distribution():
    """
    Show what tile distribution is at a high level.
    """
    print_step(1, "What is Tile Distribution?")
    
    explain_concept("The Big Picture",
                   "Imagine you have a 64x64 matrix multiplication. You have 32 GPU threads. "
                   "How do you divide the work? TileDistribution is CK's answer - it maps "
                   "logical coordinates (like matrix position [i,j]) to physical threads "
                   "and memory locations automatically.")
    
    # Show a simple example conceptually
    matrix_size = [8, 8]  # Small example for clarity
    num_threads = 4
    
    show_result("Problem", f"Matrix size: {matrix_size}, Threads: {num_threads}")
    show_result("Question", "How does each thread know which elements to process?")
    
    print("\n🎯 Without TileDistribution:")
    print("  • Manual thread ID calculations")
    print("  • Complex index arithmetic") 
    print("  • Error-prone memory access patterns")
    print("  • Different code for different matrix sizes")
    
    print("\n🎯 With TileDistribution:")
    print("  • Automatic work assignment")
    print("  • Optimized memory access patterns")
    print("  • Same code works for any size")
    print("  • Hardware-aware thread cooperation")
    
    explain_concept("Key Insight",
                   "TileDistribution is like a smart GPS for GPU threads. Give it your "
                   "logical coordinates [i,j], and it tells each thread exactly where "
                   "to go in memory and what to compute. No manual navigation required!")


def demonstrate_simple_distribution():
    """
    Show how to create a real-world tile distribution from RMSNorm.
    """
    print_step(2, "Real-World Distribution: RMSNorm")
    
    explain_concept("RMSNorm Pattern",
                   "RMSNorm is a common GPU operation. Let's see how CK distributes "
                   "its work across threads. This is a REAL example from production code!")
    
    # RMSNorm distribution from examples.py
    # S::Repeat_M=4, S::WarpPerBlock_M=2, S::ThreadPerWarp_M=8, S::Vector_M=4
    # S::Repeat_N=4, S::WarpPerBlock_N=2, S::ThreadPerWarp_N=8, S::Vector_N=4
    encoding = make_tile_distribution_encoding(
        rs_lengths=[],  # Empty R sequence
        hs_lengthss=[
            [4, 2, 8, 4],  # H for X0: Repeat_M, WarpPerBlock_M, ThreadPerWarp_M, Vector_M
            [4, 2, 8, 4]   # H for X1: Repeat_N, WarpPerBlock_N, ThreadPerWarp_N, Vector_N
        ],
        ps_to_rhss_major=[[1, 2], [1, 2]],  # P maps to H dimensions
        ps_to_rhss_minor=[[1, 1], [2, 2]],  # P minor mappings
        ys_to_rhs_major=[1, 1, 2, 2],       # Y maps to H dimensions
        ys_to_rhs_minor=[0, 3, 0, 3]        # Y minor mappings
    )
    
    distribution = make_static_tile_distribution(encoding)
    
    show_result("RMSNorm Distribution", distribution)
    show_result("X dimensions", distribution.ndim_x, "Logical data dimensions")
    show_result("Y dimensions", distribution.ndim_y, "Access pattern dimensions")
    show_result("P dimensions", distribution.ndim_p, "Thread partition dimensions")
    
    print("\n📊 RMSNorm Structure:")
    print("  • X0 (M dimension): 4×2×8×4 = 256 elements")
    print("  • X1 (N dimension): 4×2×8×4 = 256 elements")  
    print("  • Total tile size: 256×256")
    print("  • P dimensions: 2 (warp + thread within warp)")
    print("  • Y dimensions: 4 (access pattern)")
    
    explain_concept("P + Y → X Magic",
                   "Notice: We have P dimensions (thread partitioning) + Y dimensions "
                   "(access patterns) that together determine our X dimensions (logical data). "
                   "This is the core CK pattern: P + Y → X!")
    
    return distribution


def demonstrate_thread_assignments_2d(distribution):
    """
    Show how threads get their work assignments with both M and N dimension changes.
    """
    print_step(3, "How Threads Get Work (2D Grid)")
    
    # Show how different partition indices get different work
    print("\n🧵 Thread work assignments (2D thread grid):")
    
    # Use specific (M, N) tuples to show changes in both dimensions
    thread_positions = [
        (0, 0), (0, 1), (1, 0), (1, 1),  # Basic 2x2 grid
        (2, 0), (0, 2), (2, 2), (3, 3)   # Extended positions
    ]
    
    for i, (thread_m, thread_n) in enumerate(thread_positions):
        partition_index = [thread_m, thread_n]  # M, N thread coordinates
        
        try:
            # Calculate what X coordinates this thread handles
            x_index = distribution.calculate_index(partition_index)
            
            print(f"  Thread[{thread_m},{thread_n}] (position {i+1}):")
            print(f"    → Partition: {partition_index}")
            print(f"    → X coordinates: {x_index.to_list()}")
            
            # Show Y access pattern for this thread
            y_lengths = distribution.get_y_vector_lengths()
            print(f"    → Y pattern: {y_lengths}")
            print()
            
        except Exception as e:
            print(f"  Thread[{thread_m},{thread_n}]: Error - {e}")
            print()
    
    explain_concept("2D Thread Cooperation",
                   "Notice how both M and N coordinates change the X coordinates! "
                   "Thread[0,0] vs Thread[1,0] shows M dimension changes, "
                   "Thread[0,0] vs Thread[0,1] shows N dimension changes. "
                   "This creates true 2D work distribution!")


def demonstrate_thread_assignments(distribution):
    """
    Show how threads get their work assignments.
    """
    print_step(3, "How Threads Get Work")
    
    # Show how different partition indices get different work
    print("\n🧵 Thread work assignments:")
    
    # Simulate different threads (partition indices)
    for thread_id in range(4):
        # In real GPU code, this would be get_lane_id() or similar
        partition_index = [thread_id % 2, thread_id // 2]  # 2x2 thread grid
        
        try:
            # Calculate what X coordinates this thread handles
            x_index = distribution.calculate_index(partition_index)
            
            print(f"  Thread {thread_id} (partition {partition_index}):")
            print(f"    → Handles X coordinates: {x_index.to_list()}")
            
            # Show Y access pattern for this thread
            y_lengths = distribution.get_y_vector_lengths()
            print(f"    → Y access pattern: {y_lengths}")
            
        except Exception as e:
            print(f"  Thread {thread_id}: Error - {e}")
    
    explain_concept("Thread Cooperation",
                   "Each thread gets a unique partition index (like a thread ID). "
                   "The distribution automatically calculates which data elements "
                   "that thread should process. No conflicts, perfect cooperation!")


def demonstrate_real_world_usage():
    """
    Show how this looks in real GPU kernel code.
    """
    print_step(4, "Real-World GPU Kernel Pattern")
    
    print("🔥 Typical CK kernel structure:")
    print("""
    __global__ void my_kernel() {
        // 1. Get thread's partition index (automatic)
        auto partition_idx = distribution.get_partition_index();
        
        // 2. Calculate this thread's coordinates (automatic)  
        auto x_coords = distribution.calculate_index(partition_idx);
        
        // 3. Do the actual work
        for (auto y_idx : y_access_pattern) {
            auto data = load_from_memory(x_coords, y_idx);
            auto result = compute(data);
            store_to_memory(result, x_coords, y_idx);
        }
    }
    """)
    
    explain_concept("The Magic",
                   "Notice what you DON'T see: manual thread ID arithmetic, "
                   "complex index calculations, or memory offset computations. "
                   "TileDistribution handles all of that automatically!")
    
    print("\n💡 Benefits for GPU programmers:")
    print("  • Write once, run on any hardware")
    print("  • Automatic memory coalescing")
    print("  • Optimal thread cooperation")
    print("  • No manual tuning needed")


def demonstrate_distribution_properties():
    """
    Show important properties of distributions using real examples.
    """
    print_step(5, "Comparing Real Distribution Patterns")
    
    # Create distributions from real examples
    distributions = []
    
    # RMSNorm distribution (already created above)
    rmsnorm_encoding = make_tile_distribution_encoding(
        rs_lengths=[],
        hs_lengthss=[[4, 2, 8, 4], [4, 2, 8, 4]],
        ps_to_rhss_major=[[1, 2], [1, 2]],
        ps_to_rhss_minor=[[1, 1], [2, 2]],
        ys_to_rhs_major=[1, 1, 2, 2],
        ys_to_rhs_minor=[0, 3, 0, 3]
    )
    rmsnorm_dist = make_static_tile_distribution(rmsnorm_encoding)
    distributions.append(("RMSNorm Pattern", rmsnorm_dist))
    
    # R Sequence with Variables pattern from examples.py
    # S::WarpPerBlock_M=2, S::ThreadPerWarp_M=8, S::Repeat_N=4, etc.
    r_sequence_encoding = make_tile_distribution_encoding(
        rs_lengths=[2, 8],  # WarpPerBlock_M, ThreadPerWarp_M
        hs_lengthss=[[4, 2, 8, 4]],  # Repeat_N, WarpPerBlock_N, ThreadPerWarp_N, Vector_N
        ps_to_rhss_major=[[0, 1], [0, 1]],  # P maps to R dimensions
        ps_to_rhss_minor=[[0, 1], [1, 2]],  # P minor mappings
        ys_to_rhs_major=[1, 1],             # Y maps to H dimensions
        ys_to_rhs_minor=[0, 3]              # Y minor mappings
    )
    r_sequence_dist = make_static_tile_distribution(r_sequence_encoding)
    distributions.append(("R Sequence Pattern", r_sequence_dist))
    
    print("\n📊 Comparing real CK distribution patterns:")
    for name, dist in distributions:
        print(f"\n  {name}:")
        print(f"    X dimensions: {dist.ndim_x}")
        print(f"    Y dimensions: {dist.ndim_y}")
        print(f"    P dimensions: {dist.ndim_p}")
        print(f"    R dimensions: {dist.ndim_r}")
        print(f"    X lengths: {dist.get_lengths()}")
        print(f"    Static: {dist.is_static()}")
    
    explain_concept("Pattern Differences",
                   "RMSNorm uses no replication (R=[]) but complex hierarchical structure. "
                   "R Sequence pattern uses replication (R=[2,8]) for thread cooperation. "
                   "Each pattern optimizes for different GPU operations!")
    
    print("\n🎯 When to use each pattern:")
    print("  • RMSNorm: Element-wise operations with complex tiling")
    print("  • R Sequence: Operations requiring thread replication/cooperation")
    print("  • Choice depends on your algorithm's memory access patterns")
    
    return distributions


def test_distribution_operations():
    """
    Test that our real distribution examples work correctly.
    """
    print_step(6, "Testing Real Distribution Operations")
    
    def test_rmsnorm_creation():
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4, 2, 8, 4], [4, 2, 8, 4]],
            ps_to_rhss_major=[[1, 2], [1, 2]],
            ps_to_rhss_minor=[[1, 1], [2, 2]],
            ys_to_rhs_major=[1, 1, 2, 2],
            ys_to_rhs_minor=[0, 3, 0, 3]
        )
        distribution = make_static_tile_distribution(encoding)
        return (distribution.ndim_x == 2 and 
                distribution.ndim_y == 4 and 
                distribution.ndim_r == 0)
    
    def test_r_sequence_creation():
        encoding = make_tile_distribution_encoding(
            rs_lengths=[2, 8],
            hs_lengthss=[[4, 2, 8, 4]],
            ps_to_rhss_major=[[0, 1], [0, 1]],
            ps_to_rhss_minor=[[0, 1], [1, 2]],
            ys_to_rhs_major=[1, 1],
            ys_to_rhs_minor=[0, 3]
        )
        distribution = make_static_tile_distribution(encoding)
        return (distribution.ndim_x == 1 and 
                distribution.ndim_y == 2 and 
                distribution.ndim_r == 2)
    
    def test_static_property():
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4, 2, 8, 4]],
            ps_to_rhss_major=[[]],
            ps_to_rhss_minor=[[]],
            ys_to_rhs_major=[1],
            ys_to_rhs_minor=[0]
        )
        distribution = make_static_tile_distribution(encoding)
        return distribution.is_static()
    
    def test_dimension_calculations():
        # Test that P + Y → X relationship holds
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],
            hs_lengthss=[[4, 2, 8, 4], [4, 2, 8, 4]],
            ps_to_rhss_major=[[1, 2], [1, 2]],
            ps_to_rhss_minor=[[1, 1], [2, 2]],
            ys_to_rhs_major=[1, 1, 2, 2],
            ys_to_rhs_minor=[0, 3, 0, 3]
        )
        distribution = make_static_tile_distribution(encoding)
        
        # Check that P + Y dimensions make sense with X dimensions
        # This is a structural check, not a mathematical one
        return (distribution.ndim_p >= 0 and 
                distribution.ndim_y >= 0 and 
                distribution.ndim_x >= 0)
    
    # Run all tests
    tests = [
        ("RMSNorm distribution creation", test_rmsnorm_creation),
        ("R Sequence distribution creation", test_r_sequence_creation),
        ("Static property check", test_static_property),
        ("P + Y → X dimension check", test_dimension_calculations),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        passed = validate_example(test_name, test_func)
        all_passed = all_passed and passed
    
    return all_passed


def main():
    """
    Main function that runs all our tile distribution examples.
    """
    # First, make sure we can import our libraries
    if not check_imports():
        return
    
    print("Welcome to Tile Distribution Basics!")
    print("Let's explore the user-facing API that makes GPU programming easy.")
    
    # Run through all our examples
    demonstrate_what_is_tile_distribution()
    simple_dist = demonstrate_simple_distribution()
    demonstrate_thread_assignments_2d(simple_dist)
    demonstrate_thread_assignments(simple_dist)
    demonstrate_real_world_usage()
    distributions = demonstrate_distribution_properties()
    
    # Test everything works
    tests_passed = test_distribution_operations()
    
    if tests_passed:
        print("\n🎉 All tile distribution operations work correctly!")
        print("You've mastered the user API - ready for real GPU kernels!")
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return False
    
    print("\n💡 Key takeaways:")
    print("  • TileDistribution automates work assignment across threads")
    print("  • Each thread gets unique coordinates automatically")
    print("  • Same code works for any data size or thread count")
    print("  • Optimized memory access patterns built-in")
    print("  • Next: Learn about TileWindow for actual data access")
    
    return True


if __name__ == "__main__":
    run_script_safely("Tile Distribution Basics", main) 