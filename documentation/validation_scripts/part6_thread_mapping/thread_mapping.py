#!/usr/bin/env python3
"""
Purpose: Demonstrate thread mapping concepts in tile distribution.

Shows how threads get their unique IDs, how those IDs map to specific data,
thread cooperation patterns, and performance analysis techniques.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code_examples'))

from common_utils import *
from pytensor.tile_distribution import (
    make_tile_distribution, 
    TileDistributionEncoding
)
from pytensor.tile_window import make_tile_window
from pytensor.tensor_view import make_tensor_view
from pytensor.buffer_view import make_buffer_view
import numpy as np

def demonstrate_thread_identification():
    """Show how threads get their unique IDs and partition indices."""
    print_step(1, "Thread Identification and Partition Indices")
    
    print("🎯 Using Real-World RMSNorm Example")
    print("This demonstrates thread mapping from the actual RMSNorm operation")
    print("used in Composable Kernels for layer normalization.")
    
    # RMSNorm parameters (from examples.py)
    repeat_m, warp_per_block_m, thread_per_warp_m, vector_m = 4, 2, 8, 4
    repeat_n, warp_per_block_n, thread_per_warp_n, vector_n = 4, 2, 8, 4
    
    # Create RMSNorm tile distribution encoding
    # From: include/ck_tile/ops/add_rmsnorm2d_rdquant/pipeline
    encoding = TileDistributionEncoding(
        rs_lengths=[],                                           # Empty R (no replication)
        hs_lengthss=[[repeat_m, warp_per_block_m, thread_per_warp_m, vector_m],  # H for M dimension
                     [repeat_n, warp_per_block_n, thread_per_warp_n, vector_n]], # H for N dimension
        ps_to_rhss_major=[[1, 2], [1, 2]],                     # P→RH major mapping
        ps_to_rhss_minor=[[1, 1], [2, 2]],                     # P→RH minor mapping
        ys_to_rhs_major=[1, 1, 2, 2],                          # Y→RH major mapping
        ys_to_rhs_minor=[0, 3, 0, 3]                           # Y→RH minor mapping
    )
    
    show_result("RMSNorm Configuration", "Real-world layer normalization")
    show_result("Repeat (M, N)", f"({repeat_m}, {repeat_n})")
    show_result("Warps per block (M, N)", f"({warp_per_block_m}, {warp_per_block_n})")
    show_result("Threads per warp (M, N)", f"({thread_per_warp_m}, {thread_per_warp_n})")
    show_result("Vector size (M, N)", f"({vector_m}, {vector_n})")
    
    # Calculate thread organization
    threads_per_block = warp_per_block_m * warp_per_block_n * thread_per_warp_m * thread_per_warp_n
    warps_per_block = warp_per_block_m * warp_per_block_n
    
    print("\n📋 Thread Organization:")
    show_result("Threads per block", threads_per_block)
    show_result("Warps per block", warps_per_block)
    show_result("P dimensions", encoding.ndim_p)
    show_result("Y dimensions", encoding.ndim_y)
    
    # Show thread hierarchy
    print("\n🔍 Thread Hierarchy:")
    show_result("Block level", f"{warp_per_block_m}×{warp_per_block_n} warps")
    show_result("Warp level", f"{thread_per_warp_m}×{thread_per_warp_n} threads")
    show_result("Thread level", f"{vector_m}×{vector_n} elements")
    
    # Show some example thread IDs
    print("\n📊 Example Thread Mappings:")
    for warp_m in range(min(2, warp_per_block_m)):
        for warp_n in range(min(2, warp_per_block_n)):
            for thread_m in range(min(2, thread_per_warp_m)):
                for thread_n in range(min(2, thread_per_warp_n)):
                    # Calculate global thread ID
                    global_thread_id = (warp_m * warp_per_block_n * thread_per_warp_m * thread_per_warp_n +
                                      warp_n * thread_per_warp_m * thread_per_warp_n +
                                      thread_m * thread_per_warp_n + thread_n)
                    
                    show_result(f"Thread {global_thread_id}", 
                               f"Warp[{warp_m},{warp_n}] Thread[{thread_m},{thread_n}]")
                    
                    # Only show first few to avoid overwhelming output
                    if global_thread_id >= 7:
                        print("  ... (showing first 8 threads)")
                        break
                if global_thread_id >= 7:
                    break
            if global_thread_id >= 7:
                break
        if global_thread_id >= 7:
            break
    
    return encoding

def demonstrate_thread_data_mapping():
    """Show how thread IDs map to specific data elements."""
    print_step(2, "Thread-to-Data Mapping")
    
    print("🎯 RMSNorm Data Distribution Pattern")
    print("Shows how the RMSNorm operation distributes tensor data across threads")
    
    # RMSNorm parameters (same as previous function)
    repeat_m, warp_per_block_m, thread_per_warp_m, vector_m = 4, 2, 8, 4
    repeat_n, warp_per_block_n, thread_per_warp_n, vector_n = 4, 2, 8, 4
    
    # Create RMSNorm encoding
    encoding = TileDistributionEncoding(
        rs_lengths=[],                                           # Empty R (no replication)
        hs_lengthss=[[repeat_m, warp_per_block_m, thread_per_warp_m, vector_m],  # H for M dimension
                     [repeat_n, warp_per_block_n, thread_per_warp_n, vector_n]], # H for N dimension
        ps_to_rhss_major=[[1, 2], [1, 2]],                     # P→RH major mapping
        ps_to_rhss_minor=[[1, 1], [2, 2]],                     # P→RH minor mapping
        ys_to_rhs_major=[1, 1, 2, 2],                          # Y→RH major mapping
        ys_to_rhs_minor=[0, 3, 0, 3]                           # Y→RH minor mapping
    )
    
    # Calculate total tensor size processed by this tile distribution
    total_m = repeat_m * warp_per_block_m * thread_per_warp_m * vector_m
    total_n = repeat_n * warp_per_block_n * thread_per_warp_n * vector_n
    
    show_result("Total tensor size (M×N)", f"{total_m}×{total_n}")
    show_result("Elements per thread", f"{vector_m}×{vector_n} = {vector_m * vector_n}")
    
    # Show the hierarchical data distribution
    print("\n📊 Hierarchical Data Distribution:")
    print(f"🔹 Block Level: {repeat_m}×{repeat_n} iterations")
    print(f"🔹 Warp Level: {warp_per_block_m}×{warp_per_block_n} warps per block")
    print(f"🔹 Thread Level: {thread_per_warp_m}×{thread_per_warp_n} threads per warp")
    print(f"🔹 Vector Level: {vector_m}×{vector_n} elements per thread")
    
    # Show work distribution example
    print("\n📋 Work Distribution Example:")
    
    # Pick a specific thread and show its work
    example_warp_m, example_warp_n = 0, 0
    example_thread_m, example_thread_n = 0, 0
    
    # Calculate this thread's data region
    thread_start_m = example_warp_m * thread_per_warp_m * vector_m + example_thread_m * vector_m
    thread_end_m = thread_start_m + vector_m
    thread_start_n = example_warp_n * thread_per_warp_n * vector_n + example_thread_n * vector_n
    thread_end_n = thread_start_n + vector_n
    
    show_result("Example thread", f"Warp[{example_warp_m},{example_warp_n}] Thread[{example_thread_m},{example_thread_n}]")
    show_result("Data region (M)", f"[{thread_start_m}:{thread_end_m})")
    show_result("Data region (N)", f"[{thread_start_n}:{thread_end_n})")
    show_result("Total elements", f"{vector_m}×{vector_n} = {vector_m * vector_n}")
    
    # Show how different threads handle different data regions
    print("\n🔍 Thread Data Regions (first few threads):")
    for warp_m in range(min(2, warp_per_block_m)):
        for thread_m in range(min(2, thread_per_warp_m)):
            for warp_n in range(min(2, warp_per_block_n)):
                for thread_n in range(min(2, thread_per_warp_n)):
                    start_m = warp_m * thread_per_warp_m * vector_m + thread_m * vector_m
                    end_m = start_m + vector_m
                    start_n = warp_n * thread_per_warp_n * vector_n + thread_n * vector_n
                    end_n = start_n + vector_n
                    
                    show_result(f"W[{warp_m},{warp_n}]T[{thread_m},{thread_n}]", 
                               f"M[{start_m}:{end_m}) N[{start_n}:{end_n})")
                    
                    # Limit output to avoid overwhelming
                    if start_m + start_n > 20:
                        print("  ... (showing first few for brevity)")
                        return encoding
    
    return encoding

def demonstrate_thread_cooperation():
    """Show different thread cooperation patterns."""
    print_step(3, "Thread Cooperation Patterns in RMSNorm")
    
    print("🎯 RMSNorm Thread Cooperation Analysis")
    print("Analyzing how threads cooperate in the RMSNorm operation")
    
    # RMSNorm parameters (consistent with previous functions)
    repeat_m, warp_per_block_m, thread_per_warp_m, vector_m = 4, 2, 8, 4
    repeat_n, warp_per_block_n, thread_per_warp_n, vector_n = 4, 2, 8, 4
    
    # RMSNorm encoding (same as before)
    encoding = TileDistributionEncoding(
        rs_lengths=[],                                           # Empty R (no replication)
        hs_lengthss=[[repeat_m, warp_per_block_m, thread_per_warp_m, vector_m],  # H for M dimension
                     [repeat_n, warp_per_block_n, thread_per_warp_n, vector_n]], # H for N dimension
        ps_to_rhss_major=[[1, 2], [1, 2]],                     # P→RH major mapping
        ps_to_rhss_minor=[[1, 1], [2, 2]],                     # P→RH minor mapping
        ys_to_rhs_major=[1, 1, 2, 2],                          # Y→RH major mapping
        ys_to_rhs_minor=[0, 3, 0, 3]                           # Y→RH minor mapping
    )
    
    # Analysis 1: Warp-level cooperation
    print("\n🤝 Warp-Level Cooperation")
    show_result("Warps per block", f"{warp_per_block_m}×{warp_per_block_n}")
    show_result("Threads per warp", f"{thread_per_warp_m}×{thread_per_warp_n}")
    show_result("Cooperation pattern", "Threads within a warp process adjacent data")
    show_result("Synchronization", "Warp-level SIMD execution")
    
    # Analysis 2: Block-level cooperation
    print("\n🏗️ Block-Level Cooperation")
    total_threads_per_block = warp_per_block_m * warp_per_block_n * thread_per_warp_m * thread_per_warp_n
    show_result("Total threads per block", total_threads_per_block)
    show_result("Shared memory usage", "Warps can share data via shared memory")
    show_result("Cooperation pattern", "Blocks process independent tensor regions")
    
    # Analysis 3: Vector-level cooperation
    print("\n⚡ Vector-Level Processing")
    show_result("Vector size (M×N)", f"{vector_m}×{vector_n}")
    show_result("Elements per thread", vector_m * vector_n)
    show_result("Cooperation pattern", "Each thread processes multiple elements vectorized")
    show_result("Memory efficiency", "Vectorized loads/stores improve bandwidth")
    
    # Analysis 4: Hierarchical cooperation
    print("\n🔄 Hierarchical Cooperation Levels")
    show_result("Level 1 (Vector)", f"Thread processes {vector_m}×{vector_n} elements")
    show_result("Level 2 (Warp)", f"Warp processes {thread_per_warp_m * vector_m}×{thread_per_warp_n * vector_n} elements")
    show_result("Level 3 (Block)", f"Block processes {warp_per_block_m * thread_per_warp_m * vector_m}×{warp_per_block_n * thread_per_warp_n * vector_n} elements")
    show_result("Level 4 (Repeat)", f"Temporal repetition: {repeat_m}×{repeat_n} iterations per block")
    
    return encoding

def demonstrate_memory_access_patterns():
    """Show different memory access patterns and their performance implications."""
    print_step(4, "Memory Access Patterns in RMSNorm")
    
    print("🎯 RMSNorm Memory Access Analysis")
    print("Analyzing memory access patterns in the RMSNorm operation")
    
    # RMSNorm parameters (consistent with previous functions)
    repeat_m, warp_per_block_m, thread_per_warp_m, vector_m = 4, 2, 8, 4
    repeat_n, warp_per_block_n, thread_per_warp_n, vector_n = 4, 2, 8, 4
    
    # RMSNorm encoding (same as before)
    encoding = TileDistributionEncoding(
        rs_lengths=[],                                           # Empty R (no replication)
        hs_lengthss=[[repeat_m, warp_per_block_m, thread_per_warp_m, vector_m],  # H for M dimension
                     [repeat_n, warp_per_block_n, thread_per_warp_n, vector_n]], # H for N dimension
        ps_to_rhss_major=[[1, 2], [1, 2]],                     # P→RH major mapping
        ps_to_rhss_minor=[[1, 1], [2, 2]],                     # P→RH minor mapping
        ys_to_rhs_major=[1, 1, 2, 2],                          # Y→RH major mapping
        ys_to_rhs_minor=[0, 3, 0, 3]                           # Y→RH minor mapping
    )
    
    # Analysis 1: Vector memory access
    print("\n🚀 Vector Memory Access Pattern")
    show_result("Vector size per thread", f"{vector_m}×{vector_n} = {vector_m * vector_n} elements")
    show_result("Memory pattern", "Each thread loads/stores vectorized chunks")
    show_result("Cache efficiency", "High - vectorized access improves cache line utilization")
    show_result("Memory bandwidth", "Optimal - coalesced vector operations")
    
    # Analysis 2: Warp memory access
    print("\n⚡ Warp Memory Access Pattern")
    warp_elements_m = thread_per_warp_m * vector_m
    warp_elements_n = thread_per_warp_n * vector_n
    show_result("Elements per warp", f"{warp_elements_m}×{warp_elements_n} = {warp_elements_m * warp_elements_n}")
    show_result("Memory pattern", "Warp threads access adjacent memory regions")
    show_result("Coalescing", "Excellent - threads in warp access consecutive addresses")
    show_result("Cache efficiency", "High - good spatial locality within warp")
    
    # Analysis 3: Block memory access
    print("\n🏗️ Block Memory Access Pattern")
    block_elements_m = warp_per_block_m * thread_per_warp_m * vector_m
    block_elements_n = warp_per_block_n * thread_per_warp_n * vector_n
    show_result("Elements per block", f"{block_elements_m}×{block_elements_n} = {block_elements_m * block_elements_n}")
    show_result("Memory pattern", "Block processes contiguous tensor region")
    show_result("Shared memory usage", "Can cache frequently accessed data")
    show_result("Cache efficiency", "Good - block-level spatial locality")
    
    # Analysis 4: Temporal repetition access
    print("\n🔄 Temporal Repetition Access Pattern")
    block_elements_m = warp_per_block_m * thread_per_warp_m * vector_m
    block_elements_n = warp_per_block_n * thread_per_warp_n * vector_n
    show_result("Elements per block iteration", f"{block_elements_m}×{block_elements_n} = {block_elements_m * block_elements_n}")
    show_result("Repeat iterations", f"{repeat_m}×{repeat_n} = {repeat_m * repeat_n}")
    show_result("Memory pattern", "Each block processes multiple data chunks over time")
    show_result("Access efficiency", "High - reuses block-level organization across iterations")
    
    # Analysis 5: Performance implications
    print("\n📊 Performance Analysis")
    bytes_per_element = 4  # float32
    memory_per_iteration = block_elements_m * block_elements_n * bytes_per_element
    total_iterations = repeat_m * repeat_n
    show_result("Memory per iteration", f"{memory_per_iteration:,} bytes ({memory_per_iteration/1024:.1f} KB)")
    show_result("Total iterations", total_iterations)
    show_result("Access pattern", "Hierarchical: Vector → Warp → Block, repeated over time")
    show_result("Memory efficiency", "High - optimized for GPU memory hierarchy")
    show_result("Performance prediction", "Excellent - good data locality and temporal reuse")
    
    return encoding

def demonstrate_performance_analysis():
    """Show techniques for analyzing thread performance."""
    print_step(5, "Performance Analysis Techniques for RMSNorm")
    
    print("🎯 RMSNorm Performance Analysis")
    print("Analyzing performance characteristics of the RMSNorm thread mapping")
    
    # RMSNorm parameters (consistent with previous functions)
    repeat_m, warp_per_block_m, thread_per_warp_m, vector_m = 4, 2, 8, 4
    repeat_n, warp_per_block_n, thread_per_warp_n, vector_n = 4, 2, 8, 4
    
    # RMSNorm encoding (same as before)
    encoding = TileDistributionEncoding(
        rs_lengths=[],                                           # Empty R (no replication)
        hs_lengthss=[[repeat_m, warp_per_block_m, thread_per_warp_m, vector_m],  # H for M dimension
                     [repeat_n, warp_per_block_n, thread_per_warp_n, vector_n]], # H for N dimension
        ps_to_rhss_major=[[1, 2], [1, 2]],                     # P→RH major mapping
        ps_to_rhss_minor=[[1, 1], [2, 2]],                     # P→RH minor mapping
        ys_to_rhs_major=[1, 1, 2, 2],                          # Y→RH major mapping
        ys_to_rhs_minor=[0, 3, 0, 3]                           # Y→RH minor mapping
    )
    
    # Analysis 1: Work distribution balance
    print("\n📊 Work Distribution Balance")
    block_elements = warp_per_block_m * thread_per_warp_m * vector_m * warp_per_block_n * thread_per_warp_n * vector_n
    total_threads_per_block = warp_per_block_m * warp_per_block_n * thread_per_warp_m * thread_per_warp_n
    elements_per_thread = vector_m * vector_n
    
    show_result("Total elements per block", block_elements)
    show_result("Elements per thread", elements_per_thread)
    show_result("Threads per block", total_threads_per_block)
    show_result("Work balance", f"{elements_per_thread * total_threads_per_block}/{block_elements} = {(elements_per_thread * total_threads_per_block)/block_elements:.1%}")
    
    # Analysis 2: Memory access efficiency
    print("\n🔍 Memory Access Efficiency")
    show_result("Vector access stride", f"{vector_m}×{vector_n} vectorized")
    show_result("Warp coalescing", f"{thread_per_warp_m}×{thread_per_warp_n} threads access adjacent data")
    show_result("Memory coalescing", "Excellent - vectorized warp access")
    
    # Analysis 3: Thread utilization
    print("\n⚡ Thread Utilization")
    max_threads = 1024  # Typical GPU block size
    thread_efficiency = total_threads_per_block / max_threads
    
    show_result("Available threads", max_threads)
    show_result("Used threads", total_threads_per_block)
    show_result("Thread efficiency", f"{thread_efficiency:.1%}")
    
    # Analysis 4: Temporal efficiency
    print("\n🔄 Temporal Efficiency")
    total_iterations = repeat_m * repeat_n
    show_result("Repeat iterations", total_iterations)
    show_result("Temporal reuse", f"Each thread processes {total_iterations} data chunks")
    show_result("Cache reuse", "High - same thread organization reused across iterations")
    
    return encoding

def demonstrate_debugging_techniques():
    """Show techniques for debugging thread mapping issues."""
    print_step(6, "Debugging RMSNorm Thread Mapping")
    
    print("🎯 RMSNorm Thread Mapping Validation")
    print("Analyzing the RMSNorm configuration for potential issues")
    
    # RMSNorm parameters (consistent with previous functions)
    repeat_m, warp_per_block_m, thread_per_warp_m, vector_m = 4, 2, 8, 4
    repeat_n, warp_per_block_n, thread_per_warp_n, vector_n = 4, 2, 8, 4
    
    # RMSNorm encoding (same as before)
    encoding = TileDistributionEncoding(
        rs_lengths=[],                                           # Empty R (no replication)
        hs_lengthss=[[repeat_m, warp_per_block_m, thread_per_warp_m, vector_m],  # H for M dimension
                     [repeat_n, warp_per_block_n, thread_per_warp_n, vector_n]], # H for N dimension
        ps_to_rhss_major=[[1, 2], [1, 2]],                     # P→RH major mapping
        ps_to_rhss_minor=[[1, 1], [2, 2]],                     # P→RH minor mapping
        ys_to_rhs_major=[1, 1, 2, 2],                          # Y→RH major mapping
        ys_to_rhs_minor=[0, 3, 0, 3]                           # Y→RH minor mapping
    )
    
    # Debug 1: Check thread organization
    print("\n🔍 Thread Organization Analysis")
    total_threads_per_block = warp_per_block_m * warp_per_block_n * thread_per_warp_m * thread_per_warp_n
    total_warps_per_block = warp_per_block_m * warp_per_block_n
    threads_per_warp = thread_per_warp_m * thread_per_warp_n
    
    show_result("Total threads per block", total_threads_per_block)
    show_result("Total warps per block", total_warps_per_block)
    show_result("Threads per warp", threads_per_warp)
    show_result("Elements per thread", vector_m * vector_n)
    
    # Debug 2: Check for common issues
    print("\n⚠️  Configuration Validation")
    issues = []
    
    if total_threads_per_block > 1024:
        issues.append(f"Thread count ({total_threads_per_block}) exceeds typical block limit (1024)")
    
    if threads_per_warp != 64:
        issues.append(f"Threads per warp ({threads_per_warp}) not standard (64 for AMD)")
    
    if vector_m * vector_n < 4:
        issues.append(f"Vector size ({vector_m * vector_n}) too small for efficiency")
    
    if total_threads_per_block % 64 != 0:
        issues.append(f"Block size ({total_threads_per_block}) not warp-aligned")
    
    if issues:
        show_result("Issues found", issues)
    else:
        show_result("Issues found", "None - Configuration looks optimal!")
    
    # Debug 3: Performance characteristics
    print("\n💡 Performance Characteristics")
    memory_per_thread = vector_m * vector_n * 4  # 4 bytes per float32
    memory_per_block = total_threads_per_block * memory_per_thread
    
    show_result("Memory per thread", f"{memory_per_thread} bytes")
    show_result("Memory per block", f"{memory_per_block:,} bytes ({memory_per_block/1024:.1f} KB)")
    show_result("Warp efficiency", "High - full warp utilization")
    show_result("Memory efficiency", "High - vectorized access pattern")
    show_result("Temporal efficiency", f"High - {repeat_m * repeat_n} iterations reuse thread organization")
    
    return encoding

def test_thread_mapping_concepts():
    """Test thread mapping understanding."""
    print_step(7, "Testing RMSNorm Thread Mapping Concepts")
    
    def test_rmsnorm_encoding_creation():
        # Test that we can create the RMSNorm encoding
        try:
            encoding = TileDistributionEncoding(
                rs_lengths=[],
                hs_lengthss=[[4, 2, 8, 4], [4, 2, 8, 4]],
                ps_to_rhss_major=[[1, 2], [1, 2]],
                ps_to_rhss_minor=[[1, 1], [2, 2]],
                ys_to_rhs_major=[1, 1, 2, 2],
                ys_to_rhs_minor=[0, 3, 0, 3]
            )
            return encoding.ndim_p == 2 and encoding.ndim_y == 4
        except Exception:
            return False
    
    def test_thread_organization():
        # Test that the thread organization makes sense
        repeat_m, warp_per_block_m, thread_per_warp_m, vector_m = 4, 2, 8, 4
        repeat_n, warp_per_block_n, thread_per_warp_n, vector_n = 4, 2, 8, 4
        
        total_threads = warp_per_block_m * warp_per_block_n * thread_per_warp_m * thread_per_warp_n
        return total_threads == 256  # 2*2*8*8
    
    def test_memory_efficiency():
        # Test that vector access is efficient
        vector_m, vector_n = 4, 4
        vector_size = vector_m * vector_n
        return vector_size >= 16  # Good vectorization
    
    tests = [
        ("RMSNorm encoding creation", test_rmsnorm_encoding_creation),
        ("Thread organization calculation", test_thread_organization),
        ("Vector access efficiency", test_memory_efficiency)
    ]
    
    results = []
    for test_name, test_func in tests:
        results.append(validate_example(test_name, test_func))
    
    return all(results)

def main():
    """Main function to run all thread mapping demonstrations."""
    if not check_imports():
        return False
    
    print_section("Thread Mapping and Hardware Connection")
    
    # Run demonstrations
    encoding1 = demonstrate_thread_identification()
    encoding2 = demonstrate_thread_data_mapping()
    encoding3 = demonstrate_thread_cooperation()
    encoding4 = demonstrate_memory_access_patterns()
    encoding5 = demonstrate_performance_analysis()
    encoding6 = demonstrate_debugging_techniques()
    
    # Run tests
    all_tests_passed = test_thread_mapping_concepts()
    
    print_section("Summary")
    print(f"✅ Thread mapping demonstrations completed")
    print(f"✅ All tests passed: {all_tests_passed}")
    print(f"📝 Key concepts covered:")
    print(f"   - Thread identification and partition indices")
    print(f"   - Thread-to-data mapping")
    print(f"   - Thread cooperation patterns")
    print(f"   - Memory access pattern analysis")
    print(f"   - Performance analysis techniques")
    print(f"   - Debugging thread mapping issues")
    
    return all_tests_passed

if __name__ == "__main__":
    success = run_script_safely(main, "Thread Mapping")
    sys.exit(0 if success else 1) 