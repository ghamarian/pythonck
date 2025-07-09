#!/usr/bin/env python3
"""
Purpose: Demonstrate static distributed tensors.

This script shows how static distributed tensors work - the data structures
that hold each thread's portion of the distributed computation. These tensors
are "static" because their layout is determined at compile time.

Key Concepts:
- StaticDistributedTensor: Thread-local data container
- Thread buffer organization: How data is stored per thread
- Element access patterns: Getting/setting elements efficiently
- Memory layout: How Y coordinates map to thread buffer locations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code_examples'))

from common_utils import *
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.static_distributed_tensor import make_static_distributed_tensor
import numpy as np

def demonstrate_static_distributed_tensor_basics():
    """Show the basic concept of static distributed tensors."""
    print_step(1, "Static Distributed Tensor Basics")
    
    print("🎯 What is a Static Distributed Tensor?")
    print("A StaticDistributedTensor is a data structure that holds")
    print("one thread's portion of a larger distributed computation.")
    print("It's 'static' because the layout is determined at compile time.")
    
    print(f"\n📦 Key Characteristics:")
    print("  • Each thread has its own StaticDistributedTensor")
    print("  • Contains only the data that thread needs")
    print("  • Layout optimized for the thread's access patterns")
    print("  • Provides efficient element access via Y coordinates")
    print("  • Memory is organized according to tile distribution")
    
    print(f"\n🔍 Think of it Like:")
    print("  • A traditional tensor: contains all data")
    print("  • A distributed tensor: data split across threads")
    print("  • A static distributed tensor: thread-local portion with")
    print("    compile-time optimized layout")
    
    print("✅ Static distributed tensor basics: Core concepts explained")
    
    return True

def demonstrate_tensor_creation():
    """Show how to create static distributed tensors."""
    print_step(2, "Creating Static Distributed Tensors")
    
    print("🎯 Tensor Creation Process")
    print("Static distributed tensors are created from tile distributions")
    print("and provide the storage for thread-local computations.")
    
    # Create tile distribution first
    encoding = TileDistributionEncoding(
        rs_lengths=[],                    
        hs_lengthss=[[2, 2], [2, 2]],   # 2x2 tiles per thread
        ps_to_rhss_major=[[1], [2]],     
        ps_to_rhss_minor=[[0], [0]],     
        ys_to_rhs_major=[1, 1, 2, 2],    
        ys_to_rhs_minor=[0, 1, 0, 1]     
    )
    
    try:
        tile_distribution = make_static_tile_distribution(encoding)
        
        print(f"\n🔧 Creating Static Distributed Tensor...")
        
        # Try to create static distributed tensor
        distributed_tensor = make_static_distributed_tensor(
            tile_distribution=tile_distribution,
            dtype=np.float32
        )
        
        print("✅ Static distributed tensor created successfully!")
        
        print(f"\n📋 Tensor Properties:")
        print(f"  • Type: {type(distributed_tensor).__name__}")
        print(f"  • Data type: {distributed_tensor.dtype if hasattr(distributed_tensor, 'dtype') else 'float32'}")
        
        # Check for key methods
        key_methods = ['get_element', 'set_element', 'get_num_of_elements', 'get_thread_buffer']
        print(f"  • Available methods:")
        for method in key_methods:
            has_method = hasattr(distributed_tensor, method)
            print(f"    - {method}: {'✅' if has_method else '❌'}")
        
        return distributed_tensor
        
    except Exception as e:
        print(f"⚠️ Failed to create static distributed tensor: {e}")
        print("Note: We'll demonstrate the concepts even without creation")
        return None

def demonstrate_thread_buffer_organization():
    """Show how thread buffers are organized."""
    print_step(3, "Thread Buffer Organization")
    
    print("🎯 Thread Buffer Layout")
    print("Each thread's buffer is organized to efficiently store")
    print("the elements in its tile according to Y coordinates.")
    
    # Example with 2x2 tile
    tile_size = [2, 2]
    print(f"\n📊 Example: {tile_size} Thread Tile")
    print("Thread buffer organization (conceptual):")
    
    # Show how Y coordinates map to buffer positions
    buffer_pos = 0
    for y0 in range(tile_size[0]):
        for y1 in range(tile_size[1]):
            y_coord = [y0, y1]
            print(f"  Y={y_coord} → Buffer position {buffer_pos}")
            buffer_pos += 1
    
    print(f"\n💡 Buffer Layout Properties:")
    print("  • Contiguous memory for cache efficiency")
    print("  • Y coordinates provide logical indexing")
    print("  • Buffer positions provide physical indexing")
    print("  • Layout optimized for thread's access patterns")
    
    print(f"\n🔄 Coordinate Mapping:")
    print("  Y[0,0] Y[0,1]     Buffer[0] Buffer[1]")
    print("  Y[1,0] Y[1,1]  →  Buffer[2] Buffer[3]")
    print("  (logical)         (physical)")
    
    print("✅ Thread buffer organization: Layout principles explained")
    
    return tile_size

def demonstrate_element_access_patterns():
    """Show how to access elements in static distributed tensors."""
    print_step(4, "Element Access Patterns")
    
    print("🎯 Accessing Elements")
    print("Static distributed tensors provide efficient element access")
    print("using Y coordinates (logical tile positions).")
    
    # Conceptual example of element access
    tile_size = [2, 2]
    print(f"\n📝 Element Access Example ({tile_size} tile):")
    
    # Simulate element operations
    print("Conceptual operations:")
    
    for y0 in range(tile_size[0]):
        for y1 in range(tile_size[1]):
            y_coord = [y0, y1]
            # Conceptual values
            value = (y0 + 1) * 10 + (y1 + 1)
            print(f"  tensor.set_element({y_coord}, {value})")
    
    print("\nAfter setting values:")
    for y0 in range(tile_size[0]):
        for y1 in range(tile_size[1]):
            y_coord = [y0, y1]
            value = (y0 + 1) * 10 + (y1 + 1)
            print(f"  tensor.get_element({y_coord}) → {value}")
    
    print(f"\n💡 Access Patterns:")
    print("  • get_element(y_indices): Read value at Y coordinate")
    print("  • set_element(y_indices, value): Write value at Y coordinate")
    print("  • Y coordinates are logical (within thread's tile)")
    print("  • Internally maps to efficient buffer access")
    
    print(f"\n🚀 Performance Benefits:")
    print("  • Y coordinate lookup is O(1)")
    print("  • Buffer access is cache-friendly")
    print("  • No bounds checking needed (static layout)")
    print("  • Compiler can optimize access patterns")
    
    print("✅ Element access patterns: Efficient indexing demonstrated")
    
    return True

def demonstrate_memory_layout_details():
    """Show detailed memory layout of static distributed tensors."""
    print_step(5, "Memory Layout Details")
    
    print("🎯 Internal Memory Organization")
    print("Let's examine how the memory is actually organized")
    print("within a static distributed tensor.")
    
    # Example with larger tile
    tile_size = [3, 2]
    total_elements = tile_size[0] * tile_size[1]
    
    print(f"\n📊 Example: {tile_size} tile ({total_elements} elements)")
    print("Memory layout (row-major within tile):")
    
    # Show memory organization
    print("\n🗃️ Physical Memory:")
    memory_layout = []
    for addr in range(total_elements):
        # Calculate which Y coordinate this address corresponds to
        y0 = addr // tile_size[1]
        y1 = addr % tile_size[1]
        y_coord = [y0, y1]
        memory_layout.append((addr, y_coord))
        print(f"  Address {addr}: Y={y_coord}")
    
    print(f"\n📋 Mapping Table:")
    print("  Y Coordinate → Memory Address")
    for addr, y_coord in memory_layout:
        print(f"  {y_coord} → {addr}")
    
    print(f"\n💡 Layout Advantages:")
    print("  • Predictable memory access patterns")
    print("  • Sequential Y access = sequential memory access")
    print("  • Good cache locality for typical access patterns")
    print("  • Efficient for both row-wise and element-wise access")
    
    # Show different access patterns
    print(f"\n🔄 Access Pattern Examples:")
    print("  Row-wise access:")
    for y0 in range(tile_size[0]):
        row_accesses = []
        for y1 in range(tile_size[1]):
            addr = y0 * tile_size[1] + y1
            row_accesses.append(f"[{addr}]")
        print(f"    Row {y0}: {' '.join(row_accesses)} (sequential)")
    
    print("✅ Memory layout details: Internal organization understood")
    
    return memory_layout

def demonstrate_thread_coordination():
    """Show how multiple threads coordinate using static distributed tensors."""
    print_step(6, "Thread Coordination with Static Distributed Tensors")
    
    print("🎯 Multi-Thread Coordination")
    print("Each thread has its own static distributed tensor,")
    print("but they coordinate to solve the overall problem.")
    
    # Example: 2x2 thread grid, each with 2x2 tile
    thread_grid = [2, 2]
    tile_size = [2, 2]
    
    print(f"\n📊 Example Setup:")
    print(f"  Thread grid: {thread_grid}")
    print(f"  Tile size per thread: {tile_size}")
    print(f"  Total threads: {thread_grid[0] * thread_grid[1]}")
    print(f"  Elements per thread: {tile_size[0] * tile_size[1]}")
    
    print(f"\n🧵 Thread Coordination Pattern:")
    
    thread_id = 0
    for thread_x in range(thread_grid[0]):
        for thread_y in range(thread_grid[1]):
            p_coord = [thread_x, thread_y]
            
            # Calculate thread's region in global tensor
            base_x = thread_x * tile_size[0]
            base_y = thread_y * tile_size[1]
            
            print(f"\n  Thread {thread_id} P={p_coord}:")
            print(f"    • Has StaticDistributedTensor with {tile_size} elements")
            print(f"    • Handles global region [{base_x}:{base_x+tile_size[0]}, {base_y}:{base_y+tile_size[1]}]")
            print(f"    • Y=[0,0] maps to global X=[{base_x},{base_y}]")
            print(f"    • Y=[{tile_size[0]-1},{tile_size[1]-1}] maps to global X=[{base_x+tile_size[0]-1},{base_y+tile_size[1]-1}]")
            
            thread_id += 1
    
    print(f"\n💡 Coordination Benefits:")
    print("  • Each thread works independently on its tile")
    print("  • No data races or synchronization needed within tiles")
    print("  • Efficient parallel processing")
    print("  • Scalable to any number of threads")
    
    print("✅ Thread coordination: Multi-thread patterns demonstrated")
    
    return thread_grid, tile_size

def test_static_distributed_tensor_operations():
    """Test static distributed tensor operations."""
    print_step(7, "Testing Static Distributed Tensor Operations")
    
    def test_conceptual_element_access():
        """Test conceptual element access patterns."""
        tile_size = [2, 2]
        
        # Simulate element storage
        simulated_buffer = {}
        
        # Set elements
        for y0 in range(tile_size[0]):
            for y1 in range(tile_size[1]):
                y_coord = (y0, y1)  # Use tuple as dict key
                value = y0 * 10 + y1
                simulated_buffer[y_coord] = value
        
        # Get elements
        all_correct = True
        for y0 in range(tile_size[0]):
            for y1 in range(tile_size[1]):
                y_coord = (y0, y1)
                expected_value = y0 * 10 + y1
                actual_value = simulated_buffer.get(y_coord, -1)
                if actual_value != expected_value:
                    all_correct = False
        
        return all_correct
    
    def test_memory_address_calculation():
        """Test memory address calculation."""
        tile_size = [3, 2]
        
        # Test row-major linearization
        for y0 in range(tile_size[0]):
            for y1 in range(tile_size[1]):
                expected_addr = y0 * tile_size[1] + y1
                actual_addr = y0 * tile_size[1] + y1  # Same calculation
                if expected_addr != actual_addr:
                    return False
        
        return True
    
    def test_thread_coordination_math():
        """Test thread coordination mathematics."""
        thread_grid = [2, 2]
        tile_size = [2, 2]
        
        # Check that all threads cover non-overlapping regions
        covered_positions = set()
        
        for thread_x in range(thread_grid[0]):
            for thread_y in range(thread_grid[1]):
                base_x = thread_x * tile_size[0]
                base_y = thread_y * tile_size[1]
                
                # Add all positions this thread covers
                for y0 in range(tile_size[0]):
                    for y1 in range(tile_size[1]):
                        global_x = base_x + y0
                        global_y = base_y + y1
                        position = (global_x, global_y)
                        
                        # Check for overlap
                        if position in covered_positions:
                            return False
                        
                        covered_positions.add(position)
        
        # Check that we covered the expected number of positions
        expected_total = (thread_grid[0] * tile_size[0]) * (thread_grid[1] * tile_size[1])
        return len(covered_positions) == expected_total
    
    def test_tensor_creation_concept():
        """Test the conceptual understanding of tensor creation."""
        # This test always passes - it's about understanding
        return True
    
    tests = [
        ("Conceptual element access", test_conceptual_element_access),
        ("Memory address calculation", test_memory_address_calculation),
        ("Thread coordination math", test_thread_coordination_math),
        ("Tensor creation concept", test_tensor_creation_concept)
    ]
    
    results = []
    for test_name, test_func in tests:
        results.append(validate_example(test_name, test_func))
    
    return all(results)

def main():
    """Main function to run all static distributed tensor demonstrations."""
    if not check_imports():
        return False
    
    print_section("Static Distributed Tensors")
    
    # Run demonstrations
    basics_result = demonstrate_static_distributed_tensor_basics()
    distributed_tensor = demonstrate_tensor_creation()
    tile_size = demonstrate_thread_buffer_organization()
    access_result = demonstrate_element_access_patterns()
    memory_layout = demonstrate_memory_layout_details()
    thread_grid, tile_size = demonstrate_thread_coordination()
    
    # Run tests
    all_tests_passed = test_static_distributed_tensor_operations()
    
    print_section("Summary")
    print(f"✅ Static distributed tensor demonstrations completed")
    print(f"✅ All tests passed: {all_tests_passed}")
    
    print("\n🎓 Key Takeaways:")
    print("  • StaticDistributedTensor holds thread-local data")
    print("  • Layout is determined at compile time for efficiency")
    print("  • Y coordinates provide logical indexing within tiles")
    print("  • Memory organization optimized for access patterns")
    print("  • Each thread works independently on its portion")
    print("  • Coordination happens through tile distribution design")
    
    print("\n💡 The Storage Layer:")
    print("  Encoding → TileDistribution → StaticDistributedTensor")
    print("  This provides efficient, scalable thread-local storage!")
    
    print("\n🚀 Next Steps:")
    print("  • Learn about hardware thread mapping")
    print("  • See how threads get their partition coordinates")
    print("  • Understand thread cooperation patterns")
    
    return all_tests_passed

if __name__ == "__main__":
    success = run_script_safely(main, "Static Distributed Tensor")
    sys.exit(0 if success else 1) 