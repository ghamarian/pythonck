#!/usr/bin/env python3
"""
Purpose: Demonstrate tile distribution encoding internals.

This script shows how tile distribution encoding works internally - how the 
mathematical encoding creates the adaptors and descriptors that implement
the P+Y → X transformations we learned about in coordinate systems.

Key Concepts:
- TileDistributionEncoding: Mathematical specification
- Adaptor creation: How encoding builds transformation chains
- Descriptor creation: How Y coordinates map to linearized storage
- Internal components: ps_ys_to_xs_adaptor and ys_to_d_descriptor
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code_examples'))

from common_utils import *
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
import numpy as np

def demonstrate_encoding_basics():
    """Show the basic structure of tile distribution encoding."""
    print_step(1, "Tile Distribution Encoding Basics")
    
    print("🎯 What is Tile Distribution Encoding?")
    print("The encoding is a mathematical specification that defines:")
    print("• How threads are organized (P-space)")
    print("• How work is distributed (Y-space)")  
    print("• How these map to tensor coordinates (X-space)")
    print("• How data is shared across threads (R-space)")
    
    # Simple 2D example
    print(f"\n📋 Example: 2x2 thread grid, 2x2 tiles per thread")
    
    encoding = TileDistributionEncoding(
        rs_lengths=[],                    # No replication
        hs_lengthss=[[2, 2], [2, 2]],   # 2x2 hierarchical tiles
        ps_to_rhss_major=[[1], [2]],     # P0→H1, P1→H2  
        ps_to_rhss_minor=[[0], [0]],     # Use first component
        ys_to_rhs_major=[1, 1, 2, 2],    # Y mapping to H
        ys_to_rhs_minor=[0, 1, 0, 1]     # Y component selection
    )
    
    print("📐 Encoding Structure:")
    print(f"  rs_lengths: {encoding.rs_lengths} (replication)")
    print(f"  hs_lengthss: {encoding.hs_lengthss} (hierarchical tiles)")
    print(f"  ps_to_rhss_major: {encoding.ps_to_rhss_major} (P→RH major)")
    print(f"  ps_to_rhss_minor: {encoding.ps_to_rhss_minor} (P→RH minor)")
    print(f"  ys_to_rhs_major: {encoding.ys_to_rhs_major} (Y→RH major)")
    print(f"  ys_to_rhs_minor: {encoding.ys_to_rhs_minor} (Y→RH minor)")
    
    print("✅ Encoding basics: Mathematical specification created")
    
    return encoding

def demonstrate_encoding_to_distribution():
    """Show how encoding creates tile distribution."""
    print_step(2, "From Encoding to Tile Distribution")
    
    print("🎯 The Magic Transformation")
    print("make_static_tile_distribution() takes the mathematical encoding")
    print("and creates the runtime components that implement the transformations.")
    
    # Create encoding
    encoding = TileDistributionEncoding(
        rs_lengths=[],                    
        hs_lengthss=[[2, 2], [2, 2]],   
        ps_to_rhss_major=[[1], [2]],     
        ps_to_rhss_minor=[[0], [0]],     
        ys_to_rhs_major=[1, 1, 2, 2],    
        ys_to_rhs_minor=[0, 1, 0, 1]     
    )
    
    print(f"\n🔧 Creating Tile Distribution...")
    try:
        tile_distribution = make_static_tile_distribution(encoding)
        
        print("✅ Tile distribution created successfully!")
        print("📦 Internal Components:")
        print("  • ps_ys_to_xs_adaptor: Maps (P,Y) coordinates to X coordinates")
        print("  • ys_to_d_descriptor: Maps Y coordinates to linearized storage")
        print("  • encoding: Original mathematical specification")
        
        # Show the components exist
        print(f"\n🔍 Component Inspection:")
        print(f"  Has ps_ys_to_xs_adaptor: {hasattr(tile_distribution, 'ps_ys_to_xs_adaptor')}")
        print(f"  Has ys_to_d_descriptor: {hasattr(tile_distribution, 'ys_to_d_descriptor')}")
        print(f"  Has encoding: {hasattr(tile_distribution, 'encoding')}")
        
        return tile_distribution
        
    except Exception as e:
        print(f"⚠️ Failed to create tile distribution: {e}")
        print("Note: This demonstrates the concept even if creation fails")
        return None

def demonstrate_adaptor_internals():
    """Show how the adaptor implements P+Y → X transformation."""
    print_step(3, "Adaptor Internals: P+Y → X Transformation")
    
    print("🎯 The P+Y → X Adaptor")
    print("The ps_ys_to_xs_adaptor is a chain of transformations that")
    print("converts combined (P,Y) coordinates to final X coordinates.")
    
    print(f"\n🔗 Transformation Chain Concept:")
    print("  1. Start with P coordinates (which thread)")
    print("  2. Add Y coordinates (which element in thread's tile)")
    print("  3. Apply replication transforms (R-space)")
    print("  4. Apply hierarchical transforms (H-space)")
    print("  5. Merge into final X coordinates")
    
    print(f"\n💡 Why This Works:")
    print("  • Each transform handles one aspect of the mapping")
    print("  • Transforms are composable and efficient")
    print("  • The chain is built automatically from encoding")
    print("  • Same pattern works for any distribution strategy")
    
    # Show conceptual example
    print(f"\n📝 Conceptual Example:")
    print("  Input: P=[1,0] + Y=[0,1] → Combined=[1,0,0,1]")
    print("  Transform 1: Handle replication (none in this case)")
    print("  Transform 2: Handle hierarchical structure")
    print("  Transform 3: Merge to final coordinates")
    print("  Output: X=[0,3] (final tensor position)")
    
    print("✅ Adaptor internals: P+Y → X transformation chain")
    
    return True

def demonstrate_descriptor_internals():
    """Show how the descriptor implements Y → D transformation."""
    print_step(4, "Descriptor Internals: Y → D Transformation")
    
    print("🎯 The Y → D Descriptor")
    print("The ys_to_d_descriptor handles the linearization of")
    print("Y coordinates to memory addresses within a thread's tile.")
    
    print(f"\n🔄 Y → D Transformation:")
    print("  1. Start with Y coordinates [y0, y1, y2, y3]")
    print("  2. Apply thread's local layout (usually row-major)")
    print("  3. Compute linear offset within thread's buffer")
    print("  4. Result: D coordinate (memory address)")
    
    # Show example
    tile_shape = [2, 2]
    print(f"\n📝 Example with {tile_shape} tile:")
    
    for y0 in range(tile_shape[0]):
        for y1 in range(tile_shape[1]):
            y_coord = [y0, y1]
            # Row-major linearization within tile
            d_coord = y0 * tile_shape[1] + y1
            print(f"  Y={y_coord} → D={d_coord}")
    
    print(f"\n💡 Why Separate from Adaptor:")
    print("  • Adaptor handles inter-thread coordination (P+Y → X)")
    print("  • Descriptor handles intra-thread layout (Y → D)")
    print("  • This separation enables different memory layouts")
    print("  • Each thread can have its own descriptor")
    
    print("✅ Descriptor internals: Y → D linearization")
    
    return tile_shape

def demonstrate_encoding_parameters():
    """Show how different encoding parameters affect behavior."""
    print_step(5, "Encoding Parameters Deep Dive")
    
    print("🎯 Understanding Encoding Parameters")
    print("Each parameter in the encoding controls a specific aspect")
    print("of how threads and data are organized.")
    
    print(f"\n📋 Parameter Breakdown:")
    
    # rs_lengths
    print("🔹 rs_lengths (Replication Lengths):")
    print("  • Controls data sharing across threads")
    print("  • [] = no replication")
    print("  • [2] = each data element shared by 2 threads")
    print("  • [2,2] = 2x2 replication pattern")
    
    # hs_lengthss  
    print("🔹 hs_lengthss (Hierarchical Lengths):")
    print("  • Defines tile sizes per dimension")
    print("  • [[2,2], [2,2]] = 2x2 tiles for both X dimensions")
    print("  • [[4,4], [4,4]] = 4x4 tiles for both X dimensions")
    print("  • Controls thread workload size")
    
    # ps_to_rhss mappings
    print("🔹 ps_to_rhss_major/minor (P→RH Mappings):")
    print("  • Maps partition coordinates to RH space")
    print("  • [[1], [2]] = P0→H1, P1→H2")
    print("  • Controls which H dimensions each P dimension affects")
    
    # ys_to_rhs mappings
    print("🔹 ys_to_rhs_major/minor (Y→RH Mappings):")
    print("  • Maps logical tile coordinates to RH space")
    print("  • [1,1,2,2] = Y0,Y1→H1, Y2,Y3→H2")
    print("  • Controls how Y coordinates navigate the tile structure")
    
    print("✅ Encoding parameters: Complete specification understood")
    
    return True

def demonstrate_practical_examples():
    """Show practical encoding examples."""
    print_step(6, "Practical Encoding Examples")
    
    print("🎯 Real-World Encoding Patterns")
    print("Let's see how different use cases translate to encodings.")
    
    # Example 1: Simple thread distribution
    print(f"\n📝 Example 1: Simple 2x2 Thread Distribution")
    print("Use case: 4 threads, each handles 2x2 elements")
    
    simple_encoding = TileDistributionEncoding(
        rs_lengths=[],                    # No sharing
        hs_lengthss=[[2, 2], [2, 2]],   # 2x2 tiles
        ps_to_rhss_major=[[1], [2]],     # Standard mapping
        ps_to_rhss_minor=[[0], [0]],     
        ys_to_rhs_major=[1, 1, 2, 2],    
        ys_to_rhs_minor=[0, 1, 0, 1]     
    )
    
    print("  ✅ Result: Clean 2x2 thread grid, no overlap")
    
    # Example 2: With replication
    print(f"\n📝 Example 2: With Replication for Broadcasting")
    print("Use case: Share some data across multiple threads")
    
    replicated_encoding = TileDistributionEncoding(
        rs_lengths=[2],                   # 2-way replication
        hs_lengthss=[[2, 2], [2, 2]],   
        ps_to_rhss_major=[[0], [2]],     # P0 affects replication
        ps_to_rhss_minor=[[0], [0]],     
        ys_to_rhs_major=[0, 0, 2, 2],    # Y maps to replication too
        ys_to_rhs_minor=[0, 1, 0, 1]     
    )
    
    print("  ✅ Result: Some data shared between thread pairs")
    
    # Example 3: Different tile sizes
    print(f"\n📝 Example 3: Larger Tiles for Better Performance")
    print("Use case: Each thread handles more work (4x4 tiles)")
    
    large_tile_encoding = TileDistributionEncoding(
        rs_lengths=[],                    
        hs_lengthss=[[4, 4], [4, 4]],   # Larger tiles
        ps_to_rhss_major=[[1], [2]],     
        ps_to_rhss_minor=[[0], [0]],     
        ys_to_rhs_major=[1, 1, 2, 2],    
        ys_to_rhs_minor=[0, 1, 0, 1]     
    )
    
    print("  ✅ Result: Better cache utilization, fewer threads needed")
    
    print("✅ Practical examples: Common encoding patterns demonstrated")
    
    return [simple_encoding, replicated_encoding, large_tile_encoding]

def test_encoding_internals():
    """Test encoding internal operations."""
    print_step(7, "Testing Encoding Internals")
    
    def test_encoding_creation():
        """Test that encodings can be created."""
        try:
            encoding = TileDistributionEncoding(
                rs_lengths=[],
                hs_lengthss=[[2, 2], [2, 2]],
                ps_to_rhss_major=[[1], [2]],
                ps_to_rhss_minor=[[0], [0]],
                ys_to_rhs_major=[1, 1, 2, 2],
                ys_to_rhs_minor=[0, 1, 0, 1]
            )
            return True
        except Exception:
            return False
    
    def test_encoding_has_required_fields():
        """Test that encoding has all required fields."""
        encoding = TileDistributionEncoding(
            rs_lengths=[],
            hs_lengthss=[[2, 2], [2, 2]],
            ps_to_rhss_major=[[1], [2]],
            ps_to_rhss_minor=[[0], [0]],
            ys_to_rhs_major=[1, 1, 2, 2],
            ys_to_rhs_minor=[0, 1, 0, 1]
        )
        
        required_fields = [
            'rs_lengths', 'hs_lengthss', 'ps_to_rhss_major',
            'ps_to_rhss_minor', 'ys_to_rhs_major', 'ys_to_rhs_minor'
        ]
        
        return all(hasattr(encoding, field) for field in required_fields)
    
    def test_tile_distribution_creation():
        """Test that tile distributions can be created from encodings."""
        try:
            encoding = TileDistributionEncoding(
                rs_lengths=[],
                hs_lengthss=[[2, 2], [2, 2]],
                ps_to_rhss_major=[[1], [2]],
                ps_to_rhss_minor=[[0], [0]],
                ys_to_rhs_major=[1, 1, 2, 2],
                ys_to_rhs_minor=[0, 1, 0, 1]
            )
            tile_distribution = make_static_tile_distribution(encoding)
            return True
        except Exception:
            return False
    
    def test_different_encodings():
        """Test that different encoding parameters work."""
        try:
            # Test with replication
            encoding1 = TileDistributionEncoding(
                rs_lengths=[2],
                hs_lengthss=[[2, 2], [2, 2]],
                ps_to_rhss_major=[[0], [2]],
                ps_to_rhss_minor=[[0], [0]],
                ys_to_rhs_major=[0, 0, 2, 2],
                ys_to_rhs_minor=[0, 1, 0, 1]
            )
            
            # Test with larger tiles
            encoding2 = TileDistributionEncoding(
                rs_lengths=[],
                hs_lengthss=[[4, 4], [4, 4]],
                ps_to_rhss_major=[[1], [2]],
                ps_to_rhss_minor=[[0], [0]],
                ys_to_rhs_major=[1, 1, 2, 2],
                ys_to_rhs_minor=[0, 1, 0, 1]
            )
            
            return True
        except Exception:
            return False
    
    tests = [
        ("Encoding creation", test_encoding_creation),
        ("Required fields", test_encoding_has_required_fields),
        ("Tile distribution creation", test_tile_distribution_creation),
        ("Different encodings", test_different_encodings)
    ]
    
    results = []
    for test_name, test_func in tests:
        results.append(validate_example(test_name, test_func))
    
    return all(results)

def main():
    """Main function to run all encoding internals demonstrations."""
    if not check_imports():
        return False
    
    print_section("Tile Distribution Encoding Internals")
    
    # Run demonstrations
    encoding = demonstrate_encoding_basics()
    tile_distribution = demonstrate_encoding_to_distribution()
    adaptor_result = demonstrate_adaptor_internals()
    tile_shape = demonstrate_descriptor_internals()
    params_result = demonstrate_encoding_parameters()
    examples = demonstrate_practical_examples()
    
    # Run tests
    all_tests_passed = test_encoding_internals()
    
    print_section("Summary")
    print(f"✅ Encoding internals demonstrations completed")
    print(f"✅ All tests passed: {all_tests_passed}")
    
    print("\n🎓 Key Takeaways:")
    print("  • TileDistributionEncoding is a mathematical specification")
    print("  • make_static_tile_distribution() creates runtime components")
    print("  • ps_ys_to_xs_adaptor handles P+Y → X transformations")
    print("  • ys_to_d_descriptor handles Y → D linearization")
    print("  • Different encodings enable different distribution strategies")
    print("  • All coordinate transformations are built from this foundation")
    
    print("\n💡 The Internal Pipeline:")
    print("  Encoding → Adaptors/Descriptors → Coordinate Transformations")
    print("  This abstraction enables flexible, efficient parallel processing!")
    
    print("\n🚀 Next Steps:")
    print("  • Learn about static distributed tensors")
    print("  • See how threads use these components")
    print("  • Understand hardware thread mapping")
    
    return all_tests_passed

if __name__ == "__main__":
    success = run_script_safely(main, "Encoding Internals")
    sys.exit(0 if success else 1) 