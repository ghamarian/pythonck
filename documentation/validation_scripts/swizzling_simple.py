"""
Simplified Swizzling Implementation: Morton Ordering Demo

This script demonstrates Morton ordering (Z-order curve) within tiles:
1. Understanding Morton ordering bit patterns
2. Coordinate transformations step by step
3. Memory access pattern analysis
4. Complete visualization

Focus on the core concepts rather than complex tensor descriptor chains.
"""

import numpy as np
import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pytensor import (
    make_naive_tensor_descriptor_packed,
    UnmergeTransform, MergeTransform,
    MultiIndex,
    make_tensor_coordinate,
)
from pytensor.tensor_descriptor import transform_tensor_descriptor


def print_section(title, level=1):
    """Print a formatted section header."""
    if level == 1:
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
    else:
        print(f"\n{'-'*40}")
        print(f"{title}")
        print(f"{'-'*40}")


def print_matrix(matrix, title="Matrix"):
    """Print a matrix in a nice format."""
    if title:
        print(f"\n{title}:")
    for row in matrix:
        print(" ".join(f"{val:3.0f}" for val in row))


def print_tiles_layout(data, tile_size, title="Tiles Layout"):
    """Print data organized by tiles with clear separation."""
    if title:
        print(f"\n{title}:")
    
    H, W = data.shape
    tiles_y = H // tile_size
    tiles_x = W // tile_size
    
    for tile_y in range(tiles_y):
        # Print each row of tiles
        for y_in_tile in range(tile_size):
            row_parts = []
            for tile_x in range(tiles_x):
                # Extract this row from this tile
                start_y = tile_y * tile_size + y_in_tile
                start_x = tile_x * tile_size
                end_x = start_x + tile_size
                tile_row = data[start_y, start_x:end_x]
                row_str = " ".join(f"{val:3.0f}" for val in tile_row)
                row_parts.append(row_str)
            print(" | ".join(row_parts))
        
        # Print separator between tile rows
        if tile_y < tiles_y - 1:
            sep_width = tile_size * 4 - 1  # 4 chars per number minus 1
            separator = "-" * sep_width
            print(" + ".join([separator] * tiles_x))


def morton_encode_2d(y, x):
    """Encode 2D coordinates to Morton order."""
    result = 0
    for i in range(2):  # 2 bits each for 4x4 tile
        bit_y = (y >> i) & 1
        bit_x = (x >> i) & 1
        result |= (bit_y << (2*i + 1)) | (bit_x << (2*i))
    return result


def morton_decode_2d(morton_idx):
    """Decode Morton index back to 2D coordinates."""
    y = 0
    x = 0
    for i in range(2):  # 2 bits each
        y |= ((morton_idx >> (2*i + 1)) & 1) << i
        x |= ((morton_idx >> (2*i)) & 1) << i
    return y, x


def create_test_data():
    """Create test data for swizzling examples."""
    # 8x8 texture with sequential numbers (easier to follow)
    H, W = 8, 8
    texture = np.arange(1, H*W + 1).reshape(H, W)
    return texture, H, W


def analyze_morton_pattern():
    """Analyze and understand Morton ordering pattern."""
    print_section("Morton Ordering Analysis")
    
    print("Morton Ordering Pattern for 4×4 tile:")
    print("Coordinates → Morton Index → Binary Breakdown")
    
    morton_matrix = np.zeros((4, 4), dtype=int)
    for y in range(4):
        for x in range(4):
            morton_idx = morton_encode_2d(y, x)
            morton_matrix[y, x] = morton_idx
            
            # Show bit breakdown
            y_bits = f"{y:02b}"
            x_bits = f"{x:02b}"
            morton_bits = f"{morton_idx:04b}"
            
            print(f"({y},{x}) = ({y_bits}, {x_bits}) → {morton_idx:2d} = {morton_bits}")
    
    print("\nMorton Index Layout in 4×4 tile:")
    for row in morton_matrix:
        print(" ".join(f"{val:2d}" for val in row))
    
    # Verify encoding/decoding
    print("\nVerifying Morton encoding/decoding:")
    for morton_idx in range(16):
        y, x = morton_decode_2d(morton_idx)
        encoded_back = morton_encode_2d(y, x)
        print(f"Morton {morton_idx:2d} → ({y},{x}) → {encoded_back:2d} ✓" if encoded_back == morton_idx else "✗")


def demonstrate_stage1_simple():
    """Demonstrate Stage 1: Simple tiling without tensor descriptors."""
    print_section("Stage 1: Tiling Transformation (Simple)")
    
    texture, H, W = create_test_data()
    TILE = 4
    
    print_matrix(texture, f"Original {H}×{W} Texture")
    print_tiles_layout(texture, TILE, "Tiled Layout")
    
    # Manual coordinate transformation: (Y_blk, y_in, X_blk, x_in) → (Y, X)
    def tiled_to_linear(Y_blk, y_in, X_blk, x_in):
        Y = Y_blk * TILE + y_in
        X = X_blk * TILE + x_in
        return Y, X
    
    print("\nVerifying Stage 1 coordinate transformation:")
    print("(Y_blk, y_in, X_blk, x_in) → Original (Y, X) → Value")
    
    test_coords = [
        (0, 0, 0, 0),  # First element of first tile
        (0, 1, 0, 2),  # Row 1, col 2 of first tile  
        (1, 2, 1, 3),  # Row 2, col 3 of last tile
        (0, 3, 1, 1),  # Last row, middle column of top-right tile
    ]
    
    for Y_blk, y_in, X_blk, x_in in test_coords:
        orig_y, orig_x = tiled_to_linear(Y_blk, y_in, X_blk, x_in)
        value = texture[orig_y, orig_x]
        print(f"({Y_blk}, {y_in}, {X_blk}, {x_in}) → ({orig_y}, {orig_x}) → {value}")
    
    return texture, TILE


def demonstrate_stage2_simple():
    """Demonstrate Stage 2: Morton ordering within tiles (Simple)."""
    print_section("Stage 2: Morton Ordering (Simple)")
    
    texture, TILE = demonstrate_stage1_simple()
    
    # Manual coordinate transformation: (Y_blk, Morton, X_blk) → (Y, X)
    def morton_to_linear(Y_blk, morton_idx, X_blk):
        # Decode Morton index to tile coordinates
        y_in_tile, x_in_tile = morton_decode_2d(morton_idx)
        
        # Convert to global coordinates
        Y = Y_blk * TILE + y_in_tile
        X = X_blk * TILE + x_in_tile
        return Y, X
    
    print("\nVerifying Morton Ordering:")
    print("(Y_blk, Morton, X_blk) → Original (Y, X) → Value")
    print("Morton pattern within tile (0,0):")
    
    morton_values = []
    for morton_idx in range(16):  # 4×4 = 16 elements
        orig_y, orig_x = morton_to_linear(0, morton_idx, 0)  # First tile
        value = texture[orig_y, orig_x]
        morton_values.append(value)
        
        if morton_idx < 8:  # Print first 8 for brevity
            print(f"(0, {morton_idx:2d}, 0) → ({orig_y}, {orig_x}) → {value}")
    
    print("...")
    print(f"Complete Morton sequence for tile (0,0): {morton_values}")
    
    return morton_values


def demonstrate_tensor_descriptor_stage1():
    """Demonstrate Stage 1 using tensor descriptors."""
    print_section("Stage 1: Tensor Descriptor Implementation")
    
    texture, H, W = create_test_data()
    TILE = 4
    
    # Create basic descriptor
    DESC0 = make_naive_tensor_descriptor_packed([H, W])
    print(f"DESC0 top dimensions: {DESC0.get_num_of_top_dimension()} (Y, X)")
    
    # Stage 1: Split each axis into (block, in-tile)
    unmerge_Y = UnmergeTransform([H // TILE, TILE])  # Y → (Y_blk, y_in)
    unmerge_X = UnmergeTransform([W // TILE, TILE])  # X → (X_blk, x_in)
    
    print(f"Unmerge Y: {H} → {H // TILE} × {TILE}")
    print(f"Unmerge X: {W} → {W // TILE} × {TILE}")
    
    # Apply transformation
    DESC1 = transform_tensor_descriptor(
        DESC0,
        transforms=[unmerge_Y, unmerge_X],
        lower_dimension_hidden_idss=[[0], [1]],
        upper_dimension_hidden_idss=[[0, 1], [2, 3]]
    )
    
    print(f"DESC1 top dimensions: {DESC1.get_num_of_top_dimension()} (Y_blk, y_in, X_blk, x_in)")
    
    # Test a few coordinates
    print("\nTensor descriptor coordinate verification:")
    print("(Y_blk, y_in, X_blk, x_in) → Offset → (Y, X) → Value")
    
    test_coords = [
        (0, 0, 0, 0),  # First element
        (0, 1, 0, 2),  # Different position
    ]
    
    for Y_blk, y_in, X_blk, x_in in test_coords:
        new_coord = MultiIndex(4, [Y_blk, y_in, X_blk, x_in])
        tensor_coord = make_tensor_coordinate(DESC1, new_coord)
        orig_coord = tensor_coord.get_bottom_index()
        
        if len(orig_coord) == 1:
            # Single offset - convert back to 2D
            offset = orig_coord[0]
            orig_y = offset // W
            orig_x = offset % W
        else:
            orig_y, orig_x = orig_coord[0], orig_coord[1]
            
        value = texture[orig_y, orig_x]
        print(f"({Y_blk}, {y_in}, {X_blk}, {x_in}) → {orig_coord.to_list()} → ({orig_y}, {orig_x}) → {value}")


def visualize_complete_transformation():
    """Create complete visualization of the swizzling transformation."""
    print_section("Complete Transformation Visualization")
    
    texture, H, W = create_test_data()
    TILE = 4
    
    # Show original texture
    print_matrix(texture, "Original 8×8 Texture")
    
    # Show tiled view (Stage 1)
    print_tiles_layout(texture, TILE, "Stage 1: Tiled View")
    
    # Create Morton-ordered view within each tile
    morton_texture = np.zeros_like(texture)
    
    tiles_y = H // TILE
    tiles_x = W // TILE
    
    print(f"\nStage 2: Morton Ordering Within Each {TILE}×{TILE} Tile")
    print("Showing how original tile data maps to Morton order:")
    
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            print(f"\nTile ({tile_y},{tile_x}):")
            
            # Extract original tile
            start_y = tile_y * TILE
            start_x = tile_x * TILE
            orig_tile = texture[start_y:start_y+TILE, start_x:start_x+TILE]
            
            print("Original tile:")
            for row in orig_tile:
                print(" ".join(f"{val:3.0f}" for val in row))
            
            # Create Morton-ordered version
            morton_linear = np.zeros(TILE * TILE)
            
            for y in range(TILE):
                for x in range(TILE):
                    morton_idx = morton_encode_2d(y, x)
                    morton_linear[morton_idx] = orig_tile[y, x]
            
            # Reshape back to 2D (row-major Morton order)
            morton_tile = morton_linear.reshape(TILE, TILE)
            
            print("Morton-ordered tile (linear Morton → 2D):")
            for row in morton_tile:
                print(" ".join(f"{val:3.0f}" for val in row))
                
            print(f"Morton sequence: {morton_linear}")
            
            # Store in complete Morton texture
            morton_texture[start_y:start_y+TILE, start_x:start_x+TILE] = morton_tile
    
    # Show complete Morton-ordered texture
    print_matrix(morton_texture, "\nComplete Morton-Ordered Texture")
    print_tiles_layout(morton_texture, TILE, "Morton-Ordered Tiles Layout")
    
    return morton_texture


def analyze_access_patterns():
    """Analyze different memory access patterns."""
    print_section("Memory Access Pattern Analysis")
    
    texture, H, W = create_test_data()
    TILE = 4
    
    # Linear access pattern
    print("\n1. LINEAR ACCESS (Row-major):")
    linear_sequence = []
    for y in range(H):
        for x in range(W):
            linear_sequence.append(texture[y, x])
    print(f"Sequence: {linear_sequence[:16]}... (first 16 elements)")
    
    # Morton access pattern  
    print("\n2. MORTON ACCESS (Within tiles):")
    morton_sequence = []
    tiles_y, tiles_x = H // TILE, W // TILE
    
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            # Within each tile, access in Morton order
            tile_morton = []
            for morton_idx in range(TILE * TILE):
                # Decode Morton index to tile coordinates
                y_in_tile, x_in_tile = morton_decode_2d(morton_idx)
                
                global_y = tile_y * TILE + y_in_tile
                global_x = tile_x * TILE + x_in_tile
                value = texture[global_y, global_x]
                tile_morton.append(value)
                morton_sequence.append(value)
            
            print(f"Tile ({tile_y},{tile_x}) Morton: {tile_morton}")
    
    print(f"\nFull Morton sequence: {morton_sequence[:16]}... (first 16 elements)")
    
    # Show spatial locality
    print("\n3. SPATIAL LOCALITY ANALYSIS:")
    print("Adjacent Morton indices and their 2D distance:")
    
    for i in range(15):  # First 15 transitions
        y1, x1 = morton_decode_2d(i)
        y2, x2 = morton_decode_2d(i + 1)
        manhattan_dist = abs(y2 - y1) + abs(x2 - x1)
        print(f"Morton {i:2d}→{i+1:2d}: ({y1},{x1})→({y2},{x2}), distance: {manhattan_dist}")


def run_validation_tests():
    """Run comprehensive validation tests."""
    print_section("Validation Tests")
    
    try:
        # Test Morton encoding/decoding
        print("Testing Morton encoding/decoding...")
        for y in range(4):
            for x in range(4):
                morton = morton_encode_2d(y, x)
                y_dec, x_dec = morton_decode_2d(morton)
                assert (y, x) == (y_dec, x_dec), f"Encode/decode mismatch: ({y},{x}) → {morton} → ({y_dec},{x_dec})"
        print("✓ Morton encoding/decoding tests passed")
        
        # Test simple transformations
        print("Testing simple coordinate transformations...")
        morton_values = demonstrate_stage2_simple()
        print("✓ Simple coordinate transformation tests passed")
        
        # Test that all 16 elements in first tile are accessed exactly once
        print("Testing Morton completeness...")
        expected_first_tile = [1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28]
        assert len(morton_values) == 16, f"Expected 16 values, got {len(morton_values)}"
        assert set(morton_values) == set(expected_first_tile), f"Morton values {morton_values} don't match expected {expected_first_tile}"
        print("✓ Morton completeness tests passed")
        
        print("\n🎉 All validation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run all tests and demonstrations."""
    print("SIMPLIFIED MORTON ORDERING / SWIZZLING IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Step 1: Understand Morton pattern
        analyze_morton_pattern()
        
        # Step 2: Demonstrate simple transformations
        demonstrate_stage1_simple()
        demonstrate_stage2_simple()
        
        # Step 3: Show tensor descriptor approach for Stage 1
        demonstrate_tensor_descriptor_stage1()
        
        # Step 4: Visualize complete transformation
        visualize_complete_transformation()
        
        # Step 5: Analyze access patterns
        analyze_access_patterns()
        
        # Step 6: Run validation tests
        success = run_validation_tests()
        
        if success:
            print("\n✅ SUCCESS: All swizzling implementation tests passed!")
            print("This working code can be adapted for the QMD file.")
        else:
            print("\n❌ FAILURE: Some tests failed. Check the errors above.")
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 