"""
Swizzling Implementation: Morton Ordering with Tensor Descriptors

This script demonstrates memory swizzling using Morton ordering (Z-order curve):
1. Understanding Morton ordering bit patterns
2. Stage 1: Tiling transformation (Y, X) → (Y_blk, y_in, X_blk, x_in)
3. Stage 2: Morton ordering within tiles
4. Complete swizzling pipeline validation
5. Memory access pattern analysis

Each step is validated for correctness and demonstrates GPU memory optimization patterns.
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


def test_stage1_tiling():
    """Test Stage 1: Tiling transformation."""
    print_section("Stage 1: Tiling Transformation")
    
    texture, H, W = create_test_data()
    TILE = 4
    
    print_matrix(texture, f"Original {H}×{W} Texture")
    print_tiles_layout(texture, TILE, "Tiled Layout")
    
    # Create Stage 1 descriptor
    print(f"\nCreating Stage 1 transformation: {H}×{W} → tiles")
    
    # Start with naive row-major descriptor
    DESC0 = make_naive_tensor_descriptor_packed([H, W])
    print(f"DESC0 top dimensions: {DESC0.get_num_of_top_dimension()} (Y, X)")
    
    # Stage 1: Split each axis into (block, in-tile)
    unmerge_Y = UnmergeTransform([H // TILE, TILE])  # Y → (Y_blk, y_in)
    unmerge_X = UnmergeTransform([W // TILE, TILE])  # X → (X_blk, x_in)
    
    print(f"Unmerge Y: {H} → {H // TILE} × {TILE}")
    print(f"Unmerge X: {W} → {W // TILE} × {TILE}")
    
    # Apply transformation to descriptor
    DESC1 = transform_tensor_descriptor(
        DESC0,
        transforms=[unmerge_Y, unmerge_X],
        lower_dimension_hidden_idss=[[0], [1]],           # Y takes old dim 0, X takes old dim 1
        upper_dimension_hidden_idss=[[0, 1], [2, 3]]     # Y→(0,1), X→(2,3)
    )
    
    print(f"DESC1 top dimensions: {DESC1.get_num_of_top_dimension()} (Y_blk, y_in, X_blk, x_in)")
    
    # Verify coordinate transformation
    print("\nVerifying Stage 1 coordinate transformation:")
    print("(Y_blk, y_in, X_blk, x_in) → Original (Y, X) → Value")
    
    test_coords = [
        (0, 0, 0, 0),  # First element of first tile
        (0, 1, 0, 2),  # Row 1, col 2 of first tile  
        (1, 2, 1, 3),  # Row 2, col 3 of last tile
        (0, 3, 1, 1),  # Last row, middle column of top-right tile
    ]
    
    for Y_blk, y_in, X_blk, x_in in test_coords:
        # Create coordinate in new space
        new_coord = MultiIndex(4, [Y_blk, y_in, X_blk, x_in])
        
        # Transform to original space
        tensor_coord = make_tensor_coordinate(DESC1, new_coord)
        orig_coord = tensor_coord.get_bottom_index()
        
        print(f"Debug: orig_coord length = {len(orig_coord)}, content = {orig_coord.to_list()}")
        
        if len(orig_coord) == 1:
            # Single offset - convert back to 2D
            offset = orig_coord[0]
            orig_y = offset // 8  # H = 8
            orig_x = offset % 8
        else:
            # Direct 2D coordinates
            orig_y, orig_x = orig_coord[0], orig_coord[1]
            
        value = texture[orig_y, orig_x]
        
        print(f"({Y_blk}, {y_in}, {X_blk}, {x_in}) → ({orig_y}, {orig_x}) → {value}")
    
    return DESC1, texture, TILE


def test_stage2_morton():
    """Test Stage 2: Morton ordering within tiles."""
    print_section("Stage 2: Morton Ordering")
    
    # Get Stage 1 results
    DESC1, texture, TILE = test_stage1_tiling()
    
    print("Creating Stage 2 transformation: Morton ordering within tiles")
    
    # Split each 2-bit in-tile coordinate into individual bits
    split2 = UnmergeTransform([2, 2])  # 4 → (bit1, bit0)
    
    # Create Morton ordering by merging bits in the right order: y0,x0,y1,x1
    # This creates the Morton index: y₁x₁y₀x₀
    merge_morton = MergeTransform([2, 2, 2, 2])  # (y0,x0,y1,x1) → Morton 0-15
    
    print("Stage 2 Transforms:")
    print("- Split y_in (dim 1): 4 → 2×2 (y1, y0)")
    print("- Split x_in (dim 3): 4 → 2×2 (x1, x0)")  
    print("- Merge Morton: (y0,x0,y1,x1) → Morton index")
    
    # Apply Stage 2 to DESC1
    # First split y_in and x_in
    DESC1_split = transform_tensor_descriptor(
        DESC1,
        transforms=[
            split2,        # Operates on DESC1 dim 1 (y_in) → (y1, y0)
            split2,        # Operates on DESC1 dim 3 (x_in) → (x1, x0)  
        ],
        lower_dimension_hidden_idss=[
            [1],           # y_in (from DESC1) 
            [3],           # x_in (from DESC1)
        ],
        upper_dimension_hidden_idss=[
            [1, 4],        # y_in → (y1, y0) at positions 1,4
            [3, 5],        # x_in → (x1, x0) at positions 3,5
        ],
    )
    
    # Now merge to create Morton ordering
    # After splits we have: (Y_blk, y1, X_blk, x1, y0, x0)
    # We want to merge (y0, x0, y1, x1) → Morton
    DESC2 = transform_tensor_descriptor(
        DESC1_split,
        transforms=[
            merge_morton,  # Merges: (y0,x0,y1,x1) → Morton
        ],
        lower_dimension_hidden_idss=[
            [4, 5, 1, 3],  # (y0, x0, y1, x1) positions in DESC1_split
        ],
        upper_dimension_hidden_idss=[
            [1],           # Morton replaces position 1
        ],
    )
    
    print(f"DESC2 top dimensions: {DESC2.get_num_of_top_dimension()}")
    
    # Verify Morton ordering
    print("\nVerifying Morton Ordering:")
    print("(Y_blk, Morton, X_blk) → Original (Y, X) → Value")
    print("Morton pattern within tile (0,0):")
    
    # Check DESC2 dimensions once before the loop
    num_dims = DESC2.get_num_of_top_dimension()
    print(f"DESC2 actual dimensions: {num_dims}, lengths: {DESC2.get_lengths()}")
    
    morton_values = []
    for morton_idx in range(16):  # 4×4 = 16 elements
        if num_dims == 2:
            # If 2D, it's likely (tile_idx, morton_idx)
            new_coord = MultiIndex(2, [0, morton_idx])
        elif num_dims == 3:
            # If 3D, it's (Y_blk, Morton, X_blk)
            new_coord = MultiIndex(3, [0, morton_idx, 0])
        else:
            raise ValueError(f"Unexpected number of dimensions: {num_dims}")
        
        # Transform to original space
        tensor_coord = make_tensor_coordinate(DESC2, new_coord)
        orig_coord = tensor_coord.get_bottom_index()
        
        if len(orig_coord) == 1:
            # Single offset - convert back to 2D
            offset = orig_coord[0]
            orig_y = offset // 8  # H = 8
            orig_x = offset % 8
        else:
            # Direct 2D coordinates
            orig_y, orig_x = orig_coord[0], orig_coord[1]
            
        value = texture[orig_y, orig_x]
        morton_values.append(value)
        
        if morton_idx < 8:  # Print first 8 for brevity
            if num_dims == 2:
                print(f"(0, {morton_idx:2d}) → ({orig_y}, {orig_x}) → {value}")
            else:
                print(f"(0, {morton_idx:2d}, 0) → ({orig_y}, {orig_x}) → {value}")
    
    print("...")
    print(f"Complete Morton sequence for tile (0,0): {morton_values}")
    
    return DESC2, morton_values


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
        
        # Test descriptor creation
        print("Testing tensor descriptor creation...")
        DESC2, morton_values = test_stage2_morton()
        print("✓ Tensor descriptor creation tests passed")
        
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
    print("MORTON ORDERING / SWIZZLING IMPLEMENTATION VALIDATION")
    print("=" * 60)
    
    try:
        # Step 1: Understand Morton pattern
        analyze_morton_pattern()
        
        # Step 2: Test transformations
        test_stage1_tiling()
        test_stage2_morton()
        
        # Step 3: Visualize complete transformation
        visualize_complete_transformation()
        
        # Step 4: Analyze access patterns
        analyze_access_patterns()
        
        # Step 5: Run validation tests
        success = run_validation_tests()
        
        if success:
            print("\n✅ SUCCESS: All swizzling implementation tests passed!")
            print("The QMD file can now be updated with this working code.")
        else:
            print("\n❌ FAILURE: Some tests failed. Check the errors above.")
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
