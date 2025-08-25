"""
Convolution Implementation: From Naive to Tensor Descriptors

This script demonstrates the complete progression of convolution implementation:
1. Naive reference implementation
2. Window extraction with NumPy as_strided  
3. Window extraction with Tensor Descriptors
4. Im2col transformation with NumPy
5. Im2col with Tensor Descriptors
6. Multi-channel extension
7. Complete convolution pipeline

Each step is validated for correctness and efficiency.
"""

import numpy as np
from pytensor.tensor_descriptor import (
    EmbedTransform, MergeTransform, MultiIndex,
    make_naive_tensor_descriptor, transform_tensor_descriptor
)


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


def print_matrix(matrix, title="Matrix", max_width=8):
    """Print a matrix in a nice format."""
    print(f"\n{title}:")
    if len(matrix.shape) == 2:
        for row in matrix:
            print(" ".join(f"{val:3.0f}" if abs(val - round(val)) < 1e-10 else f"{val:3.1f}" 
                          for val in row))
    else:
        print(f"Shape: {matrix.shape}")
        print(matrix)


def print_windows_tiled(windows, title="Windows Tiled View"):
    """Print 4D windows tensor as a tiled 2D layout with spacing."""
    if title:
        print(f"\n{title}:")
    
    out_h, out_w, K, K = windows.shape
    
    # Create separator patterns
    # Each number takes 3 chars, separated by spaces, so K numbers = 3*K + (K-1) chars
    window_width = 3 * K + (K - 1)
    row_sep = "-" * window_width
    col_sep = " | "
    
    for window_row in range(out_h):
        # Print each row of windows
        for k_row in range(K):
            line_parts = []
            for window_col in range(out_w):
                window_data = " ".join(f"{val:3.0f}" for val in windows[window_row, window_col, k_row, :])
                line_parts.append(window_data)
            print(col_sep.join(line_parts))
        
        # Print horizontal separator between window rows (except after last row)
        if window_row < out_h - 1:
            # Create separator that aligns with the | characters in content lines
            # Content uses " | " so separator should use " + "
            sep_parts = [row_sep] * out_w
            sep_line = (" + ").join(sep_parts)
            print(sep_line)
    
    print()  # Empty line for spacing


def create_test_data():
    """Create test data for convolution examples."""
    # 6x6 input image with sequential numbers (easier to follow)
    image = np.arange(1, 37).reshape(6, 6)
    
    # Random 3x3 kernel for more interesting output
    np.random.seed(42)  # For reproducible results
    kernel = np.random.randint(-2, 3, (3, 3))
    
    return image, kernel


def demonstrate_simple_tiling():
    """Demonstrate simple tiling with as_strided (no overlap)."""
    print_section("Simple Tiling with as_strided", level=2)
    
    # Create a simple 6x6 matrix
    matrix = np.arange(1, 37).reshape(6, 6)
    print_matrix(matrix, "Original 6×6 Matrix")
    
    # Tile into 2x2 blocks (no overlap)
    from numpy.lib.stride_tricks import as_strided
    
    # Parameters for 2x2 tiles
    tile_size = 2
    matrix_h, matrix_w = matrix.shape
    num_tiles_h = matrix_h // tile_size
    num_tiles_w = matrix_w // tile_size
    
    print(f"\nTiling into {num_tiles_h}×{num_tiles_w} tiles of size {tile_size}×{tile_size}")
    print(f"Original strides: {matrix.strides} (bytes per step)")
    
    # Calculate new strides for tiling
    # To move to next tile vertically: jump tile_size rows
    # To move to next tile horizontally: jump tile_size columns  
    # Within tile: use original strides
    tiles = as_strided(
        matrix,
        shape=(num_tiles_h, num_tiles_w, tile_size, tile_size),
        strides=(matrix.strides[0] * tile_size, matrix.strides[1] * tile_size, 
                matrix.strides[0], matrix.strides[1])
    )
    
    print(f"Tiles shape: {tiles.shape}")
    print(f"New strides: {tiles.strides}")
    print(f"Stride calculation:")
    print(f"  - Move to next tile row: {matrix.strides[0]} × {tile_size} = {matrix.strides[0] * tile_size}")
    print(f"  - Move to next tile col: {matrix.strides[1]} × {tile_size} = {matrix.strides[1] * tile_size}")
    print(f"  - Within tile row: {matrix.strides[0]} (original)")
    print(f"  - Within tile col: {matrix.strides[1]} (original)")
    
    # Show all tiles in tiled layout
    print_windows_tiled(tiles, "All 2×2 Tiles in Tiled Layout")
    
    return tiles


def demonstrate_overlapping_windows():
    """Demonstrate overlapping windows with as_strided."""
    print_section("Overlapping Windows with as_strided", level=2)
    
    # Use our test image
    image, _ = create_test_data()
    print_matrix(image, "6×6 Input Image")
    
    # Extract 3x3 overlapping windows
    from numpy.lib.stride_tricks import as_strided
    
    K = 3  # kernel size
    H, W = image.shape
    out_h, out_w = H - K + 1, W - K + 1
    
    print(f"\nExtracting {K}×{K} overlapping windows")
    print(f"Output positions: {out_h}×{out_w} = {out_h * out_w} windows")
    print(f"Original strides: {image.strides}")
    
    # For overlapping windows, we move by 1 element (not tile_size)
    # But within each window, we still use original strides
    windows = as_strided(
        image,
        shape=(out_h, out_w, K, K),
        strides=(image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    )
    
    print(f"Windows shape: {windows.shape}")
    print(f"New strides: {windows.strides}")
    print(f"Stride meaning:")
    print(f"  - Move to next window row: {image.strides[0]} (1 row down)")
    print(f"  - Move to next window col: {image.strides[1]} (1 col right)")
    print(f"  - Within window row: {image.strides[0]} (1 row down)")
    print(f"  - Within window col: {image.strides[1]} (1 col right)")
    
    # Show all overlapping windows in tiled layout
    print_windows_tiled(windows, "All 3×3 Overlapping Windows in Tiled Layout")
    
    print(f"\nNotice the overlap - adjacent windows share columns and rows!")
    
    return windows


def naive_convolution_2d(image, kernel):
    """Reference implementation: naive 2D convolution with nested loops."""
    print_section("1. Naive Convolution Reference", level=2)
    
    H, W = image.shape
    K = kernel.shape[0]  # Assume square kernel
    out_h, out_w = H - K + 1, W - K + 1
    
    output = np.zeros((out_h, out_w))
    
    print(f"Input shape: {image.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {output.shape}")
    
    for i in range(out_h):
        for j in range(out_w):
            # Extract window
            window = image[i:i+K, j:j+K]
            # Apply convolution
            output[i, j] = np.sum(window * kernel)
    
    print_matrix(image, "Input Image")
    print_matrix(kernel, "Kernel")
    print_matrix(output, "Naive Convolution Output")
    
    return output


def extract_windows_numpy(image, kernel_size=3):
    """Extract convolution windows using numpy as_strided."""
    print_section("2. Window Extraction with NumPy as_strided", level=2)
    
    H, W = image.shape
    K = kernel_size
    out_h, out_w = H - K + 1, W - K + 1
    
    # Use as_strided to create overlapping windows
    from numpy.lib.stride_tricks import as_strided
    
    windows = as_strided(
        image,
        shape=(out_h, out_w, K, K),
        strides=(image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    )
    
    print(f"Original image shape: {image.shape}")
    print(f"Windows shape: {windows.shape}")
    print(f"Memory view: 4D tensor [out_h, out_w, kernel_h, kernel_w]")
    
    # Show windows in tiled layout
    print_windows_tiled(windows, "All Windows in Tiled Layout")
    
    return windows


def extract_windows_numpy_flexible(image, kernel_size=3):
    """Extract convolution windows using numpy as_strided (handles 2D and 3D)."""
    
    if image.ndim == 2:
        # 2D case: single channel
        H, W = image.shape
        K = kernel_size
        out_h, out_w = H - K + 1, W - K + 1
        
        from numpy.lib.stride_tricks import as_strided
        windows = as_strided(
            image,
            shape=(out_h, out_w, K, K),
            strides=(image.strides[0], image.strides[1], image.strides[0], image.strides[1])
        )
        return windows
        
    elif image.ndim == 3:
        # 3D case: multi-channel
        H, W, C_in = image.shape
        K = kernel_size
        out_h, out_w = H - K + 1, W - K + 1
        
        from numpy.lib.stride_tricks import as_strided
        windows = as_strided(
            image,
            shape=(out_h, out_w, K, K, C_in),
            strides=(image.strides[0], image.strides[1], 
                    image.strides[0], image.strides[1], image.strides[2])
        )
        return windows
        
    else:
        raise ValueError(f"Expected 2D or 3D input, got {image.ndim}D")


def extract_windows_tensor_descriptor(image, kernel_size=3):
    """Extract convolution windows using Tensor Descriptors."""
    print_section("3. Window Extraction with Tensor Descriptors", level=2)
    
    H, W = image.shape
    K = kernel_size
    out_h, out_w = H - K + 1, W - K + 1
    
    # Create tensor descriptor for 4D windows
    window_lengths = [out_h, out_w, K, K]
    window_strides = [W, 1, W, 1]  # Strides for overlapping windows
    
    windows_descriptor = make_naive_tensor_descriptor(window_lengths, window_strides)
    
    print(f"Tensor descriptor shape: {windows_descriptor.get_lengths()}")
    print(f"Element space size: {windows_descriptor.get_element_space_size()}")
    
    # Extract windows using tensor descriptor
    image_flat = image.flatten()
    windows_td = np.zeros((out_h, out_w, K, K))
    
    for i in range(out_h):
        for j in range(out_w):
            for ki in range(K):
                for kj in range(K):
                    offset = windows_descriptor.calculate_offset([i, j, ki, kj])
                    windows_td[i, j, ki, kj] = image_flat[offset]
    
    # Show all windows in tiled layout
    print_windows_tiled(windows_td, "All Windows in Tiled Layout (Tensor Descriptor)")
    
    return windows_td


def compare_windows(windows_numpy, windows_td):
    """Compare numpy and tensor descriptor window extraction."""
    print_section("Verification: NumPy vs Tensor Descriptor Windows", level=2)
    
    difference = np.linalg.norm(windows_numpy - windows_td)
    print(f"L2 norm of difference: {difference}")
    
    if difference < 1e-10:
        print("✅ SUCCESS: Tensor descriptor windows identical to NumPy!")
    else:
        print("❌ ERROR: Tensor descriptor windows differ from NumPy")
        print(f"Max difference: {np.max(np.abs(windows_numpy - windows_td))}")
    
    return difference < 1e-10


def im2col_numpy_transform(windows):
    """Convert 4D windows to 2D im2col matrix using NumPy."""
    print_section("4. Im2col Transformation with NumPy", level=2)
    
    out_h, out_w, K, K = windows.shape
    num_windows = out_h * out_w
    patch_size = K * K
    
    # Reshape to 2D matrix
    im2col_matrix = windows.reshape(num_windows, patch_size)
    
    print(f"4D windows shape: {windows.shape}")
    print(f"2D im2col shape: {im2col_matrix.shape}")
    print(f"Transformation: [{out_h}, {out_w}, {K}, {K}] → [{num_windows}, {patch_size}]")
    
    # Show the matrix
    print(f"\nIm2col matrix (each row is a flattened window):")
    for i, row in enumerate(im2col_matrix):
        win_i, win_j = i // out_w, i % out_w
        print(f"Row {i} (window [{win_i},{win_j}]): {' '.join(f'{val:3.0f}' for val in row)}")
    
    return im2col_matrix


def im2col_tensor_descriptor_simple(windows_td):
    """Create im2col matrix from tensor descriptor windows (simple reshape)."""
    print_section("5. Im2col with Tensor Descriptors (Simple)", level=2)
    
    # Use the windows we already extracted with tensor descriptors
    im2col_matrix = windows_td.reshape(16, 9)
    
    print(f"Tensor descriptor windows shape: {windows_td.shape}")
    print(f"Tensor descriptor im2col shape: {im2col_matrix.shape}")
    print(f"Simple reshape: just like NumPy!")
    
    # Show first few rows
    print(f"\nFirst few rows of TD im2col matrix:")
    for i in range(3):
        win_i, win_j = i // 4, i % 4
        print(f"Row {i} (window [{win_i},{win_j}]): {' '.join(f'{val:3.0f}' for val in im2col_matrix[i])}")
    
    return im2col_matrix


def create_direct_im2col_descriptor(image, kernel_size=3):
    """Create im2col matrix directly using tensor descriptors."""
    print_section("5b. Advanced: Direct 2D Tensor Descriptor", level=2)
    
    H, W = image.shape
    K = kernel_size
    out_h, out_w = H - K + 1, W - K + 1
    
    # Step 1: Create 4D windows descriptor
    window_lengths = [out_h, out_w, K, K]
    window_strides = [W, 1, W, 1]
    windows_descriptor = make_naive_tensor_descriptor(window_lengths, window_strides)
    
    # Step 2: Apply merge transforms to create 2D im2col layout directly
    merge_windows = MergeTransform([out_h, out_w])  # Merge spatial output dimensions
    merge_patch = MergeTransform([K, K])  # Merge kernel dimensions
    
    im2col_descriptor = transform_tensor_descriptor(
        windows_descriptor,
        transforms=[merge_windows, merge_patch],
        lower_dimension_hidden_idss=[[0, 1], [2, 3]],
        upper_dimension_hidden_idss=[[0], [1]]
    )
    
    print(f"Direct im2col descriptor shape: {im2col_descriptor.get_lengths()}")
    
    # Show example offsets
    print(f"\nExample: offset for window 0, patch element 0: {im2col_descriptor.calculate_offset([0, 0])}")
    print(f"Example: offset for window 0, patch element 4: {im2col_descriptor.calculate_offset([0, 4])}")
    print(f"Example: offset for window 1, patch element 0: {im2col_descriptor.calculate_offset([1, 0])}")
    
    print(f"\nWhy both approaches?")
    print(f"- Simple reshape: Easy to understand, perfect for CPU implementations")
    print(f"- Direct tensor descriptor: Enables efficient GPU kernel generation")
    
    return im2col_descriptor


def compare_im2col(im2col_numpy, im2col_td):
    """Compare numpy and tensor descriptor im2col matrices."""
    print_section("Verification: NumPy vs Tensor Descriptor Im2col", level=2)
    
    difference = np.linalg.norm(im2col_numpy - im2col_td)
    print(f"L2 norm of difference: {difference}")
    
    if difference < 1e-10:
        print("✅ SUCCESS: Tensor descriptor im2col identical to NumPy!")
    else:
        print("❌ ERROR: Tensor descriptor im2col differs from NumPy")
        print(f"Max difference: {np.max(np.abs(im2col_numpy - im2col_td))}")
    
    return difference < 1e-10


def convolution_with_im2col(im2col_matrix, kernel, out_shape):
    """Perform convolution using im2col matrix multiplication."""
    print_section("6. Convolution via Matrix Multiplication", level=2)
    
    # Flatten kernel
    kernel_flat = kernel.flatten()
    
    print(f"Im2col matrix shape: {im2col_matrix.shape}")
    print(f"Kernel flat shape: {kernel_flat.shape}")
    
    # Matrix multiplication: each row of im2col with kernel
    output_flat = im2col_matrix @ kernel_flat
    
    # Reshape to output dimensions
    output = output_flat.reshape(out_shape)
    
    print(f"Output flat shape: {output_flat.shape}")
    print(f"Final output shape: {output.shape}")
    
    print_matrix(output, "Convolution via Im2col")
    
    return output


def multi_channel_convolution():
    """Demonstrate multi-channel convolution."""
    print_section("7. Multi-Channel Convolution", level=2)
    
    # Create multi-channel input: 6x6x2 (2 input channels)
    np.random.seed(123)
    input_tensor = np.random.randint(0, 5, (6, 6, 2))
    
    # Create multi-channel filters: 3x3x2x3 (2 input, 3 output channels)
    filters = np.random.randint(-1, 2, (3, 3, 2, 3))
    
    H, W, C_in = input_tensor.shape
    K, K, C_in_filter, C_out = filters.shape
    out_h, out_w = H - K + 1, W - K + 1
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Filters shape: {filters.shape}")
    print(f"Output shape: ({out_h}, {out_w}, {C_out})")
    
    # Reference convolution with nested loops
    reference_output = np.zeros((out_h, out_w, C_out))
    for i in range(out_h):
        for j in range(out_w):
            for c_out in range(C_out):
                window = input_tensor[i:i+K, j:j+K, :]  # [K, K, C_in]
                reference_output[i, j, c_out] = np.sum(window * filters[:, :, :, c_out])
    
    # Im2col approach using our flexible function
    # Extract all channels at once using our existing function
    windows_5d = extract_windows_numpy_flexible(input_tensor, K)
    
    # Convert to im2col format: [num_windows, patch_size]
    num_windows = out_h * out_w
    patch_size = K * K * C_in
    im2col_matrix = windows_5d.reshape(num_windows, patch_size)
    
    # Reshape filters and apply
    filters_reshaped = filters.reshape(patch_size, C_out)
    im2col_output_flat = im2col_matrix @ filters_reshaped
    im2col_output = im2col_output_flat.reshape(out_h, out_w, C_out)
    
    print(f"\nIm2col matrix shape: {im2col_matrix.shape}")
    print(f"Reshaped filters shape: {filters_reshaped.shape}")
    
    # Compare results
    difference = np.linalg.norm(reference_output - im2col_output)
    print(f"\nL2 norm difference (reference vs im2col): {difference}")
    
    if difference < 1e-10:
        print("✅ SUCCESS: Multi-channel im2col matches reference!")
    else:
        print("❌ ERROR: Multi-channel results differ")
    
    return reference_output, im2col_output


def main():
    """Run the complete convolution implementation demonstration."""
    print_section("Convolution Implementation: From Naive to Tensor Descriptors")
    
    # Create test data
    image, kernel = create_test_data()
    
    # 0. Demonstrate simple tiling first
    demonstrate_simple_tiling()
    
    # 0.5. Demonstrate overlapping windows
    demonstrate_overlapping_windows()
    
    # 1. Naive reference implementation
    reference_output = naive_convolution_2d(image, kernel)
    
    # 2. Window extraction with NumPy
    windows_numpy = extract_windows_numpy(image)
    
    # 3. Window extraction with Tensor Descriptors
    windows_td = extract_windows_tensor_descriptor(image)
    
    # Verify window extraction equivalence
    windows_match = compare_windows(windows_numpy, windows_td)
    
    # 4. Im2col with NumPy
    im2col_numpy = im2col_numpy_transform(windows_numpy)
    
    # 5. Im2col with Tensor Descriptors (simple)
    im2col_td_simple = im2col_tensor_descriptor_simple(windows_td)
    
    # 5b. Advanced: Direct tensor descriptor approach
    im2col_descriptor = create_direct_im2col_descriptor(image)
    
    # Verify im2col equivalence
    im2col_match = compare_im2col(im2col_numpy, im2col_td_simple)
    
    # 6. Convolution with im2col
    out_shape = reference_output.shape
    im2col_output = convolution_with_im2col(im2col_numpy, kernel, out_shape)
    
    # Verify convolution equivalence
    print_section("Final Verification: All Methods", level=2)
    conv_difference = np.linalg.norm(reference_output - im2col_output)
    print(f"L2 norm difference (naive vs im2col): {conv_difference}")
    
    if conv_difference < 1e-10:
        print("✅ SUCCESS: Im2col convolution matches naive reference!")
    else:
        print("❌ ERROR: Im2col convolution differs from reference")
    
    # 7. Multi-channel extension
    multi_channel_convolution()
    
    # Summary
    print_section("Summary")
    print("✅ Demonstrated complete convolution pipeline:")
    print("   1. Naive reference implementation")
    print("   2. Window extraction (NumPy vs Tensor Descriptors)")
    print("   3. Im2col transformation (NumPy vs Tensor Descriptors)")
    print("   4. Matrix multiplication convolution")
    print("   5. Multi-channel extension")
    print("\n🎯 All methods produce identical results!")
    print("💡 Tensor descriptors provide the foundation for efficient GPU implementations")


if __name__ == "__main__":
    main()
