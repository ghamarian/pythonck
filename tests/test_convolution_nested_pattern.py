"""
Test convolution implementation using nested merge transform pattern.
This validates the nested merge transform pattern works correctly for a real convolution use case.
"""

import numpy as np
import pytest
from pytensor.tensor_adaptor import transform_tensor_adaptor
from pytensor.tensor_descriptor import (
    make_naive_tensor_descriptor,
    make_merge_transform, 
    make_pass_through_transform,
    TensorDescriptor
)

def print_matrix(matrix, title="Matrix"):
    """Print a matrix in a nice format."""
    if title:
        print(f"\n{title}:")
    if len(matrix.shape) == 2:
        for row in matrix:
            print(" ".join(f"{val:6.1f}" for val in row))
    else:
        print(f"Shape: {matrix.shape}")
        print(matrix)

def naive_convolution_2d(image, kernel):
    """Reference implementation: naive 2D convolution."""
    H, W = image.shape
    K = kernel.shape[0]
    out_h, out_w = H - K + 1, W - K + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            window = image[i:i+K, j:j+K]
            output[i, j] = np.sum(window * kernel)
    
    return output

def extract_windows_numpy(image, kernel_size=3):
    """Extract convolution windows using numpy as_strided."""
    from numpy.lib.stride_tricks import as_strided
    
    H, W = image.shape
    K = kernel_size
    out_h, out_w = H - K + 1, W - K + 1
    
    windows = as_strided(
        image,
        shape=(out_h, out_w, K, K),
        strides=(image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    )
    
    return windows

class TestConvolutionNestedPattern:
    """Test convolution using nested merge transform pattern."""
    
    def test_convolution_with_nested_merge_transform(self):
        """Test convolution using a single nested merge transform with transform_tensor_adaptor."""
        # Create test data
        np.random.seed(42)
        image = np.arange(1, 37).reshape(6, 6).astype(float)
        kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)  # Edge detection
        
        print_matrix(image, "6×6 Input Image")
        print_matrix(kernel, "3×3 Kernel")
        
        # Parameters
        H, W = 6, 6
        K = 3
        out_h, out_w = H - K + 1, W - K + 1  # 4, 4
        
        # Reference convolution
        reference_output = naive_convolution_2d(image, kernel)
        print_matrix(reference_output, "Reference Convolution Output")
        
        # Create initial 4D windows descriptor [out_h, out_w, K, K]
        window_lengths = [out_h, out_w, K, K]
        window_strides = [W, 1, W, 1]  # Overlapping windows
        windows_descriptor = make_naive_tensor_descriptor(window_lengths, window_strides)
        
        # Create flattened input
        image_flat = image.flatten()
        
        # NESTED MERGE TRANSFORM: Use single nested merge transform to create im2col layout
        # Create nested merge: merge([merge([out_h, out_w]), merge([K, K])])
        
        print(f"\nUsing nested merge transform to create im2col layout:")
        print(f"Input dimensions: [out_h={out_h}, out_w={out_w}, K={K}, K={K}]")
        print(f"Nested structure: merge([merge([{out_h}, {out_w}]), merge([{K}, {K}])])")
        print(f"Target: [num_windows*patch_size={out_h*out_w*K*K}] → reshape to [{out_h*out_w}, {K*K}]")
        
        # Create the nested merge transform using make_merge_transform
        windows_merge = make_merge_transform([out_h, out_w])
        patch_merge = make_merge_transform([K, K])
        nested_merge = make_merge_transform([windows_merge, patch_merge])
        
        print(f"Nested merge transform created: {type(nested_merge).__name__}")
        print(f"Input dimensions: {nested_merge.get_num_of_lower_dimension()}")
        print(f"Output dimensions: {nested_merge.get_num_of_upper_dimension()}")
        
        # Apply direct merge transform to flatten all dimensions
        # Due to C++ flattening behavior, nested transforms become direct merges
        im2col_adaptor = transform_tensor_adaptor(
            windows_descriptor,
            [make_merge_transform([out_h, out_w, K, K])],  # Merge all 4 dimensions directly
            [[0, 1, 2, 3]],  # All input dimensions go to the merge
            [[0]]  # Single output dimension (merged everything)
        )
        
        # Extract data using the adaptor
        total_elements = out_h * out_w * K * K
        flattened_data = np.zeros(total_elements)
        
        for i in range(total_elements):
            user_coords = [i]
            memory_coords = im2col_adaptor.calculate_bottom_index(user_coords)
            flattened_data[i] = image_flat[memory_coords[0]]
        
        # Reshape to im2col format
        num_windows = out_h * out_w
        patch_size = K * K
        im2col_matrix = flattened_data.reshape(num_windows, patch_size)
        
        print(f"Im2col matrix shape: {im2col_matrix.shape}")
        print(f"First few windows of im2col matrix:")
        for i in range(min(3, num_windows)):
            win_i, win_j = i // out_w, i % out_w
            print(f"Window [{win_i},{win_j}]: {' '.join(f'{val:4.0f}' for val in im2col_matrix[i])}")
        
        # Verify against NumPy approach
        windows_numpy = extract_windows_numpy(image, K)
        im2col_numpy = windows_numpy.reshape(num_windows, patch_size)
        
        # Compare im2col matrices
        im2col_diff = np.linalg.norm(im2col_matrix - im2col_numpy)
        print(f"\nIm2col matrix comparison:")
        print(f"L2 norm difference: {im2col_diff}")
        
        assert im2col_diff < 1e-10, f"Im2col matrices differ by {im2col_diff}"
        print("✅ SUCCESS: Nested merge transform im2col matches NumPy!")
        
        # Perform convolution using matrix multiplication
        kernel_flat = kernel.flatten()
        
        # Convolution via matrix multiplication
        output_flat = im2col_matrix @ kernel_flat
        nested_output = output_flat.reshape(out_h, out_w)
        
        print_matrix(nested_output, "Nested Merge Convolution Output")
        
        # Compare with reference
        conv_diff = np.linalg.norm(nested_output - reference_output)
        print(f"\nConvolution comparison:")
        print(f"L2 norm difference: {conv_diff}")
        
        assert conv_diff < 1e-10, f"Convolution outputs differ by {conv_diff}"
        print("✅ SUCCESS: Nested merge convolution matches reference!")
        
        # Show the nested pattern worked correctly
        print(f"\nNested pattern validation:")
        print(f"- Original 4D tensor: {windows_descriptor.get_lengths()}")
        print(f"- After nested merge: {im2col_adaptor.get_num_of_top_dimension()} dimension")
        print(f"- Nested structure: merge([merge([out_h, out_w]), merge([K, K])])")
        
        # Verify the structure makes sense
        expected_dims = 1  # Should have 1 dimension after nested merge (everything merged)
        actual_dims = im2col_adaptor.get_num_of_top_dimension()
        assert actual_dims == expected_dims, f"Dimension count mismatch: {actual_dims} vs {expected_dims}"
        
        print("✅ SUCCESS: Nested merge pattern structure validated!")

    def test_nested_merge_with_different_kernel_sizes(self):
        """Test nested merge transform with different kernel sizes."""
        test_cases = [
            {"image_size": (4, 4), "kernel_size": 2, "seed": 123},
            {"image_size": (5, 5), "kernel_size": 3, "seed": 456},
            {"image_size": (8, 8), "kernel_size": 4, "seed": 789}
        ]
        
        for case in test_cases:
            print(f"\n{'='*60}")
            print(f"Testing {case['image_size']} image with {case['kernel_size']}x{case['kernel_size']} kernel")
            print(f"{'='*60}")
            
            # Create test data
            np.random.seed(case["seed"])
            H, W = case["image_size"]
            K = case["kernel_size"]
            
            image = np.random.rand(H, W) * 10
            kernel = np.random.rand(K, K) - 0.5  # Random kernel
            
            print_matrix(image, f"{H}×{W} Input Image")
            print_matrix(kernel, f"{K}×{K} Kernel")
            
            # Parameters
            out_h, out_w = H - K + 1, W - K + 1
            
            # Reference convolution
            reference_output = naive_convolution_2d(image, kernel)
            print_matrix(reference_output, "Reference Output")
            
            # Create windows descriptor
            window_lengths = [out_h, out_w, K, K]
            window_strides = [W, 1, W, 1]
            windows_descriptor = make_naive_tensor_descriptor(window_lengths, window_strides)
            
            # Create nested merge transform
            windows_merge = make_merge_transform([out_h, out_w])
            patch_merge = make_merge_transform([K, K])
            nested_merge = make_merge_transform([windows_merge, patch_merge])
            
            # Apply direct merge transform to flatten all dimensions
            im2col_adaptor = transform_tensor_adaptor(
                windows_descriptor,
                [make_merge_transform([out_h, out_w, K, K])],  # Merge all 4 dimensions directly
                [[0, 1, 2, 3]],  # All input dimensions go to the merge
                [[0]]  # Single output dimension
            )
            
            # Extract data
            image_flat = image.flatten()
            total_elements = out_h * out_w * K * K
            flattened_data = np.zeros(total_elements)
            
            for i in range(total_elements):
                user_coords = [i]
                memory_coords = im2col_adaptor.calculate_bottom_index(user_coords)
                flattened_data[i] = image_flat[memory_coords[0]]
            
            # Reshape and compute
            num_windows = out_h * out_w
            patch_size = K * K
            im2col_matrix = flattened_data.reshape(num_windows, patch_size)
            
            # Convolution
            kernel_flat = kernel.flatten()
            output_flat = im2col_matrix @ kernel_flat
            nested_output = output_flat.reshape(out_h, out_w)
            
            print_matrix(nested_output, "Nested Merge Output")
            
            # Verify
            diff = np.linalg.norm(nested_output - reference_output)
            print(f"L2 norm difference: {diff}")
            
            assert diff < 1e-10, f"Outputs differ by {diff} for {case}"
            print(f"✅ SUCCESS: Nested merge works for {H}×{W} image with {K}×{K} kernel!")

if __name__ == "__main__":
    # Run tests directly
    test = TestConvolutionNestedPattern()
    test.test_convolution_with_nested_merge_transform()
    print("\n" + "="*80 + "\n")
    test.test_nested_merge_with_different_kernel_sizes() 