"""
Example usage of the PyTensor library.

This script demonstrates how to use buffer views, tensor descriptors,
and tensor coordinates together.
"""

import sys
sys.path.append('..')

import numpy as np
from pytensor import (
    # Buffer view
    make_buffer_view, AddressSpaceEnum,
    # Tensor descriptor
    make_naive_tensor_descriptor, make_naive_tensor_descriptor_packed,
    make_naive_tensor_descriptor_aligned,
    # Tensor coordinate
    make_tensor_coordinate, move_tensor_coordinate
)


def example_strided_tensor():
    """Example of using a strided tensor descriptor."""
    print("=== Strided Tensor Example ===")
    
    # Create a 3x4 matrix in row-major layout
    rows, cols = 3, 4
    data = np.arange(rows * cols, dtype=np.float32)
    print(f"Data: {data}")
    print(f"As matrix:\n{data.reshape(rows, cols)}")
    
    # Create buffer view
    buf = make_buffer_view(data, len(data), AddressSpaceEnum.GLOBAL)
    
    # Create tensor descriptor (row-major: stride for row is cols, stride for col is 1)
    desc = make_naive_tensor_descriptor([rows, cols], [cols, 1])
    print(f"\nTensor descriptor: {desc}")
    print(f"Element space size: {desc.get_element_space_size()}")
    
    # Access elements using tensor coordinates
    print("\nAccessing elements:")
    for i in range(rows):
        for j in range(cols):
            offset = desc.calculate_offset([i, j])
            value = buf.get(offset, 0, True)
            print(f"  [{i},{j}] -> offset {offset:2d} -> value {value:4.1f}")
    
    # Move coordinates
    print("\nMoving coordinates:")
    coord = make_tensor_coordinate(desc, [0, 0])
    print(f"  Start at [0,0], offset = {coord.get_offset()}")
    
    move_tensor_coordinate(desc, coord, [1, 0])  # Move down one row
    print(f"  After moving [1,0], offset = {coord.get_offset()}")
    
    move_tensor_coordinate(desc, coord, [0, 2])  # Move right two columns
    print(f"  After moving [0,2], offset = {coord.get_offset()}")


def example_packed_tensor():
    """Example of using a packed tensor descriptor."""
    print("\n=== Packed Tensor Example ===")
    
    # Create a 2x3x4 3D tensor
    dims = [2, 3, 4]
    total_elements = np.prod(dims)
    data = np.arange(total_elements, dtype=np.float32)
    print(f"Data shape: {dims}")
    print(f"Total elements: {total_elements}")
    
    # Create buffer view
    buf = make_buffer_view(data, len(data), AddressSpaceEnum.GLOBAL)
    
    # Create packed tensor descriptor
    desc = make_naive_tensor_descriptor_packed(dims)
    print(f"\nPacked tensor descriptor: {desc}")
    print(f"Element space size: {desc.get_element_space_size()}")
    
    # Verify packed layout (consecutive elements)
    print("\nVerifying packed layout:")
    prev_offset = -1
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                offset = desc.calculate_offset([i, j, k])
                if prev_offset >= 0:
                    assert offset == prev_offset + 1, "Not packed!"
                prev_offset = offset
    print("  ✓ All elements are consecutive (packed)")
    
    # Show some element accesses
    print("\nSample element accesses:")
    test_indices = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 2, 3]]
    for idx in test_indices:
        offset = desc.calculate_offset(idx)
        value = buf.get(offset, 0, True)
        print(f"  {idx} -> offset {offset:2d} -> value {value:4.1f}")


def example_aligned_tensor():
    """Example of using an aligned tensor descriptor."""
    print("\n=== Aligned Tensor Example ===")
    
    # Create a matrix with aligned rows
    rows, cols = 3, 5
    alignment = 8  # Align each row to 8 elements
    
    # Calculate actual storage size with alignment
    aligned_cols = ((cols + alignment - 1) // alignment) * alignment
    total_elements = rows * aligned_cols
    
    # Create data with padding
    data = np.full(total_elements, -1.0, dtype=np.float32)  # -1 for padding
    
    # Fill actual data
    for i in range(rows):
        for j in range(cols):
            data[i * aligned_cols + j] = i * cols + j
    
    print(f"Matrix: {rows}x{cols}, alignment: {alignment}")
    print(f"Aligned columns: {aligned_cols}")
    print(f"Data with padding: {data}")
    
    # Create buffer view
    buf = make_buffer_view(data, len(data), AddressSpaceEnum.GLOBAL)
    
    # Create aligned tensor descriptor
    desc = make_naive_tensor_descriptor_aligned([rows, cols], alignment)
    print(f"\nAligned tensor descriptor: {desc}")
    
    # Show memory layout
    print("\nMemory layout:")
    for i in range(rows):
        row_data = []
        for j in range(aligned_cols):
            offset = i * aligned_cols + j
            value = data[offset]
            if j < cols:
                row_data.append(f"{value:4.0f}")
            else:
                row_data.append("  - ")  # Padding
        print(f"  Row {i}: {' '.join(row_data)}")
    
    # Verify alignment
    print("\nVerifying alignment:")
    for i in range(rows):
        # Check that each row starts at an aligned offset
        offset = desc.calculate_offset([i, 0])
        print(f"  Row {i} starts at offset {offset} (aligned: {offset % alignment == 0})")


def example_vectorized_access():
    """Example of vectorized buffer access."""
    print("\n=== Vectorized Access Example ===")
    
    # Create data
    data = np.arange(16, dtype=np.float32)
    buf = make_buffer_view(data, len(data), AddressSpaceEnum.GLOBAL)
    
    print(f"Data: {data}")
    
    # Single element access
    print("\nSingle element access:")
    value = buf.get(5, 0, True, vector_size=1)
    print(f"  buf.get(5) = {value}")
    
    # Vector access
    print("\nVector access (size=4):")
    vec = buf.get(4, 0, True, vector_size=4)
    print(f"  buf.get(4, vector_size=4) = {vec}")
    
    # Vector write
    print("\nVector write:")
    new_vec = np.array([100, 101, 102, 103], dtype=np.float32)
    buf.set(8, 0, True, new_vec, vector_size=4)
    print(f"  After writing {new_vec} at offset 8:")
    print(f"  Data: {data}")


def example_tensor_view():
    """Example of using tensor view for unified access."""
    print("\n=== Tensor View Example ===")
    
    # Import tensor view functions
    from pytensor import make_naive_tensor_view, make_tensor_coordinate, move_tensor_coordinate
    
    # Create a 4x5 matrix
    rows, cols = 4, 5
    data = np.arange(rows * cols, dtype=np.float32)
    
    # Create tensor view
    view = make_naive_tensor_view(data, [rows, cols], [cols, 1])
    
    print(f"Created {rows}x{cols} tensor view")
    print(f"Data: {data}")
    
    # Array-style indexing
    print("\nArray-style indexing:")
    print(f"  view[0, 0] = {view[0, 0]}")
    print(f"  view[1, 2] = {view[1, 2]}")
    print(f"  view[3, 4] = {view[3, 4]}")
    
    # Modify elements
    print("\nModifying elements:")
    view[0, 1] = 100.0
    view[2, 3] = 200.0
    print(f"  Set view[0, 1] = 100.0")
    print(f"  Set view[2, 3] = 200.0")
    print(f"  Data: {data}")
    
    # Using coordinates
    print("\nUsing tensor coordinates:")
    coord = make_tensor_coordinate(view.tensor_desc, [1, 0])
    print(f"  Created coordinate at [1, 0]")
    
    # Get vectorized elements
    vec = view.get_vectorized_elements(coord, 0, 5)
    print(f"  Vector of 5 elements from [1, 0]: {vec}")
    
    # Move coordinate and access
    move_tensor_coordinate(view.tensor_desc, coord, [1, 2])
    print(f"  Moved coordinate by [1, 2] to position [2, 2]")
    print(f"  Element at new position: {view.get_element(coord)}")
    
    # Update with ADD operation
    print("\nUpdate operations:")
    from pytensor import MemoryOperationEnum
    view.dst_in_mem_op = MemoryOperationEnum.ADD
    
    coord = make_tensor_coordinate(view.tensor_desc, [3, 0])
    update_vec = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    print(f"  Adding {update_vec} at position [3, 0]")
    
    original = data[15:20].copy()
    view.update_vectorized_elements(coord, update_vec, 0, 5)
    print(f"  Original values: {original}")
    print(f"  New values: {data[15:20]}")


if __name__ == "__main__":
    example_strided_tensor()
    example_packed_tensor()
    example_aligned_tensor()
    example_vectorized_access()
    example_tensor_view() 