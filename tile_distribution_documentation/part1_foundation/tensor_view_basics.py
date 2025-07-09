#!/usr/bin/env python3
"""
Purpose: Demonstrate pytensor TensorView concepts - multi-dimensional views of buffer data.

Shows how to create TensorView objects that provide structured access to
BufferView data with shape, strides, and coordinate transformations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code_examples'))

from common_utils import *
from pytensor.buffer_view import make_buffer_view
from pytensor.tensor_view import (
    TensorView,
    make_tensor_view,
    make_naive_tensor_view,
    make_naive_tensor_view_packed,
    transform_tensor_view
)
from pytensor.tensor_descriptor import (
    make_naive_tensor_descriptor,
    make_naive_tensor_descriptor_packed
)
import numpy as np

def demonstrate_tensor_view_creation():
    """Show how to create TensorView objects from BufferView."""
    print_step(1, "Creating TensorView objects")
    
    # Create buffer data
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32)
    buffer_view = make_buffer_view(data)
    show_result("Buffer data", data)
    
    # Create tensor descriptor for 2D view (3x4)
    tensor_desc = make_naive_tensor_descriptor([3, 4])
    show_tensor_info(tensor_desc, "2D tensor descriptor")
    
    # Create tensor view
    tensor_view = make_tensor_view(data, tensor_desc)
    show_result("TensorView created", type(tensor_view).__name__)
    
    return tensor_view, tensor_desc

def demonstrate_tensor_view_properties():
    """Show TensorView properties and methods."""
    print_step(2, "TensorView properties")
    
    data = np.array(range(24), dtype=np.float32)
    tensor_desc = make_naive_tensor_descriptor([4, 6])
    tensor_view = make_tensor_view(data, tensor_desc)
    
    show_result("Number of dimensions", tensor_view.get_num_of_dimension())
    show_result("Tensor lengths", tensor_view.get_lengths())
    show_result("Element space size", tensor_view.get_element_space_size())
    show_result("Is static", tensor_view.is_static())
    
    return tensor_view

def demonstrate_tensor_coordinate_access():
    """Show how to access tensor elements using coordinates."""
    print_step(3, "Tensor coordinate access")
    
    data = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
    tensor_desc = make_naive_tensor_descriptor([2, 3])
    tensor_view = make_tensor_view(data, tensor_desc)
    
    from pytensor.tensor_coordinate import MultiIndex
    
    # Access elements using coordinates
    coord_00 = MultiIndex(2, [0, 0])
    coord_12 = MultiIndex(2, [1, 2])
    
    show_result("Element at [0,0]", tensor_view.get_element(coord_00))
    show_result("Element at [1,2]", tensor_view.get_element(coord_12))
    
    # Modify elements
    tensor_view.set_element(coord_00, 999.0)
    show_result("After setting [0,0] to 999", tensor_view.get_element(coord_00))
    show_result("Original data after modification", data)
    
    return tensor_view

def demonstrate_packed_tensor_view():
    """Show packed tensor view creation."""
    print_step(4, "Packed tensor view")
    
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    
    # Create packed tensor view (row-major layout)
    tensor_view = make_naive_tensor_view_packed(data, [2, 4])
    
    show_result("Packed tensor dimensions", tensor_view.get_num_of_dimension())
    show_result("Packed tensor lengths", tensor_view.get_lengths())
    
    from pytensor.tensor_coordinate import MultiIndex
    
    # Test coordinate access
    coords_to_test = [(0, 0), (0, 3), (1, 0), (1, 3)]
    for row, col in coords_to_test:
        coord = MultiIndex(2, [row, col])
        value = tensor_view.get_element(coord)
        show_result(f"Element at [{row},{col}]", value)
    
    return tensor_view

def demonstrate_tensor_view_transforms():
    """Show tensor view transformations."""
    print_step(5, "Tensor view transformations")
    
    data = np.array(range(12), dtype=np.float32)
    original_view = make_naive_tensor_view_packed(data, [3, 4])
    
    show_result("Original shape", original_view.get_lengths())
    
    # Create transpose transformation
    from pytensor.tensor_adaptor import make_transpose_adaptor
    transpose_adaptor = make_transpose_adaptor([3, 4])
    show_adaptor_info(transpose_adaptor, "Transpose adaptor")
    
    # Apply transformation
    transposed_view = transform_tensor_view(original_view, transpose_adaptor)
    show_result("Transposed shape", transposed_view.get_lengths())
    
    return original_view, transposed_view

def demonstrate_multi_dimensional_views():
    """Show multi-dimensional tensor views."""
    print_step(6, "Multi-dimensional tensor views")
    
    data = np.array(range(24), dtype=np.float32)
    
    # Create 3D tensor view
    tensor_view_3d = make_naive_tensor_view_packed(data, [2, 3, 4])
    show_result("3D tensor shape", tensor_view_3d.get_lengths())
    
    # Create 4D tensor view
    tensor_view_4d = make_naive_tensor_view_packed(data, [2, 2, 3, 2])
    show_result("4D tensor shape", tensor_view_4d.get_lengths())
    
    from pytensor.tensor_coordinate import MultiIndex
    
    # Access 3D element
    coord_3d = MultiIndex(3, [1, 2, 1])
    value_3d = tensor_view_3d.get_element(coord_3d)
    show_result("3D element at [1,2,1]", value_3d)
    
    # Access 4D element  
    coord_4d = MultiIndex(4, [1, 1, 2, 1])
    value_4d = tensor_view_4d.get_element(coord_4d)
    show_result("4D element at [1,1,2,1]", value_4d)
    
    return tensor_view_3d, tensor_view_4d

def test_tensor_view_operations():
    """Test TensorView operations."""
    print_step(7, "Testing TensorView operations")
    
    def test_creation():
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        tensor_view = make_naive_tensor_view_packed(data, [2, 2])
        return tensor_view.get_num_of_dimension() == 2
    
    def test_coordinate_access():
        data = np.array([10, 20, 30, 40], dtype=np.float32)
        tensor_view = make_naive_tensor_view_packed(data, [2, 2])
        from pytensor.tensor_coordinate import MultiIndex
        coord = MultiIndex(2, [1, 1])
        return tensor_view.get_element(coord) == 40.0
    
    def test_modification():
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        tensor_view = make_naive_tensor_view_packed(data, [2, 2])
        from pytensor.tensor_coordinate import MultiIndex
        coord = MultiIndex(2, [0, 0])
        tensor_view.set_element(coord, 999.0)
        return tensor_view.get_element(coord) == 999.0
    
    tests = [
        ("TensorView creation", test_creation),
        ("Coordinate access", test_coordinate_access),
        ("Element modification", test_modification)
    ]
    
    results = []
    for test_name, test_func in tests:
        results.append(validate_example(test_name, test_func))
    
    return all(results)

def main():
    """Main function to run all TensorView demonstrations."""
    if not check_imports():
        return False
    
    print_section("TensorView Basics")
    
    # Run demonstrations
    tensor_view1, tensor_desc1 = demonstrate_tensor_view_creation()
    tensor_view2 = demonstrate_tensor_view_properties()
    tensor_view3 = demonstrate_tensor_coordinate_access()
    tensor_view4 = demonstrate_packed_tensor_view()
    original_view, transposed_view = demonstrate_tensor_view_transforms()
    tensor_view_3d, tensor_view_4d = demonstrate_multi_dimensional_views()
    
    # Run tests
    all_tests_passed = test_tensor_view_operations()
    
    print_section("Summary")
    print(f"✅ TensorView demonstrations completed")
    print(f"✅ All tests passed: {all_tests_passed}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = run_script_safely(main, "TensorView Basics")
    sys.exit(0 if success else 1) 