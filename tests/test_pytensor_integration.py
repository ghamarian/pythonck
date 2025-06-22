"""
Test script for pytensor integration with the parser.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_transforms import TensorTransformParser
import sympy as sp

def test_simple_transforms():
    """Test creating pytensor transforms and using their sympy methods."""
    parser = TensorTransformParser()
    
    # Test 1: PassThrough Transform
    print("=== Test 1: PassThrough Transform ===")
    transform_dict = {
        'type': 'pass_through',
        'value': sp.Symbol('N')
    }
    variables = {'N': 8}
    
    transform = parser.create_pytensor_transform(transform_dict, variables)
    print(f"Created transform: {transform}")
    
    # Test sympy methods
    input_syms = [sp.Symbol('x')]
    forward_result = transform.sympy_forward(input_syms)
    print(f"Forward: {input_syms} -> {forward_result}")
    
    backward_result = transform.sympy_backward(forward_result)
    print(f"Backward: {forward_result} -> {backward_result}")
    
    # Test 2: Merge Transform
    print("\n=== Test 2: Merge Transform ===")
    transform_dict = {
        'type': 'merge',
        'values': [
            {'type': 'pass_through', 'value': 4},
            {'type': 'pass_through', 'value': 8}
        ]
    }
    
    transform = parser.create_pytensor_transform(transform_dict, variables)
    print(f"Created transform: {transform}")
    
    # Test sympy methods
    input_syms = [sp.Symbol('a'), sp.Symbol('b')]
    forward_result = transform.sympy_forward(input_syms)
    print(f"Forward: {input_syms} -> {forward_result}")
    
    backward_result = transform.sympy_backward(forward_result)
    print(f"Backward: {forward_result} -> {backward_result}")

def test_naive_descriptor():
    """Test creating a naive tensor descriptor."""
    print("\n=== Test 3: Naive Descriptor ===")
    parser = TensorTransformParser()
    
    desc_str = "make_naive_tensor_descriptor_packed(make_tuple(number<4>{}, number<8>{}))"
    variables = {}
    
    try:
        tensor_desc = parser.create_pytensor_descriptor(desc_str, variables)
        print(f"Created descriptor: {tensor_desc}")
        print(f"Number of dimensions: {tensor_desc.get_num_of_dimension()}")
        print(f"Lengths: {tensor_desc.get_lengths()}")
        print(f"Element space size: {tensor_desc.get_element_space_size()}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_transform_descriptor():
    """Test creating a transform tensor descriptor."""
    print("\n=== Test 4: Transform Descriptor ===")
    parser = TensorTransformParser()
    
    desc_str = """
    transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(number<16>{})),
        make_tuple(make_merge_transform(make_tuple(number<4>{}, number<4>{}))),
        make_tuple(sequence<0>{}),
        make_tuple(sequence<0>{})
    )
    """
    variables = {}
    
    try:
        tensor_desc = parser.create_pytensor_descriptor(desc_str, variables)
        print(f"Created descriptor: {tensor_desc}")
        print(f"Number of dimensions: {tensor_desc.get_num_of_dimension()}")
        print(f"Number of transforms: {tensor_desc.get_num_of_transform()}")
        
        # Test the transforms
        transforms = tensor_desc.get_transforms()
        for i, transform in enumerate(transforms):
            print(f"Transform {i}: {transform}")
            
            # Test sympy methods if applicable
            if hasattr(transform, 'sympy_forward'):
                try:
                    if transform.__class__.__name__ == 'MergeTransform':
                        input_syms = [sp.Symbol(f'x{j}') for j in range(transform.get_num_of_lower_dimension())]
                        forward_result = transform.sympy_forward(input_syms)
                        print(f"  Forward: {input_syms} -> {forward_result}")
                except Exception as e:
                    print(f"  Error in sympy_forward: {e}")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_transforms()
    test_naive_descriptor()
    test_transform_descriptor()
    print("\n=== All tests completed ===") 