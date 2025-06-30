"""
Test script for pytensor integration with the parser.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_transforms import TensorTransformParser
from pytensor.tensor_descriptor import UnmergeTransform, MergeTransform, PassThroughTransform
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
    forward_result = transform.sympy_calculate_lower(input_syms)
    print(f"Forward result: {forward_result}")
    
    backward_result = transform.sympy_calculate_upper(forward_result)
    print(f"Backward result: {backward_result}")
    
    # Test PassThrough Transform
    input_syms = [sp.Symbol('x')]
    forward_result = transform.sympy_calculate_lower(input_syms)
    print(f"  Forward result: {forward_result}")
    
    backward_result = transform.sympy_calculate_upper(forward_result)
    print(f"  Backward result: {backward_result}")
    
    # For passthrough, forward and backward should be identity
    assert forward_result == input_syms
    assert backward_result == input_syms
    
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
    
    # Test with multiple inputs
    input_syms = [sp.Symbol('x'), sp.Symbol('y')]
    forward_result = transform.sympy_calculate_upper(input_syms)
    print(f"  Forward result: {forward_result}")
    
    backward_result = transform.sympy_calculate_lower(forward_result)
    print(f"  Backward result: {backward_result}")
    
    # Test with specific values
    test_vals = {sp.Symbol('x'): 2, sp.Symbol('y'): 3}
    forward_val = forward_result[0].subs(test_vals)
    backward_vals = [expr.subs(test_vals) for expr in backward_result]
    print(f"  Test values x=2, y=3:")
    print(f"    Forward: {forward_val}")
    print(f"    Backward: {backward_vals}")
    
    # Test UnmergeTransform too  
    unmerge_transform = UnmergeTransform(lengths=[4, 8])
    multiple_inputs = [sp.Symbol('a'), sp.Symbol('b')]  # UnmergeTransform needs multiple inputs for calculate_lower  
    unmerge_forward = unmerge_transform.sympy_calculate_lower(multiple_inputs)
    print(f"  Unmerge forward: {unmerge_forward}")
    
    print("\n=== Testing all transform types ===")
    transforms_to_test = [
        PassThroughTransform(length=5),
        MergeTransform(lengths=[2, 3]),
        UnmergeTransform(lengths=[2, 3]),
        # Add more transforms as needed
    ]
    
    for transform in transforms_to_test:
        print(f"\nTesting {transform}")
        try:
            if hasattr(transform, 'sympy_calculate_lower'):
                # Determine appropriate input based on transform type
                if isinstance(transform, MergeTransform):
                    input_syms = [sp.Symbol(f'x_{i}') for i in range(len(transform.lengths))]
                    forward_result = transform.sympy_calculate_upper(input_syms)
                    print(f"  Forward result: {forward_result}")
                else:
                    input_syms = [sp.Symbol('x')]
                    forward_result = transform.sympy_calculate_lower(input_syms)
                    print(f"  Forward result: {forward_result}")
            else:
                print(f"  Transform {transform} doesn't have sympy methods")
        except Exception as e:
            print(f"  Error in testing transform: {e}")

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
            if hasattr(transform, 'sympy_calculate_lower'):
                try:
                    if transform.__class__.__name__ == 'MergeTransform':
                        input_syms = [sp.Symbol(f'x{j}') for j in range(transform.get_num_of_lower_dimension())]
                        forward_result = transform.sympy_calculate_upper(input_syms)
                        print(f"  Forward result: {forward_result}")
                except Exception as e:
                    print(f"  Error in sympy_calculate_upper: {e}")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_transforms()
    test_naive_descriptor()
    test_transform_descriptor()
    print("\n=== All tests completed ===") 