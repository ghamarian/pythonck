"""
Test file for analyzing a complex merge transformation example.

This test examines whether the combined formulas correctly handle the dimension groupings
specified in the tensor descriptor transformation.
"""

import pytest
from tensor_transforms import TensorTransformParser
from tensor_transform_app import build_combined_formula, build_combined_backward_formula
import sympy as sp


def test_complex_merge_example():
    """Test the specific complex merge example to verify formula correctness."""
    
    # The tensor descriptor transformation from the user
    descriptor_code = """
    transform_tensor_descriptor(
        k_lds_block_desc_0,
        make_tuple(
            make_merge_transform(make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
            make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                            number<kKVector / kKPack>{},
                                            number<kKPack>{}))),
        make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
        make_tuple(sequence<0>{}, sequence<1>{})
    )
    """
    
    # Set up variables
    variables = {
        'NumKLdsBuffers': 4,
        'kNPerBlock': 32,
        'kKPerBlock': 64,
        'kKVector': 8,
        'kKPack': 2
    }
    
    # Parse the descriptor
    parser = TensorTransformParser()
    descriptor = parser.parse_tensor_descriptor(descriptor_code)
    
    print("=== PARSED DESCRIPTOR ===")
    print(f"Input descriptor: {descriptor['input_descriptor']}")
    print(f"Number of transforms: {len(descriptor['transforms'])}")
    print(f"Lower dimensions: {descriptor['lower_dimensions']}")
    print(f"Upper dimensions: {descriptor['upper_dimensions']}")
    print()
    
    # Analyze the transforms
    transforms = descriptor['transforms']
    lower_dims = descriptor['lower_dimensions']
    
    print("=== TRANSFORM ANALYSIS ===")
    for i, (transform, dims) in enumerate(zip(transforms, lower_dims)):
        print(f"Transform {i}:")
        print(f"  Type: {transform['type']}")
        print(f"  Input dimensions: {dims}")
        if transform['type'] == 'merge':
            print(f"  Merge values: {len(transform['values'])}")
            for j, val in enumerate(transform['values']):
                if hasattr(val['value'], 'subs'):
                    substituted = val['value'].subs(variables)
                    print(f"    Value {j}: {val['value']} = {substituted}")
                else:
                    print(f"    Value {j}: {val['value']}")
        print()
    
    # Generate combined formulas
    print("=== COMBINED FORMULAS ===")
    try:
        # Forward formula
        combined_expr, input_symbols = build_combined_formula(transforms, lower_dims, variables)
        print("Forward transformation:")
        print(f"  Input symbols: {input_symbols}")
        print(f"  Combined expression: {combined_expr}")
        print(f"  LaTeX: {sp.latex(combined_expr)}")
        print()
        
        # Let's break down each output dimension
        if hasattr(combined_expr, 'args') and len(combined_expr.args) > 0:
            print("Individual output dimensions:")
            for i, expr in enumerate(combined_expr.args):
                print(f"  y_{i} = {expr}")
                print(f"  LaTeX: {sp.latex(expr)}")
            print()
        
        # Backward formula
        backward_exprs, input_symbols, output_symbols = build_combined_backward_formula(transforms, lower_dims, variables)
        print("Backward transformation:")
        print(f"  Output symbols: {output_symbols}")
        print(f"  Input expressions: {backward_exprs}")
        print()
        
        print("Individual backward dimensions:")
        for i, expr in enumerate(backward_exprs):
            if i < len(input_symbols):
                print(f"  {input_symbols[i]} = {expr}")
                print(f"  LaTeX: {sp.latex(expr)}")
        print()
        
    except Exception as e:
        print(f"Error generating formulas: {e}")
        import traceback
        traceback.print_exc()
    
    # Test specific expectations
    print("=== VERIFICATION ===")
    
    # Expected behavior based on the sequences:
    # - Transform 0 should use dimensions 0 and 3: d_0 and d_3
    # - Transform 1 should use dimensions 1, 2, and 4: d_1, d_2, and d_4
    
    # Check that input symbols match expected dimensions
    expected_input_dims = {0, 1, 2, 3, 4}  # From sequences
    actual_input_dims = set()
    for dims in lower_dims:
        actual_input_dims.update(dims)
    
    print(f"Expected input dimensions: {sorted(expected_input_dims)}")
    print(f"Actual input dimensions: {sorted(actual_input_dims)}")
    assert actual_input_dims == expected_input_dims, f"Input dimensions mismatch: expected {expected_input_dims}, got {actual_input_dims}"
    
    # Check that we have 2 output dimensions (2 transforms)
    assert len(transforms) == 2, f"Expected 2 transforms, got {len(transforms)}"
    
    # Check that lower dimensions are grouped correctly
    assert lower_dims[0] == [0, 3], f"Transform 0 should use dimensions [0, 3], got {lower_dims[0]}"
    assert lower_dims[1] == [1, 2, 4], f"Transform 1 should use dimensions [1, 2, 4], got {lower_dims[1]}"
    
    print("✓ All verifications passed!")
    print()
    
    # Manual formula verification
    print("=== MANUAL FORMULA VERIFICATION ===")
    
    # For transform 0: merge(d_0, d_3) with lengths (NumKLdsBuffers=4, kNPerBlock=32)
    # Should be: d_0 * 32 + d_3
    expected_y0 = sp.Symbol('d_0') * 32 + sp.Symbol('d_3')
    print(f"Expected y_0: {expected_y0}")
    
    # For transform 1: merge(d_1, d_2, d_4) with lengths (kKPerBlock/kKVector=8, kKVector/kKPack=4, kKPack=2)
    # Should be: d_1 * (4*2) + d_2 * 2 + d_4 = d_1 * 8 + d_2 * 2 + d_4
    expected_y1 = sp.Symbol('d_1') * 8 + sp.Symbol('d_2') * 2 + sp.Symbol('d_4')
    print(f"Expected y_1: {expected_y1}")
    
    if hasattr(combined_expr, 'args') and len(combined_expr.args) >= 2:
        actual_y0 = combined_expr.args[0]
        actual_y1 = combined_expr.args[1]
        
        print(f"Actual y_0: {actual_y0}")
        print(f"Actual y_1: {actual_y1}")
        
        # Check if they match (allowing for equivalent expressions)
        diff_y0 = sp.simplify(expected_y0 - actual_y0)
        diff_y1 = sp.simplify(expected_y1 - actual_y1)
        
        print(f"Difference y_0: {diff_y0}")
        print(f"Difference y_1: {diff_y1}")
        
        if diff_y0 == 0:
            print("✓ y_0 formula matches expected!")
        else:
            print("✗ y_0 formula does NOT match expected!")
            
        if diff_y1 == 0:
            print("✓ y_1 formula matches expected!")
        else:
            print("✗ y_1 formula does NOT match expected!")


def test_with_specific_values():
    """Test the formulas with specific input values to verify correctness."""
    print("\n=== TESTING WITH SPECIFIC VALUES ===")
    
    # Test with specific input values
    test_inputs = {
        'd_0': 1,
        'd_1': 2, 
        'd_2': 3,
        'd_3': 4,
        'd_4': 5
    }
    
    variables = {
        'NumKLdsBuffers': 4,
        'kNPerBlock': 32,
        'kKPerBlock': 64,
        'kKVector': 8,
        'kKPack': 2
    }
    
    # Expected outputs using manual calculation
    expected_y0 = test_inputs['d_0'] * 32 + test_inputs['d_3']  # 1*32 + 4 = 36
    expected_y1 = test_inputs['d_1'] * 8 + test_inputs['d_2'] * 2 + test_inputs['d_4']  # 2*8 + 3*2 + 5 = 27
    
    print(f"Input values: {test_inputs}")
    print(f"Expected y_0: {expected_y0}")
    print(f"Expected y_1: {expected_y1}")
    
    # Parse and compute using our implementation
    descriptor_code = """
    transform_tensor_descriptor(
        k_lds_block_desc_0,
        make_tuple(
            make_merge_transform(make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
            make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                            number<kKVector / kKPack>{},
                                            number<kKPack>{}))),
        make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
        make_tuple(sequence<0>{}, sequence<1>{})
    )
    """
    
    parser = TensorTransformParser()
    descriptor = parser.parse_tensor_descriptor(descriptor_code)
    
    transforms = descriptor['transforms']
    lower_dims = descriptor['lower_dimensions']
    
    combined_expr, input_symbols = build_combined_formula(transforms, lower_dims, variables)
    
    if hasattr(combined_expr, 'args') and len(combined_expr.args) >= 2:
        actual_y0 = combined_expr.args[0].subs(test_inputs)
        actual_y1 = combined_expr.args[1].subs(test_inputs)
        
        print(f"Computed y_0: {actual_y0}")
        print(f"Computed y_1: {actual_y1}")
        
        assert actual_y0 == expected_y0, f"y_0 mismatch: expected {expected_y0}, got {actual_y0}"
        assert actual_y1 == expected_y1, f"y_1 mismatch: expected {expected_y1}, got {actual_y1}"
        
        print("✓ All value tests passed!")


if __name__ == "__main__":
    test_complex_merge_example()
    test_with_specific_values() 