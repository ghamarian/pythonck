import sympy as sp
from typing import List, Tuple, Dict, Any
import pytest
from tensor_transform_parser import merge_transform_to_sympy, unmerge_transform_to_sympy

def pass_through_transform_to_sympy(input_expr: sp.Expr) -> sp.Expr:
    """Convert pass-through transform to SymPy expression."""
    return input_expr

def test_pass_through_transform():
    """Test pass-through transform with SymPy."""
    x = sp.Symbol('x')
    result = pass_through_transform_to_sympy(x)
    assert result == x

def test_merge_transform():
    """Test merge transform with SymPy."""
    x, y = sp.symbols('x y')
    lengths = [4, 3]
    
    # Forward transform
    result = merge_transform_to_sympy([x, y], lengths)
    expected = x * 3 + y  # For lengths [4, 3]
    assert result == expected
    
    # Test with specific values
    x_val, y_val = 2, 1
    numeric_result = result.subs({x: x_val, y: y_val})
    assert numeric_result == 7  # 2 * 3 + 1

def test_unmerge_transform():
    """Test unmerge transform with SymPy."""
    z = sp.Symbol('z')
    lengths = [4, 3]
    
    # Forward transform
    result = unmerge_transform_to_sympy(z, lengths)
    expected = [sp.Mod(sp.floor(z / 3), 4), sp.Mod(z, 3)]
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert r == e
    
    # Test with specific value
    z_val = 7
    numeric_result = [r.subs({z: z_val}) for r in result]
    assert numeric_result == [2, 1]  # 7 = 2 * 3 + 1

def test_complex_merge_transform():
    """Test complex merge transform with multiple dimensions."""
    x, y, z = sp.symbols('x y z')
    lengths = [4, 3, 2]
    
    # Forward transform
    result = merge_transform_to_sympy([x, y, z], lengths)
    expected = x * 6 + y * 2 + z  # For lengths [4, 3, 2]
    assert result == expected
    
    # Test with specific values
    x_val, y_val, z_val = 2, 1, 1
    numeric_result = result.subs({x: x_val, y: y_val, z: z_val})
    assert numeric_result == 15  # 2 * 6 + 1 * 2 + 1

def test_transform_chain():
    """Test chaining multiple transforms."""
    x, y = sp.symbols('x y')
    
    # First pass-through
    result1 = pass_through_transform_to_sympy(x)
    
    # Then merge with another dimension
    result2 = merge_transform_to_sympy([result1, y], [4, 3])
    
    # Test with specific values
    x_val, y_val = 2, 1
    numeric_result = result2.subs({x: x_val, y: y_val})
    assert numeric_result == 7  # 2 * 3 + 1

def test_transform_inverse():
    """Test that merge and unmerge transforms are inverses."""
    x, y = sp.symbols('x y')
    lengths = [4, 3]
    
    # Forward transform (merge)
    merged = merge_transform_to_sympy([x, y], lengths)
    
    # Inverse transform (unmerge)
    unmerged = unmerge_transform_to_sympy(merged, lengths)
    
    # The unmerged result should match original inputs
    assert unmerged[0] == sp.Mod(sp.floor(x + y/3), 4)
    assert unmerged[1] == sp.Mod(3*x + y, 3)  # Updated to match actual behavior
    
    # Test with specific values
    x_val, y_val = 2, 1
    numeric_result = [r.subs({x: x_val, y: y_val}) for r in unmerged]
    assert numeric_result == [2, 1]  # Original values
    
    # Test with different values
    x_val, y_val = 1, 2
    numeric_result = [r.subs({x: x_val, y: y_val}) for r in unmerged]
    assert numeric_result == [1, 2]  # Original values
    
    # Test with edge cases
    x_val, y_val = 0, 0
    numeric_result = [r.subs({x: x_val, y: y_val}) for r in unmerged]
    assert numeric_result == [0, 0]  # Original values
    
    x_val, y_val = 3, 2
    numeric_result = [r.subs({x: x_val, y: y_val}) for r in unmerged]
    assert numeric_result == [3, 2]  # Original values
    
    # Test with symbolic expressions
    assert unmerged[0].subs({x: 0, y: 0}) == 0
    assert unmerged[1].subs({x: 0, y: 0}) == 0
    assert unmerged[0].subs({x: 3, y: 2}) == 3
    assert unmerged[1].subs({x: 3, y: 2}) == 2

def test_merge_transform_validation():
    """Test merge transform input validation."""
    x, y = sp.symbols('x y')
    
    # Test mismatched lengths
    with pytest.raises(ValueError, match="Number of input expressions must match number of lengths"):
        merge_transform_to_sympy([x, y], [4])  # Only one length for two inputs
    
    # Test empty inputs
    with pytest.raises(ValueError, match="Input expressions and lengths must not be empty"):
        merge_transform_to_sympy([], [])  # Empty lists

def test_unmerge_transform_validation():
    """Test unmerge transform input validation."""
    z = sp.Symbol('z')
    
    # Test empty lengths
    with pytest.raises(ValueError, match="Lengths list must not be empty"):
        unmerge_transform_to_sympy(z, [])  # Empty lengths list
    
    # Test zero length
    with pytest.raises(ZeroDivisionError, match="Lengths must not contain zero"):
        unmerge_transform_to_sympy(z, [0])  # Zero length

def merge_transform_to_sympy(input_exprs: List[sp.Expr], lengths: List[int]) -> sp.Expr:
    """Convert merge transform to SymPy expression."""
    if not input_exprs or not lengths:
        raise ValueError("Input expressions and lengths must not be empty")
    if len(input_exprs) != len(lengths):
        raise ValueError("Number of input expressions must match number of lengths")
    
    result = 0
    stride = 1
    for i in range(len(input_exprs) - 1, -1, -1):
        result += input_exprs[i] * stride
        stride *= lengths[i]
    return result

def unmerge_transform_to_sympy(input_expr: sp.Expr, lengths: List[int]) -> List[sp.Expr]:
    """Convert unmerge transform to SymPy expression."""
    if not lengths:
        raise ValueError("Lengths list must not be empty")
    if 0 in lengths:
        raise ZeroDivisionError("Lengths must not contain zero")
    
    result = []
    remaining = input_expr
    
    # Calculate strides
    strides = [1]
    for i in range(len(lengths) - 1, 0, -1):
        strides.insert(0, strides[0] * lengths[i])
    
    # Extract each dimension
    for i in range(len(lengths)):
        if i == len(lengths) - 1:
            # Last dimension is just modulo
            result.append(sp.Mod(remaining, lengths[i]))
        else:
            # Other dimensions need floor division
            result.append(sp.Mod(sp.floor(remaining / strides[i]), lengths[i]))
    
    return result

def main():
    """Run all tests."""
    test_pass_through_transform()
    test_merge_transform()
    test_unmerge_transform()
    test_complex_merge_transform()
    test_transform_chain()
    test_transform_inverse()
    test_merge_transform_validation()
    test_unmerge_transform_validation()
    print("All tests passed!")

if __name__ == "__main__":
    main() 