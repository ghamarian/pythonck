import sympy as sp
from typing import List, Tuple, Dict, Any
import pytest
import sys
import os

# Add pytensor to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytensor.tensor_descriptor import (
    Transform, PassThroughTransform, MergeTransform, UnmergeTransform,
    EmbedTransform, OffsetTransform, PadTransform, ReplicateTransform, XorTransform
)

def test_pass_through_transform():
    """Test pass-through transform with SymPy."""
    x = sp.Symbol('x')
    transform = PassThroughTransform(length=4)
    
    # PassThrough is identity in both directions
    result = transform.sympy_calculate_lower([x])
    assert result[0] == x
    
    result = transform.sympy_calculate_upper([x])
    assert result[0] == x

def test_merge_transform():
    """Test merge transform with SymPy."""
    x, y = sp.symbols('x y')
    lengths = [4, 3]
    transform = MergeTransform(lengths=lengths)
    
    # Composition: multiple lower → single upper
    result = transform.sympy_calculate_upper([x, y])
    expected = x * 3 + y  # For lengths [4, 3]
    assert result[0] == expected
    
    # Test with specific values
    x_val, y_val = 2, 1
    numeric_result = result[0].subs({x: x_val, y: y_val})
    assert numeric_result == 7  # 2 * 3 + 1

def test_unmerge_transform():
    """Test unmerge transform with SymPy."""
    z = sp.Symbol('z')
    lengths = [4, 3]
    transform = UnmergeTransform(lengths=lengths)
    
    # Decomposition: single lower → multiple upper
    result = transform.sympy_calculate_upper([z])
    
    # Test with specific value
    z_val = 7  # This should give [2, 1] for lengths [4, 3]
    numeric_results = [expr.subs(z, z_val) for expr in result]
    expected = [2, 1]  # 7 = 2 * 3 + 1
    assert numeric_results == expected

def test_complex_merge_transform():
    """Test complex merge transform with multiple dimensions."""
    x, y, z = sp.symbols('x y z')
    lengths = [4, 3, 2]
    transform = MergeTransform(lengths=lengths)
    
    # Composition: multiple lower → single upper
    result = transform.sympy_calculate_upper([x, y, z])
    expected = x * 6 + y * 2 + z  # For lengths [4, 3, 2]
    assert result[0] == expected
    
    # Test with specific values
    x_val, y_val, z_val = 2, 1, 1
    numeric_result = result[0].subs({x: x_val, y: y_val, z: z_val})
    assert numeric_result == 15  # 2 * 6 + 1 * 2 + 1

def test_transform_chain():
    """Test chaining multiple transforms."""
    x, y = sp.symbols('x y')
    
    # First pass-through
    pass_through = PassThroughTransform(length=1)
    result1 = pass_through.sympy_calculate_lower([x])
    
    # Then merge with another dimension (composition: multiple → single)
    merge = MergeTransform(lengths=[4, 3])
    result2 = merge.sympy_calculate_upper([result1[0], y])
    
    # Test with specific values
    x_val, y_val = 2, 1
    numeric_result = result2[0].subs({x: x_val, y: y_val})
    assert numeric_result == 7  # 2 * 3 + 1

def test_transform_inverse():
    """Test that merge and unmerge transforms are inverses."""
    x_s, y_s = sp.symbols('x y')
    lengths = [4, 3]
    
    # Composition (merge): multiple → single
    merge = MergeTransform(lengths=lengths)
    merged_expr_list = merge.sympy_calculate_upper([x_s, y_s])
    
    z_s = sp.Symbol('z')
    # Decomposition (unmerge): single → multiple
    unmerge = UnmergeTransform(lengths=lengths)
    unmerged_exprs = unmerge.sympy_calculate_upper([z_s])
    
    # The unmerged result should match original inputs
    # Test with substitution
    for x_val in range(lengths[0]):
        for y_val in range(lengths[1]):
            # Calculate composition transform value
            subs_dict = {x_s: x_val, y_s: y_val}
            merged_val = merged_expr_list[0].subs(subs_dict)
            
            # Calculate decomposition transform value
            unmerged_vals = [expr.subs(z_s, merged_val) for expr in unmerged_exprs]
            
            # Should recover original values
            assert unmerged_vals == [x_val, y_val]

def test_embed_transform():
    """Test embed transform with SymPy."""
    x, y = sp.symbols('x y')
    lengths = [4, 3]
    strides = [3, 1]
    transform = EmbedTransform(lengths=lengths, strides=strides)
    
    # Composition: multiple upper → single lower (coordinates → offset)
    result = transform.sympy_calculate_lower([x, y])
    expected = x * 3 + y * 1
    assert result[0] == expected
    
    # Test with specific values
    x_val, y_val = 2, 1
    numeric_result = result[0].subs({x: x_val, y: y_val})
    assert numeric_result == 7  # 2 * 3 + 1
    
    # Decomposition: single lower → multiple upper (offset → coordinates)
    z = sp.Symbol('z')
    result = transform.sympy_calculate_upper([z])
    # Test with the offset we computed above
    coords = [expr.subs(z, 7) for expr in result]
    # The decomposition should give us back [2, 1]
    assert coords == [2, 1]

def test_offset_transform():
    """Test offset transform with SymPy."""
    x = sp.Symbol('x')
    transform = OffsetTransform(element_space_size=100, offset=10)
    
    # Forward: upper → lower (add offset)
    result = transform.sympy_calculate_lower([x])
    assert result[0] == x + 10
    
    # Backward: lower → upper (subtract offset)
    result = transform.sympy_calculate_upper([x])
    assert result[0] == x - 10

def test_pad_transform():
    """Test pad transform with SymPy."""
    x = sp.Symbol('x')
    transform = PadTransform(lower_length=8, left_pad=2, right_pad=3)
    
    # Forward: upper → lower (subtract left pad and clamp)
    result = transform.sympy_calculate_lower([x])
    # Should be Max(0, Min(x - 2, 7)) but simplified form depends on SymPy version
    assert sp.simplify(result[0].subs(x, 5)) == 3  # 5 - 2 = 3
    
    # Backward: lower → upper (add left pad)
    result = transform.sympy_calculate_upper([x])
    assert result[0] == x + 2

def test_replicate_transform_sympy():
    """Test replicate transform with SymPy."""
    transform = ReplicateTransform(upper_lengths=[8, 4])
    
    # Replicate: 0 inputs → multiple outputs (creates zeros)
    result = transform.sympy_calculate_upper([])
    assert len(result) == 2
    assert all(r == 0 for r in result)
    
    # Test legacy compatibility: 1 input → 0 outputs
    x = sp.Symbol('x')
    result = transform.sympy_calculate_upper([x])
    assert len(result) == 0

def test_unmerge_edge_cases():
    """Test edge cases for unmerge transform."""
    z = sp.Symbol('z')
    
    # Test with single length
    transform = UnmergeTransform(lengths=[4])
    result = transform.sympy_calculate_upper([z])
    assert len(result) == 1
    # Test with a numeric value
    numeric_result = result[0].subs(z, 3)
    assert numeric_result == 3  # For single dimension, should return input value
    
    # Test with very large lengths
    transform = UnmergeTransform(lengths=[1024, 32])
    result = transform.sympy_calculate_upper([z])
    assert len(result) == 2
    # Test with numeric values
    test_val = 1000
    numeric_results = [expr.subs(z, test_val) for expr in result]
    assert numeric_results[0] == test_val // 32  # 1000 // 32 = 31
    assert numeric_results[1] == test_val % 32   # 1000 % 32 = 8

def test_transform_composition():
    """Test composing multiple transforms together."""
    x = sp.Symbol('x')
    
    # Create a chain of transforms:
    # 1. Offset by 2
    # 2. Replicate
    # 3. Embed
    offset_transform = OffsetTransform(element_space_size=10, offset=2)
    replicate_transform = ReplicateTransform(upper_lengths=[2, 3])  # 2 dimensions
    embed_transform = EmbedTransform(lengths=[3], strides=[1])
    
    # Apply transforms in sequence
    result1 = offset_transform.sympy_calculate_lower([x])  # x + 2
    result2 = replicate_transform.sympy_calculate_upper([])  # [0, 0]
    result3 = embed_transform.sympy_calculate_lower([x])  # [x] coordinate → offset
    
    # Test with specific value
    x_val = 3
    numeric_result1 = result1[0].subs({x: x_val})
    assert numeric_result1 == 5  # 3 + 2
    assert len(result2) == 2  # Replicate produces zeros (one per dimension)
    assert all(r == 0 for r in result2)  # All zeros
    numeric_result3 = result3[0].subs({x: x_val})
    assert numeric_result3 == 3  # Simple embedding

def test_xor_transform():
    """Test XOR transform with SymPy."""
    x, y = sp.symbols('x y')
    transform = XorTransform(lengths=[8, 4])
    
    # Forward: upper → lower (apply XOR)
    result = transform.sympy_calculate_lower([x, y])
    # First dimension passes through, second gets XORed
    assert result[0] == x
    # Result[1] should be Xor(y, x % 4) but we'll test the evaluation
    
    # Test with specific values
    x_val, y_val = 5, 2
    from pytensor.tensor_descriptor import Xor
    expected_y = Xor(y_val, x_val % 4)  # Xor(2, 5 % 4) = Xor(2, 1) = 3
    numeric_results = [expr.subs({x: x_val, y: y_val}) for expr in result]
    assert numeric_results[0] == x_val
    assert numeric_results[1] == 3  # Xor(2, 1) = 3 

def test_edge_cases():
    """Test edge cases for various transforms."""
    x = sp.Symbol('x', integer=True)
    z = sp.Symbol('z')
    
    # Test with single length
    transform = UnmergeTransform(lengths=[4])
    result = transform.sympy_calculate_upper([z])
    assert len(result) == 1
    # Test with numeric value instead of symbolic
    numeric_result = result[0].subs(z, 3)
    assert numeric_result == 3  # For single dimension, should return input value
    
    # Test with very large lengths
    transform = UnmergeTransform(lengths=[1024, 32])
    result = transform.sympy_calculate_upper([z])
    assert len(result) == 2
    # Test with numeric values
    test_val = 1000
    numeric_results = [expr.subs(z, test_val) for expr in result]
    assert numeric_results[0] == test_val // 32  # 1000 // 32 = 31
    assert numeric_results[1] == test_val % 32   # 1000 % 32 = 8

def test_complex_chaining():
    """Test complex chaining of different transforms."""
    x = sp.Symbol('x')
    
    # Create chain of transforms
    offset_transform = OffsetTransform(element_space_size=10, offset=2)
    replicate_transform = ReplicateTransform(upper_lengths=[3])
    embed_transform = EmbedTransform(lengths=[4], strides=[1])
    
    # Apply transforms in sequence
    result1 = offset_transform.sympy_calculate_lower([x])  # x + 2
    result2 = replicate_transform.sympy_calculate_upper([])  # []
    result3 = embed_transform.sympy_calculate_lower([x])  # [x]
    
    # Test with specific value
    x_val = 3
    numeric_result1 = result1[0].subs({x: x_val})
    assert numeric_result1 == 5  # 3 + 2
    
    numeric_result3 = result3[0].subs({x: x_val})
    assert numeric_result3 == 3  # Simple embedding 