import sympy as sp
from typing import List, Tuple, Dict, Any
import pytest
import sys
import os

# Add pytensor to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytensor.tensor_descriptor import (
    Transform, PassThroughTransform, MergeTransform, UnmergeTransform,
    EmbedTransform, OffsetTransform, PadTransform, ReplicateTransform
)

def test_pass_through_transform():
    """Test pass-through transform with SymPy."""
    x = sp.Symbol('x')
    transform = PassThroughTransform(length=1)
    result = transform.sympy_forward([x])
    assert result[0] == x

def test_merge_transform():
    """Test merge transform with SymPy."""
    x, y = sp.symbols('x y')
    lengths = [4, 3]
    transform = MergeTransform(lengths=lengths)
    
    # Forward transform
    result = transform.sympy_forward([x, y])
    expected = x * 3 + y  # For lengths [4, 3]
    assert result[0] == expected
    
    # Test with specific values
    x_val, y_val = 2, 1
    numeric_result = result[0].subs({x: x_val, y: y_val})
    assert numeric_result == 7  # 2 * 3 + 1

def test_unmerge_transform():
    """Test unmerge transform with SymPy."""
    z = sp.Symbol('z', integer=True)
    lengths = [4, 3]
    transform = UnmergeTransform(lengths=lengths)
    
    # Backward transform
    result = transform.sympy_backward([z])
    # For lengths [4, 3], we expect:
    # x = z // 3
    # y = z % 3
    expected = [z // 3, z % 3]
    assert len(result) == len(expected)
    assert sp.simplify(result[0] - expected[0]) == 0
    assert sp.simplify(result[1] - expected[1]) == 0
    
    # Test with specific value
    z_val = 7
    numeric_result = [r.subs({z: z_val}) for r in result]
    assert numeric_result == [2, 1]  # 7 = 2 * 3 + 1

def test_complex_merge_transform():
    """Test complex merge transform with multiple dimensions."""
    x, y, z = sp.symbols('x y z')
    lengths = [4, 3, 2]
    transform = MergeTransform(lengths=lengths)
    
    # Forward transform
    result = transform.sympy_forward([x, y, z])
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
    result1 = pass_through.sympy_forward([x])
    
    # Then merge with another dimension
    merge = MergeTransform(lengths=[4, 3])
    result2 = merge.sympy_forward([result1[0], y])
    
    # Test with specific values
    x_val, y_val = 2, 1
    numeric_result = result2[0].subs({x: x_val, y: y_val})
    assert numeric_result == 7  # 2 * 3 + 1

def test_transform_inverse():
    """Test that merge and unmerge transforms are inverses."""
    x_s, y_s = sp.symbols('x y')
    lengths = [4, 3]
    
    # Forward transform (merge)
    merge = MergeTransform(lengths=lengths)
    merged_expr_list = merge.sympy_forward([x_s, y_s])
    
    z_s = sp.Symbol('z')
    # Inverse transform (unmerge)
    unmerge = UnmergeTransform(lengths=lengths)
    unmerged_exprs = unmerge.sympy_backward([z_s])
    
    # The unmerged result should match original inputs
    # Test with substitution
    for x_val in range(lengths[0]):
        for y_val in range(lengths[1]):
            # Calculate forward transform value
            subs_dict = {x_s: x_val, y_s: y_val}
            merged_val = merged_expr_list[0].subs(subs_dict)
            
            # Calculate backward transform with the result
            unmerged_vals = [e.subs({z_s: merged_val}) for e in unmerged_exprs]

            assert unmerged_vals[0] == x_val
            assert unmerged_vals[1] == y_val 

def test_complex_unmerge_transform():
    """Test unmerge transform with three dimensions."""
    z = sp.Symbol('z', integer=True)
    lengths = [4, 3, 2]  # Three dimensions
    transform = UnmergeTransform(lengths=lengths)
    
    # Backward transform
    result = transform.sympy_backward([z])
    # For lengths [4, 3, 2], we expect:
    # x = z // (3 * 2)
    # y = (z % (3 * 2)) // 2
    # w = z % 2
    expected = [z // 6, (z % 6) // 2, z % 2]
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert sp.simplify(r - e) == 0
    
    # Test with specific value
    z_val = 15  # 15 = 2 * 6 + 1 * 2 + 1
    numeric_result = [r.subs({z: z_val}) for r in result]
    assert numeric_result == [2, 1, 1]

def test_embed_transform_sympy():
    """Test embed transform with SymPy expressions."""
    x = sp.Symbol('x')
    transform = EmbedTransform(lengths=[4], strides=[1])
    
    # Forward transform
    result = transform.sympy_forward([x])
    assert len(result) == 1
    assert result[0] == x  # Simple case with stride 1
    
    # Test with specific value
    x_val = 2
    numeric_result = [r.subs({x: x_val}) for r in result]
    assert numeric_result == [2]

def test_offset_transform_sympy():
    """Test offset transform with SymPy expressions."""
    x = sp.Symbol('x')
    transform = OffsetTransform(element_space_size=10, offset=5)
    
    # Forward transform
    result = transform.sympy_forward([x])
    assert result[0] == x + 5
    
    # Test with specific value
    x_val = 3
    numeric_result = result[0].subs({x: x_val})
    assert numeric_result == 8  # 3 + 5

def test_pad_transform_sympy():
    """Test pad transform with SymPy expressions."""
    x = sp.Symbol('x')
    transform = PadTransform(lower_length=4, left_pad=1, right_pad=1)
    
    # Forward transform
    result = transform.sympy_forward([x])
    # Should clamp to valid range [0, lower_length-1]
    assert result[0] == sp.Max(0, sp.Min(x - 1, 3))  # -1 for left pad, clamp to [0,3]
    
    # Test with specific values
    x_val = 2  # 2 - 1 = 1, which is in valid range
    numeric_result = result[0].subs({x: x_val})
    assert numeric_result == 1

def test_replicate_transform_sympy():
    """Test replicate transform with SymPy expressions."""
    transform = ReplicateTransform(upper_lengths=[3])
    
    # Forward transform (no input symbols)
    result = transform.sympy_forward([])
    assert len(result) == 0  # Replicate has no input symbols
    
    # Backward transform
    result = transform.sympy_backward([sp.Symbol('x')])
    assert len(result) == 0  # Replicate has no output symbols

def test_unmerge_edge_cases():
    """Test unmerge transform with edge cases."""
    z = sp.Symbol('z', integer=True)
    
    # Test with single length
    transform = UnmergeTransform(lengths=[4])
    result = transform.sympy_backward([z])
    assert len(result) == 1
    assert result[0] == z
    
    # Test with very large lengths
    transform = UnmergeTransform(lengths=[1024, 32])
    result = transform.sympy_backward([z])
    assert len(result) == 2
    assert sp.simplify(result[0] - z // 32) == 0
    assert sp.simplify(result[1] - z % 32) == 0

def test_transform_composition():
    """Test composing multiple transforms together."""
    x = sp.Symbol('x')
    
    # Create a chain of transforms:
    # 1. Offset by 2
    # 2. Replicate
    # 3. Embed
    offset_transform = OffsetTransform(element_space_size=10, offset=2)
    replicate_transform = ReplicateTransform(upper_lengths=[2])
    embed_transform = EmbedTransform(lengths=[3], strides=[1])
    
    # Apply transforms in sequence
    result1 = offset_transform.sympy_forward([x])  # x + 2
    result2 = replicate_transform.sympy_forward([])  # []
    result3 = embed_transform.sympy_forward([x])  # [x]
    
    # Test with specific value
    x_val = 3
    numeric_result1 = result1[0].subs({x: x_val})
    assert numeric_result1 == 5  # 3 + 2
    assert len(result2) == 0  # Replicate has no output
    numeric_result3 = result3[0].subs({x: x_val})
    assert numeric_result3 == 3  # Simple embedding 