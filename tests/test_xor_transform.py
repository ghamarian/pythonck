#!/usr/bin/env python3

import pytest
from pytensor.tensor_descriptor import XorTransform, MultiIndex
import sympy as sp

class TestXorTransform:
    """Comprehensive test suite for XOR transform."""
    
    def test_basic_functionality_with_modulo(self):
        """Test XOR with modulo=True."""
        xor1 = XorTransform([4, 8], apply_modulo=True)
        upper = MultiIndex(2, [2, 5])
        lower = xor1.calculate_lower_index(upper)
        upper_back = xor1.calculate_upper_index(lower)
        
        assert upper._values == [2, 5]
        assert lower._values == [2, 7]  # 5 ^ (2 % 8) = 5 ^ 2 = 7
        assert upper_back._values == [2, 5]
        assert upper._values == upper_back._values
    
    def test_basic_functionality_without_modulo(self):
        """Test XOR with modulo=False."""
        xor2 = XorTransform([4, 8], apply_modulo=False)
        upper = MultiIndex(2, [2, 5])
        lower2 = xor2.calculate_lower_index(upper)
        upper_back2 = xor2.calculate_upper_index(lower2)
        
        assert upper._values == [2, 5]
        assert lower2._values == [2, 7]  # 5 ^ 2 = 7
        assert upper_back2._values == [2, 5]
        assert upper._values == upper_back2._values
    
    def test_sympy_expressions(self):
        """Test SymPy expressions for forward and backward directions."""
        xor1 = XorTransform([4, 8], apply_modulo=True)
        x, y = sp.Symbol('x'), sp.Symbol('y')
        
        # Test basic sympy operations
        forward = xor1.sympy_calculate_lower([x, y])  # upper → lower
        backward = xor1.sympy_calculate_upper([x, y])  # lower → upper (same for XOR)
        
        # Both should have same mathematical form but different semantic meaning
        assert len(forward) == 2
        assert len(backward) == 2
        assert forward[0] == x  # First dimension passes through
        assert backward[0] == x  # First dimension passes through
        
        # Second dimension should be XOR with modulo
        from pytensor.tensor_descriptor import XorFunction
        expected_expr = XorFunction(y, sp.Mod(x, 8))
        assert forward[1] == expected_expr
        assert backward[1] == expected_expr
    
    def test_cpp_equivalence_verification(self):
        """Test that Python implementation matches C++ CalculateLowerIndex."""
        xor1 = XorTransform([4, 8], apply_modulo=True)
        test_upper = [2, 5]
        upper_idx = MultiIndex(2, test_upper)
        lower_idx = xor1.calculate_lower_index(upper_idx)
        
        # Manual calculation matching C++ code:
        # idx_low[0] = idx_up[0]
        # idx_low[1] = idx_up[1] ^ (idx_up[0] % lengths[1])
        manual_lower_0 = test_upper[0]  
        manual_lower_1 = test_upper[1] ^ (test_upper[0] % 8)  
        
        assert lower_idx._values == [manual_lower_0, manual_lower_1]
        assert lower_idx._values == [2, 7]
    
    def test_edge_cases(self):
        """Test various edge cases for round-trip consistency."""
        xor1 = XorTransform([4, 8], apply_modulo=True)
        test_cases = [
            [0, 0], [1, 1], [3, 7], [0, 7], [3, 0]
        ]
        
        for test_upper in test_cases:
            upper_idx = MultiIndex(2, test_upper)
            lower_idx = xor1.calculate_lower_index(upper_idx)
            upper_back_idx = xor1.calculate_upper_index(lower_idx)
            
            assert test_upper == upper_back_idx._values, f"Round-trip failed for {test_upper}"
    
    def test_modulo_vs_no_modulo_comparison(self):
        """Test comparison between modulo and non-modulo modes."""
        xor_with_mod = XorTransform([4, 8], apply_modulo=True)
        xor_without_mod = XorTransform([4, 8], apply_modulo=False)
        
        test_upper = [3, 6]
        upper_idx = MultiIndex(2, test_upper)
        
        lower_with_mod = xor_with_mod.calculate_lower_index(upper_idx)
        lower_without_mod = xor_without_mod.calculate_lower_index(upper_idx)
        
        # Manual verification
        with_mod_expected = [3, 6 ^ (3 % 8)]  # 6 ^ 3 = 5
        without_mod_expected = [3, 6 ^ 3]     # 6 ^ 3 = 5  
        
        assert lower_with_mod._values == with_mod_expected
        assert lower_without_mod._values == without_mod_expected
        
        # In this case they should be the same since 3 % 8 = 3
        assert lower_with_mod._values == lower_without_mod._values
    
    def test_invalid_dimensions(self):
        """Test that XOR transform validates dimension requirements."""
        with pytest.raises(ValueError, match="XOR transform requires exactly 2 dimensions"):
            XorTransform([4])  # Only 1 dimension
        
        with pytest.raises(ValueError, match="XOR transform requires exactly 2 dimensions"):
            XorTransform([4, 8, 16])  # 3 dimensions
        
        xor = XorTransform([4, 8])
        
        with pytest.raises(ValueError, match="XOR transform expects 2D upper index"):
            xor.calculate_lower_index(MultiIndex(1, [5]))  # 1D input
        
        with pytest.raises(ValueError, match="XOR transform expects 2D lower index"):
            xor.calculate_upper_index(MultiIndex(3, [1, 2, 3]))  # 3D input 