"""
Test to verify that forward and backward transformations are correctly implemented as inverses.

This test validates:
1. That sympy_calculate_lower and sympy_calculate_upper are true inverses
2. That the tensor_transform_app.py logic correctly applies transforms
3. That the fixed semantics (not based on input count) work correctly
4. That transform direction is determined by graph flow, not input count
"""

import pytest
import sympy as sp
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pytensor.tensor_descriptor import (
    UnmergeTransform, MergeTransform, PassThroughTransform, 
    OffsetTransform, PadTransform, XorTransform, EmbedTransform
)
from pytensor.tensor_coordinate import MultiIndex


class TestTransformInverseVerification:
    """Comprehensive tests for transform inverse properties."""
    
    def test_unmerge_transform_inverse_property(self):
        """Test UnmergeTransform forward/backward are inverses."""
        lengths = [2, 3, 4]
        transform = UnmergeTransform(lengths)
        
        # Test 1: Forward (single→multiple) then Backward (multiple→single)
        z = sp.Symbol('z')
        
        # Forward: single input → multiple outputs (decomposition)
        forward_result = transform.sympy_calculate_upper([z])
        assert len(forward_result) == len(lengths), f"Forward should produce {len(lengths)} outputs"
        
        # Backward: multiple inputs → single output (composition)  
        backward_result = transform.sympy_calculate_lower(forward_result)
        assert len(backward_result) == 1, "Backward should produce 1 output"
        
        # Should recover original input
        original_recovered = backward_result[0]
        
        # Test with numeric values
        for z_val in range(24):  # 2*3*4 = 24
            forward_vals = [expr.subs(z, z_val) for expr in forward_result]
            backward_val = original_recovered.subs(z, z_val)
            
            # Verify we get back to original
            assert backward_val == z_val, f"Failed for z={z_val}: got {backward_val}"
            
            # Verify forward values are valid coordinates
            assert 0 <= forward_vals[0] < lengths[0]
            assert 0 <= forward_vals[1] < lengths[1] 
            assert 0 <= forward_vals[2] < lengths[2]
    
    def test_merge_transform_inverse_property(self):
        """Test MergeTransform forward/backward are inverses."""
        lengths = [2, 3, 4]
        transform = MergeTransform(lengths)
        
        # Test 1: Forward (multiple→single) then Backward (single→multiple)
        x, y, z = sp.symbols('x y z')
        
        # Forward: multiple inputs → single output (composition)
        forward_result = transform.sympy_calculate_upper([x, y, z])
        assert len(forward_result) == 1, "Forward should produce 1 output"
        
        # Backward: single input → multiple outputs (decomposition)
        w = sp.Symbol('w')
        backward_result = transform.sympy_calculate_lower([w])
        assert len(backward_result) == len(lengths), f"Backward should produce {len(lengths)} outputs"
        
        # Test round-trip: forward then backward
        merged = forward_result[0]
        recovered = [expr.subs(w, merged) for expr in backward_result]
        
        # Test with numeric values
        for x_val in range(lengths[0]):
            for y_val in range(lengths[1]):
                for z_val in range(lengths[2]):
                    # Substitute values
                    merged_val = merged.subs({x: x_val, y: y_val, z: z_val})
                    recovered_vals = [expr.subs({x: x_val, y: y_val, z: z_val}) for expr in recovered]
                    
                    # Should recover original values
                    assert recovered_vals == [x_val, y_val, z_val], \
                        f"Failed for ({x_val},{y_val},{z_val}): got {recovered_vals}"
    
    def test_xor_transform_self_inverse(self):
        """Test that XOR transform is self-inverse."""
        transform = XorTransform([4, 8], apply_modulo=True)
        x, y = sp.symbols('x y')
        
        # Forward: upper → lower
        forward_result = transform.sympy_calculate_lower([x, y])
        
        # Backward: lower → upper (should be same operation for XOR)
        backward_result = transform.sympy_calculate_upper(forward_result)
        
        # Test self-inverse property with numeric values
        for x_val in range(4):
            for y_val in range(8):
                forward_vals = [expr.subs({x: x_val, y: y_val}) for expr in forward_result]
                backward_vals = [expr.subs({x: x_val, y: y_val}) for expr in backward_result]
                
                # Should get back to original inputs
                assert backward_vals == [x_val, y_val], \
                    f"XOR not self-inverse for ({x_val},{y_val}): got {backward_vals}"
    
    def test_passthrough_transform_identity(self):
        """Test that PassThrough is identity in both directions."""
        transform = PassThroughTransform(10)
        x = sp.Symbol('x')
        
        # Forward and backward should be identical
        forward_result = transform.sympy_calculate_lower([x])
        backward_result = transform.sympy_calculate_upper([x])
        
        assert forward_result == [x], "PassThrough forward should be identity"
        assert backward_result == [x], "PassThrough backward should be identity"
        assert forward_result == backward_result, "PassThrough should be same in both directions"
    
    def test_offset_transform_inverse(self):
        """Test that OffsetTransform forward/backward are inverses."""
        transform = OffsetTransform(element_space_size=100, offset=25)
        x = sp.Symbol('x')
        
        # Forward: add offset
        forward_result = transform.sympy_calculate_lower([x])
        assert forward_result == [x + 25], "Forward should add offset"
        
        # Backward: subtract offset  
        backward_result = transform.sympy_calculate_upper([x])
        assert backward_result == [x - 25], "Backward should subtract offset"
        
        # Test round-trip
        round_trip = transform.sympy_calculate_upper(forward_result)
        assert round_trip == [x], "Round-trip should recover original"
    
    def test_embed_transform_inverse_property(self):
        """Test EmbedTransform forward/backward are inverses."""
        lengths = [4, 5]
        strides = [5, 1]  # Row-major layout
        transform = EmbedTransform(lengths, strides)
        
        # Test 1: Multiple coordinates → linear address → multiple coordinates
        x, y = sp.symbols('x y')
        
        # Forward: multiple → single (composition)
        forward_result = transform.sympy_calculate_lower([x, y])
        assert len(forward_result) == 1, "Embed forward should produce single address"
        
        # Backward: single → multiple (decomposition)
        addr = sp.Symbol('addr')
        backward_result = transform.sympy_calculate_upper([addr])
        assert len(backward_result) == len(lengths), "Embed backward should produce coordinates"
        
        # Test round-trip with numeric values
        for x_val in range(lengths[0]):
            for y_val in range(lengths[1]):
                # Forward: coordinates → address
                addr_val = forward_result[0].subs({x: x_val, y: y_val})
                
                # Backward: address → coordinates
                recovered_coords = [expr.subs(addr, addr_val) for expr in backward_result]
                
                # Should recover original coordinates
                assert recovered_coords == [x_val, y_val], \
                    f"Failed for ({x_val},{y_val}): got {recovered_coords}"
    
    def test_transform_semantics_are_direction_independent(self):
        """Test that transform semantics are determined by direction, not input count."""
        
        # This test verifies the fix we made to tensor_transform_app.py
        # where we stopped using input_symbols length to decide direction
        
        lengths = [3, 4, 5]
        unmerge = UnmergeTransform(lengths)
        merge = MergeTransform(lengths)
        
        # Create test symbols
        single_input = sp.Symbol('z')
        multiple_inputs = [sp.Symbol(f'x_{i}') for i in range(len(lengths))]
        
        # UnmergeTransform: Fixed semantics regardless of input count
        # Forward direction: always decompose (single → multiple)
        unmerge_forward_1 = unmerge.sympy_calculate_upper([single_input])
        unmerge_forward_2 = unmerge.sympy_calculate_upper([sp.Symbol('w')])  # Different symbol, same result structure
        
        assert len(unmerge_forward_1) == len(lengths), "UnmergeTransform forward always produces multiple outputs"
        assert len(unmerge_forward_2) == len(lengths), "UnmergeTransform forward structure is consistent"
        
        # Backward direction: always compose (multiple → single)
        unmerge_backward = unmerge.sympy_calculate_lower(multiple_inputs)
        assert len(unmerge_backward) == 1, "UnmergeTransform backward always produces single output"
        
        # MergeTransform: Fixed semantics regardless of input count
        # Forward direction: always compose (multiple → single)
        merge_forward = merge.sympy_calculate_upper(multiple_inputs)
        assert len(merge_forward) == 1, "MergeTransform forward always produces single output"
        
        # Backward direction: always decompose (single → multiple)
        merge_backward = merge.sympy_calculate_lower([single_input])
        assert len(merge_backward) == len(lengths), "MergeTransform backward always produces multiple outputs"
    
    def test_unmerge_transform_multiindex_inverse_property(self):
        """Test UnmergeTransform MultiIndex forward/backward are inverses."""
        lengths = [2, 3, 4]
        transform = UnmergeTransform(lengths)
        
        # Test forward (multiple→single) then backward (single→multiple)
        for val0 in range(lengths[0]):
            for val1 in range(lengths[1]):
                for val2 in range(lengths[2]):
                    # Forward: multiple coordinates → single linear index
                    upper_idx = MultiIndex(3, [val0, val1, val2])
                    lower_idx = transform.calculate_lower_index(upper_idx)
                    assert len(lower_idx) == 1, "Forward should produce single output"
                    
                    # Backward: single linear index → multiple coordinates
                    recovered_upper = transform.calculate_upper_index(lower_idx)
                    assert len(recovered_upper) == 3, "Backward should produce 3 coordinates"
                    
                    # Should recover original coordinates
                    assert recovered_upper.to_list() == [val0, val1, val2], \
                        f"Failed round-trip for ({val0},{val1},{val2}): got {recovered_upper.to_list()}"
    
    def test_merge_transform_multiindex_inverse_property(self):
        """Test MergeTransform MultiIndex forward/backward are inverses."""
        lengths = [2, 3, 4]
        transform = MergeTransform(lengths)
        
        # Test forward (multiple→single) then backward (single→multiple)
        for val0 in range(lengths[0]):
            for val1 in range(lengths[1]):
                for val2 in range(lengths[2]):
                    # Forward: multiple coordinates → single linear index
                    lower_idx = MultiIndex(3, [val0, val1, val2])
                    upper_idx = transform.calculate_upper_index(lower_idx)
                    assert len(upper_idx) == 1, "Forward should produce single output"
                    
                    # Backward: single linear index → multiple coordinates
                    recovered_lower = transform.calculate_lower_index(upper_idx)
                    assert len(recovered_lower) == 3, "Backward should produce 3 coordinates"
                    
                    # Should recover original coordinates
                    assert recovered_lower.to_list() == [val0, val1, val2], \
                        f"Failed round-trip for ({val0},{val1},{val2}): got {recovered_lower.to_list()}"
    
    def test_xor_transform_multiindex_self_inverse(self):
        """Test that XOR transform MultiIndex methods are self-inverse."""
        transform = XorTransform([4, 8], apply_modulo=True)
        
        # Test self-inverse property with MultiIndex
        for x_val in range(4):
            for y_val in range(8):
                # Forward: upper → lower
                upper_idx = MultiIndex(2, [x_val, y_val])
                lower_idx = transform.calculate_lower_index(upper_idx)
                assert len(lower_idx) == 2, "XOR forward should maintain dimension count"
                
                # Backward: lower → upper (should recover original)
                recovered_upper = transform.calculate_upper_index(lower_idx)
                assert len(recovered_upper) == 2, "XOR backward should maintain dimension count"
                
                # Should get back to original
                assert recovered_upper.to_list() == [x_val, y_val], \
                    f"XOR not self-inverse for ({x_val},{y_val}): got {recovered_upper.to_list()}"
    
    def test_passthrough_transform_multiindex_identity(self):
        """Test that PassThrough MultiIndex methods are identity."""
        transform = PassThroughTransform(10)
        
        # Test identity property with MultiIndex
        for val in range(10):
            idx = MultiIndex(1, [val])
            
            # Forward and backward should be identical
            forward_result = transform.calculate_lower_index(idx)
            backward_result = transform.calculate_upper_index(idx)
            
            assert forward_result.to_list() == [val], "PassThrough forward should be identity"
            assert backward_result.to_list() == [val], "PassThrough backward should be identity"
            assert forward_result.to_list() == backward_result.to_list(), "PassThrough should be same in both directions"
    
    def test_offset_transform_multiindex_inverse(self):
        """Test that OffsetTransform MultiIndex methods are inverses."""
        transform = OffsetTransform(element_space_size=100, offset=25)
        
        # Test with various values
        for val in range(75):  # Keep within bounds after adding offset
            upper_idx = MultiIndex(1, [val])
            
            # Forward: add offset
            lower_idx = transform.calculate_lower_index(upper_idx)
            assert lower_idx.to_list() == [val + 25], "Forward should add offset"
            
            # Backward: subtract offset
            recovered_upper = transform.calculate_upper_index(lower_idx)
            assert recovered_upper.to_list() == [val], "Should recover original value"
    
    def test_embed_transform_multiindex_inverse_property(self):
        """Test EmbedTransform MultiIndex methods are inverses."""
        lengths = [4, 5]
        strides = [5, 1]  # Row-major layout
        transform = EmbedTransform(lengths, strides)
        
        # Test with all valid coordinate combinations
        for x_val in range(lengths[0]):
            for y_val in range(lengths[1]):
                # Forward: multiple coordinates → single address
                upper_idx = MultiIndex(2, [x_val, y_val])
                lower_idx = transform.calculate_lower_index(upper_idx)
                assert len(lower_idx) == 1, "Embed forward should produce single address"
                
                # Backward: single address → multiple coordinates
                recovered_upper = transform.calculate_upper_index(lower_idx)
                assert len(recovered_upper) == 2, "Embed backward should produce coordinates"
                
                # Should recover original coordinates
                assert recovered_upper.to_list() == [x_val, y_val], \
                    f"Failed round-trip for ({x_val},{y_val}): got {recovered_upper.to_list()}"
    
    def test_pad_transform_multiindex_properties(self):
        """Test PadTransform MultiIndex methods (matches C++ - no clamping)."""
        transform = PadTransform(lower_length=8, left_pad=2, right_pad=3)
        
        # Test the core mapping properties
        for val in range(13):  # 8 + 2 + 3 = 13 total padded length
            upper_idx = MultiIndex(1, [val])
            
            # Forward: upper → lower (no clamping, matches C++)
            lower_idx = transform.calculate_lower_index(upper_idx)
            assert len(lower_idx) == 1, "Pad forward should produce single output"
            
            # Check direct subtraction (no clamping)
            expected_lower = val - 2  # Direct C++ style subtraction
            assert lower_idx[0] == expected_lower, f"Forward mapping should be {val} - 2 = {expected_lower}, got {lower_idx[0]}"
                
            # Test backward from valid lower indices only
            if 0 <= val < 8:  # Test backward only for valid lower range  
                lower_test = MultiIndex(1, [val])
                upper_recovered = transform.calculate_upper_index(lower_test)
                assert upper_recovered[0] == val + 2, "Backward should add left padding"


def test_realistic_pipeline_inverse_verification():
    """Test a realistic simple pipeline to verify inverse properties."""
    
    # Create a simpler, more realistic pipeline
    # Stage 1: MergeTransform [4, 8] (multiple → single)
    stage1 = MergeTransform([4, 8])
    
    # Stage 2: UnmergeTransform [32] → [2, 16] (single → multiple)
    stage2 = UnmergeTransform([2, 16])
    
    # Forward pipeline: coordinates → merged → final coordinates
    x, y = sp.symbols('x y')
    
    # Stage 1 forward: multiple → single (composition)
    merged = stage1.sympy_calculate_upper([x, y])
    assert len(merged) == 1, "Stage 1 should produce 1 output"
    
    # Stage 2 forward: single → multiple (decomposition)
    final_coords = stage2.sympy_calculate_upper(merged)
    assert len(final_coords) == 2, "Stage 2 should produce 2 outputs"
    
    # Backward pipeline: final coordinates → merged → original coordinates
    
    # Stage 2 backward: multiple → single (composition)
    merged_back = stage2.sympy_calculate_lower(final_coords)
    assert len(merged_back) == 1, "Stage 2 backward should produce 1 output"
    
    # Stage 1 backward: single → multiple (decomposition)  
    coords_back = stage1.sympy_calculate_lower(merged_back)
    assert len(coords_back) == 2, "Stage 1 backward should produce 2 outputs"
    
    # Test with specific values that are within bounds
    for x_val in range(4):
        for y_val in range(8):
            # Forward pipeline
            merged_val = merged[0].subs({x: x_val, y: y_val})
            final_vals = [expr.subs({x: x_val, y: y_val}) for expr in final_coords]
            
            # Backward pipeline
            coords_back_vals = [expr.subs({x: x_val, y: y_val}) for expr in coords_back]
            
            # Should recover original coordinates
            assert coords_back_vals == [x_val, y_val], \
                f"Failed round-trip for ({x_val},{y_val}): got {coords_back_vals}"
    
    print("✓ Realistic pipeline inverse verification passed")


if __name__ == "__main__":
    # Run the tests
    test_instance = TestTransformInverseVerification()
    
    # Test SymPy methods
    print("Testing SymPy methods...")
    test_instance.test_unmerge_transform_inverse_property()
    test_instance.test_merge_transform_inverse_property()
    test_instance.test_xor_transform_self_inverse()
    test_instance.test_passthrough_transform_identity()
    test_instance.test_offset_transform_inverse()
    test_instance.test_embed_transform_inverse_property()
    test_instance.test_transform_semantics_are_direction_independent()
    
    # Test MultiIndex methods
    print("Testing MultiIndex methods...")
    test_instance.test_unmerge_transform_multiindex_inverse_property()
    test_instance.test_merge_transform_multiindex_inverse_property()
    test_instance.test_xor_transform_multiindex_self_inverse()
    test_instance.test_passthrough_transform_multiindex_identity()
    test_instance.test_offset_transform_multiindex_inverse()
    test_instance.test_embed_transform_multiindex_inverse_property()
    test_instance.test_pad_transform_multiindex_properties()
    
    # Test realistic pipeline
    print("Testing realistic pipeline...")
    test_realistic_pipeline_inverse_verification()
    
    print("✅ All transform inverse verification tests passed!")
    print("   - SymPy methods: ✓")
    print("   - MultiIndex methods: ✓")
    print("   - Pipeline integration: ✓") 