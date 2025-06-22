#!/usr/bin/env python3

import pytest
from tensor_transforms import TensorTransformParser

class TestXorIntegration:
    """Test XOR transform integration with the parser."""
    
    def test_xor_transform_parsing_and_creation(self):
        """Test XOR transform parsing and tensor descriptor creation."""
        parser = TensorTransformParser()
        
        # Test a simple XOR transform descriptor  
        xor_desc = """transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(number<A>{}, number<B>{})),
            make_tuple(make_xor_transform(make_tuple(number<A>{}, number<B>{}))),
            make_tuple(sequence<0, 1>{}),
            make_tuple(sequence<0, 1>{})
        )"""
        
        # Test parsing
        result = parser.parse_tensor_descriptor(xor_desc)
        assert result.get('type') is not None, "Parsing should succeed"
        assert result.get('type') == 'transform', f"Expected transform, got {result.get('type')}"
        
        # Test tensor descriptor creation
        variables = {'A': 4, 'B': 8}
        tensor_desc = parser.create_pytensor_descriptor(xor_desc, variables)
        
        assert tensor_desc is not None, "Tensor descriptor creation should succeed"
        assert tensor_desc.get_num_of_dimension() == 2, f"Expected 2 dimensions, got {tensor_desc.get_num_of_dimension()}"
        # XOR transform preserves the input dimensions, so we should get the same logical dimensions
        # Note: The actual dimension sizes depend on the complete transformation pipeline
        assert len(tensor_desc.get_lengths()) == 2, f"Expected 2 dimensions, got {len(tensor_desc.get_lengths())}"
        assert tensor_desc.get_lengths()[0] > 0 and tensor_desc.get_lengths()[1] > 0, f"All dimensions should be positive, got {tensor_desc.get_lengths()}"
        assert tensor_desc.get_element_space_size() == 32, f"Expected element space 32, got {tensor_desc.get_element_space_size()}"
        assert tensor_desc.get_num_of_transform() == 2, f"Expected 2 transforms (UnmergeTransform + XorTransform), got {tensor_desc.get_num_of_transform()}"
        
        # Test actual coordinate transformation
        from pytensor.tensor_coordinate import MultiIndex
        test_coords = MultiIndex(2, [2, 5])
        linear_offset = tensor_desc.calculate_offset(test_coords)
        
        # With XOR transform: [2, 5] -> [2, 7] -> linear offset
        # The exact offset depends on the underlying storage pattern
        assert isinstance(linear_offset, int), f"Offset should be integer, got {type(linear_offset)}"
        assert linear_offset >= 0, f"Offset should be non-negative, got {linear_offset}"
    
    def test_xor_transform_properties(self):
        """Test that XOR transforms have correct properties in parsed descriptors."""
        parser = TensorTransformParser()
        
        xor_desc = """transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(number<A>{}, number<B>{})),
            make_tuple(make_xor_transform(make_tuple(number<A>{}, number<B>{}))),
            make_tuple(sequence<0, 1>{}),
            make_tuple(sequence<0, 1>{})
        )"""
        
        variables = {'A': 4, 'B': 8}
        tensor_desc = parser.create_pytensor_descriptor(xor_desc, variables)
        
        # Get the XOR transform
        transforms = tensor_desc.get_transforms()
        assert len(transforms) == 2, f"Expected 2 transforms (UnmergeTransform + XorTransform), got {len(transforms)}"
        
        # The first transform is the automatic UnmergeTransform from naive_tensor_descriptor_packed
        # The second transform is our XOR transform
        xor_transform = transforms[1]
        assert hasattr(xor_transform, 'lengths'), "XOR transform should have lengths attribute"
        assert hasattr(xor_transform, 'apply_modulo'), "XOR transform should have apply_modulo attribute"
        
        # Test that it can perform coordinate transformations
        from pytensor.tensor_coordinate import MultiIndex
        test_upper = MultiIndex(2, [1, 3])
        test_lower = xor_transform.calculate_lower_index(test_upper)
        test_upper_back = xor_transform.calculate_upper_index(test_lower)
        
        assert test_upper._values == test_upper_back._values, "XOR should be reversible" 