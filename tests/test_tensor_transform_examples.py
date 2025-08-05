"""
Unit tests for tensor transform examples parsing.

This module tests that all examples from tensor_transform_examples.py
can be successfully parsed by the TensorTransformParser.
"""

import pytest
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_transforms import get_transform_examples, get_default_variables, TensorTransformParser


class TestTensorTransformExamples:
    """Test cases for all tensor transform examples."""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser instance for each test."""
        return TensorTransformParser()
    
    @pytest.fixture
    def examples_and_variables(self):
        """Get all examples and their default variables."""
        return get_transform_examples(), get_default_variables()
    
    def test_simple_pass_through_merge(self, parser):
        """Test the simple pass-through and merge example."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["Simple Pass-Through & Merge"]
        variables = variables_dict["Simple Pass-Through & Merge"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        assert descriptor['type'] == 'transform'
        assert len(descriptor['transforms']) == 2
        assert descriptor['transforms'][0]['type'] == 'pass_through'
        assert descriptor['transforms'][1]['type'] == 'merge'
        
        # Test that we can create pytensor objects
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
    
    def test_all_pass_through(self, parser):
        """Test the all pass-through example."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["All Pass-Through"]
        variables = variables_dict["All Pass-Through"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        assert descriptor['type'] == 'transform'
        assert len(descriptor['transforms']) == 3
        for transform in descriptor['transforms']:
            assert transform['type'] == 'pass_through'
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
    
    def test_k_lds_block_desc(self, parser):
        """Test the K LDS block descriptor example."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["K LDS Block Desc"]
        variables = variables_dict["K LDS Block Desc"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        assert descriptor['type'] == 'transform'
        assert len(descriptor['transforms']) == 2
        assert descriptor['transforms'][0]['type'] == 'merge'
        assert descriptor['transforms'][1]['type'] == 'merge'
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
    
    def test_v_lds_block_desc(self, parser):
        """Test the V LDS block descriptor example."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["V LDS Block Desc"]
        variables = variables_dict["V LDS Block Desc"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        assert descriptor['type'] == 'transform'
        assert len(descriptor['transforms']) == 2
        assert descriptor['transforms'][0]['type'] == 'merge'
        assert descriptor['transforms'][1]['type'] == 'merge'
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
    
    def test_b_lds_block_desc_raw_vars(self, parser):
        """Test the B LDS block descriptor with raw variables."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["B LDS Block Desc (Raw Vars)"]
        variables = variables_dict["B LDS Block Desc (Raw Vars)"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        assert descriptor['type'] == 'transform'
        assert len(descriptor['transforms']) == 2
        assert descriptor['transforms'][0]['type'] == 'pass_through'
        assert descriptor['transforms'][1]['type'] == 'merge'
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
    
    def test_realistic_multi_descriptor_example(self, parser):
        """Test the realistic multi-descriptor example."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["Realistic Multi-Descriptor Example"]
        variables = variables_dict["Realistic Multi-Descriptor Example"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        # This example is a transform_tensor_descriptor containing make_naive_tensor_descriptor_packed
        # The correct type should be 'transform', not 'naive_packed'
        assert descriptor['type'] == 'transform'
        assert 'transforms' in descriptor
        assert len(descriptor['transforms']) > 0
        
        # Create pytensor to verify it works
        pytensor = parser.create_pytensor_descriptor(example, variables)
        assert pytensor is not None
    
    def test_arithmetic_sequence_transform(self, parser):
        """Test the arithmetic sequence transform example (newly fixed)."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["Arithmetic Sequence Transform"]
        variables = variables_dict["Arithmetic Sequence Transform"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        assert descriptor['type'] == 'transform'
        assert len(descriptor['transforms']) == 1
        assert descriptor['transforms'][0]['type'] == 'unmerge'
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
    
    def test_arithmetic_sequence_with_explicit_lengths(self, parser):
        """Test the new arithmetic sequence example with explicit scalar variables."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["Arithmetic Sequence with Explicit Lengths"]
        variables = variables_dict["Arithmetic Sequence with Explicit Lengths"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        assert descriptor['type'] == 'transform'
        assert len(descriptor['transforms']) == 1
        assert descriptor['transforms'][0]['type'] == 'unmerge'
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
        
        # Test variable detection
        var_info = parser.get_variable_info(example)
        list_vars = [k for k, v in var_info.items() if v.get('type') == 'list']
        scalar_vars = [k for k, v in var_info.items() if v.get('type') != 'list']
        
        # Should detect A, B, C, N as scalars, no list variables
        assert 'A' in scalar_vars
        assert 'B' in scalar_vars
        assert 'C' in scalar_vars
        assert 'N' in scalar_vars
        assert len(list_vars) == 0
    
    def test_arithmetic_sequence_with_list_variable(self, parser):
        """Test the new arithmetic sequence example with list variable."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["Arithmetic Sequence with List Variable"]
        variables = variables_dict["Arithmetic Sequence with List Variable"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        assert descriptor['type'] == 'transform'
        assert len(descriptor['transforms']) == 1
        assert descriptor['transforms'][0]['type'] == 'unmerge'
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
        
        # Test variable detection
        var_info = parser.get_variable_info(example)
        list_vars = [k for k, v in var_info.items() if v.get('type') == 'list']
        scalar_vars = [k for k, v in var_info.items() if v.get('type') != 'list']
        
        # Should detect N as scalar, lengths as list variable
        assert 'N' in scalar_vars
        assert 'lengths' in list_vars
        assert len(list_vars) == 1
        
        # Check list variable info
        lengths_info = var_info['lengths']
        assert lengths_info['type'] == 'list'
        assert lengths_info['context'] == 'unmerge_lengths'
        assert isinstance(lengths_info['default_value'], list)
    
    def test_a_lds_block_desc_example(self, parser):
        """Test the A LDS block descriptor example."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["A LDS Block Desc Example"]
        variables = variables_dict["A LDS Block Desc Example"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        # This example is a transform_tensor_descriptor (correct parsing)
        assert descriptor['type'] == 'transform'
        assert 'transforms' in descriptor
        assert len(descriptor['transforms']) > 0
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
    
    def test_x_lds_block_desc_example(self, parser):
        """Test the X LDS block descriptor example."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["X LDS Block Desc Example"]
        variables = variables_dict["X LDS Block Desc Example"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        # This example is a transform_tensor_descriptor (correct parsing)
        assert descriptor['type'] == 'transform'
        assert 'transforms' in descriptor
        assert len(descriptor['transforms']) > 0
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
    
    def test_xt_lds_block_desc_example(self, parser):
        """Test the XT LDS block descriptor example."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["XT LDS Block Desc Example"]
        variables = variables_dict["XT LDS Block Desc Example"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        # This example is a transform_tensor_descriptor (correct parsing)
        assert descriptor['type'] == 'transform'
        assert 'transforms' in descriptor
        assert len(descriptor['transforms']) > 0
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None
    
    def test_b_lds_block_desc_example(self, parser):
        """Test the B LDS block descriptor example (newly fixed)."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["B LDS Block Desc Example"]
        variables = variables_dict["B LDS Block Desc Example"]
        
        descriptor = parser.parse_tensor_descriptor(example)
        
        # This example is a transform_tensor_descriptor (correct parsing)
        assert descriptor['type'] == 'transform'
        assert 'transforms' in descriptor
        assert len(descriptor['transforms']) > 0
        
        # Test pytensor creation
        tensor_desc = parser.create_pytensor_descriptor(example, variables)
        assert tensor_desc is not None

    def test_a_grid_desc_multi_transform_example(self, parser):
        """Test the A Grid Desc Multi-Transform Example with user's original format."""
        examples, variables_dict = get_transform_examples(), get_default_variables()
        example = examples["A Grid Desc Multi-Transform Example"]
        variables = variables_dict["A Grid Desc Multi-Transform Example"]
        
        # Extract all individual descriptors
        descriptors = self._extract_descriptors_from_multi_example(example)
        assert len(descriptors) == 4, f"Expected 4 descriptors, got {len(descriptors)}"
        
        # Test each descriptor individually
        descriptor_types = []
        for i, desc in enumerate(descriptors):
            descriptor = parser.parse_tensor_descriptor(desc)
            assert descriptor is not None
            descriptor_types.append(descriptor['type'])
            
            # Test pytensor creation
            tensor_desc = parser.create_pytensor_descriptor(desc, variables)
            assert tensor_desc is not None
        
        # Verify expected types: naive, transform, transform, transform
        expected_types = ['naive', 'transform', 'transform', 'transform']
        assert descriptor_types == expected_types, f"Expected {expected_types}, got {descriptor_types}"
        
        # Verify specific features
        # First descriptor should be naive with 2-parameter format
        first_desc = parser.parse_tensor_descriptor(descriptors[0])
        assert first_desc['type'] == 'naive'
        assert len(first_desc['lengths']) == 2  # M, K
        assert len(first_desc['strides']) == 2  # StrideA, I1
        
        # Second descriptor should have unmerge transform
        second_desc = parser.parse_tensor_descriptor(descriptors[1])
        assert second_desc['type'] == 'transform'
        assert any(t['type'] == 'unmerge' for t in second_desc['transforms'])
        
        # Third descriptor should have xor_with_modulo transform
        third_desc = parser.parse_tensor_descriptor(descriptors[2])
        assert third_desc['type'] == 'transform'
        assert any(t['type'] == 'xor_with_modulo' for t in third_desc['transforms'])
        
        # Fourth descriptor should have merge_v3_division_mod transform
        fourth_desc = parser.parse_tensor_descriptor(descriptors[3])
        assert fourth_desc['type'] == 'transform'
        # Check if any transform is merge (merge_v3_division_mod is treated as merge)
        assert any(t['type'] == 'merge' for t in fourth_desc['transforms'])
    
    def _extract_descriptors_from_multi_example(self, example_text):
        """Extract individual descriptors from multi-descriptor examples."""
        lines = [line.strip() for line in example_text.strip().split('\n') if line.strip()]
        descriptors = []
        current_desc = []
        
        for line in lines:
            if line.startswith(('const auto', 'constexpr auto')):
                if current_desc:
                    # Process previous descriptor
                    desc_text = ' '.join(current_desc)
                    if '=' in desc_text:
                        rhs = desc_text.split('=', 1)[1].strip()
                        if rhs.endswith(';'):
                            rhs = rhs[:-1]
                        descriptors.append(rhs)
                    current_desc = []
                current_desc.append(line)
            else:
                current_desc.append(line)
        
        # Don't forget the last descriptor
        if current_desc:
            desc_text = ' '.join(current_desc)
            if '=' in desc_text:
                rhs = desc_text.split('=', 1)[1].strip()
                if rhs.endswith(';'):
                    rhs = rhs[:-1]
                descriptors.append(rhs)
        
        return descriptors

    def test_all_examples_parse_successfully(self, parser, examples_and_variables):
        """Test that all examples can be parsed without errors."""
        examples, variables_dict = examples_and_variables
        
        failed_examples = []
        
        # Multi-descriptor examples that need special handling
        multi_descriptor_examples = {
            "A Grid Desc Multi-Transform Example",
            "Realistic Multi-Descriptor Example", 
            "A LDS Block Desc Example",
            "X LDS Block Desc Example", 
            "XT LDS Block Desc Example",
            "B LDS Block Desc Example"
        }
        
        for name, example in examples.items():
            try:
                variables = variables_dict.get(name, {})
                
                if name in multi_descriptor_examples:
                    # Handle multi-descriptor examples by extracting individual descriptors
                    descriptors = self._extract_descriptors_from_multi_example(example)
                    if descriptors:
                        # Test the first descriptor (usually the naive one)
                        descriptor = parser.parse_tensor_descriptor(descriptors[0])
                        tensor_desc = parser.create_pytensor_descriptor(descriptors[0], variables)
                    else:
                        # Fallback to original parsing if extraction fails
                        descriptor = parser.parse_tensor_descriptor(example)
                        tensor_desc = parser.create_pytensor_descriptor(example, variables)
                else:
                    # Handle single-descriptor examples normally
                    descriptor = parser.parse_tensor_descriptor(example)
                    tensor_desc = parser.create_pytensor_descriptor(example, variables)
                
                # Basic validation
                assert descriptor is not None
                assert 'type' in descriptor
                assert descriptor['type'] in ['transform', 'naive', 'naive_packed', 'variable']
                assert tensor_desc is not None
                
            except Exception as e:
                failed_examples.append((name, str(e)))
        
        # All examples should parse successfully
        assert len(failed_examples) == 0, f"Failed examples: {failed_examples}"
    
    def test_parser_handles_new_patterns(self, parser):
        """Test that the parser correctly handles newly added patterns."""
        
        # Test typename arithmetic_sequence_gen pattern
        arithmetic_expr = "typename arithmetic_sequence_gen<0, N, 1>::type{}"
        result = parser._parse_value_expr(arithmetic_expr)
        assert result is not None
        
        # Test variable * number<expression> pattern
        mult_expr = "BK0 * number<NLdsLayer>{}"
        result = parser._parse_value_expr(mult_expr)
        assert result is not None
        
        # Test regular number<expression> pattern still works
        number_expr = "number<kKPerBlock / kKPack>{}"
        result = parser._parse_value_expr(number_expr)
        assert result is not None


class TestParserEdgeCases:
    """Test edge cases and error handling in the parser."""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser instance for each test."""
        return TensorTransformParser()
    
    def test_invalid_tensor_descriptor(self, parser):
        """Test that invalid tensor descriptors raise appropriate errors."""
        invalid_desc = "invalid_tensor_descriptor()"
        
        with pytest.raises(ValueError):
            parser.parse_tensor_descriptor(invalid_desc)
    
    def test_invalid_expression(self, parser):
        """Test that invalid expressions raise appropriate errors."""
        with pytest.raises(ValueError):
            parser._parse_value_expr("invalid@#$%expression")
    
    def test_empty_sequence(self, parser):
        """Test parsing empty sequences."""
        empty_seq = "sequence<>{}"
        result = parser.parse_sequence(empty_seq)
        assert result == []
    
    def test_nested_make_tuple(self, parser):
        """Test parsing nested make_tuple expressions."""
        nested_tuple = "make_tuple(make_tuple(number<A>{}, number<B>{}), number<C>{})"
        result = parser.parse_make_tuple(nested_tuple)
        
        assert len(result) == 2
        assert result[0]['type'] == 'merge'  # Nested tuple becomes merge
        assert result[1]['type'] == 'pass_through'


if __name__ == "__main__":
    pytest.main([__file__]) 