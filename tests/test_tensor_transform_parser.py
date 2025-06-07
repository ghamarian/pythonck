import pytest
from tensor_transform_parser import TensorTransformParser
import sympy as sp

def test_parse_number(parser, example_variables):
    """Test parsing number expressions."""
    # Test literal number
    assert parser.parse_number("42") == 42
    
    # Test template parameter
    parser.set_variables(example_variables)
    assert parser.parse_number("number<kKPerBlock>{}") == 64
    
    # Test unknown variable
    assert parser.parse_number("number<UnknownVar>{}") == 1

def test_parse_make_tuple(parser):
    """Test parsing make_tuple expressions."""
    # Test simple tuple
    result = parser.parse_make_tuple("make_tuple(1, 2, 3)")
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    
    # Test nested tuple
    result = parser.parse_make_tuple("make_tuple(1, make_tuple(2, 3))")
    assert len(result) == 2
    assert result[0] == 1
    assert result[1]['type'] == 'merge'
    assert len(result[1]['values']) == 2

def test_parse_sequence(parser):
    """Test parsing sequence expressions."""
    # Test empty sequence
    assert parser.parse_sequence("sequence<>{}") == []
    
    # Test single element
    assert parser.parse_sequence("sequence<1>{}") == [1]
    
    # Test multiple elements
    assert parser.parse_sequence("sequence<1, 2, 3>{}") == [1, 2, 3]

def test_parse_transform(parser, example_variables):
    """Test parsing transform expressions."""
    parser.set_variables(example_variables)
    
    # Test pass-through transform
    result = parser.parse_transform("make_pass_through_transform(number<kKPerBlock>{})")
    assert result['type'] == 'pass_through'
    assert result['value'] == 64
    
    # Test merge transform
    result = parser.parse_transform("make_merge_transform(make_tuple(1, 2))")
    assert result['type'] == 'merge'
    assert len(result['values']) == 2

def test_parse_tensor_descriptor(parser, example_variables):
    """Test parsing complete tensor descriptor."""
    parser.set_variables(example_variables)
    
    # Example descriptor
    descriptor_str = """
    transform_tensor_descriptor(
        k_lds_block_desc_0,
        make_tuple(
            make_pass_through_transform(number<kNPerBlock>{}),
            make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}))
    """
    
    # Parse descriptor
    result = parser.parse_tensor_descriptor(descriptor_str)
    
    # Verify structure
    assert 'input_descriptor' in result
    assert 'transforms' in result
    assert 'lower_dimensions' in result
    assert 'upper_dimensions' in result
    
    # Verify transforms
    assert len(result['transforms']) == 2
    assert result['transforms'][0]['type'] == 'pass_through'
    assert result['transforms'][1]['type'] == 'merge'

def test_to_sympy(parser):
    """Test conversion to SymPy expressions."""
    # Create a simple descriptor
    descriptor = {
        'input_descriptor': 'test_desc',
        'transforms': [
            {
                'type': 'pass_through',
                'value': 0
            },
            {
                'type': 'merge',
                'values': [1, 2]
            }
        ],
        'lower_dimensions': [[1], [0, 2]],
        'upper_dimensions': [[0], [1]]
    }
    
    # Convert to SymPy
    result = parser.to_sympy(descriptor)
    
    # Verify structure
    assert 'transforms' in result
    assert 'symbols' in result
    
    # Verify transforms
    assert len(result['transforms']) == 2
    assert result['transforms'][0]['type'] == 'pass_through'
    assert result['transforms'][1]['type'] == 'merge'
    
    # Verify symbols
    assert 'dim_0' in result['symbols']
    assert 'dim_1' in result['symbols']
    assert 'dim_2' in result['symbols']

def test_complex_tensor_descriptor(parser, example_variables):
    """Test parsing a complex tensor descriptor with multiple transforms."""
    parser.set_variables(example_variables)
    
    # Complex descriptor with multiple transforms
    descriptor_str = """
    transform_tensor_descriptor(
        k_lds_block_desc_0,
        make_tuple(
            make_pass_through_transform(number<NumIssues>{}),
            make_pass_through_transform(number<NumWarps>{}),
            make_merge_transform(make_tuple(
                number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
        make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}))
    """
    
    # Parse descriptor
    result = parser.parse_tensor_descriptor(descriptor_str)
    
    # Verify structure
    assert len(result['transforms']) == 3
    assert result['transforms'][0]['type'] == 'pass_through'
    assert result['transforms'][1]['type'] == 'pass_through'
    assert result['transforms'][2]['type'] == 'merge'
    
    # Verify merge transform values
    merge_transform = result['transforms'][2]
    assert len(merge_transform['values']) == 3

def main():
    """Run all tests."""
    parser = TensorTransformParser()
    example_variables = {'kKPerBlock': 64}
    test_parse_number(parser, example_variables)
    test_parse_make_tuple(parser)
    test_parse_sequence(parser)
    test_parse_transform(parser, example_variables)
    test_parse_tensor_descriptor(parser, example_variables)
    test_to_sympy(parser)
    test_complex_tensor_descriptor(parser, example_variables)
    print("All tests passed!")

if __name__ == "__main__":
    main() 