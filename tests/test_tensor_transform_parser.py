import pytest
from tensor_transform_parser import TensorTransformParser
import sympy as sp

def test_parse_value_expr(parser):
    """Test parsing of symbolic value expressions."""
    parser.set_variables({'X': 16, 'Y': 2})
    # Test simple number
    assert parser._parse_value_expr("32") == sp.Integer(32)
    # Test variable
    assert parser._parse_value_expr("X") == sp.Integer(16)
    # Test arithmetic
    assert parser._parse_value_expr("X / Y") == sp.Integer(8)
    # Test number<> template
    assert parser._parse_value_expr("number<X/Y>{}") == sp.Integer(8)
    # Test unknown variable defaults to a symbol
    assert parser._parse_value_expr("Z") == sp.Symbol("Z")

def test_parse_make_tuple(parser):
    """Test parsing of make_tuple expressions."""
    # Test with numbers and transforms
    tuple_str = "make_tuple(32, make_pass_through_transform(16))"
    result = parser.parse_make_tuple(tuple_str)
    assert len(result) == 2
    assert result[0] == {'type': 'pass_through', 'value': sp.Integer(32)}
    assert result[1] == {'type': 'pass_through', 'value': sp.Integer(16)}

def test_parse_sequence(parser):
    """Test parsing of sequence expressions."""
    parser.set_variables({'X': 8})
    # Test simple sequence
    assert parser.parse_sequence("sequence<1, 2, 3>") == [1, 2, 3]
    # Test sequence with variables/expressions
    assert parser.parse_sequence("sequence<X, X*2, 3>{}") == [8, 16, 3]
    # Test empty sequence
    assert parser.parse_sequence("sequence<>") == []

def test_parse_transform(parser):
    """Test parsing of transform expressions."""
    # Test pass_through
    transform = parser.parse_transform("make_pass_through_transform(32)")
    assert transform['type'] == 'pass_through'
    assert transform['value'] == sp.Integer(32)
    # Test merge
    transform = parser.parse_transform("make_merge_transform(make_tuple(8, 4))")
    assert transform['type'] == 'merge'
    assert len(transform['values']) == 2
    assert transform['values'][0]['value'] == sp.Integer(8)

def test_parse_tensor_descriptor(parser, example_variables):
    """Test parsing a simple tensor descriptor."""
    parser.set_variables(example_variables)
    descriptor_str = """
    transform_tensor_descriptor(
        k_lds_block_desc_0,
        make_tuple(
            make_pass_through_transform(number<kNPerBlock>{}),
            make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}))
    """
    result = parser.parse_tensor_descriptor(descriptor_str)
    assert result['input_descriptor'] == 'k_lds_block_desc_0'
    assert len(result['transforms']) == 2
    assert result['transforms'][0]['type'] == 'pass_through'
    assert result['transforms'][0]['value'] == sp.Integer(32) # From example_variables in conftest
    assert result['transforms'][1]['type'] == 'merge'

def test_complex_tensor_descriptor(parser, example_variables):
    """Test parsing a complex tensor descriptor with multiple transforms."""
    parser.set_variables(example_variables)
    descriptor_str = """
    transform_tensor_descriptor(
        input_desc,
        make_tuple(
            make_merge_transform(
                make_tuple(
                    make_pass_through_transform(number<A>{}),
                    make_merge_transform(make_tuple(number<B>{}, number<C>{}))
                )
            ),
            make_pass_through_transform(number<D>{})
        ),
        make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
        make_tuple(sequence<0>{}, sequence<1>{})
    )
    """
    result = parser.parse_tensor_descriptor(descriptor_str)
    assert len(result['transforms']) == 2
    # Check the nested structure
    outer_merge = result['transforms'][0]
    assert outer_merge['type'] == 'merge'
    assert len(outer_merge['values']) == 2
    
    inner_pass_through = outer_merge['values'][0]
    assert inner_pass_through['type'] == 'pass_through'
    assert inner_pass_through['value'] == sp.Symbol('A') # Not subs'd here, which is fine

    inner_merge = outer_merge['values'][1]
    assert inner_merge['type'] == 'merge'
    assert len(inner_merge['values']) == 2
    assert inner_merge['values'][0]['value'] == sp.Symbol('B')
    assert inner_merge['values'][1]['value'] == sp.Symbol('C')
    
    pass_through_d = result['transforms'][1]
    assert pass_through_d['type'] == 'pass_through'
    assert pass_through_d['value'] == sp.Symbol('D')

def main():
    """Run all tests."""
    parser = TensorTransformParser()
    example_variables = {'kKPerBlock': 64}
    test_parse_value_expr(parser)
    test_parse_make_tuple(parser)
    test_parse_sequence(parser)
    test_parse_transform(parser)
    test_parse_tensor_descriptor(parser, example_variables)
    test_complex_tensor_descriptor(parser, example_variables)
    print("All tests passed!")

if __name__ == "__main__":
    main() 