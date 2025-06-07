import pytest
from tensor_transform_parser import TensorTransformParser
import sympy as sp

@pytest.fixture
def parser():
    """Fixture for the TensorTransformParser."""
    return TensorTransformParser()

def test_parse_value_expr(parser):
    """Test parsing of symbolic value expressions."""
    variables = {'X': 16, 'Y': 2}
    # Test simple number
    assert parser._parse_value_expr("32") == sp.Integer(32)
    # Test variable
    expr_x = parser._parse_value_expr("X")
    assert expr_x == sp.Symbol("X")
    assert expr_x.subs(variables) == sp.Integer(16)
    # Test arithmetic
    expr_div = parser._parse_value_expr("X / Y")
    assert expr_div == sp.Symbol("X") / sp.Symbol("Y")
    assert expr_div.subs(variables) == sp.Integer(8)
    # Test number<> template
    expr_template = parser._parse_value_expr("number<X/Y>{}")
    assert expr_template == sp.Symbol("X") / sp.Symbol("Y")
    assert expr_template.subs(variables) == sp.Integer(8)
    # Test unknown variable defaults to a symbol
    assert parser._parse_value_expr("Z") == sp.Symbol("Z")

def test_parse_make_tuple(parser):
    """Test parsing of make_tuple expressions."""
    # Test simple tuple
    result = parser.parse_make_tuple("make_tuple(1, 2, 3)")
    assert len(result) == 3
    assert result[0]['value'] == sp.Integer(1)
    
    # Test with transforms
    result = parser.parse_make_tuple("make_tuple(make_pass_through_transform(0), make_merge_transform(make_tuple(1,2)))")
    assert len(result) == 2
    assert result[0]['type'] == 'pass_through'
    assert result[1]['type'] == 'merge'

def test_parse_sequence(parser):
    """Test parsing of sequence expressions."""
    # Test simple sequence
    assert parser.parse_sequence("sequence<0, 1, 2>{}") == [0, 1, 2]
    # Test empty sequence
    assert parser.parse_sequence("sequence<>{}") == []
    # Test sequence with expressions. This is expected to fail because sequences
    # must resolve to integers, but the parser now keeps variables symbolic.
    with pytest.raises(ValueError):
        parser.parse_sequence("sequence<0, X>{}")

def test_parse_transform(parser):
    """Test parsing of different transform types."""
    # Test pass_through
    transform = parser.parse_transform("make_pass_through_transform(number<3>{})")
    assert transform['type'] == 'pass_through'
    assert transform['value'] == sp.Integer(3)
    
    # Test merge
    transform = parser.parse_transform("make_merge_transform(make_tuple(4, 5))")
    assert transform['type'] == 'merge'
    assert len(transform['values']) == 2
    assert transform['values'][0]['value'] == sp.Integer(4)

@pytest.fixture
def example_variables():
    return {
        'NumIssues': 4, 'LaneGroups': 2, 'LanesPerK': 4, 'KVector': 2
    }

def test_parse_tensor_descriptor(parser, example_variables):
    """Test parsing a simple tensor descriptor."""
    example = """
    transform_tensor_descriptor(
        in_desc,
        make_tuple(make_pass_through_transform(NumIssues / LaneGroups)),
        make_tuple(sequence<0>{}),
        make_tuple(sequence<1>{})
    )
    """
    descriptor = parser.parse_tensor_descriptor(example)
    assert descriptor['input_descriptor'] == 'in_desc'
    assert len(descriptor['transforms']) == 1
    
    transform = descriptor['transforms'][0]
    assert transform['type'] == 'pass_through'
    
    # Check symbolic expression and then substitute
    expected_expr = sp.Symbol('NumIssues') / sp.Symbol('LaneGroups')
    assert transform['value'] == expected_expr
    assert transform['value'].subs(example_variables) == 2


def test_complex_tensor_descriptor(parser, example_variables):
    """Test parsing a complex tensor descriptor with multiple transforms."""
    example = """
    transform_tensor_descriptor(
        in_desc,
        make_tuple(
            make_merge_transform(make_tuple(LanesPerK, KVector)),
            make_pass_through_transform(NumIssues)
        ),
        make_tuple(sequence<0, 1>{}, sequence<2>{}),
        make_tuple(sequence<0>{}, sequence<1>{})
    )
    """
    descriptor = parser.parse_tensor_descriptor(example)
    assert len(descriptor['transforms']) == 2
    
    merge_transform = descriptor['transforms'][0]
    assert merge_transform['type'] == 'merge'
    assert len(merge_transform['values']) == 2
    
    val1 = merge_transform['values'][0]['value']
    val2 = merge_transform['values'][1]['value']
    
    assert val1 == sp.Symbol('LanesPerK')
    assert val2 == sp.Symbol('KVector')
    assert val1.subs(example_variables) == 4
    assert val2.subs(example_variables) == 2

    pass_through = descriptor['transforms'][1]
    assert pass_through['type'] == 'pass_through'
    assert pass_through['value'] == sp.Symbol('NumIssues')
    assert pass_through['value'].subs(example_variables) == 4

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