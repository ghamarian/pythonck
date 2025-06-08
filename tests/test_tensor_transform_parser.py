import pytest
import sympy as sp
from tensor_transform_parser import TensorTransformParser
from tensor_transform_examples import get_transform_examples, get_default_variables
from tensor_transform_app import build_transformation_graph_from_pytensor
from extract_descriptors import extract_descriptors_from_text

def test_parse_tensor_descriptor():
    """Test parsing a full tensor descriptor."""
    parser = TensorTransformParser()
    variables = {"G": 1, "M": 128, "K": 256, "N": 256}
    text = """
    transform_tensor_descriptor(
        k_lds_block_desc_0,
        make_tuple(
            make_merge_transform(make_tuple(G, M)),
            make_pass_through_transform(K)
        ),
        make_tuple(sequence<0, 1>{}, sequence<2>{}),
        make_tuple(sequence<0>{}, sequence<1>{})
    )
    """
    descriptor = parser.parse_tensor_descriptor(text)  # Changed from parse() to parse_tensor_descriptor()
    assert descriptor is not None
    assert len(descriptor['transforms']) == 2
    assert descriptor['transforms'][0]['type'] == 'merge'
    assert len(descriptor['transforms'][0]['values']) == 2

def test_complex_tensor_descriptor():
    """Test parsing a complex tensor descriptor with multiple merges."""
    parser = TensorTransformParser()
    variables = {"G": 1, "M": 128, "K": 256, "N": 256}
    text = """
    transform_tensor_descriptor(
        k_lds_block_desc_0,
        make_tuple(
            make_merge_transform(make_tuple(M, K)),
            make_pass_through_transform(G),
            make_unmerge_transform(make_tuple(N, 1))
        ),
        make_tuple(sequence<0, 1>{}, sequence<2>{}, sequence<3, 4>{}),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{})
    )
    """
    descriptor = parser.parse_tensor_descriptor(text)  # Changed from parse() to parse_tensor_descriptor()
    assert descriptor is not None
    assert len(descriptor['transforms']) == 3
    assert descriptor['transforms'][2]['type'] == 'unmerge'

def test_to_sympy():
    """Test full conversion to SymPy expressions."""
    parser = TensorTransformParser()

    # Test with merge and pass_through
    descriptor = {
        'input_descriptor': 'test_desc',
        'transforms': [
            {
                'type': 'merge',
                'values': [
                    {'type': 'value', 'value': 32},  # M
                    {'type': 'value', 'value': 64}   # K
                ]
            },
            {
                'type': 'pass_through',
                'value': {'type': 'value', 'value': 128}  # N
            }
        ],
        'lower_dimensions': [[0, 1], [2]],
        'upper_dimensions': [[0], [1]]
    }
    
    result = parser.to_sympy(descriptor)
    assert len(result['transforms']) == 2
    assert isinstance(result['transforms'][0]['expr'], sp.Expr)
    assert isinstance(result['transforms'][1]['expr'], sp.Expr)

def test_xt_lds_block_desc_example():
    """Test parsing and graph generation for the XT LDS Block Desc Example."""
    print("\n=== Testing XT LDS Block Desc Example ===")
    
    # Get the example and variables
    examples = get_transform_examples()
    variables = get_default_variables()
    
    xt_code = examples['XT LDS Block Desc Example']
    xt_vars = variables['XT LDS Block Desc Example']
    
    print(f"Variables: {xt_vars}")
    print(f"Code length: {len(xt_code)} characters")
    
    # Test 1: Extract descriptors
    descriptors = extract_descriptors_from_text(xt_code)
    print(f"Extracted descriptors count: {len(descriptors)}")
    assert len(descriptors) > 0, "Should extract at least one descriptor"
    
    for i, desc in enumerate(descriptors):
        print(f"Descriptor {i}: {desc[:100]}...")
    
    # Test 2: Parse descriptor
    parser = TensorTransformParser()
    parsed_descriptor = parser.parse_tensor_descriptor(descriptors[0])
    print(f"Parsed descriptor type: {parsed_descriptor['type']}")
    
    if parsed_descriptor['type'] == 'transform':
        print(f"Number of transforms: {len(parsed_descriptor['transforms'])}")
        for i, transform in enumerate(parsed_descriptor['transforms']):
            print(f"  Transform {i}: {transform['type']}")
        print(f"Lower dimensions: {parsed_descriptor['lower_dimensions']}")
        print(f"Upper dimensions: {parsed_descriptor['upper_dimensions']}")
    elif parsed_descriptor['type'] == 'naive':
        print(f"Naive descriptor keys: {list(parsed_descriptor.keys())}")
        if 'dimensions' in parsed_descriptor:
            print(f"Dimensions count: {len(parsed_descriptor['dimensions'])}")
    
    # Test 3: Create pytensor object
    try:
        tensor_desc = parser.create_pytensor_descriptor(descriptors[0], xt_vars)
        print(f"Pytensor creation: SUCCESS - {type(tensor_desc)}")
        
        # Analyze the pytensor object
        transforms = tensor_desc.get_transforms()
        print(f"Pytensor transforms count: {len(transforms)}")
        
        for i, transform in enumerate(transforms):
            print(f"  Pytensor transform {i}: {type(transform).__name__}")
        
        all_lower_ids = tensor_desc.get_lower_dimension_hidden_idss()
        all_upper_ids = tensor_desc.get_upper_dimension_hidden_idss()
        print(f"Lower dimension IDs: {all_lower_ids}")
        print(f"Upper dimension IDs: {all_upper_ids}")
        
        # Check dimension count
        num_dims = tensor_desc.get_num_of_dimension()
        print(f"Number of dimensions: {num_dims}")
        
    except Exception as e:
        print(f"Pytensor creation FAILED: {e}")
        assert False, f"Failed to create pytensor object: {e}"
    
    # Test 4: Build graph
    try:
        dot = build_transformation_graph_from_pytensor(descriptors, xt_vars)
        dot_source = dot.source
        
        # Count nodes and edges
        node_count = dot_source.count('label=')
        edge_count = dot_source.count(' -> ')
        
        print(f"Graph nodes: {node_count}")
        print(f"Graph edges: {edge_count}")
        print(f"DOT source length: {len(dot_source)}")
        
        # Check for specific patterns
        input_nodes = dot_source.count('input_d')
        s_nodes = dot_source.count('s1_') + dot_source.count('s2_') + dot_source.count('s3_')
        
        print(f"Input nodes: {input_nodes}")
        print(f"Stage nodes: {s_nodes}")
        
        # This should pass - if it generates a very small graph, we have a problem
        assert edge_count >= 1, f"Graph should have at least 1 edge, got {edge_count}"
        assert node_count >= 2, f"Graph should have at least 2 nodes, got {node_count}"
        
        print("Graph generation: SUCCESS")
        
    except Exception as e:
        print(f"Graph generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Failed to build graph: {e}"

def test_xt_vs_realistic_comparison():
    """Compare XT example with Realistic Multi-Descriptor example."""
    print("\n=== Comparing XT vs Realistic Multi ===")
    
    examples = get_transform_examples()
    variables = get_default_variables()
    
    # Test both examples
    test_cases = [
        ('XT LDS Block Desc Example', examples['XT LDS Block Desc Example'], variables['XT LDS Block Desc Example']),
        ('Realistic Multi-Descriptor Example', examples['Realistic Multi-Descriptor Example'], variables['Realistic Multi-Descriptor Example'])
    ]
    
    for name, code, vars in test_cases:
        print(f"\n--- {name} ---")
        
        try:
            descriptors = extract_descriptors_from_text(code)
            print(f"Descriptors: {len(descriptors)}")
            
            dot = build_transformation_graph_from_pytensor(descriptors, vars)
            dot_source = dot.source
            
            node_count = dot_source.count('label=')
            edge_count = dot_source.count(' -> ')
            
            print(f"Nodes: {node_count}, Edges: {edge_count}")
            
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 