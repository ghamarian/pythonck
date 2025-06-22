"""
Detailed tests for the XT LDS Block Desc Example.

This module provides comprehensive testing of the XT LDS Block Desc Example
to help debug graph generation issues in the visualization app.
"""

import pytest
import sys
import os
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_transforms import get_transform_examples, get_default_variables, TensorTransformParser


class TestXTLDSBlockDescDetailed:
    """Detailed tests for the XT LDS Block Desc Example."""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser instance for each test."""
        return TensorTransformParser()
    
    @pytest.fixture
    def xt_example_data(self):
        """Get the XT LDS Block Desc example and its variables."""
        examples = get_transform_examples()
        variables = get_default_variables()
        return {
            'code': examples["XT LDS Block Desc Example"],
            'variables': variables["XT LDS Block Desc Example"]
        }
    
    def test_xt_example_variables(self, xt_example_data):
        """Test that all required variables are present."""
        variables = xt_example_data['variables']
        
        required_vars = [
            'KThreadWrite', 'kfold', 'KThreadReadPerm', 'K0PerThreadWrite',
            'MN1', 'MN0', 'mnpair', 'KPackT'
        ]
        
        for var in required_vars:
            assert var in variables, f"Missing required variable: {var}"
            assert isinstance(variables[var], int), f"Variable {var} should be an integer"
        
        print(f"Variables: {variables}")
    
    def test_xt_example_parsing_stages(self, parser, xt_example_data):
        """Test parsing of each stage in the XT example."""
        code = xt_example_data['code']
        variables = xt_example_data['variables']
        
        # The XT example contains multiple descriptor definitions
        # Let's parse the entire thing and see what we get
        descriptor = parser.parse_tensor_descriptor(code)
        
        print(f"Parsed descriptor type: {descriptor['type']}")
        print(f"Descriptor keys: {list(descriptor.keys())}")
        
        if descriptor['type'] == 'transform':
            print(f"Number of transforms: {len(descriptor['transforms'])}")
            for i, transform in enumerate(descriptor['transforms']):
                print(f"Transform {i}: {transform['type']}")
                if 'values' in transform:
                    print(f"  Values: {len(transform['values'])} items")
        
        # Test that we can create a pytensor descriptor
        tensor_desc = parser.create_pytensor_descriptor(code, variables)
        assert tensor_desc is not None
        print(f"Created tensor descriptor: {type(tensor_desc)}")
    
    def test_xt_individual_stages(self, parser, xt_example_data):
        """Test parsing individual stages of the XT example."""
        variables = xt_example_data['variables']
        
        # Test the raw descriptor stage
        raw_stage = """make_naive_tensor_descriptor(
    make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
               number<K0PerThreadWrite>{},
               number<KThreadReadPerm * MN1>{},
               number<kfold * MN0 / mnpair>{},
               number<mnpair>{},
               KPackT),
    make_tuple(number<KPackT * kfold * MN0 * KThreadReadPerm * MN1 * K0PerThreadWrite>{},
               number<KPackT * kfold * MN0 * KThreadReadPerm * MN1>{},
               number<KPackT * kfold * MN0>{},
               number<KPackT * mnpair>{},
               number<KPackT>{},
               number<1>{}),
    number<KPackT>{},
    number<1>{})"""
        
        raw_desc = parser.parse_tensor_descriptor(raw_stage)
        assert raw_desc['type'] == 'naive'
        print(f"Raw stage lengths: {len(raw_desc['lengths'])}")
        
        # Test the permuted stage
        permuted_stage = """transform_tensor_descriptor(
    xt_lds_block_desc_raw,
    make_tuple(
        make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
        make_pass_through_transform(number<K0PerThreadWrite>{}),
        make_xor_transform(
            make_tuple(number<KThreadReadPerm * MN1>{}, number<kfold * MN0 / mnpair>{})),
        make_pass_through_transform(number<mnpair>{}),
        make_pass_through_transform(KPackT)),
    make_tuple(
        sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
    make_tuple(
        sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}))"""
        
        permuted_desc = parser.parse_tensor_descriptor(permuted_stage)
        assert permuted_desc['type'] == 'transform'
        assert len(permuted_desc['transforms']) == 5
        print(f"Permuted stage transforms: {[t['type'] for t in permuted_desc['transforms']]}")
        
        # Check that XOR transform is parsed correctly
        xor_transform = permuted_desc['transforms'][2]
        assert xor_transform['type'] == 'xor'
        assert len(xor_transform['values']) == 2
        print(f"XOR transform values: {xor_transform['values']}")
    
    def test_xt_dimension_mapping(self, parser, xt_example_data):
        """Test dimension mapping in the XT example."""
        variables = xt_example_data['variables']
        
        # Test a simple transform to understand dimension mapping
        simple_transform = """transform_tensor_descriptor(
    input_desc,
    make_tuple(
        make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
        make_pass_through_transform(number<K0PerThreadWrite>{})),
    make_tuple(sequence<0>{}, sequence<1>{}),
    make_tuple(sequence<0>{}, sequence<1>{}))"""
        
        desc = parser.parse_tensor_descriptor(simple_transform)
        
        print(f"Lower dimensions: {desc['lower_dimensions']}")
        print(f"Upper dimensions: {desc['upper_dimensions']}")
        
        # Check that dimensions are parsed correctly
        assert len(desc['lower_dimensions']) == 2
        assert len(desc['upper_dimensions']) == 2
        assert desc['lower_dimensions'][0] == [0]
        assert desc['lower_dimensions'][1] == [1]
    
    def test_xt_variable_substitution(self, parser, xt_example_data):
        """Test variable substitution in expressions."""
        variables = xt_example_data['variables']
        
        # Test individual expressions
        test_expressions = [
            "number<KThreadWrite / kfold / KThreadReadPerm>{}",
            "number<K0PerThreadWrite>{}",
            "number<KThreadReadPerm * MN1>{}",
            "number<kfold * MN0 / mnpair>{}",
            "KPackT"
        ]
        
        for expr in test_expressions:
            try:
                parsed_expr = parser._parse_value_expr(expr)
                print(f"Expression '{expr}' -> {parsed_expr}")
                
                # Try to substitute variables
                if hasattr(parsed_expr, 'subs'):
                    substituted = parsed_expr.subs(variables)
                    print(f"  Substituted: {substituted}")
                    if substituted.is_number:
                        print(f"  Numeric value: {int(substituted)}")
                
            except Exception as e:
                print(f"Error parsing '{expr}': {e}")
    
    def test_xt_pytensor_creation_detailed(self, parser, xt_example_data):
        """Test detailed pytensor object creation."""
        code = xt_example_data['code']
        variables = xt_example_data['variables']
        
        # Create the tensor descriptor
        tensor_desc = parser.create_pytensor_descriptor(code, variables)
        
        print(f"Tensor descriptor type: {type(tensor_desc)}")
        print(f"Tensor descriptor attributes: {dir(tensor_desc)}")
        
        # Check if it has the expected attributes
        if hasattr(tensor_desc, 'get_upper_lengths'):
            upper_lengths = tensor_desc.get_upper_lengths()
            print(f"Upper lengths: {upper_lengths}")
        
        if hasattr(tensor_desc, 'get_lower_lengths'):
            lower_lengths = tensor_desc.get_lower_lengths()
            print(f"Lower lengths: {lower_lengths}")
        
        if hasattr(tensor_desc, 'transforms'):
            print(f"Number of transforms: {len(tensor_desc.transforms)}")
            for i, transform in enumerate(tensor_desc.transforms):
                print(f"Transform {i}: {type(transform)}")
    
    def test_xt_graph_data_generation(self, parser, xt_example_data):
        """Test generation of graph data for visualization."""
        code = xt_example_data['code']
        variables = xt_example_data['variables']
        
        # Parse the descriptor
        parsed_desc = parser.parse_tensor_descriptor(code)
        
        # Create a simple graph representation
        graph_data = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'type': parsed_desc['type'],
                'variables': variables
            }
        }
        
        if parsed_desc['type'] == 'transform':
            # Add input node
            graph_data['nodes'].append({
                'id': 'input',
                'label': parsed_desc.get('input_descriptor', 'input'),
                'type': 'input'
            })
            
            # Add transform nodes
            for i, transform in enumerate(parsed_desc['transforms']):
                node_id = f'transform_{i}'
                graph_data['nodes'].append({
                    'id': node_id,
                    'label': f"{transform['type']}_{i}",
                    'type': 'transform',
                    'transform_type': transform['type']
                })
                
                # Add edge from previous node
                if i == 0:
                    source = 'input'
                else:
                    source = f'transform_{i-1}'
                
                graph_data['edges'].append({
                    'source': source,
                    'target': node_id,
                    'label': f'dim_{i}'
                })
            
            # Add output node
            graph_data['nodes'].append({
                'id': 'output',
                'label': 'output',
                'type': 'output'
            })
            
            # Connect last transform to output
            if parsed_desc['transforms']:
                last_transform = f"transform_{len(parsed_desc['transforms'])-1}"
                graph_data['edges'].append({
                    'source': last_transform,
                    'target': 'output',
                    'label': 'final'
                })
        elif parsed_desc['type'] == 'naive':
            # For naive descriptors, create a simple input-output graph
            graph_data['nodes'].append({
                'id': 'input',
                'label': 'naive_input',
                'type': 'input'
            })
            
            graph_data['nodes'].append({
                'id': 'naive_desc',
                'label': f"naive_descriptor ({len(parsed_desc['lengths'])} dims)",
                'type': 'descriptor'
            })
            
            graph_data['nodes'].append({
                'id': 'output',
                'label': 'output',
                'type': 'output'
            })
            
            graph_data['edges'].append({
                'source': 'input',
                'target': 'naive_desc',
                'label': 'data'
            })
            
            graph_data['edges'].append({
                'source': 'naive_desc',
                'target': 'output',
                'label': 'result'
            })
        
        print(f"Generated graph data:")
        print(f"Nodes: {len(graph_data['nodes'])}")
        print(f"Edges: {len(graph_data['edges'])}")
        
        for node in graph_data['nodes']:
            print(f"  Node: {node}")
        
        for edge in graph_data['edges']:
            print(f"  Edge: {edge}")
        
        # Ensure we have a meaningful graph
        assert len(graph_data['nodes']) > 2, "Graph should have more than just input/output"
        assert len(graph_data['edges']) > 0, "Graph should have connections"
    
    def test_xt_sympy_conversion(self, parser, xt_example_data):
        """Test SymPy conversion for the XT example."""
        code = xt_example_data['code']
        variables = xt_example_data['variables']
        
        # Parse the descriptor
        parsed_desc = parser.parse_tensor_descriptor(code)
        
        if parsed_desc['type'] == 'transform':
            # Try to convert to SymPy
            try:
                sympy_desc = parser.to_sympy(parsed_desc)
                print(f"SymPy conversion successful")
                print(f"Transforms: {len(sympy_desc['transforms'])}")
                print(f"Symbols: {sympy_desc['symbols']}")
                
                for i, transform in enumerate(sympy_desc['transforms']):
                    print(f"Transform {i}: {transform}")
                    
            except Exception as e:
                print(f"SymPy conversion failed: {e}")
                # This might be expected if the to_sympy method needs updates
    
    def test_xt_stage_by_stage_parsing(self, parser, xt_example_data):
        """Test parsing each stage of the XT example separately."""
        variables = xt_example_data['variables']
        
        stages = [
            ("raw", """make_naive_tensor_descriptor(
    make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
               number<K0PerThreadWrite>{},
               number<KThreadReadPerm * MN1>{},
               number<kfold * MN0 / mnpair>{},
               number<mnpair>{},
               KPackT),
    make_tuple(number<KPackT * kfold * MN0 * KThreadReadPerm * MN1 * K0PerThreadWrite>{},
               number<KPackT * kfold * MN0 * KThreadReadPerm * MN1>{},
               number<KPackT * kfold * MN0>{},
               number<KPackT * mnpair>{},
               number<KPackT>{},
               number<1>{}),
    number<KPackT>{},
    number<1>{})"""),
            
            ("permuted", """transform_tensor_descriptor(
    xt_lds_block_desc_raw,
    make_tuple(
        make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
        make_pass_through_transform(number<K0PerThreadWrite>{}),
        make_xor_transform(
            make_tuple(number<KThreadReadPerm * MN1>{}, number<kfold * MN0 / mnpair>{})),
        make_pass_through_transform(number<mnpair>{}),
        make_pass_through_transform(KPackT)),
    make_tuple(
        sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
    make_tuple(
        sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}))"""),
            
            ("unmerged", """transform_tensor_descriptor(
    xt_lds_block_desc_permuted,
    make_tuple(
        make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
        make_pass_through_transform(number<K0PerThreadWrite>{}),
        make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<MN1>{})),
        make_unmerge_transform(make_tuple(number<kfold>{}, number<MN0 / mnpair>{})),
        make_pass_through_transform(number<mnpair>{}),
        make_pass_through_transform(KPackT)),
    make_tuple(sequence<0>{},
               sequence<1>{},
               sequence<2>{},
               sequence<3>{},
               sequence<4>{},
               sequence<5>{}),
    make_tuple(sequence<1>{},
               sequence<2>{},
               sequence<0, 3>{},
               sequence<4, 5>{},
               sequence<6>{},
               sequence<7>{}))"""),
            
            ("final", """transform_tensor_descriptor(
    xt_lds_block_desc_unmerged,
    make_tuple(make_merge_transform_v3_division_mod(
                   make_tuple(number<KThreadReadPerm>{},
                              number<KThreadWrite / kfold / KThreadReadPerm>{},
                              number<kfold>{},
                              number<K0PerThreadWrite>{},
                              number<KPackT>{})),
               make_merge_transform_v3_division_mod(
                   make_tuple(number<MN0 / mnpair>{}, number<mnpair>{}, number<MN1>{}))),
    make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
    make_tuple(sequence<0>{}, sequence<1>{}))""")
        ]
        
        for stage_name, stage_code in stages:
            print(f"\n=== Testing {stage_name} stage ===")
            try:
                desc = parser.parse_tensor_descriptor(stage_code)
                print(f"Stage type: {desc['type']}")
                
                if desc['type'] == 'transform':
                    print(f"Transforms: {[t['type'] for t in desc['transforms']]}")
                    print(f"Lower dims: {desc['lower_dimensions']}")
                    print(f"Upper dims: {desc['upper_dimensions']}")
                elif desc['type'] == 'naive':
                    print(f"Dimensions: {len(desc['dimensions'])}")
                
                # Try to create pytensor object
                tensor_desc = parser.create_pytensor_descriptor(stage_code, variables)
                print(f"Pytensor creation: SUCCESS")
                
            except Exception as e:
                print(f"Error in {stage_name} stage: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 