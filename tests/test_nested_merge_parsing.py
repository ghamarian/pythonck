import pytest
import sys
import os

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensor_transform_parser import TensorTransformParser
from tensor_transform_app import build_transformation_graph_from_pytensor


class TestNestedMergeParsing:
    """Test parsing and graph generation for nested merge transforms."""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser instance for each test."""
        return TensorTransformParser()
    
    def test_nested_merge_parsing_structure(self, parser):
        """Test that nested merge structures are parsed correctly."""
        print("\n" + "="*80)
        print("TEST: Nested Merge Parsing Structure")
        print("="*80)
        
        # Your specific problematic transformation
        transform_str = """
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
        
        variables = {"A": 2, "B": 4, "C": 8, "D": 16, "input_desc": 1}
        
        # Parse the descriptor
        parsed = parser.parse_tensor_descriptor(transform_str)
        print(f"Parsed descriptor type: {parsed['type']}")
        print(f"Number of transforms: {len(parsed['transforms'])}")
        
        # Examine the first transform (the nested merge)
        first_transform = parsed['transforms'][0]
        print(f"\nFirst transform type: {first_transform['type']}")
        print(f"First transform values: {first_transform['values']}")
        
        # The structure should be:
        # First transform: merge with two values:
        #   1. A pass-through of A
        #   2. A merge of B and C
        assert first_transform['type'] == 'merge'
        assert len(first_transform['values']) == 2
        
        # First value should be pass-through of A
        first_value = first_transform['values'][0]
        print(f"\nFirst value structure: {first_value}")
        
        # Second value should be a nested merge
        second_value = first_transform['values'][1]
        print(f"Second value structure: {second_value}")
        
        # Check if second value is properly structured as nested merge
        if isinstance(second_value, str):
            # If it's a string representation, it's already been flattened
            print("WARNING: Second value is a string, indicates flattening occurred")
        elif isinstance(second_value, dict) and second_value.get('type') == 'merge':
            print("GOOD: Second value is a proper nested merge structure")
            print(f"Nested merge values: {second_value.get('values', [])}")
        else:
            print(f"UNEXPECTED: Second value has type {type(second_value)}")
    
    def test_nested_merge_pytensor_creation(self, parser):
        """Test that nested merge creates appropriate pytensor objects."""
        print("\n" + "="*80)
        print("TEST: Nested Merge PyTensor Creation")
        print("="*80)
        
        # Simplified test case focusing on the nested merge
        transform_str = """
        transform_tensor_descriptor(
            input_desc,
            make_tuple(
                make_merge_transform(
                    make_tuple(
                        make_pass_through_transform(number<A>{}),
                        make_merge_transform(make_tuple(number<B>{}, number<C>{}))
                    )
                )
            ),
            make_tuple(sequence<0, 1, 2>{}),
            make_tuple(sequence<0>{})
        )
        """
        
        variables = {"A": 2, "B": 4, "C": 8, "input_desc": 1}
        
        # Create pytensor descriptor
        pytensor_desc = parser.create_pytensor_descriptor(transform_str, variables)
        
        print(f"PyTensor descriptor type: {type(pytensor_desc)}")
        if hasattr(pytensor_desc, 'transforms'):
            print(f"Number of transforms: {len(pytensor_desc.transforms)}")
            for i, transform in enumerate(pytensor_desc.transforms):
                print(f"Transform {i}: {type(transform).__name__}")
                if hasattr(transform, 'lengths'):
                    print(f"  Lengths: {transform.lengths}")
                    
        # The key question: should we have:
        # Option 1: One MergeTransform with lengths [2, 4, 8] (current behavior)
        # Option 2: Two separate transforms representing the hierarchy
        
        if hasattr(pytensor_desc, 'transforms') and len(pytensor_desc.transforms) > 0:
            first_transform = pytensor_desc.transforms[0]
            if hasattr(first_transform, 'lengths'):
                print(f"\nFirst transform lengths: {first_transform.lengths}")
                if first_transform.lengths == [2, 4, 8]:
                    print("CURRENT BEHAVIOR: Flattened to single merge with 3 inputs")
                    print("EXPECTED: Should this be hierarchical instead?")
    
    def test_nested_merge_graph_generation(self, parser):
        """Test that nested merge generates correct graph structure."""
        print("\n" + "="*80)
        print("TEST: Nested Merge Graph Generation")
        print("="*80)
        
        # Your original problematic case
        descriptors = ["""
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
        """]
        
        variables = {"A": 2, "B": 4, "C": 8, "D": 16, "input_desc": 1}
        
        try:
            # Build the graph
            graph = build_transformation_graph_from_pytensor(descriptors, variables)
            
            # Analyze the graph structure
            graph_source = graph.source
            print(f"Graph source length: {len(graph_source)}")
            
            # Count nodes and edges
            merge_count = graph_source.count('Merge')
            passthrough_count = graph_source.count('PassThrough')
            edge_count = graph_source.count('->')
            
            print(f"Merge operations in graph: {merge_count}")
            print(f"PassThrough operations in graph: {passthrough_count}")
            print(f"Total edges: {edge_count}")
            
            # Print first part of graph source for debugging
            print(f"\nGraph source (first 500 chars):")
            print(graph_source[:500])
            
            # The question: Do we see two merge nodes (correct) or one merge node with 3 inputs (incorrect)?
            print(f"\n" + "="*40)
            print("ANALYSIS:")
            print(f"If we see 1 merge node, the structure is flattened (current)")
            print(f"If we see 2 merge nodes, the structure is hierarchical (expected)")
            print(f"Actual merge count: {merge_count}")
            
            return graph
            
        except Exception as e:
            print(f"Graph generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_simple_nested_merge_case(self, parser):
        """Test a simpler nested merge case to isolate the issue."""
        print("\n" + "="*80)
        print("TEST: Simple Nested Merge Case")
        print("="*80)
        
        # Simplest possible nested merge
        simple_nested = """
        make_merge_transform(
            make_tuple(
                number<A>{},
                make_merge_transform(make_tuple(number<B>{}, number<C>{}))
            )
        )
        """
        
        variables = {"A": 2, "B": 4, "C": 8}
        
        # Parse just the transform
        parsed_transform = parser.parse_transform(simple_nested)
        print(f"Parsed transform: {parsed_transform}")
        
        # Create pytensor transform
        pytensor_transform = parser.create_pytensor_transform(parsed_transform, variables)
        print(f"PyTensor transform type: {type(pytensor_transform).__name__}")
        if hasattr(pytensor_transform, 'lengths'):
            print(f"Lengths: {pytensor_transform.lengths}")
            
        # This should tell us definitively if the flattening happens in:
        # 1. The parsing stage, or 
        # 2. The pytensor creation stage
        
        print(f"\nExpected behavior:")
        print(f"- Parse should preserve nested structure")
        print(f"- PyTensor creation should either:")
        print(f"  a) Create hierarchical transforms, or")
        print(f"  b) Flatten but indicate the hierarchy for graph generation")
    
    def test_compare_flat_vs_nested(self, parser):
        """Compare explicit flat merge vs nested merge parsing."""
        print("\n" + "="*80)
        print("TEST: Compare Flat vs Nested Merge")
        print("="*80)
        
        # Flat version: explicitly merge A, B, C
        flat_merge = """
        make_merge_transform(make_tuple(number<A>{}, number<B>{}, number<C>{}))
        """
        
        # Nested version: merge A with (merge of B, C)
        nested_merge = """
        make_merge_transform(
            make_tuple(
                number<A>{},
                make_merge_transform(make_tuple(number<B>{}, number<C>{}))
            )
        )
        """
        
        variables = {"A": 2, "B": 4, "C": 8}
        
        # Parse both
        flat_parsed = parser.parse_transform(flat_merge)
        nested_parsed = parser.parse_transform(nested_merge)
        
        print(f"Flat merge parsed: {flat_parsed}")
        print(f"Nested merge parsed: {nested_parsed}")
        
        # Create pytensor objects
        flat_pytensor = parser.create_pytensor_transform(flat_parsed, variables)
        nested_pytensor = parser.create_pytensor_transform(nested_parsed, variables)
        
        print(f"\nFlat PyTensor lengths: {flat_pytensor.lengths}")
        print(f"Nested PyTensor lengths: {nested_pytensor.lengths}")
        
        # Are they the same? If so, the flattening is the issue
        if flat_pytensor.lengths == nested_pytensor.lengths:
            print(f"\nPROBLEM IDENTIFIED: Nested merge is flattened to same as flat merge")
            print(f"This happens in the _flatten_merge_lengths method")
        else:
            print(f"\nSURPRISE: Nested and flat merges produce different results")


if __name__ == "__main__":
    # Run the tests directly
    test_instance = TestNestedMergeParsing()
    parser = TensorTransformParser()
    
    test_instance.test_nested_merge_parsing_structure(parser)
    test_instance.test_nested_merge_pytensor_creation(parser)
    test_instance.test_nested_merge_graph_generation(parser)  
    test_instance.test_simple_nested_merge_case(parser)
    test_instance.test_compare_flat_vs_nested(parser) 