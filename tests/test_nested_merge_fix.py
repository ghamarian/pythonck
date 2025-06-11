import pytest
import sys
import os

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensor_transform_parser import TensorTransformParser
from tensor_transform_app import build_transformation_graph_from_pytensor


class TestNestedMergeFix:
    """Test the fix for nested merge transform hierarchy preservation."""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser instance for each test."""
        return TensorTransformParser()
    
    def test_hierarchical_merge_creation(self, parser):
        """Test that we can create hierarchical merge transforms."""
        print("\n" + "="*80)
        print("TEST: Hierarchical Merge Creation")
        print("="*80)
        
        # We need to modify the parser to create hierarchical structures
        # Instead of flattening, it should create:
        # 1. Inner merge: B + C -> intermediate
        # 2. Outer merge: A + intermediate -> final
        
        # For now, let's test what the expected structure should look like
        variables = {"A": 2, "B": 4, "C": 8}
        
        # Manually create what we expect:
        # Inner merge: B=4, C=8 -> 32
        # Outer merge: A=2, intermediate=32 -> 64
        
        from pytensor.tensor_descriptor import MergeTransform
        
        # Inner merge (B + C)
        inner_merge = MergeTransform(lengths=[4, 8])  # B + C
        print(f"Inner merge lengths: {inner_merge.lengths}")
        print(f"Inner merge total length: {sum(inner_merge.lengths) if hasattr(inner_merge, 'lengths') else 'unknown'}")
        
        # For the outer merge, we need to know the output length of inner merge
        # In merge operations: output_length = sum(input_lengths)
        inner_output_length = 4 * 8  # For merge: output = product of inputs
        
        # Outer merge (A + inner_result)
        outer_merge = MergeTransform(lengths=[2, inner_output_length])  # A + inner_result
        print(f"Outer merge lengths: {outer_merge.lengths}")
        
        print(f"\nExpected hierarchical structure:")
        print(f"1. Inner: Merge([B=4, C=8]) -> {inner_output_length}")
        print(f"2. Outer: Merge([A=2, inner={inner_output_length}]) -> {2 * inner_output_length}")
        
        # Compare with current flattened approach
        flat_merge = MergeTransform(lengths=[2, 4, 8])  # A + B + C flattened
        print(f"\nCurrent flattened: Merge([A=2, B=4, C=8]) -> ?")
        
        # The key insight: hierarchical and flat merges should produce different results!
        
    def test_expected_graph_structure(self, parser):
        """Test what the graph structure should look like for hierarchical merges."""
        print("\n" + "="*80)
        print("TEST: Expected Graph Structure")
        print("="*80)
        
        print("For nested merge: make_merge_transform(make_tuple(A, make_merge_transform(make_tuple(B, C))))")
        print("\nExpected graph should have:")
        print("1. Input nodes: d0(A), d1(B), d2(C)")
        print("2. First merge node: merge(d1, d2) -> intermediate")
        print("3. Second merge node: merge(d0, intermediate) -> output")
        print("4. Total merge nodes: 2")
        print("5. Total input nodes: 3")
        print("6. Total output nodes: 1")
        
        print("\nCurrent behavior produces:")
        print("1. Input nodes: d0(A), d1(B), d2(C)")
        print("2. Single merge node: merge(d0, d1, d2) -> output")
        print("3. Total merge nodes: 1")
        print("4. This loses the hierarchical information!")
        
    def test_manual_hierarchical_descriptor(self, parser):
        """Test creating a descriptor that should produce hierarchical transforms."""
        print("\n" + "="*80)
        print("TEST: Manual Hierarchical Descriptor")
        print("="*80)
        
        # Try to create a descriptor with explicit staging
        # This would require modifying the descriptor to have multiple stages
        hierarchical_desc = """
        transform_tensor_descriptor(
            transform_tensor_descriptor(
                input_desc,
                make_tuple(
                    make_pass_through_transform(number<A>{}),
                    make_merge_transform(make_tuple(number<B>{}, number<C>{}))
                ),
                make_tuple(sequence<0>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{})
            ),
            make_tuple(
                make_merge_transform(make_tuple(number<A>{}, number<BC>{}))
            ),
            make_tuple(sequence<0, 1>{}),
            make_tuple(sequence<0>{})
        )
        """
        
        print("Attempting to create hierarchical descriptor with two stages:")
        print("Stage 1: A passthrough, BC merge")
        print("Stage 2: A + BC merge")
        
        # This might not work with current parser, but shows the concept
        variables = {"A": 2, "B": 4, "C": 8, "BC": 32, "input_desc": 1}
        
        try:
            parsed = parser.parse_tensor_descriptor(hierarchical_desc)
            print(f"Parsed successfully: {parsed['type']}")
        except Exception as e:
            print(f"Failed to parse hierarchical descriptor: {e}")
            print("This indicates we need a different approach")


if __name__ == "__main__":
    # Run the tests directly
    test_instance = TestNestedMergeFix()
    parser = TensorTransformParser()
    
    test_instance.test_hierarchical_merge_creation(parser)
    test_instance.test_expected_graph_structure(parser)
    test_instance.test_manual_hierarchical_descriptor(parser) 