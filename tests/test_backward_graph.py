"""
Unit tests for backward transformation graph generation.

This module tests the backward graph generation functionality,
ensuring that backward transforms are correctly applied and
that the resulting graphs are valid.
"""

import pytest
import sys
import os
import sympy as sp
import graphviz

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_transform_app import build_backward_transformation_graph_from_pytensor
from tensor_transforms import get_transform_examples, get_default_variables, extract_descriptors_from_text, TensorTransformParser
import pytensor.tensor_descriptor as ptd


class TestBackwardGraph:
    """Test backward graph generation functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser instance for each test."""
        return TensorTransformParser()
    
    @pytest.fixture
    def simple_example_data(self):
        """Get a simple example for testing."""
        examples = get_transform_examples()
        variables = get_default_variables()
        return {
            'code': examples["Simple Pass-Through & Merge"],
            'variables': variables["Simple Pass-Through & Merge"]
        }
    
    @pytest.fixture
    def a_lds_example_data(self):
        """Get the A LDS Block Desc example for testing."""
        examples = get_transform_examples()
        variables = get_default_variables()
        return {
            'code': examples["A LDS Block Desc Example"],
            'variables': variables["A LDS Block Desc Example"]
        }
    
    def test_backward_graph_creation(self, simple_example_data):
        """Test that backward graph can be created without errors."""
        descriptors = extract_descriptors_from_text(simple_example_data['code'])
        variables = simple_example_data['variables']
        
        # Should not raise an exception
        dot = build_backward_transformation_graph_from_pytensor(descriptors, variables)
        
        assert isinstance(dot, graphviz.Digraph)
        assert dot.format == "png"
        
        # Check that the graph has some content
        dot_source = dot.source
        assert "backward_title" in dot_source
        assert "Variables:" in dot_source
    
    def test_backward_graph_with_empty_descriptors(self):
        """Test backward graph with empty descriptors."""
        dot = build_backward_transformation_graph_from_pytensor([], {})
        
        assert isinstance(dot, graphviz.Digraph)
        dot_source = dot.source
        assert "backward_title" in dot_source
    
    def test_backward_graph_rankdir(self, simple_example_data):
        """Test that backward graph has correct direction (LR)."""
        descriptors = extract_descriptors_from_text(simple_example_data['code'])
        variables = simple_example_data['variables']
        
        dot = build_backward_transformation_graph_from_pytensor(descriptors, variables)
        dot_source = dot.source
        
        # Should have left-to-right direction like forward graph
        assert 'rankdir=LR' in dot_source
    
    def test_backward_graph_node_colors(self, simple_example_data):
        """Test that backward graph nodes have correct colors."""
        descriptors = extract_descriptors_from_text(simple_example_data['code'])
        variables = simple_example_data['variables']
        
        dot = build_backward_transformation_graph_from_pytensor(descriptors, variables)
        dot_source = dot.source
        
        # Should have light green for starting nodes and light purple for backward nodes
        assert 'fillcolor="#ccffcc"' in dot_source  # Start nodes
        # May or may not have purple nodes depending on transforms
    
    def test_backward_graph_edge_labels(self, a_lds_example_data):
        """Test that backward graph edges have correct labels with inverse notation."""
        descriptors = extract_descriptors_from_text(a_lds_example_data['code'])
        variables = a_lds_example_data['variables']
        
        dot = build_backward_transformation_graph_from_pytensor(descriptors, variables)
        dot_source = dot.source
        
        # Should have transform labels with inverse notation (⁻¹)
        assert any(transform_type in dot_source for transform_type in 
                  ["Embed⁻¹", "Merge⁻¹", "Unmerge⁻¹", "PassThrough⁻¹", "Xor⁻¹"])
    
    def test_backward_graph_variable_substitution(self, a_lds_example_data):
        """Test that variables are correctly substituted in backward graph."""
        descriptors = extract_descriptors_from_text(a_lds_example_data['code'])
        variables = a_lds_example_data['variables']
        
        # Test with different variable values
        test_variables = variables.copy()
        test_variables['kKPerBlock'] = 64  # Different from default
        
        dot = build_backward_transformation_graph_from_pytensor(descriptors, test_variables)
        dot_source = dot.source
        
        # Variables should be included in the comment
        assert "64" in dot_source  # Our test value should appear
    
    def test_backward_transform_direction_handling(self, parser):
        """Test that different transform types use correct backward methods."""
        # Create simple transforms for testing
        merge_transform = ptd.MergeTransform(lengths=[2, 3])
        unmerge_transform = ptd.UnmergeTransform(lengths=[2, 3])
        passthrough_transform = ptd.PassThroughTransform(length=4)
        xor_transform = ptd.XorTransform(lengths=[4, 4])
        
        # Test merge transform backward (should use sympy_calculate_lower for 1 input)
        x = sp.Symbol('x')
        merge_backward_result = merge_transform.sympy_calculate_lower([x])
        assert len(merge_backward_result) == 2
        
        # Test unmerge transform backward (should use sympy_calculate_upper for single input)
        z = sp.Symbol('z')
        unmerge_backward_result = unmerge_transform.sympy_calculate_upper([z])
        assert len(unmerge_backward_result) == 2
        
        # Test passthrough backward
        z = sp.Symbol('z')
        passthrough_backward_result = passthrough_transform.sympy_calculate_lower([z])  # or calculate_upper, same for passthrough
        assert len(passthrough_backward_result) == 1
        
        # Test XOR backward
        a, b = sp.symbols('a b')
        xor_backward_result = xor_transform.sympy_calculate_lower([a, b])  # or calculate_upper for XOR
        assert len(xor_backward_result) == 2
    
    def test_backward_graph_complex_example(self, a_lds_example_data):
        """Test backward graph with a complex multi-descriptor example."""
        descriptors = extract_descriptors_from_text(a_lds_example_data['code'])
        variables = a_lds_example_data['variables']
        
        assert len(descriptors) > 1  # Should be multi-descriptor
        
        dot = build_backward_transformation_graph_from_pytensor(descriptors, variables)
        dot_source = dot.source
        
        # Should have multiple stages
        assert "back_s" in dot_source  # Backward stage nodes
        
        # Should have starting nodes based on final descriptor
        assert "backward_start_d" in dot_source
        
        # Should have some transform operations
        assert any(transform_type in dot_source for transform_type in 
                  ["Embed⁻¹", "Merge⁻¹", "Unmerge⁻¹", "PassThrough⁻¹", "Xor⁻¹"])
    
    def test_backward_graph_error_handling(self):
        """Test backward graph error handling with invalid descriptors."""
        # Test with malformed descriptor
        invalid_descriptors = ["invalid descriptor syntax"]
        variables = {"test": 1}
        
        # Should not crash, but may produce error nodes
        dot = build_backward_transformation_graph_from_pytensor(invalid_descriptors, variables)
        assert isinstance(dot, graphviz.Digraph)
    
    def test_backward_vs_forward_consistency(self, simple_example_data):
        """Test that backward graph is consistent with forward graph structure."""
        from tensor_transform_app import build_transformation_graph_from_pytensor
        
        descriptors = extract_descriptors_from_text(simple_example_data['code'])
        variables = simple_example_data['variables']
        
        # Build both graphs
        forward_dot = build_transformation_graph_from_pytensor(descriptors, variables)
        backward_dot = build_backward_transformation_graph_from_pytensor(descriptors, variables)
        
        forward_source = forward_dot.source
        backward_source = backward_dot.source
        
        # Both should have content
        assert len(forward_source) > 100
        assert len(backward_source) > 100
        
        # Both should include variable information
        for var_name, var_value in variables.items():
            if isinstance(var_value, int):
                assert str(var_value) in forward_source
                assert str(var_value) in backward_source
    
    def test_transform_inverse_property(self):
        """Test that some transforms satisfy the inverse property."""
        # Test MergeTransform and UnmergeTransform inverse relationship
        lengths = [3, 4]
        merge = ptd.MergeTransform(lengths=lengths)
        unmerge = ptd.UnmergeTransform(lengths=lengths)
        
        # Create test symbols
        x, y = sp.symbols('x y')
        
        # Forward: merge([x, y]) -> z (merge: multiple → single)
        merged = merge.sympy_calculate_upper([x, y])[0]
        
        # Backward: unmerge([z]) -> [x', y'] (unmerge: single → multiple)
        unmerged = unmerge.sympy_calculate_upper([merged])
        
        # Test with specific values to verify inverse property
        for x_val in range(lengths[0]):
            for y_val in range(lengths[1]):
                # Substitute values
                merged_val = merged.subs({x: x_val, y: y_val})
                unmerged_vals = [expr.subs({x: x_val, y: y_val}) for expr in unmerged]
                
                # The unmerged values should match original inputs
                assert unmerged_vals[0] == x_val
                assert unmerged_vals[1] == y_val
    
    def test_xor_self_inverse_property(self):
        """Test that XOR transform is self-inverse."""
        xor_transform = ptd.XorTransform(lengths=[4, 4])
        
        x, y = sp.symbols('x y')
        
        # Forward transformation (upper → lower)
        forward_result = xor_transform.sympy_calculate_lower([x, y])
        
        # Backward transformation of the forward result (lower → upper)
        backward_result = xor_transform.sympy_calculate_upper(forward_result)
        
        # Should get back to original inputs
        # Note: XOR might reorder outputs, so check the values rather than exact match
        for x_val in range(4):
            for y_val in range(4):
                forward_vals = [expr.subs({x: x_val, y: y_val}) for expr in forward_result]
                backward_vals = [expr.subs({x: x_val, y: y_val}) for expr in backward_result]
                
                # The backward transformation should yield values that,
                # when forward transformed again, give the same result
                forward_again = xor_transform.sympy_calculate_lower(backward_vals)
                forward_again_vals = [expr.subs({x: x_val, y: y_val}) for expr in forward_again]
                
                assert forward_again_vals == forward_vals


if __name__ == "__main__":
    pytest.main([__file__]) 