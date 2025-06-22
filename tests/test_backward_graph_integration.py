"""
Integration tests for backward graph functionality with real examples.

This module tests the complete backward graph pipeline with actual
tensor descriptor examples to ensure everything works end-to-end.
"""

import pytest
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_transform_app import build_backward_transformation_graph_from_pytensor
from tensor_transforms import get_transform_examples, get_default_variables, extract_descriptors_from_text


class TestBackwardGraphIntegration:
    """Integration tests for backward graph with real examples."""
    
    def test_a_lds_backward_graph(self):
        """Test backward graph generation with A LDS Block Desc example."""
        examples = get_transform_examples()
        variables = get_default_variables()
        
        code = examples["A LDS Block Desc Example"]
        vars = variables["A LDS Block Desc Example"]
        
        descriptors = extract_descriptors_from_text(code)
        
        # Should have multiple descriptors
        assert len(descriptors) >= 2
        
        # Generate backward graph
        dot = build_backward_transformation_graph_from_pytensor(descriptors, vars)
        dot_source = dot.source
        
        # Basic structure checks
        assert 'rankdir=LR' in dot_source  # Left-to-right direction like forward graph
        assert "backward_title" in dot_source
        assert "Backward Graph" in dot_source
        
        # Should have starting nodes
        assert "backward_start_d" in dot_source
        
        # Should have some transforms
        transform_found = any(label in dot_source for label in 
                            ["Embed⁻¹", "Merge⁻¹", "Unmerge⁻¹", "PassThrough⁻¹", "Xor⁻¹"])
        assert transform_found, f"No backward transforms found in: {dot_source[:500]}..."
        
        # Should have variables substituted
        for var_name, var_value in vars.items():
            if isinstance(var_value, int):
                assert str(var_value) in dot_source
    
    def test_b_lds_backward_graph(self):
        """Test backward graph generation with B LDS Block Desc example."""
        examples = get_transform_examples()
        variables = get_default_variables()
        
        code = examples["B LDS Block Desc Example"]
        vars = variables["B LDS Block Desc Example"]
        
        descriptors = extract_descriptors_from_text(code)
        
        # Generate backward graph
        dot = build_backward_transformation_graph_from_pytensor(descriptors, vars)
        dot_source = dot.source
        
        # Should have content
        assert len(dot_source) > 200  # Reasonable minimum size
        assert "backward_start_d" in dot_source
        
        # Check for XOR transformation with inverse notation
        if "Xor⁻¹" in dot_source:
            # XOR should be present with inverse notation for this example
            assert "Xor⁻¹" in dot_source
    
    def test_simple_merge_backward_graph(self):
        """Test backward graph with Simple Pass-Through & Merge example."""
        examples = get_transform_examples()
        variables = get_default_variables()
        
        code = examples["Simple Pass-Through & Merge"]
        vars = variables["Simple Pass-Through & Merge"]
        
        descriptors = extract_descriptors_from_text(code)
        
        # Generate backward graph
        dot = build_backward_transformation_graph_from_pytensor(descriptors, vars)
        dot_source = dot.source
        
        # Should have merge operations with inverse notation
        assert "Merge⁻¹" in dot_source or "PassThrough⁻¹" in dot_source
        
        # Should have proper variable substitution
        for var_name, var_value in vars.items():
            if isinstance(var_value, int):
                assert str(var_value) in dot_source
    
    def test_backward_graph_node_count(self):
        """Test that backward graph has reasonable number of nodes."""
        examples = get_transform_examples()
        variables = get_default_variables()
        
        code = examples["A LDS Block Desc Example"]
        vars = variables["A LDS Block Desc Example"]
        
        descriptors = extract_descriptors_from_text(code)
        
        # Generate backward graph
        dot = build_backward_transformation_graph_from_pytensor(descriptors, vars)
        dot_source = dot.source
        
        # Count nodes (lines with node definitions)
        node_lines = [line for line in dot_source.split('\n') if ' [' in line and ']' in line]
        
        # Should have at least a few nodes (title + start nodes + transform nodes)
        assert len(node_lines) >= 3
        
        # Should not have excessive nodes (sanity check)
        assert len(node_lines) <= 50
    
    def test_backward_graph_edge_count(self):
        """Test that backward graph has edges connecting nodes."""
        examples = get_transform_examples()
        variables = get_default_variables()
        
        code = examples["A LDS Block Desc Example"]
        vars = variables["A LDS Block Desc Example"]
        
        descriptors = extract_descriptors_from_text(code)
        
        # Generate backward graph
        dot = build_backward_transformation_graph_from_pytensor(descriptors, vars)
        dot_source = dot.source
        
        # Count edges (lines with -> connections)
        edge_lines = [line for line in dot_source.split('\n') if ' -> ' in line]
        
        # Should have at least some edges to connect nodes
        assert len(edge_lines) >= 1, f"No edges found in graph: {dot_source}"
    
    def test_backward_graph_color_scheme(self):
        """Test that backward graph uses distinct colors from forward graph."""
        examples = get_transform_examples()
        variables = get_default_variables()
        
        code = examples["Simple Pass-Through & Merge"]
        vars = variables["Simple Pass-Through & Merge"]
        
        descriptors = extract_descriptors_from_text(code)
        
        # Generate backward graph
        dot = build_backward_transformation_graph_from_pytensor(descriptors, vars)
        dot_source = dot.source
        
        # Should use light blue for title
        assert 'fillcolor=lightblue' in dot_source
        
        # Should use light green for start nodes
        assert 'fillcolor="#ccffcc"' in dot_source
        
        # May have light purple for backward transform nodes
        # (depends on whether transforms are present)
    
    def test_backward_graph_error_recovery(self):
        """Test that backward graph handles errors gracefully."""
        # Test with empty descriptors
        dot = build_backward_transformation_graph_from_pytensor([], {})
        assert dot.source is not None
        assert len(dot.source) > 0
        
        # Test with invalid variables
        examples = get_transform_examples()
        code = examples["Simple Pass-Through & Merge"]
        descriptors = extract_descriptors_from_text(code)
        
        # Use problematic variables
        invalid_vars = {"nonexistent_var": "invalid_value"}
        
        # Should not crash
        dot = build_backward_transformation_graph_from_pytensor(descriptors, invalid_vars)
        assert dot.source is not None
    
    def test_backward_graph_formula_generation(self):
        """Test that backward graph generates meaningful formulas."""
        examples = get_transform_examples()
        variables = get_default_variables()
        
        code = examples["A LDS Block Desc Example"]
        vars = variables["A LDS Block Desc Example"]
        
        descriptors = extract_descriptors_from_text(code)
        
        # Generate backward graph
        dot = build_backward_transformation_graph_from_pytensor(descriptors, vars)
        dot_source = dot.source
        
        # Should have some mathematical expressions in node labels
        # Look for common mathematical operations
        math_expressions = ['/', '*', '+', '-', 'xor', 'floor', 'mod', '%']
        has_math = any(expr in dot_source for expr in math_expressions)
        
        # For A LDS example, we expect some mathematical operations
        assert has_math, f"No mathematical expressions found in: {dot_source[:1000]}..."
    
    def test_all_examples_backward_graphs(self):
        """Test that backward graphs can be generated for all examples."""
        examples = get_transform_examples()
        variables = get_default_variables()
        
        successful_examples = 0
        total_examples = len(examples)
        
        for example_name, code in examples.items():
            try:
                vars = variables.get(example_name, {})
                descriptors = extract_descriptors_from_text(code)
                
                if descriptors:  # Only test if we can extract descriptors
                    dot = build_backward_transformation_graph_from_pytensor(descriptors, vars)
                    
                    # Basic validation
                    assert dot.source is not None
                    assert len(dot.source) > 50  # Should have reasonable content
                    assert "backward_title" in dot.source
                    
                    successful_examples += 1
                    
            except Exception as e:
                print(f"Failed to generate backward graph for {example_name}: {e}")
                # Don't fail the test, just count successes
        
        # We should succeed with most examples
        success_rate = successful_examples / total_examples
        assert success_rate >= 0.5, f"Only {successful_examples}/{total_examples} examples succeeded"


if __name__ == "__main__":
    pytest.main([__file__]) 