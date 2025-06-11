#!/usr/bin/env python3
"""
Unit tests for the four-descriptor tensor transformation pipeline.

Tests the complete pipeline:
1. b_lds_block_desc (naive) - 6 dimensions
2. b_lds_block_desc_permuted (XOR) - 6 dimensions  
3. b_lds_block_desc_unmerged (Unmerge) - 8 dimensions
4. b_lds_block_desc_kn (Merge) - 2 dimensions
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_transform_app import build_transformation_graph_from_pytensor, build_backward_transformation_graph_from_pytensor
from extract_descriptors import extract_descriptors_from_text
from tensor_transform_parser import TensorTransformParser


class TestFourDescriptorPipeline(unittest.TestCase):
    """Test the four-descriptor pipeline with comprehensive validation."""
    
    def setUp(self):
        """Set up test data for four-descriptor pipeline."""
        self.code = '''
        constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
            make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
                       number<K0PerThreadWrite>{},
                       number<KThreadReadPerm * N1>{},
                       number<kfold * N0 / npair>{},
                       number<npair>{},
                       BK1));

        constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
            b_lds_block_desc,
            make_tuple(
                make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                make_pass_through_transform(number<K0PerThreadWrite>{}),
                make_xor_transform(
                    make_tuple(number<KThreadReadPerm * N1>{}, number<kfold * N0 / npair>{})),
                make_pass_through_transform(number<npair>{}),
                make_pass_through_transform(BK1)),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

        constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
            b_lds_block_desc_permuted,
            make_tuple(
                make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                make_pass_through_transform(number<K0PerThreadWrite>{}),
                make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<N1>{})),
                make_unmerge_transform(make_tuple(number<kfold>{}, number<N0 / npair>{})),
                make_pass_through_transform(number<npair>{}),
                make_pass_through_transform(BK1)),
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
                       sequence<7>{}));

        constexpr auto b_lds_block_desc_kn = transform_tensor_descriptor(
            b_lds_block_desc_unmerged,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<KThreadReadPerm>{},
                                      number<KThreadWrite / kfold / KThreadReadPerm>{},
                                      number<kfold>{},
                                      number<K0PerThreadWrite>{},
                                      BK1)),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<N0 / npair>{}, number<npair>{}, number<N1>{}))),
            make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));
        '''
        
        self.test_vars = {
            'KThreadWrite': 32, 'kfold': 2, 'KThreadReadPerm': 4, 'K0PerThreadWrite': 2,
            'N1': 16, 'N0': 64, 'npair': 2, 'BK1': 8
        }
        
        self.descriptors = extract_descriptors_from_text(self.code)

    def test_descriptor_extraction(self):
        """Test that all four descriptors are extracted correctly."""
        self.assertEqual(len(self.descriptors), 4)
        # Check that descriptors contain expected patterns (variable names are stripped during extraction)
        self.assertIn('make_naive_tensor_descriptor_packed', self.descriptors[0])
        self.assertIn('b_lds_block_desc', self.descriptors[1])  # References first descriptor
        self.assertIn('b_lds_block_desc_permuted', self.descriptors[2])  # References second descriptor
        self.assertIn('b_lds_block_desc_unmerged', self.descriptors[3])  # References third descriptor

    def test_descriptor_dimensions(self):
        """Test that descriptors have expected dimension counts."""
        # Expected progression: [6, 6, 8, 2]
        expected_dims = [6, 6, 8, 2]
        
        parser = TensorTransformParser()
        descriptor_registry = {}
        
        for i, desc_str in enumerate(self.descriptors):
            parser.descriptor_registry = descriptor_registry
            tensor_desc = parser.create_pytensor_descriptor(desc_str, self.test_vars)
            actual_dims = tensor_desc.get_num_of_dimension()
            
            self.assertEqual(actual_dims, expected_dims[i], 
                           f"Descriptor {i} should have {expected_dims[i]} dimensions, got {actual_dims}")
            
            # Register for next descriptor
            if i < 4:
                var_names = ['b_lds_block_desc', 'b_lds_block_desc_permuted', 
                           'b_lds_block_desc_unmerged', 'b_lds_block_desc_kn']
                if i < len(var_names):
                    descriptor_registry[var_names[i]] = tensor_desc

    def test_forward_graph_generation(self):
        """Test that forward graph generates without errors."""
        dot = build_transformation_graph_from_pytensor(self.descriptors, self.test_vars)
        self.assertIsNotNone(dot)
        self.assertIn('digraph', dot.source.lower())

    def test_backward_graph_generation(self):
        """Test that backward graph generates without errors."""
        dot = build_backward_transformation_graph_from_pytensor(self.descriptors, self.test_vars)
        self.assertIsNotNone(dot)
        self.assertIn('digraph', dot.source.lower())

    def test_forward_graph_input_nodes(self):
        """Test that forward graph has correct number of input nodes."""
        dot = build_transformation_graph_from_pytensor(self.descriptors, self.test_vars)
        # After storage perspective change: naive_packed descriptor has 1 storage input
        storage_input_nodes = len([line for line in dot.source.split('\n') 
                                 if 'input_storage [label=' in line and 'node' not in line])
        logical_input_nodes = len([line for line in dot.source.split('\n') 
                                 if 'input_d' in line and '[label=' in line and 'node' not in line])
        total_input_nodes = storage_input_nodes + logical_input_nodes
        # Expected: 1 storage input + logical inputs from subsequent descriptors
        self.assertGreaterEqual(total_input_nodes, 1, f"Expected at least 1 input node (storage), got {total_input_nodes}")
        self.assertEqual(storage_input_nodes, 1, f"Expected 1 storage input node, got {storage_input_nodes}")

    def test_forward_graph_output_nodes(self):
        """Test that forward graph has correct number of output nodes."""
        dot = build_transformation_graph_from_pytensor(self.descriptors, self.test_vars)
        output_nodes = len([line for line in dot.source.split('\n') 
                          if 'forward_output_d' in line and '[label=' in line])
        self.assertEqual(output_nodes, 2, f"Expected 2 forward output nodes, got {output_nodes}")

    def test_backward_graph_start_nodes(self):
        """Test that backward graph has correct number of start nodes."""
        dot = build_backward_transformation_graph_from_pytensor(self.descriptors, self.test_vars)
        start_nodes = len([line for line in dot.source.split('\n') 
                         if 'backward_start_d' in line and '[label=' in line and '->' not in line])
        # Backward graph starts with final descriptor's output dimensions (2)
        self.assertEqual(start_nodes, 2, f"Expected 2 backward start nodes, got {start_nodes}")

    def test_backward_graph_output_nodes(self):
        """Test that backward graph has correct number of output nodes."""
        dot = build_backward_transformation_graph_from_pytensor(self.descriptors, self.test_vars)
        # After storage perspective change: naive_packed descriptor has 1 storage output
        storage_output_nodes = len([line for line in dot.source.split('\n') 
                                  if 'backward_output_storage [label=' in line])
        logical_output_nodes = len([line for line in dot.source.split('\n') 
                                  if 'backward_output_d' in line and '[label=' in line])
        total_output_nodes = storage_output_nodes + logical_output_nodes
        # Expected: 1 storage output (consistent with forward storage input)
        self.assertGreaterEqual(total_output_nodes, 1, f"Expected at least 1 output node (storage), got {total_output_nodes}")
        self.assertEqual(storage_output_nodes, 1, f"Expected 1 storage output node, got {storage_output_nodes}")

    def test_inverse_notation_in_backward_graph(self):
        """Test that backward graph uses inverse notation (⁻¹)."""
        dot = build_backward_transformation_graph_from_pytensor(self.descriptors, self.test_vars)
        dot_source = dot.source
        
        # Check for inverse notation
        self.assertIn('⁻¹', dot_source, "Backward graph should contain inverse notation (⁻¹)")
        
        # Check for specific inverse transforms
        inverse_transforms = ['PassThrough⁻¹', 'Xor⁻¹', 'Merge⁻¹', 'Unmerge⁻¹']
        found_inverse = False
        for transform in inverse_transforms:
            if transform in dot_source:
                found_inverse = True
                break
        
        self.assertTrue(found_inverse, f"Should find at least one inverse transform from {inverse_transforms}")

    def test_dimension_progression(self):
        """Test the expected dimension progression through the pipeline."""
        parser = TensorTransformParser()
        descriptor_registry = {}
        
        expected_progression = [6, 6, 8, 2]
        actual_progression = []
        
        for i, desc_str in enumerate(self.descriptors):
            parser.descriptor_registry = descriptor_registry
            tensor_desc = parser.create_pytensor_descriptor(desc_str, self.test_vars)
            actual_progression.append(tensor_desc.get_num_of_dimension())
            
            # Register for next descriptor
            var_names = ['b_lds_block_desc', 'b_lds_block_desc_permuted', 
                        'b_lds_block_desc_unmerged', 'b_lds_block_desc_kn']
            if i < len(var_names):
                descriptor_registry[var_names[i]] = tensor_desc
        
        self.assertEqual(actual_progression, expected_progression,
                        f"Expected dimension progression {expected_progression}, got {actual_progression}")

    def test_transform_types_in_descriptors(self):
        """Test that descriptors contain expected transform types."""
        parser = TensorTransformParser()
        descriptor_registry = {}
        
        expected_transform_counts = [1, 6, 12, 14]  # Cumulative transform counts
        
        for i, desc_str in enumerate(self.descriptors):
            parser.descriptor_registry = descriptor_registry
            tensor_desc = parser.create_pytensor_descriptor(desc_str, self.test_vars)
            
            transform_count = len(tensor_desc.get_transforms())
            self.assertEqual(transform_count, expected_transform_counts[i],
                           f"Descriptor {i} should have {expected_transform_counts[i]} transforms, got {transform_count}")
            
            # Register for next descriptor
            var_names = ['b_lds_block_desc', 'b_lds_block_desc_permuted', 
                        'b_lds_block_desc_unmerged', 'b_lds_block_desc_kn']
            if i < len(var_names):
                descriptor_registry[var_names[i]] = tensor_desc

    def test_naive_descriptor_properties(self):
        """Test properties of the naive descriptor."""
        parser = TensorTransformParser()
        
        tensor_desc = parser.create_pytensor_descriptor(self.descriptors[0], self.test_vars)
        
        # Should have 6 dimensions
        self.assertEqual(tensor_desc.get_num_of_dimension(), 6)
        
        # Should have expected lengths
        expected_lengths = [4, 2, 64, 64, 2, 8]  # Based on test variables
        actual_lengths = [tensor_desc.get_length(i) for i in range(6)]
        self.assertEqual(actual_lengths, expected_lengths)

    def test_final_descriptor_properties(self):
        """Test properties of the final descriptor."""
        parser = TensorTransformParser()
        descriptor_registry = {}
        
        # Build all descriptors in sequence
        for i, desc_str in enumerate(self.descriptors):
            parser.descriptor_registry = descriptor_registry
            tensor_desc = parser.create_pytensor_descriptor(desc_str, self.test_vars)
            
            var_names = ['b_lds_block_desc', 'b_lds_block_desc_permuted', 
                        'b_lds_block_desc_unmerged', 'b_lds_block_desc_kn']
            if i < len(var_names):
                descriptor_registry[var_names[i]] = tensor_desc
        
        # Final descriptor should have 2 dimensions
        final_desc = tensor_desc
        self.assertEqual(final_desc.get_num_of_dimension(), 2)

    def test_graph_consistency(self):
        """Test that forward and backward graphs are consistent."""
        forward_dot = build_transformation_graph_from_pytensor(self.descriptors, self.test_vars)
        backward_dot = build_backward_transformation_graph_from_pytensor(self.descriptors, self.test_vars)
        
        # Count storage and logical nodes separately after storage perspective change
        forward_storage_inputs = len([line for line in forward_dot.source.split('\n') 
                                    if 'input_storage [label=' in line and 'node' not in line])
        backward_storage_outputs = len([line for line in backward_dot.source.split('\n') 
                                      if 'backward_output_storage [label=' in line])
        forward_outputs = len([line for line in forward_dot.source.split('\n') 
                             if 'forward_output_d' in line and '[label=' in line])
        backward_inputs = len([line for line in backward_dot.source.split('\n') 
                             if 'backward_start_d' in line and '[label=' in line and '->' not in line])
        
        # Test storage consistency: forward storage input should match backward storage output
        self.assertEqual(forward_storage_inputs, backward_storage_outputs, 
                        "Forward storage inputs should match backward storage outputs")
        
        # Test pipeline consistency: forward outputs should equal backward inputs
        self.assertEqual(forward_outputs, backward_inputs,
                        f"Forward outputs ({forward_outputs}) should match backward inputs ({backward_inputs})")
        
        # Final descriptor should have 2 dimensions
        self.assertEqual(forward_outputs, 2, "Forward graph should have 2 final outputs (from last descriptor)")
        self.assertEqual(backward_inputs, 2, "Backward graph should start with 2 inputs (from last descriptor)")

if __name__ == '__main__':
    unittest.main() 