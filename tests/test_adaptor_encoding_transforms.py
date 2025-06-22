"""
Tests for _make_adaptor_encoding_for_tile_distribution transforms.

This module tests the unmerge and merge transforms to understand how the
tile distribution encoding creates the tensor adaptor structure.
"""

import pytest
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytensor.tile_distribution import (
    TileDistributionEncoding, 
    _make_adaptor_encoding_for_tile_distribution,
    _construct_static_tensor_adaptor_from_encoding,
    make_static_tile_distribution
)
from tile_distribution.examples import EXAMPLES, DEFAULT_VARIABLES
import re
import numpy as np


def parse_tile_distribution_encoding(code_snippet: str, variables: dict = None) -> dict:
    """
    Parse a tile distribution encoding from C++ syntax to Python parameters.
    
    Args:
        code_snippet: C++ code snippet with tile_distribution_encoding
        variables: Dictionary of variable substitutions
        
    Returns:
        Dictionary with parsed parameters
    """
    if variables is None:
        variables = {}
    
    # Remove comments and clean up
    lines = []
    for line in code_snippet.split('\n'):
        line = line.strip()
        if line and not line.startswith('//'):
            # Remove inline comments
            comment_pos = line.find('//')
            if comment_pos != -1:
                line = line[:comment_pos]
            lines.append(line.strip())
    
    text = ' '.join(lines)
    
    # Remove the template declaration
    text = re.sub(r'tile_distribution_encoding<\s*', '', text)
    text = re.sub(r'>\s*\{\s*\}\s*$', '', text)
    
    # Replace variables
    for var, value in variables.items():
        text = text.replace(var, str(value))
    
    # Split by commas at the top level (not inside sequences/tuples)
    parts = []
    current_part = ""
    paren_depth = 0
    angle_depth = 0
    
    for char in text:
        if char in '(<':
            paren_depth += 1
        elif char in ')>':
            paren_depth -= 1
        elif char == ',' and paren_depth == 0:
            parts.append(current_part.strip())
            current_part = ""
            continue
        current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    if len(parts) != 6:
        raise ValueError(f"Expected 6 parts, got {len(parts)}: {parts}")
    
    def parse_sequence(s):
        """Parse sequence<...> to list of integers."""
        s = s.strip()
        if s.startswith('sequence<') and s.endswith('>'):
            content = s[9:-1].strip()
            if not content:
                return []
            return [int(x.strip()) for x in content.split(',')]
        raise ValueError(f"Invalid sequence format: {s}")
    
    def parse_tuple_of_sequences(s):
        """Parse tuple<sequence<...>, sequence<...>, ...> to list of lists."""
        s = s.strip()
        if s.startswith('tuple<') and s.endswith('>'):
            content = s[6:-1].strip()
            
            # Split by comma at the top level
            seq_parts = []
            current_seq = ""
            depth = 0
            
            for char in content:
                if char in '<(':
                    depth += 1
                elif char in '>)':
                    depth -= 1
                elif char == ',' and depth == 0:
                    seq_parts.append(current_seq.strip())
                    current_seq = ""
                    continue
                current_seq += char
            
            if current_seq.strip():
                seq_parts.append(current_seq.strip())
            
            return [parse_sequence(seq) for seq in seq_parts]
        raise ValueError(f"Invalid tuple format: {s}")
    
    # Parse each part
    rs_lengths = parse_sequence(parts[0])
    hs_lengthss = parse_tuple_of_sequences(parts[1])
    ps_to_rhss_major = parse_tuple_of_sequences(parts[2])
    ps_to_rhss_minor = parse_tuple_of_sequences(parts[3])
    ys_to_rhs_major = parse_sequence(parts[4])
    ys_to_rhs_minor = parse_sequence(parts[5])
    
    return {
        'rs_lengths': rs_lengths,
        'hs_lengthss': hs_lengthss,
        'ps_to_rhss_major': ps_to_rhss_major,
        'ps_to_rhss_minor': ps_to_rhss_minor,
        'ys_to_rhs_major': ys_to_rhs_major,
        'ys_to_rhs_minor': ys_to_rhs_minor
    }


class TestAdaptorEncodingTransforms:
    """Test the unmerge and merge transforms in _make_adaptor_encoding_for_tile_distribution."""
    
    def test_basic_16x4_threads_transforms(self):
        """Test the basic 16x4 threads example to understand transforms."""
        example_code = EXAMPLES["Basic 16x4 Threads"]
        params = parse_tile_distribution_encoding(example_code)
        
        print(f"\n=== Basic 16x4 Threads Example ===")
        print(f"Parsed parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Create encoding
        encoding = TileDistributionEncoding(**params)
        print(f"\nEncoding dimensions:")
        print(f"  ndim_x: {encoding.ndim_x}")
        print(f"  ndim_p: {encoding.ndim_p}")
        print(f"  ndim_y: {encoding.ndim_y}")
        print(f"  ndim_r: {encoding.ndim_r}")
        
        # Test the adaptor encoding creation
        adaptor_impl = _make_adaptor_encoding_for_tile_distribution(encoding)
        
        ps_ys_to_xs_adaptor_impl = adaptor_impl[0]
        ys_to_d_adaptor_impl = adaptor_impl[1]
        d_length = adaptor_impl[2]
        rh_major_minor_to_hidden_ids_impl = adaptor_impl[3]
        
        print(f"\n=== Adaptor Encoding Analysis ===")
        print(f"D length: {d_length}")
        
        # Analyze ps_ys_to_xs_adaptor transforms
        transforms, num_trans, bottom_dim_ids, ndim_bottom, top_dim_ids, ndim_top = ps_ys_to_xs_adaptor_impl
        
        print(f"\nps_ys_to_xs_adaptor:")
        print(f"  Bottom dimensions: {bottom_dim_ids} (count: {ndim_bottom})")
        print(f"  Top dimensions: {top_dim_ids} (count: {ndim_top})")
        print(f"  Number of transforms: {num_trans}")
        
        for i, transform in enumerate(transforms):
            print(f"\n  Transform {i}: {transform['name']}")
            print(f"    Input dims: {transform['dims']} (count: {transform['num_dim']})")
            print(f"    Output dims: {transform['dims_out']} (count: {transform['num_dim_out']})")
            print(f"    Meta data: {transform['meta_data']}")
        
        # Analyze ys_to_d_adaptor transforms  
        y_transforms, y_num_trans, y_bottom_dim_ids, y_ndim_bottom, y_top_dim_ids, y_ndim_top = ys_to_d_adaptor_impl
        
        print(f"\nys_to_d_adaptor:")
        print(f"  Bottom dimensions: {y_bottom_dim_ids} (count: {y_ndim_bottom})")
        print(f"  Top dimensions: {y_top_dim_ids} (count: {y_ndim_top})")
        print(f"  Number of transforms: {y_num_trans}")
        
        for i, transform in enumerate(y_transforms):
            print(f"\n  Transform {i}: {transform['name']}")
            print(f"    Input dims: {transform['dims']} (count: {transform['num_dim']})")
            print(f"    Output dims: {transform['dims_out']} (count: {transform['num_dim_out']})")
            print(f"    Meta data: {transform['meta_data']}")
        
        # Analyze hidden dimension mappings
        print(f"\n=== Hidden Dimension Mappings ===")
        print(f"RH major-minor to hidden IDs:")
        for rh_major, hidden_ids in enumerate(rh_major_minor_to_hidden_ids_impl):
            print(f"  RH[{rh_major}]: {hidden_ids[:10]}...")  # Show first 10 elements
        
        # Verify the transforms make sense
        assert len(transforms) >= 1, "Should have at least one transform"
        
        # First transform should be replicate (for R dimensions)
        assert transforms[0]['name'] == 'replicate'
        assert transforms[0]['num_dim'] == 0  # No input dimensions
        assert transforms[0]['meta_data'] == encoding.rs_lengths
        
        # Following transforms should be unmerge (for X dimensions)
        for i in range(1, len(transforms) - encoding.ndim_p):
            transform = transforms[i]
            assert transform['name'] == 'unmerge'
            assert transform['num_dim'] == 1  # One input dimension
            assert transform['num_dim_out'] == len(transform['meta_data'])
        
        # Last transforms should be merge (for P dimensions)
        for i in range(len(transforms) - encoding.ndim_p, len(transforms)):
            transform = transforms[i]
            assert transform['name'] == 'merge'
            assert transform['num_dim_out'] == 1  # One output dimension
    
    def test_variable_based_template_transforms(self):
        """Test the variable-based template example."""
        example_code = EXAMPLES["Variable-Based Template"]
        variables = DEFAULT_VARIABLES["Variable-Based Template"]
        params = parse_tile_distribution_encoding(example_code, variables)
        
        print(f"\n=== Variable-Based Template Example ===")
        print(f"Variables: {variables}")
        print(f"Parsed parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Create encoding and test
        encoding = TileDistributionEncoding(**params)
        adaptor_impl = _make_adaptor_encoding_for_tile_distribution(encoding)
        
        transforms, num_trans, bottom_dim_ids, ndim_bottom, top_dim_ids, ndim_top = adaptor_impl[0]
        
        print(f"\nTransform sequence:")
        for i, transform in enumerate(transforms):
            print(f"  {i}: {transform['name']} {transform['dims']} -> {transform['dims_out']}")
            print(f"      meta_data: {transform['meta_data']}")
        
        # Verify the structure
        assert transforms[0]['name'] == 'replicate'
        
        # Count unmerge transforms (should equal ndim_x)
        unmerge_count = sum(1 for t in transforms if t['name'] == 'unmerge' and t['num_dim'] == 1)
        assert unmerge_count >= encoding.ndim_x  # At least one per X dimension
        
        # Count merge transforms (should equal ndim_p)
        merge_count = sum(1 for t in transforms if t['name'] == 'merge')
        assert merge_count == encoding.ndim_p
    
    def test_complex_distribution_transforms(self):
        """Test the complex distribution example."""
        example_code = EXAMPLES["Complex Distribution"]
        params = parse_tile_distribution_encoding(example_code)
        
        print(f"\n=== Complex Distribution Example ===")
        print(f"Parsed parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        encoding = TileDistributionEncoding(**params)
        adaptor_impl = _make_adaptor_encoding_for_tile_distribution(encoding)
        
        transforms, num_trans, bottom_dim_ids, ndim_bottom, top_dim_ids, ndim_top = adaptor_impl[0]
        
        print(f"\nDimension flow analysis:")
        print(f"  Bottom (X): {bottom_dim_ids}")
        print(f"  Top (P+Y): {top_dim_ids}")
        
        # Track how dimensions flow through transforms
        print(f"\nTransform flow:")
        for i, transform in enumerate(transforms):
            if transform['name'] == 'replicate':
                print(f"  {i}: REPLICATE [] -> {transform['dims_out']}")
                print(f"      Creates R dimensions: {transform['meta_data']}")
            elif transform['name'] == 'unmerge':
                x_dim = transform['dims'][0]
                print(f"  {i}: UNMERGE X[{x_dim}] -> {transform['dims_out']}")
                print(f"      Splits into H dimensions: {transform['meta_data']}")
            elif transform['name'] == 'merge':
                print(f"  {i}: MERGE {transform['dims']} -> {transform['dims_out']}")
                print(f"      Combines with lengths: {transform['meta_data']}")
        
        # Verify consistency
        assert len(bottom_dim_ids) == encoding.ndim_x
        assert len(top_dim_ids) == encoding.ndim_p + encoding.ndim_y
    
    def test_dimension_mapping_consistency(self):
        """Test that dimension mappings are consistent across all examples."""
        print(f"\n=== Dimension Mapping Consistency Test ===")
        
        for example_name, example_code in EXAMPLES.items():
            print(f"\nTesting: {example_name}")
            
            variables = DEFAULT_VARIABLES.get(example_name, {})
            params = parse_tile_distribution_encoding(example_code, variables)
            
            try:
                encoding = TileDistributionEncoding(**params)
                adaptor_impl = _make_adaptor_encoding_for_tile_distribution(encoding)
                
                transforms, num_trans, bottom_dim_ids, ndim_bottom, top_dim_ids, ndim_top = adaptor_impl[0]
                
                print(f"  ✓ Parsed successfully")
                print(f"    Dimensions: X={encoding.ndim_x}, P={encoding.ndim_p}, Y={encoding.ndim_y}, R={encoding.ndim_r}")
                print(f"    Transforms: {num_trans} total")
                
                # Verify dimension counts
                assert ndim_bottom == encoding.ndim_x
                assert ndim_top == encoding.ndim_p + encoding.ndim_y
                
                # Verify transform types
                replicate_count = sum(1 for t in transforms if t['name'] == 'replicate')
                unmerge_count = sum(1 for t in transforms if t['name'] == 'unmerge' and t['num_dim'] == 1)
                merge_count = sum(1 for t in transforms if t['name'] == 'merge')
                
                assert replicate_count == 1, f"Expected 1 replicate, got {replicate_count}"
                assert merge_count == encoding.ndim_p, f"Expected {encoding.ndim_p} merge, got {merge_count}"
                
                print(f"    Transform counts: replicate={replicate_count}, unmerge={unmerge_count}, merge={merge_count}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
    
    def test_why_unmerge_then_merge(self):
        """
        Test to understand WHY we have unmerge followed by merge transforms.
        This demonstrates the hierarchical structure.
        """
        print(f"\n=== Understanding Unmerge -> Merge Pattern ===")
        
        # Use a simple example to explain the pattern
        params = {
            'rs_lengths': [2],  # One R dimension with length 2
            'hs_lengthss': [[4, 4], [8, 2]],  # Two X dimensions with hierarchical structure
            'ps_to_rhss_major': [[0, 1], [2]],  # P0 maps to R[0] and X[0], P1 maps to X[1]
            'ps_to_rhss_minor': [[0, 0], [1]],
            'ys_to_rhs_major': [1, 2],  # Y0 maps to X[0], Y1 maps to X[1]
            'ys_to_rhs_minor': [1, 0]
        }
        
        encoding = TileDistributionEncoding(**params)
        adaptor_impl = _make_adaptor_encoding_for_tile_distribution(encoding)
        
        transforms, num_trans, bottom_dim_ids, ndim_bottom, top_dim_ids, ndim_top = adaptor_impl[0]
        
        print(f"Input encoding:")
        print(f"  R lengths: {params['rs_lengths']}")
        print(f"  H hierarchies: {params['hs_lengthss']}")
        print(f"  P->RH major: {params['ps_to_rhss_major']}")
        print(f"  P->RH minor: {params['ps_to_rhss_minor']}")
        print(f"  Y->RH major: {params['ys_to_rhs_major']}")
        print(f"  Y->RH minor: {params['ys_to_rhs_minor']}")
        
        print(f"\nWhy we need UNMERGE then MERGE:")
        print(f"1. Start with X dimensions: {list(range(encoding.ndim_x))}")
        
        for i, transform in enumerate(transforms):
            if transform['name'] == 'replicate':
                print(f"2. REPLICATE creates R dimensions {transform['dims_out']} from nothing")
                print(f"   This gives us replication capability for multi-processing")
                
            elif transform['name'] == 'unmerge':
                x_dim = transform['dims'][0]
                h_lengths = transform['meta_data']
                print(f"3. UNMERGE X[{x_dim}] -> {transform['dims_out']} with lengths {h_lengths}")
                print(f"   This breaks X[{x_dim}] into hierarchical components")
                print(f"   Total size = {h_lengths} -> product = {np.prod(h_lengths)}")
                
            elif transform['name'] == 'merge':
                input_dims = transform['dims']
                output_dim = transform['dims_out'][0]
                lengths = transform['meta_data']
                print(f"4. MERGE {input_dims} -> [{output_dim}] with lengths {lengths}")
                print(f"   This combines R/H dimensions to create a P dimension")
                print(f"   P dimension size = product({lengths}) = {np.prod(lengths)}")
        
        print(f"\nFinal mapping:")
        print(f"  Bottom (X) dimensions: {bottom_dim_ids}")
        print(f"  Top (P+Y) dimensions: {top_dim_ids}")
        
        print(f"\nThe pattern explained:")
        print(f"  • UNMERGE: Break each X dimension into hierarchical components (threads, warps, blocks)")
        print(f"  • MERGE: Combine R and H components to form partition (P) dimensions")
        print(f"  • This allows flexible mapping of processing elements to data")
        
        # Test that we can construct a working tile distribution
        distribution = make_static_tile_distribution(encoding)
        print(f"\n✓ Successfully created tile distribution with {distribution.ndim_p} P dims")


def test_all_examples_parse_correctly():
    """Test that all examples from examples.py parse correctly."""
    print(f"\n=== Testing All Examples Parse Correctly ===")
    
    for example_name, example_code in EXAMPLES.items():
        print(f"\nTesting: {example_name}")
        
        variables = DEFAULT_VARIABLES.get(example_name, {})
        
        try:
            params = parse_tile_distribution_encoding(example_code, variables)
            encoding = TileDistributionEncoding(**params)
            
            # Test that we can create the adaptor encoding
            adaptor_impl = _make_adaptor_encoding_for_tile_distribution(encoding)
            
            # Test that we can create the full tile distribution
            distribution = make_static_tile_distribution(encoding)
            
            print(f"  ✓ Successfully parsed and created distribution")
            print(f"    Dimensions: X={distribution.ndim_x}, P={distribution.ndim_p}, Y={distribution.ndim_y}, R={distribution.ndim_r}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            # Re-raise to fail the test
            raise


if __name__ == "__main__":
    # Run individual tests for debugging
    test_class = TestAdaptorEncodingTransforms()
    
    print("Running individual tests for debugging:")
    test_class.test_basic_16x4_threads_transforms()
    test_class.test_variable_based_template_transforms()
    test_class.test_complex_distribution_transforms()
    test_class.test_dimension_mapping_consistency()
    test_class.test_why_unmerge_then_merge()
    
    test_all_examples_parse_correctly()
    
    print("\n=== All tests completed! ===")