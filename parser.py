"""
Parser for tile_distribution_encoding structures in Composable Kernels.

This module contains functions for parsing C++ template parameters from
tile_distribution_encoding structures used in the Composable Kernels library.
"""

import re
from typing import Dict, List, Tuple, Any, Optional, Union
import ast
import json

class TileDistributionParser:
    """Parser for tile_distribution_encoding structures."""
    
    @staticmethod
    def parse_sequence(sequence_str: str) -> List[Any]:
        """Parse a C++ sequence<x, y, z> string into a list of integers or strings.
        
        Args:
            sequence_str: String like 'sequence<1, 2, 3>'
            
        Returns:
            List of integers or strings, e.g. [1, 2, 3]
        """
        # First, ensure we have a sequence pattern
        if not "sequence<" in sequence_str:
            return []
            
        # Extract everything between the outermost angle brackets
        start_idx = sequence_str.find("sequence<") + len("sequence<")
        content = ""
        bracket_level = 1  # We're already inside one bracket
        
        for i in range(start_idx, len(sequence_str)):
            char = sequence_str[i]
            if char == '<':
                bracket_level += 1
                content += char
            elif char == '>':
                bracket_level -= 1
                if bracket_level == 0:  # End of the sequence
                    break
                content += char
            else:
                content += char
                
        if not content.strip():
            return []
            
        # Split the content by commas, handling nested structures
        values = []
        bracket_level = 0
        current_value = ""
        
        for char in content:
            if char == '<':
                bracket_level += 1
                current_value += char
            elif char == '>':
                bracket_level -= 1
                current_value += char
            elif char == ',' and bracket_level == 0:
                values.append(current_value.strip())
                current_value = ""
            else:
                current_value += char
                
        if current_value.strip():
            values.append(current_value.strip())
            
        # Process each value
        result = []
        for val in values:
            val = val.strip()
            if not val:
                continue
                
            # Try to convert to integer
            try:
                result.append(int(val))
            except ValueError:
                # Check if it's a nested structure
                if "sequence<" in val:
                    result.append(TileDistributionParser.parse_sequence(val))
                elif "tuple<" in val:
                    result.append(TileDistributionParser.parse_tuple(val))
                else:
                    # Must be a variable name
                    result.append(val)
                    
        return result

    @staticmethod
    def parse_tuple(tuple_str: str) -> List[Any]:
        """Parse a C++ tuple<...> string into a list of Python objects.
        
        Args:
            tuple_str: String like 'tuple<sequence<1, 2>, sequence<3, 4>>'
            
        Returns:
            List of parsed objects
        """
        # First, ensure we have a tuple pattern
        if not "tuple<" in tuple_str:
            return []
            
        # Extract everything between the outermost angle brackets
        start_idx = tuple_str.find("tuple<") + len("tuple<")
        content = ""
        bracket_level = 1  # We're already inside one bracket
        
        for i in range(start_idx, len(tuple_str)):
            char = tuple_str[i]
            if char == '<':
                bracket_level += 1
                content += char
            elif char == '>':
                bracket_level -= 1
                if bracket_level == 0:  # End of the tuple
                    break
                content += char
            else:
                content += char
                
        if not content.strip():
            return []
            
        # Split the content by commas, handling nested structures
        items = []
        bracket_level = 0
        current_item = ""
        
        for char in content:
            if char == '<':
                bracket_level += 1
                current_item += char
            elif char == '>':
                bracket_level -= 1
                current_item += char
            elif char == ',' and bracket_level == 0:
                items.append(current_item.strip())
                current_item = ""
            else:
                current_item += char
                
        if current_item.strip():
            items.append(current_item.strip())
            
        # Process each item
        result = []
        for item in items:
            item = item.strip()
            if not item:
                continue
                
            if "sequence<" in item:
                result.append(TileDistributionParser.parse_sequence(item))
            elif "tuple<" in item:
                result.append(TileDistributionParser.parse_tuple(item))
            else:
                result.append(item)
                
        return result

    @staticmethod
    def parse_tile_distribution_encoding(cpp_code: str) -> Dict[str, Any]:
        """Parse a tile_distribution_encoding structure from C++ code.
        
        Args:
            cpp_code: C++ code containing tile_distribution_encoding
            
        Returns:
            Dictionary with parsed values
        """
        # First, ensure we have a tile_distribution_encoding pattern
        if not "tile_distribution_encoding<" in cpp_code:
            return {}
            
        # Extract everything between the outermost angle brackets
        start_idx = cpp_code.find("tile_distribution_encoding<") + len("tile_distribution_encoding<")
        content = ""
        bracket_level = 1  # We're already inside one bracket
        
        for i in range(start_idx, len(cpp_code)):
            char = cpp_code[i]
            if char == '<':
                bracket_level += 1
                content += char
            elif char == '>':
                bracket_level -= 1
                if bracket_level == 0:  # End of the encoding
                    break
                content += char
            else:
                content += char
                
        if not content.strip():
            return {}
            
        # Split the content by commas, handling nested structures
        components = []
        bracket_level = 0
        current_component = ""
        
        for char in content:
            if char == '<':
                bracket_level += 1
                current_component += char
            elif char == '>':
                bracket_level -= 1
                current_component += char
            elif char == ',' and bracket_level == 0:
                components.append(current_component.strip())
                current_component = ""
            else:
                current_component += char
                
        if current_component.strip():
            components.append(current_component.strip())
            
        # Check if we have enough components
        if len(components) < 6:
            return {}
            
        # Parse the components
        rs_lengths = TileDistributionParser.parse_sequence(components[0])
        hs_lengthss = TileDistributionParser.parse_tuple(components[1])
        ps_rhss_major = TileDistributionParser.parse_tuple(components[2])
        ps_rhss_minor = TileDistributionParser.parse_tuple(components[3])
        ys_rhs_major = TileDistributionParser.parse_sequence(components[4])
        ys_rhs_minor = TileDistributionParser.parse_sequence(components[5])
        
        # Build the result dictionary
        result = {
            "RsLengths": rs_lengths,
            "HsLengthss": hs_lengthss,
            "Ps2RHssMajor": ps_rhss_major,
            "Ps2RHssMinor": ps_rhss_minor,
            "Ys2RHsMajor": ys_rhs_major,
            "Ys2RHsMinor": ys_rhs_minor
        }
        
        # Extract variable names
        variables = set()
        
        def extract_variables(obj):
            """Recursively extract variable names from parsed structures."""
            if isinstance(obj, str) and re.match(r'^[A-Za-z][A-Za-z0-9_]*$', obj):
                # Skip common keywords
                if obj not in ["sequence", "tuple", "int", "float", "double", "char", "bool", 
                           "auto", "void", "const", "static", "constexpr", "index_t",
                           "R", "H", "X0", "X1", "p", "major", "minor", "Y", "y"]:
                    variables.add(obj)
            elif isinstance(obj, list):
                for item in obj:
                    extract_variables(item)
                    
        # Extract variables from all components
        extract_variables(rs_lengths)
        extract_variables(hs_lengthss)
        extract_variables(ps_rhss_major)
        extract_variables(ps_rhss_minor)
        extract_variables(ys_rhs_major)
        extract_variables(ys_rhs_minor)
        
        result["variable_names"] = list(variables)
        
        return result
    
    @staticmethod
    def extract_template_variables(cpp_code: str) -> Dict[str, Any]:
        """Extract template variables used in the tile_distribution_encoding.
        
        Args:
            cpp_code: C++ code containing tile_distribution_encoding
            
        Returns:
            Dictionary mapping variable names to their values
        """
        variables = {}
        
        # Find all variable declarations like "constexpr index_t K0 = 16;"
        variable_decls = re.findall(r'constexpr\s+index_t\s+([A-Za-z0-9_]+)\s*=\s*(\d+);', cpp_code)
        
        for name, value in variable_decls:
            variables[name] = int(value)
        
        # Look for struct/namespace-scoped variables like S::Repeat_M
        scoped_vars = re.findall(r'([A-Za-z0-9_]+)::([A-Za-z0-9_]+)', cpp_code)
        
        # Extract all variable references in sequences
        sequence_vars = re.findall(r'sequence<([^>]+)>', cpp_code)
        all_vars = []
        for seq in sequence_vars:
            var_candidates = seq.split(',')
            for var in var_candidates:
                var = var.strip()
                if not var:
                    continue
                # Check if it's not a number
                try:
                    int(var)
                except ValueError:
                    # If it looks like a variable name or scoped variable
                    if re.match(r'^[A-Za-z0-9_:]+$', var):
                        all_vars.append(var)
        
        # Add default values for scoped variables and any vars found in sequences
        for var in all_vars:
            if var not in variables:
                if '::' in var:
                    # Extract the variable name after the scope
                    var_name = var.split('::')[-1]
                    variables[var] = 4  # Default value
                elif re.match(r'^[A-Za-z][A-Za-z0-9_]*$', var):
                    variables[var] = 4  # Default value
        
        # Also add any scoped variables
        for scope, var in scoped_vars:
            scoped_name = f"{scope}::{var}"
            if scoped_name not in variables:
                variables[scoped_name] = 4  # Default value
        
        return variables

def parse_from_file(filename: str) -> Dict[str, Any]:
    """Parse tile_distribution_encoding from a file.
    
    Args:
        filename: Path to C++ file
        
    Returns:
        Dictionary with parsed structure
    """
    with open(filename, 'r') as f:
        cpp_code = f.read()
    
    parser = TileDistributionParser()
    encoding = parser.parse_tile_distribution_encoding(cpp_code)
    variables = parser.extract_template_variables(cpp_code)
    
    return {
        "encoding": encoding,
        "variables": variables
    }

def extract_variables_from_encoding(encoding: Dict[str, Any]) -> List[str]:
    """Extract variable names used in the encoding.
    
    Args:
        encoding: Parsed encoding structure
        
    Returns:
        List of variable names
    """
    variables = set()
    
    # Helper function to find variables in a structure
    def find_variables(obj):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, str) and re.match(r'^[A-Za-z][A-Za-z0-9_]*$', item):
                    variables.add(item)
                elif isinstance(item, (list, dict)):
                    find_variables(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                find_variables(value)
    
    find_variables(encoding)
    return list(variables)

def debug_indexing_relationships(parsed_encoding: Dict[str, Any], variables: Dict[str, Any] = None) -> Dict[str, Any]:
    """Creates a debug representation of the indexing relationships in the encoding.
    
    Args:
        parsed_encoding: The parsed tile_distribution_encoding structure
        variables: Dictionary mapping variable names to their values
        
    Returns:
        Dictionary with debug information about indices
        
    This creates a detailed mapping showing:
    - MajorIndices: Shows that R is at major index 0, H0 at 1, H1 at 2, etc.
    - MinorIndices: Shows the actual values of each sequence (R sequence and H sequences)
    - IndexMapping: Shows how each P and Y entry maps to specific elements in R or H sequences
      For example, P0[0] might map to H0[1], meaning it indexes the second element of the first H sequence.
      The Value field shows the actual value at that position when variables are resolved.
    """
    if variables is None:
        variables = {}
        
    result = {
        "MajorIndices": {
            "R": 0,  # R is always at major index 0
        },
        "MinorIndices": {},
        "IndexMapping": {}
    }
    
    # Get the basic structures
    r_lengths = parsed_encoding.get("RsLengths", [])
    hs_lengthss = parsed_encoding.get("HsLengthss", [])
    ps_rhss_major = parsed_encoding.get("Ps2RHssMajor", [])
    ps_rhss_minor = parsed_encoding.get("Ps2RHssMinor", [])
    ys_rhs_major = parsed_encoding.get("Ys2RHsMajor", [])
    ys_rhs_minor = parsed_encoding.get("Ys2RHsMinor", [])
    
    # Add H0, H1, etc. to major indices
    for i, h_lengths in enumerate(hs_lengthss):
        result["MajorIndices"][f"H{i}"] = i + 1
    
    # Store actual values of R and H sequences, but preserve variable names
    r_values = []
    r_orig_values = []
    for val in r_lengths:
        r_orig_values.append(val)
        if isinstance(val, str) and val in variables:
            r_values.append(variables[val])
        else:
            r_values.append(val)
    
    h_values = []
    h_orig_values = []
    for h_seq in hs_lengthss:
        h_seq_values = []
        h_seq_orig = []
        for val in h_seq:
            h_seq_orig.append(val)
            if isinstance(val, str) and val in variables:
                h_seq_values.append(variables.get(val, val))
            else:
                h_seq_values.append(val)
        h_values.append(h_seq_values)
        h_orig_values.append(h_seq_orig)
    
    # Store actual lengths in MinorIndices
    result["MinorIndices"]["R"] = r_orig_values
    for i, h_vals in enumerate(h_orig_values):
        result["MinorIndices"][f"H{i}"] = h_vals
    
    # Add P0, P1, etc. information
    for i, (p_major, p_minor) in enumerate(zip(ps_rhss_major, ps_rhss_minor)):
        p_key = f"P{i}"
        
        # Record P to R/H mappings
        mappings = []
        for j, (maj, min_idx) in enumerate(zip(p_major, p_minor)):
            # Determine target (R or H)
            if maj == 0:  # R dimension
                target = "R"
                orig_value = r_orig_values[min_idx] if 0 <= min_idx < len(r_orig_values) else None
                numeric_value = r_values[min_idx] if 0 <= min_idx < len(r_values) else None
            else:  # H dimension
                target = f"H{maj-1}"
                if 0 <= maj-1 < len(h_orig_values) and 0 <= min_idx < len(h_orig_values[maj-1]):
                    orig_value = h_orig_values[maj-1][min_idx]
                    numeric_value = h_values[maj-1][min_idx]
                else:
                    orig_value = None
                    numeric_value = None
                
            mappings.append({
                "Target": target,
                "MajorIndex": maj,
                "MinorIndex": min_idx,
                "SymbolicValue": orig_value,  # Original symbolic value (variable name)
                "Value": numeric_value  # Numeric value if variable was resolved
            })
            
        result["IndexMapping"][p_key] = mappings
    
    # Add Y0, Y1, etc. information
    for i in range(len(ys_rhs_major)):
        y_key = f"Y{i}"
        
        # Record Y to R/H mappings
        mappings = []
        if i < len(ys_rhs_major) and i < len(ys_rhs_minor):
            maj = ys_rhs_major[i]
            min_idx = ys_rhs_minor[i]
            
            # Determine target (R or H)
            if maj == 0:  # R dimension
                target = "R"
                orig_value = r_orig_values[min_idx] if 0 <= min_idx < len(r_orig_values) else None
                numeric_value = r_values[min_idx] if 0 <= min_idx < len(r_values) else None
            else:  # H dimension
                target = f"H{maj-1}"
                if 0 <= maj-1 < len(h_orig_values) and 0 <= min_idx < len(h_orig_values[maj-1]):
                    orig_value = h_orig_values[maj-1][min_idx]
                    numeric_value = h_values[maj-1][min_idx]
                else:
                    orig_value = None
                    numeric_value = None
                
            mappings.append({
                "Target": target,
                "MajorIndex": maj,
                "MinorIndex": min_idx,
                "SymbolicValue": orig_value,  # Original symbolic value (variable name)
                "Value": numeric_value  # Numeric value if variable was resolved
            })
            
        result["IndexMapping"][y_key] = mappings
    
    return result

# Example usage
if __name__ == "__main__":
    # Example code to test the parser
    example_code = """
    tile_distribution_encoding<
        sequence<1>,                           // 0 R
        tuple<sequence<Nr_y, Nr_p, Nw>,        // H 
              sequence<Kr_y, Kr_p, Kw, Kv>>,
        tuple<sequence<1, 2>,                  // p major
              sequence<2, 1>>,
        tuple<sequence<1, 1>,                  // p minor
              sequence<2, 2>>,
        sequence<1, 2, 2>,                     // Y major
        sequence<0, 0, 3>>{}                   // y minor
    """
    
    # Debug: Print out the raw example string
    print("Raw example:")
    print(example_code)
    print()
    
    parser = TileDistributionParser()
    
    # Test each parsing function individually for debugging
    print("Testing sequence parsing:")
    seq_test = "sequence<Nr_y, Nr_p, Nw>"
    seq_result = parser.parse_sequence(seq_test)
    print(f"Input: {seq_test}")
    print(f"Output: {seq_result}")
    print()
    
    print("Testing tuple parsing:")
    tuple_test = "tuple<sequence<Nr_y, Nr_p, Nw>, sequence<Kr_y, Kr_p, Kw, Kv>>"
    tuple_result = parser.parse_tuple(tuple_test)
    print(f"Input: {tuple_test}")
    print(f"Output: {tuple_result}")
    print()
    
    # Parse the full example
    result = parser.parse_tile_distribution_encoding(example_code)
    print("Parsed Structure:")
    print(result)
    
    # Create test variables
    test_variables = {
        "Nr_y": 4,
        "Nr_p": 4,
        "Nw": 8,
        "Kr_y": 4,
        "Kr_p": 8,
        "Kw": 8,
        "Kv": 4
    }
    
    # Debug indexing relationships
    print("\nIndexing Relationships:")
    indexing_debug = debug_indexing_relationships(result, test_variables)
    print(json.dumps(indexing_debug, indent=2)) 