"""
Parser for tensor descriptor transformations.

This module provides functionality to parse C++ tensor descriptor transformation
expressions and convert them to SymPy expressions for analysis.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import sympy as sp

class TensorTransformParser:
    """Parser for tensor descriptor transformations."""
    
    def __init__(self):
        """Initialize the parser."""
        self.variables: Dict[str, int] = {}
    
    def _parse_value_expr(self, expr_str: str) -> sp.Expr:
        """Parse a string into a SymPy expression, handling variables."""
        s = expr_str.strip()

        # Handle number<...>{} template
        match = re.match(r'number<([^>]+)>{}', s)
        if match:
            s = match.group(1).strip()

        # Find all identifiers (potential variables) in the string
        identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', s)
        
        # Create a dictionary of SymPy symbols for the identifiers
        local_dict = {i: sp.Symbol(i) for i in identifiers}

        try:
            # Use sympify to parse the string into a SymPy expression
            # and substitute known numeric variables
            expr = sp.sympify(s, locals=local_dict)
            return expr.subs(self.variables)
        except (sp.SympifyError, TypeError, SyntaxError):
            raise ValueError(f"Could not parse '{expr_str}' as a symbolic expression.")
    
    def parse_make_tuple(self, tuple_str: str) -> List[Any]:
        """Parse a make_tuple expression."""
        # Remove outer make_tuple and braces
        content = tuple_str.strip()
        if content.startswith('make_tuple('):
            content = content[len('make_tuple('):-1].strip()
        
        # Split by commas, but not within nested parentheses or angle brackets
        items = []
        current = ""
        paren_depth = 0
        angle_depth = 0
        for char in content:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == '<':
                angle_depth += 1
            elif char == '>':
                angle_depth -= 1
            elif char == ',' and paren_depth == 0 and angle_depth == 0:
                items.append(current.strip())
                current = ""
                continue
            current += char
        if current:
            items.append(current.strip())
        
        # Parse each item
        result = []
        for item in items:
            if not item: continue
            # Try parsing as a transform first
            if 'make_pass_through_transform' in item or 'make_merge_transform' in item:
                result.append(self.parse_transform(item))
            elif item.startswith('sequence<'):
                result.append(self.parse_sequence(item))
            elif item.startswith('make_tuple('):
                # A nested tuple implies a merge
                nested_result = self.parse_make_tuple(item)
                result.append({
                    'type': 'merge',
                    'values': nested_result
                })
            else:
                # Otherwise, it's a value expression that we treat as a pass_through length
                result.append({
                    'type': 'pass_through',
                    'value': self._parse_value_expr(item)
                })
        return result
    
    def parse_sequence(self, sequence_str: str) -> List[int]:
        """Parse a sequence expression."""
        content = sequence_str.strip()
        # Use regex to robustly find content within sequence<...> or sequence<...>{}
        # This will capture the content between the angle brackets.
        match = re.match(r'sequence<(.*?)>(?:\{\})?$', content)
        if not match:
            raise ValueError(f"Invalid sequence format: {sequence_str}")
        
        content = match.group(1)

        if not content.strip():
            return []
        
        items = content.split(',')
        result = []
        for item in items:
            item_str = item.strip()
            if item_str:
                # Sequences in descriptors are indices, so they should be integers.
                # We use the parser to evaluate expressions like 'kKPerBlock / 8'
                # and then cast to int.
                expr = self._parse_value_expr(item_str)
                if not expr.is_number:
                    raise ValueError(f"Sequence item '{item_str}' must be a numeric value.")
                result.append(int(expr))
        return result
    
    def parse_transform(self, transform_str: str) -> Dict[str, Any]:
        """Parse a transform expression."""
        transform_str = transform_str.strip()
        
        if transform_str.startswith('make_pass_through_transform'):
            match = re.match(r'make_pass_through_transform\((.*?)\)', transform_str)
            if match:
                return {
                    'type': 'pass_through',
                    'value': self._parse_value_expr(match.group(1))
                }
        
        # Handle both merge transform variants
        elif transform_str.startswith(('make_merge_transform', 'make_merge_transform_v3_division_mod')):
            match = re.match(r'make_merge_transform(?:_v3_division_mod)?\((.*)\)', transform_str, re.DOTALL)
            if match:
                tuple_content = match.group(1).strip()
                values = self.parse_make_tuple(tuple_content)
                return {
                    'type': 'merge',
                    'values': values
                }
        
        try:
            # Fallback for raw values/expressions, treat as pass_through
            value = self._parse_value_expr(transform_str)
            return {
                'type': 'pass_through',
                'value': value
            }
        except ValueError:
            raise ValueError(f"Unknown transform type: {transform_str}")
    
    def parse_tensor_descriptor(self, descriptor_str: str) -> Dict[str, Any]:
        """Parse a tensor descriptor transformation."""
        # Extract the main components with a more precise pattern
        pattern = r'transform_tensor_descriptor\s*\(\s*(.*?)\s*,\s*make_tuple\s*\((.*?)\)\s*,\s*make_tuple\s*\((.*?)\)\s*,\s*make_tuple\s*\((.*?)\)\s*\)'
        match = re.search(pattern, descriptor_str, re.DOTALL)
        
        if not match:
            raise ValueError("Invalid tensor descriptor format")
        
        input_desc = match.group(1).strip()
        transforms_str = match.group(2).strip()
        lower_dims_str = match.group(3).strip()
        upper_dims_str = match.group(4).strip()
        
        # Parse each component
        transforms = self.parse_make_tuple(transforms_str)
        lower_dims = self.parse_make_tuple(lower_dims_str)
        upper_dims = self.parse_make_tuple(upper_dims_str)
        
        # Ensure transforms are properly typed
        for i, transform in enumerate(transforms):
            if isinstance(transform, dict):
                # Keep existing transforms as is
                continue
            elif isinstance(transform, (int, float)):
                # Convert numbers to pass-through transforms
                transforms[i] = {
                    'type': 'pass_through',
                    'value': transform
                }
            elif isinstance(transform, list):
                # Convert lists to merge transforms
                transforms[i] = {
                    'type': 'merge',
                    'values': transform
                }
        
        return {
            'input_descriptor': input_desc,
            'transforms': transforms,
            'lower_dimensions': lower_dims,
            'upper_dimensions': upper_dims
        }
    
    def set_variables(self, variables: Dict[str, int]):
        """Set variable values for parsing."""
        self.variables = variables
    
    def to_sympy(self, descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed descriptor to SymPy expressions."""
        # Create symbols for each dimension
        dim_symbols = {}
        max_dim = -1
        
        # Find max dimension from lower and upper dimensions
        all_dims = descriptor.get('lower_dimensions', []) + descriptor.get('upper_dimensions', [])
        for dims in all_dims:
            if isinstance(dims, list):
                max_dim = max(max_dim, max(dims, default=-1))
            else:
                max_dim = max(max_dim, dims)

        for i in range(max_dim + 1):
            dim_symbols[f'dim_{i}'] = sp.Symbol(f'dim_{i}')
        
        # Convert transforms to SymPy expressions
        sympy_transforms = []
        for transform in descriptor['transforms']:
            if transform['type'] == 'pass_through':
                dim_idx = transform['value']
                if f'dim_{dim_idx}' not in dim_symbols:
                     # This can happen if a pass_through refers to a dim not in lower/upper
                     dim_symbols[f'dim_{dim_idx}'] = sp.Symbol(f'dim_{dim_idx}')
                sympy_transforms.append({
                    'type': 'pass_through',
                    'expr': dim_symbols[f'dim_{dim_idx}']
                })
            elif transform['type'] == 'merge':
                # Create merge expression
                # This part needs to be implemented to handle expressions
                sympy_transforms.append(transform)  # Placeholder

        return {
            'input_descriptor': descriptor.get('input_descriptor'),
            'transforms': sympy_transforms,
            'symbols': dim_symbols
        }

def merge_transform_to_sympy(input_exprs: List[sp.Expr], lengths: List[int]) -> sp.Expr:
    """Convert merge transform to SymPy expression."""
    if len(input_exprs) != len(lengths):
        raise ValueError("Number of input expressions must match number of lengths")
    
    result = 0
    stride = 1
    for i in range(len(input_exprs) - 1, -1, -1):
        result += input_exprs[i] * stride
        stride *= lengths[i]
    return result

def unmerge_transform_to_sympy(input_expr: sp.Expr, lengths: List[int]) -> List[sp.Expr]:
    """Convert unmerge transform to SymPy expression."""
    result = []
    remaining = input_expr
    stride = 1
    
    # Calculate strides
    strides = [1]
    for i in range(len(lengths) - 1, 0, -1):
        strides.insert(0, strides[0] * lengths[i])
    
    # Extract each dimension
    for i in range(len(lengths)):
        result.append((remaining // strides[i]) % lengths[i])
    
    return result

def main():
    """Test the parser with example input."""
    parser = TensorTransformParser()
    
    # Example tensor descriptor
    example = """
    transform_tensor_descriptor(
        k_lds_block_desc_0,
        make_tuple(
            make_pass_through_transform(number<kNPerBlock>{}),
            make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}))
    """
    
    # Set some example variables
    parser.set_variables({
        'kNPerBlock': 32,
        'kKPerBlock': 64,
        'kKPack': 8
    })
    
    # Parse and convert to SymPy
    descriptor = parser.parse_tensor_descriptor(example)
    sympy_desc = parser.to_sympy(descriptor)
    
    print("Parsed descriptor:", descriptor)
    print("\nSymPy expressions:", sympy_desc)

if __name__ == "__main__":
    main() 