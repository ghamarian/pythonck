"""
Parser for tensor descriptor transformations.

This module provides functionality to parse C++ tensor descriptor transformation
expressions and convert them to pytensor objects.
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Union
import sympy as sp
from pytensor.tensor_descriptor import (
    Transform, PassThroughTransform, MergeTransform, UnmergeTransform,
    EmbedTransform, OffsetTransform, PadTransform, ReplicateTransform
)

def extract_descriptor_references(code: str) -> set:
    """Extract descriptor names that are used as first parameters in tensor descriptor functions or declared as descriptors."""
    import re
    
    descriptor_names = set()
    
    # Pattern 1: First parameter in transform_tensor_descriptor calls
    # This matches: transform_tensor_descriptor(first_param, ...)
    transform_pattern = r'transform_tensor_descriptor\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,'
    matches = re.findall(transform_pattern, code)
    descriptor_names.update(matches)
    
    # Pattern 2: Variable declarations that create descriptors
    # This matches: constexpr auto variable_name = make_naive_tensor_descriptor... or transform_tensor_descriptor...
    declaration_pattern = r'(?:constexpr\s+auto|auto)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:make_naive_tensor_descriptor|transform_tensor_descriptor)'
    matches = re.findall(declaration_pattern, code)
    descriptor_names.update(matches)
    
    return descriptor_names

def get_cpp_keywords():
    """Get the complete set of C++ keywords and function names that should not be treated as variables."""
    return {
        # C++ keywords
        'typename', 'type', 'constexpr', 'auto', 'const', 'static', 'namespace', 'class', 'struct',
        'template', 'using', 'true', 'false', 'nullptr', 'void', 'int', 'float', 'double', 'char',
        'bool', 'size_t', 'index_t', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
        'default', 'break', 'continue', 'public', 'private', 'protected', 'virtual', 'override',
        'final', 'inline', 'extern', 'friend', 'operator', 'new', 'delete', 'this', 'sizeof',
        
        # Tensor library specific keywords
        'arithmetic_sequence_gen', 'sequence', 'Sequence', 'number', 'make_tuple',
        'transform_tensor_descriptor', 'make_naive_tensor_descriptor', 'make_naive_tensor_descriptor_packed',
        'make_pass_through_transform', 'make_merge_transform', 'make_unmerge_transform',
        'make_xor_transform', 'make_xor_with_modulo_transform', 'make_merge_transform_v3_division_mod',
        'make_embed_transform', 'make_offset_transform', 'make_pad_transform', 'make_replicate_transform',
        
        # Common variable patterns that aren't actual variables
        'gen', 'transform', 'descriptor', 'tensor', 'make', 'get', 'set', 'create', 'build'
    }

class TensorTransformParser:
    """Parser for tensor descriptor transformations."""
    
    def __init__(self):
        """Initialize the parser."""
        self.descriptor_registry = {}  # Registry for resolving variable references
        self.default_warnings = []  # Track when default values are used
    
    def get_default_warnings(self):
        """Get list of warnings about default values used."""
        return self.default_warnings.copy()
    
    def clear_default_warnings(self):
        """Clear the list of default warnings."""
        self.default_warnings = []
    
    def extract_list_variables(self, descriptor_str: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract variables that should be lists based on their usage context.
        
        Returns:
            Dict mapping variable names to their list metadata:
            {
                'variable_name': {
                    'type': 'list',
                    'context': 'unmerge_lengths',  # or 'merge_lengths', etc.
                    'expected_type': 'int',
                    'default_length': 2,
                    'default_value': [2, 2]
                }
            }
        """
        list_variables = {}
        
        try:
            # Parse the descriptor to find list variable contexts
            parsed_dict = self.parse_tensor_descriptor(descriptor_str)
            
            if parsed_dict['type'] == 'transform':
                # Analyze transforms to find list variables
                for transform in parsed_dict['transforms']:
                    if transform['type'] == 'unmerge':
                        # Check if unmerge has a single variable reference (not make_tuple)
                        # This means: make_unmerge_transform(lengths) vs make_unmerge_transform(make_tuple(A, B, C))
                        if (len(transform['values']) == 1 and 
                            isinstance(transform['values'][0], dict) and 
                            transform['values'][0].get('type') == 'variable_reference'):
                            
                            var_name = transform['values'][0].get('name')
                            if var_name:
                                list_variables[var_name] = {
                                    'type': 'list',
                                    'context': 'unmerge_lengths',
                                    'expected_type': 'int',
                                    'default_length': 3,
                                    'default_value': [2, 4, 8],
                                    'description': f'List of output dimensions for unmerge transform'
                                }
                    
                    elif transform['type'] == 'merge':
                        # Similar logic for merge - only if it's a single variable reference
                        if (len(transform['values']) == 1 and 
                            isinstance(transform['values'][0], dict) and 
                            transform['values'][0].get('type') == 'variable_reference'):
                            
                            var_name = transform['values'][0].get('name')
                            if var_name:
                                list_variables[var_name] = {
                                    'type': 'list',
                                    'context': 'merge_lengths',
                                    'expected_type': 'int',
                                    'default_length': 3,
                                    'default_value': [2, 4, 8],
                                    'description': f'List of input dimensions for merge transform'
                                }
                
                # Also check upper dimensions for arithmetic sequence parameters
                for upper_dim in parsed_dict['upper_dimensions']:
                    if isinstance(upper_dim, dict) and upper_dim.get('type') == 'arithmetic_sequence_gen':
                        # Extract N parameter from arithmetic sequence
                        n_param = upper_dim.get('n')
                        if isinstance(n_param, sp.Symbol):
                            var_name = str(n_param)
                            # N should correspond to the length of any list variables
                            # Update default_length for list variables to match N
                            for list_var in list_variables.values():
                                list_var['linked_to_N'] = var_name
                                
        except Exception as e:
            # If parsing fails, we'll just return empty dict
            pass
        
        return list_variables

    def get_variable_info(self, descriptor_str: str) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive information about all variables in the descriptor.
        
        Returns:
            Dict mapping variable names to their metadata:
            {
                'variable_name': {
                    'type': 'scalar' | 'list',
                    'default_value': value,
                    'description': str,
                    ...
                }
            }
        """
        variable_info = {}
        
        # Get scalar variables (existing functionality)
        scalar_vars = self.extract_variables(descriptor_str)
        for var in scalar_vars:
            variable_info[var] = {
                'type': 'scalar',
                'default_value': 1,
                'description': f'Scalar parameter {var}'
            }
        
        # Get list variables (new functionality)
        list_vars = self.extract_list_variables(descriptor_str)
        for var_name, var_data in list_vars.items():
            variable_info[var_name] = var_data
        
        return variable_info

    def extract_variables(self, descriptor_str: str) -> set:
        """Extract scalar variables from the descriptor string."""
        # Get all identifiers
        identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', descriptor_str)
        
        # Filter out C++ keywords and descriptor references
        cpp_keywords = get_cpp_keywords()
        descriptor_references = extract_descriptor_references(descriptor_str)
        
        # Only keep actual variables
        variables = set()
        for identifier in identifiers:
            if identifier not in cpp_keywords and identifier not in descriptor_references:
                variables.add(identifier)
        
        return variables
    
    def _parse_value_expr(self, expr_str: str) -> sp.Expr:
        """Parse a string into a SymPy expression, keeping variables symbolic."""
        s = expr_str.strip()

        # Handle typename arithmetic_sequence_gen<start, N, step>::type{} template
        typename_match = re.match(r'typename\s+arithmetic_sequence_gen<([^>]+)>::type\{\}', s)
        if typename_match:
            # For arithmetic sequence generation, we need to return a special marker
            # that indicates this should generate a sequence from start to start+N*step-step
            # The content inside <> should be start, N, step
            content = typename_match.group(1).strip()
            parts = [p.strip() for p in content.split(',')]
            if len(parts) >= 3:
                start_param = parts[0]
                n_param = parts[1]  
                step_param = parts[2]
                # Return a special dict that marks this as an arithmetic sequence generator
                return {
                    'type': 'arithmetic_sequence_gen',
                    'start': sp.Symbol(start_param) if start_param.isalpha() else sp.Integer(int(start_param)),
                    'n': sp.Symbol(n_param) if n_param.isalpha() else sp.Integer(int(n_param)),
                    'step': sp.Symbol(step_param) if step_param.isalpha() else sp.Integer(int(step_param))
                }
            else:
                return sp.Symbol('seq_len')

        # Handle number<...>{} template
        match = re.match(r'number<([^>]+)>{}', s)
        if match:
            s = match.group(1).strip()

        # Handle expressions like "BK0 * number<NLdsLayer>{}"
        # Look for pattern: variable * number<expression>{}
        mult_number_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\*\s*number<([^>]+)>\{\}', s)
        if mult_number_match:
            var_name = mult_number_match.group(1)
            number_expr = mult_number_match.group(2).strip()
            
            # Parse both parts as symbols/expressions
            var_symbol = sp.Symbol(var_name)
            number_symbol = self._parse_value_expr(f'number<{number_expr}>{{}}')
            return var_symbol * number_symbol

        # Find all identifiers (potential variables) in the string
        identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', s)
        
        # Filter out C++ keywords and function names that shouldn't be treated as variables
        cpp_keywords = get_cpp_keywords()
        
        # Only create symbols for actual variables, not C++ keywords
        variable_identifiers = [i for i in identifiers if i not in cpp_keywords]
        
        # Create a dictionary of SymPy symbols for the actual variables
        local_dict = {i: sp.Symbol(i) for i in variable_identifiers}

        try:
            # Use sympify to parse the string into a SymPy expression.
            # Variables are kept as symbols.
            return sp.sympify(s, locals=local_dict)
        except (sp.SympifyError, TypeError, SyntaxError):
            raise ValueError(f"Failed to parse expression: {expr_str}")
    
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
            if any(t in item for t in ['make_pass_through_transform', 'make_merge_transform', 
                                     'make_xor_transform', 'make_xor_with_modulo_transform', 'make_unmerge_transform']):
                result.append(self.parse_transform(item))
            elif item.startswith(('sequence<', 'Sequence<')):
                result.append(self.parse_sequence(item))
            elif item.startswith('make_tuple('):
                # A nested tuple implies a merge
                nested_result = self.parse_make_tuple(item)
                result.append({
                    'type': 'merge',
                    'values': nested_result
                })
            else:
                # Parse the value expression and check if it's an arithmetic sequence generator
                parsed_value = self._parse_value_expr(item)
                if isinstance(parsed_value, dict) and parsed_value.get('type') == 'arithmetic_sequence_gen':
                    # This is an arithmetic sequence generator - add it directly
                    result.append(parsed_value)
                elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', item):
                    # This is a bare variable name - could be a list reference
                    result.append({
                        'type': 'variable_reference',
                        'name': item
                    })
                else:
                    # Otherwise, it's a value expression that we treat as a pass_through length
                    result.append({
                        'type': 'pass_through',
                        'value': parsed_value
                    })
        return result
    
    def parse_sequence(self, sequence_str: str) -> List[int]:
        """Parse a sequence expression."""
        content = sequence_str.strip()
        # Use regex to robustly find content within sequence<...> or Sequence<...> (case-insensitive)
        # This will capture the content between the angle brackets.
        match = re.match(r'[Ss]equence<(.*?)>(?:\{\})?$', content)
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
        
        # Handle xor transform
        elif transform_str.startswith('make_xor_transform'):
            match = re.match(r'make_xor_transform\((.*)\)', transform_str, re.DOTALL)
            if match:
                tuple_content = match.group(1).strip()
                values = self.parse_make_tuple(tuple_content)
                return {
                    'type': 'xor',
                    'values': values
                }
        
        # Handle xor_with_modulo transform
        elif transform_str.startswith('make_xor_with_modulo_transform'):
            match = re.match(r'make_xor_with_modulo_transform\((.*)\)', transform_str, re.DOTALL)
            if match:
                tuple_content = match.group(1).strip()
                values = self.parse_make_tuple(tuple_content)
                return {
                    'type': 'xor_with_modulo',
                    'values': values
                }
        
        # Handle unmerge transform
        elif transform_str.startswith('make_unmerge_transform'):
            match = re.match(r'make_unmerge_transform\((.*)\)', transform_str, re.DOTALL)
            if match:
                tuple_content = match.group(1).strip()
                # Check if the content is just a variable name (not a make_tuple)
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tuple_content):
                    # This is a variable reference, not a tuple
                    return {
                        'type': 'unmerge',
                        'values': [{
                            'type': 'variable_reference',
                            'name': tuple_content
                        }]
                    }
                else:
                    # This is a tuple or more complex expression
                    values = self.parse_make_tuple(tuple_content)
                    return {
                        'type': 'unmerge',
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
        descriptor_str = descriptor_str.strip()
        
        # Check if it's just a variable reference (e.g., "input_desc", "k_lds_block_desc_0")
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', descriptor_str):
            return {
                'type': 'variable',
                'name': descriptor_str,
                'transforms': [],
                'lower_dimensions': [],
                'upper_dimensions': []
            }
        
        # Check for transform_tensor_descriptor FIRST (before naive patterns)
        # This prevents naive patterns from matching inside transform_tensor_descriptor
        pattern = r'transform_tensor_descriptor\s*\(\s*(.*?)\s*,\s*make_tuple\s*\((.*?)\)\s*,\s*make_tuple\s*\((.*?)\)\s*,\s*make_tuple\s*\((.*?)\)\s*\)'
        match = re.search(pattern, descriptor_str, re.DOTALL)
        
        if match:
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
                'type': 'transform',
                'input_descriptor': input_desc,
                'transforms': transforms,
                'lower_dimensions': lower_dims,
                'upper_dimensions': upper_dims
            }
        
        # Check for make_naive_tensor_descriptor_packed
        naive_packed_match = re.search(r'make_naive_tensor_descriptor_packed\s*\(\s*make_tuple\s*\((.*?)\)\s*\)', descriptor_str, re.DOTALL)
        if naive_packed_match:
            # Parse the tuple of dimensions
            dimensions_str = naive_packed_match.group(1).strip()
            dimensions = self.parse_make_tuple(dimensions_str)
            return {
                'type': 'naive_packed',
                'dimensions': dimensions,
                'transforms': [],
                'lower_dimensions': [],
                'upper_dimensions': []
            }
        
        # Check for regular make_naive_tensor_descriptor
        # Support both formats:
        # 1) make_naive_tensor_descriptor(lengths_tuple, strides_tuple, vector_length, offset)
        # 2) make_naive_tensor_descriptor(lengths_tuple, strides_tuple) - with defaults
        
        # Try 4-parameter version first
        naive_match_4 = re.search(r'make_naive_tensor_descriptor\s*\(\s*make_tuple\s*\((.*?)\)\s*,\s*make_tuple\s*\((.*?)\)\s*,\s*(.*?)\s*,\s*(.*?)\s*\)', descriptor_str, re.DOTALL)
        if naive_match_4:
            # Parse all components (4-parameter version)
            lengths_str = naive_match_4.group(1).strip()
            strides_str = naive_match_4.group(2).strip()
            vector_length_str = naive_match_4.group(3).strip()
            offset_str = naive_match_4.group(4).strip()
            
            lengths = self.parse_make_tuple(lengths_str)
            strides = self.parse_make_tuple(strides_str)
            vector_length = self._parse_value_expr(vector_length_str)
            offset = self._parse_value_expr(offset_str)
            
            return {
                'type': 'naive',
                'lengths': lengths,
                'strides': strides,
                'vector_length': vector_length,
                'offset': offset,
                'transforms': [],
                'lower_dimensions': [],
                'upper_dimensions': []
            }
        
        # Try 2-parameter version (with defaults)
        naive_match_2 = re.search(r'make_naive_tensor_descriptor\s*\(\s*make_tuple\s*\((.*?)\)\s*,\s*make_tuple\s*\((.*?)\)\s*\)', descriptor_str, re.DOTALL)
        if naive_match_2:
            # Parse components (2-parameter version with defaults)
            lengths_str = naive_match_2.group(1).strip()
            strides_str = naive_match_2.group(2).strip()
            
            lengths = self.parse_make_tuple(lengths_str)
            strides = self.parse_make_tuple(strides_str)
            # Use defaults for optional parameters
            vector_length = sp.sympify(-1)  # number<-1>{}
            offset = sp.sympify(-1)         # number<-1>{}
            
            return {
                'type': 'naive',
                'lengths': lengths,
                'strides': strides,
                'vector_length': vector_length,
                'offset': offset,
                'transforms': [],
                'lower_dimensions': [],
                'upper_dimensions': []
            }
        
        # If no patterns matched, this is an error
        raise ValueError(f"Invalid tensor descriptor format. Unable to parse: {descriptor_str[:200]}...")
    
    def _flatten_merge_lengths(self, val, variables):
        """Recursively flatten merge structures to get individual input lengths."""
        if isinstance(val, dict):
            if val.get('type') == 'pass_through':
                # PassThrough: contributes its length as one input
                value = val.get('value', 1)
                if isinstance(value, sp.Basic):
                    # Handle case where variables might contain lists
                    safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                    return [int(value.subs(safe_variables))]
                else:
                    return [int(value)]
            elif val.get('type') == 'variable_reference':
                # Handle variable references that might be lists
                var_name = val.get('name')
                if var_name in variables:
                    var_value = variables[var_name]
                    if isinstance(var_value, list):
                        # If the variable is a list, use it as the lengths
                        return [int(x) for x in var_value]
                    else:
                        # If it's a scalar, treat it as a single length
                        return [int(var_value)]
                else:
                    # Variable not found, default to 1
                    warning = f"Variable '{var_name}' not defined, using default value [1]"
                    self.default_warnings.append(warning)
                    return [1]
            elif val.get('type') == 'merge':
                # Nested merge: flatten its components
                nested_lengths = []
                for nested_val in val.get('values', []):
                    nested_lengths.extend(self._flatten_merge_lengths(nested_val, variables))
                return nested_lengths
            else:
                # Other transform types
                nested_transform = self.create_pytensor_transform(val, variables)
                if hasattr(nested_transform, 'length'):
                    return [nested_transform.length]
                else:
                    return [1]
        elif isinstance(val, sp.Basic):
            # Handle case where variables might contain lists
            safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
            return [int(val.subs(safe_variables))]
        else:
            return [int(val)]
    
    def create_pytensor_transform(self, transform_dict: Dict[str, Any], variables: Dict[str, int] = None) -> 'Transform':
        """Convert a parsed transform dictionary to a pytensor Transform object."""
        from pytensor.tensor_descriptor import (
            PassThroughTransform, MergeTransform, UnmergeTransform,
            PadTransform, OffsetTransform, ReplicateTransform, XorTransform
        )
        
        if variables is None:
            variables = {}
        
        transform_type = transform_dict.get('type')
        
        if transform_type == 'pass_through':
            value = transform_dict.get('value', 1)
            # Substitute variables if it's a symbolic expression
            if isinstance(value, sp.Basic):
                # Handle case where variables might contain lists
                safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                value = int(value.subs(safe_variables))
            return PassThroughTransform(length=value)
        
        elif transform_type == 'variable_reference':
            # Handle standalone variable references as pass-through transforms
            var_name = transform_dict.get('name')
            if var_name in variables:
                var_value = variables[var_name]
                if isinstance(var_value, list):
                    # If the variable is a list, we can't use it directly as a pass-through
                    # This might need special handling depending on context
                    raise ValueError(f"Variable '{var_name}' is a list and cannot be used as a standalone transform")
                else:
                    return PassThroughTransform(length=int(var_value))
            else:
                # Variable not found, default to 1
                warning = f"Variable '{var_name}' not defined, using default value 1"
                self.default_warnings.append(warning)
                return PassThroughTransform(length=1)
        
        elif transform_type == 'merge':
            values = transform_dict.get('values', [])
            
            # Check if any values contain nested merges
            has_nested_merge = any(isinstance(val, dict) and val.get('type') == 'merge' for val in values)
            
            if has_nested_merge:
                # Create a hierarchical transform representation
                # Store the hierarchy information for the graph builder
                hierarchical_info = {
                    'type': 'hierarchical_merge',
                    'structure': values,
                    'is_hierarchical': True
                }
                
                # For now, still create a single MergeTransform but with hierarchy metadata
                # The graph builder will use this information to create proper nodes
                lengths = []
                for val in values:
                    lengths.extend(self._flatten_merge_lengths(val, variables))
                
                merge_transform = MergeTransform(lengths=lengths)
                # Add hierarchy information as an attribute
                merge_transform._hierarchy_info = hierarchical_info
                return merge_transform
            else:
                # Regular flat merge - use existing logic
                lengths = []
                for val in values:
                    lengths.extend(self._flatten_merge_lengths(val, variables))
                return MergeTransform(lengths=lengths)
        
        elif transform_type == 'unmerge':
            values = transform_dict.get('values', [])
            lengths = []
            for val in values:
                if isinstance(val, dict):
                    if val.get('type') == 'variable_reference':
                        # Handle variable references that might be lists
                        var_name = val.get('name')
                        if var_name in variables:
                            var_value = variables[var_name]
                            if isinstance(var_value, list):
                                # If the variable is a list, use it as the lengths
                                lengths.extend([int(x) for x in var_value])
                            else:
                                # If it's a scalar, treat it as a single length
                                lengths.append(int(var_value))
                        else:
                            # Variable not found, default to 1
                            warning = f"Variable '{var_name}' not defined, using default value 1"
                            self.default_warnings.append(warning)
                            lengths.append(1)
                    else:
                        nested_transform = self.create_pytensor_transform(val, variables)
                        if hasattr(nested_transform, 'length'):
                            lengths.append(nested_transform.length)
                        elif hasattr(nested_transform, 'get_upper_lengths'):
                            lengths.extend(nested_transform.get_upper_lengths())
                        else:
                            lengths.append(1)
                elif isinstance(val, sp.Basic):
                    # Handle case where variables might contain lists
                    safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                    lengths.append(int(val.subs(safe_variables)))
                else:
                    lengths.append(int(val))
            from pytensor.tensor_descriptor import UnmergeTransform
            return UnmergeTransform(lengths=lengths)
        
        elif transform_type == 'pad':
            lower_length = transform_dict.get('lower_length', 1)
            left_pad = transform_dict.get('left_pad', 0)
            right_pad = transform_dict.get('right_pad', 0)
            
            # Substitute variables if they're symbolic
            safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
            if isinstance(lower_length, sp.Basic):
                lower_length = int(lower_length.subs(safe_variables))
            if isinstance(left_pad, sp.Basic):
                left_pad = int(left_pad.subs(safe_variables))
            if isinstance(right_pad, sp.Basic):
                right_pad = int(right_pad.subs(safe_variables))
                
            return PadTransform(lower_length=lower_length, left_pad=left_pad, right_pad=right_pad)
        
        elif transform_type == 'offset':
            element_space_size = transform_dict.get('element_space_size', 1)
            offset = transform_dict.get('offset', 0)
            
            safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
            if isinstance(element_space_size, sp.Basic):
                element_space_size = int(element_space_size.subs(safe_variables))
            if isinstance(offset, sp.Basic):
                offset = int(offset.subs(safe_variables))
                
            return OffsetTransform(element_space_size=element_space_size, offset=offset)
        
        elif transform_type == 'replicate':
            upper_lengths = transform_dict.get('upper_lengths', [1])
            
            # Substitute variables in lengths
            safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
            resolved_lengths = []
            for length in upper_lengths:
                if isinstance(length, sp.Basic):
                    resolved_lengths.append(int(length.subs(safe_variables)))
                else:
                    resolved_lengths.append(int(length))
                    
            return ReplicateTransform(upper_lengths=resolved_lengths)
        
        elif transform_type == 'xor':
            # XOR transform - proper implementation
            values = transform_dict.get('values', [])
            lengths = []
            for val in values:
                if isinstance(val, dict):
                    # For XOR, we expect the values to be numbers or expressions, not transforms
                    # So we extract the values directly
                    if val.get('type') == 'pass_through':
                        value = val.get('value', 1)
                        safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                        if isinstance(value, sp.Basic):
                            lengths.append(int(value.subs(safe_variables)))
                        else:
                            lengths.append(int(value))
                    else:
                        nested_transform = self.create_pytensor_transform(val, variables)
                        if hasattr(nested_transform, 'length'):
                            lengths.append(nested_transform.length)
                        elif hasattr(nested_transform, 'get_upper_lengths'):
                            lengths.extend(nested_transform.get_upper_lengths())
                        else:
                            lengths.append(1)
                elif isinstance(val, sp.Basic):
                    safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                    lengths.append(int(val.subs(safe_variables)))
                else:
                    lengths.append(int(val))
            
            # XOR requires exactly 2 dimensions
            if len(lengths) != 2:
                raise ValueError(f"XOR transform requires exactly 2 dimensions, got {len(lengths)}")
            
            return XorTransform(lengths=lengths)
        
        elif transform_type == 'arithmetic_sequence_gen':
            # Handle arithmetic sequence generation
            start = transform_dict.get('start', 0)
            n = transform_dict.get('n', 1)
            step = transform_dict.get('step', 1)
            
            # Substitute variables
            safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
            if isinstance(start, sp.Basic):
                start = int(start.subs(safe_variables))
            if isinstance(n, sp.Basic):
                n = int(n.subs(safe_variables))
            if isinstance(step, sp.Basic):
                step = int(step.subs(safe_variables))
            
            # Generate the sequence: [start, start+step, start+2*step, ..., start+(n-1)*step]
            sequence = [start + i * step for i in range(n)]
            
            # Return a special marker that indicates this is a sequence
            # This will be handled by the descriptor builder
            return {
                'type': 'sequence_literal',
                'sequence': sequence
            }
        
        elif transform_type == 'xor_with_modulo':
            # XOR with modulo transform - same as regular XOR for our purposes
            values = transform_dict.get('values', [])
            lengths = []
            for val in values:
                if isinstance(val, dict):
                    # For XOR, we expect the values to be numbers or expressions, not transforms
                    # So we extract the values directly
                    if val.get('type') == 'pass_through':
                        value = val.get('value', 1)
                        safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                        if isinstance(value, sp.Basic):
                            lengths.append(int(value.subs(safe_variables)))
                        else:
                            lengths.append(int(value))
                    else:
                        nested_transform = self.create_pytensor_transform(val, variables)
                        if hasattr(nested_transform, 'length'):
                            lengths.append(nested_transform.length)
                        elif hasattr(nested_transform, 'get_upper_lengths'):
                            lengths.extend(nested_transform.get_upper_lengths())
                        else:
                            lengths.append(1)
                elif isinstance(val, sp.Basic):
                    safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                    lengths.append(int(val.subs(safe_variables)))
                else:
                    lengths.append(int(val))
            
            # XOR requires exactly 2 dimensions
            if len(lengths) != 2:
                raise ValueError(f"XOR with modulo transform requires exactly 2 dimensions, got {len(lengths)}")
            
            return XorTransform(lengths=lengths)
        
        elif transform_type == 'merge_v3_division_mod':
            # Merge transform v3 with division/mod - use same flattening logic as regular merge
            values = transform_dict.get('values', [])
            lengths = []
            
            # Flatten all values to get individual input lengths
            for val in values:
                lengths.extend(self._flatten_merge_lengths(val, variables))
                
            return MergeTransform(lengths=lengths)
        
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
    
    def create_pytensor_descriptor(self, descriptor_str: str, variables: Dict[str, int] = None) -> 'TensorDescriptor':
        """Parse a tensor descriptor string and create a pytensor TensorDescriptor object."""
        from pytensor.tensor_descriptor import (
            make_naive_tensor_descriptor_packed, make_naive_tensor_descriptor, transform_tensor_descriptor
        )
        
        if variables is None:
            variables = {}
        
        # Clear warnings at the start of each parsing operation
        self.clear_default_warnings()
        
        # Parse the descriptor string
        parsed_dict = self.parse_tensor_descriptor(descriptor_str)
        
        if parsed_dict['type'] == 'naive_packed':
            # Create a naive tensor descriptor (packed version)
            dimensions = parsed_dict['dimensions']
            lengths = []
            for dim in dimensions:
                if isinstance(dim, dict):
                    # Handle nested transform structures in dimensions
                    nested_transform = self.create_pytensor_transform(dim, variables)
                    if hasattr(nested_transform, 'length'):
                        lengths.append(nested_transform.length)
                    else:
                        lengths.append(1)
                elif isinstance(dim, sp.Basic):
                    # Handle case where variables might contain lists
                    safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                    lengths.append(int(dim.subs(safe_variables)))
                else:
                    lengths.append(int(dim))
            
            return make_naive_tensor_descriptor_packed(lengths)
        
        elif parsed_dict['type'] == 'naive':
            # Create a naive tensor descriptor (regular version with lengths, strides, vector_length, offset)
            lengths_data = parsed_dict['lengths']
            strides_data = parsed_dict['strides']
            vector_length = parsed_dict['vector_length']
            offset = parsed_dict['offset']
            
            # Process lengths
            lengths = []
            for length_item in lengths_data:
                if isinstance(length_item, dict):
                    nested_transform = self.create_pytensor_transform(length_item, variables)
                    if hasattr(nested_transform, 'length'):
                        lengths.append(nested_transform.length)
                    else:
                        lengths.append(1)
                elif isinstance(length_item, sp.Basic):
                    safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                    lengths.append(int(length_item.subs(safe_variables)))
                else:
                    lengths.append(int(length_item))
            
            # Process strides
            strides = []
            for stride_item in strides_data:
                if isinstance(stride_item, dict):
                    nested_transform = self.create_pytensor_transform(stride_item, variables)
                    if hasattr(nested_transform, 'length'):
                        strides.append(nested_transform.length)
                    else:
                        strides.append(1)
                elif isinstance(stride_item, sp.Basic):
                    safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                    strides.append(int(stride_item.subs(safe_variables)))
                else:
                    strides.append(int(stride_item))
            
            # Process vector_length and offset
            safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
            if isinstance(vector_length, sp.Basic):
                vector_length_val = int(vector_length.subs(safe_variables))
            else:
                vector_length_val = int(vector_length)
            
            if isinstance(offset, sp.Basic):
                offset_val = int(offset.subs(safe_variables))
            else:
                offset_val = int(offset)
            
            return make_naive_tensor_descriptor(lengths, strides, vector_length_val, offset_val)
        
        elif parsed_dict['type'] == 'variable':
            # Handle variable references by looking them up in the registry
            var_name = parsed_dict['name']
            
            # Check if we have this variable in our registry
            if hasattr(self, 'descriptor_registry') and var_name in self.descriptor_registry:
                return self.descriptor_registry[var_name]
            
            # If not found, we need to create a descriptor with the right number of dimensions
            # To do this properly, we need context about how this variable is used
            # For now, create a reasonable default that can be overridden
            from pytensor.tensor_descriptor import TensorDescriptor
            
            # Try to infer dimensions from usage context if available
            if hasattr(self, '_current_context'):
                context = self._current_context
                required_dims = context.get('required_input_dims', 0)
                if required_dims > 0:
                    return TensorDescriptor(
                        transforms=[],
                        lower_dimension_hidden_idss=[],
                        upper_dimension_hidden_idss=[],
                        top_dimension_hidden_ids=list(range(required_dims)),
                        element_space_size=128,
                    )
            
            # Default fallback
            return TensorDescriptor(
                transforms=[],
                lower_dimension_hidden_idss=[],
                upper_dimension_hidden_idss=[],
                top_dimension_hidden_ids=[0, 1],  # Default 2D
                element_space_size=128,
            )
        
        elif parsed_dict['type'] == 'transform':
            # Analyze dimension requirements first
            all_lower_dims = parsed_dict['lower_dimensions'] 
            all_upper_dims = parsed_dict['upper_dimensions']
            
            # Find the maximum dimension index referenced
            max_input_dim = 0
            for dims in all_lower_dims:
                if dims:
                    max_input_dim = max(max_input_dim, max(dims) + 1)
            
            # Set context for variable resolution
            self._current_context = {
                'lower_dimensions': all_lower_dims,
                'upper_dimensions': all_upper_dims,
                'required_input_dims': max_input_dim
            }
            
            # Then, recursively parse the input descriptor
            input_desc_str = parsed_dict['input_descriptor']
            input_descriptor = self.create_pytensor_descriptor(input_desc_str, variables)
            
            # Clear context
            if hasattr(self, '_current_context'):
                delattr(self, '_current_context')
            
            # Create transform objects
            transforms = []
            for transform_dict in parsed_dict['transforms']:
                transform_obj = self.create_pytensor_transform(transform_dict, variables)
                transforms.append(transform_obj)
            
            # Convert dimension indices, handling sequence literals
            lower_dims = []
            for dim_item in parsed_dict['lower_dimensions']:
                if isinstance(dim_item, dict) and dim_item.get('type') == 'sequence_literal':
                    # Convert sequence literal to actual sequence
                    lower_dims.append(dim_item['sequence'])
                elif isinstance(dim_item, dict) and dim_item.get('type') == 'arithmetic_sequence_gen':
                    # Handle arithmetic sequence generator directly
                    seq_transform = self.create_pytensor_transform(dim_item, variables)
                    if isinstance(seq_transform, dict) and seq_transform.get('type') == 'sequence_literal':
                        lower_dims.append(seq_transform['sequence'])
                    else:
                        lower_dims.append([0])  # fallback
                elif isinstance(dim_item, list):
                    lower_dims.append(dim_item)
                else:
                    lower_dims.append(dim_item)
            
            upper_dims = []
            for dim_item in parsed_dict['upper_dimensions']:
                if isinstance(dim_item, dict) and dim_item.get('type') == 'sequence_literal':
                    # Convert sequence literal to actual sequence
                    upper_dims.append(dim_item['sequence'])
                elif isinstance(dim_item, dict) and dim_item.get('type') == 'arithmetic_sequence_gen':
                    # Handle arithmetic sequence generator directly
                    seq_transform = self.create_pytensor_transform(dim_item, variables)
                    if isinstance(seq_transform, dict) and seq_transform.get('type') == 'sequence_literal':
                        upper_dims.append(seq_transform['sequence'])
                    else:
                        upper_dims.append([0])  # fallback
                elif isinstance(dim_item, list):
                    upper_dims.append(dim_item)
                else:
                    upper_dims.append(dim_item)
            
            # Validate dimension compatibility before creating descriptor
            for i, (transform, lower_dim_list, upper_dim_list) in enumerate(zip(transforms, lower_dims, upper_dims)):
                if hasattr(transform, 'ndim'):
                    expected_inputs = transform.ndim
                    if isinstance(upper_dim_list, list) and len(upper_dim_list) != expected_inputs:
                        # Provide helpful suggestion for common arithmetic sequence issues
                        if len(upper_dim_list) > expected_inputs:
                            suggestion = f"Consider reducing N from {len(upper_dim_list)} to {expected_inputs} in your arithmetic_sequence_gen"
                        else:
                            suggestion = f"Consider increasing N from {len(upper_dim_list)} to {expected_inputs} in your arithmetic_sequence_gen"
                        
                        raise ValueError(
                            f"Transform {i} ({transform.__class__.__name__}) with lengths {transform.lengths} "
                            f"expects {expected_inputs} upper dimensions but got {len(upper_dim_list)}. "
                            f"{suggestion}."
                        )
            
            return transform_tensor_descriptor(
                input_descriptor=input_descriptor,
                transforms=transforms,
                lower_dimension_hidden_idss=lower_dims,
                upper_dimension_hidden_idss=upper_dims
            )
        
        else:
            raise ValueError(f"Unknown descriptor type: {parsed_dict['type']}")
    
    def to_sympy(self, descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a descriptor dictionary to SymPy expressions.
        
        Args:
            descriptor: Dictionary containing transform information
            
        Returns:
            Dictionary with SymPy expressions for transforms and symbols
        """
        result = {
            'transforms': [],
            'symbols': set()
        }
        
        # Create symbols for each dimension
        symbols = {}  # Use a dictionary to store symbols by dimension number
        
        # Process each transform
        for transform, lower_dims in zip(descriptor.get('transforms', []), descriptor.get('lower_dimensions', [])):
            # Create symbols for any new dimensions we encounter
            for dim in lower_dims:
                if dim not in symbols:
                    symbols[dim] = sp.Symbol(f'dim_{dim}')
            
            if transform['type'] == 'pass_through':
                # Create PassThroughTransform
                transform_obj = PassThroughTransform(length=1)  # Length will be updated later
                # Get input symbol
                input_symbol = symbols[lower_dims[0]]
                # Use sympy_calculate_upper to get output
                output = transform_obj.sympy_calculate_upper([input_symbol])
                result['transforms'].append({
                    'type': 'pass_through',
                    'expr': output[0]
                })
            
            elif transform['type'] == 'merge':
                # Get input symbols for this transform
                input_symbols = [symbols[i] for i in lower_dims]
                
                # Get lengths for this transform
                lengths = []
                for val in transform['values']:
                    if isinstance(val['value'], sp.Basic):
                        length = val['value']
                    else:
                        length = sp.Integer(val['value'])
                    lengths.append(length)
                
                # Create MergeTransform
                transform_obj = MergeTransform(lengths=lengths)
                # Use sympy_calculate_upper to get output
                output = transform_obj.sympy_calculate_upper(input_symbols)
                result['transforms'].append({
                    'type': 'merge',
                    'expr': output[0]
                })
        
        # Add all symbols to the result
        result['symbols'] = set(symbols.values())
        
        # Add symbol names for testing
        result['symbol_names'] = {str(symbol) for symbol in symbols.values()}
        
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
    
    # Parse and convert to SymPy
    descriptor = parser.parse_tensor_descriptor(example)
    sympy_desc = parser.to_sympy(descriptor)
    
    print("Parsed descriptor:", descriptor)
    print("\nSymPy expressions:", sympy_desc)

if __name__ == "__main__":
    main() 