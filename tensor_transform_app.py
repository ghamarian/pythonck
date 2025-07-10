"""
Streamlit application for visualizing tensor descriptor transformations.
"""

import streamlit as st
import sympy as sp
import re
import time
import math
from tensor_transforms import (
    TensorTransformParser, get_cpp_keywords, extract_descriptor_references,
    get_transform_examples, get_default_variables, extract_descriptors_from_text
)
from pytensor.tensor_descriptor import (
    Transform, PassThroughTransform, MergeTransform, UnmergeTransform,
    EmbedTransform, OffsetTransform, PadTransform, ReplicateTransform
)
from typing import Dict, Any, List, Tuple
import graphviz
import pytensor.tensor_descriptor

def format_cpp_code(code: str) -> str:
    """
    Format C++ tensor descriptor code with proper indentation and spacing.
    """
    if not code.strip():
        return code
    
    lines = code.split('\n')
    formatted_lines = []
    indent_level = 0
    
    # Keywords that should be followed by proper spacing
    cpp_keywords = [
        'constexpr', 'auto', 'const', 'make_tuple', 'make_naive_tensor_descriptor',
        'transform_tensor_descriptor', 'make_pass_through_transform', 'make_merge_transform',
        'make_unmerge_transform', 'make_xor_transform', 'sequence', 'number', 'typename'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append("")
            continue
        
        # Adjust indent level based on braces and parentheses
        open_count = line.count('(') + line.count('{')
        close_count = line.count(')') + line.count('}')
        
        # Decrease indent before adding line if it closes more than it opens
        if close_count > open_count:
            indent_level = max(0, indent_level - (close_count - open_count))
        
        # Add proper indentation
        indent = "    " * indent_level  # 4 spaces per level
        
        # Clean up spacing around operators and keywords
        formatted_line = line
        
        # Add spaces around operators, but avoid template brackets
        # First handle = operators
        formatted_line = re.sub(r'([^<>=!])=([^=])', r'\1 = \2', formatted_line)
        # Handle comparison operators (==, !=, <=, >=) but not template brackets
        formatted_line = re.sub(r'([^<>])([!<>]=?)([^<>=])', r'\1 \2 \3', formatted_line)
        formatted_line = re.sub(r'\s+', ' ', formatted_line)  # Remove extra spaces
        
        # Fix template brackets - ensure no spaces inside
        formatted_line = re.sub(r'<\s+', '<', formatted_line)
        formatted_line = re.sub(r'\s+>', '>', formatted_line)
        # Fix spaces before template brackets (e.g., "sequence <0>" -> "sequence<0>")
        formatted_line = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+<', r'\1<', formatted_line)
        # Fix spaces before :: (scope resolution operator)
        formatted_line = re.sub(r'\s+::', r'::', formatted_line)
        
        # Fix spacing around parentheses
        formatted_line = re.sub(r'\(\s+', '(', formatted_line)
        formatted_line = re.sub(r'\s+\)', ')', formatted_line)
        
        # Add proper spacing after commas, but not before {} 
        formatted_line = re.sub(r',(?!\s)', ', ', formatted_line)
        # Fix unwanted space before {} that might have been introduced
        formatted_line = re.sub(r'>\s+{', r'>{', formatted_line)
        
        # Add the formatted line with indentation
        formatted_lines.append(indent + formatted_line)
        
        # Increase indent after adding line if it opens more than it closes
        if open_count > close_count:
            indent_level += (open_count - close_count)
    
    # Join lines and clean up
    formatted_code = '\n'.join(formatted_lines)
    
    # Remove trailing whitespace from each line
    formatted_code = '\n'.join(line.rstrip() for line in formatted_code.split('\n'))
    
    # Ensure proper line ending
    return formatted_code.strip() + '\n' if formatted_code.strip() else ""

def get_transform_examples_with_multi():
    examples = get_transform_examples()
    return examples

def handle_parser_warnings(parser, context=""):
    """Handle default value warnings from the parser."""
    warnings = parser.get_default_warnings()
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        if context:
            st.info(f"💡 Tip: Set values for these variables in the 'Template Variables' section for {context}.")
        else:
            st.info("💡 Tip: Set values for these variables in the 'Template Variables' section above for more control.")
    return warnings

def initialize_session_state():
    """Initialize all required session state variables."""
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = list(get_transform_examples_with_multi().keys())[0]
    
    if 'variables' not in st.session_state:
        # Get default variables for the selected example
        examples = get_transform_examples_with_multi()
        default_vars = get_default_variables()
        if st.session_state.selected_example in default_vars:
            st.session_state.variables = default_vars[st.session_state.selected_example].copy()
        else:
            st.session_state.variables = {}

    if 'current_code' not in st.session_state:
        st.session_state.current_code = get_transform_examples_with_multi()[st.session_state.selected_example]
        
    if 'parsed_descriptor' not in st.session_state:
        st.session_state.parsed_descriptor = None

def get_transform_length(transform: Dict[str, Any], variables: Dict[str, Any]) -> sp.Expr:
    """Recursively calculate the total length of a transform's output space, substituting variables."""
    transform_type = transform.get('type')
    
    if transform_type == 'pass_through':
        value = transform.get('value', sp.Integer(1))
        if isinstance(value, sp.Basic):
            # Filter variables to only include numeric values that SymPy can handle
            safe_vars = {k: v for k, v in variables.items() if isinstance(v, (int, float, complex, sp.Basic))}
            return value.subs(safe_vars)
        else:
            return value
        
    elif transform_type in ('merge', 'xor'):
        total_length = sp.Integer(1)
        for sub_transform in transform.get('values', []):
            total_length *= get_transform_length(sub_transform, variables)
        return total_length
        
    return sp.Integer(1)

def display_variable_controls():
    """Display controls for adjusting template variables (both scalars and lists)."""
    st.subheader("Template Variables")
    
    updated_variables = {}
    
    if not hasattr(st.session_state, 'variable_info') or not st.session_state.variable_info:
        st.info("No template variables detected in code. Parse your code to find variables.")
        return
    
    # Get default values for the current example
    default_vars = get_default_variables()
    example_defaults = default_vars.get(st.session_state.selected_example, {})
    
    # Group variables by type
    scalar_vars = {}
    list_vars = {}
    
    for var_name, var_info in st.session_state.variable_info.items():
        if var_info.get('type') == 'list':
            list_vars[var_name] = var_info
        else:
            scalar_vars[var_name] = var_info
    
    # Display scalar variables
    if scalar_vars:
        st.markdown("**Scalar Variables**")
        for var_name, var_info in scalar_vars.items():
            # Get current value
            current_value = st.session_state.variables.get(var_name)
            if current_value is None:
                current_value = example_defaults.get(var_name, var_info.get('default_value', 1))
            
            # Ensure current_value is numeric
            try:
                current_value = int(current_value) if isinstance(current_value, (int, float)) else 1
            except (ValueError, TypeError):
                current_value = 1
            
            # Display control
            display_name = var_name
            if 'description' in var_info:
                display_name += f" ({var_info['description']})"
            
            value = st.number_input(
                display_name,
                min_value=1,
                value=current_value,
                key=f"var_{var_name}"
            )
            updated_variables[var_name] = value
    
    # Display list variables
    if list_vars:
        st.markdown("**List Variables**")
        for var_name, var_info in list_vars.items():
            # Get current value
            current_value = st.session_state.variables.get(var_name)
            if current_value is None:
                current_value = example_defaults.get(var_name, var_info.get('default_value', [2, 2]))
            
            # Ensure current_value is a list
            if not isinstance(current_value, list):
                current_value = var_info.get('default_value', [2, 2])
            
            # Display list control
            st.markdown(f"**{var_name}**")
            if 'description' in var_info:
                st.caption(var_info['description'])
            
            # Convert list to comma-separated string for text input
            list_str = ", ".join(map(str, current_value))
            
            new_list_str = st.text_input(
                f"Values for {var_name} (comma-separated)",
                value=list_str,
                key=f"list_{var_name}",
                help=f"Enter {var_info.get('expected_type', 'int')} values separated by commas, e.g., '2, 4, 8'"
            )
            
            # Parse the input back to a list
            try:
                if new_list_str.strip():
                    new_list = [int(x.strip()) for x in new_list_str.split(',') if x.strip()]
                    if not new_list:  # Empty list
                        new_list = var_info.get('default_value', [2, 4, 8])
                else:
                    new_list = var_info.get('default_value', [2, 4, 8])
                updated_variables[var_name] = new_list
            except ValueError:
                st.error(f"Invalid input for {var_name}. Please enter integers separated by commas.")
                updated_variables[var_name] = current_value
            
            # Show preview of current list
            st.caption(f"Current list: {updated_variables.get(var_name, current_value)}")
    
    # Check for changes
    variables_changed = False
    if hasattr(st.session_state, 'variables'):
        for var, value in updated_variables.items():
            if var not in st.session_state.variables or st.session_state.variables[var] != value:
                variables_changed = True
                break
    
    st.session_state.variables = updated_variables

def substitute_descriptor(descriptor, user_vars):
    """Recursively substitute user variable values into all SymPy expressions in the descriptor."""
    if isinstance(descriptor, dict):
        return {k: substitute_descriptor(v, user_vars) for k, v in descriptor.items()}
    elif isinstance(descriptor, list):
        return [substitute_descriptor(item, user_vars) for item in descriptor]
    elif isinstance(descriptor, sp.Basic):
        # Filter user_vars to only include numeric values that SymPy can handle
        safe_vars = {}
        for k, v in user_vars.items():
            if isinstance(v, (int, float, complex, sp.Basic)):
                safe_vars[k] = v
            # Skip lists, dicts, and other complex types
        return descriptor.subs(safe_vars)
    else:
        return descriptor

def create_hierarchical_merge_nodes(transform, input_symbols, lower_indices, upper_indices,
                                   structure, variables, dot, stage_idx, transform_idx, 
                                   actual_stage_num, prev_stage_output_nodes):
    """
    Create separate nodes for hierarchical merge structures.
    
    This function processes nested merge structures by creating intermediate nodes
    for each level of the hierarchy.
    
    Returns:
        tuple: (output_formulas, intermediate_node_ids) where intermediate_node_ids 
               contains information about which intermediate nodes were created
    """
    import sympy as sp
    from pytensor.tensor_descriptor import MergeTransform, PassThroughTransform
    from tensor_transforms import TensorTransformParser
    
    print(f"DEBUG: Creating hierarchical merge nodes for structure: {structure}")
    
    parser = TensorTransformParser()
    intermediate_nodes = {}
    intermediate_formulas = {}
    symbol_to_input_idx = {}
    created_intermediate_nodes = []  # Track created intermediate nodes for edge connections
    
    # Map input symbols to their indices for tracking
    for idx, symbol in zip(lower_indices, input_symbols):
        symbol_to_input_idx[symbol] = idx
    
    def process_structure_recursive(struct, current_inputs, depth=0):
        """Recursively process the hierarchical structure."""
        indent = "  " * depth
        print(f"DEBUG: {indent}Processing structure at depth {depth}: {struct}")
        
        result_symbols = []
        
        for i, item in enumerate(struct):
            if isinstance(item, dict):
                if item.get('type') == 'pass_through':
                    # Simple pass-through - use corresponding input symbol
                    if i < len(current_inputs):
                        result_symbols.append(current_inputs[i])
                        print(f"DEBUG: {indent}Pass-through: {current_inputs[i]}")
                    else:
                        # Fallback symbol
                        fallback_symbol = sp.Symbol(f"pt_{depth}_{i}")
                        result_symbols.append(fallback_symbol)
                        print(f"DEBUG: {indent}Pass-through fallback: {fallback_symbol}")
                
                elif item.get('type') == 'merge':
                    # Nested merge - create intermediate node
                    nested_values = item.get('values', [])
                    print(f"DEBUG: {indent}Processing nested merge with {len(nested_values)} values")
                    
                    # For the nested merge, we need to take the next available inputs
                    # Based on the structure: [A, merge(B, C)] - the nested merge should get B and C
                    nested_inputs = []
                    needed_inputs = len(nested_values)  # Each value in the nested merge needs one input
                    
                    # Take the next 'needed_inputs' symbols from the remaining current_inputs
                    start_idx = len(result_symbols)  # Skip inputs already consumed by previous items
                    for j in range(needed_inputs):
                        if start_idx + j < len(input_symbols):
                            nested_inputs.append(input_symbols[start_idx + j])
                    
                    print(f"DEBUG: {indent}Nested merge using inputs: {nested_inputs}")
                    
                    # Create the intermediate merge transform
                    nested_lengths = []
                    for nested_item in nested_values:
                        lengths = parser._flatten_merge_lengths(nested_item, variables)
                        nested_lengths.extend(lengths)
                    
                    print(f"DEBUG: {indent}Creating intermediate merge with lengths: {nested_lengths}")
                    intermediate_merge = MergeTransform(lengths=nested_lengths)
                    
                    # Create intermediate node
                    intermediate_node_id = f"s{actual_stage_num}_t{transform_idx}_intermediate_{depth}_{i}"
                    
                    # Calculate the merge formula
                    if nested_inputs:
                        try:
                            intermediate_formula = intermediate_merge.sympy_calculate_upper(nested_inputs)[0]
                            print(f"DEBUG: {indent}Intermediate formula: {intermediate_formula}")
                        except Exception as e:
                            print(f"DEBUG: {indent}Failed to create intermediate formula: {e}")
                            intermediate_formula = sum(nested_inputs) if nested_inputs else sp.Symbol("intermediate")
                    else:
                        intermediate_formula = sp.Symbol("intermediate")
                    
                    # Store intermediate result
                    intermediate_nodes[intermediate_node_id] = intermediate_formula
                    intermediate_formulas[intermediate_node_id] = intermediate_formula
                    
                    # Track this intermediate node for later edge connections
                    intermediate_info = {
                        'node_id': intermediate_node_id,
                        'input_indices': [start_idx + j for j in range(needed_inputs)],
                        'formula': intermediate_formula
                    }
                    created_intermediate_nodes.append(intermediate_info)
                    
                    # Substitute variables and create the DOT node
                    # Filter variables to only include numeric values that SymPy can handle
                    safe_vars = {k: v for k, v in variables.items() if isinstance(v, (int, float, complex, sp.Basic))}
                    substituted_formula = intermediate_formula.subs(safe_vars)
                    simplified_formula = sp.simplify(substituted_formula)
                    
                    try:
                        formula_str = str(simplified_formula)
                        if (not any(func in formula_str for func in ['floor', 'ceiling', 'sqrt']) 
                            and '/' not in formula_str and '**' not in formula_str):
                            try:
                                pretty_str = sp.pretty(simplified_formula, use_unicode=False)
                                if '\n' not in pretty_str:
                                    formula_str = pretty_str
                            except:
                                pass
                    except Exception:
                        formula_str = f"intermediate_{depth}_{i}"
                    
                    label = f'<<FONT POINT-SIZE="12">{formula_str}</FONT>>'
                    dot.node(intermediate_node_id, label, fillcolor="#ffcc99")  # Orange for intermediate
                    print(f"DEBUG: {indent}Created intermediate node {intermediate_node_id}")
                    
                    # Create edges from inputs to intermediate node - use the actual input indices
                    for j, nested_input in enumerate(nested_inputs):
                        actual_input_idx = start_idx + j
                        if actual_input_idx in prev_stage_output_nodes:
                            dot.edge(prev_stage_output_nodes[actual_input_idx], intermediate_node_id, 
                                    label="Merge")
                    
                    # Use the intermediate result as a symbol for further processing
                    result_symbols.append(intermediate_formula)
                    
            else:
                # Handle non-dict items (shouldn't happen in proper structure)
                print(f"DEBUG: {indent}Unexpected item type: {type(item)}")
                if current_inputs:
                    result_symbols.append(current_inputs.pop(0))
        
        return result_symbols
    
    # Process the hierarchical structure
    working_inputs = input_symbols.copy()
    processed_symbols = process_structure_recursive(structure, working_inputs)
    
    # Create the final merge if needed
    if len(processed_symbols) > 1:
        # For hierarchical merge, we need to create the correct lengths for the final merge
        # The final merge should combine the outputs of the hierarchical structure
        # Calculate the effective lengths for the final merge
        final_lengths = []
        
        # For each processed symbol, determine its contribution
        for i, (symbol, struct_item) in enumerate(zip(processed_symbols, structure)):
            if isinstance(struct_item, dict):
                if struct_item.get('type') == 'pass_through':
                    # Pass-through contributes its original length
                    value = struct_item.get('value', 1)
                    if hasattr(value, 'subs'):
                        # It's a SymPy expression
                        safe_variables = {k: v for k, v in variables.items() if not isinstance(v, list)}
                        final_lengths.append(int(value.subs(safe_variables)))
                    else:
                        final_lengths.append(int(value))
                elif struct_item.get('type') == 'merge':
                    # Nested merge contributes the product of its input lengths
                    nested_lengths = []
                    for nested_item in struct_item.get('values', []):
                        nested_lengths.extend(parser._flatten_merge_lengths(nested_item, variables))
                    final_lengths.append(int(sp.prod(nested_lengths)))
            else:
                # Fallback
                final_lengths.append(1)
        
        print(f"DEBUG: Final merge lengths calculated as: {final_lengths}")
        
        # Create final merge transform with correct lengths
        final_merge = MergeTransform(lengths=final_lengths)
        try:
            final_formula = final_merge.sympy_calculate_upper(processed_symbols)[0]
            print(f"DEBUG: Final hierarchical merge formula: {final_formula}")
            return ([final_formula], created_intermediate_nodes)
        except Exception as e:
            print(f"DEBUG: Failed to create final merge formula: {e}")
            return ([sum(processed_symbols)], created_intermediate_nodes)
    elif len(processed_symbols) == 1:
        return (processed_symbols, created_intermediate_nodes)
    else:
        # Fallback
        return ([sum(input_symbols) if input_symbols else sp.Symbol("merged")], created_intermediate_nodes)

def build_transformation_graph_from_pytensor(descriptors, variables):
    """
    Build a Graphviz DOT graph using pytensor objects and their sympy methods.
    
    This creates the Upper → Lower transformation graph, showing how logical 
    dimensions (upper) are transformed to physical memory layout (lower).
    """
    import math  # Needed for math.prod() in FixedFinalDescriptor
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", splines="ortho", compound="true")
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Define stage colors for visual separation
    stage_colors = [
        "#ffeeee",  # Light red for input stage
        "#eeffee",  # Light green for stage 1
        "#eeeeff",  # Light blue for stage 2 
        "#ffffe0",  # Light yellow for stage 3
        "#f0e0ff",  # Light purple for stage 4
        "#e0ffff",  # Light cyan for stage 5
        "#ffe0e0",  # Light pink for stage 6
    ]
    


    timestamp = int(time.time() * 1000)
    var_comment = f"Variables: {sorted(variables.items())} - Time: {timestamp}"
    dot.attr(comment=var_comment)
    
    vars_display = ", ".join(f"{k}={v}" for k, v in sorted(variables.items())[:5])
    dot.node("title", f"Upper → Lower Graph - Variables: {vars_display}", shape="note", style="filled", fillcolor="lightyellow")

    if not descriptors:
        return dot

    parser = TensorTransformParser()
    
    all_lines = [line.strip() for line in " ".join(descriptors).split('\n') if line.strip()]
    
    realistic_var_names = [
        'b_lds_block_desc',
        'b_lds_block_desc_permuted', 
        'b_lds_block_desc_unmerged',
        'b_lds_block_desc_kn'
    ]
    
    pytensor_descriptors = []
    descriptor_registry = {}
    
    for i, desc_str in enumerate(descriptors):
        try:
            parser.descriptor_registry = descriptor_registry
            tensor_desc = parser.create_pytensor_descriptor(desc_str, variables)
            pytensor_descriptors.append(tensor_desc)
            
            print(f"DEBUG: Created pytensor descriptor {i} with {len(tensor_desc.get_transforms())} transforms")
            
            if i < len(realistic_var_names):
                var_name = realistic_var_names[i]
                descriptor_registry[var_name] = tensor_desc
                
        except Exception as e:
            st.error(f"Failed to create pytensor descriptor {i}: {e}")
            dot.node(f"error_{i}", f"Error: {str(e)[:50]}...", fillcolor="#ffcccc")
            continue

    if not pytensor_descriptors:
        st.error("No valid descriptors could be created")
        return dot

    # For multi-descriptor examples, process each descriptor as a sequential stage
    # Each descriptor builds on the previous one, so we need to show the progression
    
    # For the first descriptor, if it's a naive descriptor, use its actual dimension count
    # This fixes the issue where naive descriptors were getting too many input dimensions
    first_desc = pytensor_descriptors[0]
    first_desc_str = descriptors[0].strip()
    
    # Special handling for the first descriptor if it's make_naive_tensor_descriptor_packed
    is_first_naive_packed = "make_naive_tensor_descriptor_packed" in first_desc_str
    
    if is_first_naive_packed:
        # For naive_packed descriptors, we use storage perspective (1 input)
        # The max_input_dim calculation is not relevant since we create storage input
        max_input_dim = 1  # Will be overridden by storage logic anyway
        print(f"DEBUG: First descriptor is naive_packed, will use storage perspective")
    elif "make_naive_tensor_descriptor(" in first_desc_str:
        # For non-packed naive descriptors, use the actual dimension count
        max_input_dim = first_desc.get_num_of_dimension()
        print(f"DEBUG: Using naive descriptor dimension count: {max_input_dim}")
    else:
        # For other descriptors, analyze all descriptors to find the maximum input dimension needed
        max_input_dim = 0
        for tensor_desc in pytensor_descriptors:
            all_lower_idss = tensor_desc.get_lower_dimension_hidden_idss()
            for lower_ids in all_lower_idss:
                if lower_ids:
                    max_input_dim = max(max_input_dim, max(lower_ids) + 1)
        
        print(f"DEBUG: max_input_dim calculated as {max_input_dim}")
        
        # If no transforms found, use the first descriptor's dimension count
        if max_input_dim == 0:
            max_input_dim = first_desc.get_num_of_dimension()
            print(f"DEBUG: max_input_dim fallback to first descriptor dimension count: {max_input_dim}")
        
        # For naive descriptors, we need to ensure we have enough input dimensions
        # by checking the actual transforms in the pytensor object
        if max_input_dim < 6:  # Assume at least 6 dimensions are often needed
            all_upper_idss = first_desc.get_upper_dimension_hidden_idss()
            all_lower_idss = first_desc.get_lower_dimension_hidden_idss()
            
            # Find the maximum INPUT dimension referenced in any transform
            # Only look at lower_ids since those represent input dimensions
            max_dim_needed = 0
            for lower_ids in all_lower_idss:
                if lower_ids:
                    max_dim_needed = max(max_dim_needed, max(lower_ids) + 1)
            # NOTE: Don't look at upper_ids - those are OUTPUT dimensions, not input!
            
            if max_dim_needed > max_input_dim:
                max_input_dim = max_dim_needed
                print(f"DEBUG: max_input_dim updated to {max_input_dim} based on transform INPUT requirements")
    
    print(f"DEBUG: is_first_naive_packed = {is_first_naive_packed}")
    
    # Create initial input nodes with stage clustering
    prev_stage_output_nodes = {}
    current_formulas = {}
    
    # Create input stage subgraph
    with dot.subgraph(name='cluster_input') as input_cluster:
        input_cluster.attr(style='filled', fillcolor=stage_colors[0], label='Input Stage')
        input_cluster.attr(fontsize='14', fontweight='bold')
    
    if is_first_naive_packed:
        print("DEBUG: Using naive_packed path - consistent storage perspective")
        # For naive_tensor_descriptor_packed, show the storage perspective consistently:
        # Forward: 1 linear storage -> N logical dimensions (via UnmergeTransform)
        # Backward: N logical dimensions -> 1 linear storage (via UnmergeTransform⁻¹)
        first_desc = pytensor_descriptors[0]
        num_logical_dims = first_desc.get_num_of_dimension()
        
        print(f"DEBUG: num_logical_dims = {num_logical_dims}")
        
        # Create single input node for the linear storage (dimension 0 in hidden space)
        storage_node_id = "input_storage"
        prev_stage_output_nodes[0] = storage_node_id
        
        try:
            element_space_size = first_desc.get_element_space_size()
            storage_label = f"storage ({element_space_size})"
        except:
            storage_label = "storage"
        
        # Add input node to the input cluster
        with dot.subgraph(name='cluster_input') as input_cluster:
            input_cluster.node(storage_node_id, storage_label, fillcolor="#ffcccc")
        current_formulas[storage_node_id] = sp.Symbol("storage")
        
        print(f"DEBUG: Created single storage input {storage_node_id} for linear dimension 0")
    else:
        print(f"DEBUG: Using non-packed path, creating {max_input_dim} input dimensions")
        
        # Create all input nodes within the input cluster
        with dot.subgraph(name='cluster_input') as input_cluster:
            for k in range(max_input_dim):
                node_id = f"input_d{k}"
                prev_stage_output_nodes[k] = node_id
                
                print(f"DEBUG: Created input {node_id} mapped to index {k}")
                
                # Get dimension length - for unmerge transforms, calculate required input size
                try:
                    first_desc = pytensor_descriptors[0]
                    
                    # Check if this is an unmerge transform case
                    all_transforms = first_desc.get_transforms()
                    if (len(all_transforms) > 0 and 
                        isinstance(all_transforms[0], pytensor.tensor_descriptor.UnmergeTransform)):
                        
                        # For unmerge: input dimension should have size = product of unmerge lengths
                        unmerge_transform = all_transforms[0]
                        if hasattr(unmerge_transform, 'lengths') and unmerge_transform.lengths:
                            import math
                            required_input_size = math.prod(unmerge_transform.lengths)
                            dim_label = f"d{k} ({required_input_size})"
                            print(f"DEBUG: Input d{k} calculated for unmerge with lengths {unmerge_transform.lengths} -> size {required_input_size}")
                        else:
                            dim_label = f"d{k}"
                    else:
                        # Regular case: use descriptor's reported length
                        if k < first_desc.get_num_of_dimension():
                            dim_length = first_desc.get_length(k) if hasattr(first_desc, 'get_length') else '?'
                            dim_label = f"d{k} ({dim_length})"
                        else:
                            dim_label = f"d{k}"
                except:
                    dim_label = f"d{k}"
                input_cluster.node(node_id, dim_label, fillcolor="#ffcccc")
                current_formulas[node_id] = sp.Symbol(f"d{k}")
    
    # Process each descriptor as a separate stage
    actual_stage_num = 1  # Track actual stage numbers starting from 1
    for stage_idx, tensor_desc in enumerate(pytensor_descriptors):
        print(f"DEBUG: Processing stage_idx={stage_idx}")
        stage_output_nodes = {}
        next_formulas = {}
        
        all_transforms = tensor_desc.get_transforms()
        all_lower_idss = tensor_desc.get_lower_dimension_hidden_idss()
        all_upper_idss = tensor_desc.get_upper_dimension_hidden_idss()
        
        print(f"DEBUG: Stage {stage_idx} has {len(all_transforms)} total transforms")
        
        # Each descriptor defines its own complete transformation pipeline
        # For transform_tensor_descriptor calls, we process the transforms defined in THAT descriptor
        # not accumulated transforms from previous descriptors
        
        # Check if this is a naive descriptor or transform descriptor
        desc_str = descriptors[stage_idx].strip() if stage_idx < len(descriptors) else ""
        is_naive_descriptor = "make_naive_tensor_descriptor" in desc_str
        is_transform_descriptor = "transform_tensor_descriptor" in desc_str
        
        if is_naive_descriptor:
            # Naive descriptors: process all transforms (usually just EmbedTransform)
            new_transforms = all_transforms
            new_lower_idss = all_lower_idss
            new_upper_idss = all_upper_idss
        elif is_transform_descriptor:
            # Transform descriptors: process the transforms specified in make_tuple(...)
            # These are the transforms that get applied to the input descriptor
            # For A LDS example: stage 1 should process [XorTransform, PassThroughTransform]
            
            # Extract the transform count from the original descriptor parsing
            # The transforms in the tuple are what we want to visualize as this stage
            parser = TensorTransformParser()
            try:
                parsed_desc = parser.parse_tensor_descriptor(desc_str)
                if parsed_desc['type'] == 'transform':
                    stage_transform_count = len(parsed_desc['transforms'])
                    print(f"DEBUG: Stage {stage_idx} descriptor defines {stage_transform_count} transforms")
                    
                    # Process the last N transforms where N is the number defined in this descriptor
                    new_transforms = all_transforms[-stage_transform_count:] if stage_transform_count > 0 else []
                    new_lower_idss = all_lower_idss[-stage_transform_count:] if stage_transform_count > 0 else []
                    new_upper_idss = all_upper_idss[-stage_transform_count:] if stage_transform_count > 0 else []
                else:
                    # Fallback: use all transforms
                    new_transforms = all_transforms
                    new_lower_idss = all_lower_idss
                    new_upper_idss = all_upper_idss
            except Exception as e:
                print(f"DEBUG: Failed to parse descriptor for stage {stage_idx}, using all transforms: {e}")
                new_transforms = all_transforms
                new_lower_idss = all_lower_idss
                new_upper_idss = all_upper_idss
        else:
            # Unknown descriptor type: use all transforms
            new_transforms = all_transforms
            new_lower_idss = all_lower_idss
            new_upper_idss = all_upper_idss
        
        print(f"DEBUG: Stage {stage_idx} will process {len(new_transforms)} new transforms")
        
        # Special handling for naive_tensor_descriptor_packed first transform
        # For naive_packed descriptors, we want to show the logical interface (user perspective)
        # while still displaying the internal UnmergeTransform that handles storage mapping
        if stage_idx == 0 and is_first_naive_packed and len(new_transforms) > 0:
            if isinstance(new_transforms[0], pytensor.tensor_descriptor.UnmergeTransform):
                print(f"DEBUG: Stage {stage_idx} has UnmergeTransform, will show both logical and storage perspectives")
        
        # Skip stages with no new transforms to process
        if not new_transforms:
            print(f"DEBUG: Stage {stage_idx} has no new transforms to process, skipping")
            # Don't increment actual_stage_num for skipped stages
            continue
            
        print(f"DEBUG: Stage {stage_idx} proceeding with {len(new_transforms)} new transforms, using actual stage number {actual_stage_num}")
        
        # Create stage subgraph for visual separation
        stage_color = stage_colors[min(actual_stage_num, len(stage_colors)-1)]
        with dot.subgraph(name=f'cluster_stage_{actual_stage_num}') as stage_cluster:
            stage_cluster.attr(style='filled', fillcolor=stage_color, 
                             label=f'Stage {actual_stage_num}', 
                             fontsize='14', fontweight='bold')
            
            # Determine if this is the final stage
            is_final_stage = (stage_idx == len(pytensor_descriptors) - 1)
            
            # For the final stage, we need to determine the actual final output dimensions
            # by looking at the final descriptor's dimension count
            if is_final_stage:
                final_output_dims = tensor_desc.get_num_of_dimension()
                print(f"DEBUG: Final stage should have {final_output_dims} output dimensions")
            
            # Process each new transform in this stage
            for i, transform in enumerate(new_transforms):
                if i >= len(new_lower_idss) or i >= len(new_upper_idss):
                    continue
                    
                lower_indices = new_lower_idss[i]
                upper_indices = new_upper_idss[i]
                
                print(f"DEBUG: Transform {i} - lower_indices: {lower_indices}, upper_indices: {upper_indices}")
                print(f"DEBUG: Transform {i} type: {transform.__class__.__name__}")
                print(f"DEBUG: prev_stage_output_nodes keys: {list(prev_stage_output_nodes.keys())}")
                
                # Get input symbols for this transform
                input_symbols = []
                for idx in lower_indices:
                    if idx in prev_stage_output_nodes:
                        prev_node_id = prev_stage_output_nodes[idx]
                        if prev_node_id in current_formulas:
                            input_symbols.append(current_formulas[prev_node_id])
                        else:
                            input_symbols.append(sp.Symbol(f"d_{idx}"))
                    else:
                        input_symbols.append(sp.Symbol(f"d_{idx}"))

                try:
                    # Initialize variable for hierarchical merge tracking
                    hierarchical_intermediate_nodes = []
                    
                    # Apply the transform - use correct directional terminology  
                    # In forward graph, we're going from inputs to outputs (upper → lower)
                    if isinstance(transform, pytensor.tensor_descriptor.UnmergeTransform):
                        # UnmergeTransform: Forward direction always decomposes (single → multiple)
                        # Use sympy_calculate_upper: takes 1 lower symbol → multiple upper symbols
                        if len(input_symbols) == 1:
                            output_formulas = transform.sympy_calculate_upper(input_symbols)
                        else:
                            # If we have multiple inputs, combine them first then decompose
                            merged_input = sum(input_symbols) if len(input_symbols) > 1 else sp.Symbol("merged")
                            output_formulas = transform.sympy_calculate_upper([merged_input])
                    elif isinstance(transform, pytensor.tensor_descriptor.EmbedTransform):
                        # EmbedTransform direction depends on the context:
                        # - For the first (naive) descriptor: multi-dimensional coordinates -> linear address  
                        # - For decomposition: linear address -> multi-dimensional coordinates
                        
                        # Check if this is the first naive descriptor
                        is_first_descriptor = (stage_idx == 0)
                        first_desc_str = descriptors[0].strip() if descriptors else ""
                        is_naive_first = "make_naive_tensor_descriptor" in first_desc_str
                        
                        if is_first_descriptor and is_naive_first:
                            # First descriptor: combines coordinates into linear address
                            # Multiple inputs → single output: use calculate_upper_index
                            expected_input_dims = len(transform.lengths)
                            
                            # Create input symbols for each coordinate
                            coordinate_symbols = []
                            for dim_idx in range(expected_input_dims):
                                if dim_idx in prev_stage_output_nodes:
                                    # Use the input node's formula
                                    input_node = prev_stage_output_nodes[dim_idx]
                                    coordinate_symbols.append(current_formulas[input_node])
                                else:
                                    coordinate_symbols.append(sp.Symbol(f"d{dim_idx}"))
                            
                            try:
                                output_formulas = transform.sympy_calculate_upper(coordinate_symbols)
                                print(f"DEBUG: EmbedTransform.sympy_calculate_upper({coordinate_symbols}) -> {output_formulas}")
                            except Exception as embed_error:
                                print(f"DEBUG: EmbedTransform.sympy_calculate_upper failed: {embed_error}")
                                # Fallback: create simple sum
                                output_formulas = [sum(coordinate_symbols)]
                        else:
                            # Decomposition case: linear address -> coordinates
                            # Single input → multiple outputs: use calculate_lower_index
                            if input_symbols:
                                single_input = input_symbols[0]
                            else:
                                single_input = sp.Symbol("d0")
                            
                            try:
                                output_formulas = transform.sympy_calculate_lower([single_input])
                                print(f"DEBUG: EmbedTransform.sympy_calculate_lower([{single_input}]) -> {len(output_formulas)} outputs")
                            except Exception as embed_error:
                                print(f"DEBUG: EmbedTransform.sympy_calculate_lower failed: {embed_error}")
                                # Fallback: create coordinate symbols
                                output_formulas = [sp.Symbol(f"coord{i}") for i in range(len(upper_indices))]
                    elif isinstance(transform, pytensor.tensor_descriptor.MergeTransform):
                        # Check if this is a hierarchical merge
                        if hasattr(transform, '_hierarchy_info') and transform._hierarchy_info.get('is_hierarchical'):
                            print(f"DEBUG: Processing hierarchical merge with structure: {transform._hierarchy_info['structure']}")
                            # Create separate nodes for hierarchical merge
                            output_formulas, intermediate_node_info = create_hierarchical_merge_nodes(
                                transform, input_symbols, lower_indices, upper_indices,
                                transform._hierarchy_info['structure'], variables, dot,
                                stage_idx, i, actual_stage_num, prev_stage_output_nodes
                            )
                            # Store intermediate node info for custom edge creation
                            hierarchical_intermediate_nodes = intermediate_node_info
                        else:
                            # Regular flat merge - use directional semantics
                            # MergeTransform: upper (multiple) → lower (single)
                            if len(lower_indices) > 1 and len(upper_indices) == 1:
                                # Case: multiple inputs → 1 output: use calculate_upper_index (composition)
                                try:
                                    output_formulas = transform.sympy_calculate_upper(input_symbols)
                                    print(f"DEBUG: MergeTransform.sympy_calculate_upper({input_symbols}) -> {len(output_formulas)} outputs")
                                except Exception as merge_error:
                                    print(f"DEBUG: MergeTransform.sympy_calculate_upper failed: {merge_error}")
                                    formula = sum(input_symbols) if input_symbols else sp.Symbol("merged")
                                    output_formulas = [formula]
                            else:
                                # Case: 1 input → multiple outputs: use calculate_lower_index (decomposition)
                                single_input = input_symbols[0] if input_symbols else sp.Symbol("merged")
                                try:
                                    output_formulas = transform.sympy_calculate_lower([single_input])
                                    print(f"DEBUG: MergeTransform.sympy_calculate_lower({single_input}) -> {len(output_formulas)} outputs")
                                except Exception as merge_error:
                                    print(f"DEBUG: MergeTransform.sympy_calculate_lower failed: {merge_error}")
                                    output_formulas = [sp.Symbol(f"comp{i}") for i in range(len(upper_indices))]
                    elif isinstance(transform, pytensor.tensor_descriptor.UnmergeTransform):
                        # Duplicate UnmergeTransform handling - use same forward logic
                        if len(input_symbols) == 1:
                            output_formulas = transform.sympy_calculate_upper(input_symbols)
                        else:
                            merged_input = sum(input_symbols) if len(input_symbols) > 1 else sp.Symbol("merged")
                            output_formulas = transform.sympy_calculate_upper([merged_input])
                    else:
                        # Default: use calculate_lower_index for upper → lower direction
                        output_formulas = transform.sympy_calculate_lower(input_symbols)
                    
                    # Create output nodes for this transform
                    print(f"DEBUG: Transform {i} has {len(output_formulas)} output formulas for {len(upper_indices)} upper indices")
                    
                    # For final stage, only create nodes for dimensions that actually exist in the final output
                    if is_final_stage:
                        # Filter upper_indices to only include those within the final output dimension count
                        filtered_upper_indices = [idx for idx in upper_indices if idx < final_output_dims]
                        print(f"DEBUG: Final stage filtering upper_indices {upper_indices} to {filtered_upper_indices}")
                    else:
                        filtered_upper_indices = upper_indices
                    
                    # Create output nodes for each formula within the stage cluster
                    for j, output_idx in enumerate(filtered_upper_indices):
                        if j < len(output_formulas):
                            # Use the actual stage number for node naming
                            node_id = f"s{actual_stage_num}_t{i}_d{output_idx}"
                            stage_output_nodes[output_idx] = node_id
                            
                            print(f"DEBUG: Creating node {node_id} for stage {stage_idx} transform {i}")
                            
                            formula = output_formulas[j]
                            next_formulas[node_id] = formula
                            
                            # Substitute variables and simplify
                            # Filter variables to only include numeric values that SymPy can handle
                            safe_vars = {k: v for k, v in variables.items() if isinstance(v, (int, float, complex, sp.Basic))}
                            substituted_formula = formula.subs(safe_vars)
                            simplified_formula = sp.simplify(substituted_formula)
                            
                            try:
                                formula_str = str(simplified_formula)
                                
                                # Pretty print simple formulas
                                if (not any(func in formula_str for func in ['floor', 'ceiling', 'sqrt', 'sin', 'cos', 'tan']) 
                                    and '/' not in formula_str and '**' not in formula_str):
                                    try:
                                        pretty_str = sp.pretty(simplified_formula, use_unicode=False)
                                        if '\n' not in pretty_str:
                                            formula_str = pretty_str
                                    except:
                                        pass
                                
                                # Clean up XOR notation
                                import re
                                formula_str = formula_str.replace('XorFunction', 'xor')
                                formula_str = formula_str.replace('⊕', ' xor ')
                                formula_str = re.sub(r'xor\(([^,]+),\s*([^)]+)\)', r'\1 xor \2', formula_str)
                                
                            except Exception:
                                formula_str = f"d{output_idx}"
                            
                            label = f'<<FONT POINT-SIZE="12">{formula_str}</FONT>>'
                            stage_cluster.node(node_id, label, fillcolor="#c0ffc0")
                            print(f"DEBUG: Added DOT node {node_id} with label {formula_str}")
                            
                            # Create edges from input nodes to this output node
                            # Special handling for hierarchical merges
                            if (isinstance(transform, pytensor.tensor_descriptor.MergeTransform) and 
                                hasattr(transform, '_hierarchy_info') and transform._hierarchy_info.get('is_hierarchical')):
                                # For hierarchical merges, create custom edges
                                print(f"DEBUG: Creating custom edges for hierarchical merge")
                                
                                # Get the structure to understand the hierarchy
                                structure = transform._hierarchy_info['structure']
                                
                                # Create edges based on the hierarchical structure
                                consumed_inputs = set()  # Track which inputs have been consumed by intermediate nodes
                                
                                # First, mark inputs consumed by intermediate nodes
                                if 'hierarchical_intermediate_nodes' in locals():
                                    for intermediate_info in hierarchical_intermediate_nodes:
                                        consumed_inputs.update(intermediate_info['input_indices'])
                                        print(f"DEBUG: Inputs {intermediate_info['input_indices']} consumed by intermediate {intermediate_info['node_id']}")
                                
                                # Create edges from remaining inputs and intermediate nodes to final node
                                for struct_idx, struct_item in enumerate(structure):
                                    if isinstance(struct_item, dict):
                                        if struct_item.get('type') == 'pass_through':
                                            # Pass-through connects directly from input to final
                                            input_idx = struct_idx  # For the first element (A)
                                            if input_idx not in consumed_inputs and input_idx in prev_stage_output_nodes:
                                                transform_name = "Merge"
                                                dot.edge(prev_stage_output_nodes[input_idx], node_id, 
                                                       label=transform_name)
                                                print(f"DEBUG: Created edge from input {input_idx} to final node {node_id}")
                                        
                                        elif struct_item.get('type') == 'merge':
                                            # Nested merge connects from intermediate node to final
                                            if 'hierarchical_intermediate_nodes' in locals():
                                                for intermediate_info in hierarchical_intermediate_nodes:
                                                    if struct_idx == 1:  # This is the second item in structure (the nested merge)
                                                        transform_name = "Merge"
                                                        dot.edge(intermediate_info['node_id'], node_id, 
                                                               label=transform_name)
                                                        print(f"DEBUG: Created edge from intermediate {intermediate_info['node_id']} to final node {node_id}")
                            
                            # Special handling for EmbedTransform in naive descriptors
                            elif (isinstance(transform, pytensor.tensor_descriptor.EmbedTransform) and 
                                stage_idx == 0 and "make_naive_tensor_descriptor" in descriptors[0]):
                                # For naive descriptor EmbedTransform, connect all input dimensions
                                expected_input_dims = len(transform.lengths)
                                for dim_idx in range(expected_input_dims):
                                    if dim_idx in prev_stage_output_nodes:
                                        transform_name = transform.__class__.__name__.replace('Transform', '')
                                        dot.edge(prev_stage_output_nodes[dim_idx], node_id, 
                                               label=transform_name)
                            else:
                                # Standard edge creation for other transforms
                                for input_idx in lower_indices:
                                    if input_idx in prev_stage_output_nodes:
                                        transform_name = transform.__class__.__name__.replace('Transform', '')
                                        dot.edge(prev_stage_output_nodes[input_idx], node_id, 
                                               label=transform_name)
                    
                except Exception as e:
                    # Provide more helpful error messages for common issues
                    error_msg = str(e)
                    if "doesn't match transform dimension" in error_msg:
                        if hasattr(transform, 'ndim') and hasattr(transform, 'lengths'):
                            error_msg += f"\nTransform '{transform.__class__.__name__}' with lengths {transform.lengths} expects {transform.ndim} input symbols."
                            error_msg += f"\nCheck that the upper_dimensions in your descriptor match the transform's expected input count."
                    elif "SympifyError" in error_msg and "[" in error_msg and "]" in error_msg:
                        error_msg += f"\nThe transform received a list instead of a numeric value. Check your variable definitions."
                    
                    st.warning(f"Failed to generate formula for transform {transform}: {error_msg}")
                    # Create error nodes with actual stage number
                    for j, output_idx in enumerate(upper_indices):
                        node_id = f"s{actual_stage_num}_t{i}_d{output_idx}"
                        stage_output_nodes[output_idx] = node_id
                        next_formulas[node_id] = sp.Symbol(f"d{output_idx}")
                        stage_cluster.node(node_id, f"d{output_idx}", fillcolor="#ffcccc")
                        
                        # Create edges for error case - use same logic as successful case
                        if (isinstance(transform, pytensor.tensor_descriptor.EmbedTransform) and 
                            stage_idx == 0 and "make_naive_tensor_descriptor" in descriptors[0]):
                            # For naive descriptor EmbedTransform, connect all input dimensions
                            expected_input_dims = len(transform.lengths)
                            for dim_idx in range(expected_input_dims):
                                if dim_idx in prev_stage_output_nodes:
                                    transform_name = transform.__class__.__name__.replace('Transform', '')
                                    dot.edge(prev_stage_output_nodes[dim_idx], node_id, 
                                           label=transform_name)
                        else:
                            # Standard edge creation for other transforms
                            for input_idx in lower_indices:
                                if input_idx in prev_stage_output_nodes:
                                    transform_name = transform.__class__.__name__.replace('Transform', '')
                                    dot.edge(prev_stage_output_nodes[input_idx], node_id, 
                                           label=transform_name)
        
        # Update for next stage
        if stage_output_nodes:
            # Merge new stage outputs with existing nodes instead of replacing
            # This preserves access to original inputs and previous stage outputs
            prev_stage_output_nodes.update(stage_output_nodes)
            current_formulas.update(next_formulas)
            print(f"DEBUG: Stage {stage_idx} completed, merged stage outputs with existing nodes")
            print(f"DEBUG: Available nodes for next stage: {list(prev_stage_output_nodes.keys())}")
            

            
            # Increment actual stage number only when we actually process a stage
            actual_stage_num += 1
        else:
            print(f"DEBUG: Stage {stage_idx} completed with no output nodes")

    # Create final output nodes for forward graph
    if pytensor_descriptors:
        # BUGFIX: The last descriptor contains ALL accumulated transforms, but we need 
        # only the final stage transforms. Extract just the final transforms.
        full_descriptor = pytensor_descriptors[-1]
        
        # Get the final stage's transform count
        final_stage_idx = len(pytensor_descriptors) - 1
        if final_stage_idx < len(descriptors):
            desc_str = descriptors[final_stage_idx].strip()
            parser = TensorTransformParser()
            try:
                parsed_desc = parser.parse_tensor_descriptor(desc_str)
                if parsed_desc['type'] == 'transform':
                    final_stage_transform_count = len(parsed_desc['transforms'])
                    print(f"DEBUG: Final stage has {final_stage_transform_count} transforms")
                    
                    # Create a final descriptor with only the last stage's transforms
                    all_transforms = full_descriptor.get_transforms()
                    all_lower_idss = full_descriptor.get_lower_dimension_hidden_idss()
                    all_upper_idss = full_descriptor.get_upper_dimension_hidden_idss()
                    
                    final_transforms = all_transforms[-final_stage_transform_count:]
                    final_lower_idss = all_lower_idss[-final_stage_transform_count:]
                    final_upper_idss = all_upper_idss[-final_stage_transform_count:]
                    
                    # Build the final descriptor using the pytensor library
                    from pytensor.tensor_descriptor import transform_tensor_descriptor_for_visualization as transform_tensor_descriptor
                    
                    # Use the previous stage's descriptor as input
                    if len(pytensor_descriptors) > 1:
                        input_descriptor = pytensor_descriptors[-2]
                    else:
                        input_descriptor = pytensor_descriptors[0]
                    
                    # FIXED: Instead of trying to create a new descriptor (which accumulates all transforms),
                    # manually calculate the lengths from the final MergeTransforms
                    print(f"DEBUG: Created corrected final descriptor with {len(final_transforms)} transforms")
                    print(f"DEBUG: Final transforms: {[t.__class__.__name__ for t in final_transforms]}")
                    
                    # FIXED: Count actual output dimensions like backward graph does
                    # Instead of just counting transforms, look at upper dimension indices
                    all_upper_idss = full_descriptor.get_upper_dimension_hidden_idss()
                    final_upper_idss = all_upper_idss[-final_stage_transform_count:]
                    
                    # Count actual output dimensions by looking at the upper dimension indices
                    all_final_upper_dims = []
                    for upper_ids in final_upper_idss:
                        all_final_upper_dims.extend(upper_ids)
                    
                    # The number of unique upper dimension indices is the actual output count
                    actual_output_dims = len(set(all_final_upper_dims)) if all_final_upper_dims else 1
                    
                    print(f"DEBUG: Forward graph final stage upper dimension indices: {final_upper_idss}")
                    print(f"DEBUG: Forward graph all final upper dims: {all_final_upper_dims}")
                    print(f"DEBUG: Forward graph unique final upper dims count: {actual_output_dims}")
                    
                    # Create a wrapper that returns the actual output dimension count
                    class FixedFinalDescriptor:
                        def __init__(self, transforms, actual_dims):
                            self.final_transforms = transforms
                            self.actual_output_dims = actual_dims
                            
                        def get_num_of_dimension(self):
                            return self.actual_output_dims
                            
                        def get_length(self, dim):
                            # FIXED: For multiple transforms, we need to find which transform 
                            # outputs to this dimension and get its output length
                            if not self.final_transforms or dim >= self.actual_output_dims:
                                return 1
                            
                            # Get the upper dimension mappings to find which transform outputs to this dim
                            all_upper_idss = full_descriptor.get_upper_dimension_hidden_idss()
                            final_upper_idss = all_upper_idss[-len(self.final_transforms):]
                            
                            # Find which transform outputs to dimension 'dim'
                            for i, (transform, upper_ids) in enumerate(zip(self.final_transforms, final_upper_idss)):
                                if dim in upper_ids:
                                    # Found the transform that outputs to this dimension
                                    if isinstance(transform, pytensor.tensor_descriptor.UnmergeTransform):
                                        # For UnmergeTransform: each output dimension has its individual length
                                        if hasattr(transform, 'lengths') and transform.lengths:
                                            # Find which output position this dimension corresponds to
                                            dim_position = upper_ids.index(dim)
                                            if dim_position < len(transform.lengths):
                                                return transform.lengths[dim_position]
                                        return 1
                                    elif hasattr(transform, 'lengths'):
                                        # For merge transforms, output length = product of input lengths
                                        result = math.prod(transform.lengths) if transform.lengths else 1
                                        return result
                                    elif hasattr(transform, 'get_upper_lengths'):
                                        lengths = transform.get_upper_lengths()
                                        # For transforms with single output, get that length
                                        return lengths[0] if lengths else 1
                                    else:
                                        return 1
                            
                            # Fallback if not found
                            return 1
                    
                    final_descriptor = FixedFinalDescriptor(final_transforms, actual_output_dims)
                    
                else:
                    final_descriptor = full_descriptor
            except Exception as e:
                print(f"DEBUG: Failed to create corrected final descriptor: {e}")
                final_descriptor = full_descriptor
        else:
            final_descriptor = full_descriptor
            
        final_output_dims = final_descriptor.get_num_of_dimension()
        
        # FIXED: For multi-stage examples, only look at the FINAL stage transforms
        # not all accumulated transforms in the descriptor
        max_transform_output_dim = final_output_dims - 1
        if len(pytensor_descriptors) > 1:
            # Multi-stage: only consider the final stage's expected output dimensions
            final_stage_idx = len(pytensor_descriptors) - 1
            if final_stage_idx < len(descriptors):
                desc_str = descriptors[final_stage_idx].strip()
                parser = TensorTransformParser()
                try:
                    parsed_desc = parser.parse_tensor_descriptor(desc_str)
                    if parsed_desc['type'] == 'transform' and 'output_sequences' in parsed_desc:
                        # Look at the output sequences to determine actual output dimensions
                        output_sequences = parsed_desc['output_sequences']
                        for seq in output_sequences:
                            if seq:  # Non-empty sequence
                                max_transform_output_dim = max(max_transform_output_dim, max(seq))
                        print(f"DEBUG: Multi-stage final output sequences: {output_sequences}, max output dim: {max_transform_output_dim}")
                except Exception as e:
                    print(f"DEBUG: Failed to parse final stage output sequences: {e}")
                    # Fallback: use descriptor's dimension count
                    pass
        else:
            # Single stage: look at transform outputs as before
            if pytensor_descriptors:
                tensor_desc = pytensor_descriptors[-1]
                all_upper_idss = tensor_desc.get_upper_dimension_hidden_idss()
                # Only look at the final stage transforms
                final_stage_idx = len(pytensor_descriptors) - 1
                if final_stage_idx < len(descriptors):
                    desc_str = descriptors[final_stage_idx].strip()
                    if "transform_tensor_descriptor" in desc_str:
                        parser = TensorTransformParser()
                        try:
                            parsed_desc = parser.parse_tensor_descriptor(desc_str)
                            if parsed_desc['type'] == 'transform':
                                final_stage_transform_count = len(parsed_desc['transforms'])
                                # Only consider the last N transforms where N is the final stage count
                                final_upper_idss = all_upper_idss[-final_stage_transform_count:] if final_stage_transform_count > 0 else []
                                for upper_ids in final_upper_idss:
                                    if upper_ids:
                                        max_transform_output_dim = max(max_transform_output_dim, max(upper_ids))
                                print(f"DEBUG: Single-stage final transforms output to max dim: {max_transform_output_dim}")
                        except:
                            # Fallback to all transforms
                            for upper_ids in all_upper_idss:
                                if upper_ids:
                                    max_transform_output_dim = max(max_transform_output_dim, max(upper_ids))
                    else:
                        # Non-transform descriptor, use all
                        for upper_ids in all_upper_idss:
                            if upper_ids:
                                max_transform_output_dim = max(max_transform_output_dim, max(upper_ids))
        
        actual_final_output_dims = max(final_output_dims, max_transform_output_dim + 1)
        
        # Debug the final descriptor issue
        print(f"DEBUG: Creating {actual_final_output_dims} final output nodes (descriptor dims: {final_output_dims}, max transform output: {max_transform_output_dim})")
        print(f"DEBUG: Final descriptor type: {type(final_descriptor)}")
        
        # Check the final dimension lengths
        for k in range(actual_final_output_dims):
            if k < final_output_dims:
                expected_length = final_descriptor.get_length(k)
                print(f"DEBUG: Final descriptor get_length({k}) = {expected_length}")
            else:
                print(f"DEBUG: Final descriptor get_length({k}) = ? (beyond descriptor dims)")
        
        # Create output stage subgraph for visual separation
        with dot.subgraph(name='cluster_output') as output_cluster:
            output_cluster.attr(style='filled', fillcolor='#e8ffe8', 
                              label='Output Stage', 
                              fontsize='14', fontweight='bold')
            
            for k in range(actual_final_output_dims):
                final_node_id = f"forward_output_d{k}"
                
                try:
                    if k < final_output_dims and hasattr(final_descriptor, 'get_length'):
                        dim_length = final_descriptor.get_length(k)
                        dim_label = f"out{k} ({dim_length})"
                    else:
                        dim_label = f"out{k}"
                except:
                    dim_label = f"out{k}"
                
                print(f"DEBUG: Final label for dimension {k}: {dim_label}")
                
                # Create final output node with distinct styling within output cluster
                output_cluster.node(final_node_id, dim_label, fillcolor="#66ff66", style="filled,bold", shape="box")
                
                # Connect from the last stage node to final output if connection exists
                if k in prev_stage_output_nodes:
                    source_node = prev_stage_output_nodes[k]
                    dot.edge(source_node, final_node_id, color="green", style="bold")
                    print(f"DEBUG: Created final output node {final_node_id} connected from {source_node}")
                else:
                    print(f"DEBUG: Created final output node {final_node_id} with no incoming connection (dimension {k} not in prev_stage_output_nodes)")
                    # This can happen when transforms create new dimensions or when there's a mismatch
                    # between the number of outputs from the last stage and the final descriptor's dimensions


    
    # Debug: Check if s4 nodes are in the DOT source
    dot_source = dot.source
    s4_count = dot_source.count('s4_')
    print(f"DEBUG: DOT source contains {s4_count} s4 nodes")
    if s4_count > 0:
        print("DEBUG: s4 nodes found in DOT source")
    else:
        print("DEBUG: No s4 nodes found in DOT source!")

    return dot

def build_combined_formula(transforms: List[Dict[str, Any]], 
                         lower_dims: List[List[int]], 
                         variables: Dict[str, Any]) -> Tuple[sp.Expr, List[sp.Symbol]]:
    """Build a combined formula for all transforms."""
    input_symbols = []
    for dims in lower_dims:
        for dim in dims:
            symbol = sp.Symbol(f'd_{dim}')
            if symbol not in input_symbols:
                input_symbols.append(symbol)
    
    output_exprs = []
    for transform, dims in zip(transforms, lower_dims):
        # Create appropriate transform object
        if transform['type'] == 'merge':
            lengths = [get_transform_length(val, variables) for val in transform['values']]
            transform_obj = MergeTransform(lengths=lengths)
        elif transform['type'] == 'unmerge':
            lengths = [get_transform_length(val, variables) for val in transform['values']]
            transform_obj = UnmergeTransform(lengths=lengths)
        elif transform['type'] == 'pass_through':
            transform_obj = PassThroughTransform(length=1)
        else:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        
        # Get input symbols for this transform
        transform_inputs = [sp.Symbol(f'd_{dim}') for dim in dims]
        
        # Apply transform
        if transform['type'] == 'unmerge':
            result = transform_obj.sympy_calculate_lower(transform_inputs)
        else:
            result = transform_obj.sympy_calculate_upper(transform_inputs)
        
        output_exprs.extend(result)
    
    return sp.Matrix(output_exprs), input_symbols

def build_combined_backward_formula(transforms: List[Dict[str, Any]], 
                                  lower_dims: List[List[int]], 
                                  variables: Dict[str, Any]) -> Tuple[List[sp.Expr], List[sp.Symbol], List[sp.Symbol]]:
    """Build a combined backward formula for all transforms."""
    input_symbols = []
    for dims in lower_dims:
        for dim in dims:
            symbol = sp.Symbol(f'd_{dim}')
            if symbol not in input_symbols:
                input_symbols.append(symbol)
    
    output_symbols = []
    for i in range(len(transforms)):
        output_symbols.append(sp.Symbol(f'y_{i}'))
    
    backward_exprs = []
    for transform, dims, output_symbol in zip(transforms, lower_dims, output_symbols):
        # Create appropriate transform object
        if transform['type'] == 'merge':
            lengths = [get_transform_length(val, variables) for val in transform['values']]
            transform_obj = MergeTransform(lengths=lengths)
        elif transform['type'] == 'unmerge':
            lengths = [get_transform_length(val, variables) for val in transform['values']]
            transform_obj = UnmergeTransform(lengths=lengths)
        elif transform['type'] == 'pass_through':
            transform_obj = PassThroughTransform(length=1)
        else:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        
        # Get input symbols for this transform
        transform_inputs = [sp.Symbol(f'd_{dim}') for dim in dims]
        
        # Apply transform in reverse
        if transform['type'] == 'merge':
            result = transform_obj.sympy_calculate_lower([output_symbol])
        else:
            result = transform_obj.sympy_calculate_upper([output_symbol])
        
        backward_exprs.extend(result)
    
    return backward_exprs, input_symbols, output_symbols

def build_backward_transformation_graph_from_pytensor(descriptors, variables):
    """
    Build a Graphviz DOT graph using pytensor objects in lower → upper direction.
    
    This shows the reverse transformation: starting from physical memory layout (lower)
    and working backwards to logical dimensions (upper) using sympy_lower_to_upper methods.
    """
    import math  # Needed for math.prod() in FixedFinalDescriptor
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", splines="ortho")  # Left to Right like forward graph
    dot.attr('node', shape='box', style='rounded,filled')

    timestamp = int(time.time() * 1000)
    var_comment = f"Backward Variables: {sorted(variables.items())} - Time: {timestamp}"
    dot.attr(comment=var_comment)
    
    vars_display = ", ".join(f"{k}={v}" for k, v in sorted(variables.items())[:5])
    dot.node("backward_title", f"Backward Graph - Variables: {vars_display}", shape="note", style="filled", fillcolor="lightblue")

    if not descriptors:
        return dot

    parser = TensorTransformParser()
    
    realistic_var_names = [
        'b_lds_block_desc',
        'b_lds_block_desc_permuted', 
        'b_lds_block_desc_unmerged',
        'b_lds_block_desc_kn'
    ]
    
    pytensor_descriptors = []
    descriptor_registry = {}
    
    # First, build all descriptors (same as forward)
    for i, desc_str in enumerate(descriptors):
        try:
            parser.descriptor_registry = descriptor_registry
            tensor_desc = parser.create_pytensor_descriptor(desc_str, variables)
            pytensor_descriptors.append(tensor_desc)
            
            print(f"DEBUG BACKWARD: Created pytensor descriptor {i} with {len(tensor_desc.get_transforms())} transforms")
            
            if i < len(realistic_var_names):
                var_name = realistic_var_names[i]
                descriptor_registry[var_name] = tensor_desc
                
        except Exception as e:
            st.error(f"Failed to create pytensor descriptor {i}: {e}")
            dot.node(f"backward_error_{i}", f"Error: {str(e)[:50]}...", fillcolor="#ffcccc")
            continue

    if not pytensor_descriptors:
        st.error("No valid descriptors could be created for backward graph")
        return dot

    # Start from the final descriptor's output dimensions (these become our "input" nodes for backward)
    # BUGFIX: Apply the same fix as forward graph - use only final stage transforms
    full_descriptor = pytensor_descriptors[-1]
    
    # Get the final stage's transform count (same logic as forward graph)
    final_stage_idx = len(pytensor_descriptors) - 1
    if final_stage_idx < len(descriptors):
        desc_str = descriptors[final_stage_idx].strip()
        parser = TensorTransformParser()
        try:
            parsed_desc = parser.parse_tensor_descriptor(desc_str)
            if parsed_desc['type'] == 'transform':
                final_stage_transform_count = len(parsed_desc['transforms'])
                print(f"DEBUG BACKWARD: Final stage has {final_stage_transform_count} transforms")
                
                # Create a final descriptor with only the last stage's transforms
                all_transforms = full_descriptor.get_transforms()
                final_transforms = all_transforms[-final_stage_transform_count:]
                
                # Use the SAME FixedFinalDescriptor logic as forward graph
                all_upper_idss = full_descriptor.get_upper_dimension_hidden_idss()
                final_upper_idss = all_upper_idss[-final_stage_transform_count:]
                
                # Count actual output dimensions by looking at the upper dimension indices
                all_final_upper_dims = []
                for upper_ids in final_upper_idss:
                    all_final_upper_dims.extend(upper_ids)
                
                # The number of unique upper dimension indices is the actual output count
                actual_output_dims = len(set(all_final_upper_dims)) if all_final_upper_dims else 1
                
                print(f"DEBUG BACKWARD: Backward graph final stage upper dimension indices: {final_upper_idss}")
                print(f"DEBUG BACKWARD: Backward graph all final upper dims: {all_final_upper_dims}")
                print(f"DEBUG BACKWARD: Backward graph unique final upper dims count: {actual_output_dims}")
                
                # Create the same FixedFinalDescriptor as forward graph
                class FixedFinalDescriptor:
                    def __init__(self, transforms, actual_dims):
                        self.final_transforms = transforms
                        self.actual_output_dims = actual_dims
                        
                    def get_num_of_dimension(self):
                        return self.actual_output_dims
                        
                    def get_length(self, dim):
                        # FIXED: Use same logic as forward graph
                        if not self.final_transforms or dim >= self.actual_output_dims:
                            return 1
                        
                        # Get the upper dimension mappings to find which transform outputs to this dim
                        all_upper_idss = full_descriptor.get_upper_dimension_hidden_idss()
                        final_upper_idss = all_upper_idss[-len(self.final_transforms):]
                        
                        # Find which transform outputs to dimension 'dim'
                        for i, (transform, upper_ids) in enumerate(zip(self.final_transforms, final_upper_idss)):
                            if dim in upper_ids:
                                # Found the transform that outputs to this dimension
                                if isinstance(transform, pytensor.tensor_descriptor.UnmergeTransform):
                                    # For UnmergeTransform: each output dimension has its individual length
                                    if hasattr(transform, 'lengths') and transform.lengths:
                                        # Find which output position this dimension corresponds to
                                        dim_position = upper_ids.index(dim)
                                        if dim_position < len(transform.lengths):
                                            return transform.lengths[dim_position]
                                    return 1
                                elif hasattr(transform, 'lengths'):
                                    # For merge transforms, output length = product of input lengths
                                    result = math.prod(transform.lengths) if transform.lengths else 1
                                    return result
                                elif hasattr(transform, 'get_upper_lengths'):
                                    lengths = transform.get_upper_lengths()
                                    # For transforms with single output, get that length
                                    return lengths[0] if lengths else 1
                                else:
                                    return 1
                        
                        # Fallback if not found
                        return 1
                
                final_descriptor = FixedFinalDescriptor(final_transforms, actual_output_dims)
                print(f"DEBUG BACKWARD: Created corrected backward final descriptor with {len(final_transforms)} transforms")
                
            else:
                final_descriptor = full_descriptor
        except Exception as e:
            print(f"DEBUG BACKWARD: Failed to create corrected final descriptor: {e}")
            final_descriptor = full_descriptor
    else:
        final_descriptor = full_descriptor
        
    final_output_dims = final_descriptor.get_num_of_dimension()
    
    print(f"DEBUG BACKWARD: Final descriptor has {final_output_dims} output dimensions")
    
    # Create starting nodes (final outputs become backward inputs)
    prev_stage_output_nodes = {}
    current_formulas = {}
    
    # FIXED: Use the same final_descriptor as forward graph for consistency
    actual_output_dims = final_descriptor.get_num_of_dimension()
    
    print(f"DEBUG BACKWARD: Using {actual_output_dims} final descriptor dimensions for starting nodes")
    
    for k in range(actual_output_dims):
        node_id = f"backward_start_d{k}"
        prev_stage_output_nodes[k] = node_id
        
        try:
            dim_length = final_descriptor.get_length(k) if hasattr(final_descriptor, 'get_length') else '?'
            dim_label = f"out{k} ({dim_length})"
        except:
            dim_label = f"out{k}"
        print(f"DEBUG BACKWARD: Creating starting node {node_id} with label {dim_label}")
        dot.node(node_id, dim_label, fillcolor="#ccffcc")  # Light green for final outputs
        current_formulas[node_id] = sp.Symbol(f"out{k}")
    
    # Process descriptors in REVERSE order for backward graph
    actual_stage_num = len(pytensor_descriptors)
    for stage_idx in range(len(pytensor_descriptors) - 1, -1, -1):
        tensor_desc = pytensor_descriptors[stage_idx]
        print(f"DEBUG BACKWARD: Processing stage_idx={stage_idx} in reverse")
        
        stage_output_nodes = {}
        next_formulas = {}
        
        all_transforms = tensor_desc.get_transforms()
        all_lower_idss = tensor_desc.get_lower_dimension_hidden_idss()
        all_upper_idss = tensor_desc.get_upper_dimension_hidden_idss()
        
        print(f"DEBUG BACKWARD: Stage {stage_idx} has {len(all_transforms)} total transforms")
        
        # Determine which transforms to process for this stage (same logic as forward)
        desc_str = descriptors[stage_idx].strip() if stage_idx < len(descriptors) else ""
        is_naive_descriptor = "make_naive_tensor_descriptor" in desc_str
        is_transform_descriptor = "transform_tensor_descriptor" in desc_str
        
        if is_naive_descriptor:
            new_transforms = all_transforms
            new_lower_idss = all_lower_idss
            new_upper_idss = all_upper_idss
        elif is_transform_descriptor:
            parser = TensorTransformParser()
            try:
                parsed_desc = parser.parse_tensor_descriptor(desc_str)
                if parsed_desc['type'] == 'transform':
                    stage_transform_count = len(parsed_desc['transforms'])
                    print(f"DEBUG BACKWARD: Stage {stage_idx} descriptor defines {stage_transform_count} transforms")
                    
                    new_transforms = all_transforms[-stage_transform_count:] if stage_transform_count > 0 else []
                    new_lower_idss = all_lower_idss[-stage_transform_count:] if stage_transform_count > 0 else []
                    new_upper_idss = all_upper_idss[-stage_transform_count:] if stage_transform_count > 0 else []
                else:
                    new_transforms = all_transforms
                    new_lower_idss = all_lower_idss
                    new_upper_idss = all_upper_idss
            except Exception as e:
                print(f"DEBUG BACKWARD: Failed to parse descriptor for stage {stage_idx}, using all transforms: {e}")
                new_transforms = all_transforms
                new_lower_idss = all_lower_idss
                new_upper_idss = all_upper_idss
        else:
            new_transforms = all_transforms
            new_lower_idss = all_lower_idss
            new_upper_idss = all_upper_idss
        
        print(f"DEBUG BACKWARD: Stage {stage_idx} will process {len(new_transforms)} new transforms")
        
        # Skip stages with no transforms
        if not new_transforms:
            print(f"DEBUG BACKWARD: Stage {stage_idx} has no new transforms to process, skipping")
            continue
            
        print(f"DEBUG BACKWARD: Stage {stage_idx} proceeding, using actual stage number {actual_stage_num}")
        
        # Process transforms in REVERSE order for backward direction
        for i in range(len(new_transforms) - 1, -1, -1):
            transform = new_transforms[i]
            if i >= len(new_lower_idss) or i >= len(new_upper_idss):
                continue
                
            # For backward: we swap the interpretation
            # What was "upper" in forward becomes "lower" in backward (input)
            # What was "lower" in forward becomes "upper" in backward (output)
            backward_input_indices = new_upper_idss[i]  # These are our inputs for backward
            backward_output_indices = new_lower_idss[i]  # These are our outputs for backward
            
            print(f"DEBUG BACKWARD: Transform {i} - backward_input_indices: {backward_input_indices}, backward_output_indices: {backward_output_indices}")
            print(f"DEBUG BACKWARD: Transform {i} type: {transform.__class__.__name__}")
            print(f"DEBUG BACKWARD: prev_stage_output_nodes keys: {list(prev_stage_output_nodes.keys())}")
            
            # Get input symbols for this backward transform
            # FIXED: Map hidden dimension IDs to logical indices
            top_dim_ids = tensor_desc.get_top_dimension_hidden_ids()
            hidden_to_logical = {hidden_id: logical_idx for logical_idx, hidden_id in enumerate(top_dim_ids)}
            
            input_symbols = []
            for idx in backward_input_indices:
                # Map hidden dimension ID to logical index
                logical_idx = hidden_to_logical.get(idx, idx)
                
                if logical_idx in prev_stage_output_nodes:
                    prev_node_id = prev_stage_output_nodes[logical_idx]
                    if prev_node_id in current_formulas:
                        input_symbols.append(current_formulas[prev_node_id])
                    else:
                        input_symbols.append(sp.Symbol(f"back_d_{idx}"))
                else:
                    input_symbols.append(sp.Symbol(f"back_d_{idx}"))

            try:
                # Initialize variable for hierarchical merge tracking
                hierarchical_intermediate_nodes = []
                
                # Apply the transform - use correct directional terminology
                # In backward graph, we're going from final outputs back to inputs (lower → upper)
                if isinstance(transform, pytensor.tensor_descriptor.UnmergeTransform):
                    # UnmergeTransform: Backward direction composes (multiple → single)
                    # Use sympy_calculate_lower: takes multiple upper symbols → 1 lower symbol
                    if len(input_symbols) == len(transform.lengths):
                        output_formulas = transform.sympy_calculate_lower(input_symbols)
                    else:
                        # If wrong number of inputs, try to pad or truncate to expected length
                        if len(input_symbols) < len(transform.lengths):
                            # Pad with zeros
                            padded_inputs = input_symbols + [sp.Integer(0)] * (len(transform.lengths) - len(input_symbols))
                            output_formulas = transform.sympy_calculate_lower(padded_inputs)
                        else:
                            # Truncate to expected length
                            truncated_inputs = input_symbols[:len(transform.lengths)]
                            output_formulas = transform.sympy_calculate_lower(truncated_inputs)
                elif isinstance(transform, pytensor.tensor_descriptor.EmbedTransform):
                    # EmbedTransform: upper (multiple) → lower (single) 
                    # In backward direction (lower → upper): single input → multiple outputs
                    if len(input_symbols) == 1:
                        # Single input → multiple outputs: use calculate_lower_index (decomposition)
                        output_formulas = transform.sympy_calculate_lower(input_symbols)
                    else:
                        # Multiple inputs → single output: use calculate_upper_index (composition)
                        output_formulas = transform.sympy_calculate_upper(input_symbols)
                elif isinstance(transform, pytensor.tensor_descriptor.MergeTransform):
                    # MergeTransform: upper (multiple) → lower (single)
                    # In backward direction (lower → upper): 
                    if len(input_symbols) == 1:
                        # Single input → multiple outputs: use calculate_lower_index (decomposition)
                        output_formulas = transform.sympy_calculate_lower(input_symbols)
                    else:
                        # Multiple inputs → single output: use calculate_upper_index (composition)
                        output_formulas = transform.sympy_calculate_upper(input_symbols)
                elif isinstance(transform, pytensor.tensor_descriptor.XorTransform):
                    # XOR: upper (2D) → lower (2D), self-inverse
                    # For backward direction: use calculate_upper_index (lower → upper)
                    output_formulas = transform.sympy_calculate_upper(input_symbols)
                elif isinstance(transform, pytensor.tensor_descriptor.PassThroughTransform):
                    # PassThrough is identity in both directions
                    output_formulas = transform.sympy_calculate_upper(input_symbols)
                else:
                    # Default: use calculate_upper_index for lower → upper direction
                    output_formulas = transform.sympy_calculate_upper(input_symbols)
                
                # Create output nodes for this backward transform
                print(f"DEBUG BACKWARD: Transform {i} has {len(output_formulas)} output formulas for {len(backward_output_indices)} backward output indices")
                
                # Create output nodes for each formula
                for j, output_idx in enumerate(backward_output_indices):
                    if j < len(output_formulas):
                        node_id = f"back_s{actual_stage_num}_t{i}_d{output_idx}"
                        stage_output_nodes[output_idx] = node_id
                        
                        print(f"DEBUG BACKWARD: Creating node {node_id} for stage {stage_idx} transform {i}")
                        
                        formula = output_formulas[j]
                        next_formulas[node_id] = formula
                        
                        # Substitute variables and simplify
                        # Filter variables to only include numeric values that SymPy can handle
                        safe_vars = {k: v for k, v in variables.items() if isinstance(v, (int, float, complex, sp.Basic))}
                        substituted_formula = formula.subs(safe_vars)
                        simplified_formula = sp.simplify(substituted_formula)
                        
                        try:
                            formula_str = str(simplified_formula)
                            
                            # Pretty print simple formulas
                            if (not any(func in formula_str for func in ['floor', 'ceiling', 'sqrt', 'sin', 'cos', 'tan']) 
                                and '/' not in formula_str and '**' not in formula_str):
                                try:
                                    pretty_str = sp.pretty(simplified_formula, use_unicode=False)
                                    if '\n' not in pretty_str:
                                        formula_str = pretty_str
                                except:
                                    pass
                            
                            # Clean up XOR notation
                            import re
                            formula_str = formula_str.replace('XorFunction', 'xor')
                            formula_str = formula_str.replace('⊕', ' xor ')
                            formula_str = re.sub(r'xor\(([^,]+),\s*([^)]+)\)', r'\1 xor \2', formula_str)
                            
                        except Exception:
                            formula_str = f"d{output_idx}"
                        
                        label = f'<<FONT POINT-SIZE="12">{formula_str}</FONT>>'
                        dot.node(node_id, label, fillcolor="#ffccff")  # Light purple for backward nodes
                        print(f"DEBUG BACKWARD: Added DOT node {node_id} with label {formula_str}")
                        
                        # Create edges FROM input nodes TO this output node (normal left-to-right flow)
                        for input_idx in backward_input_indices:
                            if input_idx in prev_stage_output_nodes:
                                transform_name = f"{transform.__class__.__name__.replace('Transform', '')}⁻¹"
                                dot.edge(prev_stage_output_nodes[input_idx], node_id, 
                                       label=transform_name, color="blue")
                
            except Exception as e:
                st.warning(f"Failed to generate backward formula for transform {transform}: {e}")
                # Create error nodes
                for j, output_idx in enumerate(backward_output_indices):
                    node_id = f"back_s{actual_stage_num}_t{i}_d{output_idx}"
                    stage_output_nodes[output_idx] = node_id
                    next_formulas[node_id] = sp.Symbol(f"back_d{output_idx}")
                    dot.node(node_id, f"back_d{output_idx}", fillcolor="#ffcccc")
                    
                    # Create edges for error case
                    for input_idx in backward_input_indices:
                        if input_idx in prev_stage_output_nodes:
                            transform_name = f"{transform.__class__.__name__.replace('Transform', '')}⁻¹"
                            dot.edge(prev_stage_output_nodes[input_idx], node_id, 
                                   label=transform_name, color="red")
        
        # Update for next stage (going backwards)
        if stage_output_nodes:
            prev_stage_output_nodes.update(stage_output_nodes)
            current_formulas.update(next_formulas)
            print(f"DEBUG BACKWARD: Stage {stage_idx} completed, merged stage outputs with existing nodes")
            print(f"DEBUG BACKWARD: Available nodes for next stage: {list(prev_stage_output_nodes.keys())}")
            actual_stage_num -= 1
        else:
            print(f"DEBUG BACKWARD: Stage {stage_idx} completed with no output nodes")

    # Create final output nodes for backward graph (original input dimensions)
    if pytensor_descriptors:
        first_descriptor = pytensor_descriptors[0]
        first_desc_str = descriptors[0].strip() if descriptors else ""
        is_first_naive_packed = "make_naive_tensor_descriptor_packed" in first_desc_str
        
        if is_first_naive_packed:
            # For naive_packed descriptors, create single storage output (consistent with forward input)
            print(f"DEBUG BACKWARD: Creating single storage output for naive_packed descriptor")
            
            if 0 in prev_stage_output_nodes:
                final_node_id = "backward_output_storage"
                
                try:
                    element_space_size = first_descriptor.get_element_space_size()
                    storage_label = f"storage ({element_space_size})"
                except:
                    storage_label = "storage"
                
                # Create final storage output node
                dot.node(final_node_id, storage_label, fillcolor="#ff6666", style="filled,bold", shape="box")
                
                # Connect from the transform node to final storage output
                source_node = prev_stage_output_nodes[0]
                dot.edge(source_node, final_node_id, color="red", style="bold")
                
                print(f"DEBUG BACKWARD: Created final storage output {final_node_id} connected from {source_node}")
        else:
            # For regular descriptors, create logical dimension outputs
            # FIXED: For multi-stage examples, be more conservative about final output dimensions
            first_input_dims = first_descriptor.get_num_of_dimension()
            
            # For multi-stage examples, the final stage should determine the output dimensions
            max_output_dim = first_input_dims - 1
            if len(pytensor_descriptors) > 1:
                # Multi-stage: look at the final stage's input sequences to determine expected inputs
                final_stage_idx = len(pytensor_descriptors) - 1
                if final_stage_idx < len(descriptors):
                    desc_str = descriptors[final_stage_idx].strip()
                    parser = TensorTransformParser()
                    try:
                        parsed_desc = parser.parse_tensor_descriptor(desc_str)
                        if parsed_desc['type'] == 'transform' and 'input_sequences' in parsed_desc:
                            # Look at input sequences to determine what the original inputs should be
                            input_sequences = parsed_desc['input_sequences']
                            for seq in input_sequences:
                                if seq:  # Non-empty sequence
                                    max_output_dim = max(max_output_dim, max(seq))
                            print(f"DEBUG BACKWARD: Multi-stage final input sequences: {input_sequences}, max input dim: {max_output_dim}")
                    except Exception as e:
                        print(f"DEBUG BACKWARD: Failed to parse final stage input sequences: {e}")
                        # Fallback: use first descriptor's dimension count
                        pass
            else:
                # Single stage: look at transform outputs as before, but only for final stage transforms
                if pytensor_descriptors:
                    tensor_desc = pytensor_descriptors[-1]
                    all_lower_idss = tensor_desc.get_lower_dimension_hidden_idss()
                    # Only consider the final stage transforms
                    final_stage_idx = len(pytensor_descriptors) - 1
                    if final_stage_idx < len(descriptors):
                        desc_str = descriptors[final_stage_idx].strip()
                        if "transform_tensor_descriptor" in desc_str:
                            parser = TensorTransformParser()
                            try:
                                parsed_desc = parser.parse_tensor_descriptor(desc_str)
                                if parsed_desc['type'] == 'transform':
                                    final_stage_transform_count = len(parsed_desc['transforms'])
                                    # Only consider the last N transforms where N is the final stage count
                                    final_lower_idss = all_lower_idss[-final_stage_transform_count:] if final_stage_transform_count > 0 else []
                                    for lower_ids in final_lower_idss:
                                        if lower_ids:
                                            max_output_dim = max(max_output_dim, max(lower_ids))
                                    print(f"DEBUG BACKWARD: Single-stage final transforms expect inputs up to dim: {max_output_dim}")
                            except:
                                # Fallback: only add one extra dimension beyond descriptor count
                                max_output_dim = max(max_output_dim, first_input_dims)
                        else:
                            # Non-transform descriptor: use descriptor count
                            pass
            
            actual_output_dims = max_output_dim + 1
            print(f"DEBUG BACKWARD: Creating {actual_output_dims} final output nodes (descriptor dims: {first_input_dims}, max transform output: {max_output_dim})")
            
            for k in range(actual_output_dims):
                final_node_id = f"backward_output_d{k}"
                
                try:
                    dim_length = first_descriptor.get_length(k) if hasattr(first_descriptor, 'get_length') else '?'
                    dim_label = f"in{k} ({dim_length})"
                except:
                    dim_label = f"in{k}"
                
                # Create final output node with distinct styling
                dot.node(final_node_id, dim_label, fillcolor="#ff6666", style="filled,bold", shape="box")
                
                # Connect from the last stage node to final output if connection exists
                if k in prev_stage_output_nodes:
                    source_node = prev_stage_output_nodes[k]
                    dot.edge(source_node, final_node_id, color="red", style="bold")
                    print(f"DEBUG BACKWARD: Created final output node {final_node_id} connected from {source_node}")
                else:
                    print(f"DEBUG BACKWARD: Created final output node {final_node_id} with no incoming connection (dimension {k} not in prev_stage_output_nodes)")
                    # This can happen when transforms create new dimensions or when there's a mismatch
                    # between the number of outputs from the last stage and the final descriptor's dimensions

    return dot

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Tensor Transformation Visualizer")

    initialize_session_state()

    with st.sidebar:
        st.header("Inputs")
        examples = get_transform_examples_with_multi()
        
        def on_example_change():
            """Handle example selection change."""
            st.session_state.selected_example = st.session_state.example_selectbox
            st.session_state.current_code = examples[st.session_state.selected_example]
            st.session_state.parsed_descriptor = None
            
            # Exit format mode when changing examples so user sees the new code
            st.session_state.format_mode = False
            
            # Update variables with defaults for the new example
            default_vars = get_default_variables()
            if st.session_state.selected_example in default_vars:
                st.session_state.variables = default_vars[st.session_state.selected_example].copy()
            else:
                st.session_state.variables = {}

        st.selectbox(
            "Select Example:",
            list(examples.keys()),
            key="example_selectbox",
            on_change=on_example_change
        )
        
        # Initialize format mode state
        if 'format_mode' not in st.session_state:
            st.session_state.format_mode = False
        if 'original_code' not in st.session_state:
            st.session_state.original_code = ""
        
        # Code editor
        if st.session_state.format_mode:
            # Show formatted code preview
            st.markdown("**Formatted Code Preview**")
            formatted_code = format_cpp_code(st.session_state.original_code)
            st.code(formatted_code, language="cpp", line_numbers=True)
        else:
            # Normal code editor
            st.session_state.current_code = st.text_area(
                "Tensor Descriptor Code",
                value=st.session_state.current_code,
                height=300,
                key="code_editor"
            )

        parser = TensorTransformParser()
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            parse_clicked = st.button("Parse and Generate Formulas")
        
        with col2:
            # Format/Edit toggle button
            if not st.session_state.format_mode:
                if st.button("🎨 Format Code", help="Preview formatted C++ code"):
                    st.session_state.original_code = st.session_state.current_code
                    st.session_state.format_mode = True
                    st.rerun()
            else:
                if st.button("✏️ Edit Code", help="Return to code editor"):
                    # Return to original code without applying formatting
                    st.session_state.current_code = st.session_state.original_code
                    st.session_state.format_mode = False
                    st.rerun()
        
        # Handle parse button click
        if parse_clicked:
            try:
                descriptor_texts = extract_descriptors_from_text(st.session_state.current_code)
                parsed_descriptors = [parser.parse_tensor_descriptor(desc) for desc in descriptor_texts]
                st.session_state.parsed_descriptor = parsed_descriptors
                
                # Extract variable information from the code using enhanced parser
                all_code = "\n".join(descriptor_texts)
                variable_info = parser.get_variable_info(all_code)
                st.session_state.variable_info = variable_info
                
                # Get default values for the current example
                default_vars = get_default_variables()
                example_defaults = default_vars.get(st.session_state.selected_example, {})
                
                # Create new variables dict preserving existing values where possible
                new_variables = {}
                for var_name, var_data in variable_info.items():
                    # First try to get existing value from session state
                    if var_name in st.session_state.variables:
                        new_variables[var_name] = st.session_state.variables[var_name]
                    # Then try defaults from example
                    elif var_name in example_defaults:
                        new_variables[var_name] = example_defaults[var_name]
                    # Finally use variable's default value
                    else:
                        new_variables[var_name] = var_data.get('default_value', 1)
                
                st.session_state.variables = new_variables
                
                # Keep variable_names for backward compatibility
                st.session_state.variable_names = sorted(variable_info.keys())
                st.success(f"Parsed {len(parsed_descriptors)} descriptor(s) successfully!")
            except Exception as e:
                st.error(f"Failed to parse descriptor(s): {e}")
                st.session_state.parsed_descriptor = None

        display_variable_controls()

    if st.session_state.parsed_descriptor:
        parsed_descriptors = st.session_state.parsed_descriptor
        user_vars = st.session_state.variables
        parsed_descriptors = [substitute_descriptor(d, user_vars) for d in parsed_descriptors]

        st.markdown("---")
        st.header("Transformation Pipeline Graphs")
        
        if 'last_variables' not in st.session_state:
            st.session_state.last_variables = {}
        
        variables_changed = st.session_state.last_variables != st.session_state.variables
        if variables_changed:
            st.session_state.last_variables = st.session_state.variables.copy()
        
        try:
            descriptors = extract_descriptors_from_text(st.session_state.current_code)
            
            if not descriptors:
                st.error("No descriptors found in the current code")
                return
            
            if variables_changed:
                st.success("Graphs updated with new variable values!")
            
            # Always display both graphs vertically
            st.subheader("Upper → Lower Transformation Graph")
            st.caption("Shows how logical dimensions (upper) transform to physical memory layout (lower)")
            dot_forward = build_transformation_graph_from_pytensor(descriptors, st.session_state.variables)
            st.graphviz_chart(dot_forward)
            
            st.subheader("Lower → Upper Transformation Graph")
            st.caption("Shows how physical memory layout (lower) transforms back to logical dimensions (upper)")
            dot_backward = build_backward_transformation_graph_from_pytensor(descriptors, st.session_state.variables)
            st.graphviz_chart(dot_backward)
                    
        except Exception as e:
            st.error(f"Failed to build graphs: {e}")
            import traceback
            st.text(traceback.format_exc())

        # Add JSON parsing view for debugging
        st.markdown("---")
        st.header("Parsing Debug Information")
        
        if st.checkbox("Show JSON Parsing Results", value=False):
            try:
                descriptors = extract_descriptors_from_text(st.session_state.current_code)
                parser = TensorTransformParser()
                
                st.subheader("Descriptor Extraction Results")
                st.write(f"**Number of descriptors found:** {len(descriptors)}")
                
                for i, desc_str in enumerate(descriptors):
                    with st.expander(f"Descriptor {i} - Raw Text ({len(desc_str)} chars)"):
                        st.code(desc_str, language="cpp")
                
                st.subheader("Parsing Results")
                
                parsing_results = {
                    'total_descriptors': len(descriptors),
                    'variables': st.session_state.variables,
                    'descriptors': []
                }
                
                descriptor_registry = {}
                variable_names = ['desc_0', 'desc_1', 'desc_2', 'desc_3', 'desc_4']  # Generic names
                
                for i, desc_str in enumerate(descriptors):
                    try:
                        parser.descriptor_registry = descriptor_registry
                        
                        # Parse the descriptor
                        parsed_dict = parser.parse_tensor_descriptor(desc_str)
                        
                        # Create pytensor descriptor
                        parser.clear_default_warnings()  # Clear any previous warnings
                        tensor_desc = parser.create_pytensor_descriptor(desc_str, st.session_state.variables)
                        
                        # Check for default value warnings
                        warnings = parser.get_default_warnings()
                        if warnings:
                            for warning in warnings:
                                st.warning(f"⚠️ {warning}")
                            st.info("💡 Tip: Set values for these variables in the 'Template Variables' section above for more control.")
                        
                        if i < len(variable_names):
                            descriptor_registry[variable_names[i]] = tensor_desc
                        
                        # Convert to JSON-serializable format
                        json_desc = {
                            'index': i,
                            'type': parsed_dict['type'],
                            'dimensions': tensor_desc.get_num_of_dimension(),
                            'transform_count': len(tensor_desc.get_transforms()),
                            'top_dimension_ids': tensor_desc.get_top_dimension_hidden_ids(),
                        }
                        
                        # Add dimension lengths
                        try:
                            lengths = [tensor_desc.get_length(d) for d in range(tensor_desc.get_num_of_dimension())]
                            json_desc['dimension_lengths'] = lengths
                        except:
                            json_desc['dimension_lengths'] = 'Could not compute'
                        
                        # Add parsing details with proper JSON serialization
                        def make_json_serializable(obj):
                            """Convert any object to a JSON-serializable format."""
                            import sympy as sp
                            if isinstance(obj, sp.Basic):
                                return str(obj)
                            elif isinstance(obj, dict):
                                return {k: make_json_serializable(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [make_json_serializable(item) for item in obj]
                            elif hasattr(obj, '__dict__'):
                                return str(obj)  # Convert complex objects to strings
                            else:
                                return obj
                        
                        if parsed_dict['type'] == 'naive':
                            json_desc['lengths'] = make_json_serializable(parsed_dict['lengths'])
                            json_desc['strides'] = make_json_serializable(parsed_dict['strides'])
                            json_desc['vector_length'] = make_json_serializable(parsed_dict['vector_length'])
                            json_desc['offset'] = make_json_serializable(parsed_dict['offset'])
                        elif parsed_dict['type'] == 'transform':
                            json_desc['input_descriptor'] = str(parsed_dict['input_descriptor'])
                            json_desc['transforms'] = []
                            for j, transform in enumerate(parsed_dict['transforms']):
                                json_transform = {
                                    'index': j,
                                    'type': transform['type']
                                }
                                if 'values' in transform:
                                    json_transform['values'] = make_json_serializable(transform['values'])
                                if 'value' in transform:
                                    json_transform['value'] = make_json_serializable(transform['value'])
                                json_desc['transforms'].append(json_transform)
                            
                            json_desc['lower_dimensions'] = make_json_serializable(parsed_dict['lower_dimensions'])
                            json_desc['upper_dimensions'] = make_json_serializable(parsed_dict['upper_dimensions'])
                        
                        parsing_results['descriptors'].append(json_desc)
                        
                    except Exception as e:
                        error_desc = {
                            'index': i,
                            'error': str(e),
                            'descriptor_preview': desc_str[:200] + "..." if len(desc_str) > 200 else desc_str
                        }
                        parsing_results['descriptors'].append(error_desc)
                
                # Display as formatted JSON
                st.json(parsing_results)
                
                # Add download button for JSON
                import json
                json_str = json.dumps(parsing_results, indent=2)
                st.download_button(
                    label="Download Parsing Results as JSON",
                    data=json_str,
                    file_name="tensor_parsing_results.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Failed to generate parsing debug info: {e}")
                import traceback
                st.text(traceback.format_exc())

# Function aliases for proper upper/lower terminology
build_upper_to_lower_transformation_graph = build_transformation_graph_from_pytensor
build_lower_to_upper_transformation_graph = build_backward_transformation_graph_from_pytensor

if __name__ == "__main__":
    main() 