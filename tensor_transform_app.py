"""
Streamlit application for visualizing tensor descriptor transformations.
"""

import streamlit as st
import sympy as sp
import re
import time
from tensor_transform_parser import (
    TensorTransformParser, 
    merge_transform_to_sympy, 
    unmerge_transform_to_sympy
)
from tensor_transform_examples import get_transform_examples, get_default_variables
from extract_descriptors import extract_descriptors_from_text
from tensor_transform_analyzer import TensorTransformAnalyzer
from typing import Dict, Any, List, Tuple
import graphviz

# --- Add a multi-descriptor example ---
MULTI_DESCRIPTOR_EXAMPLE = '''
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

# Add to examples (with initial dims as a comment for now)
def get_transform_examples_with_multi():
    examples = get_transform_examples()
    examples["Realistic Multi-Descriptor Example"] = MULTI_DESCRIPTOR_EXAMPLE
    return examples

def initialize_session_state():
    """Initialize all required session state variables."""
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = list(get_transform_examples().keys())[0]
    
    if 'variables' not in st.session_state:
        st.session_state.variables = {}

    if 'current_code' not in st.session_state:
        st.session_state.current_code = get_transform_examples()[st.session_state.selected_example]
        
    if 'parsed_descriptor' not in st.session_state:
        st.session_state.parsed_descriptor = None

# --- New Recursive Formula Generation ---

def get_transform_length(transform: Dict[str, Any], variables: Dict[str, Any]) -> sp.Expr:
    """Recursively calculate the total length of a transform's output space, substituting variables."""
    transform_type = transform.get('type')
    
    if transform_type == 'pass_through':
        # Substitute the value immediately.
        value = transform.get('value', sp.Integer(1))
        return value.subs(variables) if isinstance(value, sp.Basic) else value
        
    elif transform_type in ('merge', 'xor'):
        total_length = sp.Integer(1)
        for sub_transform in transform.get('values', []):
            # Pass variables down in the recursive call.
            total_length *= get_transform_length(sub_transform, variables)
        return total_length
        
    return sp.Integer(1)

def build_sympy_forward_expr(transform: Dict[str, Any], symbols_iterator, variables: Dict[str, int]) -> sp.Expr:
    """Recursively build the SymPy expression for a forward transformation."""
    if transform['type'] == 'pass_through':
        return next(symbols_iterator)
    elif transform['type'] == 'merge':
        sub_exprs = [build_sympy_forward_expr(val, symbols_iterator, variables) for val in transform.get('values', [])]
        # Pass variables to get_transform_length.
        lengths = [get_transform_length(val, variables) for val in transform.get('values', [])]
        return merge_transform_to_sympy(sub_exprs, lengths)
    return sp.Integer(0)

def build_sympy_backward_exprs(y: sp.Symbol, transform: Dict[str, Any], variables: Dict[str, int]) -> List[sp.Expr]:
    """Recursively build the SymPy expressions for a backward transformation."""
    if transform['type'] == 'pass_through':
        return [y]
    elif transform['type'] == 'merge':
        # Pass variables to get_transform_length.
        lengths = [get_transform_length(val, variables) for val in transform.get('values', [])]
        unmerged_ys = unmerge_transform_to_sympy(y, lengths)
        
        backward_exprs = []
        for i, sub_transform in enumerate(transform.get('values', [])):
            sub_backward = build_sympy_backward_exprs(unmerged_ys[i], sub_transform, variables)
            backward_exprs.extend(sub_backward)
        return backward_exprs
    return []

def build_combined_formula(transforms: List[Dict[str, Any]], lower_dims: List[List[int]], variables: Dict[str, int]) -> Tuple[sp.Expr, List[sp.Symbol]]:
    """Build the final combined formula for all transforms."""
    # Create input symbols for all dimensions
    all_input_dims = set()
    for dims in lower_dims:
        all_input_dims.update(dims)
    input_symbols = [sp.Symbol(f"d_{k}") for k in sorted(all_input_dims)]
    
    # Build forward expressions for each transform
    forward_exprs = []
    for i, transform in enumerate(transforms):
        # Get the specific input dimensions for this transform
        transform_dims = lower_dims[i] if i < len(lower_dims) else []
        # Create symbols for just this transform's input dimensions
        transform_symbols = [sp.Symbol(f"d_{k}") for k in transform_dims]
        symbols_iter = iter(transform_symbols)
        forward_exprs.append(build_sympy_forward_expr(transform, symbols_iter, variables))
    
    # Combine all forward expressions into a tuple
    return sp.Tuple(*forward_exprs), input_symbols

def build_combined_backward_formula(transforms: List[Dict[str, Any]], lower_dims: List[List[int]], variables: Dict[str, int]) -> Tuple[List[sp.Expr], List[sp.Symbol], List[sp.Symbol]]:
    """Build the final combined backward formula for all transforms."""
    # Create output symbols for all transforms
    output_symbols = [sp.Symbol(f"y_{i}") for i in range(len(transforms))]
    
    # Create input symbols for all dimensions in the correct order
    all_input_dims = set()
    for dims in lower_dims:
        all_input_dims.update(dims)
    input_symbols = [sp.Symbol(f"d_{k}") for k in sorted(all_input_dims)]
    
    # Build backward expressions for each transform and properly order them
    # We need to collect all backward expressions and then order them by dimension index
    all_backward_exprs = {}  # dimension_index -> expression
    
    for i, transform in enumerate(transforms):
        transform_dims = lower_dims[i] if i < len(lower_dims) else []
        sub_exprs = build_sympy_backward_exprs(output_symbols[i], transform, variables)
        
        # Map the sub_expressions back to their dimension indices
        for j, dim_idx in enumerate(transform_dims):
            if j < len(sub_exprs):
                all_backward_exprs[dim_idx] = sub_exprs[j]
    
    # Order the backward expressions by dimension index
    backward_exprs = []
    for dim_idx in sorted(all_input_dims):
        if dim_idx in all_backward_exprs:
            backward_exprs.append(all_backward_exprs[dim_idx])
        else:
            # This shouldn't happen if everything is consistent
            backward_exprs.append(sp.Symbol(f"d_{dim_idx}"))
    
    return backward_exprs, input_symbols, output_symbols

def display_variable_controls():
    """Display number inputs for adjusting template variables."""
    st.subheader("Template Variables")
    
    # Initialize a new dictionary to store the updated variable values
    updated_variables = {}
    
    if not hasattr(st.session_state, 'variable_names') or not st.session_state.variable_names:
        st.info("No template variables detected in code. Parse your code to find variables.")
        return
    
    for var in st.session_state.variable_names:
        # Get current value if exists, otherwise default to 1
        current_value = st.session_state.variables.get(var, 1)
        
        # Format display name for namespace-prefixed variables
        if '::' in var:
            namespace, var_name = var.split('::')
            display_name = f"{namespace}::{var_name}"
        else:
            display_name = var
        
        # Create a number input for this variable with a unique key
        value = st.number_input(
            display_name,
            min_value=1,
            value=current_value,
            key=f"var_{var}"  # Use unique key to prevent conflicts
        )
        
        updated_variables[var] = value
    
    # Check if any variable has changed
    variables_changed = False
    if hasattr(st.session_state, 'variables'):
        for var, value in updated_variables.items():
            if var in st.session_state.variables and st.session_state.variables[var] != value:
                variables_changed = True
                break
    
    # Store the updated variables in the session state
    st.session_state.variables = updated_variables

def substitute_descriptor(descriptor, user_vars):
    """Recursively substitute user variable values into all SymPy expressions in the descriptor."""
    if isinstance(descriptor, dict):
        return {k: substitute_descriptor(v, user_vars) for k, v in descriptor.items()}
    elif isinstance(descriptor, list):
        return [substitute_descriptor(item, user_vars) for item in descriptor]
    elif isinstance(descriptor, sp.Basic):
        return descriptor.subs(user_vars)
    else:
        return descriptor

def build_formula_for_transform(transform, input_symbols_iter, variables):
    """
    Recursively builds a SymPy formula for a potentially nested transform object.
    It consumes symbols from the provided iterator as it encounters leaf nodes.
    """
    transform_type = transform.get('type')

    if transform_type == 'pass_through':
        # A pass_through is a leaf node in the formula tree; consumes one symbol.
        return next(input_symbols_iter)
    
    elif transform_type == 'merge':
        # Recursively build formulas for all sub-transforms (the children).
        sub_expressions = [
            build_formula_for_transform(sub_transform, input_symbols_iter, variables)
            for sub_transform in transform.get('values', [])
        ]
        
        # Get the lengths of the direct children of this merge transform.
        # Pass variables down and remove the now-redundant .subs() call.
        lengths = [get_transform_length(val, variables) for val in transform.get('values', [])]
        
        # Merge the generated sub-expressions into a single formula.
        return merge_transform_to_sympy(sub_expressions, lengths)

    elif transform_type == 'xor':
        # Handle XOR nested inside another transform.
        sub_expressions = [
            build_formula_for_transform(sub_transform, input_symbols_iter, variables)
            for sub_transform in transform.get('values', [])
        ]
        # Create a symbolic representation of the XOR operation.
        return sp.Function('XOR')(*sub_expressions)

    # Fallback for unknown or unsupported transform types within this recursive context.
    return sp.Integer(0)

def build_transformation_graph(parsed_descriptors, variables):
    """
    Build a single, connected Graphviz DOT graph that correctly visualizes the
    entire transformation pipeline, including symbolic formulas on the nodes.
    This version correctly handles pipelines that start with a transform or a naive descriptor.
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", splines="ortho")
    dot.attr('node', shape='box', style='rounded,filled')

    # Add variable values and timestamp to force graph refresh
    timestamp = int(time.time() * 1000)  # millisecond precision
    var_comment = f"Variables: {sorted(variables.items())} - Time: {timestamp}"
    dot.attr(comment=var_comment)
    
    # Add a title node with current variable values to make changes visible
    vars_display = ", ".join(f"{k}={v}" for k, v in sorted(variables.items())[:5])
    dot.node("title", f"Variables: {vars_display}", shape="note", style="filled", fillcolor="lightyellow")

    if not parsed_descriptors:
        return dot

    # 1. Determine the initial state and where transforms begin
    first_desc = parsed_descriptors[0]
    num_initial_dims = 0
    transform_start_index = 0

    if first_desc.get('type') == 'naive':
        dimensions = first_desc.get('dimensions', [])
        num_initial_dims = len(dimensions)
        transform_start_index = 1
    elif first_desc.get('type') == 'transform':
        lower_dims = first_desc.get('lower_dimensions', [])
        if not lower_dims: return dot
        num_initial_dims = max(idx for indices in lower_dims for idx in indices) + 1
        transform_start_index = 0
    else:
        return dot

    # 2. Create nodes for the initial state
    prev_stage_output_nodes = {}
    for k in range(num_initial_dims):
        node_id = f"s0_d{k}"
        prev_stage_output_nodes[k] = node_id
        dot.node(node_id, f"d{k}", fillcolor="#e0e0e0")

    if transform_start_index >= len(parsed_descriptors):
        return dot

    # 3. Loop through all transformations
    for i in range(transform_start_index, len(parsed_descriptors)):
        transform_desc = parsed_descriptors[i]
        
        if transform_desc.get('type') != 'transform':
            continue

        transforms = transform_desc.get('transforms', [])
        lower_dims_indices = transform_desc.get('lower_dimensions', [])
        upper_dims_indices = transform_desc.get('upper_dimensions', [])
        
        if not all([transforms, lower_dims_indices, upper_dims_indices]):
            continue

        # --- Formula Generation for this Stage (Now handles nesting and xor) ---
        num_inputs = len(prev_stage_output_nodes)
        input_symbols = [sp.Symbol(f"d_{k}") for k in range(num_inputs)]
        output_formulas = {}

        for j, transform in enumerate(transforms):
            transform_type = transform.get('type', 'unknown')
            input_indices = lower_dims_indices[j]
            output_indices = upper_dims_indices[j]
            
            # Defensive check to prevent IndexError
            if any(k >= num_inputs for k in input_indices):
                st.error(
                    f"Internal Error: Invalid dimension index in descriptor {i}. "
                    f"Attempted to access index >= {num_inputs}. "
                    f"Available input dimensions: {num_inputs}."
                )
                return dot

            # Create an iterator that provides the symbols for this specific transform.
            # This is crucial for the recursive builder to consume symbols correctly.
            current_input_symbols_iter = iter([input_symbols[k] for k in input_indices])

            if transform_type == 'pass_through':
                # Pass-through is 1-to-1 and consumes one symbol.
                if input_indices and output_indices:
                    output_formulas[output_indices[0]] = next(current_input_symbols_iter)
            elif transform_type == 'merge':
                # Use the recursive builder for potentially nested merges.
                formula = build_formula_for_transform(transform, current_input_symbols_iter, variables)
                if output_indices:
                    output_formulas[output_indices[0]] = formula
            elif transform_type == 'unmerge':
                # Unmerge is 1-to-many and consumes one symbol.
                if input_indices:
                    input_symbol = next(current_input_symbols_iter)
                    # Pass variables to get_transform_length.
                    lengths = [get_transform_length(val, variables) for val in transform.get('values', [])]
                    formulas = unmerge_transform_to_sympy(input_symbol, lengths)
                    for k_out_idx, form in zip(output_indices, formulas):
                        output_formulas[k_out_idx] = form
            elif transform_type == 'xor':
                # XOR produces one result. That one result is mapped to all output dimensions.
                symbols = [input_symbols[k] for k in input_indices]
                formula = sp.Function('XOR')(*symbols)
                for k_out_idx in output_indices:
                    output_formulas[k_out_idx] = formula

        # --- Node and Edge Creation for this Stage ---
        stage_num = i - transform_start_index + 1
        
        # Robustly calculate the number of output dimensions from upper_dimensions.
        if not upper_dims_indices or not any(upper_dims_indices):
            num_output_dims = 0
        else:
            # Get all unique indices from all upper_dimensions
            all_output_indices = set()
            for indices in upper_dims_indices:
                all_output_indices.update(indices)
            num_output_dims = max(all_output_indices) + 1 if all_output_indices else 0
            
        # Debug info
        st.write(f"Debug: Stage {i}, Transform start: {transform_start_index}")
        st.write(f"Debug: Upper dims indices: {upper_dims_indices}")
        st.write(f"Debug: Calculated output dims: {num_output_dims}")
        st.write(f"Debug: Previous stage had {len(prev_stage_output_nodes)} output nodes")
            
        current_stage_output_nodes = {}

        for k in range(num_output_dims):
            node_id = f"s{stage_num}_d{k}"
            current_stage_output_nodes[k] = node_id
            
            label = f"d{k}"
            formula = output_formulas.get(k)
            if formula is not None:
                # Substitute variables before simplifying and rendering
                substituted_formula = formula.subs(variables)
                formula_str = str(sp.simplify(substituted_formula))
                # Use HTML-like labels for formatting (no variable values)
                label = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" ALIGN="LEFT"><TR><TD>{label}</TD></TR><HR/><TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">{formula_str}</FONT></TD></TR></TABLE>>'
            
            dot.node(node_id, label, fillcolor="#c0ffc0")

        # Create edges from previous stage's nodes to this stage's nodes
        for j, transform in enumerate(transforms):
            transform_type = transform.get('type', 'unknown')
            
            input_indices = lower_dims_indices[j]
            input_node_ids = [prev_stage_output_nodes[k] for k in input_indices]
            
            output_indices = upper_dims_indices[j]
            output_node_ids = [current_stage_output_nodes[k] for k in output_indices]
            
            for in_node in input_node_ids:
                for out_node in output_node_ids:
                    dot.edge(in_node, out_node, label=transform_type)

        # Only update if we actually created output nodes
        if current_stage_output_nodes:
            prev_stage_output_nodes = current_stage_output_nodes
        
    return dot

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Tensor Transformation Visualizer")

    initialize_session_state()

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("Inputs")
        examples = get_transform_examples_with_multi()
        
        def on_example_change():
            st.session_state.selected_example = st.session_state.example_selectbox
            st.session_state.current_code = examples[st.session_state.selected_example]
            st.session_state.parsed_descriptor = None # Reset parsing result
            st.session_state.variables = {} # Reset variables

        st.selectbox(
            "Select Example:",
            list(examples.keys()),
            key="example_selectbox",
            on_change=on_example_change
        )
        
        st.session_state.current_code = st.text_area(
            "Tensor Descriptor Code",
            value=st.session_state.current_code,
            height=300,
            key="code_editor"
        )

        parser = TensorTransformParser()
        
        if st.button("Parse and Generate Formulas"):
            try:
                # --- Multi-descriptor support ---
                descriptor_texts = extract_descriptors_from_text(st.session_state.current_code)
                parsed_descriptors = [parser.parse_tensor_descriptor(desc) for desc in descriptor_texts]
                st.session_state.parsed_descriptor = parsed_descriptors
                # Extract all variable names from all parsed descriptors
                all_code = "\n".join(descriptor_texts)
                variable_names = set()
                template_vars = re.findall(r'number<([a-zA-Z0-9_ / * + -]+)>{}', all_code)
                for var in template_vars:
                    variable_names.update(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', var))
                raw_vars = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', all_code)
                cpp_keywords = {'make_tuple', 'sequence', 'transform_tensor_descriptor', 'make_pass_through_transform', 
                               'make_merge_transform', 'number', 'true', 'false', 'nullptr'}
                variable_names.update(var for var in raw_vars if var not in cpp_keywords)
                st.session_state.variable_names = sorted(variable_names)
                st.success(f"Parsed {len(parsed_descriptors)} descriptor(s) successfully!")
            except Exception as e:
                st.error(f"Failed to parse descriptor(s): {e}")
                st.session_state.parsed_descriptor = None

        # Show variables section using the separate function
        display_variable_controls()

    # --- Main Content Area to Display Formulas and Pipeline ---
    if st.session_state.parsed_descriptor:
        parsed_descriptors = st.session_state.parsed_descriptor
        user_vars = st.session_state.variables
        # Substitute user variable values before analysis
        parsed_descriptors = [substitute_descriptor(d, user_vars) for d in parsed_descriptors]
        
        # The analyzer is no longer needed for the graph, but might be for other things.
        # We will bypass it for visualization.
        # max_dims = max(len(desc.get('lower_dimensions', [])) for desc in parsed_descriptors if 'lower_dimensions' in desc)
        # generic_dims = [f"d{i}" for i in range(max_dims)]
        # analyzer = TensorTransformAnalyzer(parsed_descriptors, generic_dims)
        # stages = analyzer.analyze()

        st.markdown("---")
        st.header("Final Combined Transformation Formulas (last descriptor)")
        # Show formulas for the last descriptor in the chain
        last_parsed = parsed_descriptors[-1]
        transforms = last_parsed.get('transforms', [])
        lower_dims = last_parsed.get('lower_dimensions', [])
        try:
            combined_expr, input_symbols = build_combined_formula(transforms, lower_dims, st.session_state.variables)
            st.markdown("**Forward Transformation (Input → Output):**")
            st.latex(f"\\text{{Input}} = {sp.latex(sp.Tuple(*input_symbols))}")
            st.latex(f"\\text{{Output}} = {sp.latex(combined_expr.subs(st.session_state.variables))}")
            backward_exprs, input_symbols, output_symbols = build_combined_backward_formula(transforms, lower_dims, st.session_state.variables)
            st.markdown("**Backward Transformation (Output → Input):**")
            st.latex(f"\\text{{Output}} = {sp.latex(sp.Tuple(*output_symbols))}")
            st.latex(f"\\text{{Input}} = {sp.latex(sp.Tuple(*backward_exprs).subs(st.session_state.variables))}")
        except Exception as e:
            st.error(f"Failed to generate combined formulas: {e}")

        st.markdown("---")
        st.header("Transformation Pipeline Graph")
        
        # Force a rerun by checking if variables have changed
        if 'last_variables' not in st.session_state:
            st.session_state.last_variables = {}
        
        variables_changed = st.session_state.last_variables != st.session_state.variables
        if variables_changed:
            st.session_state.last_variables = st.session_state.variables.copy()
        
        try:
            # Pass the original parsed descriptors and the current variables.
            # The build function will handle all substitutions internally.
            dot = build_transformation_graph(st.session_state.parsed_descriptor, st.session_state.variables)
            
            # Show a status message when variables change
            if variables_changed:
                st.success("Graph updated with new variable values!")
            
            # The graph now includes variable values and timestamp in its content
            st.graphviz_chart(dot)
        except Exception as e:
            st.error(f"Failed to build graph: {e}")

        # ... after the graph ...
        st.markdown("---")
        st.header("Debug: Raw Stage Data")
        # Use the substituted descriptors for debug display
        substituted_descriptors = [substitute_descriptor(d, st.session_state.variables) for d in st.session_state.parsed_descriptor]
        for i, stage in enumerate(substituted_descriptors):
            st.markdown(f"**Stage {i}:**")
            st.json(stage)

if __name__ == "__main__":
    main() 