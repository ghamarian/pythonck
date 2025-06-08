"""
Streamlit application for visualizing tensor descriptor transformations.
"""

import streamlit as st
import sympy as sp
import re
import time
from tensor_transform_parser import TensorTransformParser
from pytensor.tensor_descriptor import (
    Transform, PassThroughTransform, MergeTransform, UnmergeTransform,
    EmbedTransform, OffsetTransform, PadTransform, ReplicateTransform
)
from tensor_transform_examples import get_transform_examples, get_default_variables
from extract_descriptors import extract_descriptors_from_text
from typing import Dict, Any, List, Tuple
import graphviz
import pytensor.tensor_descriptor

def get_transform_examples_with_multi():
    examples = get_transform_examples()
    return examples

def initialize_session_state():
    """Initialize all required session state variables."""
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = list(get_transform_examples_with_multi().keys())[0]
    
    if 'variables' not in st.session_state:
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
        return value.subs(variables) if isinstance(value, sp.Basic) else value
        
    elif transform_type in ('merge', 'xor'):
        total_length = sp.Integer(1)
        for sub_transform in transform.get('values', []):
            total_length *= get_transform_length(sub_transform, variables)
        return total_length
        
    return sp.Integer(1)

def display_variable_controls():
    """Display number inputs for adjusting template variables."""
    st.subheader("Template Variables")
    
    updated_variables = {}
    
    if not hasattr(st.session_state, 'variable_names') or not st.session_state.variable_names:
        st.info("No template variables detected in code. Parse your code to find variables.")
        return
    
    for var in st.session_state.variable_names:
        current_value = st.session_state.variables.get(var, 1)
        
        if '::' in var:
            namespace, var_name = var.split('::')
            display_name = f"{namespace}::{var_name}"
        else:
            display_name = var
        
        value = st.number_input(
            display_name,
            min_value=1,
            value=current_value,
            key=f"var_{var}"
        )
        
        updated_variables[var] = value
    
    variables_changed = False
    if hasattr(st.session_state, 'variables'):
        for var, value in updated_variables.items():
            if var in st.session_state.variables and st.session_state.variables[var] != value:
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
        return descriptor.subs(user_vars)
    else:
        return descriptor

def build_transformation_graph_from_pytensor(descriptors, variables):
    """
    Build a Graphviz DOT graph using pytensor objects and their sympy methods.
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", splines="ortho")
    dot.attr('node', shape='box', style='rounded,filled')

    timestamp = int(time.time() * 1000)
    var_comment = f"Variables: {sorted(variables.items())} - Time: {timestamp}"
    dot.attr(comment=var_comment)
    
    vars_display = ", ".join(f"{k}={v}" for k, v in sorted(variables.items())[:5])
    dot.node("title", f"Variables: {vars_display}", shape="note", style="filled", fillcolor="lightyellow")

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
    
    # Analyze all descriptors to find the maximum input dimension needed
    max_input_dim = 0
    for tensor_desc in pytensor_descriptors:
        all_lower_idss = tensor_desc.get_lower_dimension_hidden_idss()
        for lower_ids in all_lower_idss:
            if lower_ids:
                max_input_dim = max(max_input_dim, max(lower_ids) + 1)
    
    print(f"DEBUG: max_input_dim calculated as {max_input_dim}")
    
    # If no transforms found, use the first descriptor's dimension count
    if max_input_dim == 0:
        first_desc = pytensor_descriptors[0]
        max_input_dim = first_desc.get_num_of_dimension()
        print(f"DEBUG: max_input_dim fallback to first descriptor dimension count: {max_input_dim}")
    
    # For naive descriptors, we need to ensure we have enough input dimensions
    # by checking the actual transforms in the pytensor object
    if max_input_dim < 6:  # Assume at least 6 dimensions are often needed
        first_desc = pytensor_descriptors[0]
        all_upper_idss = first_desc.get_upper_dimension_hidden_idss()
        all_lower_idss = first_desc.get_lower_dimension_hidden_idss()
        
        # Find the maximum dimension referenced in any transform
        max_dim_needed = 0
        for lower_ids in all_lower_idss:
            if lower_ids:
                max_dim_needed = max(max_dim_needed, max(lower_ids) + 1)
        for upper_ids in all_upper_idss:
            if upper_ids:
                max_dim_needed = max(max_dim_needed, max(upper_ids) + 1)
        
        if max_dim_needed > max_input_dim:
            max_input_dim = max_dim_needed
            print(f"DEBUG: max_input_dim updated to {max_input_dim} based on pytensor transforms")
    
    # Special handling for the first descriptor if it's make_naive_tensor_descriptor_packed
    first_desc_str = descriptors[0].strip()
    is_first_naive_packed = "make_naive_tensor_descriptor_packed" in first_desc_str
    
    print(f"DEBUG: is_first_naive_packed = {is_first_naive_packed}")
    
    # Create initial input nodes
    prev_stage_output_nodes = {}
    current_formulas = {}
    
    if is_first_naive_packed:
        print("DEBUG: Using naive_packed path")
        # For naive_tensor_descriptor_packed, create logical input dimensions
        first_desc = pytensor_descriptors[0]
        num_logical_dims = first_desc.get_num_of_dimension()
        
        print(f"DEBUG: num_logical_dims = {num_logical_dims}")
        
        for k in range(num_logical_dims):
            node_id = f"input_d{k}"
            # The logical dimensions should map directly to their indices
            prev_stage_output_nodes[k] = node_id
            
            print(f"DEBUG: Created input {node_id} mapped to index {k}")
            
            try:
                dim_length = first_desc.get_length(k) if hasattr(first_desc, 'get_length') else '?'
                dim_label = f"d{k} ({dim_length})"
            except:
                dim_label = f"d{k}"
            dot.node(node_id, dim_label, fillcolor="#ffcccc")
            current_formulas[node_id] = sp.Symbol(f"d{k}")
    else:
        print(f"DEBUG: Using non-packed path, creating {max_input_dim} input dimensions")
        for k in range(max_input_dim):
            node_id = f"input_d{k}"
            prev_stage_output_nodes[k] = node_id
            
            print(f"DEBUG: Created input {node_id} mapped to index {k}")
            
            # Get dimension length from first descriptor if possible
            try:
                first_desc = pytensor_descriptors[0]
                if k < first_desc.get_num_of_dimension():
                    dim_length = first_desc.get_length(k) if hasattr(first_desc, 'get_length') else '?'
                    dim_label = f"d{k} ({dim_length})"
                else:
                    dim_label = f"d{k}"
            except:
                dim_label = f"d{k}"
            dot.node(node_id, dim_label, fillcolor="#ffcccc")
            current_formulas[node_id] = sp.Symbol(f"d{k}")
    
    # Process each descriptor as a separate stage
    for stage_idx, tensor_desc in enumerate(pytensor_descriptors):
        print(f"DEBUG: Processing stage_idx={stage_idx}")
        stage_output_nodes = {}
        next_formulas = {}
        
        all_transforms = tensor_desc.get_transforms()
        all_lower_idss = tensor_desc.get_lower_dimension_hidden_idss()
        all_upper_idss = tensor_desc.get_upper_dimension_hidden_idss()
        
        print(f"DEBUG: Stage {stage_idx} has {len(all_transforms)} total transforms")
        
        # Determine which transforms to process for this stage
        if len(pytensor_descriptors) > 1 and stage_idx > 0:
            # For multi-descriptor pipelines after the first, get only new transforms
            prev_desc = pytensor_descriptors[stage_idx - 1]
            prev_transform_count = len(prev_desc.get_transforms())
            
            print(f"DEBUG: Previous descriptor had {prev_transform_count} transforms")
            
            # Get only the new transforms for this stage
            if prev_transform_count < len(all_transforms):
                new_transforms = all_transforms[prev_transform_count:]
                new_lower_idss = all_lower_idss[prev_transform_count:]
                new_upper_idss = all_upper_idss[prev_transform_count:]
                print(f"DEBUG: Stage {stage_idx} will process {len(new_transforms)} new transforms")
            else:
                # No new transforms in this stage, skip
                print(f"DEBUG: Stage {stage_idx} has no new transforms, skipping")
                continue
        else:
            # First descriptor - process all transforms
            new_transforms = all_transforms
            new_lower_idss = all_lower_idss
            new_upper_idss = all_upper_idss
            
            print(f"DEBUG: Stage {stage_idx} (first) will process {len(new_transforms)} transforms")
            
            # Special handling for naive_tensor_descriptor_packed first transform
            if is_first_naive_packed and len(new_transforms) > 0:
                if isinstance(new_transforms[0], pytensor.tensor_descriptor.UnmergeTransform):
                    # Skip the first transform (UnmergeTransform) but set up connections
                    first_upper_indices = new_upper_idss[0]
                    for j, output_idx in enumerate(first_upper_indices):
                        if j < num_logical_dims:
                            input_node_id = f"input_d{j}"
                            stage_output_nodes[j] = input_node_id
                            next_formulas[input_node_id] = current_formulas[input_node_id]
                    
                    # Skip remaining processing for this stage if only UnmergeTransform
                    if len(new_transforms) == 1:
                        print(f"DEBUG: Stage {stage_idx} only has UnmergeTransform, setting up connections and continuing")
                        prev_stage_output_nodes = stage_output_nodes
                        current_formulas = next_formulas
                        continue
                    
                    # Process remaining transforms  
                    new_transforms = new_transforms[1:]
                    new_lower_idss = new_lower_idss[1:]
                    new_upper_idss = new_upper_idss[1:]
                    print(f"DEBUG: Stage {stage_idx} after skipping UnmergeTransform will process {len(new_transforms)} transforms")
        
        # Skip stages with no transforms to process
        if not new_transforms:
            print(f"DEBUG: Stage {stage_idx} has no transforms to process, skipping")
            continue
            
        print(f"DEBUG: Stage {stage_idx} proceeding with {len(new_transforms)} transforms")
        
        # Process each transform in this stage
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
                # Apply the transform
                if isinstance(transform, pytensor.tensor_descriptor.UnmergeTransform):
                    if len(input_symbols) != 1:
                        merged_input = input_symbols[0] if len(input_symbols) == 1 else sum(input_symbols)
                        output_formulas = transform.sympy_backward([merged_input])
                    else:
                        output_formulas = transform.sympy_backward(input_symbols)
                elif isinstance(transform, pytensor.tensor_descriptor.EmbedTransform):
                    # EmbedTransform takes inputs from lower_indices and produces outputs for upper_indices
                    # In the XT example: lower_indices=[0], upper_indices=[1,2,3,4,5,6]
                    # This means: 1 input -> 6 outputs
                    
                    if len(lower_indices) == 1 and len(upper_indices) > 1:
                        # Case: 1 input -> multiple outputs 
                        # Use calculate_upper_index direction: lower -> upper (1D -> 6D)
                        # This corresponds to sympy_backward for EmbedTransform
                        single_input = input_symbols[0] if input_symbols else sp.Symbol("d0")
                        try:
                            output_formulas = transform.sympy_backward([single_input])
                            print(f"DEBUG: EmbedTransform.sympy_backward({single_input}) -> {len(output_formulas)} outputs")
                        except Exception as embed_error:
                            print(f"DEBUG: EmbedTransform.sympy_backward failed: {embed_error}")
                            # Fallback: create individual coordinate symbols
                            output_formulas = [sp.Symbol(f"coord{i}") for i in range(len(upper_indices))]
                    else:
                        # Case: multiple inputs -> 1 output 
                        # Use calculate_lower_index direction: upper -> lower (6D -> 1D)
                        # This corresponds to sympy_forward for EmbedTransform
                        expected_input_dims = len(transform.lengths)
                        
                        # Create the required number of input symbols
                        embed_input_symbols = []
                        for dim_idx in range(expected_input_dims):
                            if dim_idx < len(input_symbols):
                                embed_input_symbols.append(input_symbols[dim_idx])
                            else:
                                embed_input_symbols.append(sp.Symbol(f"d{dim_idx}"))
                        
                        try:
                            output_formulas = transform.sympy_forward(embed_input_symbols)
                            print(f"DEBUG: EmbedTransform.sympy_forward({embed_input_symbols}) -> {len(output_formulas)} outputs")
                        except Exception as embed_error:
                            print(f"DEBUG: EmbedTransform.sympy_forward failed: {embed_error}")
                            # Fallback: create simple formula
                            formula = sum(input_symbols[:expected_input_dims]) if input_symbols else sp.Symbol("combined")
                            output_formulas = [formula]
                elif isinstance(transform, pytensor.tensor_descriptor.MergeTransform):
                    # MergeTransform has INCONSISTENT sympy method naming!
                    # For MergeTransform: sympy_forward does calculate_upper_index (lower->upper)
                    # We need to handle this inconsistency
                    if len(lower_indices) > 1 and len(upper_indices) == 1:
                        # Case: multiple inputs -> 1 output (6D -> 1D)
                        # This should use calculate_upper_index: lower -> upper
                        # For MergeTransform, this is sympy_forward (inconsistent naming!)
                        try:
                            output_formulas = transform.sympy_forward(input_symbols)
                            print(f"DEBUG: MergeTransform.sympy_forward({input_symbols}) -> {len(output_formulas)} outputs")
                        except Exception as merge_error:
                            print(f"DEBUG: MergeTransform.sympy_forward failed: {merge_error}")
                            formula = sum(input_symbols) if input_symbols else sp.Symbol("merged")
                            output_formulas = [formula]
                    else:
                        # Case: 1 input -> multiple outputs (1D -> 6D)
                        # This should use calculate_lower_index: upper -> lower
                        # For MergeTransform, this is sympy_backward (inconsistent naming!)
                        single_input = input_symbols[0] if input_symbols else sp.Symbol("merged")
                        try:
                            output_formulas = transform.sympy_backward([single_input])
                            print(f"DEBUG: MergeTransform.sympy_backward({single_input}) -> {len(output_formulas)} outputs")
                        except Exception as merge_error:
                            print(f"DEBUG: MergeTransform.sympy_backward failed: {merge_error}")
                            output_formulas = [sp.Symbol(f"comp{i}") for i in range(len(upper_indices))]
                elif isinstance(transform, pytensor.tensor_descriptor.UnmergeTransform):
                    # UnmergeTransform should follow same pattern as MergeTransform
                    # (they likely have the same inconsistent naming)
                    if len(input_symbols) != 1:
                        merged_input = input_symbols[0] if len(input_symbols) == 1 else sum(input_symbols)
                        output_formulas = transform.sympy_backward([merged_input])
                    else:
                        output_formulas = transform.sympy_backward(input_symbols)
                else:
                    output_formulas = transform.sympy_forward(input_symbols)
                
                # Create output nodes for this transform
                print(f"DEBUG: Transform {i} has {len(output_formulas)} output formulas for {len(upper_indices)} upper indices")
                
                # Create output nodes for each formula
                for j, output_idx in enumerate(upper_indices):
                    if j < len(output_formulas):
                        # Use sequential stage numbering starting from 1
                        actual_stage = stage_idx + 1
                        node_id = f"s{actual_stage}_t{i}_d{output_idx}"
                        stage_output_nodes[output_idx] = node_id
                        
                        print(f"DEBUG: Creating node {node_id} for stage {stage_idx} transform {i}")
                        
                        formula = output_formulas[j]
                        next_formulas[node_id] = formula
                        
                        # Substitute variables and simplify
                        substituted_formula = formula.subs(variables)
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
                        dot.node(node_id, label, fillcolor="#c0ffc0")
                        print(f"DEBUG: Added DOT node {node_id} with label {formula_str}")
                        
                        # Create edges from input nodes to this output node
                        for input_idx in lower_indices:
                            if input_idx in prev_stage_output_nodes:
                                transform_name = transform.__class__.__name__.replace('Transform', '')
                                dot.edge(prev_stage_output_nodes[input_idx], node_id, 
                                       label=transform_name)
                
            except Exception as e:
                st.warning(f"Failed to generate formula for transform {transform}: {e}")
                # Create error nodes
                for j, output_idx in enumerate(upper_indices):
                    actual_stage = stage_idx + 1
                    node_id = f"s{actual_stage}_t{i}_d{output_idx}"
                    stage_output_nodes[output_idx] = node_id
                    next_formulas[node_id] = sp.Symbol(f"d{output_idx}")
                    dot.node(node_id, f"d{output_idx}", fillcolor="#ffcccc")
                    
                    for input_idx in lower_indices:
                        if input_idx in prev_stage_output_nodes:
                            transform_name = transform.__class__.__name__.replace('Transform', '')
                            dot.edge(prev_stage_output_nodes[input_idx], node_id, 
                                   label=transform_name)
        
        # Update for next stage
        if stage_output_nodes:
            prev_stage_output_nodes = stage_output_nodes
            current_formulas = next_formulas
            print(f"DEBUG: Stage {stage_idx} completed, updated prev_stage_output_nodes")
        else:
            print(f"DEBUG: Stage {stage_idx} completed with no output nodes")

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
            result = transform_obj.sympy_backward(transform_inputs)
        else:
            result = transform_obj.sympy_forward(transform_inputs)
        
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
            result = transform_obj.sympy_backward([output_symbol])
        else:
            result = transform_obj.sympy_forward([output_symbol])
        
        backward_exprs.extend(result)
    
    return backward_exprs, input_symbols, output_symbols

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Tensor Transformation Visualizer")

    initialize_session_state()

    with st.sidebar:
        st.header("Inputs")
        examples = get_transform_examples_with_multi()
        
        def on_example_change():
            st.session_state.selected_example = st.session_state.example_selectbox
            st.session_state.current_code = examples[st.session_state.selected_example]
            st.session_state.parsed_descriptor = None
            st.session_state.variables = {}

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
                descriptor_texts = extract_descriptors_from_text(st.session_state.current_code)
                parsed_descriptors = [parser.parse_tensor_descriptor(desc) for desc in descriptor_texts]
                st.session_state.parsed_descriptor = parsed_descriptors
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

        display_variable_controls()

    if st.session_state.parsed_descriptor:
        parsed_descriptors = st.session_state.parsed_descriptor
        user_vars = st.session_state.variables
        parsed_descriptors = [substitute_descriptor(d, user_vars) for d in parsed_descriptors]

        st.markdown("---")
        st.header("Transformation Pipeline Graph")
        
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
            
            dot = build_transformation_graph_from_pytensor(descriptors, st.session_state.variables)
            
            if variables_changed:
                st.success("Graph updated with new variable values!")
            
            st.graphviz_chart(dot)
        except Exception as e:
            st.error(f"Failed to build graph: {e}")
            import traceback
            st.text(traceback.format_exc())

if __name__ == "__main__":
    main() 