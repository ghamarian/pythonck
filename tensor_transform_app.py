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
    
    # Get default values for the current example
    default_vars = get_default_variables()
    example_defaults = default_vars.get(st.session_state.selected_example, {})
    
    for var in st.session_state.variable_names:
        # First try to get current value from session state
        current_value = st.session_state.variables.get(var)
        
        # If not in session state, try to get from defaults
        if current_value is None:
            current_value = example_defaults.get(var, 1)
        
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
    
    # For the first descriptor, if it's a naive descriptor, use its actual dimension count
    # This fixes the issue where naive descriptors were getting too many input dimensions
    first_desc = pytensor_descriptors[0]
    first_desc_str = descriptors[0].strip()
    
    if "make_naive_tensor_descriptor(" in first_desc_str and "make_naive_tensor_descriptor_packed(" not in first_desc_str:
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
        if stage_idx == 0 and is_first_naive_packed and len(new_transforms) > 0:
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
                    # Don't increment actual_stage_num for skipped stages
                    continue
                
                # Process remaining transforms  
                new_transforms = new_transforms[1:]
                new_lower_idss = new_lower_idss[1:]
                new_upper_idss = new_upper_idss[1:]
                print(f"DEBUG: Stage {stage_idx} after skipping UnmergeTransform will process {len(new_transforms)} transforms")
        
        # Skip stages with no new transforms to process
        if not new_transforms:
            print(f"DEBUG: Stage {stage_idx} has no new transforms to process, skipping")
            # Don't increment actual_stage_num for skipped stages
            continue
            
        print(f"DEBUG: Stage {stage_idx} proceeding with {len(new_transforms)} new transforms, using actual stage number {actual_stage_num}")
        
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
                # Apply the transform
                if isinstance(transform, pytensor.tensor_descriptor.UnmergeTransform):
                    if len(input_symbols) != 1:
                        merged_input = input_symbols[0] if len(input_symbols) == 1 else sum(input_symbols)
                        output_formulas = transform.sympy_backward([merged_input])
                    else:
                        output_formulas = transform.sympy_backward(input_symbols)
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
                        # Use sympy_forward: multiple inputs -> 1 output
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
                            output_formulas = transform.sympy_forward(coordinate_symbols)
                            print(f"DEBUG: EmbedTransform.sympy_forward({coordinate_symbols}) -> {output_formulas}")
                        except Exception as embed_error:
                            print(f"DEBUG: EmbedTransform.sympy_forward failed: {embed_error}")
                            # Fallback: create simple sum
                            output_formulas = [sum(coordinate_symbols)]
                    else:
                        # Decomposition case: linear address -> coordinates
                        # Use sympy_backward: 1 input -> multiple outputs
                        if input_symbols:
                            single_input = input_symbols[0]
                        else:
                            single_input = sp.Symbol("d0")
                        
                        try:
                            output_formulas = transform.sympy_backward([single_input])
                            print(f"DEBUG: EmbedTransform.sympy_backward([{single_input}]) -> {len(output_formulas)} outputs")
                        except Exception as embed_error:
                            print(f"DEBUG: EmbedTransform.sympy_backward failed: {embed_error}")
                            # Fallback: create coordinate symbols
                            output_formulas = [sp.Symbol(f"coord{i}") for i in range(len(upper_indices))]
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
                
                # For final stage, only create nodes for dimensions that actually exist in the final output
                if is_final_stage:
                    # Filter upper_indices to only include those within the final output dimension count
                    filtered_upper_indices = [idx for idx in upper_indices if idx < final_output_dims]
                    print(f"DEBUG: Final stage filtering upper_indices {upper_indices} to {filtered_upper_indices}")
                else:
                    filtered_upper_indices = upper_indices
                
                # Create output nodes for each formula
                for j, output_idx in enumerate(filtered_upper_indices):
                    if j < len(output_formulas):
                        # Use the actual stage number for node naming
                        node_id = f"s{actual_stage_num}_t{i}_d{output_idx}"
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
                        # Special handling for EmbedTransform in naive descriptors
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
                
            except Exception as e:
                st.warning(f"Failed to generate formula for transform {transform}: {e}")
                # Create error nodes with actual stage number
                for j, output_idx in enumerate(upper_indices):
                    node_id = f"s{actual_stage_num}_t{i}_d{output_idx}"
                    stage_output_nodes[output_idx] = node_id
                    next_formulas[node_id] = sp.Symbol(f"d{output_idx}")
                    dot.node(node_id, f"d{output_idx}", fillcolor="#ffcccc")
                    
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
            """Handle example selection change."""
            st.session_state.selected_example = st.session_state.example_selectbox
            st.session_state.current_code = examples[st.session_state.selected_example]
            st.session_state.parsed_descriptor = None
            
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
                
                # Extract variable names from the code
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
                
                # Get default values for the current example
                default_vars = get_default_variables()
                example_defaults = default_vars.get(st.session_state.selected_example, {})
                
                # Create new variables dict preserving existing values where possible
                new_variables = {}
                for var in st.session_state.variable_names:
                    # First try to get existing value from session state
                    if var in st.session_state.variables:
                        new_variables[var] = st.session_state.variables[var]
                    # Then try defaults
                    elif var in example_defaults:
                        new_variables[var] = example_defaults[var]
                    # Finally fall back to 1
                    else:
                        new_variables[var] = 1
                
                st.session_state.variables = new_variables
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
                        tensor_desc = parser.create_pytensor_descriptor(desc_str, st.session_state.variables)
                        
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
                        
                        # Add parsing details
                        if parsed_dict['type'] == 'naive':
                            json_desc['lengths'] = [str(l) for l in parsed_dict['lengths']]
                            json_desc['strides'] = [str(s) for s in parsed_dict['strides']]
                            json_desc['vector_length'] = str(parsed_dict['vector_length'])
                            json_desc['offset'] = str(parsed_dict['offset'])
                        elif parsed_dict['type'] == 'transform':
                            json_desc['input_descriptor'] = parsed_dict['input_descriptor']
                            json_desc['transforms'] = []
                            for j, transform in enumerate(parsed_dict['transforms']):
                                json_transform = {
                                    'index': j,
                                    'type': transform['type']
                                }
                                if 'values' in transform:
                                    json_transform['values'] = [str(v) for v in transform['values']]
                                if 'value' in transform:
                                    json_transform['value'] = str(transform['value'])
                                json_desc['transforms'].append(json_transform)
                            
                            json_desc['lower_dimensions'] = parsed_dict['lower_dimensions']
                            json_desc['upper_dimensions'] = parsed_dict['upper_dimensions']
                        
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

if __name__ == "__main__":
    main() 