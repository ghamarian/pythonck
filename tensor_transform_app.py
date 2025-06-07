"""
Streamlit application for visualizing tensor descriptor transformations.
"""

import streamlit as st
import sympy as sp
import re
from tensor_transform_parser import (
    TensorTransformParser, 
    merge_transform_to_sympy, 
    unmerge_transform_to_sympy
)
from tensor_transform_examples import get_transform_examples, get_default_variables
from typing import Dict, Any, List

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

def get_transform_length(transform: Dict[str, Any]) -> sp.Expr:
    """Recursively calculate the total length of a transform's output space."""
    if transform['type'] == 'pass_through':
        return transform['value']
    elif transform['type'] == 'merge':
        total_length = sp.Integer(1)
        for sub_transform in transform.get('values', []):
            total_length *= get_transform_length(sub_transform)
        return total_length
    return sp.Integer(1)

def build_sympy_forward_expr(transform: Dict[str, Any], symbols_iterator) -> sp.Expr:
    """Recursively build the SymPy expression for a forward transformation."""
    if transform['type'] == 'pass_through':
        return next(symbols_iterator)
    elif transform['type'] == 'merge':
        sub_exprs = [build_sympy_forward_expr(val, symbols_iterator) for val in transform.get('values', [])]
        lengths = [get_transform_length(val) for val in transform.get('values', [])]
        return merge_transform_to_sympy(sub_exprs, lengths)
    return sp.Integer(0)

def build_sympy_backward_exprs(y: sp.Symbol, transform: Dict[str, Any]) -> List[sp.Expr]:
    """Recursively build the SymPy expressions for a backward transformation."""
    if transform['type'] == 'pass_through':
        return [y]
    elif transform['type'] == 'merge':
        lengths = [get_transform_length(val) for val in transform.get('values', [])]
        unmerged_ys = unmerge_transform_to_sympy(y, lengths)
        
        backward_exprs = []
        for i, sub_transform in enumerate(transform.get('values', [])):
            sub_backward = build_sympy_backward_exprs(unmerged_ys[i], sub_transform)
            backward_exprs.extend(sub_backward)
        return backward_exprs
    return []

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Tensor Transformation Visualizer")

    initialize_session_state()

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("Inputs")
        
        examples = get_transform_examples()
        
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
        # Find all potential variables (e.g., number<X>{}, number<Y/Z>{})
        variable_names = set(re.findall(r'number<([a-zA-Z0-9_ / * + -]+)>{}', st.session_state.current_code))
        # Extract individual variable names from expressions
        cleaned_vars = set()
        for var in variable_names:
            cleaned_vars.update(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', var))

        if cleaned_vars:
            st.subheader("Template Variables")
            
            # Use default variables for the selected example
            default_vars = get_default_variables().get(st.session_state.selected_example, {})
            
            for var in sorted(list(cleaned_vars)):
                # If we have a value in session state, use it, otherwise use default.
                current_val = st.session_state.variables.get(var, default_vars.get(var, 1))
                st.session_state.variables[var] = st.number_input(
                    f"Value for {var}",
                    value=current_val,
                    key=f"var_{var}"
                )

        if st.button("Parse and Generate Formulas"):
            try:
                parser.set_variables(st.session_state.variables)
                st.session_state.parsed_descriptor = parser.parse_tensor_descriptor(st.session_state.current_code)
                st.success("Descriptor parsed successfully!")
            except Exception as e:
                st.error(f"Failed to parse descriptor: {e}")
                st.session_state.parsed_descriptor = None

    # --- Main Content Area to Display Formulas ---
    if st.session_state.parsed_descriptor:
        st.header("Transformation Formulas")
        
        parsed = st.session_state.parsed_descriptor
        transforms = parsed.get('transforms', [])
        lower_dims = parsed.get('lower_dimensions', [])

        if not transforms:
            st.info("No transformations found in the descriptor.")
            return

        for i, transform in enumerate(transforms):
            st.subheader(f"Analysis of Transform #{i+1}")
            
            with st.expander("Parsed Structure", expanded=False):
                st.json(transform)
            
            with st.container():
                st.markdown(f"**Type:** `{transform['type']}`")
                
                # Get the input dimension symbols for this transform
                try:
                    input_indices = lower_dims[i] if i < len(lower_dims) else []
                    input_symbols = [sp.Symbol(f"d_{k}") for k in input_indices]
                    symbols_iter = iter(input_symbols)

                    # --- Forward Transformation ---
                    st.markdown("**Forward Formula:**")
                    y = sp.Symbol(f"y_{i}")
                    forward_expr = build_sympy_forward_expr(transform, symbols_iter)
                    st.latex(f"{sp.latex(y)} = {sp.latex(forward_expr)}")

                    # --- Backward Transformation ---
                    st.markdown("**Backward Formulas:**")
                    backward_exprs = build_sympy_backward_exprs(y, transform)
                    
                    if len(input_symbols) != len(backward_exprs):
                         st.warning("Could not fully determine backward formula for all inputs.")

                    for j, expr in enumerate(backward_exprs):
                        if j < len(input_symbols):
                            st.latex(f"{sp.latex(input_symbols[j])} = {sp.latex(expr)}")

                except Exception as e:
                    st.error(f"Failed to generate formulas for transform #{i+1}: {e}")

if __name__ == "__main__":
    main() 