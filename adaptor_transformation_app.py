"""
Streamlit app to understand adaptor transformation functions in tile distribution.

This app explains step-by-step:
1. _make_adaptor_encoding_for_tile_distribution
2. _construct_static_tensor_adaptor_from_encoding  
3. make_tensor_descriptor_from_adaptor

Run with: streamlit run adaptor_transformation_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import json
from dataclasses import dataclass, asdict

# Configure page - must be first streamlit command
st.set_page_config(
    page_title="Adaptor Transformation Deep Dive",
    page_icon="🔄",
    layout="wide"
)

# Import our modules
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import (
    _make_adaptor_encoding_for_tile_distribution,
    _construct_static_tensor_adaptor_from_encoding,
    make_tensor_descriptor_from_adaptor,
    TileDistribution
)
from pytensor.tensor_adaptor import TensorAdaptor
from pytensor.tensor_descriptor import TensorDescriptor
from examples import get_examples, get_default_variables
from parser import TileDistributionParser

def create_encoding_from_parsed(parsed_result: Dict[str, Any], variables: Dict[str, Any]) -> TileDistributionEncoding:
    """Create encoding from parsed result."""
    try:
        def resolve_variable(value):
            if isinstance(value, str) and value in variables:
                return variables[value]
            return value

        # Use the correct keys from the parser
        rs_lengths = [resolve_variable(v) for v in parsed_result.get('RsLengths', [])]
        hs_lengthss = [[resolve_variable(v) for v in hs] for hs in parsed_result.get('HsLengthss', [])]
        ps_to_rhss_major = parsed_result.get('Ps2RHssMajor', [])
        ps_to_rhss_minor = parsed_result.get('Ps2RHssMinor', [])
        ys_to_rhs_major = parsed_result.get('Ys2RHsMajor', [])
        ys_to_rhs_minor = parsed_result.get('Ys2RHsMinor', [])

        return TileDistributionEncoding(
            rs_lengths=rs_lengths,
            hs_lengthss=hs_lengthss,
            ps_to_rhss_major=ps_to_rhss_major,
            ps_to_rhss_minor=ps_to_rhss_minor,
            ys_to_rhs_major=ys_to_rhs_major,
            ys_to_rhs_minor=ys_to_rhs_minor
        )
    except Exception as e:
        return None

def get_example_encodings():
    """Get example encodings from the examples module using proper parsing."""
    examples = get_examples()
    encodings = {}
    errors = []
    
    for name, code in examples.items():
        try:
            # Parse the code to get the encoding
            parser = TileDistributionParser()
            parsed = parser.parse_tile_distribution_encoding(code)
            if parsed:
                # Get default variables for this example
                variables = get_default_variables(name)
                encoding = create_encoding_from_parsed(parsed, variables)
                if encoding:
                    encodings[name] = encoding
        except Exception as e:
            errors.append(f"Failed to parse {name}: {e}")
            continue
    
    return encodings, errors

def show_step_by_step_analysis(encoding: TileDistributionEncoding):
    """Show detailed step-by-step analysis similar to spans_app.py"""
    
    st.header("🔍 Step-by-Step Adaptor Transformation Analysis")
    
    # Step 1: Input Analysis
    with st.expander("📋 Step 1: Input Encoding Analysis", expanded=True):
        st.subheader("Input TileDistributionEncoding")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Basic Properties:**")
            st.json({
                "ndim_x": encoding.ndim_x,
                "ndim_y": len(encoding.ys_to_rhs_major),
                "ndim_p": len(encoding.ps_to_rhss_major),
                "ndim_r": len(encoding.rs_lengths)
            })
            
        with col2:
            st.write("**R Dimensions (Replication):**")
            for i, length in enumerate(encoding.rs_lengths):
                st.write(f"R{i}: length = {length}")
        
        st.write("**H Dimensions (Hidden/Intermediate):**")
        for i, lengths in enumerate(encoding.hs_lengthss):
            st.write(f"H{i}: {lengths}")
            
        st.write("**P → RH Mappings:**")
        for i, (major_list, minor_list) in enumerate(zip(encoding.ps_to_rhss_major, encoding.ps_to_rhss_minor)):
            for j, (major, minor) in enumerate(zip(major_list, minor_list)):
                st.write(f"P{i}[{j}] → RH{major}[{minor}]")
            
        st.write("**Y → RH Mappings:**")
        for i, (major, minor) in enumerate(zip(encoding.ys_to_rhs_major, encoding.ys_to_rhs_minor)):
            st.write(f"Y{i} → RH{major}[{minor}]")

    # Step 2: Adaptor Encoding Creation
    with st.expander("🏗️ Step 2: Adaptor Encoding Creation", expanded=True):
        st.subheader("_make_adaptor_encoding_for_tile_distribution")
        
        st.write("**Purpose:** Convert tile distribution encoding into transformation blueprint")
        
        try:
            adaptor_encoding = _make_adaptor_encoding_for_tile_distribution(encoding)
            
            st.write("**Generated Adaptor Encoding:**")
            st.write("✅ Successfully created adaptor encoding object")
            
            # Try to show some info about the encoding without converting to string
            st.write("**Adaptor Encoding Information:**")
            st.write(f"- Type: {type(adaptor_encoding).__name__}")
            
            # Try to extract useful information safely
            try:
                if hasattr(adaptor_encoding, '__len__'):
                    st.write(f"- Number of transformations: {len(adaptor_encoding)}")
                
                # Show the first few transformations if it's iterable
                if hasattr(adaptor_encoding, '__iter__'):
                    st.write("**First few transformations:**")
                    for i, transform in enumerate(adaptor_encoding):
                        if i >= 3:  # Show only first 3
                            st.write("... (more transformations)")
                            break
                        
                        # Try to extract transform info safely
                        transform_info = f"Transform {i+1}: "
                        if hasattr(transform, 'name'):
                            transform_info += f"{transform.name}"
                        elif isinstance(transform, dict) and 'name' in transform:
                            transform_info += f"{transform['name']}"
                        else:
                            transform_info += f"{type(transform).__name__}"
                        
                        st.write(transform_info)
                        
            except Exception as e:
                st.write(f"Could not extract detailed info: {e}")
                    
            # Explain the logic
            st.write("**Logic Explanation:**")
            st.write("""
            The adaptor encoding creation process typically involves:
            1. **Replicate**: Creates R dimensions for thread replication
            2. **Unmerge**: Splits combined dimensions into constituent parts  
            3. **Merge**: Combines P and Y dimensions into final X dimensions
            4. **Freeze**: Fixes certain dimensions at specific values
            """)
            
            # Show RH space structure
            st.write("**RH Space Structure Analysis:**")
            total_rh_dims = len(encoding.rs_lengths) + len(encoding.hs_lengthss)
            st.write(f"- Total RH dimensions: {total_rh_dims}")
            st.write(f"- R dimensions: {len(encoding.rs_lengths)} (replication)")
            st.write(f"- H dimensions: {len(encoding.hs_lengthss)} (hidden/intermediate)")
            
            # Show what the encoding represents
            st.write("**What this encoding represents:**")
            st.write(f"""
            - Input space: P dimensions ({len(encoding.ps_to_rhss_major)}) + Y dimensions ({len(encoding.ys_to_rhs_major)})
            - Output space: X dimensions ({encoding.ndim_x})
            - Transformation: Maps (P,Y) coordinates → X coordinates
            """)
            
        except Exception as e:
            st.error(f"Error creating adaptor encoding: {e}")
            st.write("**This is expected** - the adaptor encoding functions are designed for C++ compilation context.")
            st.write("**Conceptual Explanation:**")
            st.write("""
            Even though we can't execute this in Python, here's what happens conceptually:
            
            1. **Analyze RH Space**: The function examines the R and H dimension structure
            2. **Create Replication**: Adds transforms for R dimension replication
            3. **Create Unmerge**: Adds transforms to split H dimensions into sub-components
            4. **Create Merge**: Adds transforms to combine P/Y into final coordinates
            5. **Generate String**: Produces a transformation string like "replicate<1>, unmerge<4,4>, merge<...>"
            """)

    # Step 3: Tensor Adaptor Construction
    with st.expander("🔧 Step 3: Tensor Adaptor Construction", expanded=True):
        st.subheader("_construct_static_tensor_adaptor_from_encoding")
        
        st.write("**Purpose:** Turn blueprint into functional coordinate transformer")
        
        st.write("**Conceptual Process:**")
        st.write("""
        1. **Parse Encoding**: The adaptor encoding string is parsed into individual transforms
        2. **Build Chain**: Each transform is instantiated and chained together
        3. **Validate**: Input/output dimensions are validated for compatibility
        4. **Optimize**: The transformation chain may be optimized for performance
        """)
        
        # Show what the adaptor would do conceptually
        st.write("**Coordinate Transformation Concept:**")
        st.write(f"""
        **Input dimensions:** P({len(encoding.ps_to_rhss_major)}) + Y({len(encoding.ys_to_rhs_major)}) = {len(encoding.ps_to_rhss_major) + len(encoding.ys_to_rhs_major)} total
        
        **Output dimensions:** X({encoding.ndim_x})
        
        **Example transformation:**
        - Input: [p0=0, p1=5, y0=0, y1=2] (thread 5 in warp 0, local position (0,2))
        - Processing: Apply replicate, unmerge, merge transforms
        - Output: [x0=computed_value, x1=computed_value] (global tensor coordinates)
        """)
        
        # Try to create it but handle errors gracefully
        try:
            adaptor_encoding = _make_adaptor_encoding_for_tile_distribution(encoding)
            # Extract the ps_ys_to_xs_adaptor_encoding (first element) from the result
            ps_ys_to_xs_adaptor_encoding = adaptor_encoding[0]
            tensor_adaptor = _construct_static_tensor_adaptor_from_encoding(ps_ys_to_xs_adaptor_encoding)
            
            st.success("✅ Successfully constructed tensor adaptor!")
            st.write("**Adaptor Properties:**")
            st.json({
                "top_dimensions": tensor_adaptor.get_num_of_top_dimension(),
                "bottom_dimensions": tensor_adaptor.get_num_of_bottom_dimension(),
                "type": str(type(tensor_adaptor).__name__)
            })
            
        except Exception as e:
            st.warning(f"Construction failed (expected in Python context): {e}")
            st.write("**In C++ context, this would:**")
            st.write(f"- Create adaptor with {len(encoding.ps_to_rhss_major) + len(encoding.ys_to_rhs_major)} input dimensions")
            st.write(f"- Create adaptor with {encoding.ndim_x} output dimensions")
            st.write("- Enable coordinate transformation: (P,Y) → X")

    # Step 4: Tensor Descriptor Creation
    with st.expander("📐 Step 4: Tensor Descriptor Creation", expanded=True):
        st.subheader("make_tensor_descriptor_from_adaptor")
        
        st.write("**Purpose:** Define final tensor memory layout and access patterns")
        
        # Show conceptual tensor descriptor creation
        tensor_lengths = [64, 128, 256][:encoding.ndim_x]  # Sample lengths
        if len(tensor_lengths) < encoding.ndim_x:
            tensor_lengths.extend([64] * (encoding.ndim_x - len(tensor_lengths)))
        
        st.write("**Conceptual Tensor Descriptor:**")
        st.write(f"- Tensor dimensions: {encoding.ndim_x}")
        st.write(f"- Example tensor shape: {tensor_lengths}")
        st.write(f"- Total elements: {np.prod(tensor_lengths):,}")
        st.write(f"- Memory footprint (float32): {np.prod(tensor_lengths) * 4:,} bytes")
        
        st.write("**Usage in Tile Distribution:**")
        st.write("""
        The TensorDescriptor enables:
        1. **Shape Definition**: Defines the actual tensor dimensions and sizes
        2. **Stride Calculation**: Computes memory strides for efficient access
        3. **Coordinate Mapping**: Maps logical coordinates to linear memory offsets
        4. **Distributed Access**: Supports thread-local tensor access patterns
        """)
        
        # Try to create it but handle errors gracefully
        try:
            adaptor_encoding = _make_adaptor_encoding_for_tile_distribution(encoding)
            # Extract the ps_ys_to_xs_adaptor_encoding (first element) from the result
            ps_ys_to_xs_adaptor_encoding = adaptor_encoding[0]
            tensor_adaptor = _construct_static_tensor_adaptor_from_encoding(ps_ys_to_xs_adaptor_encoding)
            tensor_descriptor = make_tensor_descriptor_from_adaptor(tensor_adaptor, tensor_lengths)
            
            st.success("✅ Successfully created tensor descriptor!")
            st.write("**Descriptor Properties:**")
            st.json({
                "num_dimensions": len(tensor_lengths),
                "total_elements": np.prod(tensor_lengths),
                "tensor_lengths": tensor_lengths
            })
            
        except Exception as e:
            st.warning(f"Creation failed (expected in Python context): {e}")
            st.write("**In C++ context, this would create a descriptor enabling:**")
            st.write("- Efficient coordinate-to-memory-offset conversion")
            st.write("- Optimized memory access patterns for GPU kernels")
            st.write("- Integration with distributed tensor operations")

    # Step 5: Complete Pipeline
    with st.expander("🔄 Step 5: Complete Transformation Pipeline", expanded=True):
        st.subheader("End-to-End Transformation")
        
        st.write("**Complete Pipeline Flow:**")
        st.code("""
TileDistributionEncoding
        ↓
_make_adaptor_encoding_for_tile_distribution()
        ↓
AdaptorEncoding (transformation blueprint)
        ↓
_construct_static_tensor_adaptor_from_encoding()
        ↓
TensorAdaptor (functional transformer)
        ↓
make_tensor_descriptor_from_adaptor()
        ↓
TensorDescriptor (memory layout)
        """)
        
        st.write("**Key Insights:**")
        st.write("""
        1. **Encoding → Blueprint**: The tile distribution encoding becomes a transformation blueprint
        2. **Blueprint → Function**: The blueprint becomes a functional coordinate transformer
        3. **Function → Layout**: The transformer defines the final tensor memory layout
        4. **Coordinate Flow**: (P,Y) coordinates → intermediate transformations → X coordinates
        5. **Memory Mapping**: Final coordinates map to actual memory locations
        """)
        
        # Show practical example
        st.write("**Practical Example:**")
        st.write(f"""
        For this encoding:
        - P dimensions: {len(encoding.ps_to_rhss_major)} (thread coordinates)
        - Y dimensions: {len(encoding.ys_to_rhs_major)} (local tile coordinates)  
        - X dimensions: {encoding.ndim_x} (global tensor coordinates)
        - R dimensions: {len(encoding.rs_lengths)} (replication factors)
        
        A thread with P coordinates [p0, p1, ...] working on local tile Y coordinates [y0, y1, ...]
        gets transformed to global tensor X coordinates [x0, x1, ...] for memory access.
        """)

def show_interactive_editor():
    """Show interactive encoding editor similar to other apps"""
    st.header("✏️ Interactive Encoding Editor")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Edit Encoding")
        
        # Basic parameters
        ndim_x = st.number_input("Number of X dimensions", min_value=1, max_value=5, value=2)
        ndim_r = st.number_input("Number of R dimensions", min_value=0, max_value=3, value=1)
        
        # R lengths
        rs_lengths = []
        for i in range(ndim_r):
            length = st.number_input(f"R{i} length", min_value=1, value=4, key=f"r_{i}")
            rs_lengths.append(length)
        
        # H lengths
        st.write("**H Dimensions:**")
        ndim_h = st.number_input("Number of H dimension groups", min_value=0, max_value=3, value=1)
        hs_lengthss = []
        for i in range(ndim_h):
            h_count = st.number_input(f"H{i} sub-dimensions", min_value=1, max_value=4, value=2, key=f"h_count_{i}")
            h_lengths = []
            for j in range(h_count):
                length = st.number_input(f"H{i}[{j}] length", min_value=1, value=8, key=f"h_{i}_{j}")
                h_lengths.append(length)
            hs_lengthss.append(h_lengths)
        
        # P mappings
        st.write("**P Mappings:**")
        ndim_p = st.number_input("Number of P dimensions", min_value=1, max_value=4, value=2)
        ps_to_rhss_major = []
        ps_to_rhss_minor = []
        for i in range(ndim_p):
            st.write(f"P{i} mappings:")
            p_count = st.number_input(f"P{i} mapping count", min_value=1, max_value=3, value=1, key=f"p_count_{i}")
            major_list = []
            minor_list = []
            for j in range(p_count):
                major = st.number_input(f"P{i}[{j}] RH major", min_value=0, value=0, key=f"p_major_{i}_{j}")
                minor = st.number_input(f"P{i}[{j}] RH minor", min_value=0, value=0, key=f"p_minor_{i}_{j}")
                major_list.append(major)
                minor_list.append(minor)
            ps_to_rhss_major.append(major_list)
            ps_to_rhss_minor.append(minor_list)
        
        # Y mappings
        st.write("**Y Mappings:**")
        ndim_y = st.number_input("Number of Y dimensions", min_value=1, max_value=4, value=2)
        ys_to_rhs_major = []
        ys_to_rhs_minor = []
        for i in range(ndim_y):
            major = st.number_input(f"Y{i} RH major", min_value=0, value=1, key=f"y_major_{i}")
            minor = st.number_input(f"Y{i} RH minor", min_value=0, value=0, key=f"y_minor_{i}")
            ys_to_rhs_major.append(major)
            ys_to_rhs_minor.append(minor)
    
    with col2:
        st.subheader("Generated Encoding")
        
        try:
            # Create encoding from user input
            encoding = TileDistributionEncoding(
                rs_lengths=rs_lengths,
                hs_lengthss=hs_lengthss,
                ps_to_rhss_major=ps_to_rhss_major,
                ps_to_rhss_minor=ps_to_rhss_minor,
                ys_to_rhs_major=ys_to_rhs_major,
                ys_to_rhs_minor=ys_to_rhs_minor
            )
            
            st.write("**Created Encoding:**")
            st.json({
                "ndim_x": encoding.ndim_x,
                "rs_lengths": encoding.rs_lengths,
                "hs_lengthss": encoding.hs_lengthss,
                "ps_to_rhss_major": encoding.ps_to_rhss_major,
                "ps_to_rhss_minor": encoding.ps_to_rhss_minor,
                "ys_to_rhs_major": encoding.ys_to_rhs_major,
                "ys_to_rhs_minor": encoding.ys_to_rhs_minor
            })
            
            if st.button("🔍 Analyze This Encoding"):
                st.session_state.custom_encoding = encoding
                
        except Exception as e:
            st.error(f"Invalid encoding: {e}")

def show_code_examples():
    """Show code examples similar to spans_app.py"""
    st.header("💻 Code Examples")
    
    st.subheader("Basic Usage Pattern")
    st.code("""
# Step 1: Create or load a TileDistributionEncoding
encoding = TileDistributionEncoding(
    rs_lengths=[4],
    hs_lengthss=[[8, 16]],
    ps_to_rhss_major=[[0], [1]],
    ps_to_rhss_minor=[[0], [0]],
    ys_to_rhs_major=[1, 1],
    ys_to_rhs_minor=[0, 1]
)

# Step 2: Create adaptor encoding (transformation blueprint)
adaptor_encoding_result = _make_adaptor_encoding_for_tile_distribution(encoding)

# Step 3: Construct functional tensor adaptor
# Extract the ps_ys_to_xs_adaptor_encoding (first element)
ps_ys_to_xs_adaptor_encoding = adaptor_encoding_result[0]
tensor_adaptor = _construct_static_tensor_adaptor_from_encoding(ps_ys_to_xs_adaptor_encoding)

# Step 4: Create tensor descriptor with actual tensor dimensions
tensor_lengths = [128, 256]  # Example tensor shape
tensor_descriptor = make_tensor_descriptor_from_adaptor(tensor_adaptor, tensor_lengths)

# Now you can use the tensor_descriptor for coordinate transformations
# in your kernel code
""", language="python")
    
    st.subheader("Kernel Integration Example")
    st.code("""
// In CUDA kernel
__global__ void my_kernel(float* tensor, TensorDescriptor desc) {
    // Get thread coordinates (P dimensions)
    auto thread_id = make_tuple(blockIdx.x, threadIdx.x);
    
    // Get local tile coordinates (Y dimensions)  
    auto local_coords = make_tuple(local_y0, local_y1);
    
    // Combine P and Y coordinates
    auto py_coords = make_tuple(thread_id, local_coords);
    
    // Transform to global tensor coordinates (X dimensions)
    auto global_coords = desc.calculate_coordinate(py_coords);
    
    // Calculate memory offset
    auto offset = desc.calculate_offset(global_coords);
    
    // Access tensor data
    float value = tensor[offset];
}
""", language="cpp")

def main():
    st.title("🔄 Adaptor Transformation Deep Dive")
    st.markdown("Understanding the step-by-step process of coordinate transformation in tile distribution")
    
    # Sidebar for example selection
    st.sidebar.header("📚 Examples")
    
    # Load examples
    examples, errors = get_example_encodings()
    example_names = list(examples.keys())
    
    # Show parsing errors if any
    if errors:
        with st.sidebar:
            st.warning("Some examples failed to parse:")
            for error in errors:
                st.text(error)
    
    if not example_names:
        st.error("No valid examples found. Please check the examples.py file.")
        return
    
    selected_example = st.sidebar.selectbox(
        "Choose an example:",
        example_names,
        index=0
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Step-by-Step Analysis", 
        "✏️ Interactive Editor",
        "💻 Code Examples",
        "📊 Comparison View",
        "🎯 Key Concepts"
    ])
    
    with tab1:
        if selected_example and selected_example in examples:
            encoding = examples[selected_example]
            st.subheader(f"Analysis: {selected_example}")
            show_step_by_step_analysis(encoding)
        elif 'custom_encoding' in st.session_state:
            st.subheader("Analysis: Custom Encoding")
            show_step_by_step_analysis(st.session_state.custom_encoding)
    
    with tab2:
        show_interactive_editor()
    
    with tab3:
        show_code_examples()
    
    with tab4:
        st.header("📊 Multi-Example Comparison")
        
        if len(examples) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                example1 = st.selectbox("First example:", example_names, key="comp1")
                if example1:
                    encoding1 = examples[example1]
                    st.write(f"**{example1}:**")
                    st.json({
                        "ndim_x": encoding1.ndim_x,
                        "ndim_y": len(encoding1.ys_to_rhs_major),
                        "ndim_p": len(encoding1.ps_to_rhss_major),
                        "ndim_r": len(encoding1.rs_lengths)
                    })
            
            with col2:
                example2 = st.selectbox("Second example:", example_names, key="comp2")
                if example2:
                    encoding2 = examples[example2]
                    st.write(f"**{example2}:**")
                    st.json({
                        "ndim_x": encoding2.ndim_x,
                        "ndim_y": len(encoding2.ys_to_rhs_major),
                        "ndim_p": len(encoding2.ps_to_rhss_major),
                        "ndim_r": len(encoding2.rs_lengths)
                    })
        else:
            st.write("Need at least 2 examples for comparison")
    
    with tab5:
        st.header("🎯 Key Concepts")
        
        st.subheader("🏗️ Adaptor Encoding Creation")
        st.write("""
        **Purpose**: Convert tile distribution encoding into transformation blueprint
        
        **Process**:
        1. Analyze RH space structure (R + H dimensions)
        2. Create replication transforms for R dimensions
        3. Create unmerge transforms for H dimensions
        4. Create merge transforms for final X dimensions
        5. Add freeze transforms for fixed coordinates
        """)
        
        st.subheader("🔧 Tensor Adaptor Construction")
        st.write("""
        **Purpose**: Turn blueprint into functional coordinate transformer
        
        **Process**:
        1. Parse transformation encoding string
        2. Build transformation chain from individual transforms
        3. Create composite adaptor that applies all transforms
        4. Validate input/output dimension compatibility
        """)
        
        st.subheader("📐 Tensor Descriptor Creation")
        st.write("""
        **Purpose**: Define final tensor memory layout and access patterns
        
        **Process**:
        1. Apply adaptor to tensor shape information
        2. Calculate strides and memory layout
        3. Create descriptor for coordinate-to-offset mapping
        4. Enable efficient tensor access in kernels
        """)
        
        st.subheader("🔄 Complete Pipeline")
        st.write("""
        **Coordinate Flow**: (P,Y) → Transformations → X → Memory Offset
        
        **Key Insight**: Each step adds a layer of abstraction:
        - Encoding: Mathematical description
        - Adaptor: Functional transformation
        - Descriptor: Physical memory layout
        """)

if __name__ == "__main__":
    main() 