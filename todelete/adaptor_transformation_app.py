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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

def generate_make_adaptor_encoding_explanation(encoding: TileDistributionEncoding) -> str:
    """Generate detailed explanation for _make_adaptor_encoding_for_tile_distribution function."""
    explanation = []
    
    explanation.append("=== STEP 1: _make_adaptor_encoding_for_tile_distribution ===\n")
    
    explanation.append("FUNCTION PURPOSE:")
    explanation.append("Convert TileDistributionEncoding into transformation blueprint")
    explanation.append("Input: TileDistributionEncoding object")
    explanation.append("Output: Tuple with adaptor encodings and mappings\n")
    
    # Calculate actual dimensions
    ndim_p = len(encoding.ps_to_rhss_major)
    ndim_y = len(encoding.ys_to_rhs_major)
    ndim_r = len(encoding.rs_lengths)
    ndim_h = len(encoding.hs_lengthss)
    total_h_dims = sum(len(h_lengths) for h_lengths in encoding.hs_lengthss)
    
    explanation.append("STEP 1.1: Analyze Input Encoding Structure")
    explanation.append(f"  ndim_x = {encoding.ndim_x}  // Output tensor dimensions")
    explanation.append(f"  ndim_y = {ndim_y}  // Local tile dimensions")
    explanation.append(f"  ndim_p = {ndim_p}  // Partition dimensions")
    explanation.append(f"  ndim_r = {ndim_r}  // Replication dimensions")
    explanation.append(f"  ndim_h = {ndim_h}  // Hidden dimension groups")
    explanation.append(f"  total_h_subdims = {total_h_dims}  // Total hidden sub-dimensions")
    explanation.append("")
    
    total_rh = ndim_r + ndim_h
    explanation.append("STEP 1.2: Calculate RH Space Structure")
    explanation.append(f"  total_rh_dims = {ndim_r} + {ndim_h} = {total_rh}")
    explanation.append("  RH space organization:")
    
    if ndim_r > 0:
        explanation.append("    RH[0]: R sequence (replication space)")
        for r_idx, r_len in enumerate(encoding.rs_lengths):
            explanation.append(f"      R[{r_idx}] = {r_len}")
    else:
        explanation.append("    No R dimensions (no replication)")
    
    for h_idx, h_lengths in enumerate(encoding.hs_lengthss):
        explanation.append(f"    RH[{h_idx + 1}]: H{h_idx} sequence")
        for h_minor, h_len in enumerate(h_lengths):
            explanation.append(f"      H{h_idx}[{h_minor}] = {h_len}")
    
    if not encoding.hs_lengthss:
        explanation.append("    No H dimensions (no hidden transformations)")
    explanation.append("")
    
    explanation.append("STEP 1.3: Build Transformation Chain")
    explanation.append("Creates sequence of coordinate transformations:")
    
    transform_steps = []
    transform_count = 0
    
    # Replicate operations
    if encoding.rs_lengths:
        for i, length in enumerate(encoding.rs_lengths):
            transform_steps.append(f"replicate<{length}>")
            explanation.append(f"  - replicate<{length}>: Create R{i} dimension with {length} replications")
            transform_count += 1
    else:
        explanation.append("  - No replicate transforms (no R dimensions)")
    
    # Unmerge operations  
    if encoding.hs_lengthss:
        for i, lengths in enumerate(encoding.hs_lengthss):
            lengths_str = ",".join(map(str, lengths))
            transform_steps.append(f"unmerge<{lengths_str}>")
            explanation.append(f"  - unmerge<{lengths_str}>: Split H{i} into {len(lengths)} components")
            transform_count += 1
    else:
        explanation.append("  - No unmerge transforms (no H dimensions)")
    
    # Merge operations
    transform_steps.append("merge<...>")
    explanation.append(f"  - merge<...>: Combine into {encoding.ndim_x} final X coordinates")
    explanation.append(f"  **Total transformation steps: {transform_count + 1}**")
    explanation.append("")
    
    explanation.append("STEP 1.4: Create Dimension Mappings")
    explanation.append("Maps P and Y dimensions to RH space:")
    
    # P mappings with validation
    for i, (major_list, minor_list) in enumerate(zip(encoding.ps_to_rhss_major, encoding.ps_to_rhss_minor)):
        explanation.append(f"  P{i} mappings ({len(major_list)} components):")
        for j, (major, minor) in enumerate(zip(major_list, minor_list)):
            if major == 0:
                target = f"R[{minor}]"
                value = encoding.rs_lengths[minor] if minor < len(encoding.rs_lengths) else "INVALID"
            else:
                target = f"H{major-1}[{minor}]"
                if major-1 < len(encoding.hs_lengthss) and minor < len(encoding.hs_lengthss[major-1]):
                    value = encoding.hs_lengthss[major-1][minor]
                else:
                    value = "INVALID"
            explanation.append(f"    P{i}[{j}] → RH[{major}] ({target} = {value})")
    
    # Y mappings with validation
    for i, (major, minor) in enumerate(zip(encoding.ys_to_rhs_major, encoding.ys_to_rhs_minor)):
        if major == 0:
            target = f"R[{minor}]"
            value = encoding.rs_lengths[minor] if minor < len(encoding.rs_lengths) else "INVALID"
        else:
            target = f"H{major-1}[{minor}]"
            if major-1 < len(encoding.hs_lengthss) and minor < len(encoding.hs_lengthss[major-1]):
                value = encoding.hs_lengthss[major-1][minor]
            else:
                value = "INVALID"
        explanation.append(f"  Y{i} → RH[{major}] ({target} = {value})")
    explanation.append("")
    
    explanation.append("STEP 1.5: Generate Hidden ID Mappings")
    explanation.append("Creates rh_major_minor_to_hidden_ids mapping:")
    explanation.append("  - Maps (rh_major, rh_minor) coordinates to unique hidden IDs")
    explanation.append("  - Enables efficient coordinate transformation")
    explanation.append("  - Supports compile-time optimization")
    rh_combinations = 0
    for i in range(total_rh + 1):  # +1 for potential merge dimensions
        if i == 0:
            rh_combinations += len(encoding.rs_lengths)
        elif i <= len(encoding.hs_lengthss):
            rh_combinations += len(encoding.hs_lengthss[i-1]) if i-1 < len(encoding.hs_lengthss) else 0
    explanation.append(f"  - Estimated mapping entries: ~{rh_combinations}")
    explanation.append("")
    
    explanation.append("STEP 1.6: Build Output Tuple")
    explanation.append("Function returns tuple containing:")
    explanation.append("  [0] ps_ys_to_xs_adaptor_encoding:")
    explanation.append(f"      - Input dims: {ndim_p + ndim_y} (P+Y)")
    explanation.append(f"      - Output dims: {encoding.ndim_x} (X)")
    if transform_steps:
        explanation.append(f"      - Transform chain: {' ∘ '.join(transform_steps)}")
    else:
        explanation.append("      - Transform chain: direct mapping")
    explanation.append("  [1] ys_to_d_adaptor_encoding:")
    explanation.append(f"      - Maps {ndim_y} Y coordinates to distributed indices")
    explanation.append("  [2] d_length:")
    explanation.append("      - Length of distributed dimension")
    explanation.append("  [3] rh_major_minor_to_hidden_ids:")
    explanation.append(f"      - RH coordinate to hidden ID mapping (~{rh_combinations} entries)")
    explanation.append("")
    
    explanation.append("KEY INSIGHTS:")
    explanation.append("1. **Blueprint Creation**: This function creates the transformation blueprint")
    explanation.append("2. **No Actual Transformation**: Only describes what transformations to apply")
    explanation.append("3. **Compile-time Computation**: All mappings computed at compile time")
    explanation.append("4. **Foundation for Step 2**: Output becomes input to tensor adaptor construction")
    explanation.append(f"5. **Complexity**: Handles {ndim_p}P + {ndim_y}Y → {encoding.ndim_x}X coordinate mapping")
    
    return "\n".join(explanation)

def generate_construct_tensor_adaptor_explanation(encoding: TileDistributionEncoding) -> str:
    """Generate detailed explanation for _construct_static_tensor_adaptor_from_encoding function."""
    explanation = []
    
    explanation.append("=== STEP 2: _construct_static_tensor_adaptor_from_encoding ===\n")
    
    explanation.append("FUNCTION PURPOSE:")
    explanation.append("Convert transformation blueprint into functional coordinate transformer")
    explanation.append("Input: ps_ys_to_xs_adaptor_encoding (from Step 1)")
    explanation.append("Output: TensorAdaptor object with coordinate transformation capability\n")
    
    input_dims = len(encoding.ps_to_rhss_major) + len(encoding.ys_to_rhs_major)
    explanation.append("STEP 2.1: Parse Adaptor Encoding")
    explanation.append("Extracts transformation information:")
    explanation.append("  - Transformation chain string")
    explanation.append(f"  - Input dimension count: {input_dims} (P+Y)")
    explanation.append(f"  - Output dimension count: {encoding.ndim_x} (X)")
    explanation.append("  - Dimension ID mappings")
    explanation.append("  - Hidden coordinate mappings")
    explanation.append("")
    
    explanation.append("STEP 2.2: Validate Dimensions")
    explanation.append("Ensures transformation compatibility:")
    explanation.append(f"  Input (top) dimensions: {input_dims}")
    p_dims = [f"P{i}" for i in range(len(encoding.ps_to_rhss_major))]
    y_dims = [f"Y{i}" for i in range(len(encoding.ys_to_rhs_major))]
    explanation.append(f"    P coordinates: [{', '.join(p_dims)}]")
    explanation.append(f"    Y coordinates: [{', '.join(y_dims)}]")
    explanation.append(f"  Output (bottom) dimensions: {encoding.ndim_x}")
    x_dims = [f"X{i}" for i in range(encoding.ndim_x)]
    explanation.append(f"    X coordinates: [{', '.join(x_dims)}]")
    explanation.append("")
    
    explanation.append("STEP 2.3: Parse Transformation Chain")
    explanation.append("Breaks down transformation string into individual operations:")
    
    # Show the actual transformations based on encoding
    transform_count = 1
    total_transforms = 0
    
    for i, length in enumerate(encoding.rs_lengths):
        explanation.append(f"  Transform {transform_count}: replicate<{length}>")
        explanation.append(f"    - Creates R{i} dimension")
        explanation.append(f"    - Adds {length}-fold replication")
        explanation.append(f"    - Input dims: n → Output dims: n+1")
        transform_count += 1
        total_transforms += 1
    
    for i, lengths in enumerate(encoding.hs_lengthss):
        lengths_str = ",".join(map(str, lengths))
        explanation.append(f"  Transform {transform_count}: unmerge<{lengths_str}>")
        explanation.append(f"    - Splits H{i} dimension")
        explanation.append(f"    - Components: {lengths}")
        explanation.append(f"    - Input dims: n → Output dims: n+{len(lengths)-1}")
        transform_count += 1
        total_transforms += 1
    
    explanation.append(f"  Transform {transform_count}: merge<...>")
    explanation.append(f"    - Combines dimensions into final X coordinates")
    explanation.append(f"    - Input dims: intermediate → Output dims: {encoding.ndim_x}")
    explanation.append(f"  **Total transformation operations: {total_transforms + 1}**")
    explanation.append("")
    
    explanation.append("STEP 2.4: Create Transformation Objects")
    explanation.append("Instantiates C++ transformation objects:")
    explanation.append(f"  - {total_transforms + 1} transformation objects created")
    explanation.append("  - Each transform stores transformation parameters")
    explanation.append("  - Enable efficient coordinate computation")
    explanation.append("  - Support compile-time optimization")
    explanation.append("")
    
    explanation.append("STEP 2.5: Chain Transformations")
    explanation.append("Links transformations into functional pipeline:")
    explanation.append("  Algorithm (conceptual):")
    
    # Generate actual transformation chain based on encoding
    transform_names = []
    for i, length in enumerate(encoding.rs_lengths):
        transform_names.append(f"replicate_{length}")
    for i, lengths in enumerate(encoding.hs_lengthss):
        lengths_str = "_".join(map(str, lengths))
        transform_names.append(f"unmerge_{lengths_str}")
    transform_names.append("merge_final")
    
    for i, name in enumerate(transform_names):
        explanation.append(f"    auto transform{i+1} = create_{name}_transform(...)")
    
    explanation.append("    ")
    chain_call = "compose(" + ", ".join([f"transform{i+1}" for i in range(len(transform_names))]) + ")"
    explanation.append(f"    auto chained = {chain_call}")
    explanation.append("    return TensorAdaptor(chained)")
    explanation.append("")
    
    explanation.append("STEP 2.6: Create TensorAdaptor Object")
    explanation.append("Final adaptor object properties:")
    explanation.append(f"  - get_num_of_top_dimension(): {input_dims}")
    explanation.append(f"  - get_num_of_bottom_dimension(): {encoding.ndim_x}")
    explanation.append("  - calculate_bottom_coordinate(top_coord): Transforms coordinates")
    explanation.append("  - calculate_top_coordinate(bottom_coord): Inverse transformation")
    explanation.append("")
    
    explanation.append("STEP 2.7: Coordinate Transformation Example")
    explanation.append("How the adaptor transforms coordinates:")
    
    # Create realistic example coordinates based on encoding structure
    p_coords = []
    for i in range(len(encoding.ps_to_rhss_major)):
        # Use small example values that make sense for thread IDs
        coord_val = (i + 1) * 2
        p_coords.append(coord_val)
    
    y_coords = []
    for i in range(len(encoding.ys_to_rhs_major)):
        # Use small example values for tile coordinates
        coord_val = i + 1
        y_coords.append(coord_val)
    
    all_coords = p_coords + y_coords
    explanation.append(f"  Input coordinates: {all_coords}")
    explanation.append("  Transformation process:")
    explanation.append(f"    1. Apply {len(encoding.rs_lengths)} replicate transforms → add R dimensions")
    explanation.append(f"    2. Apply {len(encoding.hs_lengthss)} unmerge transforms → split H dimensions") 
    explanation.append(f"    3. Apply merge transforms → combine to X coordinates")
    explanation.append(f"  Output coordinates: [computed_x0, computed_x1, ...] ({encoding.ndim_x} values)")
    explanation.append("")
    
    explanation.append("STEP 2.8: Memory and Performance")
    explanation.append("Optimization characteristics:")
    explanation.append("  - **Compile-time**: All transformations resolved at compile time")
    explanation.append("  - **Zero overhead**: No runtime transformation cost")
    explanation.append("  - **Inlined**: Coordinate calculations inlined into kernel")
    explanation.append(f"  - **Template depth**: ~{total_transforms + 1} nested template instantiations")
    explanation.append("  - **Optimized**: Compiler can optimize entire pipeline")
    explanation.append("")
    
    explanation.append("KEY INSIGHTS:")
    explanation.append("1. **Functional Transformer**: Creates working coordinate transformer")
    explanation.append("2. **Template Magic**: Uses C++ templates for zero-cost abstraction")
    explanation.append(f"3. **Coordinate Mapping**: Enables ({len(p_dims)}P,{len(y_dims)}Y) → {encoding.ndim_x}X coordinate transformation")
    explanation.append("4. **Ready for Step 3**: Output becomes input to tensor descriptor creation")
    
    return "\n".join(explanation)

def generate_make_tensor_descriptor_explanation(encoding: TileDistributionEncoding) -> str:
    """Generate detailed explanation for make_tensor_descriptor_from_adaptor function."""
    explanation = []
    
    explanation.append("=== STEP 3: make_tensor_descriptor_from_adaptor ===\n")
    
    explanation.append("FUNCTION PURPOSE:")
    explanation.append("Define final tensor memory layout and access patterns")
    explanation.append("Input: TensorAdaptor (from Step 2) + tensor shape")
    explanation.append("Output: TensorDescriptor for efficient memory access\n")
    
    # Calculate adaptive tensor shape based on encoding
    # Use reasonable defaults that scale with the encoding dimensions
    base_sizes = [64, 128, 256, 512, 1024]
    example_shape = []
    for i in range(encoding.ndim_x):
        # Use encoding structure to inform tensor sizes
        size_factor = 1
        
        # Consider R dimensions influence
        for r_len in encoding.rs_lengths:
            size_factor *= max(1, r_len // 2)
        
        # Consider H dimensions influence  
        for h_lengths in encoding.hs_lengthss:
            for h_len in h_lengths:
                size_factor = max(size_factor, h_len)
        
        # Select appropriate base size
        base_size = base_sizes[min(i, len(base_sizes)-1)]
        adapted_size = max(32, min(base_size, base_size * size_factor // 8))
        example_shape.append(adapted_size)
    
    explanation.append("STEP 3.1: Analyze Input Parameters")
    explanation.append("Function receives:")
    explanation.append("  - TensorAdaptor: Coordinate transformation function")
    explanation.append(f"  - Tensor lengths: [{', '.join(map(str, example_shape))}] (adapted to encoding)")
    explanation.append(f"  - Number of dimensions: {encoding.ndim_x}")
    explanation.append(f"  - Total tensor elements: {np.prod(example_shape):,}")
    explanation.append("")
    
    explanation.append("STEP 3.2: Calculate Memory Layout")
    explanation.append("Determines tensor memory organization:")
    explanation.append("  Memory layout calculation:")
    
    # Calculate strides (row-major order)
    strides = [1]
    for i in range(len(example_shape) - 1, 0, -1):
        strides.insert(0, strides[0] * example_shape[i])
    
    explanation.append("    Row-major stride calculation:")
    for i, (dim, stride) in enumerate(zip(example_shape, strides)):
        explanation.append(f"      X{i}: length={dim}, stride={stride}")
    explanation.append(f"    Total memory: {np.prod(example_shape) * 4:,} bytes (float32)")
    explanation.append("")
    
    explanation.append("STEP 3.3: Integrate Adaptor")
    explanation.append("Combines adaptor with memory layout:")
    explanation.append("  - Adaptor provides coordinate transformation")
    explanation.append("  - Descriptor provides memory offset calculation")
    p_dims = len(encoding.ps_to_rhss_major)
    y_dims = len(encoding.ys_to_rhs_major)
    explanation.append(f"  - Together: ({p_dims}P,{y_dims}Y) coordinates → memory address")
    explanation.append("")
    explanation.append("  Integration process:")
    p_coords = ", ".join([f"P{i}" for i in range(p_dims)])
    y_coords = ", ".join([f"Y{i}" for i in range(y_dims)])
    x_coords = ", ".join([f"X{i}" for i in range(encoding.ndim_x)])
    explanation.append(f"    1. Use adaptor to transform ({p_coords},{y_coords}) → ({x_coords})")
    explanation.append("    2. Use descriptor to calculate memory offset")
    explanation.append("    3. Access tensor_data[offset]")
    explanation.append("")
    
    explanation.append("STEP 3.4: Create TensorDescriptor Object")
    explanation.append("Final descriptor object properties:")
    explanation.append(f"  - get_lengths(): Returns [{', '.join(map(str, example_shape))}]")
    explanation.append(f"  - get_strides(): Returns [{', '.join(map(str, strides))}]")
    explanation.append("  - calculate_offset(coordinate): Computes memory offset")
    explanation.append(f"  - element_space_size(): Returns {np.prod(example_shape):,}")
    explanation.append("")
    
    explanation.append("STEP 3.5: Memory Offset Calculation")
    explanation.append("How coordinates map to memory:")
    explanation.append("  Algorithm:")
    explanation.append("    offset = 0")
    for i, (stride, dim) in enumerate(zip(strides, example_shape)):
        explanation.append(f"    offset += X{i} * {stride}  // X{i} ∈ [0, {dim-1}]")
    explanation.append("    return offset")
    explanation.append("")
    
    explanation.append("STEP 3.6: Complete Access Pattern")
    explanation.append("Full memory access example:")
    explanation.append("  // In kernel:")
    
    p_coords_example = ", ".join([f"p{i}" for i in range(p_dims)])
    y_coords_example = ", ".join([f"y{i}" for i in range(y_dims)])
    
    explanation.append(f"  auto p_coords = make_tuple({p_coords_example});")
    explanation.append(f"  auto y_coords = make_tuple({y_coords_example});")
    explanation.append("  auto py_coords = container_concat(p_coords, y_coords);")
    explanation.append("")
    explanation.append("  // Step 1: Apply adaptor transformation")
    explanation.append("  auto x_coords = tensor_adaptor.calculate_bottom_coordinate(py_coords);")
    explanation.append("")
    explanation.append("  // Step 2: Calculate memory offset")
    explanation.append("  auto offset = tensor_descriptor.calculate_offset(x_coords);")
    explanation.append("")
    explanation.append("  // Step 3: Access memory")
    explanation.append("  float value = tensor_data[offset];")
    explanation.append("")
    
    explanation.append("STEP 3.7: Bounds Checking and Safety")
    explanation.append("Ensuring safe memory access:")
    bounds_checks = []
    for i, dim in enumerate(example_shape):
        bounds_checks.append(f"X{i} < {dim}")
    explanation.append(f"  - Coordinate bounds: {' && '.join(bounds_checks)}")
    explanation.append(f"  - Memory bounds: offset < {np.prod(example_shape):,}")
    explanation.append("  - Thread safety: Different threads access different offsets")
    explanation.append("  - Alignment: Memory access follows GPU alignment rules")
    explanation.append("")
    
    explanation.append("STEP 3.8: Performance Optimizations")
    explanation.append("GPU memory access optimizations:")
    explanation.append("  **Coalesced Access**: Adjacent threads access adjacent memory")
    # Calculate typical thread access pattern
    min_stride = min(strides)
    explanation.append(f"    - Thread 0: accesses offset[base + 0]")
    explanation.append(f"    - Thread 1: accesses offset[base + {min_stride}]")
    explanation.append(f"    - Thread N: accesses offset[base + N*{min_stride}]")
    explanation.append("")
    explanation.append("  **Cache Efficiency**: Spatial locality improves cache hits")
    cache_line_elements = 128 // 4  # 128-byte cache line, 4-byte floats
    explanation.append(f"    - Cache line: {cache_line_elements} float32 elements")
    explanation.append("    - Spatial locality: Nearby addresses cached together")
    explanation.append("")
    explanation.append("  **Bank Conflict Avoidance**: Proper stride patterns")
    explanation.append("    - Shared memory: avoid same bank access")
    explanation.append("    - Global memory: optimize for memory controller")
    explanation.append("")
    
    explanation.append("KEY INSIGHTS:")
    explanation.append("1. **Memory Mapping**: Creates final memory access pattern")
    explanation.append("2. **Performance Critical**: Determines memory efficiency")
    explanation.append(f"3. **Complete Pipeline**: Completes the ({p_dims}P,{y_dims}Y) → memory transformation")
    explanation.append("4. **Ready for Use**: Descriptor can be used in GPU kernels")
    explanation.append("5. **Zero Runtime Cost**: All calculations optimized at compile time")
    
    return "\n".join(explanation)

def generate_complete_pipeline_explanation(encoding: TileDistributionEncoding) -> str:
    """Generate explanation for the complete 3-step pipeline."""
    explanation = []
    
    explanation.append("=== COMPLETE 3-STEP TRANSFORMATION PIPELINE ===\n")
    
    input_dims = len(encoding.ps_to_rhss_major) + len(encoding.ys_to_rhs_major)
    p_count = len(encoding.ps_to_rhss_major)
    y_count = len(encoding.ys_to_rhs_major)
    
    explanation.append("OVERVIEW:")
    explanation.append("The three functions work together to create efficient GPU memory access:")
    explanation.append("1. _make_adaptor_encoding_for_tile_distribution → Blueprint")
    explanation.append("2. _construct_static_tensor_adaptor_from_encoding → Transformer")
    explanation.append("3. make_tensor_descriptor_from_adaptor → Memory accessor")
    explanation.append("")
    
    explanation.append("STEP-BY-STEP PIPELINE EXECUTION:")
    explanation.append("")
    
    explanation.append("🏗️ STEP 1: Create Transformation Blueprint")
    explanation.append("  Input: TileDistributionEncoding")
    explanation.append("  Process: Analyze encoding structure")
    explanation.append("  Output: Adaptor encoding (transformation instructions)")
    explanation.append("  Key Result: ps_ys_to_xs_adaptor_encoding")
    explanation.append("")
    
    explanation.append("🔧 STEP 2: Build Functional Transformer")
    explanation.append("  Input: ps_ys_to_xs_adaptor_encoding")
    explanation.append("  Process: Parse and instantiate transformations")
    explanation.append("  Output: TensorAdaptor (working coordinate transformer)")
    explanation.append(f"  Key Result: ({p_count}P,{y_count}Y) → {encoding.ndim_x}X coordinate transformation ({input_dims} → {encoding.ndim_x} dims)")
    explanation.append("")
    
    explanation.append("📐 STEP 3: Create Memory Layout")
    explanation.append("  Input: TensorAdaptor + tensor shape")
    explanation.append("  Process: Combine transformation with memory layout")
    explanation.append("  Output: TensorDescriptor (memory access calculator)")
    explanation.append(f"  Key Result: ({p_count}P,{y_count}Y) coordinates → memory offset")
    explanation.append("")
    
    explanation.append("COMPLETE DATA FLOW:")
    explanation.append("")
    
    # Show the complete data flow with actual encoding dimensions
    explanation.append("Thread Hardware Input:")
    p_example = [f"p{i}" for i in range(p_count)]
    y_example = [f"y{i}" for i in range(y_count)]
    explanation.append(f"  P coordinates: [{', '.join(p_example)}] (from get_warp_id(), get_lane_id())")
    explanation.append(f"  Y coordinates: [{', '.join(y_example)}] (from tile iteration)")
    explanation.append("    ↓")
    
    explanation.append("Pipeline Processing:")
    explanation.append("  Step 1: Blueprint defines transformation chain")
    explanation.append("  Step 2: Adaptor applies coordinate transformation")
    explanation.append("  Step 3: Descriptor calculates memory offset")
    explanation.append("    ↓")
    
    explanation.append("Final Memory Access:")
    explanation.append("  tensor_data[calculated_offset] → actual data value")
    explanation.append("")
    
    explanation.append("PERFORMANCE CHARACTERISTICS:")
    explanation.append("")
    
    # Calculate complexity metrics
    total_transforms = len(encoding.rs_lengths) + len(encoding.hs_lengthss) + 1
    rh_space_size = len(encoding.rs_lengths) + sum(len(h) for h in encoding.hs_lengthss)
    
    explanation.append("Compile-time Optimization:")
    explanation.append("  - All three steps resolved at compile time")
    explanation.append("  - No runtime overhead for coordinate calculation")
    explanation.append("  - Entire pipeline inlined into kernel code")
    explanation.append(f"  - Template instantiation depth: ~{total_transforms}")
    explanation.append(f"  - Coordinate space complexity: {input_dims} → {rh_space_size} → {encoding.ndim_x}")
    explanation.append("  - Compiler can optimize across all three steps")
    explanation.append("")
    
    explanation.append("Memory Access Efficiency:")
    explanation.append("  - Coalesced memory access patterns")
    explanation.append("  - Adjacent threads access adjacent memory")
    explanation.append("  - Optimal cache utilization")
    explanation.append("  - Minimized memory bandwidth usage")
    explanation.append("")
    
    explanation.append("Thread Coordination:")
    explanation.append("  - Each thread gets unique coordinate mapping")
    explanation.append("  - No thread conflicts or race conditions")
    explanation.append("  - Scalable across different GPU architectures")
    explanation.append("  - Supports complex tensor operations")
    explanation.append("")
    
    explanation.append("PRACTICAL USAGE EXAMPLE:")
    explanation.append("")
    
    explanation.append("Complete kernel integration:")
    explanation.append("```cpp")
    explanation.append("// Kernel setup (compile time)")
    explanation.append("constexpr auto encoding = TileDistributionEncoding(...);")
    explanation.append("constexpr auto adaptor_result = _make_adaptor_encoding_for_tile_distribution(encoding);")
    explanation.append("constexpr auto tensor_adaptor = _construct_static_tensor_adaptor_from_encoding(adaptor_result[0]);")
    explanation.append("constexpr auto tensor_descriptor = make_tensor_descriptor_from_adaptor(tensor_adaptor, tensor_lengths);")
    explanation.append("")
    explanation.append("// Kernel execution (runtime)")
    explanation.append("__global__ void my_kernel(float* tensor_data) {")
    explanation.append(f"    // Get thread coordinates ({p_count} P dimensions)")
    p_coord_getters = [f'get_thread_id_{i}()' for i in range(p_count)]
    explanation.append(f"    auto p_coords = make_tuple({', '.join(p_coord_getters)});")
    explanation.append("")
    explanation.append(f"    // Iterate over tile ({y_count} Y dimensions)")
    y_tile_sizes = [f'tile_size_{i}' for i in range(y_count)]
    explanation.append(f"    for(auto y_coords : make_tile_iterator({', '.join(y_tile_sizes)})) {{")
    explanation.append("        // Combine coordinates")
    explanation.append("        auto py_coords = container_concat(p_coords, y_coords);")
    explanation.append("")
    explanation.append("        // Apply complete pipeline")
    explanation.append("        auto x_coords = tensor_adaptor.calculate_bottom_coordinate(py_coords);")
    explanation.append("        auto offset = tensor_descriptor.calculate_offset(x_coords);")
    explanation.append("")
    explanation.append("        // Access memory")
    explanation.append("        float value = tensor_data[offset];")
    explanation.append("        // ... process value ...")
    explanation.append("    }")
    explanation.append("}")
    explanation.append("```")
    explanation.append("")
    
    explanation.append("KEY PIPELINE INSIGHTS:")
    explanation.append("1. **Separation of Concerns**: Each step handles different aspect")
    explanation.append("2. **Compile-time Computation**: Zero runtime cost")
    explanation.append("3. **Type Safety**: C++ templates ensure correctness")
    explanation.append("4. **Performance**: Optimal memory access patterns")
    explanation.append("5. **Composability**: Pipeline supports complex transformations")
    explanation.append("6. **Hardware Mapping**: Direct connection to GPU architecture")
    explanation.append(f"7. **Scalability**: Handles {input_dims}→{encoding.ndim_x} dimensional transformations efficiently")
    
    return "\n".join(explanation)

def visualize_coordinate_transformation(encoding: TileDistributionEncoding, max_coords: int = 8, context: str = "default"):
    """Visualize coordinate transformation examples."""
    st.subheader("🎯 Coordinate Transformation Examples")
    st.write("**Shows actual coordinate mappings from the adaptor encoding function**")
    
    # Create example coordinate sets
    p_dims = len(encoding.ps_to_rhss_major)
    y_dims = len(encoding.ys_to_rhs_major)
    
    if p_dims == 0 and y_dims == 0:
        st.write("No P or Y dimensions to transform")
        return
    
    # Generate coordinate examples
    examples = []
    for p_val in range(min(max_coords // 2, 3)):
        for y_val in range(min(max_coords // 2, 3)):
            p_coords = [p_val] * p_dims if p_dims > 0 else []
            y_coords = [y_val] * y_dims if y_dims > 0 else []
            
            # Show actual mapping based on encoding structure
            rh_mappings = []
            
            # P coordinate mappings - these are the actual transformations Step 1 creates
            for i, (major_list, minor_list) in enumerate(zip(encoding.ps_to_rhss_major, encoding.ps_to_rhss_minor)):
                for j, (major, minor) in enumerate(zip(major_list, minor_list)):
                    if major == 0 and minor < len(encoding.rs_lengths):
                        rh_mappings.append(f"P{i}[{j}]({p_val})→R[{minor}]")
                    elif major > 0 and major-1 < len(encoding.hs_lengthss) and minor < len(encoding.hs_lengthss[major-1]):
                        rh_mappings.append(f"P{i}[{j}]({p_val})→H{major-1}[{minor}]")
            
            # Y coordinate mappings
            for i, (major, minor) in enumerate(zip(encoding.ys_to_rhs_major, encoding.ys_to_rhs_minor)):
                if major == 0 and minor < len(encoding.rs_lengths):
                    rh_mappings.append(f"Y{i}({y_val})→R[{minor}]")
                elif major > 0 and major-1 < len(encoding.hs_lengthss) and minor < len(encoding.hs_lengthss[major-1]):
                    rh_mappings.append(f"Y{i}({y_val})→H{major-1}[{minor}]")
            
            examples.append({
                'P_coords': p_coords,
                'Y_coords': y_coords,
                'Combined_Input': f"({', '.join(map(str, p_coords + y_coords))})",
                'RH_Transformations': '; '.join(rh_mappings),
                'Final_X_Dims': encoding.ndim_x
            })
    
    # Create DataFrame showing actual transformation results
    df = pd.DataFrame(examples)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Step 1 Function Results: Coordinate Mapping Table**")
        st.write("This shows what `_make_adaptor_encoding_for_tile_distribution` produces:")
        st.dataframe(df, use_container_width=True)
        
        st.write("**What this means:**")
        st.write("- Input: (P,Y) coordinates from GPU threads")
        st.write("- RH Transformations: How Step 1 maps coordinates to intermediate space")
        st.write(f"- Final Output: Will be {encoding.ndim_x} X coordinates after Step 2")
    
    with col2:
        st.write("**Transformation Flow Visualization**")
        st.write("Visual representation of the coordinate mapping process:")
        
        # Create a meaningful flow chart
        fig = go.Figure()
        
        # Input coordinates (left side)
        input_labels = [ex['Combined_Input'] for ex in examples]
        input_y = list(range(len(examples)))
        
        fig.add_trace(go.Scatter(
            x=[0] * len(input_y),
            y=input_y,
            mode='markers+text',
            text=input_labels,
            textposition="middle right",
            marker=dict(size=15, color='lightblue', symbol='circle'),
            name="Input (P,Y)",
            hovertemplate="<b>Input Coordinates</b><br>%{text}<extra></extra>"
        ))
        
        # Output space (right side)
        output_labels = [f"X({ex['Final_X_Dims']}D)" for ex in examples]
        
        fig.add_trace(go.Scatter(
            x=[3] * len(input_y),
            y=input_y,
            mode='markers+text',
            text=output_labels,
            textposition="middle left",
            marker=dict(size=15, color='lightgreen', symbol='square'),
            name="Output (X)",
            hovertemplate="<b>Output Space</b><br>%{text}<extra></extra>"
        ))
        
        # Add transformation arrows with labels
        for i in range(len(input_y)):
            fig.add_annotation(
                x=0.3, y=input_y[i],
                ax=2.7, ay=input_y[i],
                xref='x', yref='y',
                axref='x', ayref='y',
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=3,
                arrowcolor='orange',
                text="RH Transform",
                showarrow=True,
                textangle=0,
                font=dict(size=10)
            )
        
        fig.update_layout(
            title="Coordinate Transformation: (P,Y) → RH → (X)",
            xaxis=dict(range=[-0.5, 3.5], showticklabels=False, title="Transformation Steps"),
            yaxis=dict(showticklabels=False, title="Example Coordinates"),
            showlegend=True,
            height=400,
            annotations=[
                dict(x=0, y=len(examples), text="Step 1 Input", showarrow=False, font=dict(size=12, color="blue")),
                dict(x=3, y=len(examples), text="Step 2 Output", showarrow=False, font=dict(size=12, color="green"))
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"coord_transform_chart_{context}")

def visualize_memory_layout(encoding: TileDistributionEncoding, context: str = "default"):
    """Visualize tensor memory layout and access patterns."""
    st.subheader("🧠 Memory Layout from Step 3 Function")
    st.write("**Shows actual memory layout that `make_tensor_descriptor_from_adaptor` creates**")
    
    # Calculate adaptive tensor shape based on encoding
    base_sizes = [32, 64, 128, 256]
    tensor_shape = []
    for i in range(encoding.ndim_x):
        # Make size depend on encoding complexity
        complexity = len(encoding.rs_lengths) + sum(len(h) for h in encoding.hs_lengthss)
        base_size = base_sizes[min(i, len(base_sizes)-1)]
        adapted_size = max(16, base_size // max(1, complexity))
        tensor_shape.append(adapted_size)
    
    # Calculate actual strides (what Step 3 function computes)
    strides = [1]
    for i in range(len(tensor_shape) - 1, 0, -1):
        strides.insert(0, strides[0] * tensor_shape[i])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Step 3 Function Output: Tensor Descriptor Properties**")
        st.json({
            "tensor_shape": tensor_shape,
            "memory_strides": strides,
            "total_elements": int(np.prod(tensor_shape)),
            "memory_bytes": int(np.prod(tensor_shape) * 4),
            "access_pattern": "row-major"
        })
        
        st.write("**Actual Memory Access Formula:**")
        st.code(f"""
// What tensor_descriptor.calculate_offset() does:
offset = 0;""" + 
"\n".join([f"offset += x{i} * {stride};" for i, stride in enumerate(strides)]) + f"""

// Example for coordinates [1, 2, ...]:
offset = {" + ".join([f"1*{stride}" for stride in strides])}
       = {sum(strides)}""", language="cpp")
    
    with col2:
        st.write("**Memory Offset Pattern Visualization**")
        st.write("This shows how coordinates map to actual memory addresses:")
        
        if encoding.ndim_x == 1:
            # 1D visualization - show actual offsets
            coords = list(range(min(tensor_shape[0], 20)))
            offsets = [i * strides[0] for i in coords]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=coords,
                y=offsets,
                mode='markers+lines',
                name='Memory Offsets',
                marker=dict(size=8, color='blue'),
                line=dict(width=2),
                hovertemplate="<b>Coordinate:</b> %{x}<br><b>Memory Offset:</b> %{y}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"1D Memory Access Pattern (stride={strides[0]})",
                xaxis_title="X0 coordinate",
                yaxis_title="Memory offset",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True, key=f"memory_1d_chart_{context}")
            
        elif encoding.ndim_x >= 2:
            # 2D visualization - show memory layout as heatmap
            rows, cols = min(tensor_shape[0], 12), min(tensor_shape[1], 12)
            
            memory_grid = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    offset = i * strides[0] + j * strides[1] if len(strides) > 1 else i * strides[0]
                    memory_grid[i, j] = offset
            
            fig = px.imshow(
                memory_grid,
                labels=dict(x="X1 coordinate", y="X0 coordinate", color="Memory Offset"),
                title=f"2D Memory Layout (strides: [{strides[0]}, {strides[1] if len(strides) > 1 else 0}])",
                color_continuous_scale="Viridis",
                aspect="auto"
            )
            
            # Add text annotations for small grids
            if rows <= 8 and cols <= 8:
                annotations = []
                for i in range(rows):
                    for j in range(cols):
                        annotations.append(
                            dict(
                                x=j, y=i,
                                text=str(int(memory_grid[i, j])),
                                showarrow=False,
                                font=dict(color="white" if memory_grid[i, j] > np.max(memory_grid)/2 else "black")
                            )
                        )
                fig.update_layout(annotations=annotations)
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key=f"memory_2d_chart_{context}")

def visualize_transformation_pipeline(encoding: TileDistributionEncoding, context: str = "default"):
    """Visualize the complete transformation pipeline."""
    st.subheader("🔄 Complete 3-Step Function Pipeline")
    st.write("**Shows the data flow through all three functions**")
    
    # Calculate actual dimensions at each stage
    p_count = len(encoding.ps_to_rhss_major)
    y_count = len(encoding.ys_to_rhs_major)
    input_dims = p_count + y_count
    
    r_count = len(encoding.rs_lengths)
    h_total = sum(len(h) for h in encoding.hs_lengthss)
    intermediate_dims = r_count + h_total
    
    # Create realistic pipeline data
    pipeline_data = [
        {
            'stage': 'Input\n(P,Y)',
            'function': 'GPU provides',
            'dimensions': input_dims,
            'example': f'[p0..p{p_count-1}, y0..y{y_count-1}]' if input_dims > 0 else '[empty]',
            'description': f'{p_count}P + {y_count}Y coords'
        },
        {
            'stage': 'Step 1\nRH Space',
            'function': 'make_adaptor_encoding',
            'dimensions': intermediate_dims,
            'example': f'[r0..r{r_count-1}, h0..h{h_total-1}]' if intermediate_dims > 0 else '[transformed]',
            'description': f'{r_count}R + {h_total}H space'
        },
        {
            'stage': 'Step 2\nX Coords',
            'function': 'construct_tensor_adaptor',
            'dimensions': encoding.ndim_x,
            'example': f'[x0..x{encoding.ndim_x-1}]' if encoding.ndim_x > 0 else '[empty]',
            'description': f'{encoding.ndim_x}X final coords'
        },
        {
            'stage': 'Step 3\nMemory',
            'function': 'make_tensor_descriptor',
            'dimensions': 1,
            'example': 'memory_offset',
            'description': '1 memory address'
        }
    ]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Pipeline Stage Details:**")
        df_pipeline = pd.DataFrame(pipeline_data)
        st.dataframe(df_pipeline, use_container_width=True)
        
        st.write("**Function Complexity:**")
        st.write(f"- Step 1: Creates {len(encoding.rs_lengths)} replicate + {len(encoding.hs_lengthss)} unmerge transforms")
        st.write(f"- Step 2: Instantiates ~{len(encoding.rs_lengths) + len(encoding.hs_lengthss) + 1} transformation objects")
        st.write(f"- Step 3: Calculates strides for {encoding.ndim_x}D tensor")
    
    with col2:
        st.write("**Dimension Flow Through Pipeline:**")
        
        # Create dimension flow chart
        fig = go.Figure()
        
        stages = [data['stage'] for data in pipeline_data]
        dimensions = [data['dimensions'] for data in pipeline_data]
        functions = [data['function'] for data in pipeline_data]
        
        # Bar chart showing dimension count at each stage
        colors = ['lightblue', 'orange', 'lightgreen', 'red']
        
        fig.add_trace(go.Bar(
            x=stages,
            y=dimensions,
            text=[f"{dim}D" for dim in dimensions],
            textposition='outside',
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Dimensions: %{y}<br>Function: %{customdata}<extra></extra>",
            customdata=functions
        ))
        
        # Add arrows between bars
        for i in range(len(stages) - 1):
            fig.add_annotation(
                x=i + 0.35, y=max(dimensions) * 0.7,
                ax=i + 0.65, ay=max(dimensions) * 0.7,
                xref='x', yref='y',
                axref='x', ayref='y',
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=3,
                arrowcolor='gray'
            )
        
        fig.update_layout(
            title="Dimension Transformation Flow",
            yaxis_title="Number of Dimensions",
            xaxis_title="Pipeline Stage",
            height=450,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"pipeline_flow_chart_{context}")

def visualize_rh_space_structure(encoding: TileDistributionEncoding, context: str = "default"):
    """Visualize the RH space structure and mappings."""
    st.subheader("🗂️ RH Space Structure from Step 1")
    st.write("**Shows the intermediate coordinate space that Step 1 creates**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**RH Dimensions Created by Step 1:**")
        
        # Create RH space data
        rh_data = []
        rh_index = 0
        
        # R dimensions
        for i, r_len in enumerate(encoding.rs_lengths):
            rh_data.append({
                'RH_Index': rh_index,
                'Type': 'R',
                'Name': f'R[{i}]',
                'Length': r_len,
                'Purpose': f'Replication ({r_len}x)',
                'Transform': f'replicate<{r_len}>'
            })
            rh_index += 1
        
        # H dimensions
        for i, h_lengths in enumerate(encoding.hs_lengthss):
            for j, h_len in enumerate(h_lengths):
                rh_data.append({
                    'RH_Index': rh_index,
                    'Type': 'H',
                    'Name': f'H{i}[{j}]',
                    'Length': h_len,
                    'Purpose': f'Hidden group {i}, component {j}',
                    'Transform': f'unmerge<{",".join(map(str, h_lengths))}>'
                })
                rh_index += 1
        
        if rh_data:
            df_rh = pd.DataFrame(rh_data)
            st.dataframe(df_rh, use_container_width=True)
            
            # Bar chart of RH dimensions
            fig = px.bar(
                df_rh, 
                x='Name', 
                y='Length', 
                color='Type',
                title="RH Space Dimension Lengths",
                color_discrete_map={'R': 'lightblue', 'H': 'orange'},
                hover_data=['Purpose', 'Transform']
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True, key=f"rh_space_chart_{context}")
        else:
            st.write("No RH dimensions (direct mapping)")
    
    with col2:
        st.write("**Coordinate Mappings (P,Y → RH):**")
        
        # Create mapping data with actual coordinate flow
        mapping_data = []
        
        # P mappings
        for i, (major_list, minor_list) in enumerate(zip(encoding.ps_to_rhss_major, encoding.ps_to_rhss_minor)):
            for j, (major, minor) in enumerate(zip(major_list, minor_list)):
                source = f'P{i}[{j}]'
                if major == 0:
                    target = f"R[{minor}]"
                    target_len = encoding.rs_lengths[minor] if minor < len(encoding.rs_lengths) else "Invalid"
                else:
                    target = f"H{major-1}[{minor}]"
                    if major-1 < len(encoding.hs_lengthss) and minor < len(encoding.hs_lengthss[major-1]):
                        target_len = encoding.hs_lengthss[major-1][minor]
                    else:
                        target_len = "Invalid"
                
                mapping_data.append({
                    'Source': source,
                    'Target': target,
                    'Target_Size': target_len,
                    'Mapping': f'{source} → {target}',
                    'Type': 'P coordinate'
                })
        
        # Y mappings
        for i, (major, minor) in enumerate(zip(encoding.ys_to_rhs_major, encoding.ys_to_rhs_minor)):
            source = f'Y{i}'
            if major == 0:
                target = f"R[{minor}]"
                target_len = encoding.rs_lengths[minor] if minor < len(encoding.rs_lengths) else "Invalid"
            else:
                target = f"H{major-1}[{minor}]"
                if major-1 < len(encoding.hs_lengthss) and minor < len(encoding.hs_lengthss[major-1]):
                    target_len = encoding.hs_lengthss[major-1][minor]
                else:
                    target_len = "Invalid"
            
            mapping_data.append({
                'Source': source,
                'Target': target,
                'Target_Size': target_len,
                'Mapping': f'{source} → {target}',
                'Type': 'Y coordinate'
            })
        
        if mapping_data:
            df_mappings = pd.DataFrame(mapping_data)
            st.dataframe(df_mappings, use_container_width=True)
            
            # Network diagram showing mappings
            if len(mapping_data) <= 10:  # Only show network for simple cases
                fig = go.Figure()
                
                sources = df_mappings['Source'].unique()
                targets = df_mappings['Target'].unique()
                
                # Source nodes (left)
                fig.add_trace(go.Scatter(
                    x=[0] * len(sources),
                    y=list(range(len(sources))),
                    mode='markers+text',
                    text=sources,
                    textposition="middle right",
                    marker=dict(size=12, color='lightblue'),
                    name="Input (P,Y)",
                    hovertemplate="<b>Source:</b> %{text}<extra></extra>"
                ))
                
                # Target nodes (right)
                fig.add_trace(go.Scatter(
                    x=[2] * len(targets),
                    y=list(range(len(targets))),
                    mode='markers+text',
                    text=targets,
                    textposition="middle left",
                    marker=dict(size=12, color='orange'),
                    name="RH Space",
                    hovertemplate="<b>Target:</b> %{text}<extra></extra>"
                ))
                
                # Connection lines
                for _, row in df_mappings.iterrows():
                    source_idx = list(sources).index(row['Source'])
                    target_idx = list(targets).index(row['Target'])
                    
                    fig.add_trace(go.Scatter(
                        x=[0.1, 1.9],
                        y=[source_idx, target_idx],
                        mode='lines',
                        line=dict(width=2, color='gray'),
                        showlegend=False,
                        hovertemplate=f"<b>{row['Mapping']}</b><br>Size: {row['Target_Size']}<extra></extra>"
                    ))
                
                fig.update_layout(
                    title="Coordinate Mapping Network",
                    xaxis=dict(range=[-0.5, 2.5], showticklabels=False),
                    yaxis=dict(showticklabels=False),
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"mapping_network_chart_{context}")
        else:
            st.write("No coordinate mappings defined")

def show_step_by_step_analysis(encoding: TileDistributionEncoding):
    """Show detailed step-by-step analysis similar to spans_app.py"""
    
    st.header("🔍 Step-by-Step Adaptor Transformation Analysis")
    
    # Create tabs for different aspects of analysis like spans_app.py
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏗️ Step 1: make_adaptor_encoding", 
        "🔧 Step 2: construct_tensor_adaptor", 
        "📐 Step 3: make_tensor_descriptor",
        "🔄 Complete Pipeline",
        "🧮 Interactive Testing",
        "📊 Encoding Structure",
        "🎯 Function Visualizations"
    ])
    
    with tab1:
        st.subheader("🏗️ _make_adaptor_encoding_for_tile_distribution")
        st.markdown("**Step 1 of 3: Convert tile distribution encoding into transformation blueprint**")
        
        explanation1 = generate_make_adaptor_encoding_explanation(encoding)
        st.code(explanation1, language="text")
        
        # Add RH space visualization
        st.markdown("---")
        visualize_rh_space_structure(encoding, context="step1_tab")
        
        # Show intermediate variables
        st.markdown("---")
        st.subheader("🔍 Intermediate Variables Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Function Input:**")
            st.write(f"- TileDistributionEncoding object")
            st.write(f"- ndim_x: {encoding.ndim_x}")
            st.write(f"- ndim_y: {len(encoding.ys_to_rhs_major)}")
            st.write(f"- ndim_p: {len(encoding.ps_to_rhss_major)}")
            st.write(f"- ndim_r: {len(encoding.rs_lengths)}")
            
            st.write("**RH Space Analysis:**")
            total_rh = len(encoding.rs_lengths) + len(encoding.hs_lengthss)
            st.write(f"- Total RH dimensions: {total_rh}")
            st.write(f"- R sequence count: {len(encoding.rs_lengths)}")
            st.write(f"- H sequence count: {len(encoding.hs_lengthss)}")
        
        with col2:
            st.write("**Function Output (Conceptual):**")
            st.write("- ps_ys_to_xs_adaptor_encoding")
            st.write("- ys_to_d_adaptor_encoding") 
            st.write("- d_length")
            st.write("- rh_major_minor_to_hidden_ids")
            
            st.write("**Key Transformations Created:**")
            transform_count = 0
            for i, length in enumerate(encoding.rs_lengths):
                st.write(f"- Replicate<{length}> for R{i}")
                transform_count += 1
            for i, lengths in enumerate(encoding.hs_lengthss):
                lengths_str = ",".join(map(str, lengths))
                st.write(f"- Unmerge<{lengths_str}> for H{i}")
                transform_count += 1
            st.write(f"- Merge operations")
            st.write(f"**Total transforms: ~{transform_count + 2}**")
    
    with tab2:
        st.subheader("🔧 _construct_static_tensor_adaptor_from_encoding")
        st.markdown("**Step 2 of 3: Turn transformation blueprint into functional coordinate transformer**")
        
        explanation2 = generate_construct_tensor_adaptor_explanation(encoding)
        st.code(explanation2, language="text")
        
        # Add coordinate transformation visualization
        st.markdown("---")
        visualize_coordinate_transformation(encoding, context="step2_tab")
        
        # Show adaptor construction details
        st.markdown("---")
        st.subheader("🔍 Adaptor Construction Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Function Input:**")
            st.write("- ps_ys_to_xs_adaptor_encoding (from Step 1)")
            st.write("- Transformation chain string")
            st.write("- Input/output dimension specifications")
            
            st.write("**Parsing Process:**")
            st.write("1. Parse transformation string")
            st.write("2. Validate dimension compatibility")
            st.write("3. Create transformation objects")
            st.write("4. Chain transformations together")
            
        with col2:
            st.write("**Function Output:**")
            st.write("- TensorAdaptor object")
            input_dims = len(encoding.ps_to_rhss_major) + len(encoding.ys_to_rhs_major)
            st.write(f"- Top dimensions: {input_dims} (P+Y)")
            st.write(f"- Bottom dimensions: {encoding.ndim_x} (X)")
            
            st.write("**Coordinate Transformation:**")
            st.code(f"""
Input: [{', '.join([f'P{i}' for i in range(len(encoding.ps_to_rhss_major))] + [f'Y{i}' for i in range(len(encoding.ys_to_rhs_major))])}]
  ↓ (Apply transformation chain)
Output: [{', '.join([f'X{i}' for i in range(encoding.ndim_x)])}]
            """, language="text")
    
    with tab3:
        st.subheader("📐 make_tensor_descriptor_from_adaptor")
        st.markdown("**Step 3 of 3: Define final tensor memory layout and access patterns**")
        
        explanation3 = generate_make_tensor_descriptor_explanation(encoding)
        st.code(explanation3, language="text")
        
        # Add memory layout visualization
        st.markdown("---")
        visualize_memory_layout(encoding, context="step3_tab")
        
        # Show tensor descriptor details
        st.markdown("---")
        st.subheader("🔍 Tensor Descriptor Construction")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Function Input:**")
            st.write("- TensorAdaptor (from Step 2)")
            st.write("- Tensor shape/lengths")
            st.write(f"- {encoding.ndim_x} dimensional tensor")
            
            # Example tensor shape
            example_shape = [64, 128, 256][:encoding.ndim_x]
            if len(example_shape) < encoding.ndim_x:
                example_shape.extend([32] * (encoding.ndim_x - len(example_shape)))
            
            st.write("**Example Tensor Shape:**")
            for i, length in enumerate(example_shape):
                st.write(f"- X{i}: {length}")
            st.write(f"**Total elements:** {np.prod(example_shape):,}")
            
        with col2:
            st.write("**Function Output:**")
            st.write("- TensorDescriptor object")
            st.write("- Memory layout information")
            st.write("- Stride calculations")
            st.write("- Coordinate-to-offset mapping")
            
            st.write("**Memory Access Pattern:**")
            st.code(f"""
// Usage in kernel:
auto coord = make_tuple(p0, p1, ..., y0, y1, ...);
auto x_coord = adaptor.calculate_coordinate(coord);
auto offset = descriptor.calculate_offset(x_coord);
float value = tensor_data[offset];
            """, language="cpp")
    
    with tab4:
        st.subheader("🔄 Complete Transformation Pipeline")
        st.markdown("**All three functions working together:**")
        
        # Generate complete pipeline explanation
        pipeline_explanation = generate_complete_pipeline_explanation(encoding)
        st.code(pipeline_explanation, language="text")
        
        # Add pipeline visualization
        st.markdown("---")
        visualize_transformation_pipeline(encoding, context="pipeline_tab")
        
        # Visual pipeline
        st.markdown("---")
        st.subheader("📊 Visual Pipeline Flow")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🏗️ Step 1 Output:**")
            st.write("Adaptor Encoding")
            st.write("- Transformation string")
            st.write("- Dimension mappings")
            st.write("- Hidden ID mappings")
            st.write("↓")
        
        with col2:
            st.write("**🔧 Step 2 Output:**")
            st.write("Tensor Adaptor")
            st.write("- Functional transformer")
            st.write("- Coordinate calculator")
            st.write("- (P,Y) → X mapping")
            st.write("↓")
        
        with col3:
            st.write("**📐 Step 3 Output:**")
            st.write("Tensor Descriptor")
            st.write("- Memory layout")
            st.write("- Stride information")
            st.write("- Offset calculator")
            st.write("✅ Ready for kernel")
        
        # Example usage
        st.markdown("---")
        st.subheader("💻 Complete Usage Example")
        
        st.code(f"""
// Step 1: Create adaptor encoding
auto encoding_result = _make_adaptor_encoding_for_tile_distribution(tile_encoding);
auto ps_ys_to_xs_encoding = encoding_result[0];

// Step 2: Construct tensor adaptor
auto tensor_adaptor = _construct_static_tensor_adaptor_from_encoding(ps_ys_to_xs_encoding);

// Step 3: Create tensor descriptor
auto tensor_lengths = array<index_t, {encoding.ndim_x}>{{...}};
auto tensor_descriptor = make_tensor_descriptor_from_adaptor(tensor_adaptor, tensor_lengths);

// Usage in kernel:
auto thread_coords = make_tuple({', '.join([f'p{i}' for i in range(len(encoding.ps_to_rhss_major))])});
auto tile_coords = make_tuple({', '.join([f'y{i}' for i in range(len(encoding.ys_to_rhs_major))])});
auto combined_coords = container_concat(thread_coords, tile_coords);

auto tensor_coords = tensor_adaptor.calculate_coordinate(combined_coords);
auto memory_offset = tensor_descriptor.calculate_offset(tensor_coords);

// Access memory
float value = tensor_data[memory_offset];
        """, language="cpp")
    
    with tab5:
        st.subheader("🧮 Interactive Function Testing")
        st.write("Test each function step with your coordinates:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Input Coordinates:**")
            
            # P coordinates
            p_coords = []
            for i in range(len(encoding.ps_to_rhss_major)):
                val = st.number_input(f"P{i} (thread/warp index)", value=0, min_value=0, key=f"test_p_coord_{i}")
                p_coords.append(val)
            
            # Y coordinates  
            y_coords = []
            for i in range(len(encoding.ys_to_rhs_major)):
                val = st.number_input(f"Y{i} (local tile index)", value=0, min_value=0, key=f"test_y_coord_{i}")
                y_coords.append(val)
            
            # Tensor dimensions
            st.write("**Tensor Shape:**")
            tensor_shape = []
            for i in range(encoding.ndim_x):
                val = st.number_input(f"X{i} dimension size", value=64, min_value=1, key=f"tensor_dim_{i}")
                tensor_shape.append(val)
        
        with col2:
            st.write("**Step-by-Step Function Results:**")
            
            if st.button("🔄 Test Complete Pipeline"):
                st.write("**Step 1: _make_adaptor_encoding_for_tile_distribution**")
                st.write("✅ Creates transformation blueprint")
                st.write(f"- Input: TileDistributionEncoding")
                st.write(f"- Output: Adaptor encoding with ~{len(encoding.rs_lengths) + len(encoding.hs_lengthss) + 2} transforms")
                
                st.write("**Step 2: _construct_static_tensor_adaptor_from_encoding**")
                st.write("✅ Creates functional transformer")
                input_dims = len(encoding.ps_to_rhss_major) + len(encoding.ys_to_rhs_major)
                st.write(f"- Input dimensions: {input_dims} (P+Y)")
                st.write(f"- Output dimensions: {encoding.ndim_x} (X)")
                
                st.write("**Step 3: make_tensor_descriptor_from_adaptor**")
                st.write("✅ Creates memory layout descriptor")
                st.write(f"- Tensor shape: {tensor_shape}")
                st.write(f"- Total elements: {np.prod(tensor_shape):,}")
                st.write(f"- Memory size: {np.prod(tensor_shape) * 4:,} bytes (float32)")
                
                st.write("**Complete Transformation (Conceptual):**")
                combined_coords = p_coords + y_coords
                st.write(f"Input: {combined_coords}")
                st.write("↓ (Apply adaptor)")
                st.write(f"Tensor coords: [calculated X0, X1, ...]")
                st.write("↓ (Apply descriptor)")
                st.write(f"Memory offset: calculated_offset")
                
                st.success(f"✅ Pipeline complete! Ready to access tensor_data[calculated_offset]")
    
    with tab6:
        st.subheader("📊 Encoding Structure Visualization")
        
        # Reuse the existing visualization from the previous implementation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**P Space (Partition/Thread)**")
            st.write("Hardware-provided coordinates:")
            for i in range(len(encoding.ps_to_rhss_major)):
                st.write(f"**P{i}**")
                major_list = encoding.ps_to_rhss_major[i]
                minor_list = encoding.ps_to_rhss_minor[i]
                for j, (major, minor) in enumerate(zip(major_list, minor_list)):
                    if major == 0:
                        target = f"R[{minor}]"
                        value = encoding.rs_lengths[minor] if minor < len(encoding.rs_lengths) else "N/A"
                    else:
                        target = f"H{major-1}[{minor}]"
                        value = encoding.hs_lengthss[major-1][minor] if major-1 < len(encoding.hs_lengthss) and minor < len(encoding.hs_lengthss[major-1]) else "N/A"
                    st.write(f"  [{j}] ↓ {target} ({value})")
                st.write("")
        
        with col2:
            st.write("**RH Space (Intermediate)**")
            st.write("Transformation coordinate space:")
            
            st.write("**R (Replication):**")
            for i, length in enumerate(encoding.rs_lengths):
                st.write(f"R[{i}] = {length}")
            
            st.write("**H (Hidden):**")
            for i, lengths in enumerate(encoding.hs_lengthss):
                st.write(f"H{i}: {lengths}")
                for j, length in enumerate(lengths):
                    st.write(f"  H{i}[{j}] = {length}")
        
        with col3:
            st.write("**Y Space (Local Tile)**")
            st.write("Tile iteration coordinates:")
            for i, (major, minor) in enumerate(zip(encoding.ys_to_rhs_major, encoding.ys_to_rhs_minor)):
                if major == 0:
                    target = f"R[{minor}]"
                    value = encoding.rs_lengths[minor] if minor < len(encoding.rs_lengths) else "N/A"
                else:
                    target = f"H{major-1}[{minor}]"
                    value = encoding.hs_lengthss[major-1][minor] if major-1 < len(encoding.hs_lengthss) and minor < len(encoding.hs_lengthss[major-1]) else "N/A"
                st.write(f"**Y{i}**")
                st.write(f"  ↓ {target} ({value})")
                st.write("")
    
    with tab7:
        st.subheader("🎯 Interactive Function Result Visualizations")
        st.markdown("**Live visualizations of transformation function results**")
        
        # All visualizations in one tab with unique contexts
        visualize_coordinate_transformation(encoding, context="combined_tab_coord")
        st.markdown("---")
        visualize_memory_layout(encoding, context="combined_tab_memory")
        st.markdown("---")
        visualize_transformation_pipeline(encoding, context="combined_tab_pipeline")
        st.markdown("---")
        visualize_rh_space_structure(encoding, context="combined_tab_rh")

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