"""
Streamlit application for visualizing tile_distribution concepts.

This application provides an interactive interface for exploring and visualizing
tile_distribution_encoding structures from the Composable Kernels library.
"""

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import re
import json
from typing import Dict, List, Tuple, Any, Optional
import plotly.express as px

# Import local modules
from parser import TileDistributionParser, debug_indexing_relationships
from tiler_pedantic import TileDistributionPedantic as TileDistribution
import visualizer
from test_visualization import create_indexing_visualization
from visualizer import visualize_hierarchical_tiles
from visualizer import visualize_y_space_structure
from examples import get_examples, get_default_variables

# Set page config
st.set_page_config(
    page_title="Tile Distribution Visualizer",
    page_icon="📊",
    layout="wide",
)

# Initialize session state
if 'encoding' not in st.session_state:
    st.session_state.encoding = None
if 'variables' not in st.session_state:
    st.session_state.variables = {}
if 'parsed_variables' not in st.session_state:
    st.session_state.parsed_variables = []
if 'tile_distribution' not in st.session_state:
    st.session_state.tile_distribution = None
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'show_dimension_arrows' not in st.session_state:
    st.session_state.show_dimension_arrows = False

def main():
    """Main function for the Streamlit app."""
    st.title("Tile Distribution Visualizer")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Input Options")
        
        # Remove the file loading option and simplify to a single input method
        st.subheader("Code Input")
        
        # Get examples and default variables
        examples = get_examples()
        
        # Initialize the selected example in session state if not present
        if 'selected_example' not in st.session_state:
            # Default to the first example, but ensure it's a valid choice
            example_keys = list(examples.keys())
            if example_keys:
                st.session_state.selected_example = example_keys[0]
            else:
                st.session_state.selected_example = ""
        
        example_keys = list(examples.keys())
        try:
            # Find the index of the selected example or default to 0
            selected_index = 0
            if st.session_state.selected_example in example_keys:
                selected_index = example_keys.index(st.session_state.selected_example)
            
            selected_example = st.selectbox(
                "Example Template:", 
                example_keys, 
                index=selected_index,
                key="example_selectbox"
            )
            # Update the session state
            st.session_state.selected_example = selected_example
        except Exception as e:
            # Safer fallback if there's an issue with the selectbox
            if example_keys:
                selected_example = example_keys[0]
                st.session_state.selected_example = selected_example
            else:
                st.error(f"No examples available: {str(e)}")
                selected_example = ""
                st.session_state.selected_example = ""
        
        # Get the selected example code
        cpp_code = examples[selected_example]
        
        # Allow editing the code in a text area
        edited_code = st.text_area(
            "Edit Code:",
            value=cpp_code,
            height=300,
            key="code_editor"
        )
        
        # Parse button
        if st.button("Parse Code"):
            parser = TileDistributionParser()
            encoding = parser.parse_tile_distribution_encoding(edited_code)
            
            # Get default variables for this example
            variables = get_default_variables(selected_example)
            
            if encoding:
                # Also extract any template variables that might be in the code
                detected_variables = parser.extract_template_variables(edited_code)
                
                # Merge the extracted variables with the default ones, giving priority to extracted values
                for var_name, var_value in detected_variables.items():
                    if var_name not in variables:
                        variables[var_name] = var_value
                
                st.session_state.encoding = encoding
                st.session_state.variables = variables
                st.session_state.cpp_code = edited_code  # Save the code for visualization
                
                # Extract variables from encoding for sliders
                if encoding and "variable_names" in encoding:
                    var_list = encoding["variable_names"]
                else:
                    var_list = []
                    
                st.session_state.parsed_variables = list(set(
                    var_list + list(variables.keys())
                ))
                
                st.success("Code parsed successfully!")
            else:
                st.error("Failed to parse the code. Check the syntax.")
        
        # Variable sliders (shown only if encoding is parsed)
        if st.session_state.encoding is not None:
            display_variable_controls()
        
        # Add debug mode toggle
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        
        # Add dimension arrows toggle
        st.session_state.show_dimension_arrows = st.checkbox("Show Dimension Arrows", value=st.session_state.show_dimension_arrows)
    
    # Main content
    if st.session_state.encoding is not None:
        # Display parsed structure
        st.header("Parsed Tile Distribution Encoding")
        
        # If in debug mode, show raw JSON with format cleanup
        if st.session_state.debug_mode:
            try:
                # Clean up the JSON representation for better display
                display_encoding = {}
                for key, value in st.session_state.encoding.items():
                    # Convert lists of lists to more readable format
                    if isinstance(value, list):
                        if all(isinstance(item, list) for item in value if isinstance(item, list)):
                            display_encoding[key] = [
                                f"[{', '.join(str(v) for v in item)}]" if isinstance(item, list) else item
                                for item in value
                            ]
                        else:
                            display_encoding[key] = value
                    else:
                        display_encoding[key] = value
                
                st.write("### Raw Parsed Data:")
                st.json(st.session_state.encoding)
                
                # Generate indexing debug information
                indexing_debug = debug_indexing_relationships(st.session_state.encoding, st.session_state.variables)
                
                st.write("### Indexing Relationships:")
                st.json(indexing_debug)
                
                # Add a more readable version of the mapping
                st.write("### Mapping Summary (with Symbolic Values):")
                mapping_summary = {}
                for p_key, mappings in indexing_debug["IndexMapping"].items():
                    if p_key.startswith("P"):
                        p_mappings = []
                        for m in mappings:
                            symbolic = m.get("SymbolicValue")
                            value = m.get("Value")
                            target = m.get("Target")
                            idx = m.get("MinorIndex")
                            p_mappings.append(f"{target}[{idx}] = {symbolic} ({value})")
                        mapping_summary[p_key] = p_mappings
                
                for y_key, mappings in indexing_debug["IndexMapping"].items():
                    if y_key.startswith("Y"):
                        y_mappings = []
                        for m in mappings:
                            symbolic = m.get("SymbolicValue")
                            value = m.get("Value")
                            target = m.get("Target")
                            idx = m.get("MinorIndex")
                            y_mappings.append(f"{target}[{idx}] = {symbolic} ({value})")
                        mapping_summary[y_key] = y_mappings
                
                st.json(mapping_summary)
                
                st.write("### Parsed Variable Names:")
                st.write(st.session_state.parsed_variables)
                
                st.write("### Current Variable Values:")
                st.json(st.session_state.variables)
            except Exception as e:
                st.error(f"Error in debug mode: {str(e)}")
                st.exception(e)
        else:
            with st.expander("Show Raw Encoding Data"):
                st.json(st.session_state.encoding)
        
        # Calculate distribution with current variables
        try:
            calculate_distribution()
            if st.session_state.tile_distribution is None:
                st.error("Failed to calculate tile distribution. Check your encoding and variables.")
                return
        except Exception as e:
            st.error(f"Error calculating distribution: {str(e)}")
            return
        
        # Add indexing visualization
        st.subheader("Indexing Relationships Visualization")
        try:
            # Generate indexing debug information
            indexing_debug = debug_indexing_relationships(st.session_state.encoding, st.session_state.variables)
            
            # Create and display the visualization
            indexing_fig = create_indexing_visualization(indexing_debug, st.session_state.variables)
            st.pyplot(indexing_fig)
            
            # Add some explanatory text
            st.markdown("""
            **This visualization shows:**
            - **Top Ids**: P dimensions (thread mapping) and Y dimensions (spatial output mapping)
            - **Hidden Ids**: Individual elements from H0 and H1 sequences with their symbolic names and values
            - **Bottom Ids**: R sequence and H sequences that organize the computation dimensions
            """)
        except Exception as e:
            st.error(f"Error creating indexing visualization: {str(e)}")
            if st.session_state.debug_mode:
                st.exception(e)
            
        # Debug information (when debug mode is enabled)
        if st.session_state.debug_mode:
            st.subheader("Debug Information")
            try:
                viz_data = st.session_state.tile_distribution.get_visualization_data()
                
                # Show the indexing relationships in a more visual way
                indexing_debug = debug_indexing_relationships(
                    st.session_state.encoding, 
                    st.session_state.variables
                )
                
                # Create a tabbed view
                tabs = st.tabs(
                    ["Index Values", "P Mappings", "Y Mappings"]
                )
                
                with tabs[0]:
                    # Display the values of indices
                    st.write("#### Index Values")
                    st.json(indexing_debug["MinorIndices"])
                
                with tabs[1]:
                    # Display P mappings in a more readable format
                    st.write("#### P Dimension Mappings")
                    p_mappings = {}
                    for p_key, mappings in indexing_debug["IndexMapping"].items():
                        if p_key.startswith("P"):
                            formatted_mappings = []
                            for m in mappings:
                                formatted_mappings.append(
                                    f"{p_key} → {m['Target']}[{m['MinorIndex']}] = {m['Value']}"
                                )
                            p_mappings[p_key] = formatted_mappings
                    st.json(p_mappings)
                
                with tabs[2]:
                    # Display Y mappings in a more readable format
                    st.write("#### Y Dimension Mappings")
                    y_mappings = {}
                    for y_key, mappings in indexing_debug["IndexMapping"].items():
                        if y_key.startswith("Y"):
                            formatted_mappings = []
                            for m in mappings:
                                formatted_mappings.append(
                                    f"{y_key} → {m['Target']}[{m['MinorIndex']}] = {m['Value']}"
                                )
                            y_mappings[y_key] = formatted_mappings
                    st.json(y_mappings)
            except Exception as e:
                st.error(f"Error in debug information: {str(e)}")
                if st.session_state.debug_mode:
                    st.exception(e)
                
        # Hierarchical tile visualization
        st.subheader("Hierarchical Tile Structure")

        # Add radio button for view selection
        view_type = st.radio(
            "View Mode:",
            ["Thread View", "Data View"],
            horizontal=True,
            key="view_type_selector"
        )

        try:
            if st.session_state.tile_distribution is not None:
                viz_data = st.session_state.tile_distribution.get_visualization_data()
                hierarchical_structure = viz_data.get("hierarchical_structure", {})

                # --- ALWAYS VISIBLE HIERARCHICAL INFO ---
                if hierarchical_structure:
                    # Display summary information (Moved from debug h_tabs[0])
                    st.write("#### Tile Structure Overview")
                    
                    thread_per_warp = hierarchical_structure.get('ThreadPerWarp', [])
                    warp_per_block = hierarchical_structure.get('WarpPerBlock', [])
                    vector_dimensions = hierarchical_structure.get('VectorDimensions', [])
                    block_size = hierarchical_structure.get('BlockSize', [])
                    repeat_factor = hierarchical_structure.get('Repeat', [1, 1])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("ThreadPerWarp", 
                                 f"{thread_per_warp[0]}x{thread_per_warp[1]}" if len(thread_per_warp) >= 2 else "N/A")
                    
                    with col2:
                        st.metric("WarpPerBlock", 
                                 f"{warp_per_block[0]}x{warp_per_block[1]}" if len(warp_per_block) >= 2 else "N/A")
                    
                    with col3:
                        if vector_dimensions:
                            if len(vector_dimensions) > 1:
                                # Show multi-dimensional vector format
                                st.metric("Vector Dimensions", 
                                         f"{' × '.join(str(v) for v in vector_dimensions)}")
                            else:
                                # Single dimension
                                st.metric("Vector Dimensions", 
                                         f"{vector_dimensions[0]}")
                        else:
                            st.metric("Vector Dimensions", "N/A")
                    
                    with col4:
                        st.metric("Repeat Factor",
                                 f"{repeat_factor[0]}x{repeat_factor[1]}" if len(repeat_factor) >= 2 else "1x1")
                    
                    st.metric("Block Size (Total Threads)", 
                             f"{block_size[0]}x{block_size[1]}" if len(block_size) >= 2 else "N/A")
                    
                    # Display a formatted description of the structure (Moved from debug h_tabs[0])
                    if thread_per_warp and warp_per_block and vector_dimensions and block_size and repeat_factor:
                        threads_per_warp_m = thread_per_warp[0] if len(thread_per_warp) > 0 else 0
                        threads_per_warp_n = thread_per_warp[1] if len(thread_per_warp) > 1 else 0
                        warps_per_block_m = warp_per_block[0] if len(warp_per_block) > 0 else 0
                        warps_per_block_n = warp_per_block[1] if len(warp_per_block) > 1 else 0
                        # Handle multi-dimensional vector dimensions
                        if len(vector_dimensions) > 1:
                            vector_display = " × ".join(str(v) for v in vector_dimensions)
                            vector_k = 1
                            for v in vector_dimensions:
                                vector_k *= v
                        else:
                            vector_display = str(vector_dimensions[0]) if vector_dimensions else "0"
                            vector_k = vector_dimensions[0] if len(vector_dimensions) > 0 else 0
                        repeat_val_m = repeat_factor[0] if len(repeat_factor) > 0 else 1
                        repeat_val_n = repeat_factor[1] if len(repeat_factor) > 1 else 1
                        
                        total_threads_m = block_size[0]
                        total_threads_n = block_size[1]
                        total_threads = total_threads_m * total_threads_n
                        total_elements = total_threads * vector_k

                        st.write(f"""
                        ### Thread Hierarchy
                        - **ThreadPerWarp**: {threads_per_warp_m} x {threads_per_warp_n} threads per warp
                        - **WarpPerBlock**: {warps_per_block_m} x {warps_per_block_n} warps per block
                        - **Repeat Factor**: {repeat_val_m}x{repeat_val_n}
                        - **Vector Dimensions**: {vector_display}
                        - **Total Vector Size (K)**: {vector_k}
                        - **Total Thread Count (in Block)**: {total_threads} threads ({total_threads_m} x {total_threads_n})
                        - **Total Elements (covered by Block)**: {total_elements} elements
                        """)
                else:
                    # This message is shown if hierarchical_structure itself is empty
                    st.write("No hierarchical structure data available to display overview.")
                # --- END ALWAYS VISIBLE HIERARCHICAL INFO ---

                if st.session_state.debug_mode:
                    # In debug mode, display the raw hierarchical structure data and other debug tabs
                    if hierarchical_structure:
                        # Create a tabbed view for *additional* debug aspects
                        h_debug_tabs = st.tabs(
                            ["Thread Blocks (Debug)", "Raw Data (Debug)"]
                        )
                        
                        with h_debug_tabs[0]: # Was h_tabs[1]
                            # Display thread block organization
                            st.write("#### Thread Block Organization (Debug)")
                            thread_blocks = hierarchical_structure.get('ThreadBlocks', {})
                            
                            if thread_blocks:
                                for warp_key, warp_threads in thread_blocks.items():
                                    st.write(f"**{warp_key}**")
                                    thread_data = []
                                    for thread_id, details in warp_threads.items():
                                        thread_data.append({
                                            "Thread ID": thread_id,
                                            "Position": f"{details.get('position', [0, 0])}",
                                            "Global ID": details.get('global_id', 0)
                                        })
                                    if thread_data:
                                        st.dataframe(thread_data[:10])
                                        if len(thread_data) > 10:
                                            st.write(f"... and {len(thread_data) - 10} more threads")
                            else:
                                st.write("No thread block data available for debug.")
                        
                        with h_debug_tabs[1]: # Was h_tabs[2]
                            # Display raw data for debugging
                            st.write("#### Raw Hierarchical Structure Data (Debug)")
                            st.json(hierarchical_structure)
                    else:
                        st.write("No detailed hierarchical structure data available for debug tabs.")
                
                # Display the hierarchical tile visualization (THE PLOT - this part remains)
                if viz_data.get("hierarchical_structure"):
                    try:
                        # First validate the hierarchical structure data
                        hierarchical_structure = viz_data.get("hierarchical_structure", {})
                        # Check required components with default values if needed
                        thread_per_warp = hierarchical_structure.get('ThreadPerWarp', [16, 4])
                        warp_per_block = hierarchical_structure.get('WarpPerBlock', [4])
                        vector_dimensions = hierarchical_structure.get('VectorDimensions', [8])
                        
                        # Generate the visualization using the hierarchical_structure data
                        try:
                            # Pass source code to visualization if available
                            source_code = viz_data.get("source_code")
                            
                            # Add encoding to viz_data for data ID calculation
                            if st.session_state.encoding:
                                viz_data['encoding'] = st.session_state.encoding
                            
                            # Convert view_type from radio button to the view_mode parameter
                            view_mode = "data" if view_type == "Data View" else "thread"
                            
                            # When using data view mode, explicitly pass encoding and rs_lengths parameters
                            if view_mode == "data" and st.session_state.encoding:
                                # Get encoding parameter directly from session state
                                encoding_param = st.session_state.encoding
                                
                                # Get RsLengths from encoding if available
                                rs_lengths_param = encoding_param.get('RsLengths', [])
                                
                                # Resolve any variables in RsLengths
                                resolved_rs_lengths = []
                                for val in rs_lengths_param:
                                    if isinstance(val, str):
                                        # Try to find the variable in the session variables
                                        if val in st.session_state.variables:
                                            resolved_rs_lengths.append(st.session_state.variables[val])
                                        else:
                                            # Default if var name is present but value not found
                                            resolved_rs_lengths.append(1)
                                    elif isinstance(val, (int, float)):
                                        resolved_rs_lengths.append(int(val))
                                    else:
                                        resolved_rs_lengths.append(1)
                                
                                # Show information about replication in debug mode
                                if st.session_state.debug_mode:
                                    st.write("**Data View Parameters:**")
                                    st.write(f"- RsLengths (raw): {rs_lengths_param}")
                                    st.write(f"- RsLengths (resolved): {resolved_rs_lengths}")
                                    if not resolved_rs_lengths or (len(resolved_rs_lengths) == 1 and resolved_rs_lengths[0] == 1) or all(r <= 1 for r in resolved_rs_lengths):
                                        st.write("- No replication will be shown (RsLengths=[1] or empty)")
                                    else:
                                        st.write("- Replication will be visualized based on RsLengths")
                                
                                # Pass encoding and resolved rs_lengths directly to visualization
                                hierarchical_fig = visualize_hierarchical_tiles(
                                    viz_data, 
                                    code_snippet=source_code, 
                                    show_arrows=st.session_state.show_dimension_arrows,
                                    view_mode=view_mode,
                                    encoding=encoding_param,
                                    rs_lengths=resolved_rs_lengths
                                )
                            else:
                                # For thread view, no need for encoding/rs_lengths
                                hierarchical_fig = visualize_hierarchical_tiles(
                                    viz_data, 
                                    code_snippet=source_code, 
                                    show_arrows=st.session_state.show_dimension_arrows,
                                    view_mode=view_mode
                                )
                            st.pyplot(hierarchical_fig)
                            
                            # Add explanatory text
                            st.markdown("""
                            **This visualization shows:**
                            - **Top**: ThreadPerWarp, WarpPerBlock and total element calculations
                            - **Middle**: Thread layout organized by warps with thread IDs
                            - **Right**: The C++ code for this tile distribution
                            
                            **View Modes:**
                            - **Thread View**: Colors threads by warp assignment, showing the physical thread organization
                            - **Data View**: Colors threads by which data they access. Threads sharing the same color are accessing the same data due to replication.
                            
                            **Replication Logic in Data View:**
                            - Replication occurs when P dimensions map to R dimensions (where rh_major=0) in the tile distribution encoding
                            - When RsLengths=[1] or empty, each thread accesses unique data (no replication is shown)
                            - When RsLengths has values > 1, threads are colored based on which data they access:
                               - Block-level replication: When warp rows share data (RsLengths[0] > 1)
                               - Thread-level replication: When threads in the same row share data based on column position (RsLengths[1] > 1)
                            - A combination of both creates a pattern where data is repeated horizontally within rows, and these patterns repeat vertically across warp rows
                            
                            In **Data View**, threads with the same color (data ID) have independent copies of the same data element, 
                            which eliminates the need for synchronization between these threads. This replication pattern is key to 
                            understanding the data sharing structure in the computation.
                            """)
                        except IndexError as idx_err:
                            st.error(f"Index error in hierarchical tile visualization: {str(idx_err)}")
                            st.warning("This error is often caused by missing or invalid dimension data. Using default dimensions for visualization.")
                            # Display detailed data for debugging
                            if st.session_state.debug_mode:
                                st.json({
                                    "ThreadPerWarp": thread_per_warp,
                                    "WarpPerBlock": warp_per_block,
                                    "VectorDimensions": vector_dimensions
                                })
                                st.exception(idx_err)
                        except Exception as e:
                            st.error(f"Error creating hierarchical tile visualization: {str(e)}")
                            if st.session_state.debug_mode:
                                st.exception(e)
                    except Exception as e:
                        st.error(f"Error setting up hierarchical tile visualization: {str(e)}")
                        if st.session_state.debug_mode:
                            st.exception(e)
        except Exception as e:
            st.error(f"Error displaying hierarchical tile structure: {str(e)}")
            if st.session_state.debug_mode:
                st.exception(e)
        
        # --- Explanation for Hierarchical Parameters ---
        with st.expander("Understanding Hierarchical Parameters (Vector Size, Thread/Warp, Warps/Block)"):
            st.markdown("""
This visualizer, now using `tiler_pedantic.py`, aims to faithfully represent the tile distribution logic from C++ `tile_distribution.hpp` and `tile_distribution_encoding.hpp`. The hierarchical parameters like Vector Size, Thread/Warp, and Warps/Block are crucial for understanding how a computation is mapped to GPU hardware.

**Derivation of Hierarchical Parameters in `tiler_pedantic.py`:**

The `TileDistributionPedantic` class (in `tiler_pedantic.py`) calculates these parameters in its `calculate_hierarchical_tile_structure` method based on the encoding structure:

2.  **Inference Logic:**
    The system uses the number of P-dimensions (`NDimPs`) and Y-dimensions (`NDimYs`) along with their mapped R/H component lengths. The `_get_lengths_for_p_dim_component(p_idx)` method is key, as it determines the R/H lengths associated with a given P-dimension.

    *   **VectorDimensions (`[K]`):**
        *   The system identifies Y dimensions that map to the last position in H sequences.
        *   For example, if Y1 maps to H0[3] and Y3 maps to H1[3], both are identified as vector dimensions.
        *   This allows proper detection of multi-dimensional vectors (e.g., 2×2 or 4×4) rather than just a single value.
        *   The total vector size (K) is calculated as the product of all vector dimension values.
        *   Defaults to `[1]` if no vector dimensions are identified.

    *   **ThreadPerWarp (`[threads_m, threads_n]`):**
        *   If `NDimPs >= 2` (i.e., `P0` and `P1` exist):
            *   `threads_m` is inferred from the length of the *first* R/H component that `P1` maps to.
            *   `threads_n` is inferred from the length of the *second* R/H component that `P1` maps to. (If `P1` maps to only one component, `threads_n` becomes 1).
        *   If `NDimPs == 1` (only `P0` exists):
            *   `threads_m` is inferred from the length of the *first* R/H component `P0` maps to.
            *   `threads_n` is inferred from the length of the *second* R/H component `P0` maps to (or 1 if `P0` maps to only one component).
        *   If a P-dimension or its mapped component is not found/valid, the corresponding length defaults to 1.
        *   Defaults to `[1, 1]` if no P-dims are available or inference doesn't yield specific values.

    *   **WarpPerBlock (`[warps_m, warps_n]`):**
        *   If `NDimPs >= 1` (i.e., `P0` exists):
            *   `warps_m` is inferred from the length of the *first* R/H component that `P0` maps to.
            *   `warps_n` is inferred from the length of the *second* R/H component that `P0` maps to. (If `P0` maps to only one component, `warps_n` becomes 1).
        *   If `P0` or its mapped component is not found/valid, the corresponding length defaults to 1.
        *   **Heuristic Default**: If `WarpPerBlock` remains `[1, 1]` after the above, AND `NDimPs >= 1`, AND `ThreadPerWarp` is non-trivial (i.e., `threads_m * threads_n > 1`), then `WarpPerBlock` is set to `[4, 1]` for a more common visualization.
        *   Defaults to `[1, 1]` if no P-dims are available or inference doesn't yield specific values.

    *   **Repeat (`[R_m, R_n]`):**
        *   Initialized to `[1, 1]` and defaults to that in most cases.

3.  **BlockSize Calculation**:
    Once `ThreadPerWarp = [tpw_m, tpw_n]`, `WarpPerBlock = [wpb_m, wpb_n]`, and `Repeat = [repeat_m, repeat_n]` are determined:
    *   `BlockSize = [tpw_m * wpb_m * repeat_m, tpw_n * wpb_n * repeat_n]`

**Role of Y-Dimensions (including "Extra" Y-Dimensions):**

The Y-dimensions (`Ys2RHsMajor/Minor`, leading to `ys_lengths_`) collectively define the shape and internal structure of the data elements that each P-tile is responsible for. They do **not** typically mean multiple GPU blocks.

*   **Vector Dimension (from a Y-dim):** As discussed, one Y-dimension is inferred as the `VectorDimensions (K)`.
*   **"Extra" Y-Dimensions:** Any other Y-dimensions contribute to the local data structure *within* each P-tile (and within each repetition if a repeat factor is active). They add more axes to the "within-tile" addressing scheme.

*Example from a Multi-Dimensional Vector:*
Consider a tile distribution encoding with this mapping pattern:
```
tile_distribution_encoding<
    sequence<>,                             // Empty R
    tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
          sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
    tuple<sequence<1, 2>, sequence<1, 2>>,
    tuple<sequence<1, 1>, sequence<2, 2>>,
    sequence<1, 1, 2, 2>,
    sequence<0, 3, 0, 3>>{}
```

Here, the Y dimensions map to H sequences as follows:
*   `Y1` maps to H0[3] (the Vector_M position)
*   `Y3` maps to H1[3] (the Vector_N position)

The vector dimension detection algorithm correctly identifies:
*   Both Y1 and Y3 as vector dimensions
*   If both are 4, then VectorDimensions = [4, 4] (instead of just 4)
*   Total Vector Size (K) = 4 × 4 = 16

This multi-dimensional vector information is critical for understanding:
*   The true shape of the data being processed (2D vectors rather than 1D)
*   How threads access data elements in a structured way
*   The total computational work performed by each thread

These "extra" Y dimensions are fundamental to:
1.  **`d_scalar_idx` calculation:** `idx_y` is flattened to `d_scalar_idx` (e.g., `y0_coord * Y1_length + y1_coord`, so `y0_coord * 4 + y1_coord` in the example), which then acts as a high-order index into the X-space.
2.  **`xs_coords` calculation:** `idx_y` helps select specific elements from R/H components that are directly "owned" by Y-dimensions in the `PsYs2XsAdaptor`.
            """)
        # --- End Explanation ---


        # Performance metrics
        if st.session_state.tile_distribution is not None:
            st.subheader("Performance Metrics")
            
            try:
                viz_data = st.session_state.tile_distribution.get_visualization_data()
                occupancy = viz_data.get("occupancy", 0)
                utilization = viz_data.get("utilization", 0)
                
                col1, col2 = st.columns(2)
                col1.metric("Occupancy", f"{occupancy:.2f}")
                col2.metric("Utilization", f"{utilization:.2f}")
                
                # Additional metrics explanation
                st.markdown("""
                **Metrics Explanation:**
                - **Occupancy**: Ratio of active threads to maximum possible threads.
                - **Utilization**: Efficiency of thread usage for computation.
                """)
            except Exception as e:
                st.error(f"Error calculating performance metrics: {str(e)}")
                if st.session_state.debug_mode:
                    st.exception(e)
    else:
        # Display initial instructions if no encoding is loaded
        st.info("👈 Select an input method from the sidebar to get started.")
        
        st.markdown("""
        ## Welcome to the Tile Distribution Visualizer
        
        This tool helps you visualize tile_distribution_encoding concepts from the Composable Kernels library.
        
        ### How to use:
        1. Select an input method from the sidebar
        2. Load or enter a tile_distribution_encoding structure
        3. Adjust variables using the sliders
        4. Explore the visualizations
        
        ### Understanding the visualizations:
        - **Encoding Structure**: Shows how dimensions relate to each other
        - **Tile Distribution**: Shows thread mapping to tile elements
        - **Thread Access Pattern**: Animates thread access over time
        """)

def display_variable_controls():
    """Display sliders for adjusting template variables."""
    st.header("Template Variables")
    
    variables = {}
    
    for var in st.session_state.parsed_variables:
        # Get current value if exists, otherwise default to 4
        current_value = st.session_state.variables.get(var, 4)
        
        # Format display name for namespace-prefixed variables
        if '::' in var:
            namespace, var_name = var.split('::')
            display_name = f"{namespace}::{var_name}"
        else:
            display_name = var
        
        # Create a slider for this variable
        value = st.slider(
            display_name,
            min_value=1,
            max_value=32,
            value=current_value,
            step=1,
            key=f"slider_{var}"  # Use unique key to prevent conflicts
        )
        
        variables[var] = value
    
    # Update session state variables
    st.session_state.variables = variables

def calculate_distribution():
    """Calculate tile distribution based on current encoding and variables."""
    try:
        if st.session_state.encoding is not None:
            # Create tile distribution object
            tile_distribution = TileDistribution(
                st.session_state.encoding,
                st.session_state.variables
            )
            
            # Set a descriptive title based on the selected example or custom code
            if hasattr(st.session_state, 'selected_example'):
                tile_distribution.set_tile_name(st.session_state.selected_example)
            
            # Set the source code for visualization
            if hasattr(st.session_state, 'cpp_code') and st.session_state.cpp_code:
                tile_distribution.set_source_code(st.session_state.cpp_code)
            
            st.session_state.tile_distribution = tile_distribution
    except Exception as e:
        st.error(f"Error creating tile distribution: {str(e)}")
        st.session_state.tile_distribution = None

if __name__ == "__main__":
    main() 