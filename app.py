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
        
        input_method = st.radio(
            "Select input method:",
            ["Example Template", "Custom C++ Code", "Load from File"]
        )
        
        if input_method == "Example Template":
            load_example_template()
        elif input_method == "Custom C++ Code":
            load_custom_code()
        elif input_method == "Load from File":
            load_from_file()
        
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
                        st.metric("Vector Dimensions", 
                                 f"{vector_dimensions[0]}" if vector_dimensions else "N/A")
                    
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
                        - **Vector Dimensions (K)**: {vector_k}
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
                            hierarchical_fig = visualize_hierarchical_tiles(viz_data, code_snippet=source_code, show_arrows=st.session_state.show_dimension_arrows)
                            st.pyplot(hierarchical_fig)
                            
                            # Add explanatory text
                            st.markdown("""
                            **This visualization shows:**
                            - **Top**: ThreadPerWarp, WarpPerBlock and total element calculations
                            - **Middle**: Thread layout organized by warps with thread IDs
                            - **Right**: The C++ code for this tile distribution
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

The `TileDistributionPedantic` class (in `tiler_pedantic.py`) calculates these parameters in its `calculate_hierarchical_tile_structure` method. It prioritizes hints you can provide in the input encoding dictionary, but will infer them if hints are absent:

1.  **`visualization_hints` in your input `encoding_dict`**:
    *   You can guide the visualizer by adding a `visualization_hints` dictionary to your main input.
    *   `vector_dim_ys_index`: (integer) The index of the YS-dimension that represents the vector length.
    *   `thread_per_warp_p_indices`: (list of integers) P-dimension index(es) whose component lengths\\' product defines Thread/Warp.
        *   One index `[pM]` results in `ThreadPerWarp = [product(lengths(P_pM)), 1]`.
        *   Two indices `[pM, pN]` result in `ThreadPerWarp = [product(lengths(P_pM)), product(lengths(P_pN))]`.
    *   `warp_per_block_p_indices`: (list of integers) P-dimension index(es) for Warps/Block.
        *   One index `[pM]` results in `WarpPerBlock = [product(lengths(P_pM)), 1]`.
        *   Two indices `[pM, pN]` result in `WarpPerBlock = [product(lengths(P_pM)), product(lengths(P_pN))]`.
    *   `repeat_factor_ys_index`: (integer) YS-dimension index for the repeat factor (currently visualized as `Repeat = [length, 1]`).

2.  **Inference Logic (if hints are not provided or insufficient)**:
    The system uses the number of P-dimensions (`NDimPs`) and Y-dimensions (`NDimYs`) along with their mapped R/H component lengths. The `_get_lengths_for_p_dim_component(p_idx)` method is key, as it determines the R/H lengths associated with a given P-dimension.

    *   **VectorDimensions (`[K]`)**:
        *   Uses `vector_dim_ys_index` hint if available.
        *   Otherwise, if `NDimYs > 0`, it checks the length of the *last* YS-dimension (`ys_lengths_[-1]`). If this length is between 2 and 16 (inclusive), it\\'s inferred as the vector dimension.
        *   Defaults to `[1]`.

    *   **ThreadPerWarp (`[threads_m, threads_n]`)**:
        *   Uses `thread_per_warp_p_indices` hint if available (product of components for specified P-dims).
        *   Otherwise (default inference):
            *   If `NDimPs >= 2` (i.e., `P0` and `P1` exist):
                *   `threads_m` is inferred from the length of the *first* R/H component that `P1` maps to.
                *   `threads_n` is inferred from the length of the *second* R/H component that `P1` maps to. (If `P1` maps to only one component, `threads_n` becomes 1).
            *   If `NDimPs == 1` (only `P0` exists):
                *   `threads_m` is inferred from the length of the *first* R/H component `P0` maps to.
                *   `threads_n` is inferred from the length of the *second* R/H component `P0` maps to (or 1 if `P0` maps to only one component).
            *   If a P-dimension or its mapped component is not found/valid, the corresponding length defaults to 1.
        *   Defaults to `[1, 1]` if no P-dims are available or inference doesn't yield specific values.

    *   **WarpPerBlock (`[warps_m, warps_n]`)**:
        *   Uses `warp_per_block_p_indices` hint if available (product of components for specified P-dims).
        *   Otherwise (default inference):
            *   If `NDimPs >= 1` (i.e., `P0` exists):
                *   `warps_m` is inferred from the length of the *first* R/H component that `P0` maps to.
                *   `warps_n` is inferred from the length of the *second* R/H component that `P0` maps to. (If `P0` maps to only one component, `warps_n` becomes 1).
            *   If `P0` or its mapped component is not found/valid, the corresponding length defaults to 1.
        *   **Heuristic Default**: If `WarpPerBlock` remains `[1, 1]` after the above, AND `NDimPs >= 1`, AND `ThreadPerWarp` is non-trivial (i.e., `threads_m * threads_n > 1`), then `WarpPerBlock` is set to `[4, 1]` for a more common visualization.
        *   Defaults to `[1, 1]` if no P-dims are available or inference doesn't yield specific values.

    *   **Repeat (`[R_m, R_n]`)**:
        *   Initialized to `[1, 1]`.
        *   If a `visualization_hints: {'repeat_factor_ys_index': YS_INDEX}` is provided, the length of the specified YS-dimension will be used to set `R_m` (i.e., `Repeat = [length_of_hinted_YS, 1]`).
        *   Without this hint, `Repeat` remains `[1, 1]` (no automatic inference is performed for the repeat factor from specific Y-to-H mappings).

3.  **BlockSize Calculation**:
    Once `ThreadPerWarp = [tpw_m, tpw_n]`, `WarpPerBlock = [wpb_m, wpb_n]`, and `Repeat = [repeat_m, repeat_n]` are determined:
    *   `BlockSize = [tpw_m * wpb_m * repeat_m, tpw_n * wpb_n * repeat_n]`

**Role of Y-Dimensions (including "Extra" Y-Dimensions):**

The Y-dimensions (`Ys2RHsMajor/Minor`, leading to `ys_lengths_`) collectively define the shape and internal structure of the data elements that each P-tile is responsible for. They do **not** typically mean multiple GPU blocks.

*   **Vector Dimension (from a Y-dim):** As discussed, one Y-dimension might be inferred or hinted as the `VectorDimensions (K)`.
*   **Repeat Factor (from a Y-dim, if hinted):** A Y-dimension hinted via `visualization_hints: {'repeat_factor_ys_index': YS_INDEX}` scales the work within the conceptual block (currently, the hint applies to the M-dimension of repeat: `Repeat = [hinted_YS_length, 1]`).
*   **"Extra" Y-Dimensions:** Any other Y-dimensions contribute to the local data structure *within* each P-tile (and within each repetition if a repeat factor is active). They add more axes to the "within-tile" addressing scheme.

*Example from a Parsed Snippet:*
Consider an encoding like your last example where Y-dimensions are resolved to `ys_lengths_ = [4, 4]`. 
*   `Y0` (length 4)
*   `Y1` (length 4)
If the system infers `VectorDimensions (K) = 4` (e.g., from Y1, being the last Y-dim of appropriate size), and `Repeat Factor` defaults to `[1,1]` (as no hint is provided for it):

In this case, Y0 (length 4) is an "extra" Y-dimension relative to the main hierarchical parameters shown.
*   It means that for the work unit defined by a P-tile, there are 4 distinct "sub-units" or "slices" along this Y0 dimension.
*   Each of these sub-units is then a vector of 4 elements (defined by Y1, the vector dimension).
*   So, the P-tile processes a local data block conceptually shaped by `Y0_length x Y1_length` (i.e., 4x4 elements in this example).
*   The `idx_y` coordinates `[y0_coord, y1_coord]` are crucial in `calculate_index` to select a specific element (or vector start) from this local 4x4 data block.

These "extra" Y dimensions are fundamental to:
1.  **`d_scalar_idx` calculation:** `idx_y` is flattened to `d_scalar_idx` (e.g., `y0_coord * Y1_length + y1_coord`, so `y0_coord * 4 + y1_coord` in the example), which then acts as a high-order index into the X-space.
2.  **`xs_coords` calculation:** `idx_y` helps select specific elements from R/H components that are directly "owned" by Y-dimensions in the `PsYs2XsAdaptor`.

This explanation clarifies how the visualizer, powered by `tiler_pedantic.py`, arrives at the hierarchical parameters, prioritizing explicit hints and falling back to rule-based inference.
            """)
        # --- End Explanation ---

        # Y-Space Local Data Layout Visualization
        st.subheader("P-Tile Local Data Layout (Y-Dimensions)")
        try:
            if st.session_state.tile_distribution is not None:
                ys_lengths = st.session_state.tile_distribution.get_ys_lengths()
                ys_names = [f"Y{i}" for i in range(len(ys_lengths))]
                
                # Get VectorDimensionYSIndex from hierarchical_structure
                # viz_data was already fetched for Hierarchical Tile Structure, reuse if possible or re-fetch
                if hasattr(st.session_state.tile_distribution, 'get_visualization_data'): # Check if method exists
                    viz_data = st.session_state.tile_distribution.get_visualization_data()
                    hierarchical_structure = viz_data.get("hierarchical_structure", {})
                    vector_dim_ys_idx = hierarchical_structure.get("VectorDimensionYSIndex", -1)
                else:
                    vector_dim_ys_idx = -1 # Fallback if data not available
                    hierarchical_structure = {} # Ensure it exists

                if ys_lengths:
                    y_space_fig = visualize_y_space_structure(
                        ys_lengths, 
                        ys_names, 
                        vector_dim_ys_idx
                    )
                    st.pyplot(y_space_fig)
                    st.markdown("""
                    **This visualization shows the structure of the data within a single P-tile, as defined by the Y-dimensions.**
                    - Each cell represents an element or a base for a vector.
                    - If a vector dimension is identified, its elements are highlighted.
                    - For NDimY > 2, this shows a 2D slice (Y0 vs Y1).
                    """)
                else:
                    st.write("No Y-dimensions to visualize for local data layout.")
            else:
                st.write("Tile distribution not calculated yet.")
        except Exception as e:
            st.error(f"Error displaying Y-space structure: {str(e)}")
            if st.session_state.debug_mode:
                st.exception(e)

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

def load_example_template():
    """Load an example tile_distribution_encoding template."""
    # Get examples and default variables from examples.py
    examples = get_examples()
    
    # Initialize the selected example in session state if not present
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = list(examples.keys())[0]
    
    example_keys = list(examples.keys())
    try:
        selected_example = st.selectbox(
            "Select an example:", 
            example_keys, 
            index=example_keys.index(st.session_state.selected_example),
            key="example_selectbox"
        )
        # Update the session state
        st.session_state.selected_example = selected_example
    except:
        # Fallback if there's an issue with the selectbox
        selected_example = example_keys[0]
        st.session_state.selected_example = selected_example
    
    cpp_code = examples[selected_example]
    
    # Display the code
    st.code(cpp_code, language="cpp")
    
    # Parse the example code
    parser = TileDistributionParser()
    encoding = parser.parse_tile_distribution_encoding(cpp_code)
    
    # Get default variables for this example
    variables = get_default_variables(selected_example)
    
    # Save to session state
    st.session_state.encoding = encoding
    st.session_state.variables = variables
    st.session_state.cpp_code = cpp_code  # Save the code for visualization
    
    # Extract variables from encoding for sliders
    if encoding and "variable_names" in encoding:
        var_list = encoding["variable_names"]
    else:
        var_list = []
        
    st.session_state.parsed_variables = list(set(
        var_list + list(variables.keys())
    ))

def load_custom_code():
    """Parse custom C++ code from a text area."""
    cpp_code = st.text_area(
        "Enter tile_distribution_encoding C++ code:",
        height=200,
        placeholder="tile_distribution_encoding<...>{}"
    )
    
    if cpp_code:
        # Parse button
        if st.button("Parse Code"):
            parser = TileDistributionParser()
            encoding = parser.parse_tile_distribution_encoding(cpp_code)
            variables = parser.extract_template_variables(cpp_code)
            
            if encoding:
                st.session_state.encoding = encoding
                st.session_state.variables = variables
                st.session_state.cpp_code = cpp_code  # Save the code for visualization
                
                # Extract variables from encoding for sliders
                st.session_state.parsed_variables = list(set(
                    encoding.get("variable_names", []) + list(variables.keys())
                ))
                
                st.success("Code parsed successfully!")
            else:
                st.error("Failed to parse the code. Check the syntax.")

def load_from_file():
    """Load tile_distribution_encoding from a file."""
    file_path = st.text_input(
        "Enter file path (relative to workspace):",
        placeholder="/main/example.cpp"
    )
    
    if file_path:
        if os.path.exists(file_path):
            if st.button("Load File"):
                try:
                    with open(file_path, 'r') as f:
                        cpp_code = f.read()
                    
                    parser = TileDistributionParser()
                    encoding = parser.parse_tile_distribution_encoding(cpp_code)
                    variables = parser.extract_template_variables(cpp_code)
                    
                    if encoding:
                        st.session_state.encoding = encoding
                        st.session_state.variables = variables
                        st.session_state.cpp_code = cpp_code  # Save the code for visualization
                        
                        # Extract variables from encoding for sliders
                        st.session_state.parsed_variables = list(set(
                            encoding.get("variable_names", []) + list(variables.keys())
                        ))
                        
                        st.success(f"File '{file_path}' loaded successfully!")
                    else:
                        st.error("No tile_distribution_encoding structure found in the file.")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        else:
            st.error(f"File '{file_path}' not found.")

def display_variable_controls():
    """Display sliders for adjusting template variables."""
    st.header("Template Variables")
    
    variables = {}
    
    for var in st.session_state.parsed_variables:
        # Get current value if exists, otherwise default to 4
        current_value = st.session_state.variables.get(var, 4)
        
        # Create a slider for this variable
        value = st.slider(
            f"{var}",
            min_value=1,
            max_value=32,
            value=current_value,
            step=1
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