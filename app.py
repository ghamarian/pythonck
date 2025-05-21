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
## How Thread Hierarchy Parameters Are Calculated

This visualizer analyzes your tile distribution encoding to determine how threads are organized. Let's understand how each parameter is calculated:

### Vector Dimensions
The system examines which Y-dimensions map to the final positions in H-sequences:

- Y-dimensions that map to the last position in an H-sequence (e.g., Y1 maps to H0[3]) are identified as vector dimensions
- If multiple Y-dimensions map to last positions in different H-sequences (e.g., Y1 → H0[3] and Y3 → H1[3]), they're all considered vector dimensions
- The resulting VectorDimensions shows these as a list (e.g., [4, 4])
- The total vector size (K) is the product of all vector dimension values (e.g., 4 × 4 = 16)
- If no vector dimensions are found, it defaults to [1]

### ThreadPerWarp [threads_m, threads_n]
The system uses P1 (the second P-dimension) to determine threads per warp:

- P1 is specifically used for ThreadPerWarp organization:
  - `threads_m` comes from the first R/H component that P1 maps to
  - `threads_n` comes from the second R/H component that P1 maps to (or 1 if P1 only maps to one component)
- If P1 doesn't exist or its values can't be determined, defaults to [1, 1]
- Final result is [threads_m, threads_n] representing a 2D grid of threads in a warp

### WarpPerBlock [warps_m, warps_n]
The system uses P0 (the first P-dimension) to determine warps per block:

- P0 is specifically used for WarpPerBlock organization:
  - `warps_m` comes from the first R/H component that P0 maps to
  - `warps_n` comes from the second R/H component that P0 maps to (or 1 if P0 only maps to one component)
- If P0's values don't yield meaningful results, a default of [4, 1] is used when:
  - ThreadPerWarp is meaningful (not just [1, 1])
- Otherwise, defaults to [1, 1]

### Repeat Factor [repeat_m, repeat_n]
Initialized to [1, 1] and typically remains at those values unless explicitly specified.

### BlockSize [total_m, total_n]
The final BlockSize combines all the above parameters:
- `total_m = threads_m × warps_m × repeat_m`
- `total_n = threads_n × warps_n × repeat_n`

This represents the complete 2D grid of threads in a block.

## Understanding Data Replication (Data View Mode)

In Data View mode, the visualizer shows how data is replicated among threads. This replication is determined by RsLengths and how P-dimensions map to R-dimensions:

### How Replication Works
When a P-dimension maps to an R-dimension (rh_major=0 in Ps2RHssMajor), threads controlled by that P-dimension access replicated data:

1. **If P0 maps to R-dimension (block-level replication)**:
   - Different warps (rows) in the visualization will share the same data
   - This creates a pattern where data repeats vertically across warp rows
   - Example: With RsLengths=[2], every pair of warp rows will share the same data

2. **If P1 maps to R-dimension (thread-level replication)**:
   - Different threads within the same warp row will share the same data
   - This creates a pattern where data repeats horizontally within a row
   - Example: With RsLengths=[1,4], every group of 4 adjacent threads will share the same data

3. **Combined replication (P0 and P1 both map to R)**:
   - Creates a checkerboard-like pattern of data replication
   - Data repeats both horizontally and vertically

### Important Notes on Replication:
- Replication only occurs when RsLengths has values > 1
- If RsLengths=[1] or empty, each thread accesses unique data (no replication is shown)
- In the Data View, threads with the same color (data ID) are accessing the same piece of data
- This replication is key to understanding data sharing in GPU kernels and can help identify:
  - Where synchronization is/isn't needed
  - How efficiently memory access patterns are organized
  - Whether threads are working independently or on shared data

## Understanding Occupancy and Utilization

The visualizer also calculates performance metrics:

- **Occupancy**: Shows what percentage of the tile's cells have valid thread mappings. It's calculated as:
  ```
  occupancy = valid_thread_positions / total_positions_in_tile
  ```

- **Utilization**: Applies a slight efficiency factor to occupancy:
  ```
  utilization = occupancy × 0.95
  ```

Note that these are approximations based on the tile structure, not actual hardware measurements.

## Hardware Considerations

The visualizer infers thread organization based on the encoding structure, not hardware-specific constants. In practice:
- AMD GPUs use 64 threads per wavefront (AMD's term for warp)
- NVIDIA GPUs use 32 threads per warp

The visualization may show thread organizations that don't perfectly align with hardware implementations.

## Example: Multi-Dimensional Vector
Consider this tile distribution encoding:
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

Here, Y1 maps to H0[3] (Vector_M) and Y3 maps to H1[3] (Vector_N). If each has value 4, the system will:
1. Identify both as vector dimensions: VectorDimensions = [4, 4]
2. Calculate total vector size: K = 4 × 4 = 16
3. Each thread processes 16 elements in a 4×4 grid
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