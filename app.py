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

# Add new imports for transformation graph visualization
import graphviz
import sympy as sp
from pytensor.tensor_descriptor import (
    Transform, PassThroughTransform, MergeTransform, UnmergeTransform,
    EmbedTransform, OffsetTransform, PadTransform, ReplicateTransform
)

# Set page config
st.set_page_config(
    page_title="Tile Distribution Visualizer",
    page_icon="📊",
    layout="wide",
)

def main():
    """Main function for the Streamlit app."""
    # Clean slate when restarting app
    if 'first_load' not in st.session_state:
        # This is a first load, clear all session state
        for key in list(st.session_state.keys()):
            if key != 'first_load':
                del st.session_state[key]
        # Mark that we've done the first load
        st.session_state.first_load = True
    
    # Initialize all required session state variables
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
        # Define a callback to handle example selection changes
        def on_example_change():
            # Get the selected example from the session state
            selected = st.session_state.example_selectbox
            # Update our tracking of selected example
            st.session_state.selected_example = selected
            # Update the current code with the new example
            st.session_state.current_code = examples[selected]
            # Reset edit mode to True for a new example
            st.session_state.edit_mode = True
            
        try:
            # Simple selection dropdown with callback
            selected_example = st.selectbox(
                "Example Template:", 
                example_keys,
                index=0 if not st.session_state.get('selected_example') in example_keys else example_keys.index(st.session_state.selected_example),
                key="example_selectbox",
                on_change=on_example_change
            )
        except Exception as e:
            # Safer fallback if there's an issue with the selectbox
            if example_keys:
                selected_example = example_keys[0]
                st.session_state.selected_example = selected_example
            else:
                st.error(f"No examples available: {str(e)}")
                selected_example = ""
                st.session_state.selected_example = ""
        
        # Function to toggle edit mode when button is clicked
        def toggle_mode():
            st.session_state.edit_mode = not st.session_state.edit_mode
        
        # Initialize session state variables if they don't exist
        if 'current_code' not in st.session_state:
            # Get the selected example code
            if hasattr(st.session_state, 'selected_example') and st.session_state.selected_example in examples:
                st.session_state.current_code = examples[st.session_state.selected_example]
            else:
                # Fallback to first example if something went wrong
                st.session_state.current_code = examples[example_keys[0]]
        
        if 'edit_mode' not in st.session_state:
            st.session_state.edit_mode = True
            
        # Create header row with title and toggle button
        col1, col2 = st.columns([5, 2])
        
        with col1:
            st.markdown("### Code Editor")
            
        with col2:
            # Single button that changes label based on current mode
            button_label = "🎨 Format" if st.session_state.edit_mode else "✏️ Edit"
            st.button(button_label, on_click=toggle_mode, key="mode_toggle")
            
        # Display the appropriate code view based on current mode
        if st.session_state.edit_mode:
            # Edit mode: Show editable text area
            edited_code = st.text_area(
                label="",
                value=st.session_state.current_code,
                height=300,
                key="code_editor"
            )
            # Save the edited code
            st.session_state.current_code = edited_code
        else:
            # Format mode: Show syntax-highlighted code with built-in copy button
            st.code(st.session_state.current_code, language="cpp")
        
        # Parse button
        if st.button("Parse Code"):
            parser = TileDistributionParser()
            # Use the current code from session state
            current_code = st.session_state.current_code
            
            # COMPLETELY RESET ALL VARIABLE STATES
            # This ensures old variables are fully removed
            if 'parsed_variables' in st.session_state:
                del st.session_state.parsed_variables
            if 'variables' in st.session_state:
                del st.session_state.variables
            
            # Now parse the code freshly
            encoding = parser.parse_tile_distribution_encoding(current_code)
            
            if encoding:
                # Extract template variables directly from the current code
                detected_variables = parser.extract_template_variables(current_code)
                
                # Get default values for template variables only if they exist in the code
                default_variables = get_default_variables(selected_example)
                filtered_defaults = {
                    k: v for k, v in default_variables.items() 
                    if k in detected_variables
                }
                
                # Create a new clean variables dict - start with detected variables
                new_variables = {}
                
                # Add default values first (lower priority)
                new_variables.update(filtered_defaults)
                
                # Then add detected values (higher priority)
                new_variables.update(detected_variables)
                
                # Initialize parsed_variables with only what's in the current code
                st.session_state.parsed_variables = sorted(list(detected_variables.keys()))
                
                # Save everything to session state
                st.session_state.encoding = encoding
                st.session_state.variables = new_variables
                st.session_state.cpp_code = current_code  # Save the code for visualization
                
                # Switch to syntax-highlighted view after successful parsing
                st.session_state.edit_mode = False
                
                # Extract variables from encoding for sliders
                if encoding and "variable_names" in encoding:
                    var_list = encoding["variable_names"]
                else:
                    var_list = []
                    
                # We no longer need this since we're directly setting 
                # parsed_variables in the parsing function by using only detected variables
                
                st.success("Code parsed successfully!")
            else:
                st.error("Failed to parse the code. Check the syntax.")
        
        # Variable sliders (shown only if encoding is parsed)
        if st.session_state.encoding is not None:
            display_variable_controls()
        
        # Add debug mode toggle
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    
    # Main content
    if st.session_state.encoding is not None:
        # Forcibly recalculate the tile distribution when:
        # 1. It doesn't exist yet 
        # 2. It was set to None by the slider code
        # 3. Debug mode is on and force_recalculate is checked
        
        force_recalculate = False
        if st.session_state.debug_mode:
            force_recalculate = st.checkbox("Force recalculation", value=False, key="force_recalc_checkbox")
        
        if (
            'tile_distribution' not in st.session_state or
            st.session_state.tile_distribution is None or
            force_recalculate
        ):
            calculate_distribution()
        
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
                                    view_mode=view_mode,
                                    encoding=encoding_param,
                                    rs_lengths=resolved_rs_lengths
                                )
                            else:
                                # For thread view, no need for encoding/rs_lengths
                                hierarchical_fig = visualize_hierarchical_tiles(
                                    viz_data, 
                                    code_snippet=source_code, 
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

        # Transformation Pipeline Graphs
        st.subheader("Transformation Pipeline")
        
        try:
            if st.session_state.encoding is not None:
                # Build and display the transformation graph
                # Use variables if they exist, otherwise empty dict for hardcoded examples
                variables = st.session_state.variables if hasattr(st.session_state, 'variables') and st.session_state.variables else {}
                
                transformation_graph = build_tile_distribution_transformation_graph(
                    st.session_state.encoding, 
                    variables
                )
                
                st.markdown("""
                **This transformation pipeline shows:**
                - **Input**: X dimensions (logical tensor dimensions)
                - **Replicate**: Creates R dimensions for data replication
                - **Unmerge**: Splits each X dimension into H components (hierarchical structure)
                - **Merge**: Combines RH components to create P dimensions (thread mapping)
                - **Output**: P dimensions (thread indices)
                
                This is the underlying transformation that maps logical tensor coordinates to thread assignments.
                
                **Note**: Y dimensions have their own separate Y↔D transformation graph for spatial output mapping.
                """)
                
                st.graphviz_chart(transformation_graph)
                
                # Build and display the Y↔D transformation graph
                st.markdown("---")
                st.markdown("### Y↔D Transformation (Spatial Coordinates ↔ Linearized Memory)")
                
                y_to_d_graph = build_y_to_d_transformation_graph(
                    st.session_state.encoding, 
                    variables
                )
                
                st.markdown("""
                **This Y↔D transformation shows:**
                - **Input**: D (linearized memory address, single dimension)
                - **Transform**: UnmergeTransform splits linear address into spatial coordinates
                - **Output**: Y dimensions (multi-dimensional spatial coordinates)
                
                Each Y dimension gets its size from the RH space (R or H dimensions) as specified by Ys2RHsMajor/Minor mappings.
                This transformation is separate from the main X→RH→P pipeline above.
                """)
                
                st.graphviz_chart(y_to_d_graph)
                
                # Show the transformation details in debug mode
                if st.session_state.debug_mode:
                    with st.expander("Transformation Details (Debug)"):
                        try:
                            transforms, lower_dims, upper_dims = extract_transforms_from_tile_distribution(
                                st.session_state.encoding, st.session_state.variables
                            )
                            
                            st.write("**Extracted Transforms:**")
                            for i, (transform, lower, upper) in enumerate(zip(transforms, lower_dims, upper_dims)):
                                st.write(f"Transform {i+1}: {transform['name']}")
                                st.write(f"  - Type: {transform['type']}")
                                st.write(f"  - Lengths: {transform.get('lengths', 'N/A')}")
                                st.write(f"  - Lower dims: {lower}")
                                st.write(f"  - Upper dims: {upper}")
                                st.write("")
                        except Exception as e:
                            st.error(f"Error extracting transformation details: {str(e)}")
                            st.exception(e)
            else:
                st.info("Parse a tile distribution encoding to see the transformation pipeline.")
                
                # Debug information to help user understand what's missing
                if st.session_state.debug_mode:
                    st.write("**Debug Info:**")
                    st.write(f"- encoding exists: {st.session_state.encoding is not None}")
                    st.write(f"- variables exists: {bool(st.session_state.variables) if hasattr(st.session_state, 'variables') else False}")
                    if hasattr(st.session_state, 'encoding') and st.session_state.encoding:
                        st.write(f"- encoding keys: {list(st.session_state.encoding.keys())}")
                    if hasattr(st.session_state, 'variables') and st.session_state.variables:
                        st.write(f"- variables: {st.session_state.variables}")
        except Exception as e:
            st.error(f"Error creating transformation pipeline visualization: {str(e)}")
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

def display_variable_controls():
    """Display sliders for adjusting template variables."""
    st.header("Template Variables")
    
    # Initialize a new dictionary to store the updated variable values
    updated_variables = {}
    
    if not hasattr(st.session_state, 'parsed_variables') or not st.session_state.parsed_variables:
        st.info("No template variables detected in code. Parse your code to find variables.")
        return
    
    for var in st.session_state.parsed_variables:
        # Get current value if exists, otherwise default to 4
        current_value = st.session_state.variables.get(var, 4)
        
        # Format display name for namespace-prefixed variables
        if '::' in var:
            namespace, var_name = var.split('::')
            display_name = f"{namespace}::{var_name}"
        else:
            display_name = var
        
        # Create a slider for this variable with a simpler approach
        value = st.slider(
            display_name,
            min_value=1,
            max_value=32,
            value=current_value,
            step=1,
            key=f"slider_{var}"  # Use unique key to prevent conflicts
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
    
    # Force recalculation if variables changed
    if variables_changed and 'tile_distribution' in st.session_state:
        st.session_state.tile_distribution = None

def calculate_distribution():
    """Calculate tile distribution based on current encoding and variables."""
    try:
        if st.session_state.encoding is not None:
            # Use the variables directly from session state (updated by the sliders)
            variables = st.session_state.variables
            
            # Make a copy of the encoding
            encoding = st.session_state.encoding.copy()
            
            # Add PsLengths if missing (for the single P tuple case)
            if "PsLengths" not in encoding and "Ps2RHssMajor" in encoding:
                ndim_p = len(encoding["Ps2RHssMajor"])
                if ndim_p > 0:
                    encoding["PsLengths"] = [1] * ndim_p
                    if st.session_state.debug_mode:
                        st.info(f"Added missing PsLengths=[{', '.join(['1'] * ndim_p)}] based on Ps2RHssMajor length {ndim_p}")
            
            # Create the tile distribution with current variables
            tile_distribution = TileDistribution(encoding, variables)
            
            # Set name and source code
            if hasattr(st.session_state, 'selected_example'):
                tile_distribution.set_tile_name(st.session_state.selected_example)
            
            if hasattr(st.session_state, 'cpp_code') and st.session_state.cpp_code:
                tile_distribution.set_source_code(st.session_state.cpp_code)
            
            # Store in session state
            st.session_state.tile_distribution = tile_distribution
            
            return True
        return False
    except Exception as e:
        st.error(f"Error creating tile distribution: {str(e)}")
        if st.session_state.debug_mode:
            st.exception(e)
        st.session_state.tile_distribution = None
        return False

def extract_transforms_from_tile_distribution(encoding_dict: Dict, variables: Dict) -> Tuple[List[Dict], List[List[int]], List[List[int]]]:
    """
    Extract transformation data from tile distribution encoding, similar to _make_adaptor_encoding_for_tile_distribution.
    
    Args:
        encoding_dict: Tile distribution encoding dictionary
        variables: Variable values
        
    Returns:
        Tuple of (transforms, lower_dimension_idss, upper_dimension_idss)
    """
    # Extract components from encoding
    rs_lengths = encoding_dict.get('RsLengths', [])
    hs_lengthss = encoding_dict.get('HsLengthss', [])
    ps_to_rhss_major = encoding_dict.get('Ps2RHssMajor', [])
    ps_to_rhss_minor = encoding_dict.get('Ps2RHssMinor', [])
    ys_to_rhs_major = encoding_dict.get('Ys2RHsMajor', [])
    ys_to_rhs_minor = encoding_dict.get('Ys2RHsMinor', [])
    
    # Resolve variables in lengths
    def resolve_value(val):
        if isinstance(val, str) and val in variables:
            return variables[val]
        elif isinstance(val, (int, float)):
            return int(val)
        else:
            return 1
    
    # Resolve all length values
    resolved_rs_lengths = [resolve_value(val) for val in rs_lengths]
    resolved_hs_lengthss = [[resolve_value(val) for val in hs_lengths] for hs_lengths in hs_lengthss]
    
    # Get dimensions
    ndim_x = len(hs_lengthss)
    
    # Initialize arrays for hidden dimensions (similar to _make_adaptor_encoding_for_tile_distribution)
    MAX_NUM_DIM = 20
    rh_major_minor_to_hidden_ids = [[0] * MAX_NUM_DIM for _ in range(ndim_x + 1)]
    rh_major_minor_to_hidden_lengths = [[0] * MAX_NUM_DIM for _ in range(ndim_x + 1)]
    
    # Initialize transforms and dimension tracking
    transforms = []
    lower_dimension_idss = []
    upper_dimension_idss = []
    hidden_dim_cnt = ndim_x  # Start after X dimensions
    
    # 1. Add replicate transform for R dimensions
    if resolved_rs_lengths:
        ndim_r_minor = len(resolved_rs_lengths)
        transforms.append({
            'type': 'replicate',
            'lengths': resolved_rs_lengths,
            'name': 'ReplicateTransform'
        })
        
        # Lower dimensions: none (replicate creates from nothing)
        lower_dimension_idss.append([])
        
        # Upper dimensions: new hidden dimensions for R
        r_upper_dims = list(range(hidden_dim_cnt, hidden_dim_cnt + ndim_r_minor))
        upper_dimension_idss.append(r_upper_dims)
        
        # Update hidden dimension mappings for R (rh_major=0)
        for i in range(ndim_r_minor):
            rh_major_minor_to_hidden_ids[0][i] = hidden_dim_cnt
            rh_major_minor_to_hidden_lengths[0][i] = resolved_rs_lengths[i]
            hidden_dim_cnt += 1
    
    # 2. Add unmerge transforms for X dimensions → H components
    for idim_x in range(ndim_x):
        h_minor_lengths = resolved_hs_lengthss[idim_x]
        ndim_h_minor = len(h_minor_lengths)
        
        if ndim_h_minor > 0:
            transforms.append({
                'type': 'unmerge',
                'lengths': h_minor_lengths,
                'name': 'UnmergeTransform'
            })
            
            # Lower dimensions: the X dimension
            lower_dimension_idss.append([idim_x])
            
            # Upper dimensions: new hidden dimensions for H
            h_upper_dims = list(range(hidden_dim_cnt, hidden_dim_cnt + ndim_h_minor))
            upper_dimension_idss.append(h_upper_dims)
            
            # Update hidden dimension mappings for H (rh_major = idim_x + 1)
            for i in range(ndim_h_minor):
                rh_major_minor_to_hidden_ids[idim_x + 1][i] = hidden_dim_cnt
                rh_major_minor_to_hidden_lengths[idim_x + 1][i] = h_minor_lengths[i]
                hidden_dim_cnt += 1
    
    # 3. Add merge transforms for P dimensions
    ndim_p = len(ps_to_rhss_major)
    # Ensure both major and minor arrays have the same length
    ndim_p_minor = len(ps_to_rhss_minor)
    actual_ndim_p = min(ndim_p, ndim_p_minor)
    
    print(f"DEBUG MERGE: ndim_p={ndim_p}, ndim_p_minor={ndim_p_minor}, actual_ndim_p={actual_ndim_p}")
    print(f"DEBUG MERGE: ps_to_rhss_major={ps_to_rhss_major}")
    print(f"DEBUG MERGE: ps_to_rhss_minor={ps_to_rhss_minor}")
    
    for i_dim_p in range(actual_ndim_p):
        p2RHsMajor = ps_to_rhss_major[i_dim_p]
        p2RHsMinor = ps_to_rhss_minor[i_dim_p]
        
        print(f"DEBUG MERGE P{i_dim_p}: p2RHsMajor={p2RHsMajor}, p2RHsMinor={p2RHsMinor}")
        print(f"DEBUG MERGE P{i_dim_p}: len check: {len(p2RHsMajor)} == {len(p2RHsMinor)} and {len(p2RHsMajor)} > 0 = {len(p2RHsMajor) == len(p2RHsMinor) and len(p2RHsMajor) > 0}")
        
        if len(p2RHsMajor) == len(p2RHsMinor) and len(p2RHsMajor) > 0:
            # Collect the hidden dimensions that this P merges from
            merge_lower_dims = []
            merge_lengths = []
            
            for j in range(len(p2RHsMajor)):
                rh_major = p2RHsMajor[j]
                rh_minor = p2RHsMinor[j]
                
                print(f"DEBUG MERGE P{i_dim_p}[{j}]: rh_major={rh_major}, rh_minor={rh_minor}")
                print(f"DEBUG MERGE P{i_dim_p}[{j}]: bounds check: {rh_major} < {len(rh_major_minor_to_hidden_ids)} and {rh_minor} < {MAX_NUM_DIM}")
                
                if rh_major < len(rh_major_minor_to_hidden_ids) and rh_minor < MAX_NUM_DIM:
                    hidden_id = rh_major_minor_to_hidden_ids[rh_major][rh_minor]
                    hidden_length = rh_major_minor_to_hidden_lengths[rh_major][rh_minor]
                    print(f"DEBUG MERGE P{i_dim_p}[{j}]: hidden_id={hidden_id}, hidden_length={hidden_length}")
                    merge_lower_dims.append(hidden_id)
                    merge_lengths.append(hidden_length)
                else:
                    print(f"DEBUG MERGE P{i_dim_p}[{j}]: BOUNDS CHECK FAILED")
            
            print(f"DEBUG MERGE P{i_dim_p}: merge_lower_dims={merge_lower_dims}")
            
            if merge_lower_dims:
                print(f"DEBUG MERGE P{i_dim_p}: CREATING MERGE TRANSFORM")
                transforms.append({
                    'type': 'merge',
                    'lengths': merge_lengths,
                    'name': 'MergeTransform'
                })
                
                # Lower dimensions: the hidden dimensions being merged
                lower_dimension_idss.append(merge_lower_dims)
                
                # Upper dimensions: new P dimension
                p_upper_dim = hidden_dim_cnt
                upper_dimension_idss.append([p_upper_dim])
                hidden_dim_cnt += 1
            else:
                print(f"DEBUG MERGE P{i_dim_p}: NO MERGE LOWER DIMS - SKIPPING")
        else:
            print(f"DEBUG MERGE P{i_dim_p}: LENGTH CHECK FAILED - SKIPPING")
    
    return transforms, lower_dimension_idss, upper_dimension_idss

def build_tile_distribution_transformation_graph(encoding_dict: Dict, variables: Dict) -> graphviz.Digraph:
    """
    Build a transformation graph for tile distribution similar to tensor_transform_app.py.
    
    Args:
        encoding_dict: Tile distribution encoding dictionary
        variables: Variable values
        
    Returns:
        Graphviz DOT graph showing the transformation pipeline
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", splines="ortho", compound="true")
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Add title
    vars_display = ", ".join(f"{k}={v}" for k, v in sorted(variables.items())[:5])
    dot.node("title", f"Tile Distribution Transformation Graph\\nVariables: {vars_display}", 
             shape="note", style="filled", fillcolor="lightyellow")
    
    try:
        # Define resolve_value function locally
        def resolve_value(val):
            if isinstance(val, str) and val in variables:
                return variables[val]
            elif isinstance(val, (int, float)):
                return int(val)
            else:
                return 1
        
        # Get input dimensions (X dimensions) and output dimensions (P,Y dimensions)
        ndim_x = len(encoding_dict.get('HsLengthss', []))
        ndim_p = len(encoding_dict.get('Ps2RHssMajor', []))
        ndim_y = len(encoding_dict.get('Ys2RHsMajor', []))
        
        # Extract transforms from tile distribution and get RH mapping
        transforms, lower_dimension_idss, upper_dimension_idss = extract_transforms_from_tile_distribution(
            encoding_dict, variables
        )
        
        # Debug output for transforms
        print(f"DEBUG TRANSFORMS: Found {len(transforms)} transforms")
        for i, (transform, lower_dims, upper_dims) in enumerate(zip(transforms, lower_dimension_idss, upper_dimension_idss)):
            print(f"  Transform {i}: {transform['type']} - {transform['name']}")
            print(f"    Lower dims: {lower_dims}")
            print(f"    Upper dims: {upper_dims}")
            if 'lengths' in transform:
                print(f"    Lengths: {transform['lengths']}")
        
        if not transforms:
            dot.node("no_transforms", "No transforms found", fillcolor="#ffcccc")
            return dot
        
        # Also extract the RH mapping for Y dimensions
        # Re-run the RH mapping logic to get the hidden dimension mappings
        rs_lengths = encoding_dict.get('RsLengths', [])
        hs_lengthss = encoding_dict.get('HsLengthss', [])
        resolved_rs_lengths = [resolve_value(val) for val in rs_lengths]
        resolved_hs_lengthss = [[resolve_value(val) for val in hs_lengths] for hs_lengths in hs_lengthss]
        
        # Rebuild the RH mapping (same logic as extract_transforms_from_tile_distribution)
        MAX_NUM_DIM = 20
        rh_major_minor_to_hidden_ids = [[0] * MAX_NUM_DIM for _ in range(ndim_x + 1)]
        hidden_dim_cnt = ndim_x
        
        # Map R dimensions
        if resolved_rs_lengths:
            for i in range(len(resolved_rs_lengths)):
                rh_major_minor_to_hidden_ids[0][i] = hidden_dim_cnt
                hidden_dim_cnt += 1
        
        # Map H dimensions
        for idim_x in range(ndim_x):
            h_minor_lengths = resolved_hs_lengthss[idim_x]
            for i in range(len(h_minor_lengths)):
                rh_major_minor_to_hidden_ids[idim_x + 1][i] = hidden_dim_cnt
                hidden_dim_cnt += 1
        
        # Create input stage - Bottom dimensions (X dimensions)
        with dot.subgraph(name='cluster_input') as input_cluster:
            input_cluster.attr(style='filled', fillcolor='#ffeeee', label='Input Stage (X Dimensions - Logical Tensor)')
            input_cluster.attr(fontsize='14', fontweight='bold')
            
            prev_stage_nodes = {}
            for x_idx in range(ndim_x):
                node_id = f"input_x{x_idx}"
                input_cluster.node(node_id, f"X{x_idx}", fillcolor="#ffcccc")
                prev_stage_nodes[x_idx] = node_id
        
        # Process each transform as a stage
        for stage_idx, (transform, lower_dims, upper_dims) in enumerate(
            zip(transforms, lower_dimension_idss, upper_dimension_idss)
        ):
            stage_color = ['#eeffee', '#eeeeff', '#ffffe0', '#f0e0ff'][stage_idx % 4]
            
            with dot.subgraph(name=f'cluster_stage_{stage_idx}') as stage_cluster:
                stage_cluster.attr(style='filled', fillcolor=stage_color, 
                                 label=f'Stage {stage_idx + 1}: {transform["name"]}', 
                                 fontsize='14', fontweight='bold')
                
                # Create output nodes for this transform
                for out_idx, upper_dim in enumerate(upper_dims):
                    node_id = f"s{stage_idx}_out{upper_dim}"
                    
                    # Create formula based on transform type
                    if transform['type'] == 'replicate':
                        formula = f"R{out_idx} (replicated)"
                        
                    elif transform['type'] == 'unmerge':
                        # For unmerge, show which X dimension is being split
                        if lower_dims and lower_dims[0] < ndim_x:
                            x_dim = lower_dims[0]
                            formula = f"X{x_dim}_H{out_idx}"
                        else:
                            formula = f"H{out_idx}"
                            
                    elif transform['type'] == 'merge':
                        # For merge, calculate which P dimension this is
                        # Count how many merge transforms we've seen so far
                        merge_count = sum(1 for i in range(stage_idx) if transforms[i]['type'] == 'merge')
                        formula = f"P{merge_count}"
                    else:
                        formula = f"dim{upper_dim}"
                    
                    # Add length information if available
                    if 'lengths' in transform and out_idx < len(transform['lengths']):
                        length = transform['lengths'][out_idx]
                        formula += f" ({length})"
                    
                    stage_cluster.node(node_id, formula, fillcolor="#c0ffc0")
                    
                    # Create edges from inputs to this output BEFORE updating prev_stage_nodes
                    transform_name = transform['name'].replace('Transform', '')
                    
                    if transform['type'] == 'replicate':
                        # Replicate has no input connections (creates from nothing)
                        pass
                    else:
                        for lower_dim in lower_dims:
                            if lower_dim in prev_stage_nodes:
                                print(f"DEBUG EDGE: Creating edge from {prev_stage_nodes[lower_dim]} -> {node_id} (label={transform_name})")
                                dot.edge(prev_stage_nodes[lower_dim], node_id, 
                                       label=transform_name)
                            else:
                                print(f"DEBUG EDGE: Missing source node for lower_dim {lower_dim} in prev_stage_nodes: {list(prev_stage_nodes.keys())}")
                    
                    # Update for next stage (do this AFTER creating edges)
                    prev_stage_nodes[upper_dim] = node_id
        
        # Create final output stage - Top dimensions (P,Y dimensions)
        if transforms:
            with dot.subgraph(name='cluster_output') as output_cluster:
                output_cluster.attr(style='filled', fillcolor='#e8ffe8', 
                                  label='Output Stage (P,Y Dimensions - Thread Mapping)', 
                                  fontsize='14', fontweight='bold')
                
                # Debug output (will be visible in debug mode)
                print(f"DEBUG Graph: ndim_p={ndim_p}, ndim_y={ndim_y}, ndim_x={ndim_x}")
                print(f"DEBUG Graph: Ps2RHssMajor={encoding_dict.get('Ps2RHssMajor', [])}")
                print(f"DEBUG Graph: Ys2RHsMajor={encoding_dict.get('Ys2RHsMajor', [])}")
                
                # Track which nodes produce P and Y outputs
                py_source_nodes = []
                
                # Find P dimension source nodes (from merge transforms)
                rs_lengths = encoding_dict.get('RsLengths', [])
                resolved_rs_lengths_local = [resolve_value(val) for val in rs_lengths]
                merge_stage_start = (1 if resolved_rs_lengths_local else 0) + ndim_x  # After replicate and unmerge
                for p_idx in range(ndim_p):
                    merge_stage_idx = merge_stage_start + p_idx
                    if merge_stage_idx < len(transforms):
                        # The merge transform outputs to a P dimension
                        for hidden_dim, node_id in prev_stage_nodes.items():
                            if f"s{merge_stage_idx}_out" in node_id:
                                py_source_nodes.append(('P', p_idx, node_id))
                                break
                
                # Find Y dimension source nodes (they are part of top_dim_ids, not transforms)
                # Y dimensions reference existing RH hidden dimensions via Ys2RHsMajor/Minor
                ys_to_rhs_major = encoding_dict.get('Ys2RHsMajor', [])
                ys_to_rhs_minor = encoding_dict.get('Ys2RHsMinor', [])
                
                for y_idx in range(ndim_y):
                    if y_idx < len(ys_to_rhs_major) and y_idx < len(ys_to_rhs_minor):
                        rh_major = resolve_value(ys_to_rhs_major[y_idx])
                        rh_minor = resolve_value(ys_to_rhs_minor[y_idx])
                        
                        # Get the exact hidden dimension ID that this Y maps to
                        if (rh_major < len(rh_major_minor_to_hidden_ids) and 
                            rh_minor < MAX_NUM_DIM):
                            target_hidden_id = rh_major_minor_to_hidden_ids[rh_major][rh_minor]
                            
                            # Find the node with this hidden dimension ID
                            for node_id in prev_stage_nodes.values():
                                if f"_out{target_hidden_id}" in node_id:
                                    py_source_nodes.append(('Y', y_idx, node_id))
                                    break
                
                # Create final P dimension outputs
                if ndim_p > 0:
                    for p_idx in range(ndim_p):
                        final_node_id = f"final_P{p_idx}"
                        output_cluster.node(final_node_id, f"P{p_idx}\\n(thread partition)", 
                                          fillcolor="#66ff66", style="filled,bold")
                        
                        # Connect from source if found
                        for dim_type, idx, src_node in py_source_nodes:
                            if dim_type == 'P' and idx == p_idx:
                                dot.edge(src_node, final_node_id, 
                                       label="Thread Mapping", color="green", style="bold")
                                break
                
                # Create final Y dimension outputs (they are top dimension IDs, not transforms)
                if ndim_y > 0:
                    for y_idx in range(ndim_y):
                        final_node_id = f"final_Y{y_idx}"
                        output_cluster.node(final_node_id, f"Y{y_idx}\\n(spatial coord)", 
                                          fillcolor="#66ccff", style="filled,bold")
                        
                        # Connect from source RH hidden dimension if found
                        for dim_type, idx, src_node in py_source_nodes:
                            if dim_type == 'Y' and idx == y_idx:
                                dot.edge(src_node, final_node_id, 
                                       label="Top Dimension", color="blue", style="bold")
                                break
                
                # Add explanation note
                note_node_id = "explanation_note"
                note_text = f"X→RH→(P,Y) Transformation Pipeline\\nInput: {ndim_x} X dims → Output: {ndim_p} P dims + {ndim_y} Y dims"
                note_text += "\\n\\nThe PsYs→Xs Adaptor uses this pipeline\\nto map (P,Y) thread coordinates to X tensor indices"
                if ndim_y > 0:
                    note_text += "\\n\\nY dims also have separate Y↔D\\ntransformation for memory layout"
                output_cluster.node(note_node_id, note_text,
                                  fillcolor="#ffffcc", style="filled,dashed", shape="note")
        
    except Exception as e:
        error_msg = f"Error building transformation graph: {str(e)}"
        dot.node("error", error_msg, fillcolor="#ffcccc")
        if st.session_state.get('debug_mode', False):
            st.exception(e)
    
    return dot

def build_y_to_d_transformation_graph(encoding_dict: Dict, variables: Dict) -> graphviz.Digraph:
    """
    Build the Y↔D transformation graph showing spatial coordinates to linearized memory mapping.
    
    Args:
        encoding_dict: Tile distribution encoding dictionary
        variables: Variable values
        
    Returns:
        Graphviz DOT graph showing the Y↔D transformation pipeline
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", splines="ortho", compound="true")
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Add title
    vars_display = ", ".join(f"{k}={v}" for k, v in sorted(variables.items())[:3])
    dot.node("y_title", f"Y↔D Transformation Graph\\nVariables: {vars_display}", 
             shape="note", style="filled", fillcolor="lightcyan")
    
    try:
        # Define resolve_value function locally
        def resolve_value(val):
            if isinstance(val, str) and val in variables:
                return variables[val]
            elif isinstance(val, (int, float)):
                return int(val)
            else:
                return 1
        
        # Get Y dimension mappings
        ys_to_rhs_major = encoding_dict.get('Ys2RHsMajor', [])
        ys_to_rhs_minor = encoding_dict.get('Ys2RHsMinor', [])
        ndim_y = len(ys_to_rhs_major)
        
        if ndim_y == 0:
            dot.node("no_y", "No Y dimensions found", fillcolor="#ffcccc")
            return dot
        
        # Calculate Y dimension lengths from RH space (same logic as _make_adaptor_encoding_for_tile_distribution)
        rs_lengths = encoding_dict.get('RsLengths', [])
        hs_lengthss = encoding_dict.get('HsLengthss', [])
        resolved_rs_lengths = [resolve_value(val) for val in rs_lengths]
        resolved_hs_lengthss = [[resolve_value(val) for val in hs_lengths] for hs_lengths in hs_lengthss]
        
        # Build RH length lookup (same as in main transform function)
        rh_major_minor_to_lengths = {}
        
        # Add R lengths (rh_major=0)
        for i, r_length in enumerate(resolved_rs_lengths):
            rh_major_minor_to_lengths[(0, i)] = r_length
        
        # Add H lengths (rh_major=1,2,3...)
        for x_idx, h_lengths in enumerate(resolved_hs_lengthss):
            for h_idx, h_length in enumerate(h_lengths):
                rh_major_minor_to_lengths[(x_idx + 1, h_idx)] = h_length
        
        # Calculate Y lengths and total D length
        y_lengths = []
        d_length = 1
        
        for y_idx in range(ndim_y):
            rh_major = ys_to_rhs_major[y_idx]
            rh_minor = ys_to_rhs_minor[y_idx]
            
            y_length = rh_major_minor_to_lengths.get((rh_major, rh_minor), 1)
            y_lengths.append(y_length)
            d_length *= y_length
        
        # Create input stage (D dimension - linearized memory)
        with dot.subgraph(name='cluster_y_input') as input_cluster:
            input_cluster.attr(style='filled', fillcolor='#ffeeee', label='Input: Linearized Memory')
            input_cluster.attr(fontsize='14', fontweight='bold')
            
            input_cluster.node("d_input", f"D (size: {d_length})", fillcolor="#ffcccc")
        
        # Create transformation stage
        with dot.subgraph(name='cluster_y_transform') as transform_cluster:
            transform_cluster.attr(style='filled', fillcolor='#eeeeff', label='UnmergeTransform')
            transform_cluster.attr(fontsize='14', fontweight='bold')
            
            # Show the unmerge operation
            unmerge_formula = f"D → Y[{', '.join(map(str, y_lengths))}]"
            transform_cluster.node("y_unmerge", unmerge_formula, fillcolor="#c0c0ff")
            
            # Connect D input to unmerge
            dot.edge("d_input", "y_unmerge", label="Unmerge")
        
        # Create output stage (Y dimensions - spatial coordinates)
        with dot.subgraph(name='cluster_y_output') as output_cluster:
            output_cluster.attr(style='filled', fillcolor='#e8ffe8', label='Output: Spatial Coordinates')
            output_cluster.attr(fontsize='14', fontweight='bold')
            
            # Create Y dimension outputs
            for y_idx in range(ndim_y):
                y_node_id = f"y_output_{y_idx}"
                y_length = y_lengths[y_idx]
                
                # Show which RH location this Y maps to
                rh_major = ys_to_rhs_major[y_idx]
                rh_minor = ys_to_rhs_minor[y_idx]
                if rh_major == 0:
                    source_info = f"from R[{rh_minor}]"
                else:
                    source_info = f"from H[{rh_major-1}][{rh_minor}]"
                
                y_label = f"Y{y_idx} (size: {y_length})\\n{source_info}"
                output_cluster.node(y_node_id, y_label, fillcolor="#66ffff")
                
                # Connect from unmerge to Y output
                dot.edge("y_unmerge", y_node_id, label=f"Y{y_idx}")
        
        # Add explanation
        with dot.subgraph(name='cluster_y_explanation') as explain_cluster:
            explain_cluster.attr(style='filled', fillcolor='#fffff0', label='Explanation')
            explain_cluster.attr(fontsize='12')
            
            explanation = f"This transforms between:\\n• Linear memory address (D)\\n• Multi-dimensional coordinates (Y)\\n\\nFormula: D = Y0*{y_lengths[1:]} + Y1*{y_lengths[2:] if len(y_lengths) > 2 else [1]} + ..."
            if len(y_lengths) <= 3:  # Simplify for small cases
                if len(y_lengths) == 2:
                    explanation = f"Formula: D = Y0*{y_lengths[1]} + Y1"
                elif len(y_lengths) == 3:
                    explanation = f"Formula: D = Y0*{y_lengths[1]*y_lengths[2]} + Y1*{y_lengths[2]} + Y2"
            
            explain_cluster.node("y_explain", explanation, fillcolor="#fffff0", shape="note")
        
    except Exception as e:
        error_msg = f"Error building Y↔D graph: {str(e)}"
        dot.node("y_error", error_msg, fillcolor="#ffcccc")
    
    return dot

if __name__ == "__main__":
    main() 