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
from tiler import TileDistribution
import visualizer
from test_visualization import create_indexing_visualization

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
            
        # Layout for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tile Distribution")
            if st.session_state.tile_distribution is not None:
                try:
                    viz_data = st.session_state.tile_distribution.get_visualization_data()
                    
                    if st.session_state.debug_mode:
                        st.write("### Visualization Data:")
                        st.json({
                            "tile_shape": viz_data.get("tile_shape"),
                            "thread_mapping": viz_data.get("thread_mapping"),
                            "dimensions": viz_data.get("dimensions")
                        })
                    
                    fig2 = visualizer.visualize_tile_distribution(viz_data)
                    st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Error visualizing tile distribution: {str(e)}")
                    if st.session_state.debug_mode:
                        st.exception(e)
        
        with col2:
            # Left empty for layout balance or future content
            pass
        
        # Thread access pattern visualization with frame control
        st.subheader("Thread Access Pattern")
        if st.session_state.tile_distribution is not None:
            try:
                viz_data = st.session_state.tile_distribution.get_visualization_data()
                
                # Frame selector slider
                num_frames = 10
                frame_idx = st.slider("Animation Frame", 0, num_frames-1, 
                                    st.session_state.current_frame,
                                    key="frame_slider")
                st.session_state.current_frame = frame_idx
                
                # Display the visualization
                fig3 = visualizer.visualize_thread_access_pattern(
                    viz_data, frame_idx, num_frames
                )
                st.pyplot(fig3)
                
                # Add enhanced thread mapping visualization
                st.subheader("Enhanced Thread Distribution")
                try:
                    if st.session_state.debug_mode:
                        # Show the indexing relationships in a more visual way
                        indexing_debug = debug_indexing_relationships(
                            st.session_state.encoding, 
                            st.session_state.variables
                        )
                        
                        # Create a tabbed view
                        tabs = st.tabs(
                            ["Thread Mapping", "Index Values", "P Mappings", "Y Mappings"],
                            key="debug_tabs"
                        )
                        
                        with tabs[0]:
                            # Display thread mapping diagram
                            st.write("#### Thread to Memory Access Mapping")
                            thread_info = viz_data.get("thread_mapping", {})
                            st.json(thread_info)
                        
                        with tabs[1]:
                            # Display the values of indices
                            st.write("#### Index Values")
                            st.json(indexing_debug["MinorIndices"])
                        
                        with tabs[2]:
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
                        
                        with tabs[3]:
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
                    st.error(f"Error in enhanced thread distribution: {str(e)}")
                    if st.session_state.debug_mode:
                        st.exception(e)
                
            except Exception as e:
                st.error(f"Error visualizing thread access pattern: {str(e)}")
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
    examples = {
        "Basic 16x4 Threads": """
        tile_distribution_encoding<
            sequence<1>,                            // 0 R
            tuple<sequence<4, 4>,                   // H (X0)
                  sequence<4, 4>>,                  // H (X1)
            tuple<sequence<1>,                      // p major
                  sequence<2>>,                     // p minor
            tuple<sequence<1>,                      // p minor
                  sequence<0>>,                     // p minor
            sequence<1, 1>,                         // Y major
            sequence<0, 1>>{}                       // y minor
        """,
        "Variable-Based Template": """
        tile_distribution_encoding<
            sequence<1>,                            // 0 R
            tuple<sequence<Nr_y, Nr_p, Nw>,         // H
                  sequence<Kr_y, Kr_p, Kw, Kv>>,    // H
            tuple<sequence<1, 2>,                   // p major
                  sequence<2, 1>>,                  // p minor
            tuple<sequence<1, 1>,                   // p minor
                  sequence<2, 2>>,                  // p minor
            sequence<1, 2, 2>,                      // Y major
            sequence<0, 0, 3>>{}                    // y minor
        """,
        "Real-World Example (RMSNorm)": """
        // From include/ck_tile/ops/add_rmsnorm2d_rdquant/pipeline
        tile_distribution_encoding<
            sequence<>,                             // Empty R
            tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
                  sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
            tuple<sequence<1, 2>, sequence<1, 2>>,
            tuple<sequence<1, 1>, sequence<2, 2>>,
            sequence<1, 1, 2, 2>,
            sequence<0, 3, 0, 3>>{}
        """,
        "Complex Distribution": """
        tile_distribution_encoding<
            sequence<1>,                            // 0 R
            tuple<sequence<16, 4>,                  // H (X0)
                  sequence<16, 4, 4>>,              // H (X1)
            tuple<sequence<1, 1>,                   // p major
                  sequence<2, 2>>,                  // p minor
            tuple<sequence<0, 1>,                   // p minor
                  sequence<0, 0>>,                  // p minor
            sequence<1, 1, 2>,                      // Y major
            sequence<1, 0, 1>>{}                    // y minor
        """
    }
    
    # Initialize the selected example in session state if not present
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = "Basic 16x4 Threads"
    
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
    variables = {}
    
    # Default values for variables
    if selected_example == "Variable-Based Template":
        variables = {
            "Nr_y": 4,
            "Nr_p": 4,
            "Nw": 8,
            "Kr_y": 4,
            "Kr_p": 8,
            "Kw": 8,
            "Kv": 4
        }
    elif selected_example == "Real-World Example (RMSNorm)":
        variables = {
            "S::Repeat_M": 4,
            "S::WarpPerBlock_M": 2,
            "S::ThreadPerWarp_M": 8,
            "S::Vector_M": 4,
            "S::Repeat_N": 4,
            "S::WarpPerBlock_N": 2,
            "S::ThreadPerWarp_N": 8,
            "S::Vector_N": 4
        }
    
    # Save to session state
    st.session_state.encoding = encoding
    st.session_state.variables = variables
    
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
            st.session_state.tile_distribution = TileDistribution(
                st.session_state.encoding,
                st.session_state.variables
            )
    except Exception as e:
        st.error(f"Error creating tile distribution: {str(e)}")
        st.session_state.tile_distribution = None

if __name__ == "__main__":
    main() 