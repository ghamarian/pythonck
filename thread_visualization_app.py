#!/usr/bin/env python3
"""
Thread Coordinate Visualization App

This Streamlit app visualizes RMSNorm tile structure and coordinate mapping.
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import os
from typing import Dict, Any, List
import seaborn as sns

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from tile_distribution import TileDistributionParser, get_examples, get_default_variables
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.tensor_coordinate import MultiIndex

# Page configuration
st.set_page_config(
    page_title="Thread Coordinate Visualization",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state  
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

def main():
    """Main application function."""
    st.title("🧵 Thread Coordinate Visualization")
    st.markdown("Interactive visualization of tile distribution structure and coordinate mapping")
    
    # Sidebar for parameters
    setup_sidebar()
    
    # Main content
    if st.session_state.encoding is not None:
        display_visualizations()
    else:
        display_welcome_message()

def setup_sidebar():
    """Setup sidebar with example selection and parameter controls."""
    st.sidebar.header("Input Options")
    
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
        selected_example = st.sidebar.selectbox(
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
            st.sidebar.error(f"No examples available: {str(e)}")
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
    col1, col2 = st.sidebar.columns([5, 2])
    
    with col1:
        st.markdown("### Code Editor")
        
    with col2:
        # Single button that changes label based on current mode
        button_label = "🎨 Format" if st.session_state.edit_mode else "✏️ Edit"
        st.button(button_label, on_click=toggle_mode, key="mode_toggle")
        
    # Display the appropriate code view based on current mode
    if st.session_state.edit_mode:
        # Edit mode: Show editable text area
        edited_code = st.sidebar.text_area(
            label="",
            value=st.session_state.current_code,
            height=300,
            key="code_editor"
        )
        # Save the edited code
        st.session_state.current_code = edited_code
    else:
        # Format mode: Show syntax-highlighted code with built-in copy button
        st.sidebar.code(st.session_state.current_code, language="cpp")
    
    # Parse button
    if st.sidebar.button("Parse Code"):
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
            
            st.sidebar.success("Code parsed successfully!")
        else:
            st.sidebar.error("Failed to parse the code. Check the syntax.")
    
    # Variable sliders (shown only if encoding is parsed)
    if st.session_state.encoding is not None:
        display_variable_controls()

def display_variable_controls():
    """Display sliders for adjusting template variables."""
    st.sidebar.header("Template Variables")
    
    # Initialize a new dictionary to store the updated variable values
    updated_variables = {}
    
    if not hasattr(st.session_state, 'parsed_variables') or not st.session_state.parsed_variables:
        st.sidebar.info("No template variables detected in code. Parse your code to find variables.")
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
        value = st.sidebar.slider(
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

def generate_visualization():
    """Generate the visualization data automatically using parsed encoding."""
    try:
        if st.session_state.encoding is not None:
            # Use the parsed encoding and variables
            encoding_data = st.session_state.encoding
            variables = st.session_state.variables
            
            # Helper function to resolve variables
            def resolve_sequence(seq):
                """Resolve variables in a sequence."""
                resolved = []
                for val in seq:
                    if isinstance(val, str) and val in variables:
                        resolved.append(variables[val])
                    elif isinstance(val, (int, float)):
                        resolved.append(int(val))
                    else:
                        resolved.append(1)  # Default fallback
                return resolved
            
            def resolve_nested_sequence(nested_seq):
                """Resolve variables in nested sequences."""
                return [resolve_sequence(seq) for seq in nested_seq]
            
            # Resolve all sequences in the encoding
            rs_lengths = resolve_sequence(encoding_data.get("RsLengths", []))
            hs_lengthss = resolve_nested_sequence(encoding_data.get("HsLengthss", []))
            ps_to_rhss_major = encoding_data.get("Ps2RHssMajor", [])
            ps_to_rhss_minor = encoding_data.get("Ps2RHssMinor", [])
            ys_to_rhs_major = encoding_data.get("Ys2RHsMajor", [])
            ys_to_rhs_minor = encoding_data.get("Ys2RHsMinor", [])
            
            # Create TileDistributionEncoding from resolved data
            encoding = TileDistributionEncoding(
                rs_lengths=rs_lengths,
                hs_lengthss=hs_lengthss,
                ps_to_rhss_major=ps_to_rhss_major,
                ps_to_rhss_minor=ps_to_rhss_minor,
                ys_to_rhs_major=ys_to_rhs_major,
                ys_to_rhs_minor=ys_to_rhs_minor
            )
            
            # Create tile distribution
            tile_distribution = make_static_tile_distribution(encoding)
            
            # Store results in session state
            st.session_state.tile_distribution = tile_distribution
            
        else:
            st.error("No encoding available. Please parse code first.")
            
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

def display_visualizations():
    """Display the main visualizations."""
    
    # Force regeneration of visualization if not already done or if variables changed
    if (st.session_state.tile_distribution is None or 
        'last_variables' not in st.session_state or 
        st.session_state.last_variables != st.session_state.variables):
        
        generate_visualization()
        st.session_state.last_variables = st.session_state.variables.copy()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Thread Access Pattern", "Hierarchical Tile Structure", "Manual Testing"])
    
    with tab1:
        display_thread_access_pattern()
    
    with tab2:
        display_hierarchical_structure()
    
    with tab3:
        display_manual_testing()

def display_thread_access_pattern():
    """Display thread access pattern as a color-coded grid."""
    st.subheader("Thread Access Pattern Visualization")
    st.write("Shows which threads access which X coordinates in the tensor")
    
    # Check if tile distribution is available
    if st.session_state.tile_distribution is None:
        st.warning("Tile distribution not available. Please ensure code is parsed successfully.")
        return
    
    try:
        vars = st.session_state.variables
        tile_distribution = st.session_state.tile_distribution
        adaptor = tile_distribution.ps_ys_to_xs_adaptor
        encoding_data = st.session_state.encoding
        
        # Calculate tensor dimensions from the H sequences
        hs_lengthss = encoding_data.get("HsLengthss", [])
        
        # Resolve variables in H sequences
        def resolve_value(val):
            if isinstance(val, str) and val in vars:
                return vars[val]
            elif isinstance(val, (int, float)):
                return int(val)
            else:
                return 1
        
        # Calculate tensor dimensions by multiplying H components
        if len(hs_lengthss) >= 2:
            h0_lengths = [resolve_value(val) for val in hs_lengthss[0]]
            h1_lengths = [resolve_value(val) for val in hs_lengthss[1]]
            m_size = np.prod(h0_lengths) if h0_lengths else 64
            n_size = np.prod(h1_lengths) if h1_lengths else 64
        else:
            # Fallback defaults
            m_size = 64
            n_size = 64
        
        # Calculate thread space bounds from P lengths
        p_lengths = tile_distribution.get_lengths() if tile_distribution else [16, 16]
        max_threads_m = p_lengths[0] - 1 if len(p_lengths) > 0 else 15
        max_threads_n = p_lengths[1] - 1 if len(p_lengths) > 1 else 15
        
        # Get the actual block size limit from partition simulator
        max_simulator_threads = 256  # Partition simulator block size limit
        
        # Configuration options
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            show_warps = st.checkbox("Show different warps", value=True, help="Show threads from different warps")
        with col2:
            show_cycles = st.checkbox("Show Y coordinate cycles", value=True, help="Show different Y coordinate iterations")
        with col3:
            num_threads_to_show = st.slider("Number of threads", 1, 16, 8, help="Total number of threads to analyze")
        with col4:
            # Ensure slider has valid range - min must be less than max
            min_display = 64
            max_possible_display = max(512, max(m_size, n_size) * 2)  # Allow display larger than tensor
            default_display = max(min_display + 1, min(256, max(m_size, n_size)))  # Reasonable default
            
            max_display_size = st.slider("Max display size", min_display, max_possible_display, default_display)
        with col5:
            y_limit = st.slider("Y range limit", 1, 4, 4, help="Limit Y ranges for performance (1=minimal, 4=full)")
        
        # Calculate display size
        m_display = min(m_size, max_display_size)
        n_display = min(n_size, max_display_size)
        
        st.write(f"Tensor size: {m_size}×{n_size}, Displaying: {m_display}×{n_display}")
        st.write(f"Thread space: {max_threads_m+1}×{max_threads_n+1} threads")
        st.write(f"Simulator limit: {max_simulator_threads} threads max")
        
        # Create grid to track which thread accesses each coordinate
        access_grid = np.full((m_display, n_display), -1, dtype=int)  # -1 means no access
        
        # Calculate access patterns for different threads and cycles
        thread_coords_list = []
        thread_colors = []
        
        # Generate thread positions to test - show diverse threads that create different patterns
        thread_positions = []
        
        if show_warps:
            # Show threads from different warps AND different positions within warps
            warp_size = 64  # Standard warp size
            max_warps = min(4, max_simulator_threads // warp_size)  # Limit to simulator capacity
            
            # Distribute threads across warps and positions for maximum diversity
            threads_added = 0
            for warp_id in range(max_warps):
                if threads_added >= num_threads_to_show:
                    break
                    
                # Add threads from different lanes in this warp to show spatial diversity
                lanes_to_sample = min(num_threads_to_show - threads_added, 
                                    max(1, num_threads_to_show // max_warps))
                
                # Sample lanes with some spacing to get different access patterns
                for lane_offset in range(0, min(warp_size, lanes_to_sample * 8), 8):
                    if threads_added >= num_threads_to_show:
                        break
                        
                    thread_id = warp_id * warp_size + lane_offset
                    if thread_id < max_simulator_threads:  # Stay within simulator limits
                        # Calculate logical thread position (not used for partition simulation)
                        thread_m = thread_id // (max_threads_n + 1) if (max_threads_n + 1) > 0 else 0
                        thread_n = thread_id % (max_threads_n + 1) if (max_threads_n + 1) > 0 else 0
                        thread_positions.append((thread_id, thread_m, thread_n, warp_id, lane_offset))
                        threads_added += 1
        else:
            # Show threads with spatial diversity (not just sequential)
            # Limit to simulator capacity and distribute evenly
            available_threads = min(max_simulator_threads, (max_threads_m + 1) * (max_threads_n + 1))
            thread_step = max(1, available_threads // num_threads_to_show)
            
            for i in range(0, min(available_threads, num_threads_to_show * thread_step), thread_step):
                thread_id = i
                if thread_id < max_simulator_threads:  # Double-check simulator limits
                    thread_m = thread_id // (max_threads_n + 1) if (max_threads_n + 1) > 0 else 0
                    thread_n = thread_id % (max_threads_n + 1) if (max_threads_n + 1) > 0 else 0
                    warp_id = thread_id // 64
                    lane_id = thread_id % 64
                    thread_positions.append((thread_id, thread_m, thread_n, warp_id, lane_id))
        
        # Generate colors for threads (different colors for different warps)
        if show_warps:
            # Color by warp
            warp_colors = plt.cm.Set1(np.linspace(0, 1, 4))  # 4 different warp colors
            thread_colors = []
            for _, _, _, warp_id, _ in thread_positions:
                thread_colors.append(warp_colors[warp_id % 4])
        else:
            # Color by thread with more distinct colors - use multiple colormaps for better distinction
            num_threads = len(thread_positions)
            thread_colors = []
            
            if num_threads <= 10:
                # Use tab10 for small numbers
                thread_colors = plt.cm.tab10(np.linspace(0, 1, num_threads))
            elif num_threads <= 20:
                # Combine tab10 and tab20 for medium numbers
                colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
                colors2 = plt.cm.tab20(np.linspace(0.5, 1, num_threads - 10))  # Use second half of tab20
                thread_colors = np.concatenate([colors1, colors2])
            else:
                # For large numbers, use HSV colorspace for maximum distinction
                # Generate evenly spaced hues with high saturation and value
                hues = np.linspace(0, 1, num_threads, endpoint=False)
                thread_colors = []
                for i, hue in enumerate(hues):
                    # Vary saturation and value slightly to create more distinction
                    sat = 0.7 + 0.3 * (i % 3) / 2  # Saturation between 0.7-1.0
                    val = 0.8 + 0.2 * ((i + 1) % 2)  # Value between 0.8-1.0
                    color = plt.cm.hsv(hue)
                    # Apply saturation and value modifications
                    import matplotlib.colors as mcolors
                    hsv_color = mcolors.rgb_to_hsv(color[:3])
                    hsv_color[1] = sat  # Set saturation
                    hsv_color[2] = val  # Set value
                    rgb_color = mcolors.hsv_to_rgb(hsv_color)
                    thread_colors.append((*rgb_color, 1.0))  # Add alpha
                thread_colors = np.array(thread_colors)
        
        # Show which threads we'll analyze
        thread_list_str = ", ".join([f"W{t[3]}L{t[4]}" for t in thread_positions[:6]]) + ("..." if len(thread_positions) > 6 else "")
        st.write(f"Analyzing threads: {thread_list_str}")
        
        # Calculate Y dimension ranges from the parsed encoding
        ys_to_rhs_major = encoding_data.get("Ys2RHsMajor", [])
        ys_to_rhs_minor = encoding_data.get("Ys2RHsMinor", [])

        # Calculate Y dimension ranges - use user-controlled limit
        y_ranges = []
        for y_idx in range(len(ys_to_rhs_major)):
            if y_idx < len(ys_to_rhs_minor):
                rh_major = ys_to_rhs_major[y_idx]
                rh_minor = ys_to_rhs_minor[y_idx]
                
                # Get the length from the appropriate sequence
                if rh_major == 0:  # R sequence
                    rs_lengths = encoding_data.get("RsLengths", [])
                    if rh_minor < len(rs_lengths):
                        y_length = resolve_value(rs_lengths[rh_minor])
                    else:
                        y_length = 1
                else:  # H sequence
                    h_idx = rh_major - 1
                    if h_idx < len(hs_lengthss) and rh_minor < len(hs_lengthss[h_idx]):
                        y_length = resolve_value(hs_lengthss[h_idx][rh_minor])
                    else:
                        y_length = 1
                
                # Apply user-controlled Y range limit
                y_ranges.append(min(y_length, y_limit))

        # If no Y ranges found, use default
        if not y_ranges:
            y_ranges = [y_limit] * max(1, len(ys_to_rhs_major))

        # Store the actual number of Y dimensions
        num_y_dims = len(y_ranges)
        
        # Calculate total Y combinations for info
        total_y_combinations = 1
        for r in y_ranges:
            total_y_combinations *= r
        
        st.write(f"Y dimensions: {num_y_dims}, Y ranges: {y_ranges} → {total_y_combinations} Y combinations per thread")
        
        # Show expected blocks for full pattern
        if y_ranges == [4, 4, 4, 4]:
            expected_blocks = 16
            if m_display < 256 or n_display < 256:
                visible_blocks = min(expected_blocks, (m_display // 64) * (n_display // 64))
                st.warning(f"⚠️ Full pattern has {expected_blocks} blocks, but only {visible_blocks} visible with {m_display}×{n_display} display. Increase display size to 256+ to see all blocks.")
            else:
                st.success(f"✅ Display size sufficient to show all {expected_blocks} blocks of the full pattern.")

        # For each thread, calculate its access pattern
        for thread_idx, (thread_id, thread_m, thread_n, warp_id, lane_id) in enumerate(thread_positions):
            # Set the thread position using linear thread ID
            from pytensor.partition_simulation import set_global_thread_position_from_id
            
            try:
                set_global_thread_position_from_id(thread_id)
                partition_idx = tile_distribution.get_partition_index()
            except ValueError as e:
                st.error(f"Thread ID {thread_id} failed: {str(e)}")
                continue  # Skip this thread and continue with others
            
            # Calculate all X coordinates this thread accesses
            thread_x_coords = []
            
            # Generate all Y coordinate combinations dynamically
            def generate_y_combinations(y_ranges):
                """Generate all combinations of Y coordinates."""
                if not y_ranges:
                    return [[]]
                
                combinations = []
                
                def backtrack(current_combo, dim_idx):
                    if dim_idx == len(y_ranges):
                        combinations.append(current_combo[:])
                        return
                    
                    for y_val in range(y_ranges[dim_idx]):
                        current_combo.append(y_val)
                        backtrack(current_combo, dim_idx + 1)
                        current_combo.pop()
                
                backtrack([], 0)
                return combinations
            
            # Iterate through Y coordinate combinations for this thread
            if show_cycles:
                # Show all Y coordinate combinations (cycles)
                y_combinations = generate_y_combinations(y_ranges)
            else:
                # Show only Y=[0,0,0,0] (baseline coordinates)
                y_combinations = [[0] * len(y_ranges)]
            
            for y_coords in y_combinations:
                # Pad y_coords to match expected length if needed
                ps_ys_coords = partition_idx + y_coords
                
                try:
                    multi_idx = MultiIndex(len(ps_ys_coords), ps_ys_coords)
                    x_coord = adaptor.calculate_bottom_index(multi_idx)
                    x_coords = x_coord.to_list()
                    
                    x0, x1 = x_coords[0], x_coords[1]
                    
                    # Only record if within display bounds
                    if 0 <= x0 < m_display and 0 <= x1 < n_display:
                        thread_x_coords.append((x0, x1))
                        # Mark this cell as accessed by this thread
                        if access_grid[x0, x1] == -1:  # First thread to access this cell
                            access_grid[x0, x1] = thread_idx
                except Exception as ex:
                    pass
            
            thread_coords_list.append({
                'thread_id': (warp_id, lane_id),
                'coords': thread_x_coords,
                'color': thread_colors[thread_idx]
            })
        
        # Create visualization with better layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
        
        # Left plot: Access grid heatmap (larger)
        # Create custom colormap
        colors = ['white'] + [thread_colors[i] for i in range(len(thread_positions))]
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(colors)
        
        im = ax1.imshow(access_grid, cmap=custom_cmap, vmin=-1, vmax=len(thread_positions)-1, 
                       origin='upper', aspect='equal')
        
        title = f'Thread Access Pattern\n({m_display}×{n_display} tensor region)'
        if show_warps and show_cycles:
            title += '\n(Multiple Warps + Y Cycles)'
        elif show_warps:
            title += '\n(Multiple Warps)'
        elif show_cycles:
            title += '\n(Y Cycles)'
        
        ax1.set_title(title)
        ax1.set_xlabel('X1 (columns)')
        ax1.set_ylabel('X0 (rows)')
        
        # Add grid
        ax1.set_xticks(np.arange(-0.5, n_display, 16), minor=True)
        ax1.set_yticks(np.arange(-0.5, m_display, 16), minor=True)
        ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Right plot: Compact legend and statistics
        ax2.axis('off')
        
        # Create compact legend
        legend_y = 0.95
        ax2.text(0.05, legend_y, 'Threads', fontsize=14, fontweight='bold', transform=ax2.transAxes)
        legend_y -= 0.05
        
        # Show only first few threads in legend to save space
        max_legend_items = min(8, len(thread_coords_list))
        for i in range(max_legend_items):
            thread_info = thread_coords_list[i]
            warp_id, lane_id = thread_info['thread_id']
            num_accesses = len(thread_info['coords'])
            color = thread_info['color']
            
            # Draw small color patch
            rect = patches.Rectangle((0.05, legend_y - 0.02), 0.02, 0.02, 
                                   facecolor=color, edgecolor='black', transform=ax2.transAxes)
            ax2.add_patch(rect)
            
            # Add compact text
            ax2.text(0.10, legend_y - 0.01, f'W{warp_id}L{lane_id}({num_accesses})',
                    fontsize=10, transform=ax2.transAxes, va='center')
            
            legend_y -= 0.04
        
        if len(thread_coords_list) > max_legend_items:
            ax2.text(0.05, legend_y, f'... +{len(thread_coords_list) - max_legend_items} more',
                    fontsize=10, transform=ax2.transAxes, style='italic')
            legend_y -= 0.06
        
        # Add statistics
        unique_accesses = np.sum(access_grid >= 0)
        total_cells = m_display * n_display
        coverage = (unique_accesses / total_cells) * 100
        
        legend_y -= 0.05
        ax2.text(0.05, legend_y, f'Statistics:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
        legend_y -= 0.04
        ax2.text(0.05, legend_y, f'Coverage: {coverage:.1f}%', fontsize=10, transform=ax2.transAxes)
        legend_y -= 0.03
        ax2.text(0.05, legend_y, f'Threads: {len(thread_coords_list)}', fontsize=10, transform=ax2.transAxes)
        legend_y -= 0.03
        ax2.text(0.05, legend_y, f'Y ranges: {y_ranges}', fontsize=10, transform=ax2.transAxes)
        
        # Add compact pattern insights
        legend_y -= 0.06
        ax2.text(0.05, legend_y, f'Insights:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
        legend_y -= 0.04
        
        if show_warps:
            ax2.text(0.05, legend_y, f'• Colors by warp', fontsize=10, transform=ax2.transAxes)
            legend_y -= 0.03
            ax2.text(0.05, legend_y, f'• Warps spread horizontally', fontsize=10, transform=ax2.transAxes)
            legend_y -= 0.03
        
        if show_cycles:
            ax2.text(0.05, legend_y, f'• Y cycles spread spatially', fontsize=10, transform=ax2.transAxes)
            legend_y -= 0.03
        else:
            ax2.text(0.05, legend_y, f'• Y[0,0,0,0] only', fontsize=10, transform=ax2.transAxes)
            legend_y -= 0.03
        
        ax2.text(0.05, legend_y, f'• White = unaccessed', fontsize=10, transform=ax2.transAxes)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error creating thread access visualization: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

def display_hierarchical_structure():
    """Display the hierarchical tile structure."""
    st.subheader("Hierarchical Tile Structure")
    
    # Check if encoding data is available
    if st.session_state.encoding is None:
        st.warning("Encoding data not available. Please parse code first.")
        return
    
    try:
        # Display configuration metrics from parsed encoding
        encoding_data = st.session_state.encoding
        vars = st.session_state.variables
        
        # Get H sequences and resolve variables
        hs_lengthss = encoding_data.get("HsLengthss", [])
        
        def resolve_value(val):
            if isinstance(val, str) and val in vars:
                return vars[val]
            elif isinstance(val, (int, float)):
                return int(val)
            else:
                return 1
        
        # Calculate dimensions from H sequences
        if len(hs_lengthss) >= 2:
            h0_resolved = [resolve_value(val) for val in hs_lengthss[0]]
            h1_resolved = [resolve_value(val) for val in hs_lengthss[1]]
            
            # Display the hierarchical components 
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**H0 Dimension (typically M):**")
                for i, (orig, resolved) in enumerate(zip(hs_lengthss[0], h0_resolved)):
                    st.write(f"  H0[{i}]: {orig} = {resolved}")
                st.write(f"  **Total M size:** {np.prod(h0_resolved)}")
            
            with col2:
                st.write("**H1 Dimension (typically N):**")
                for i, (orig, resolved) in enumerate(zip(hs_lengthss[1], h1_resolved)):
                    st.write(f"  H1[{i}]: {orig} = {resolved}")
                st.write(f"  **Total N size:** {np.prod(h1_resolved)}")
            
            # Calculate thread and element counts
            total_elements_m = np.prod(h0_resolved)
            total_elements_n = np.prod(h1_resolved)
            total_elements = total_elements_m * total_elements_n
            
            # Get P lengths for thread count
            p_lengths = st.session_state.tile_distribution.get_lengths() if st.session_state.tile_distribution else [16, 16]
            total_threads = np.prod(p_lengths)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Threads", total_threads)
            with col2:
                st.metric("Elements per Thread", total_elements // total_threads if total_threads > 0 else 0)
            with col3:
                st.metric("Total Tensor Elements", total_elements)
        
        # Create simple hierarchical visualization if we have parsed variables
        if vars:
            fig = create_simple_hierarchical_plot_parsed(encoding_data, vars)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        
        # Show encoding details
        st.subheader("Encoding Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**H Lengths:**")
            for i, h_lengths in enumerate(hs_lengthss):
                st.write(f"H{i}: {h_lengths}")
        
        with col2:
            st.write("**Dimension Mappings:**")
            st.write(f"Ps2RHssMajor: {encoding_data.get('Ps2RHssMajor', [])}")
            st.write(f"Ps2RHssMinor: {encoding_data.get('Ps2RHssMinor', [])}")
            st.write(f"Ys2RHsMajor: {encoding_data.get('Ys2RHsMajor', [])}")
            st.write(f"Ys2RHsMinor: {encoding_data.get('Ys2RHsMinor', [])}")
            
    except Exception as e:
        st.error(f"Error creating hierarchical visualization: {str(e)}")

def create_simple_hierarchical_plot(variables: Dict[str, int]):
    """Create a simple hierarchical plot showing the tile structure."""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Colors for different levels
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        
        # Calculate dimensions
        repeat_m, repeat_n = variables["S::Repeat_M"], variables["S::Repeat_N"]
        warp_m, warp_n = variables["S::WarpPerBlock_M"], variables["S::WarpPerBlock_N"]
        thread_m, thread_n = variables["S::ThreadPerWarp_M"], variables["S::ThreadPerWarp_N"]
        vector_m, vector_n = variables["S::Vector_M"], variables["S::Vector_N"]
        
        # Draw nested rectangles to show hierarchy
        # Note: Only Vector is actual vectorization, others are sweep_tile iterations
        levels = [
            ("Repeat (sweep)", repeat_m, repeat_n, colors[0], 4),
            ("WarpPerBlock (threads)", warp_m, warp_n, colors[1], 3),
            ("ThreadPerWarp (threads)", thread_m, thread_n, colors[2], 2),
            ("Vector (SIMD)", vector_m, vector_n, colors[3], 1)
        ]
        
        y_offset = 0
        for level_name, dim_m, dim_n, color, line_width in levels:
            # Draw main rectangle
            rect = patches.Rectangle((0, y_offset), 10, 2, 
                                   linewidth=line_width, edgecolor='black', 
                                   facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Add level label
            ax.text(-0.5, y_offset + 1, level_name, fontsize=12, fontweight='bold',
                   ha='right', va='center')
            
            # Add dimension info
            ax.text(5, y_offset + 1, f"{dim_m} × {dim_n}", fontsize=11,
                   ha='center', va='center', fontweight='bold')
            
            # Draw grid to show subdivision
            if dim_m > 1:
                for i in range(1, dim_m):
                    ax.axvline(x=i * (10/dim_m), ymin=(y_offset+0.1)/10, ymax=(y_offset+1.9)/10, 
                              color='black', linewidth=1, alpha=0.5)
            if dim_n > 1:
                for j in range(1, dim_n):
                    ax.axhline(y=y_offset + j * (2/dim_n), xmin=0.01, xmax=0.99,
                              color='black', linewidth=1, alpha=0.5)
            
            y_offset += 2.5
        
        # Set axis properties
        ax.set_xlim(-2, 12)
        ax.set_ylim(-0.5, y_offset)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        ax.text(5, y_offset + 0.5, 'Hierarchical Tile Structure', 
               fontsize=16, fontweight='bold', ha='center')
        
        # Add total calculation
        total = repeat_m * repeat_n * warp_m * warp_n * thread_m * thread_n * vector_m * vector_n
        ax.text(5, -0.3, f'Total Elements: {total}', 
               fontsize=12, ha='center', style='italic')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def create_simple_hierarchical_plot_parsed(encoding_data, variables):
    """Create a simple hierarchical plot showing the tile structure from parsed data."""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Colors for different levels
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        
        # Get H sequences and resolve variables
        hs_lengthss = encoding_data.get("HsLengthss", [])
        
        def resolve_value(val):
            if isinstance(val, str) and val in variables:
                return variables[val]
            elif isinstance(val, (int, float)):
                return int(val)
            else:
                return 1
        
        # Calculate dimensions from H sequences
        if len(hs_lengthss) >= 2:
            h0_resolved = [resolve_value(val) for val in hs_lengthss[0]]
            h1_resolved = [resolve_value(val) for val in hs_lengthss[1]]
            
            # Create hierarchical plot similar to the original function
            # Draw nested rectangles to show hierarchy
            levels = []
            level_names = []
            for i, (h0_val, h1_val) in enumerate(zip(h0_resolved, h1_resolved)):
                level_name = f"Level {i}: H0={h0_val}, H1={h1_val}"
                levels.append((h0_val, h1_val, colors[i % len(colors)], 4-i))
                level_names.append(level_name)
            
            y_offset = 0
            for (dim_m, dim_n, color, line_width), level_name in zip(levels, level_names):
                # Draw main rectangle
                rect = patches.Rectangle((0, y_offset), 10, 2, 
                                       linewidth=line_width, edgecolor='black', 
                                       facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                
                # Add level label
                ax.text(-0.5, y_offset + 1, level_name, fontsize=12, fontweight='bold',
                       ha='right', va='center')
                
                # Add dimension info
                ax.text(5, y_offset + 1, f"{dim_m} × {dim_n}", fontsize=11,
                       ha='center', va='center', fontweight='bold')
                
                # Draw grid to show subdivision
                if dim_m > 1:
                    for i in range(1, dim_m):
                        ax.axvline(x=i * (10/dim_m), ymin=(y_offset+0.1)/10, ymax=(y_offset+1.9)/10, 
                                  color='black', linewidth=1, alpha=0.5)
                if dim_n > 1:
                    for j in range(1, dim_n):
                        ax.axhline(y=y_offset + j * (2/dim_n), xmin=0.01, xmax=0.99,
                                  color='black', linewidth=1, alpha=0.5)
                
                y_offset += 2.5
            
            # Set axis properties
            ax.set_xlim(-2, 12)
            ax.set_ylim(-0.5, y_offset)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Add title
            ax.text(5, y_offset + 0.5, 'Hierarchical Tile Structure (Parsed)', 
                   fontsize=16, fontweight='bold', ha='center')
            
            # Add total calculation
            total_m = np.prod(h0_resolved)
            total_n = np.prod(h1_resolved)
            total = total_m * total_n
            ax.text(5, -0.3, f'Total Elements: {total} ({total_m} × {total_n})', 
                   fontsize=12, ha='center', style='italic')
            
            plt.tight_layout()
            return fig
        else:
            # Fallback for insufficient data
            ax.text(0.5, 0.5, 'Insufficient H sequence data for visualization', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return fig
            
    except Exception as e:
        # Create error plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, f'Error creating plot: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.axis('off')
        return fig

def display_manual_testing():
    """Display manual coordinate testing interface."""
    st.subheader("Manual Coordinate Testing")
    st.write("Test how P and Y coordinates map to X coordinates in real-time")
    
    # Check if required data is available
    if st.session_state.encoding is None or st.session_state.tile_distribution is None:
        st.warning("Encoding data or tile distribution not available. Please parse code first.")
        return
    
    # Get parsed encoding data and variables
    encoding_data = st.session_state.encoding
    vars = st.session_state.variables
    
    # Calculate Y dimension ranges from parsed encoding
    ys_to_rhs_major = encoding_data.get("Ys2RHsMajor", [])
    ys_to_rhs_minor = encoding_data.get("Ys2RHsMinor", [])
    hs_lengthss = encoding_data.get("HsLengthss", [])
    
    def resolve_value(val):
        if isinstance(val, str) and val in vars:
            return vars[val]
        elif isinstance(val, (int, float)):
            return int(val)
        else:
            return 1
    
    # Calculate Y ranges
    y_ranges = []
    for y_idx in range(len(ys_to_rhs_major)):
        if y_idx < len(ys_to_rhs_minor):
            rh_major = ys_to_rhs_major[y_idx]
            rh_minor = ys_to_rhs_minor[y_idx]
            
            # Get the length from the appropriate sequence
            if rh_major == 0:  # R sequence
                rs_lengths = encoding_data.get("RsLengths", [])
                if rh_minor < len(rs_lengths):
                    y_length = resolve_value(rs_lengths[rh_minor])
                else:
                    y_length = 1
            else:  # H sequence
                h_idx = rh_major - 1
                if h_idx < len(hs_lengthss) and rh_minor < len(hs_lengthss[h_idx]):
                    y_length = resolve_value(hs_lengthss[h_idx][rh_minor])
                else:
                    y_length = 1
            
            y_ranges.append(y_length)
    
    # Store the actual number of Y dimensions
    num_y_dims = len(y_ranges)
    
    # Create input controls with calculated ranges - dynamic number of columns
    num_cols = 2 + num_y_dims  # P0, P1 + Y dimensions
    cols = st.columns(num_cols)
    
    # Calculate max values from parsed parameters
    # P coordinates come from the tile distribution lengths
    p_lengths = st.session_state.tile_distribution.get_lengths() if st.session_state.tile_distribution else [16, 16]
    max_p0 = p_lengths[0] - 1 if len(p_lengths) > 0 else 15
    max_p1 = p_lengths[1] - 1 if len(p_lengths) > 1 else 15
    
    # P coordinates
    with cols[0]:
        p0 = st.number_input("P0", value=0, min_value=0, max_value=max_p0, key="manual_p0")
    with cols[1]:
        p1 = st.number_input("P1", value=0, min_value=0, max_value=max_p1, key="manual_p1")
    
    # Y coordinates - dynamic based on actual number of dimensions
    y_coords = []
    for y_idx in range(num_y_dims):
        max_y = max(1, y_ranges[y_idx] - 1)
        with cols[2 + y_idx]:
            y_val = st.number_input(f"Y{y_idx}", value=0, min_value=0, max_value=max_y, key=f"manual_y{y_idx}")
            y_coords.append(y_val)
    
    # Test coordinate automatically (no button needed)
    if st.session_state.tile_distribution is not None:
        ps_ys_coords = [p0, p1] + y_coords
        adaptor = st.session_state.tile_distribution.ps_ys_to_xs_adaptor
        
        # Show dimension information for debugging
        ndim_p = st.session_state.tile_distribution.ndim_p
        ndim_y = st.session_state.tile_distribution.ndim_y
        expected_length = ndim_p + ndim_y
        
        st.write(f"**Debug Info:** NDimP={ndim_p}, NDimY={ndim_y}, Expected PS_YS length={expected_length}, Actual length={len(ps_ys_coords)}")
        st.write(f"**Y Calculation:** num_y_dims={num_y_dims}, y_ranges={y_ranges}")
        st.write(f"**P Calculation:** P lengths={st.session_state.tile_distribution.get_lengths()}")
        
        # Show the encoding structure for debugging
        encoding_data = st.session_state.encoding
        st.write(f"**Encoding Debug:**")
        st.write(f"  - Ps2RHssMajor: {encoding_data.get('Ps2RHssMajor', [])}")
        st.write(f"  - Ps2RHssMinor: {encoding_data.get('Ps2RHssMinor', [])}")
        st.write(f"  - Ys2RHsMajor: {encoding_data.get('Ys2RHsMajor', [])}")
        st.write(f"  - Ys2RHsMinor: {encoding_data.get('Ys2RHsMinor', [])}")
        
        try:
            # Validate coordinate length
            if len(ps_ys_coords) != expected_length:
                st.error(f"Coordinate length mismatch! Expected {expected_length} dimensions but got {len(ps_ys_coords)}. PS_YS={ps_ys_coords}")
                st.write(f"**Analysis:** P coords=[{p0}, {p1}] (length 2), Y coords={y_coords} (length {len(y_coords)})")
                
                # Try to understand the discrepancy
                if hasattr(st.session_state.tile_distribution, 'ys_to_d_descriptor'):
                    y_desc_lengths = st.session_state.tile_distribution.ys_to_d_descriptor.get_lengths()
                    st.write(f"**Y descriptor lengths:** {y_desc_lengths} (length {len(y_desc_lengths)})")
                
                return
                
            multi_idx = MultiIndex(len(ps_ys_coords), ps_ys_coords)
            x_coord = adaptor.calculate_bottom_index(multi_idx)
            x_coords = x_coord.to_list()
            
            # Display result with better formatting
            st.markdown("### Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Input:** PS_YS{ps_ys_coords}")
            with col2:
                st.success(f"**Output:** X{x_coords}")
            
            # Test a few variations to show P dimension effects
            st.subheader("P Dimension Testing")
            st.write("**Testing different P values (should affect both X dimensions):**")
            
            test_coords = [
                ([0, 0] + y_coords, "Baseline"),
                ([1, 0] + y_coords, "P0=1"),
                ([0, 1] + y_coords, "P1=1"),
                ([1, 1] + y_coords, "P0=1,P1=1")
            ]
            
            for test_ps_ys, label in test_coords:
                if test_ps_ys[0] <= max_p0 and test_ps_ys[1] <= max_p1:
                    try:
                        if len(test_ps_ys) == expected_length:
                            test_multi_idx = MultiIndex(len(test_ps_ys), test_ps_ys)
                            test_x_coord = adaptor.calculate_bottom_index(test_multi_idx)
                            test_x_coords = test_x_coord.to_list()
                            st.write(f"  • {label}: PS_YS{test_ps_ys} → X{test_x_coords}")
                        else:
                            st.write(f"  • {label}: SKIPPED (wrong length: {len(test_ps_ys)})")
                    except Exception as e:
                        st.write(f"  • {label}: ERROR - {str(e)}")
            
            # Show breakdown
            st.subheader("Breakdown")
            st.write(f"• **Partition (P):** [{p0}, {p1}] - Thread/Warp positioning")
            st.write(f"  - P0 range: 0 to {max_p0} (from tile distribution lengths)")
            st.write(f"  - P1 range: 0 to {max_p1} (from tile distribution lengths)")
            st.write(f"• **Y coordinates:** {y_coords} - Element indexing within thread")
            st.write(f"  - Y ranges: {y_ranges} (from parsed encoding)")
            st.write(f"  - Y dimensions: {num_y_dims} (actual), {ndim_y} (expected)")
            st.write(f"• **Result X coordinates:** [{x_coords[0] if len(x_coords) > 0 else 0}, {x_coords[1] if len(x_coords) > 1 else 0}] - Final tensor coordinates")
            
            # Show dimension effects
            st.subheader("Dimension Effects")
            st.write("**How each coordinate affects the mapping:**")
            st.write("- **P0**: Thread position - affects tensor coordinates based on P→RH mapping")
            st.write("- **P1**: Thread position - affects tensor coordinates based on P→RH mapping")
            for y_idx in range(num_y_dims):
                st.write(f"- **Y{y_idx}**: Spatial coordinate - maps to RH space via Y→RH mapping")
            
            st.write("**Key insight:** The exact effects depend on the parsed encoding structure's P→RH and Y→RH mappings.")
            
        except Exception as e:
            st.error(f"Error calculating coordinates: {str(e)}")
            st.write(f"Debug: PS_YS coords = {ps_ys_coords}")
            st.write(f"Debug: Expected length = {expected_length}, Actual length = {len(ps_ys_coords)}")
            import traceback
            st.text(traceback.format_exc())

def display_welcome_message():
    """Display welcome message when no data is loaded."""
    st.info("👈 Select an example and parse code in the sidebar to see the visualization.")
    
    st.markdown("""
    ## Thread Coordinate Visualization
    
    This app shows how tile distribution encodings create thread access patterns and coordinate mappings.
    
    ### Features:
    - **Thread Access Pattern**: Color-coded visualization showing which threads access which tensor coordinates
    - **Hierarchical Structure**: Visual representation of the tile hierarchy from parsed encoding
    - **Manual Testing**: Interactive coordinate mapping testing to see how P and Y coordinates map to X coordinates
    - **Real-time Updates**: Automatically updates when you change parameters
    
    ### How to use:
    1. **Select an Example**: Choose from predefined tile distribution encodings in the sidebar
    2. **Parse Code**: Click "Parse Code" to analyze the selected encoding
    3. **Adjust Variables**: Use the sliders to modify template variables
    4. **Explore Visualizations**: View thread access patterns, hierarchical structure, and test coordinates
    
    ### Understanding the Encoding:
    - **P dimensions**: Thread partition coordinates (which thread you are)
    - **Y dimensions**: Spatial output coordinates (which element in the output you're computing)
    - **X dimensions**: Final tensor coordinates (where the result goes in the tensor)
    - **H sequences**: Hierarchical tile structure components
    - **R sequences**: Replication factors for data sharing
    
    The visualization helps you understand how threads are mapped to tensor elements and how data flows through the computation.
    """)

if __name__ == "__main__":
    main() 