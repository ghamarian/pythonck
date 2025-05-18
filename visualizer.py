"""
Visualizer for tile_distribution_encoding structures in Composable Kernels.

This module contains functions for visualizing C++ template parameters from
tile_distribution_encoding structures used in the Composable Kernels library.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from parser import TileDistributionParser

def display_raw_encoding(encoding: Dict[str, Any]) -> str:
    """
    Display the raw encoding structure in a human-readable format with Y_dims mapping.
    
    Args:
        encoding: The parsed tile_distribution_encoding structure
        
    Returns:
        String representation of the encoding with explanations
    """
    # Extract key components from encoding
    rs_lengths = encoding.get('RsLengths', [])
    hs_lengthss = encoding.get('HsLengthss', [])
    ps_to_rhs_major = encoding.get('Ps2RHssMajor', [])
    ps_to_rhs_minor = encoding.get('Ps2RHssMinor', [])
    ys_to_rhs_major = encoding.get('Ys2RHsMajor', [])
    ys_to_rhs_minor = encoding.get('Ys2RHsMinor', [])
    
    # Build output string
    output = []
    output.append("=== RAW ENCODING STRUCTURE ===\n")
    
    # RsLengths
    output.append("RsLengths:       {}".format(rs_lengths))
    
    # HsLengthss
    output.append("HsLengthss:")
    for i, hs_lengths in enumerate(hs_lengthss):
        output.append(f"  H{i} (X{i}):     {hs_lengths}")
    
    # Ps mappings
    output.append("\nPs2RHssMajor:")
    for i, majors in enumerate(ps_to_rhs_major):
        output.append(f"  P{i} Major:     {majors}")
    
    output.append("Ps2RHssMinor:")
    for i, minors in enumerate(ps_to_rhs_minor):
        output.append(f"  P{i} Minor:     {minors}")
    
    # Ys mappings
    output.append("\nYs2RHsMajor:     {}".format(ys_to_rhs_major))
    output.append("Ys2RHsMinor:     {}".format(ys_to_rhs_minor))
    
    # Y dimensions interpretation
    output.append("\n=== Y DIMENSIONS MAPPING ===")
    for i, (major, minor) in enumerate(zip(ys_to_rhs_major, ys_to_rhs_minor)):
        y_value = None
        y_source = None
        
        if major == 0:  # R dimension
            if minor < len(rs_lengths):
                y_value = rs_lengths[minor]
                y_source = f"R[{minor}]"
            else:
                y_value = "?"
                y_source = f"R[{minor}] (out of bounds)"
        else:  # H dimension
            h_idx = major - 1  # H indices are 1-based in major
            if h_idx < len(hs_lengthss) and minor < len(hs_lengthss[h_idx]):
                y_value = hs_lengthss[h_idx][minor]
                y_source = f"H{h_idx}[{minor}]"
            else:
                y_value = "?"
                y_source = f"H{h_idx}[{minor}] (out of bounds)"
                
        output.append(f"Y_{i} = [major={major}, minor={minor}] => {y_source} = {y_value}")
    
    # P dimensions interpretation
    output.append("\n=== P DIMENSIONS MAPPING ===")
    for p_idx, (majors, minors) in enumerate(zip(ps_to_rhs_major, ps_to_rhs_minor)):
        if not isinstance(majors, list) or not isinstance(minors, list):
            output.append(f"P_{p_idx} = Invalid format")
            continue
            
        output.append(f"P_{p_idx} maps to:")
        
        for j, (major, minor) in enumerate(zip(majors, minors)):
            p_value = None
            p_source = None
            
            if major == 0:  # R dimension
                if minor < len(rs_lengths):
                    p_value = rs_lengths[minor]
                    p_source = f"R[{minor}]"
                else:
                    p_value = "?"
                    p_source = f"R[{minor}] (out of bounds)"
            else:  # H dimension
                h_idx = major - 1  # H indices are 1-based in major
                if h_idx < len(hs_lengthss) and minor < len(hs_lengthss[h_idx]):
                    p_value = hs_lengthss[h_idx][minor]
                    p_source = f"H{h_idx}[{minor}]"
                else:
                    p_value = "?"
                    p_source = f"H{h_idx}[{minor}] (out of bounds)"
                    
            output.append(f"  Mapping {j}: [major={major}, minor={minor}] => {p_source} = {p_value}")
    
    return "\n".join(output)

class TileDistributionVisualizer:
    """Visualizer for tile_distribution_encoding structures."""
    
    @staticmethod
    def hierarchical_view(encoding: Dict[str, Any], figsize=(10, 12), variables: Dict[str, Any] = None):
        """Creates a hierarchical visualization of tile distribution encoding.
        
        Args:
            encoding: The parsed tile_distribution_encoding structure
            figsize: Figure size tuple
            variables: Dictionary mapping variable names to their values
        """
        # Initialize variables dictionary if not provided
        if variables is None:
            variables = {}
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract key components from encoding
        rs_lengths = encoding.get('RsLengths', [])
        hs_lengthss = encoding.get('HsLengthss', [])
        ps_to_rhs_major = encoding.get('Ps2RHssMajor', [])
        ps_to_rhs_minor = encoding.get('Ps2RHssMinor', [])
        ys_to_rhs_major = encoding.get('Ys2RHsMajor', [])
        ys_to_rhs_minor = encoding.get('Ys2RHsMinor', [])
        
        # Get variables
        variable_names = encoding.get('variable_names', [])
        
        # Determine dimensions
        num_p_dims = len(ps_to_rhs_major)
        num_y_dims = len(ys_to_rhs_major)
        num_r_dims = len(rs_lengths)
        num_h_dims = len(hs_lengthss)
        num_x_dims = num_r_dims + num_h_dims
        
        # Calculate hierarchy elements and values
        hidden_ids = []
        y_values = []  # Store the actual values of Y dimensions
        
        # Calculate P to R/H connections
        if ps_to_rhs_major:
            for i, (major_list, minor_list) in enumerate(zip(ps_to_rhs_major, ps_to_rhs_minor)):
                if not isinstance(major_list, list) or not isinstance(minor_list, list):
                    continue
                for major, minor in zip(major_list, minor_list):
                    if major == 0:  # R dimension
                        if minor < len(rs_lengths):
                            hidden_ids.append(("P", i, 0, minor))
                    else:  # H dimension
                        if major-1 < len(hs_lengthss) and minor < len(hs_lengthss[major-1]):
                            hidden_ids.append(("P", i, major, minor))
        
        # Calculate Y to R/H connections and values
        if ys_to_rhs_major and ys_to_rhs_minor:
            for i, (major, minor) in enumerate(zip(ys_to_rhs_major, ys_to_rhs_minor)):
                # Get the actual value that Y maps to
                y_value = None
                y_source = None
                
                if major == 0:  # R dimension
                    if minor < len(rs_lengths):
                        y_value = rs_lengths[minor]
                        y_source = f"R[{minor}]"
                        hidden_ids.append(("Y", i, 0, minor))
                else:  # H dimension
                    h_idx = major - 1  # H indices are 1-based in major
                    if h_idx < len(hs_lengthss) and minor < len(hs_lengthss[h_idx]):
                        y_value = hs_lengthss[h_idx][minor]
                        y_source = f"H{h_idx}[{minor}]"
                        hidden_ids.append(("Y", i, major, minor))
                
                y_values.append((y_value, y_source))
        
        # Create a mapping for hidden IDs
        hidden_map = {}
        for i, (src_type, src_idx, major, minor) in enumerate(hidden_ids):
            hidden_map[(src_type, src_idx, major, minor)] = i
        
        # Define vertical positions for each level
        top_y = 0.9
        middle_y = 0.6
        bottom_y = 0.3
        section_height = 0.2
        
        # TOP SECTION: P dimensions and Y dimensions side by side
        
        # Column width and positions
        p_section_width = 0.45
        y_section_width = 0.45
        p_start_x = 0.1
        y_start_x = 0.55
        
        # Section label
        ax.text(0.05, top_y + 0.05, "Top Ids", color='darkgreen', fontsize=14, 
                bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round,pad=0.5'))
        
        # P_dims container
        p_container = patches.Rectangle((p_start_x, top_y - section_height), 
                                     p_section_width, section_height, 
                                     linewidth=2, edgecolor='red', facecolor='none', zorder=1)
        ax.add_patch(p_container)
        ax.text(p_start_x + p_section_width/2, top_y + 0.03, "P_dims", color='red', fontsize=14, 
               ha='center', va='center')
        
        # Y_dims container  
        y_container = patches.Rectangle((y_start_x, top_y - section_height), 
                                     y_section_width, section_height, 
                                     linewidth=2, edgecolor='blue', facecolor='none', zorder=1)
        ax.add_patch(y_container)
        ax.text(y_start_x + y_section_width/2, top_y + 0.03, "Y_dims", color='blue', fontsize=14,
               ha='center', va='center')
        
        # P dimension boxes
        p_box_width = p_section_width / max(num_p_dims, 1) - 0.05
        p_boxes = []
        
        for i in range(num_p_dims):
            x_pos = p_start_x + 0.025 + i * (p_box_width + 0.05)
            box = patches.Rectangle((x_pos, top_y - section_height + 0.04), 
                                  p_box_width, section_height - 0.08, 
                                  linewidth=1, edgecolor='red', facecolor='lightblue', zorder=2)
            ax.add_patch(box)
            ax.text(x_pos + p_box_width/2, top_y - section_height/2, f"P{i}", 
                   ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)
            # Store the center bottom of the box for connections
            p_boxes.append((x_pos + p_box_width/2, top_y - section_height + 0.04))
        
        # Y dimension boxes
        y_box_width = y_section_width / max(num_y_dims, 1) - 0.05
        y_boxes = []
        
        for i in range(num_y_dims):
            x_pos = y_start_x + 0.025 + i * (y_box_width + 0.05)
            box = patches.Rectangle((x_pos, top_y - section_height + 0.04), 
                                  y_box_width, section_height - 0.08, 
                                  linewidth=1, edgecolor='blue', facecolor='lightyellow', zorder=2)
            ax.add_patch(box)
            
            # Add Y dimension with its mapped value for clarity
            y_label = f"Y{i}"
            if i < len(y_values) and y_values[i][0] is not None:
                y_label = f"Y{i}={y_values[i][0]}"
                
            ax.text(x_pos + y_box_width/2, top_y - section_height/2, y_label, 
                   ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)
            
            # Add source info below
            if i < len(y_values) and y_values[i][1] is not None:
                ax.text(x_pos + y_box_width/2, top_y - section_height + 0.1, f"from {y_values[i][1]}",
                       ha='center', va='center', fontsize=8, fontweight='normal', zorder=3)
                
            # Store the center bottom of the box for connections
            y_boxes.append((x_pos + y_box_width/2, top_y - section_height + 0.04))
        
        # MIDDLE SECTION: Hidden Ids
        
        # Label for hidden section
        ax.text(0.05, middle_y + 0.05, "Hidden Ids", color='darkgreen', fontsize=14,
                bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round,pad=0.5'))
        
        # Group hidden ids by their values
        hidden_values = {}
        for i, (src_type, src_idx, major, minor) in enumerate(hidden_ids):
            # Get the value (variable name or number)
            if major == 0:  # R dimension
                if minor < len(rs_lengths):
                    value = rs_lengths[minor]
                else:
                    value = "?"
            else:  # H dimension
                if major-1 < len(hs_lengthss) and minor < len(hs_lengthss[major-1]):
                    try:
                        value = hs_lengthss[major-1][minor]
                    except:
                        value = "?"
                else:
                    value = "?"
            
            if isinstance(value, str) and value in variable_names:
                if value not in hidden_values:
                    hidden_values[value] = []
                hidden_values[value].append((i, src_type, src_idx, major, minor))
            else:
                # Use numerical value as key
                value_key = f"{value}_{src_type}{src_idx}{major}{minor}"
                hidden_values[value_key] = [(i, src_type, src_idx, major, minor)]
        
        # Draw hidden boxes evenly distributed
        hidden_box_height = section_height - 0.05
        num_hidden_boxes = len(hidden_values)
        hidden_boxes_positions = [] 

        hidden_section_width = 0.8
        hidden_start_x = 0.1
        
        # Adaptive settings for hidden boxes
        hidden_label_fontsize = 10
        hidden_value_fontsize = 8
        min_sensible_box_width = 0.06
        max_sensible_box_width = 0.12
        desired_gap_between_boxes = 0.015

        if num_hidden_boxes > 6:
            hidden_label_fontsize = 8
            hidden_value_fontsize = 6
        elif num_hidden_boxes > 4:
            hidden_label_fontsize = 9
            hidden_value_fontsize = 7

        if num_hidden_boxes <= 0: # Should not happen if hidden_values is populated
            actual_hidden_box_width = max_sensible_box_width
            actual_gap = 0
        elif num_hidden_boxes == 1:
            actual_hidden_box_width = min(max_sensible_box_width, hidden_section_width)
            actual_gap = 0
        else:
            # Calculate width based on fitting them with desired gap
            width_if_gaps_present = (hidden_section_width - (num_hidden_boxes - 1) * desired_gap_between_boxes) / num_hidden_boxes
            
            if width_if_gaps_present < min_sensible_box_width:
                actual_hidden_box_width = min_sensible_box_width
                actual_gap = (hidden_section_width - num_hidden_boxes * actual_hidden_box_width) / (num_hidden_boxes - 1)
                if actual_gap < 0: actual_gap = 0 # Ensure no negative gap, boxes might touch
            else:
                actual_hidden_box_width = min(max_sensible_box_width, width_if_gaps_present)
                actual_gap = desired_gap_between_boxes
        
        text_nudge_x = 0.006 
        text_nudge_y = 0.005 
        
        arrow_label_fontsize = 7
        arrow_label_nudge_x = 0.025 
        arrow_label_nudge_y = 0.015 

        # Place hidden boxes evenly
        current_x_offset = hidden_start_x
        sorted_values = sorted(hidden_values.keys())
        for i, value_key in enumerate(sorted_values):
            value_instances = hidden_values[value_key]
            # x_pos = hidden_start_x + (hidden_section_width / max(num_hidden_boxes, 1)) * (i + 0.5) - actual_hidden_box_width / 2 # Old centering
            x_pos = current_x_offset # Position box start
            
            value = value_key.split('_')[0] if '_' in value_key and not value_key.startswith("S::") else value_key # Handle S:: names correctly
            
            box = patches.Rectangle((x_pos, middle_y - hidden_box_height), 
                                  actual_hidden_box_width, hidden_box_height, 
                                  linewidth=1, edgecolor='blue', facecolor='white', zorder=2)
            ax.add_patch(box)
            
            # Add label with both symbolic and numeric values if it's a variable
            if value in variables:
                ax.text(x_pos + actual_hidden_box_width/2 + text_nudge_x, middle_y - hidden_box_height/2 + 0.02 + text_nudge_y, f"{value}", 
                       ha='center', va='center', fontsize=hidden_label_fontsize, fontweight='bold', zorder=3)
                ax.text(x_pos + actual_hidden_box_width/2 + text_nudge_x, middle_y - hidden_box_height/2 - 0.02 + text_nudge_y, f"({variables[value]})", 
                       ha='center', va='center', fontsize=hidden_value_fontsize, zorder=3)
            else:
                ax.text(x_pos + actual_hidden_box_width/2 + text_nudge_x, middle_y - hidden_box_height/2 + text_nudge_y, f"{value}", 
                       ha='center', va='center', fontsize=hidden_label_fontsize, fontweight='bold', zorder=3)
            
            box_top = (x_pos + actual_hidden_box_width/2, middle_y)
            box_bottom = (x_pos + actual_hidden_box_width/2, middle_y - hidden_box_height)
            
            for instance_idx, src_type, src_idx, major, minor in value_instances:
                hidden_boxes_positions.append((box_top, box_bottom)) 
                hidden_map[(src_type, src_idx, major, minor)] = len(hidden_boxes_positions) - 1 
            
            current_x_offset += actual_hidden_box_width + actual_gap # Advance x for next box
        
        # BOTTOM SECTION: X dimensions (R and H)
        
        # Label for X section
        ax.text(0.05, bottom_y + 0.05, "Bottom Ids", color='darkgreen', fontsize=14,
                bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round,pad=0.5'))
        
        # X_dims container
        x_container = patches.Rectangle((0.1, bottom_y - section_height), 
                                      0.8, section_height, 
                                      linewidth=2, edgecolor='purple', facecolor='none', zorder=1)
        ax.add_patch(x_container)
        ax.text(0.1, bottom_y - section_height - 0.03, "X_dims", color='purple', fontsize=14)
        
        # R_dims container (left side of X)
        r_container_width = 0.25
        r_container = patches.Rectangle((0.15, bottom_y - section_height + 0.04), 
                                      r_container_width, section_height - 0.08, 
                                      linewidth=1, edgecolor='orange', facecolor='none', zorder=2)
        ax.add_patch(r_container)
        ax.text(0.15, bottom_y - 0.03, "R_dims", color='orange', fontsize=12)
        
        # H_dims container (right side of X)
        h_container_width = 0.45
        h_container = patches.Rectangle((0.45, bottom_y - section_height + 0.04), 
                                      h_container_width, section_height - 0.08, 
                                      linewidth=1, edgecolor='brown', facecolor='none', zorder=2)
        ax.add_patch(h_container)
        ax.text(0.45, bottom_y - 0.03, "H_dims", color='brown', fontsize=12)
        
        # R dimension boxes
        r_box_width = r_container_width - 0.05
        r_box_height = section_height - 0.12
        r_boxes = []
        
        for i in range(num_r_dims):
            x_pos = 0.175
            y_pos = bottom_y - section_height + 0.08
            value = rs_lengths[i] if i < len(rs_lengths) else "?"
            box = patches.Rectangle((x_pos, y_pos), 
                                  r_box_width, r_box_height, 
                                  linewidth=1, edgecolor='orange', facecolor='lightyellow', zorder=3)
            ax.add_patch(box)
            ax.text(x_pos + r_box_width/2, y_pos + r_box_height/2, f"R{i}={value}", 
                   ha='center', va='center', fontsize=12, zorder=4)
            r_boxes.append((x_pos + r_box_width/2, y_pos + r_box_height))
        
        # Create individual H0 and H1 element boxes instead of just H0 and H1 boxes
        h_element_boxes = []  # Will store positions for each individual H element
        
        for h_idx, h_seq in enumerate(hs_lengthss):
            # Create a label for this H sequence
            h_seq_label_x = 0.5 + h_idx * 0.25
            ax.text(h_seq_label_x, bottom_y - 0.05, f"H{h_idx}", 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='brown')
            
            # Create boxes for each element in this H sequence
            element_box_width = 0.08
            element_box_height = 0.08
            element_spacing = 0.1
            
            for elem_idx, elem_val in enumerate(h_seq):
                # Calculate position
                x_pos = 0.5 + h_idx * 0.25 - (len(h_seq) * element_spacing / 2) + elem_idx * element_spacing
                y_pos = bottom_y - section_height + 0.1
                
                # Create box
                box = patches.Rectangle((x_pos - element_box_width/2, y_pos), 
                                      element_box_width, element_box_height, 
                                  linewidth=1, edgecolor='brown', facecolor='lightgreen', zorder=3)
            ax.add_patch(box)
                
                # Add label with both symbolic and numeric values if it's a variable
            if elem_val in variables:
                # Show both symbolic name and numeric value (more compact)
                ax.text(x_pos, y_pos + element_box_height/2, f"{elem_val}", 
                        ha='center', va='center', fontsize=8, zorder=4)
                ax.text(x_pos, y_pos + element_box_height/4, f"({variables[elem_val]})", 
                        ha='center', va='center', fontsize=6, zorder=4)
            else:
                ax.text(x_pos, y_pos + element_box_height/2, f"{elem_val}", 
                        ha='center', va='center', fontsize=8, zorder=4)
            
            # Add index label below
            ax.text(x_pos, y_pos, f"[{elem_idx}]", 
                    ha='center', va='bottom', fontsize=7, zorder=4, color='brown')
            
            # Store box position for arrows (center, top)
            h_element_boxes.append((h_idx, elem_idx, (x_pos, y_pos + element_box_height)))
        
        # CONNECTIONS
        
        # Arrow style for all connections
        arrow_style = patches.ArrowStyle.Simple(head_length=6, head_width=3)
        
        # 1. Top (P) to Hidden connections
        for i in range(num_p_dims):
            if i < len(p_boxes) and ps_to_rhs_major and ps_to_rhs_minor and i < len(ps_to_rhs_major) and i < len(ps_to_rhs_minor):
                p_box_start_coord = p_boxes[i]
                for map_idx, (major, minor) in enumerate(zip(ps_to_rhs_major[i], ps_to_rhs_minor[i])):
                    hidden_key = ("P", i, major, minor)
                    if hidden_key in hidden_map:
                        hidden_idx = hidden_map[hidden_key]
                        if hidden_idx < len(hidden_boxes_positions):
                            hidden_top_coord = hidden_boxes_positions[hidden_idx][0]
                            arrow = patches.FancyArrowPatch(
                                p_box_start_coord, hidden_top_coord,
                                connectionstyle="arc3,rad=0.1",
                                arrowstyle=arrow_style,
                                color='blue', linewidth=1, alpha=0.7, zorder=1
                            )
                            ax.add_patch(arrow)

                            # Add P-arrow label
                            target_label_p = f"{'R' if major == 0 else 'H'}{'' if major == 0 else major-1}[{minor}]"
                            label_text_p = f"P{i}[{map_idx}]→{target_label_p}"
                            mid_x_p = (p_box_start_coord[0] + hidden_top_coord[0]) / 2 + arrow_label_nudge_x
                            mid_y_p = (p_box_start_coord[1] + hidden_top_coord[1]) / 2 + arrow_label_nudge_y
                            ax.text(mid_x_p, mid_y_p, label_text_p, fontsize=arrow_label_fontsize, color='blue', ha='center', va='bottom', zorder=4)

        # 2. Top (Y) to Hidden connections
        for i in range(num_y_dims):
            if i < len(y_boxes) and i < len(ys_to_rhs_major) and i < len(ys_to_rhs_minor):
                y_box_start_coord = y_boxes[i]
                major, minor = ys_to_rhs_major[i], ys_to_rhs_minor[i]
                hidden_key = ("Y", i, major, minor)
                if hidden_key in hidden_map:
                    hidden_idx = hidden_map[hidden_key]
                    if hidden_idx < len(hidden_boxes_positions):
                        hidden_top_coord = hidden_boxes_positions[hidden_idx][0]
                        arrow = patches.FancyArrowPatch(
                            y_box_start_coord, hidden_top_coord,
                            connectionstyle="arc3,rad=-0.1",
                            arrowstyle=arrow_style,
                            color='red', linewidth=1, alpha=0.7, zorder=1 # Changed to red
                        )
                        ax.add_patch(arrow)

                        # Add Y-arrow label
                        target_label_y = f"{'R' if major == 0 else 'H'}{'' if major == 0 else major-1}[{minor}]"
                        label_text_y = f"Y{i}→{target_label_y}"
                        mid_x_y = (y_box_start_coord[0] + hidden_top_coord[0]) / 2 + arrow_label_nudge_x
                        mid_y_y = (y_box_start_coord[1] + hidden_top_coord[1]) / 2 + arrow_label_nudge_y
                        ax.text(mid_x_y, mid_y_y, label_text_y, fontsize=arrow_label_fontsize, color='red', ha='center', va='bottom', zorder=4)

        # 3. Hidden to Bottom (R and H) connections
        for i, (src_type, src_idx, major, minor) in enumerate(hidden_ids):
            if i < len(hidden_boxes_positions): # Ensure index is valid for hidden_boxes_positions
                hidden_bottom = hidden_boxes_positions[i][1]
                if major == 0 and minor < len(r_boxes):  # R dimension
                    r_box = r_boxes[minor]
                    arrow = patches.FancyArrowPatch(
                        hidden_bottom, r_box,
                        connectionstyle="arc3,rad=0",
                        arrowstyle=arrow_style,
                        color='blue', linewidth=1, alpha=0.7, zorder=1
                    )
                    ax.add_patch(arrow)
                elif major > 0:  # H dimension (connect to specific elements)
                    h_idx = major - 1
                    # Find the corresponding H element box
                    for h_seq_idx, h_elem_idx, h_elem_pos in h_element_boxes:
                        if h_seq_idx == h_idx and h_elem_idx == minor:
                            arrow = patches.FancyArrowPatch(
                                        hidden_bottom, h_elem_pos,
                                connectionstyle="arc3,rad=0",
                                arrowstyle=arrow_style,
                                color='blue', linewidth=1, alpha=0.7, zorder=1
                            )
                            ax.add_patch(arrow)
                            break
        
        # 4. Direct Top to Bottom connections (P/Y to R/H) for reference
        # Use dashed lines for direct connections
        for i in range(num_p_dims):
            if i < len(p_boxes) and ps_to_rhs_major and ps_to_rhs_minor and i < len(ps_to_rhs_major) and i < len(ps_to_rhs_minor):
                p_box = p_boxes[i]
                for major, minor in zip(ps_to_rhs_major[i], ps_to_rhs_minor[i]):
                    if major == 0 and minor < len(r_boxes):  # P to R
                        r_box = r_boxes[minor]
                        arrow = patches.FancyArrowPatch(
                            p_box, r_box,
                            connectionstyle="arc3,rad=0.2",
                            arrowstyle=arrow_style,
                            linestyle='dashed',
                            color='gray', linewidth=0.7, alpha=0.3, zorder=0
                        )
                        ax.add_patch(arrow)
                    elif major > 0:  # P to H (specific element)
                        h_idx = major - 1
                        for h_seq_idx, h_elem_idx, h_elem_pos in h_element_boxes:
                            if h_seq_idx == h_idx and h_elem_idx == minor:
                                arrow = patches.FancyArrowPatch(
                                        p_box, h_elem_pos,
                                connectionstyle="arc3,rad=0.2",
                                arrowstyle=arrow_style,
                                linestyle='dashed',
                                        color='gray', linewidth=0.7, alpha=0.3, zorder=0
                            )
                            ax.add_patch(arrow)
                            break
        
        for i in range(num_y_dims):
            if i < len(y_boxes) and i < len(ys_to_rhs_major) and i < len(ys_to_rhs_minor):
                y_box = y_boxes[i]
                major, minor = ys_to_rhs_major[i], ys_to_rhs_minor[i]
                if major == 0 and minor < len(r_boxes):  # Y to R
                    r_box = r_boxes[minor]
                    arrow = patches.FancyArrowPatch(
                        y_box, r_box,
                        connectionstyle="arc3,rad=-0.2",
                        arrowstyle=arrow_style,
                        linestyle='dashed',
                        color='gray', linewidth=0.7, alpha=0.3, zorder=0
                    )
                    ax.add_patch(arrow)
                elif major > 0:  # Y to H (specific element)
                    h_idx = major - 1
                    for h_seq_idx, h_elem_idx, h_elem_pos in h_element_boxes:
                        if h_seq_idx == h_idx and h_elem_idx == minor:
                            arrow = patches.FancyArrowPatch(
                                        y_box, h_elem_pos,
                                connectionstyle="arc3,rad=-0.2",
                                arrowstyle=arrow_style,
                                linestyle='dashed',
                                        color='gray', linewidth=0.7, alpha=0.3, zorder=0
                            )
                            ax.add_patch(arrow)
                            break
        
        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.title("Tile Distribution Encoding Hierarchy", fontsize=16)
        plt.tight_layout()
        return fig

    @staticmethod
    def graph_view(encoding: Dict[str, Any], figsize=(8, 10)):
        """Creates a graph visualization of tile distribution encoding.
        
        Args:
            encoding: The parsed tile_distribution_encoding structure
            figsize: Figure size tuple
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Extract key components from encoding
        rs_lengths = encoding.get('RsLengths', [])
        hs_lengthss = encoding.get('HsLengthss', [])
        ps_to_rhs_major = encoding.get('Ps2RHssMajor', [])
        ps_to_rhs_minor = encoding.get('Ps2RHssMinor', [])
        ys_to_rhs_major = encoding.get('Ys2RHsMajor', [])
        ys_to_rhs_minor = encoding.get('Ys2RHsMinor', [])
        
        # Add nodes
        # X dimensions
        for i in range(len(hs_lengthss)):
            dims = "*".join(str(x) for x in hs_lengthss[i] if x != "?")
            if not dims:
                dims = "?"
            G.add_node(f"X{i}", type="X", label=f"X{i}", value=dims)
        
        # R dimension
        for i, r_len in enumerate(rs_lengths):
            G.add_node(f"R{i}", type="R", label=f"R", value=r_len)
        
        # P dimensions
        for i in range(len(ps_to_rhs_major)):
            G.add_node(f"P{i}", type="P", label=f"P{i}")
        
        # Y dimensions
        for i in range(len(ys_to_rhs_major)):
            G.add_node(f"Y{i}", type="Y", label=f"Y{i}")
        
        # Add edges
        # P to R/H connections
        for i, (majors, minors) in enumerate(zip(ps_to_rhs_major, ps_to_rhs_minor)):
            if isinstance(majors, list) and isinstance(minors, list):
                for major, minor in zip(majors, minors):
                    if major == 0:  # R dimension
                        G.add_edge(f"P{i}", f"R{minor}", label=f"[{major}][{minor}]")
                    else:  # H dimension
                        G.add_edge(f"P{i}", f"X{major-1}", label=f"H[{minor}]")
        
        # Y to R/H connections
        for i, (major, minor) in enumerate(zip(ys_to_rhs_major, ys_to_rhs_minor)):
            if major == 0:  # R dimension
                G.add_edge(f"Y{i}", f"R{minor}", label=f"[{major}][{minor}]")
            else:  # H dimension
                G.add_edge(f"Y{i}", f"X{major-1}", label=f"H[{minor}]")
        
        # R to X connection (typically R feeds into X)
        if len(rs_lengths) > 0 and len(hs_lengthss) > 0:
            G.add_edge(f"R0", f"X0", label="feeds")
        
        # Create positions (layout)
        pos = {}
        
        # Position X nodes at the bottom
        x_nodes = [n for n in G.nodes() if n.startswith("X")]
        for i, node in enumerate(sorted(x_nodes)):
            pos[node] = (0, i * 1.5)
        
        # Position R nodes above X
        r_nodes = [n for n in G.nodes() if n.startswith("R")]
        for i, node in enumerate(sorted(r_nodes)):
            pos[node] = (0, len(x_nodes) * 1.5 + i * 1.5 + 1)
        
        # Position P nodes above R
        p_nodes = [n for n in G.nodes() if n.startswith("P")]
        for i, node in enumerate(sorted(p_nodes)):
            pos[node] = (0, len(x_nodes) * 1.5 + len(r_nodes) * 1.5 + i * 1.5 + 2)
        
        # Position Y nodes above P
        y_nodes = [n for n in G.nodes() if n.startswith("Y")]
        for i, node in enumerate(sorted(y_nodes)):
            pos[node] = (0, len(x_nodes) * 1.5 + len(r_nodes) * 1.5 + len(p_nodes) * 1.5 + i * 1.5 + 3)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Node colors based on type
        node_colors = {
            'X': 'lightgreen',
            'R': 'lightblue',
            'P': 'pink',
            'Y': 'lightyellow'
        }
        
        colors = [node_colors.get(G.nodes[n]['type'], 'white') for n in G.nodes()]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=colors, alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        
        # Draw edges with labels
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, arrows=True, ax=ax, 
                              arrowstyle='->', arrowsize=15)
        
        # Add edge labels
        edge_labels = {(u, v): G.edges[u, v]['label'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
        
        # Add node annotations (values)
        for node in G.nodes():
            if 'value' in G.nodes[node]:
                value = G.nodes[node]['value']
                node_pos = pos[node]
                ax.annotate(f"{value}", 
                           (node_pos[0], node_pos[1]-0.2),
                           fontsize=10,
                           ha='center')
        
        plt.title("Tile Distribution Encoding Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        return fig

def visualize_encoding_structure(encoding: Dict[str, Any], variables: Dict[str, int] = None, figsize=(10, 8)):
    """
    Creates a visualization of the tile_distribution_encoding structure.
    
    Args:
        encoding: The parsed tile_distribution_encoding structure
        variables: Dictionary mapping variable names to their values
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure with the visualization
    """
    visualizer = TileDistributionVisualizer()
    return visualizer.hierarchical_view(encoding, figsize, variables)

def visualize_tile_distribution(viz_data: Dict[str, Any], figsize=(10, 8)):
    """
    Visualize a tile distribution showing thread mapping to data.
    
    Args:
        viz_data: Visualization data from TileDistribution
        figsize: Size of the figure to create
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    tile_shape = viz_data.get('tile_shape', [])
    thread_mapping = viz_data.get('thread_mapping', {})
    
    if not tile_shape or not thread_mapping:
        ax.text(0.5, 0.5, "No tile data available", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Draw tile grid (simplified example)
    tile_dims = tile_shape[:2] if len(tile_shape) >= 2 else (1, 1)
    rows, cols = tile_dims
    
    # Create color map for threads
    threads = set()
    for pos, thread_id in thread_mapping.items():
        threads.add(thread_id)
    
    thread_colors = {}
    colormap = plt.cm.tab20
    for i, thread_id in enumerate(sorted(threads)):
        thread_colors[thread_id] = colormap(i % 20)
    
    # Draw cells with thread mapping
    cell_size = min(8 / max(rows, cols), 0.5)
    
    for i in range(rows):
        for j in range(cols):
            pos_key = f"{i},{j}"
            thread_id = thread_mapping.get(pos_key, -1)
            
            if thread_id >= 0:
                color = thread_colors.get(thread_id, 'white')
                rect = patches.Rectangle(
                    (j * cell_size, rows * cell_size - (i+1) * cell_size),
                    cell_size, cell_size, 
                    linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
                )
                ax.add_patch(rect)
                
                # Add thread ID as text
                ax.text(
                    j * cell_size + cell_size/2, 
                    rows * cell_size - (i+1) * cell_size + cell_size/2,
                    f"T{thread_id}", 
                    ha='center', va='center', fontsize=8
                )
    
    # Set limits and remove ticks
    ax.set_xlim(0, cols * cell_size)
    ax.set_ylim(0, rows * cell_size)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend for threads
    handles = []
    for thread_id, color in sorted(thread_colors.items()):
        patch = patches.Patch(color=color, label=f"Thread {thread_id}")
        handles.append(patch)
    
    # If there are many threads, use a compact legend
    if len(handles) > 10:
        ax.legend(handles=handles[:10], loc='upper right', 
                 title="Threads (showing 10 of {})".format(len(handles)))
    else:
        ax.legend(handles=handles, loc='upper right', title="Threads")
    
    plt.title("Tile Distribution Thread Mapping", fontsize=16)
    plt.tight_layout()
    return fig

def visualize_thread_access_pattern(viz_data: Dict[str, Any], frame_idx: int = 0, 
                                   num_frames: int = 10, figsize=(10, 8)):
    """
    Visualize thread access patterns over time.
    
    Args:
        viz_data: Visualization data from TileDistribution
        frame_idx: Current animation frame index
        num_frames: Total number of frames
        figsize: Size of the figure to create
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    tile_shape = viz_data.get('tile_shape', [])
    thread_mapping = viz_data.get('thread_mapping', {})
    dimensions = viz_data.get('dimensions', {})
    
    if not tile_shape or not thread_mapping:
        ax.text(0.5, 0.5, "No thread access data available", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Draw tile grid (simplified example)
    tile_dims = tile_shape[:2] if len(tile_shape) >= 2 else (1, 1)
    rows, cols = tile_dims
    
    # Create simulated thread access pattern
    threads = sorted(list(set(thread_mapping.values())))
    
    # Organize threads into warps (groups of 32 threads)
    warps = {}
    for thread_id in threads:
        warp_id = thread_id // 32
        if warp_id not in warps:
            warps[warp_id] = []
        warps[warp_id].append(thread_id)
    
    # Generate a sequence of active threads for each frame
    frame_progress = frame_idx / max(1, num_frames - 1)
    
    # Determine active warps for this frame
    active_warps = []
    warp_progress = min(1.0, frame_progress * 1.5)  # Scale to make animation complete before end
    active_warp_count = max(1, int(warp_progress * len(warps)))
    
    for warp_id in sorted(warps.keys())[:active_warp_count]:
        active_warps.append(warp_id)
    
    # Create a colormap for warps
    warp_colors = {}
    colormap = plt.cm.tab10
    for i, warp_id in enumerate(sorted(warps.keys())):
        warp_colors[warp_id] = colormap(i % 10)
    
    # Draw cells with thread mapping
    cell_size = min(8 / max(rows, cols), 0.5)
    
    for i in range(rows):
        for j in range(cols):
            pos_key = f"{i},{j}"
            thread_id = thread_mapping.get(pos_key, -1)
            
            if thread_id >= 0:
                warp_id = thread_id // 32
                lane_id = thread_id % 32
                
                if warp_id in active_warps:
                # Active thread in current frame
                    color = warp_colors[warp_id]
                alpha = 0.8
                    # Add a slight animation effect
                if frame_idx % 2 == 0:
                        alpha = 0.9
                else:
                # Inactive thread
                    color = 'lightgray'
                alpha = 0.3
            else:
                # No thread assigned
                color = 'white'
                alpha = 0.1
                
            rect = patches.Rectangle(
                (j * cell_size, rows * cell_size - (i+1) * cell_size),
                cell_size, cell_size, 
                linewidth=1, edgecolor='black', facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
            
            if thread_id >= 0:
                warp_id = thread_id // 32
                lane_id = thread_id % 32
                
                # Use different text for active vs inactive threads
                if warp_id in active_warps:
                    text = f"W{warp_id}\nL{lane_id}"
                    fontsize = 7
                    color = 'black'
                else:
                    text = f"{thread_id}"
                    fontsize = 6
                    color = 'gray'
                
                # Add thread ID as text
                ax.text(
                    j * cell_size + cell_size/2, 
                    rows * cell_size - (i+1) * cell_size + cell_size/2,
                    text, 
                    ha='center', va='center', fontsize=fontsize, color=color
                )
    
    # Set limits and remove ticks
    ax.set_xlim(0, cols * cell_size)
    ax.set_ylim(0, rows * cell_size)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add dimension labels
    if dimensions:
        dim_text = []
        for dim_name, dim_size in dimensions.items():
            dim_text.append(f"{dim_name}: {dim_size}")
        
        ax.text(0.02, 0.98, "\n".join(dim_text), 
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    # Add progress indicator
    ax.set_title(f"Thread Access Pattern (Frame {frame_idx+1}/{num_frames})", fontsize=16)
    ax.text(0.5, -0.05, f"Execution Progress: {frame_progress:.0%}", 
            transform=ax.transAxes, ha='center', fontsize=12)
    
    # Add legend for warps
    legend_elements = []
    for warp_id in sorted(warp_colors.keys()):
        if warp_id in active_warps:
            patch = patches.Patch(facecolor=warp_colors[warp_id], edgecolor='black', 
                                 alpha=0.8, label=f'Warp {warp_id} (Threads {warp_id*32}-{warp_id*32+31})')
            legend_elements.append(patch)
    
    if not active_warps:
        legend_elements.append(patches.Patch(facecolor='lightgray', edgecolor='black', 
                                           alpha=0.3, label='No Active Warps'))
    
    # Add legend with scrollable box if too many warps
    if len(legend_elements) > 5:
        ax.legend(handles=legend_elements[:5], loc='upper right', 
                 title=f"Active Warps (showing 5 of {len(active_warps)})")
    else:
        ax.legend(handles=legend_elements, loc='upper right', title="Active Warps")
    
    # Add execution information
    info_text = (
        f"Total Threads: {len(threads)}\n"
        f"Total Warps: {len(warps)}\n"
        f"Active Warps: {len(active_warps)}"
    )
    ax.text(0.98, 0.02, info_text, 
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    return fig

def visualize_hierarchical_tiles(viz_data: Dict[str, Any], figsize=(14, 10), code_snippet: str = None, show_arrows=False):
    """
    Visualize hierarchical tile structure showing block, warp, and thread organization.
    
    This creates a visualization showing:
    - ThreadPerWarp structure
    - WarpPerBlock structure 
    - Overall layout with thread IDs
    - Vector dimensions
    
    Args:
        viz_data: Visualization data from TileDistribution
        figsize: Size of the figure to create
        code_snippet: Optional source code that generated this tile distribution
        show_arrows: Whether to show dimension arrows
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract hierarchical structure
    hierarchical_structure = viz_data.get('hierarchical_structure', {})
    
    if not hierarchical_structure:
        ax.text(0.5, 0.5, "No hierarchical tile data available", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Extract key parameters with safe defaults
    thread_per_warp = hierarchical_structure.get('ThreadPerWarp', [16, 4])
    warp_per_block = hierarchical_structure.get('WarpPerBlock', [4])
    vector_dimensions = hierarchical_structure.get('VectorDimensions', [8])
    block_size = hierarchical_structure.get('BlockSize', [64, 4])
    thread_blocks = hierarchical_structure.get('ThreadBlocks', {})
    tile_name = hierarchical_structure.get('TileName', "Tile Distribution")
    
    # Determine dimensions for display with safe defaults
    threads_per_warp_m = thread_per_warp[0] if thread_per_warp and len(thread_per_warp) > 0 else 16
    threads_per_warp_n = thread_per_warp[1] if thread_per_warp and len(thread_per_warp) > 1 else 4
    warps_per_block_m = warp_per_block[0] if warp_per_block and len(warp_per_block) > 0 else 4
    warps_per_block_n = warp_per_block[1] if warp_per_block and len(warp_per_block) > 1 else 1
    
    # Calculate total warps
    total_warps = warps_per_block_m * warps_per_block_n
    
    # Maximum number of warps to display (for visual clarity)
    MAX_DISPLAY_WARPS = 16  # We'll display at most this many warps
    
    # Determine if we need to sample warps
    display_all_warps = total_warps <= MAX_DISPLAY_WARPS
    
    # If we have too many warps, we'll display a subset and indicate there are more
    display_warp_count = total_warps if display_all_warps else MAX_DISPLAY_WARPS
    
    # Extracted values for prominent display
    bs_m_val = block_size[0] if block_size and len(block_size) > 0 else '?'
    bs_n_val = block_size[1] if block_size and len(block_size) > 1 else '?'
    tpw_m_val = thread_per_warp[0] if thread_per_warp and len(thread_per_warp) > 0 else '?'
    tpw_n_val = thread_per_warp[1] if thread_per_warp and len(thread_per_warp) > 1 else '?'
    wpb_m_display_val = warps_per_block_m # Assuming warps_per_block_m is the count for M-dim
    wpb_n_display_val = warps_per_block_n # The N-dimension of warps per block
    vec_k_val = vector_dimensions[0] if vector_dimensions and len(vector_dimensions) > 0 else '?'

    # Set darker background for the plot
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Draw title in white
    plt.title(tile_name, fontsize=20, pad=20, color='white', fontweight='bold')
    
    # Define colors for different levels
    warp_colors = [
        '#00BFFF',  # Deep sky blue
        '#00FF7F',  # Spring green
        '#FFFF00',  # Yellow
        '#FF0000',  # Red
        '#1E90FF',  # Dodger blue
        '#32CD32',  # Lime green
        '#FFD700',  # Gold
        '#FF4500'   # Orange red
    ]
    
    # Define layout for the new 2x2 info boxes
    info_y_start = 0.95  # Top edge of the info box area
    info_x_start = 0.07  # Left edge
    info_box_height = 0.055 # Reduced from 0.06
    info_box_width = 0.19
    info_x_spacing = 0.02 # Restored: Horizontal spacing between boxes in a row
    info_y_spacing = 0.01   # Reduced from 0.015 (space between info box rows)
    info_text_color = 'white'
    info_edge_color = 'white'
    info_alpha = 0.6
    info_fontsize = 8

    # Box 1.1: BlockSize
    bs_box_y = info_y_start - info_box_height
    bs_box = patches.Rectangle(
        (info_x_start, bs_box_y),
        info_box_width, info_box_height,
        linewidth=1, edgecolor=info_edge_color, facecolor='#FF6347', alpha=info_alpha # Tomato
    )
    ax.add_patch(bs_box)
    ax.text(info_x_start + info_box_width / 2, bs_box_y + info_box_height / 2,
           f"Block: {bs_m_val}M×{bs_n_val}N",
           fontsize=info_fontsize, ha='center', va='center', color=info_text_color)

    # Box 1.2: WarpsPerBlock
    wpb_box_x = info_x_start + info_box_width + info_x_spacing
    wpb_box = patches.Rectangle(
        (wpb_box_x, bs_box_y), # Same Y as BlockSize box
        info_box_width, info_box_height,
        linewidth=1, edgecolor=info_edge_color, facecolor='#32CD32', alpha=info_alpha # LimeGreen
    )
    ax.add_patch(wpb_box)
    ax.text(wpb_box_x + info_box_width / 2, bs_box_y + info_box_height / 2,
           f"Warps/Blk: {wpb_m_display_val}M×{wpb_n_display_val}N",  # Updated to show both dimensions
           fontsize=info_fontsize, ha='center', va='center', color=info_text_color)

    # Box 2.1: ThreadPerWarp
    tpw_box_y = bs_box_y - info_y_spacing - info_box_height # Y position for the second row of info boxes
    tpw_box = patches.Rectangle(
        (info_x_start, tpw_box_y), # Same X as BlockSize box
        info_box_width, info_box_height,
        linewidth=1, edgecolor=info_edge_color, facecolor='#00BFFF', alpha=info_alpha # DeepSkyBlue
    )
    ax.add_patch(tpw_box)
    ax.text(info_x_start + info_box_width / 2, tpw_box_y + info_box_height / 2,
           f"Thrd/Warp: {tpw_m_val}R×{tpw_n_val}C",
           fontsize=info_fontsize, ha='center', va='center', color=info_text_color)

    # Box 2.2: VectorDimension
    vec_box_x = info_x_start + info_box_width + info_x_spacing
    vec_box = patches.Rectangle(
        (vec_box_x, tpw_box_y), # Same Y as ThreadPerWarp box, Same X as WarpsPerBlock box
        info_box_width, info_box_height,
        linewidth=1, edgecolor=info_edge_color, facecolor='#FFD700', alpha=info_alpha # Gold
    )
    ax.add_patch(vec_box)
    ax.text(vec_box_x + info_box_width / 2, tpw_box_y + info_box_height / 2,
           f"Vector(K): {vec_k_val}",
           fontsize=info_fontsize, ha='center', va='center', color=info_text_color)
    
    # Set up layout dimensions
    thread_area_width = 0.55
    code_area_width = 0.45
    
    # Define the display regions (restored)
    # warps_region_height = 0.7  # Original fixed value, will be recalculated
    
    # Create left side vertical color bar (restored variable)
    color_bar_width = 0.05
    
    # The top of the warp drawing area (and color bar) should be below the new info boxes.
    # tpw_box_y is the y-coordinate of the bottom of the tpw_box (lowest info box).
    # Effective padding below info boxes reduced from (old info_y_spacing + 0.01 = 0.015 + 0.01 = 0.025)
    # to new desired total padding of 0.015.
    padding_below_info_area = 0.015
    warp_area_top_y = tpw_box_y - padding_below_info_area
    
    # --- Revised Y-Layout for Bottom Elements ---
    bottom_plot_margin = 0.02
    repeat_box_height_val = 0.08 
    repeat_box_padding_above = 0.025 # Reduced from 0.05

    # Y-position for the *bottom* of the RepeatM box
    repeat_box_actual_y_start = bottom_plot_margin
    # Y-position for the *top* of the RepeatM box
    repeat_box_actual_y_end = repeat_box_actual_y_start + repeat_box_height_val
    
    # This is the Y where the padding above the RepeatM box ends / where the warp region bottom aligns
    repeat_section_padding_end_y = repeat_box_actual_y_end + repeat_box_padding_above

    # Recalculate warps_region_height based on available space
    warps_region_height = warp_area_top_y - repeat_section_padding_end_y
    
    # NEW: Calculate warp height based on the number of warps we want to display
    warp_height = warps_region_height / max(display_warp_count, 1)
    
    # If warp_height is too small, adjust it to a minimum value
    min_warp_height = 0.02  # Minimum height for a warp
    if warp_height < min_warp_height:
        warp_height = min_warp_height
        # This might mean not all warps we want to display will fit
        # We'll handle that by showing the pattern and indicating there are more
    # --- End Revised Y-Layout ---
    
    # Draw the color bar on left side (y-positions adjusted based on new warp_area_top_y and warp_height)
    for i in range(min(display_warp_count, total_warps)):
        # For sampling, choose warps at regular intervals if not showing all
        if display_all_warps:
            warp_idx = i
        else:
            # Sample warps at regular intervals
            warp_idx = int(i * (total_warps / display_warp_count))
            
        warp_color = warp_colors[warp_idx % len(warp_colors)]
        y_pos_color_bar = warp_area_top_y - (i + 1) * warp_height
        
        color_rect = patches.Rectangle(
            (0.02, y_pos_color_bar), 
            color_bar_width - 0.01, warp_height,
            linewidth=1, 
            edgecolor='white', 
            facecolor=warp_color,
            alpha=0.8
        )
        ax.add_patch(color_rect)
    
    # Draw the RepeatM section at the bottom with clear separation
    repeat_box = patches.Rectangle(
        (0.02, repeat_box_actual_y_start), # Use new Y start for the box
        0.3, repeat_box_height_val, # Use defined height
        linewidth=1, 
        edgecolor='white', 
        facecolor='#1E90FF', # DodgerBlue
        alpha=0.5
    )
    ax.add_patch(repeat_box)
    repeat_val = hierarchical_structure.get('Repeat', [4])[0] if hierarchical_structure.get('Repeat') and len(hierarchical_structure.get('Repeat')) > 0 else 4
    # Center text in the new RepeatM box position
    ax.text(0.02 + 0.3/2, repeat_box_actual_y_start + repeat_box_height_val/2, 
           f"{repeat_val} × ({wpb_m_display_val}W × {tpw_m_val}R × {tpw_n_val}C × {vec_k_val}V)\\nRepeatM", 
           fontsize=9, ha='center', va='center', color='white')
    
    # Initialize lists for batch drawing thread cells
    all_thread_rects = []
    all_thread_facecolors = []
    # Initialize lists for batch drawing warp containers
    all_warp_rects = []
    all_warp_facecolors_for_container = [] 

    # Threshold for rendering thread ID text to improve performance for dense layouts
    min_cell_dim_for_text = 0.01 # If cell width or height is less than 1% of figure dim, skip text

    # Draw warps and threads in the center section
    warps_shown = 0
    for warp_idx, (warp_key, warp_data) in enumerate(thread_blocks.items()):
        # For sampling, choose warps at regular intervals if not showing all
        if not display_all_warps:
            if warps_shown >= display_warp_count:
                break
                
            # Sample warps at regular intervals
            if warp_idx != int(warps_shown * (total_warps / display_warp_count)):
                continue
        
        warp_color = warp_colors[warp_idx % len(warp_colors)]
        
        y_pos_warp_block = warp_area_top_y - (warps_shown + 1) * warp_height
        
        # When sampling, modify the warp label to show the actual warp number
        if display_all_warps:
            warp_label = f"Warp{warp_idx}"
        else:
            warp_label = f"Warp{warp_idx}" if warps_shown < display_warp_count - 1 else f"... ({total_warps-display_warp_count+1} more warps)"
        
        ax.text(color_bar_width + 0.02, y_pos_warp_block + warp_height/2, warp_label, 
               fontsize=10, ha='left', va='center', color='white')
        
        thread_grid_width = thread_area_width - color_bar_width - 0.05
        thread_grid_height = warp_height * 0.9
        
        # Create warp container for collection
        warp_container_rect = patches.Rectangle(
            (color_bar_width + 0.04, y_pos_warp_block + 0.01), 
            thread_grid_width, thread_grid_height
        )
        all_warp_rects.append(warp_container_rect)
        all_warp_facecolors_for_container.append(warp_color) 
        
        cell_width = thread_grid_width / max(threads_per_warp_n, 1)
        cell_height = thread_grid_height / max(threads_per_warp_m, 1)
        
        # Only show thread details if the warps are large enough
        if thread_grid_height > 0.02:  # Only add thread details if warp is tall enough
            for thread_key, thread_data in warp_data.items():
                thread_pos = thread_data.get('position', [0, 0])
                thread_id = thread_data.get('global_id', 0)
                
                row_idx = thread_pos[0] if len(thread_pos) > 0 else 0
                col_idx = thread_pos[1] if len(thread_pos) > 1 else 0
                
                cell_x = color_bar_width + 0.04 + col_idx * cell_width
                cell_y = y_pos_warp_block + 0.01 + (threads_per_warp_m - row_idx - 1) * cell_height
                
                thread_rect = patches.Rectangle(
                    (cell_x, cell_y),
                    cell_width, cell_height
                )
                all_thread_rects.append(thread_rect)
                all_thread_facecolors.append(warp_color)
                
                # Conditionally add thread ID text to avoid clutter and improve performance
                if cell_width > min_cell_dim_for_text and cell_height > min_cell_dim_for_text:
                    ax.text(cell_x + cell_width/2, cell_y + cell_height/2, f"T{thread_id}",
                           fontsize=8, ha='center', va='center', color='white')
        
        warps_shown += 1
    
    # After looping, add the collection of warp container rectangles
    if all_warp_rects:
        warp_container_collection = PatchCollection(
            all_warp_rects,
            facecolors=all_warp_facecolors_for_container, 
            edgecolor='white',
            linewidth=1,
            alpha=0.3 # Alpha for the warp containers
        )
        ax.add_collection(warp_container_collection)

    # After looping through all warps and threads, add the collection of thread rectangles
    if all_thread_rects:
        thread_collection = PatchCollection(
            all_thread_rects,
            facecolors=all_thread_facecolors, 
            edgecolor='white',
            linewidth=1,
            alpha=0.7 # Alpha for the thread cells
        )
        ax.add_collection(thread_collection)
    
    # Add information about sampling if not showing all warps
    if not display_all_warps:
        ax.text(0.02, bottom_plot_margin + repeat_box_height_val + repeat_box_padding_above + 0.02, 
               f"Showing {display_warp_count} of {total_warps} warps ({warps_per_block_m}×{warps_per_block_n} total)",
               fontsize=9, ha='left', va='bottom', color='white',
               bbox=dict(facecolor='#444444', alpha=0.7, boxstyle='round,pad=0.3'))

    # Draw C++ code on the right side if provided
    if code_snippet:
        code_x = thread_area_width + 0.05
        
        code_box_top_y = warp_area_top_y 
        code_box_bottom_y = repeat_section_padding_end_y 
        
        code_box_bottom_y = max(code_box_bottom_y, bottom_plot_margin)
        code_box_effective_height = code_box_top_y - code_box_bottom_y

        if code_box_effective_height > 0.1 : # Only draw if there's reasonable height
            code_box = patches.Rectangle(
                (code_x, code_box_bottom_y), 
                0.42, code_box_effective_height, 
                linewidth=1,
                edgecolor='white',
                facecolor='#222222',
                alpha=0.7
            )
            ax.add_patch(code_box)

            # Define colors for syntax highlighting (Restored)
            syntax_colors = {
                'keyword': '#569CD6',    # Blue
                'type': '#4EC9B0',       # Teal
                'namespace': '#9CDCFE',  # Light blue
                'function': '#DCDCAA',   # Yellow
                'comment': '#6A9955',    # Green
                'number': '#B5CEA8',     # Light green
                'operator': '#D4D4D4',   # Light gray
                'string': '#CE9178',     # Orange
                'default': '#D4D4D4'     # Light gray
            }
            
            # Keywords to color (Restored)
            keywords = ['template', 'typename', 'constexpr', 'static', 'return', 'using', 'if', 'else', 'for', 'while']
            types = ['index_t', 'tuple', 'sequence', 'Problem', 'int', 'float', 'double', 'bool']
            
            code_lines = code_snippet.split('\n') # Restored for clarity, or use split in loop directly

            # Add code text with syntax highlighting
            # Position text lines from the top of the code_box (code_box_top_y)
            for i, line in enumerate(code_lines):
                # Calculate how many lines can fit
                num_fittable_lines = int((code_box_effective_height - 0.02) / 0.025) # 0.02 top/bottom margin
                if i >= max(0, num_fittable_lines): 
                    break
                    
                # Position for this line, from top of code box
                text_line_y = code_box_top_y - 0.015 - (0.025 * i) # Start first line below top of box
                
                # Skip completely if line is too long
                if len(line) > 70: # Simple estimate for characters fitting in 0.42 width
                    parts = []
                    current_part = ""
                    for word in line.split():
                        if len(current_part + " " + word) > 70:
                            parts.append(current_part)
                            current_part = word
                        else:
                            current_part += (" " + word if current_part else word)
                    if current_part:
                        parts.append(current_part)
                        
                    for j, part in enumerate(parts):
                        # Ensure wrapped lines also check fittable lines
                        if (i + j * 0.8) >= max(0, num_fittable_lines) : break 
                        sub_y_pos = code_box_top_y - 0.015 - (0.025 * (i + j * 0.8))
                        ax.text(code_x + 0.01, sub_y_pos, part, 
                               fontsize=8, ha='left', va='center', 
                               family='monospace', color=syntax_colors['default'])
                    continue # Move to next original line
                
                # Comments handling
                if "//" in line:
                    comment_parts = line.split("//")
                    if len(comment_parts) >= 2:
                        # Draw code part and comment part separately
                        ax.text(code_x + 0.01, text_line_y, comment_parts[0], 
                               fontsize=8, ha='left', va='center', 
                               family='monospace', color=syntax_colors['default'])
                        ax.text(code_x + 0.01 + len(comment_parts[0]) * 0.005, text_line_y, # Estimate length for positioning
                               '//' + comment_parts[1], 
                               fontsize=8, ha='left', va='center', 
                               family='monospace', color=syntax_colors['comment'])
                        continue
                
                # Simple syntax highlighting for other lines
                line_color = syntax_colors['default']
                if any(keyword in line.split() for keyword in keywords):
                    line_color = syntax_colors['keyword']
                elif any(type_name in line.split() for type_name in types):
                    line_color = syntax_colors['type']
                elif "ck_tile::" in line or "::" in line:
                    line_color = syntax_colors['namespace']
                elif "(" in line and ")" in line and not "=" in line: # Basic function detection
                    line_color = syntax_colors['function']
                    
                ax.text(code_x + 0.01, text_line_y, line, 
                       fontsize=8, ha='left', va='center', 
                       family='monospace', color=line_color)
        
        # Add dimension annotations based on detected values in the code
        dimension_values = hierarchical_structure.get('DimensionValues', [])
        
        # Only show dimension arrows if explicitly requested
        if show_arrows and dimension_values:
            # Define arrow parameters
            arrow_y = 0.2  # Even lower position to ensure they're below the code
            start_x = code_x + 0.1
            arrow_spacing = 0.07  # Increase spacing between arrows
            
            # Draw dimension values evenly spaced
            for i, value in enumerate(dimension_values[:6]):  # Limit to 6 dimensions
                x_pos = start_x + i * arrow_spacing
                
                # Draw number
                ax.text(x_pos, arrow_y, str(value), 
                       fontsize=12, ha='center', va='center', 
                       color='red', fontweight='bold')
                
                # Draw an arrow pointing to the number
                arr_len = 0.03
                ax.annotate("", xy=(x_pos, arrow_y - arr_len), 
                          xytext=(x_pos, arrow_y - 0.06),
                          arrowprops=dict(arrowstyle="->", color='red', lw=1.5))
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage
    example_code = """
    tile_distribution_encoding<
        sequence<1>,                           // 0 R
        tuple<sequence<Nr_y, Nr_p, Nw>,        // H 
              sequence<Kr_y, Kr_p, Kw, Kv>>,
        tuple<sequence<1, 2>,                  // p major
              sequence<2, 1>>,
        tuple<sequence<1, 1>,                  // p minor
              sequence<2, 2>>,
        sequence<1, 2, 2>,                     // Y major
        sequence<0, 0, 3>>{}                   // y minor
    """
    
    parser = TileDistributionParser()
    result = parser.parse_tile_distribution_encoding(example_code)
    print("Parsed Structure:")
    print(result)
    
    # Create visualizations
    fig1 = visualize_encoding_structure(result)
    plt.savefig("encoding_structure.png", dpi=150, bbox_inches='tight')
    
    # Simple mock data for visualization
    mock_viz_data = {
        'tile_shape': [4, 4],
        'thread_mapping': {
            '0,0': 0, '0,1': 1, '0,2': 2, '0,3': 3,
            '1,0': 4, '1,1': 5, '1,2': 6, '1,3': 7,
            '2,0': 8, '2,1': 9, '2,2': 10, '2,3': 11,
            '3,0': 12, '3,1': 13, '3,2': 14, '3,3': 15
        },
        'occupancy': 0.85,
        'utilization': 0.92
    }
    
    fig2 = visualize_tile_distribution(mock_viz_data)
    plt.savefig("tile_distribution.png", dpi=150, bbox_inches='tight')
    
    fig3 = visualize_thread_access_pattern(mock_viz_data, 3, 10)
    plt.savefig("thread_access.png", dpi=150, bbox_inches='tight')
    
    fig4 = visualize_hierarchical_tiles(mock_viz_data)
    plt.savefig("hierarchical_tiles.png", dpi=150, bbox_inches='tight')
    
    plt.show() 

def visualize_y_space_structure(
    ys_lengths: List[int],
    ys_names: List[str],
    vector_dim_idx: Optional[int] = -1,
    title: str = "P-Tile Local Data Layout (Y-Dimensions)"
) -> plt.Figure:
    """
    Visualizes the Y-space structure as a 2D grid.
    Highlights the vector dimension if specified.
    If NDimY > 2, it visualizes Y0 vs Y1.
    If NDimY = 1, it visualizes a 1D strip.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)

    n_dim_y = len(ys_lengths)

    if n_dim_y == 0:
        ax.text(0.5, 0.5, "No Y-Dimensions", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    dim1_len = 1
    dim2_len = 1
    dim1_idx = -1
    dim2_idx = -1
    dim1_name = ""
    dim2_name = ""

    if n_dim_y == 1:
        dim1_len = ys_lengths[0]
        dim1_idx = 0
        dim1_name = ys_names[0]
        # Visualize as a horizontal strip, so dim2_len remains 1
        # Let dim1 be "columns" and dim2 be "rows" for plotting
        plot_cols, plot_rows = dim1_len, dim2_len
        x_label, y_label = dim1_name, ""
        x_ticks_labels = range(dim1_len)
        y_ticks_labels = []

    elif n_dim_y >= 2:
        dim1_len = ys_lengths[0] # Typically Y0
        dim2_len = ys_lengths[1] # Typically Y1
        dim1_idx = 0
        dim2_idx = 1
        dim1_name = ys_names[0]
        dim2_name = ys_names[1]
        # Let dim1 be "columns" (x-axis) and dim2 be "rows" (y-axis)
        plot_cols, plot_rows = dim1_len, dim2_len
        x_label, y_label = dim1_name, dim2_name
        x_ticks_labels = range(dim1_len)
        y_ticks_labels = range(dim2_len)
        if n_dim_y > 2:
            ax.set_title(f"{title}\\n(Showing {ys_names[0]} vs {ys_names[1]}, {len(ys_names)-2} other Y-dims not shown)")


    cell_width = 1.0
    cell_height = 1.0
    
    for r in range(plot_rows): # Iterate over Y-dim for rows
        for c in range(plot_cols): # Iterate over Y-dim for columns
            cell_x = c * cell_width
            cell_y = (plot_rows - 1 - r) * cell_height # Y-axis inverted for typical matrix display

            is_vector_cell_dim1 = (vector_dim_idx == dim1_idx)
            is_vector_cell_dim2 = (vector_dim_idx == dim2_idx and n_dim_y > 1) # Only if dim2 actually exists

            # Default cell properties
            face_color = 'white'
            edge_color = 'black'
            text_val = ""
            
            if n_dim_y == 1: # 1D case
                if is_vector_cell_dim1: # If the single Y-dim is the vector dim
                    face_color = 'lightcoral'
                    text_val = f"V{c}"
                else:
                    text_val = f"{c}" # Just index
            elif n_dim_y >=2: # 2D case
                text_val = f"({c},{r})" # (dim1_coord, dim2_coord)
                is_part_of_vector_highlight = False
                
                if vector_dim_idx == dim1_idx: # Vector is along dim1 (columns)
                    face_color = 'lightcoral'
                    text_val = f"(V{c},{r})"
                elif vector_dim_idx == dim2_idx: # Vector is along dim2 (rows)
                    face_color = 'lightskyblue'
                    text_val = f"({c},V{r})"
                
                # If vector_dim_idx is neither dim1_idx nor dim2_idx but exists (i.e., n_dim_y > 2 and vector_dim_idx > 1)
                # then all cells shown are part of "other" dimensions relative to the vector.
                # No specific cell highlight unless we choose to show vector elements if vector_dim_idx is 0 or 1.

            rect = patches.Rectangle((cell_x, cell_y), cell_width, cell_height,
                                     linewidth=1, edgecolor=edge_color, facecolor=face_color, alpha=0.7)
            ax.add_patch(rect)
            ax.text(cell_x + cell_width / 2, cell_y + cell_height / 2, text_val,
                    ha='center', va='center', fontsize=8)

    ax.set_xlim(0, plot_cols * cell_width)
    ax.set_ylim(0, plot_rows * cell_height)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(x_label)
    ax.set_xticks([i * cell_width + cell_width / 2 for i in range(plot_cols)])
    ax.set_xticklabels(x_ticks_labels)
    
    if y_label or y_ticks_labels : # Only set if they are meaningful
        ax.set_ylabel(y_label)
        ax.set_yticks([i * cell_height + cell_height / 2 for i in range(plot_rows)])
        ax.set_yticklabels(reversed(y_ticks_labels)) # Match visual layout
    else:
        ax.set_yticks([])


    if n_dim_y > 0 and vector_dim_idx != -1:
        legend_patches = [patches.Patch(facecolor='lightcoral' if vector_dim_idx == dim1_idx or n_dim_y == 1 else ('lightskyblue' if vector_dim_idx == dim2_idx else 'grey'), 
                                        label=f'Vector Elements (along {ys_names[vector_dim_idx]})')]
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.4, 1))

    plt.tight_layout()
    return fig