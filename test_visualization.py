#!/usr/bin/env python
"""
Test script for visualizing thread distribution with enhanced visualization.
This script generates visualizations of a specified tile_distribution_encoding
and saves them to files or displays them interactively.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import numpy as np
from parser import TileDistributionParser, debug_indexing_relationships
import visualizer
from tiler import TileDistribution

def create_indexing_visualization(indexing_debug, variables):
    """Create a visual representation of the indexing relationships."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define vertical positions for each level
    title_height = 0.95
    p_dims_y = 0.8
    y_dims_y = 0.6
    hidden_y = 0.4
    r_h_dims_y = 0.2
    
    # Draw title
    ax.text(0.5, title_height, "The Distribution Encoding Hierarchy", 
            fontsize=18, fontweight='bold', ha='center')
    
    # Get the data
    major_indices = indexing_debug.get("MajorIndices", {})
    minor_indices = indexing_debug.get("MinorIndices", {})
    index_mapping = indexing_debug.get("IndexMapping", {})
    
    # Create boxes for P dimensions (top left)
    ax.text(0.05, p_dims_y + 0.05, "Top Ids", color='darkgreen', fontsize=14, 
            bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round,pad=0.5'))
    
    # P_dims container
    p_container_rect = (0.1, p_dims_y - 0.1)
    p_container_height = 0.15
    p_container = patches.Rectangle(p_container_rect, 
                                 0.35, p_container_height, 
                                 linewidth=2, edgecolor='red', facecolor='none', zorder=1)
    ax.add_patch(p_container)
    ax.text(0.275, p_container_rect[1] + p_container_height + 0.01, "P_dims", color='red', fontsize=14, ha='center', va='bottom')
    
    # P0 box
    p0_box = patches.Rectangle((0.15, p_dims_y - 0.08), 
                            0.1, 0.1, 
                            linewidth=1, edgecolor='red', facecolor='lightblue', zorder=2)
    ax.add_patch(p0_box)
    ax.text(0.2, p_dims_y - 0.03, "P0", fontsize=12, ha='center')
    p0_pos = (0.2, p_dims_y - 0.08)
    
    # P1 box
    p1_box = patches.Rectangle((0.3, p_dims_y - 0.08), 
                            0.1, 0.1, 
                            linewidth=1, edgecolor='red', facecolor='lightblue', zorder=2)
    ax.add_patch(p1_box)
    ax.text(0.35, p_dims_y - 0.03, "P1", fontsize=12, ha='center')
    p1_pos = (0.35, p_dims_y - 0.08)
    
    # Y_dims container (top right)
    y_container_rect = (0.55, p_dims_y - 0.1)
    y_container_height = 0.15
    y_container = patches.Rectangle(y_container_rect, 
                                 0.35, y_container_height, 
                                 linewidth=2, edgecolor='blue', facecolor='none', zorder=1)
    ax.add_patch(y_container)
    ax.text(0.725, y_container_rect[1] + y_container_height + 0.01, "Y_dims", color='blue', fontsize=14, ha='center', va='bottom')
    
    # Y0, Y1, Y2 boxes
    y0_box = patches.Rectangle((0.575, p_dims_y - 0.08), 
                            0.08, 0.1, 
                            linewidth=1, edgecolor='blue', facecolor='lightyellow', zorder=2)
    ax.add_patch(y0_box)
    ax.text(0.615, p_dims_y - 0.03, "Y0", fontsize=12, ha='center')
    y0_pos = (0.615, p_dims_y - 0.08)
    
    y1_box = patches.Rectangle((0.685, p_dims_y - 0.08), 
                            0.08, 0.1, 
                            linewidth=1, edgecolor='blue', facecolor='lightyellow', zorder=2)
    ax.add_patch(y1_box)
    ax.text(0.725, p_dims_y - 0.03, "Y1", fontsize=12, ha='center')
    y1_pos = (0.725, p_dims_y - 0.08)
    
    y2_box = patches.Rectangle((0.795, p_dims_y - 0.08), 
                            0.08, 0.1, 
                            linewidth=1, edgecolor='blue', facecolor='lightyellow', zorder=2)
    ax.add_patch(y2_box)
    ax.text(0.835, p_dims_y - 0.03, "Y2", fontsize=12, ha='center')
    y2_pos = (0.835, p_dims_y - 0.08)
    
    # Middle section - Hidden Ids
    ax.text(0.05, hidden_y + 0.05, "Hidden Ids", color='darkgreen', fontsize=14,
            bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round,pad=0.5'))
    
    # Get H0 and H1 values
    h0_values = minor_indices.get("H0", [])
    h1_values = minor_indices.get("H1", [])
    
    # Create a hidden box section for each sequence
    h0_container_rect = (0.1, hidden_y - 0.1)
    h0_container_height = 0.15
    h0_container = patches.Rectangle(h0_container_rect, 
                                  0.35, h0_container_height, 
                                  linewidth=1, edgecolor='blue', facecolor='none', zorder=1)
    ax.add_patch(h0_container)
    ax.text(0.275, h0_container_rect[1] + h0_container_height + 0.01, "H0 Elements", color='blue', fontsize=12, ha='center', va='bottom')
    
    h1_container_rect = (0.5, hidden_y - 0.1)
    h1_container_height = 0.15
    h1_container = patches.Rectangle(h1_container_rect, 
                                  0.45, h1_container_height, 
                                  linewidth=1, edgecolor='red', facecolor='none', zorder=1)
    ax.add_patch(h1_container)
    ax.text(0.725, h1_container_rect[1] + h1_container_height + 0.01, "H1 Elements", color='red', fontsize=12, ha='center', va='bottom')

    # Create boxes for each element in H0
    h0_boxes = []
    for i, val in enumerate(h0_values):
        # Calculate position within the H0 container
        box_width = 0.08
        h0_start_x = 0.15
        spacing = 0.12
        x_pos = h0_start_x + i * spacing
        
        # Create box
        box = patches.Rectangle((x_pos - 0.04, hidden_y - 0.08), 
                              box_width, 0.1, 
                              linewidth=1, edgecolor='blue', facecolor='white', zorder=2)
        ax.add_patch(box)
        
        # Add label with both symbolic and numeric values
        symbolic_val = val
        numeric_val = variables.get(symbolic_val, symbolic_val)
        ax.text(x_pos, hidden_y, f"{symbolic_val}", fontsize=9, ha='center', va='center')
        ax.text(x_pos, hidden_y - 0.04, f"({numeric_val})", fontsize=8, ha='center', va='center')
        
        # Add index label
        ax.text(x_pos, hidden_y - 0.08, f"[{i}]", fontsize=8, ha='center', va='top', color='blue')
        
        # Store box position (center of the box's top edge) for incoming arrows
        h0_boxes.append((x_pos, hidden_y - 0.08 + 0.1)) # Target top-center of the small box
    
    # Create boxes for each element in H1
    h1_boxes = []
    for i, val in enumerate(h1_values):
        # Calculate position within the H1 container
        box_width = 0.08
        h1_start_x = 0.55
        spacing = 0.1
        x_pos = h1_start_x + i * spacing
        
        # Create box
        box = patches.Rectangle((x_pos - 0.04, hidden_y - 0.08), 
                              box_width, 0.1, 
                              linewidth=1, edgecolor='red', facecolor='white', zorder=2)
        ax.add_patch(box)
        
        # Add label with both symbolic and numeric values
        symbolic_val = val
        numeric_val = variables.get(symbolic_val, symbolic_val)
        ax.text(x_pos, hidden_y, f"{symbolic_val}", fontsize=9, ha='center', va='center')
        ax.text(x_pos, hidden_y - 0.04, f"({numeric_val})", fontsize=8, ha='center', va='center')
        
        # Add index label
        ax.text(x_pos, hidden_y - 0.08, f"[{i}]", fontsize=8, ha='center', va='top', color='red')
        
        # Store box position (center of the box's top edge) for incoming arrows
        h1_boxes.append((x_pos, hidden_y - 0.08 + 0.1)) # Target top-center of the small box
    
    # Bottom section - R and H boxes
    ax.text(0.05, r_h_dims_y + 0.05, "Bottom Ids", color='darkgreen', fontsize=14,
            bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round,pad=0.5'))
    
    # R box container
    r_container_rect = (0.1, r_h_dims_y - 0.1)
    r_container_height = 0.15
    r_container = patches.Rectangle(r_container_rect, 
                                  0.2, r_container_height, 
                                  linewidth=2, edgecolor='orange', facecolor='none', zorder=1)
    ax.add_patch(r_container)
    ax.text(0.2, r_container_rect[1] + r_container_height + 0.01, "R_dims", color='orange', fontsize=14, ha='center', va='bottom')
    
    # R0 box
    r0_box = patches.Rectangle((0.15, r_h_dims_y - 0.08), 
                            0.1, 0.1, 
                            linewidth=1, edgecolor='orange', facecolor='lightyellow', zorder=2)
    ax.add_patch(r0_box)
    ax.text(0.2, r_h_dims_y - 0.03, "R0=1", fontsize=12, ha='center')
    r0_pos = (0.2, r_h_dims_y + 0.05)
    
    # H box container
    h_container_rect = (0.35, r_h_dims_y - 0.1)
    h_container_height = 0.15
    h_container = patches.Rectangle(h_container_rect, 
                                  0.55, h_container_height, 
                                  linewidth=2, edgecolor='brown', facecolor='none', zorder=1)
    ax.add_patch(h_container)
    ax.text(0.625, h_container_rect[1] + h_container_height + 0.01, "H_dims", color='brown', fontsize=14, ha='center', va='bottom')
    
    # H0 box - show the variables
    h0_inner_box_rect = patches.Rectangle((0.4, r_h_dims_y - 0.08), 
                            0.2, 0.1, 
                            linewidth=1, edgecolor='brown', facecolor='lightgreen', zorder=2)
    ax.add_patch(h0_inner_box_rect)
    h0_vars = ", ".join(str(x) for x in minor_indices.get("H0", []))
    ax.text(0.5, r_h_dims_y - 0.03, f"H0=[{h0_vars}]", fontsize=8, ha='center')
    # Target for arrows from Hidden H0 elements should be the top-center of this inner box
    h0_target_pos = (0.5, r_h_dims_y - 0.08 + 0.1) 
    
    # H1 box - show the variables
    h1_inner_box_rect = patches.Rectangle((0.65, r_h_dims_y - 0.08), 
                            0.2, 0.1, 
                            linewidth=1, edgecolor='brown', facecolor='lightgreen', zorder=2)
    ax.add_patch(h1_inner_box_rect)
    h1_vars = ", ".join(str(x) for x in minor_indices.get("H1", []))
    ax.text(0.75, r_h_dims_y - 0.03, f"H1=[{h1_vars}]", fontsize=8, ha='center')
    # Target for arrows from Hidden H1 elements should be the top-center of this inner box
    h1_target_pos = (0.75, r_h_dims_y - 0.08 + 0.1)
    
    # Arrow style for connections
    arrow_style = patches.ArrowStyle.Simple(head_length=6, head_width=4)
    
    # Extract P and Y mappings
    p_mappings = {}
    y_mappings = {}
    
    for key, mappings in index_mapping.items():
        if key.startswith("P"):
            p_mappings[key] = mappings
        elif key.startswith("Y"):
            y_mappings[key] = mappings
    
    # Draw arrows from P dimensions to H0/H1 elements
    for p_key, mappings in p_mappings.items():
        p_idx = int(p_key[1:])  # Get P index (0 or 1)
        p_start = p0_pos if p_idx == 0 else p1_pos
        
        for i, mapping in enumerate(mappings):
            target = mapping.get("Target")
            minor_idx = mapping.get("MinorIndex")
            
            if target == "R":
                # Connect to R
                arrow = patches.FancyArrowPatch(
                    p_start, r0_pos,
                    connectionstyle=f"arc3,rad={0.1 * (i+1)}",
                    arrowstyle=arrow_style,
                    color='blue', linewidth=1, alpha=0.7, zorder=1
                )
                ax.add_patch(arrow)
                label_x = (p_start[0] + r0_pos[0]) / 2
                label_y = (p_start[1] + r0_pos[1]) / 2 + 0.05
                ax.text(label_x, label_y, f"{p_key}[{i}]→R[{minor_idx}]", 
                       fontsize=8, ha='center', color='blue')
            elif target == "H0" and minor_idx < len(h0_boxes):
                # Connect to specific H0 element
                h0_element = h0_boxes[minor_idx]
                arrow = patches.FancyArrowPatch(
                    p_start, h0_element,
                    connectionstyle=f"arc3,rad={0.1 * (i+1)}",
                    arrowstyle=arrow_style,
                    color='blue', linewidth=1, alpha=0.7, zorder=1
                )
                ax.add_patch(arrow)
                label_x = (p_start[0] + h0_element[0]) / 2
                label_y = (p_start[1] + h0_element[1]) / 2 + 0.02
                ax.text(label_x, label_y, f"{p_key}[{i}]→H0[{minor_idx}]", 
                       fontsize=8, ha='center', color='blue')
            elif target == "H1" and minor_idx < len(h1_boxes):
                # Connect to specific H1 element
                h1_element = h1_boxes[minor_idx]
                arrow = patches.FancyArrowPatch(
                    p_start, h1_element,
                    connectionstyle=f"arc3,rad={0.1 * (i+1)}",
                    arrowstyle=arrow_style,
                    color='blue', linewidth=1, alpha=0.7, zorder=1
                )
                ax.add_patch(arrow)
                label_x = (p_start[0] + h1_element[0]) / 2
                label_y = (p_start[1] + h1_element[1]) / 2 + 0.02
                ax.text(label_x, label_y, f"{p_key}[{i}]→H1[{minor_idx}]", 
                       fontsize=8, ha='center', color='blue')
    
    # Draw arrows from Y dimensions to H0/H1 elements
    for y_key, mappings in y_mappings.items():
        y_idx = int(y_key[1:])  # Get Y index (0, 1, or 2)
        if y_idx == 0:
            y_start = y0_pos
        elif y_idx == 1:
            y_start = y1_pos
        else:
            y_start = y2_pos
        
        for i, mapping in enumerate(mappings):
            target = mapping.get("Target")
            minor_idx = mapping.get("MinorIndex")
            
            if target == "R":
                # Connect to R
                arrow = patches.FancyArrowPatch(
                    y_start, r0_pos,
                    connectionstyle=f"arc3,rad={-0.1 * (i+1)}",
                    arrowstyle=arrow_style,
                    color='red', linewidth=1, alpha=0.7, zorder=1
                )
                ax.add_patch(arrow)
                label_x = (y_start[0] + r0_pos[0]) / 2
                label_y = (y_start[1] + r0_pos[1]) / 2 + 0.05
                ax.text(label_x, label_y, f"{y_key}→R[{minor_idx}]", 
                       fontsize=8, ha='center', color='red')
            elif target == "H0" and minor_idx < len(h0_boxes):
                # Connect to specific H0 element
                h0_element = h0_boxes[minor_idx]
                arrow = patches.FancyArrowPatch(
                    y_start, h0_element,
                    connectionstyle=f"arc3,rad={-0.1 * (i+1)}",
                    arrowstyle=arrow_style,
                    color='red', linewidth=1, alpha=0.7, zorder=1
                )
                ax.add_patch(arrow)
                label_x = (y_start[0] + h0_element[0]) / 2
                label_y = (y_start[1] + h0_element[1]) / 2 + 0.02
                ax.text(label_x, label_y, f"{y_key}→H0[{minor_idx}]", 
                       fontsize=8, ha='center', color='red')
            elif target == "H1" and minor_idx < len(h1_boxes):
                # Connect to specific H1 element
                h1_element = h1_boxes[minor_idx]
                arrow = patches.FancyArrowPatch(
                    y_start, h1_element,
                    connectionstyle=f"arc3,rad={-0.1 * (i+1)}",
                    arrowstyle=arrow_style,
                    color='red', linewidth=1, alpha=0.7, zorder=1
                )
                ax.add_patch(arrow)
                label_x = (y_start[0] + h1_element[0]) / 2
                label_y = (y_start[1] + h1_element[1]) / 2 + 0.02
                ax.text(label_x, label_y, f"{y_key}→H1[{minor_idx}]", 
                       fontsize=8, ha='center', color='red')
    
    # Connect H0/H1 elements to H boxes in bottom level
    # These arrows should originate from the bottom of the H0/H1 element boxes
    
    # Adjust h0_element_source_positions and h1_element_source_positions 
    # to be the bottom-center of the individual H0/H1 element boxes
    
    h0_element_source_positions = []
    for i, val in enumerate(h0_values):
        box_width = 0.08
        h0_start_x = 0.15
        spacing = 0.12
        x_pos = h0_start_x + i * spacing
        h0_element_source_positions.append((x_pos, hidden_y - 0.08)) # Bottom-center of small box

    h1_element_source_positions = []
    for i, val in enumerate(h1_values):
        box_width = 0.08
        h1_start_x = 0.55
        spacing = 0.1
        x_pos = h1_start_x + i * spacing
        h1_element_source_positions.append((x_pos, hidden_y - 0.08)) # Bottom-center of small box

    for i, h0_source_pos in enumerate(h0_element_source_positions):
        arrow = patches.FancyArrowPatch(
            h0_source_pos, h0_target_pos, # Use h0_target_pos for the inner green box
            connectionstyle=f"arc3,rad={0.1 * (i+1)}",
            arrowstyle=arrow_style,
            color='purple', linewidth=1, alpha=0.5, zorder=1
        )
        ax.add_patch(arrow)
    
    for i, h1_source_pos in enumerate(h1_element_source_positions):
        arrow = patches.FancyArrowPatch(
            h1_source_pos, h1_target_pos, # Use h1_target_pos for the inner green box
            connectionstyle=f"arc3,rad={-0.1 * (i+1)}",
            arrowstyle=arrow_style,
            color='purple', linewidth=1, alpha=0.5, zorder=1
        )
        ax.add_patch(arrow)
    
    # Generate description text from the mappings
    description_lines = []
    for p_key, mappings in sorted(p_mappings.items()):
        p_desc = f"{p_key} maps to: "
        map_parts = []
        for m in mappings:
            target = m.get("Target")
            idx = m.get("MinorIndex")
            sym_val = m.get("SymbolicValue")
            num_val = m.get("Value")
            map_parts.append(f"{target}[{idx}]={sym_val} ({num_val})")
        p_desc += ", ".join(map_parts)
        description_lines.append(p_desc)
    
    for y_key, mappings in sorted(y_mappings.items()):
        for m in mappings:  # Usually only one mapping per Y
            target = m.get("Target")
            idx = m.get("MinorIndex")
            sym_val = m.get("SymbolicValue")
            num_val = m.get("Value")
            description_lines.append(f"{y_key} maps to: {target}[{idx}]={sym_val} ({num_val})")
    
    description = "\n".join(description_lines)
    
    ax.text(0.02, 0.02, description, 
            fontsize=10, va='bottom', ha='left',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Set up the axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def test_variable_based_template():
    """Test visualization with the variable-based template."""
    # Example code from the user query
    example_code = """
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
    """
    
    # Parse the encoding
    parser = TileDistributionParser()
    encoding = parser.parse_tile_distribution_encoding(example_code)
    
    # Set variable values
    variables = {
        "Nr_y": 4,
        "Nr_p": 4,
        "Nw": 8,
        "Kr_y": 4,
        "Kr_p": 8,
        "Kw": 8,
        "Kv": 4
    }
    
    # Generate indexing debug information
    indexing_debug = debug_indexing_relationships(encoding, variables)
    print("\nIndexing Relationships:")
    print(json.dumps(indexing_debug, indent=2))
    
    # Create and save the indexing visualization
    print("\nGenerating explicit indexing visualization...")
    fig0 = create_indexing_visualization(indexing_debug, variables)
    plt.figure(fig0.number)
    plt.savefig("indexing_visualization.png", dpi=200, bbox_inches='tight')
    
    # Create and save visualizations
    
    # Create tile distribution
    try:
        tile_dist = TileDistribution(encoding, variables)
        viz_data = tile_dist.get_visualization_data()
        
        print("\nTile Distribution Data:")
        print(f"- Tile Shape: {viz_data.get('tile_shape')}")
        print(f"- Threads: {len(set(viz_data.get('thread_mapping', {}).values()))}")
        
        # Tile distribution visualization
        print("\nGenerating tile distribution visualization...")
        fig2 = visualizer.visualize_tile_distribution(viz_data)
        plt.figure(fig2.number)
        plt.savefig("tile_distribution.png", dpi=150, bbox_inches='tight')
        
        # Thread access pattern visualization
        print("\nGenerating thread access pattern visualizations...")
        for frame in range(0, 10, 3):  # Generate a few frames
            fig3 = visualizer.visualize_thread_access_pattern(viz_data, frame, 10)
            plt.figure(fig3.number)
            plt.savefig(f"thread_access_frame_{frame}.png", dpi=150, bbox_inches='tight')
            plt.close(fig3)
        
        print("\nVisualizations saved to PNG files.")
        
        # Display the indexing visualization and first thread pattern frame
        plt.figure(fig0.number)
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        raise

if __name__ == "__main__":
    print("Testing tile distribution visualization")
    test_variable_based_template() 