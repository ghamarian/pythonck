"""
Visualization helpers for validation scripts.

These functions help us create clear, helpful visualizations
that make complex concepts easier to understand.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any


def setup_nice_plot(title: str, figsize: Tuple[int, int] = (10, 6)):
    """
    Set up a matplotlib plot with nice defaults.
    
    We use this to make consistent, readable plots throughout
    our validation scripts.
    """
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    return plt


def show_tensor_shape(tensor_data: np.ndarray, name: str = "Tensor"):
    """
    Display tensor shape information in a clear way.
    
    This helps you visualize what we're working with.
    """
    print(f"\n📐 {name} Shape Information:")
    print(f"  Shape: {tensor_data.shape}")
    print(f"  Dimensions: {tensor_data.ndim}")
    print(f"  Total elements: {tensor_data.size}")
    print(f"  Data type: {tensor_data.dtype}")
    
    # Show a sample of the data if it's not too big
    if tensor_data.size <= 20:
        print(f"  Data: {tensor_data}")
    else:
        print(f"  Sample: {tensor_data.flat[:5]}... (showing first 5 elements)")


def visualize_memory_layout(shape: Tuple[int, ...], strides: Tuple[int, ...], name: str = "Memory Layout"):
    """
    Visualize how a tensor is laid out in memory.
    
    This is super helpful for understanding how coordinate transforms work.
    """
    print(f"\n🧠 {name}:")
    print(f"  Shape: {shape}")
    print(f"  Strides: {strides}")
    
    # Calculate memory positions for each element
    total_elements = np.prod(shape)
    print(f"  Total elements: {total_elements}")
    
    # Show how indices map to memory positions
    if len(shape) == 2 and np.prod(shape) <= 16:
        print(f"  Memory mapping (for small 2D case):")
        for i in range(shape[0]):
            for j in range(shape[1]):
                memory_pos = i * strides[0] + j * strides[1]
                print(f"    [{i},{j}] -> memory position {memory_pos}")


def show_coordinate_transform(input_coords: List[int], output_coords: List[int], 
                            transform_name: str = "Transform"):
    """
    Show how coordinates transform through an operation.
    
    This makes it easy to see what's happening in coordinate transforms.
    """
    print(f"\n🔄 {transform_name}:")
    print(f"  Input coordinates:  {input_coords}")
    print(f"  Output coordinates: {output_coords}")
    print(f"  Transformation: {input_coords} -> {output_coords}")


def plot_access_pattern(indices: List[Tuple[int, ...]], shape: Tuple[int, int], 
                       title: str = "Access Pattern"):
    """
    Plot how threads access memory in a 2D tensor.
    
    This is great for visualizing thread cooperation patterns.
    """
    if len(shape) != 2:
        print(f"⚠️  Can only plot 2D access patterns, got shape {shape}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a grid showing the tensor
    grid = np.zeros(shape)
    
    # Mark accessed positions
    for i, (row, col) in enumerate(indices):
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            grid[row, col] = i + 1
    
    # Plot the grid
    im = ax.imshow(grid, cmap='viridis', aspect='equal')
    ax.set_title(title)
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Access Order')
    
    # Add grid lines
    ax.set_xticks(range(shape[1]))
    ax.set_yticks(range(shape[0]))
    ax.grid(True, color='white', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


def compare_layouts(layout1: dict, layout2: dict, title: str = "Layout Comparison"):
    """
    Compare two different tensor layouts side by side.
    
    This helps you see the difference between different approaches.
    """
    print(f"\n📊 {title}:")
    print(f"  Layout 1: {layout1}")
    print(f"  Layout 2: {layout2}")
    
    # Find common keys
    common_keys = set(layout1.keys()) & set(layout2.keys())
    
    if common_keys:
        print(f"  Comparing common properties:")
        for key in sorted(common_keys):
            val1, val2 = layout1[key], layout2[key]
            match = "✅" if val1 == val2 else "❌"
            print(f"    {key}: {val1} vs {val2} {match}")


def show_distribution_breakdown(distribution_info: dict):
    """
    Show how a tile distribution breaks down the work.
    
    This helps you understand what each dimension is doing.
    """
    print(f"\n🔧 Distribution Breakdown:")
    
    for key, value in distribution_info.items():
        if isinstance(value, (list, tuple)):
            print(f"  {key}: {value} (length: {len(value)})")
        else:
            print(f"  {key}: {value}")


def create_simple_diagram(elements: List[str], title: str = "Diagram"):
    """
    Create a simple text-based diagram.
    
    Sometimes a simple ASCII diagram is clearer than a complex plot.
    """
    print(f"\n📋 {title}:")
    
    # Simple box diagram
    max_len = max(len(elem) for elem in elements)
    border = "+" + "-" * (max_len + 2) + "+"
    
    print(f"  {border}")
    for elem in elements:
        print(f"  | {elem:<{max_len}} |")
    print(f"  {border}")


def explain_with_example(concept: str, example_code: str, explanation: str):
    """
    Show a concept with code and explanation.
    
    This follows our code-first approach - show the working example,
    then explain what it does.
    """
    print(f"\n💡 {concept}")
    print(f"   Code example:")
    print(f"   {example_code}")
    print(f"   What this does: {explanation}")


# Helper function to check if we can actually display plots
def can_display_plots() -> bool:
    """
    Check if we can display matplotlib plots.
    
    This is useful for scripts that might run in environments
    without display capabilities.
    """
    try:
        import matplotlib
        # Check if we have a display
        if matplotlib.get_backend() == 'Agg':
            return False
        return True
    except:
        return False 