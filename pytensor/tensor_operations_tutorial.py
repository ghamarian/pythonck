"""
Enhanced tutorial for CK Tensor Operations with C++ code correspondence.

This module provides a comprehensive tutorial on tensor operations in CK,
including load/store, shuffle, update, and sweep operations. Each operation
is explained with visualizations and corresponding C++ implementations.

Key Operations Covered:
1. load_tile: Loading data from global memory to registers
2. store_tile: Storing data from registers to global memory  
3. shuffle_tile: Data exchange between threads
4. update_tile: In-place tensor modifications
5. sweep_tile: Iterating over tensor with accumulation
6. slice_tile: Extracting sub-tensors
7. transpose_tile: Tensor dimension reordering

Author: CK Tutorial
License: MIT
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Arrow
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import networkx as nx

# Import base implementations
from .load_tile import load_tile
from .store_tile import store_tile  
from .shuffle_tile import shuffle_tile
from .update_tile import update_tile
from .sweep_tile import sweep_tile


# =============================================================================
# SECTION 1: Load and Store Operations
# =============================================================================

class LoadStoreTutorial:
    """
    Tutorial for tile load and store operations - the foundation of data movement.
    """
    
    @staticmethod
    def explain_load_operation():
        """
        Explain and visualize the load_tile operation.
        
        C++ REFERENCE:
        From ck_tile/core/tensor/load_tile.hpp:
        ```cpp
        template <typename DstTile_, typename SrcTensorView_>
        CK_TILE_DEVICE void load_tile(DstTile_& dst_tile,
                                     const SrcTensorView_& src_tensor_view)
        {
            // Each thread loads its portion of the tile
            constexpr auto I0 = number<0>{};
            const auto src_coord = src_tensor_view.get_tensor_coordinate();
            
            // Vectorized load for efficiency
            const auto [adapted_src_coord, valid_elements] = 
                make_load_tile_window(src_tensor_view, dst_tile.get_tile_distribution());
                
            dst_tile.get_thread_buffer() = src_tensor_view.get_vectorized_elements(adapted_src_coord);
        }
        ```
        """
        print("=== Load Tile Operation ===")
        print("Transfers data from global memory to thread-local registers")
        print()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Source tensor in global memory
        ax1.set_title('Global Memory (Source)', fontsize=14)
        ax1.set_xlim(0, 8)
        ax1.set_ylim(0, 8)
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        ax1.invert_yaxis()
        
        # Draw source tensor
        for i in range(8):
            for j in range(8):
                value = i * 8 + j
                color_intensity = value / 63.0
                rect = Rectangle((j, i), 1, 1,
                               linewidth=1, edgecolor='black',
                               facecolor=plt.cm.Blues(color_intensity),
                               alpha=0.7)
                ax1.add_patch(rect)
                ax1.text(j + 0.5, i + 0.5, f'{value}',
                        ha='center', va='center', fontsize=8)
        
        # Highlight tile to load
        tile_rect = Rectangle((1, 1), 4, 4,
                            linewidth=3, edgecolor='red',
                            facecolor='none')
        ax1.add_patch(tile_rect)
        ax1.text(3, 0.3, 'Tile to Load', ha='center', 
                fontsize=10, color='red', fontweight='bold')
        
        # Thread assignment
        ax2.set_title('Thread Assignment', fontsize=14)
        ax2.set_xlim(0, 4)
        ax2.set_ylim(0, 4)
        ax2.set_xlabel('Tile Column')
        ax2.set_ylabel('Tile Row')
        ax2.invert_yaxis()
        
        # Show how threads map to tile elements
        thread_colors = plt.cm.tab10(np.arange(16) / 16)
        thread_id = 0
        for i in range(4):
            for j in range(4):
                rect = Rectangle((j, i), 1, 1,
                               linewidth=1, edgecolor='black',
                               facecolor=thread_colors[thread_id],
                               alpha=0.7)
                ax2.add_patch(rect)
                ax2.text(j + 0.5, i + 0.5, f'T{thread_id}',
                        ha='center', va='center', fontsize=8)
                thread_id += 1
        
        ax2.text(2, -0.5, 'Each thread loads one element',
                ha='center', fontsize=10)
        
        # Loaded tile in registers
        ax3.set_title('Thread Registers (Destination)', fontsize=14)
        ax3.set_xlim(0, 4)
        ax3.set_ylim(0, 4)
        ax3.set_xlabel('Register Tile Column')
        ax3.set_ylabel('Register Tile Row')
        ax3.invert_yaxis()
        
        # Show loaded values
        for i in range(4):
            for j in range(4):
                global_i = 1 + i
                global_j = 1 + j
                value = global_i * 8 + global_j
                thread_id = i * 4 + j
                
                rect = Rectangle((j, i), 1, 1,
                               linewidth=1, edgecolor='black',
                               facecolor=thread_colors[thread_id],
                               alpha=0.7)
                ax3.add_patch(rect)
                ax3.text(j + 0.5, i + 0.5, f'{value}',
                        ha='center', va='center', fontsize=8,
                        fontweight='bold')
        
        ax3.text(2, -0.5, 'Values loaded to registers',
                ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        print("\nC++ Usage Example:")
        print("```cpp")
        print("// Create source tensor view")
        print("auto src_tensor = make_tensor_view<float>(global_ptr, shape, stride);")
        print("")
        print("// Create destination tile in registers")
        print("auto dst_tile = make_static_distributed_tensor<float, TileShape>();")
        print("")
        print("// Load tile from global memory")
        print("load_tile(dst_tile, src_tensor);")
        print("```")
    
    @staticmethod
    def explain_store_operation():
        """
        Explain and visualize the store_tile operation.
        """
        print("\n=== Store Tile Operation ===")
        print("Transfers data from thread registers back to global memory")
        print()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Source tile in registers
        ax1.set_title('Thread Registers (Source)', fontsize=14)
        ax1.set_xlim(0, 4)
        ax1.set_ylim(0, 4)
        ax1.invert_yaxis()
        
        thread_colors = plt.cm.tab10(np.arange(16) / 16)
        for i in range(4):
            for j in range(4):
                thread_id = i * 4 + j
                value = thread_id * 2  # Example computation result
                
                rect = Rectangle((j, i), 1, 1,
                               linewidth=1, edgecolor='black',
                               facecolor=thread_colors[thread_id],
                               alpha=0.7)
                ax1.add_patch(rect)
                ax1.text(j + 0.5, i + 0.5, f'{value}',
                        ha='center', va='center', fontsize=8,
                        fontweight='bold')
        
        # Arrow showing data flow
        ax1.annotate('', xy=(4.5, 2), xytext=(4.2, 2),
                    arrowprops=dict(arrowstyle='->', lw=3, color='green'))
        
        # Destination in global memory
        ax2.set_title('Global Memory (Destination)', fontsize=14)
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 8)
        ax2.invert_yaxis()
        
        # Draw full tensor with stored tile
        for i in range(8):
            for j in range(8):
                if 2 <= i < 6 and 2 <= j < 6:
                    # Stored tile region
                    local_i = i - 2
                    local_j = j - 2
                    thread_id = local_i * 4 + local_j
                    value = thread_id * 2
                    color = thread_colors[thread_id]
                    alpha = 0.7
                else:
                    # Rest of tensor
                    value = 0
                    color = 'lightgray'
                    alpha = 0.3
                
                rect = Rectangle((j, i), 1, 1,
                               linewidth=1, edgecolor='black',
                               facecolor=color, alpha=alpha)
                ax2.add_patch(rect)
                ax2.text(j + 0.5, i + 0.5, f'{value}',
                        ha='center', va='center', fontsize=7)
        
        # Highlight stored region
        store_rect = Rectangle((2, 2), 4, 4,
                             linewidth=3, edgecolor='green',
                             facecolor='none')
        ax2.add_patch(store_rect)
        ax2.text(4, 1.3, 'Stored Tile', ha='center',
                fontsize=10, color='green', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("\nKey Points:")
        print("1. Each thread stores its register value to the assigned memory location")
        print("2. Coalesced stores are critical for performance")
        print("3. Synchronization may be needed after store operations")
        print()
        print("C++ Pattern:")
        print("```cpp")
        print("// Store tile back to global memory")
        print("store_tile(dst_tensor_view, src_tile);")
        print("__syncthreads(); // Ensure all stores complete")
        print("```")
    
    @staticmethod
    def demonstrate_vectorized_io():
        """
        Show vectorized vs scalar load/store operations.
        """
        print("\n=== Vectorized I/O Operations ===")
        print("CK uses vector types to maximize memory bandwidth")
        print()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Scalar load
        ax = axes[0, 0]
        ax.set_title('Scalar Load (4 transactions)', fontsize=12)
        ax.set_xlim(-1, 5)
        ax.set_ylim(0, 3)
        
        # Memory locations
        for i in range(4):
            rect = Rectangle((i, 0.5), 1, 1,
                           linewidth=1, edgecolor='black',
                           facecolor='lightblue', alpha=0.5)
            ax.add_patch(rect)
            ax.text(i + 0.5, 1, f'M[{i}]', ha='center', va='center', fontsize=8)
            
            # Individual loads
            arrow = Arrow(i + 0.5, 0.4, 0, -0.3, width=0.2,
                        color='red', alpha=0.7)
            ax.add_patch(arrow)
            ax.text(i + 0.5, 0, f'LD', ha='center', fontsize=6)
        
        ax.text(2, 2.5, 'for(i=0; i<4; i++)\n  r[i] = mem[i];',
               ha='center', fontsize=9, fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax.axis('off')
        
        # Vector load
        ax = axes[0, 1]
        ax.set_title('Vector Load (1 transaction)', fontsize=12)
        ax.set_xlim(-1, 5)
        ax.set_ylim(0, 3)
        
        # Memory locations
        for i in range(4):
            rect = Rectangle((i, 0.5), 1, 1,
                           linewidth=1, edgecolor='black',
                           facecolor='lightgreen', alpha=0.5)
            ax.add_patch(rect)
            ax.text(i + 0.5, 1, f'M[{i}]', ha='center', va='center', fontsize=8)
        
        # Single vector load
        rect = Rectangle((0, -0.1), 4, 0.4,
                       linewidth=2, edgecolor='green',
                       facecolor='green', alpha=0.3)
        ax.add_patch(rect)
        ax.text(2, 0.1, 'LD.128', ha='center', fontsize=10,
               fontweight='bold', color='green')
        
        ax.text(2, 2.5, 'float4 v = \n  *(float4*)mem;',
               ha='center', fontsize=9, fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax.axis('off')
        
        # Performance comparison
        ax = axes[1, 0]
        ax.set_title('Performance Comparison', fontsize=12)
        
        categories = ['Scalar', 'Vector2', 'Vector4', 'Vector8']
        bandwidth = [25, 50, 100, 100]  # Relative bandwidth utilization
        colors = ['red', 'orange', 'green', 'blue']
        
        bars = ax.bar(categories, bandwidth, color=colors, alpha=0.7)
        ax.set_ylabel('Bandwidth Utilization (%)')
        ax.set_ylim(0, 120)
        
        for bar, bw in zip(bars, bandwidth):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{bw}%', ha='center', va='bottom')
        
        # CK vector type usage
        ax = axes[1, 1]
        ax.set_title('CK Vector Types', fontsize=12)
        ax.axis('off')
        
        vector_info = """CK Vector Type Support:
        
// Define vector types
using float2 = vector_type<float, 2>::type;
using float4 = vector_type<float, 4>::type;
using half8 = vector_type<half_t, 8>::type;

// Automatic vectorization
template <typename T, int N>
CK_TILE_DEVICE void load_tile_vectorized(
    const T* src, T* dst, index_t size) 
{
    constexpr int vec_size = 
        get_vector_size<T, N>();
    using vec_t = 
        typename vector_type<T, vec_size>::type;
    
    // Vectorized load loop
    for(index_t i = 0; i < size; i += vec_size) {
        *reinterpret_cast<vec_t*>(dst + i) = 
        *reinterpret_cast<const vec_t*>(src + i);
    }
}"""
        
        ax.text(0.05, 0.95, vector_info, transform=ax.transAxes,
               fontsize=8, fontfamily='monospace', va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# SECTION 2: Shuffle Operations
# =============================================================================

class ShuffleTutorial:
    """
    Tutorial for shuffle_tile operations - inter-thread data exchange.
    """
    
    @staticmethod
    def explain_shuffle_operation():
        """
        Explain thread shuffle operations.
        
        C++ REFERENCE:
        From ck_tile/core/tensor/shuffle_tile.hpp:
        Shuffle operations allow threads to exchange data without
        going through shared memory.
        """
        print("\n=== Shuffle Tile Operation ===")
        print("Enables direct data exchange between threads in a warp")
        print()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Before shuffle
        ax = axes[0, 0]
        ax.set_title('Before Shuffle', fontsize=12)
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 5)
        
        # Draw thread lanes
        thread_data = [10, 20, 30, 40, 50, 60, 70, 80]
        for i, data in enumerate(thread_data):
            # Thread box
            rect = Rectangle((i, 1), 1, 2,
                           linewidth=2, edgecolor='black',
                           facecolor='lightblue', alpha=0.7)
            ax.add_patch(rect)
            ax.text(i + 0.5, 2, f'{data}', ha='center', va='center',
                   fontsize=10, fontweight='bold')
            ax.text(i + 0.5, 0.5, f'T{i}', ha='center', va='center',
                   fontsize=8)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(4, 4, 'Each thread has its own value', ha='center', fontsize=10)
        
        # Shuffle pattern
        ax = axes[0, 1]
        ax.set_title('Shuffle Pattern (Shift Right by 1)', fontsize=12)
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 5)
        
        # Show shuffle arrows
        for i in range(len(thread_data)):
            src = i
            dst = (i + 1) % len(thread_data)
            
            # Source position
            src_x = src + 0.5
            dst_x = dst + 0.5
            
            # Draw arrow
            arrow = Arrow(src_x, 3.5, dst_x - src_x, 0,
                        width=0.2, color='red', alpha=0.7)
            ax.add_patch(arrow)
            
            # Thread boxes
            rect = Rectangle((i, 1), 1, 2,
                           linewidth=1, edgecolor='gray',
                           facecolor='white', alpha=0.3)
            ax.add_patch(rect)
            ax.text(i + 0.5, 0.5, f'T{i}', ha='center', va='center',
                   fontsize=8)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(4, 4, 'Data moves between threads', ha='center', 
                fontsize=10, color='red')
        
        # After shuffle
        ax = axes[1, 0]
        ax.set_title('After Shuffle', fontsize=12)
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 5)
        
        # Draw thread lanes with shuffled data
        shuffled_data = [80] + thread_data[:-1]  # Shifted right by 1
        for i, data in enumerate(shuffled_data):
            # Thread box
            rect = Rectangle((i, 1), 1, 2,
                           linewidth=2, edgecolor='black',
                           facecolor='lightgreen', alpha=0.7)
            ax.add_patch(rect)
            ax.text(i + 0.5, 2, f'{data}', ha='center', va='center',
                   fontsize=10, fontweight='bold')
            ax.text(i + 0.5, 0.5, f'T{i}', ha='center', va='center',
                   fontsize=8)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(4, 4, 'Each thread now has data from its neighbor',
                ha='center', fontsize=10)
        
        # Common shuffle patterns
        ax = axes[1, 1]
        ax.set_title('Common Shuffle Patterns', fontsize=12)
        ax.axis('off')
        
        patterns = """Common Shuffle Patterns in CK:

1. Shift/Rotate:
   __shfl_down(value, delta)
   __shfl_up(value, delta)

2. Broadcast:
   __shfl(value, src_lane)

3. Butterfly/XOR:
   __shfl_xor(value, mask)

C++ Example:
```cpp
// Warp-level reduction using shuffle
template <typename T>
CK_TILE_DEVICE T warp_reduce_sum(T value) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down(value, offset);
    }
    return value;
}
```"""
        
        ax.text(0.05, 0.95, patterns, transform=ax.transAxes,
               fontsize=9, fontfamily='monospace', va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def demonstrate_shuffle_applications():
        """
        Show practical applications of shuffle operations.
        """
        print("\n=== Shuffle Applications ===")
        print("Real-world uses of shuffle in CK")
        print()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Matrix transpose using shuffle
        ax1.set_title('Matrix Transpose via Shuffle', fontsize=14)
        ax1.set_xlim(-1, 5)
        ax1.set_ylim(-1, 5)
        ax1.invert_yaxis()
        
        # Original 4x4 matrix distributed across threads
        ax1.text(2, -0.5, 'Original (Row-major)', ha='center', fontsize=10)
        
        colors = plt.cm.tab10(np.arange(16) / 16)
        for i in range(4):
            for j in range(4):
                tid = i * 4 + j
                value = tid
                rect = Rectangle((j, i), 1, 1,
                               linewidth=1, edgecolor='black',
                               facecolor=colors[tid], alpha=0.5)
                ax1.add_patch(rect)
                ax1.text(j + 0.5, i + 0.5, f'{value}',
                        ha='center', va='center', fontsize=8)
        
        # Arrow
        ax1.annotate('', xy=(5.5, 2), xytext=(4.5, 2),
                    arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
        ax1.text(6, 2, 'Shuffle\nTranspose', ha='left', va='center',
                fontsize=10, color='blue')
        
        # Reduction tree using shuffle
        ax2.set_title('Parallel Reduction using Shuffle', fontsize=14)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 6)
        
        # Create reduction tree
        levels = 4
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        
        # Draw tree
        y_positions = [5, 3.5, 2, 0.5]
        x_starts = [1, 1.75, 2.875, 4.5]
        x_gaps = [1, 2, 4, 0]
        
        for level in range(levels):
            y = y_positions[level]
            x_start = x_starts[level]
            x_gap = x_gaps[level]
            
            if level == 0:
                # Initial values
                for i, val in enumerate(values):
                    circle = Circle((x_start + i, y), 0.3,
                                  facecolor='lightblue', edgecolor='black')
                    ax2.add_patch(circle)
                    ax2.text(x_start + i, y, f'{val}',
                            ha='center', va='center', fontsize=8)
            else:
                # Reduction results
                num_nodes = 2 ** (levels - level)
                for i in range(num_nodes):
                    x = x_start + i * x_gap
                    circle = Circle((x, y), 0.3,
                                  facecolor='lightgreen', edgecolor='black')
                    ax2.add_patch(circle)
                    
                    # Connect to children
                    if level < levels:
                        child_y = y_positions[level - 1]
                        child1_x = x - x_gap / 4
                        child2_x = x + x_gap / 4
                        
                        ax2.plot([x, child1_x], [y, child_y], 'k-', alpha=0.5)
                        ax2.plot([x, child2_x], [y, child_y], 'k-', alpha=0.5)
                    
                    # Show sum
                    if level == 1:
                        sums = [3, 7, 11, 15]
                        ax2.text(x, y, f'{sums[i]}', ha='center', va='center',
                                fontsize=8)
                    elif level == 2:
                        sums = [10, 26]
                        ax2.text(x, y, f'{sums[i]}', ha='center', va='center',
                                fontsize=8)
                    elif level == 3:
                        ax2.text(x, y, '36', ha='center', va='center',
                                fontsize=10, fontweight='bold')
        
        # Labels
        ax2.text(8, 5, 'Step 1:\nPairwise', ha='center', fontsize=8)
        ax2.text(8, 3.5, 'Step 2:\nDistance 2', ha='center', fontsize=8)
        ax2.text(8, 2, 'Step 3:\nDistance 4', ha='center', fontsize=8)
        ax2.text(8, 0.5, 'Final:\nResult', ha='center', fontsize=8)
        
        ax2.set_xlim(0, 10)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# SECTION 3: Update and Transform Operations
# =============================================================================

class UpdateTransformTutorial:
    """
    Tutorial for update_tile and transformation operations.
    """
    
    @staticmethod
    def explain_update_operation():
        """
        Explain update_tile for in-place modifications.
        
        C++ REFERENCE:
        update_tile applies element-wise operations to tensor data.
        """
        print("\n=== Update Tile Operation ===")
        print("Applies element-wise transformations to tile data")
        print()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Original tile
        ax = axes[0, 0]
        ax.set_title('Original Tile', fontsize=12)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.invert_yaxis()
        
        original_data = np.arange(16).reshape(4, 4)
        im = ax.imshow(original_data, cmap='viridis', alpha=0.7)
        
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{original_data[i, j]}',
                       ha='center', va='center', fontsize=10,
                       color='white', fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Update function
        ax = axes[0, 1]
        ax.set_title('Update Function', fontsize=12)
        ax.text(0.5, 0.5, 'f(x) = x * 2 + 1', ha='center', va='center',
               fontsize=16, fontweight='bold',
               transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow"))
        ax.axis('off')
        
        # Updated tile
        ax = axes[0, 2]
        ax.set_title('Updated Tile', fontsize=12)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.invert_yaxis()
        
        updated_data = original_data * 2 + 1
        im = ax.imshow(updated_data, cmap='plasma', alpha=0.7)
        
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{updated_data[i, j]}',
                       ha='center', va='center', fontsize=10,
                       color='white', fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Common update operations
        operations = [
            ('Scale', lambda x: x * 0.5, 'viridis'),
            ('Clamp', lambda x: np.clip(x, 5, 10), 'coolwarm'),
            ('Activation', lambda x: np.maximum(0, x - 8), 'RdYlBu')
        ]
        
        for idx, (name, func, cmap) in enumerate(operations):
            ax = axes[1, idx]
            ax.set_title(f'{name} Operation', fontsize=12)
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 4)
            ax.invert_yaxis()
            
            result_data = func(original_data)
            im = ax.imshow(result_data, cmap=cmap, alpha=0.7)
            
            for i in range(4):
                for j in range(4):
                    value = result_data[i, j]
                    # Format based on operation
                    if name == 'Scale':
                        text = f'{value:.1f}'
                    else:
                        text = f'{int(value)}'
                    
                    ax.text(j, i, text, ha='center', va='center',
                           fontsize=9, color='white', fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()
        
        print("\nC++ Update Patterns:")
        print("```cpp")
        print("// Element-wise update with lambda")
        print("update_tile(tile, [](auto& x) { return x * 2 + 1; });")
        print("")
        print("// Update with binary operation")
        print("update_tile(tile_a, tile_b, [](auto a, auto b) { return a + b; });")
        print("")
        print("// Conditional update")
        print("update_tile_if(tile, ")
        print("    [](auto x) { return x > threshold; },  // predicate")
        print("    [](auto x) { return x * scale; });     // update function")
        print("```")
    
    @staticmethod
    def demonstrate_fusion():
        """
        Show operation fusion capabilities.
        """
        print("\n=== Operation Fusion ===")
        print("Combining multiple operations for efficiency")
        print()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Unfused operations
        ax1.set_title('Unfused Operations (3 Kernel Launches)', fontsize=14)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 8)
        ax1.axis('off')
        
        # Draw operation pipeline
        operations = [
            ('Load A', 1, 6, 'lightblue'),
            ('Scale', 3, 6, 'lightgreen'),
            ('Store', 5, 6, 'lightcoral'),
            ('Load B', 1, 4, 'lightblue'),
            ('Add Bias', 3, 4, 'lightyellow'),
            ('Store', 5, 4, 'lightcoral'),
            ('Load C', 1, 2, 'lightblue'),
            ('Activation', 3, 2, 'lightpink'),
            ('Store', 5, 2, 'lightcoral')
        ]
        
        for op, x, y, color in operations:
            rect = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='black',
                                linewidth=2)
            ax1.add_patch(rect)
            ax1.text(x, y, op, ha='center', va='center', fontsize=9)
        
        # Arrows
        for i in range(3):
            y = 6 - i * 2
            ax1.arrow(2.3, y, 0.4, 0, head_width=0.15, 
                     head_length=0.1, fc='black', ec='black')
            ax1.arrow(4.3, y, 0.4, 0, head_width=0.15,
                     head_length=0.1, fc='black', ec='black')
        
        # Kernel boundaries
        for i in range(3):
            y = 7 - i * 2
            rect = Rectangle((0.2, y-1.2), 6, 1.5,
                           linewidth=2, edgecolor='red',
                           facecolor='none', linestyle='--')
            ax1.add_patch(rect)
            ax1.text(6.5, y-0.5, f'Kernel {i+1}', fontsize=8, color='red')
        
        # Memory accesses
        ax1.text(8, 6, '3x Read', fontsize=10, color='blue')
        ax1.text(8, 5, '3x Write', fontsize=10, color='red')
        ax1.text(8, 4, 'Total: 6', fontsize=10, fontweight='bold')
        
        # Fused operation
        ax2.set_title('Fused Operation (1 Kernel Launch)', fontsize=14)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 8)
        ax2.axis('off')
        
        # Single fused kernel
        fused_rect = Rectangle((0.5, 3), 7, 2,
                             linewidth=3, edgecolor='green',
                             facecolor='lightgreen', alpha=0.3)
        ax2.add_patch(fused_rect)
        
        # Operations inside
        fused_ops = [
            ('Load', 1.5, 4, 'lightblue'),
            ('Scale', 2.5, 4, 'lightgreen'),
            ('Add Bias', 3.5, 4, 'lightyellow'),
            ('Activation', 4.5, 4, 'lightpink'),
            ('Store', 5.5, 4, 'lightcoral')
        ]
        
        for op, x, y, color in fused_ops:
            circle = Circle((x, y), 0.4, facecolor=color,
                          edgecolor='black', linewidth=2)
            ax2.add_patch(circle)
            ax2.text(x, y, op, ha='center', va='center', fontsize=8)
        
        # Arrows between operations
        for i in range(len(fused_ops) - 1):
            x1 = 1.5 + i
            x2 = 2.5 + i
            ax2.arrow(x1 + 0.4, 4, x2 - x1 - 0.8, 0,
                     head_width=0.1, head_length=0.05,
                     fc='black', ec='black', alpha=0.5)
        
        ax2.text(3.5, 5.5, 'Single Fused Kernel', ha='center',
                fontsize=12, fontweight='bold', color='green')
        
        # Memory accesses
        ax2.text(8, 6, '1x Read', fontsize=10, color='blue')
        ax2.text(8, 5, '1x Write', fontsize=10, color='red')
        ax2.text(8, 4, 'Total: 2', fontsize=10, fontweight='bold')
        ax2.text(8, 3, '3x Speedup!', fontsize=12, 
                color='green', fontweight='bold')
        
        # C++ code
        ax2.text(3.5, 1.5, 
                'update_tile(tile, [](auto x) {\n'
                '    x = x * scale;\n'
                '    x = x + bias;\n'
                '    x = activation(x);\n'
                '    return x;\n'
                '});',
                ha='center', va='center', fontsize=8,
                fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# SECTION 4: Sweep Operations
# =============================================================================

class SweepTutorial:
    """
    Tutorial for sweep_tile operations - iterating with accumulation.
    """
    
    @staticmethod
    def explain_sweep_operation():
        """
        Explain sweep_tile for reduction and scanning operations.
        """
        print("\n=== Sweep Tile Operation ===")
        print("Iterates over tile elements with accumulation")
        print()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Sweep for reduction
        ax = axes[0, 0]
        ax.set_title('Sweep for Sum Reduction', fontsize=12)
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        
        # Draw tile
        values = [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
        
        for i in range(4):
            for j in range(4):
                rect = Rectangle((j, 3-i), 1, 1,
                               linewidth=1, edgecolor='black',
                               facecolor='lightblue', alpha=0.5)
                ax.add_patch(rect)
                ax.text(j + 0.5, 3-i + 0.5, f'{values[i][j]}',
                       ha='center', va='center', fontsize=9)
        
        # Show sweep order
        sweep_order = [(0,0), (0,1), (0,2), (0,3),
                      (1,0), (1,1), (1,2), (1,3)]
        
        for idx, (i, j) in enumerate(sweep_order[:8]):
            if idx < 7:
                x1, y1 = j + 0.5, 3-i + 0.5
                i2, j2 = sweep_order[idx + 1]
                x2, y2 = j2 + 0.5, 3-i2 + 0.5
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color='red',
                                         lw=2, alpha=0.7))
        
        ax.text(2, -0.5, 'Accumulator: Sum = 136',
               ha='center', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Sweep patterns
        ax = axes[0, 1]
        ax.set_title('Common Sweep Patterns', fontsize=12)
        ax.axis('off')
        
        patterns_text = """Sweep Patterns:

1. Row-major (→):
   for(i = 0; i < M; i++)
     for(j = 0; j < N; j++)
       accumulate(tile[i][j])

2. Column-major (↓):
   for(j = 0; j < N; j++)
     for(i = 0; i < M; i++)
       accumulate(tile[i][j])

3. Diagonal:
   for(k = 0; k < M+N; k++)
     for(i,j where i+j==k)
       accumulate(tile[i][j])

4. Space-filling curve:
   for(idx in hilbert_curve)
     accumulate(tile[idx])"""
        
        ax.text(0.05, 0.95, patterns_text, transform=ax.transAxes,
               fontsize=9, fontfamily='monospace', va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        # Parallel sweep (prefix sum)
        ax = axes[1, 0]
        ax.set_title('Parallel Sweep (Prefix Sum)', fontsize=12)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        
        # Initial values
        initial = [1, 2, 3, 4, 5, 6, 7, 8]
        y_pos = 4
        
        for i, val in enumerate(initial):
            rect = Rectangle((i+1, y_pos-0.3), 0.8, 0.6,
                           facecolor='lightblue', edgecolor='black')
            ax.add_patch(rect)
            ax.text(i+1.4, y_pos, f'{val}', ha='center', va='center',
                   fontsize=8)
        
        ax.text(0, y_pos, 'Input:', ha='right', va='center', fontsize=9)
        
        # Step 1
        y_pos = 3
        step1 = [1, 3, 5, 7, 9, 11, 13, 15]
        for i, val in enumerate(step1):
            color = 'lightgreen' if i % 2 == 1 else 'white'
            rect = Rectangle((i+1, y_pos-0.3), 0.8, 0.6,
                           facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(i+1.4, y_pos, f'{val}', ha='center', va='center',
                   fontsize=8)
        
        ax.text(0, y_pos, 'Step 1:', ha='right', va='center', fontsize=9)
        
        # Step 2
        y_pos = 2
        step2 = [1, 3, 6, 10, 14, 18, 22, 28]
        for i, val in enumerate(step2):
            color = 'lightyellow' if i >= 2 else 'white'
            rect = Rectangle((i+1, y_pos-0.3), 0.8, 0.6,
                           facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(i+1.4, y_pos, f'{val}', ha='center', va='center',
                   fontsize=8)
        
        ax.text(0, y_pos, 'Step 2:', ha='right', va='center', fontsize=9)
        
        # Final
        y_pos = 1
        final = [1, 3, 6, 10, 15, 21, 28, 36]
        for i, val in enumerate(final):
            rect = Rectangle((i+1, y_pos-0.3), 0.8, 0.6,
                           facecolor='lightcoral', edgecolor='black')
            ax.add_patch(rect)
            ax.text(i+1.4, y_pos, f'{val}', ha='center', va='center',
                   fontsize=8, fontweight='bold')
        
        ax.text(0, y_pos, 'Output:', ha='right', va='center', fontsize=9)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0.5, 4.5)
        ax.axis('off')
        
        # C++ implementation
        ax = axes[1, 1]
        ax.set_title('C++ Sweep Implementation', fontsize=12)
        ax.axis('off')
        
        cpp_code = """// Sum reduction using sweep
template <typename Tile>
CK_TILE_DEVICE auto sweep_reduce_sum(const Tile& tile) {
    using T = typename Tile::DataType;
    T sum = 0;
    
    sweep_tile(tile, [&sum](const T& element) {
        sum += element;
    });
    
    return sum;
}

// Parallel prefix sum
template <typename Tile>
CK_TILE_DEVICE void sweep_prefix_sum(Tile& tile) {
    constexpr auto tile_size = Tile::size();
    
    // Up-sweep phase
    for(index_t d = 0; d < log2(tile_size); d++) {
        index_t stride = 1 << (d + 1);
        
        sweep_tile_stride(tile, stride,
            [stride](auto& elem, index_t idx) {
                if(idx % stride == stride - 1) {
                    elem += tile[idx - stride/2];
                }
            });
        __syncthreads();
    }
}"""
        
        ax.text(0.05, 0.95, cpp_code, transform=ax.transAxes,
               fontsize=7, fontfamily='monospace', va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# SECTION 5: Complete Examples
# =============================================================================

class TensorOperationsExamples:
    """
    Complete examples combining multiple tensor operations.
    """
    
    @staticmethod
    def matrix_multiplication_example():
        """
        Show how tensor operations combine in matrix multiplication.
        """
        print("\n=== Matrix Multiplication with Tensor Operations ===")
        print("Complete example showing all operations")
        print()
        
        # Create visualization of GEMM pipeline
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 2, 1])
        
        # Title
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.text(0.5, 0.5, 'GEMM Pipeline with Tensor Operations',
                     ha='center', va='center', fontsize=18, fontweight='bold',
                     transform=ax_title.transAxes)
        ax_title.axis('off')
        
        # Load A
        ax_load_a = fig.add_subplot(gs[1, 0])
        ax_load_a.set_title('1. Load Tile A', fontsize=12)
        ax_load_a.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightblue'))
        ax_load_a.text(0.5, 0.5, 'load_tile(\n  a_tile,\n  a_view)',
                      ha='center', va='center', fontsize=10,
                      transform=ax_load_a.transAxes)
        ax_load_a.axis('off')
        
        # Load B
        ax_load_b = fig.add_subplot(gs[1, 1])
        ax_load_b.set_title('2. Load Tile B', fontsize=12)
        ax_load_b.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgreen'))
        ax_load_b.text(0.5, 0.5, 'load_tile(\n  b_tile,\n  b_view)',
                      ha='center', va='center', fontsize=10,
                      transform=ax_load_b.transAxes)
        ax_load_b.axis('off')
        
        # Compute
        ax_compute = fig.add_subplot(gs[1, 2])
        ax_compute.set_title('3. Compute C += A×B', fontsize=12)
        ax_compute.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightyellow'))
        ax_compute.text(0.5, 0.5, 'update_tile(\n  c_tile,\n  gemm_op)',
                      ha='center', va='center', fontsize=10,
                      transform=ax_compute.transAxes)
        ax_compute.axis('off')
        
        # Store C
        ax_store_c = fig.add_subplot(gs[1, 3])
        ax_store_c.set_title('4. Store Tile C', fontsize=12)
        ax_store_c.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightcoral'))
        ax_store_c.text(0.5, 0.5, 'store_tile(\n  c_view,\n  c_tile)',
                      ha='center', va='center', fontsize=10,
                      transform=ax_store_c.transAxes)
        ax_store_c.axis('off')
        
        # Code example
        ax_code = fig.add_subplot(gs[2, :])
        ax_code.axis('off')
        
        code = """// Complete GEMM kernel using tensor operations
template <typename TileShape>
__global__ void gemm_kernel(const float* A, const float* B, float* C, 
                           index_t M, index_t N, index_t K) {
    // 1. Create tile distributions
    auto a_distribution = make_tile_distribution<TileShape::MPerBlock, TileShape::KPerBlock>();
    auto b_distribution = make_tile_distribution<TileShape::KPerBlock, TileShape::NPerBlock>();
    auto c_distribution = make_tile_distribution<TileShape::MPerBlock, TileShape::NPerBlock>();
    
    // 2. Create tiles in registers
    auto a_tile = make_static_distributed_tensor<float, TileShape::MPerBlock, TileShape::KPerBlock>();
    auto b_tile = make_static_distributed_tensor<float, TileShape::KPerBlock, TileShape::NPerBlock>();
    auto c_tile = make_static_distributed_tensor<float, TileShape::MPerBlock, TileShape::NPerBlock>();
    
    // 3. Initialize accumulator
    clear(c_tile);
    
    // 4. Main GEMM loop
    for(index_t k = 0; k < K; k += TileShape::KPerBlock) {
        // Load tiles from global memory
        load_tile(a_tile, make_tensor_view(A + ..., a_distribution));
        load_tile(b_tile, make_tensor_view(B + ..., b_distribution));
        
        // Perform tile-level GEMM
        update_tile(c_tile, a_tile, b_tile, 
            [](auto& c, const auto& a, const auto& b) { 
                return c + a * b; 
            });
    }
    
    // 5. Store result
    store_tile(make_tensor_view(C + ..., c_distribution), c_tile);
}"""
        
        ax_code.text(0.5, 0.5, code, ha='center', va='center',
                    fontsize=8, fontfamily='monospace',
                    transform=ax_code.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def layernorm_example():
        """
        Show tensor operations in layer normalization.
        """
        print("\n=== Layer Normalization with Tensor Operations ===")
        print("Combining sweep, update, and shuffle operations")
        print()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Step 1: Load and compute mean
        ax = axes[0, 0]
        ax.set_title('Step 1: Compute Mean', fontsize=12)
        ax.text(0.5, 0.7, 'load_tile(x_tile, x_view)', ha='center',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.text(0.5, 0.5, '↓', ha='center', transform=ax.transAxes, fontsize=16)
        ax.text(0.5, 0.3, 'mean = sweep_reduce_sum(x_tile) / N',
               ha='center', transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax.axis('off')
        
        # Step 2: Compute variance
        ax = axes[0, 1]
        ax.set_title('Step 2: Compute Variance', fontsize=12)
        ax.text(0.5, 0.7, 'update_tile(x_tile, [mean](x) { \n  return (x - mean)² \n})',
               ha='center', transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax.text(0.5, 0.5, '↓', ha='center', transform=ax.transAxes, fontsize=16)
        ax.text(0.5, 0.3, 'var = sweep_reduce_sum(x_tile) / N',
               ha='center', transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax.axis('off')
        
        # Step 3: Normalize
        ax = axes[0, 2]
        ax.set_title('Step 3: Normalize', fontsize=12)
        ax.text(0.5, 0.7, 'reload original x_tile', ha='center',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.text(0.5, 0.5, '↓', ha='center', transform=ax.transAxes, fontsize=16)
        ax.text(0.5, 0.3, 'update_tile(x_tile, [](x) {\n  return (x-mean)/√(var+ε)\n})',
               ha='center', transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.axis('off')
        
        # Step 4: Scale and shift
        ax = axes[1, 0]
        ax.set_title('Step 4: Scale & Shift', fontsize=12)
        ax.text(0.5, 0.7, 'load_tile(gamma_tile, gamma_view)\n'
                         'load_tile(beta_tile, beta_view)',
               ha='center', transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.text(0.5, 0.5, '↓', ha='center', transform=ax.transAxes, fontsize=16)
        ax.text(0.5, 0.3, 'update_tile(x_tile, gamma, beta,\n'
                         '  [](x,g,b) { return g*x + b })',
               ha='center', transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink"))
        ax.axis('off')
        
        # Step 5: Store result
        ax = axes[1, 1]
        ax.set_title('Step 5: Store Result', fontsize=12)
        ax.text(0.5, 0.5, 'store_tile(y_view, x_tile)',
               ha='center', va='center', transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
        ax.axis('off')
        
        # Complete code
        ax = axes[1, 2]
        ax.set_title('Complete LayerNorm', fontsize=12)
        ax.text(0.05, 0.95, 
                '// Fused LayerNorm\n'
                'auto x = load_tile(...);\n'
                'auto mean = reduce_sum(x)/N;\n'
                'auto var = reduce_sum(\n'
                '  (x-mean)²)/N;\n'
                'x = (x-mean)/sqrt(var+eps);\n'
                'x = gamma * x + beta;\n'
                'store_tile(x);',
                transform=ax.transAxes, fontsize=8,
                fontfamily='monospace', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# SECTION 6: Performance Optimization Guide
# =============================================================================

class PerformanceOptimizationGuide:
    """
    Guide for optimizing tensor operations performance.
    """
    
    @staticmethod
    def optimization_checklist():
        """
        Present optimization checklist for tensor operations.
        """
        print("\n=== Tensor Operations Optimization Checklist ===")
        print()
        
        checklist = """
□ Memory Access Optimization
  ✓ Ensure coalesced memory access patterns
  ✓ Use vectorized loads/stores (float4, half8)
  ✓ Align data to 128-byte boundaries
  ✓ Minimize global memory transactions

□ Operation Fusion
  ✓ Combine multiple operations into single kernel
  ✓ Fuse element-wise operations with loads/stores
  ✓ Use update_tile for in-place transformations
  ✓ Minimize intermediate data movement

□ Tile Size Selection
  ✓ Choose tile sizes that maximize occupancy
  ✓ Balance register usage vs tile size
  ✓ Consider shared memory constraints
  ✓ Align with warp/wavefront size (32/64)

□ Thread Cooperation
  ✓ Use shuffle operations for warp-level reduction
  ✓ Minimize __syncthreads() calls
  ✓ Overlap computation with data movement
  ✓ Use cooperative groups when beneficial

□ Data Type Optimization
  ✓ Use appropriate precision (FP16, BF16, INT8)
  ✓ Leverage tensor cores when available
  ✓ Consider mixed precision strategies
  ✓ Pack small data types (INT4)

□ Hardware-Specific Tuning
  ✓ Target specific GPU architecture
  ✓ Use architecture-specific intrinsics
  ✓ Consider L1/L2 cache configuration
  ✓ Profile and iterate
"""
        print(checklist)
        
        # Visual guide
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Optimization impact
        ax1.set_title('Optimization Impact on Performance', fontsize=14)
        
        optimizations = ['Baseline', '+Coalescing', '+Vectorize', 
                        '+Fusion', '+Shuffle', '+TensorCore']
        performance = [100, 150, 200, 300, 350, 500]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(optimizations)))
        
        bars = ax1.barh(optimizations, performance, color=colors)
        ax1.set_xlabel('Relative Performance (%)')
        ax1.set_xlim(0, 600)
        
        for bar, perf in zip(bars, performance):
            width = bar.get_width()
            ax1.text(width + 10, bar.get_y() + bar.get_height()/2,
                    f'{perf}%', ha='left', va='center', fontweight='bold')
        
        # Memory bandwidth utilization
        ax2.set_title('Memory Bandwidth Utilization', fontsize=14)
        
        categories = ['Unoptimized', 'Optimized']
        theoretical_bw = [100, 100]
        achieved_bw = [25, 85]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, theoretical_bw, width, 
                       label='Theoretical', color='lightgray', alpha=0.7)
        bars2 = ax2.bar(x + width/2, achieved_bw, width,
                       label='Achieved', color='green', alpha=0.7)
        
        ax2.set_ylabel('Bandwidth Utilization (%)')
        ax2.set_ylim(0, 120)
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        
        # Add values
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{int(height)}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# Interactive Tutorial Runner
# =============================================================================

def run_tensor_operations_tutorial():
    """
    Run the complete tensor operations tutorial.
    """
    print("=" * 70)
    print("CK TENSOR OPERATIONS INTERACTIVE TUTORIAL")
    print("=" * 70)
    print()
    
    print("This tutorial covers the fundamental tensor operations in CK:")
    print("1. Load/Store - Moving data between memory and registers")
    print("2. Shuffle - Inter-thread data exchange")
    print("3. Update - Element-wise transformations")
    print("4. Sweep - Iteration with accumulation")
    print("5. Complete examples and optimization")
    print()
    
    # Section 1: Load/Store
    print("\n" + "="*50)
    print("SECTION 1: LOAD AND STORE OPERATIONS")
    print("="*50)
    LoadStoreTutorial.explain_load_operation()
    LoadStoreTutorial.explain_store_operation()
    LoadStoreTutorial.demonstrate_vectorized_io()
    
    # Section 2: Shuffle
    print("\n" + "="*50)
    print("SECTION 2: SHUFFLE OPERATIONS")
    print("="*50)
    ShuffleTutorial.explain_shuffle_operation()
    ShuffleTutorial.demonstrate_shuffle_applications()
    
    # Section 3: Update/Transform
    print("\n" + "="*50)
    print("SECTION 3: UPDATE AND TRANSFORM OPERATIONS")
    print("="*50)
    UpdateTransformTutorial.explain_update_operation()
    UpdateTransformTutorial.demonstrate_fusion()
    
    # Section 4: Sweep
    print("\n" + "="*50)
    print("SECTION 4: SWEEP OPERATIONS")
    print("="*50)
    SweepTutorial.explain_sweep_operation()
    
    # Section 5: Complete Examples
    print("\n" + "="*50)
    print("SECTION 5: COMPLETE EXAMPLES")
    print("="*50)
    TensorOperationsExamples.matrix_multiplication_example()
    TensorOperationsExamples.layernorm_example()
    
    # Section 6: Optimization
    print("\n" + "="*50)
    print("SECTION 6: PERFORMANCE OPTIMIZATION")
    print("="*50)
    PerformanceOptimizationGuide.optimization_checklist()
    
    print("\n" + "="*70)
    print("TUTORIAL COMPLETE!")
    print("="*70)
    print()
    print("Key Takeaways:")
    print("1. Tensor operations are the building blocks of GPU kernels")
    print("2. Efficient memory access is critical for performance")
    print("3. Operation fusion reduces memory traffic")
    print("4. Hardware-aware optimization yields best results")
    print()
    print("Next Steps:")
    print("- Experiment with different tile sizes and access patterns")
    print("- Profile your kernels to identify bottlenecks")
    print("- Study the CK examples for real-world usage patterns")
    print()


if __name__ == "__main__":
    run_tensor_operations_tutorial()