"""
Enhanced tutorial implementation of CK Tile Window with C++ code correspondence.

This module provides a comprehensive tutorial on tile windows - the fundamental
abstraction for accessing tensor data in a tiled manner on GPUs. It includes
detailed explanations, visualizations, and corresponding C++ code snippets.

Key Concepts:
1. Tile Window: A view into tensor memory organized for efficient GPU access
2. Load/Store Operations: How data moves between global memory and registers
3. Coordinate Mapping: Translation between tile and tensor coordinates
4. Memory Access Patterns: Coalescing, bank conflicts, and optimization

Author: CK Tutorial
License: MIT
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Import base implementations
from .tensor_view import TensorView
from .tensor_descriptor import TensorDescriptor
from .tile_distribution import TileDistribution
from .static_distributed_tensor import StaticDistributedTensor


# =============================================================================
# SECTION 1: Core Tile Window Concepts
# =============================================================================

"""
C++ REFERENCE:
--------------
From ck_tile/core/tensor/tile_window.hpp:

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord>
struct tile_window_with_static_distribution
{
    // Tile window provides:
    // 1. Mapping from tile coordinates to tensor coordinates
    // 2. Load/store operations with configurable access patterns
    // 3. Support for various data types and layouts
    
    struct load_store_traits {
        static constexpr index_t vector_size = /* computed based on alignment */;
        static constexpr bool is_coalesced = /* based on access pattern */;
    };
};

CONCEPT EXPLANATION:
-------------------
A tile window is a "view" into tensor memory that:
1. Maps thread/warp coordinates to tensor positions
2. Handles data type conversions and vectorization
3. Optimizes memory access patterns
4. Provides high-level load/store operations
"""


@dataclass
class TileWindowTutorial:
    """
    Enhanced tile window implementation with tutorial features.
    
    This class demonstrates how tile windows work in CK, with visualizations
    and step-by-step execution traces.
    """
    
    # Core components
    bottom_tensor_view: TensorView
    window_lengths: List[int]
    tile_distribution: TileDistribution
    
    # Tutorial features
    verbose: bool = True
    trace_memory_access: bool = True
    visualize_access: bool = True
    
    # Memory access statistics
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize and validate tile window configuration."""
        if self.verbose:
            print("=== Tile Window Configuration ===")
            print(f"Window shape: {self.window_lengths}")
            print(f"Tensor shape: {self.bottom_tensor_view.get_tensor_shape()}")
            print(f"Data type: {self.bottom_tensor_view.dtype}")
            print()
            self._show_cpp_equivalent()
    
    def _show_cpp_equivalent(self):
        """Display equivalent C++ code."""
        print("C++ Equivalent:")
        print("```cpp")
        print("// Define tile window types")
        print(f"using WindowLengths = sequence<{', '.join(map(str, self.window_lengths))}>;")
        print("using TileWindow = tile_window_with_static_distribution<")
        print("    TensorView,      // Bottom tensor view")
        print("    WindowLengths,   // Window dimensions")
        print("    TileDistribution,// Thread distribution")
        print("    1>;              // Number of coordinates")
        print()
        print("// Create tile window")
        print("TileWindow window{tensor_view, distribution};")
        print("```")
        print()
    
    def get_window_shape(self) -> List[int]:
        """Get the shape of the tile window."""
        return self.window_lengths
    
    def calculate_vector_size(self) -> int:
        """
        Calculate optimal vector size for memory operations.
        
        C++ REFERENCE:
        ```cpp
        static constexpr auto get_vector_dim_y_scalar_per_vector() {
            // Find dimension with stride 1 for vectorization
            for(index_t i = 0; i < NDimY; ++i) {
                if(ys_vector_strides[i] == 1 && ys_vector_lengths[i] > ScalarPerVector_) {
                    ScalarPerVector_ = ys_vector_lengths[i];
                    VectorDimY_ = i;
                }
            }
        }
        ```
        """
        # Simplified vector size calculation
        # In practice, this depends on:
        # 1. Data type size
        # 2. Memory alignment
        # 3. Hardware capabilities
        
        dtype_size = np.dtype(self.bottom_tensor_view.dtype).itemsize
        max_vector_bytes = 16  # 128-bit vectors
        vector_size = min(max_vector_bytes // dtype_size, 4)
        
        if self.verbose:
            print(f"Calculated vector size: {vector_size} elements")
            print(f"(Based on {dtype_size}-byte elements, {max_vector_bytes}-byte vectors)")
        
        return vector_size
    
    def load_tile(self, tile_offset: List[int], 
                  thread_idx: Optional[int] = None) -> 'DistributedTensor':
        """
        Load a tile of data from global memory.
        
        This demonstrates the tile loading process with detailed tracing.
        
        Args:
            tile_offset: Offset of tile in tensor
            thread_idx: Thread index (for demonstration)
            
        Returns:
            Distributed tensor containing tile data
        """
        if self.verbose:
            print(f"\n=== Loading Tile at offset {tile_offset} ===")
        
        # Calculate global coordinates
        access_pattern = []
        
        # Simulate thread-wise loading
        num_threads = 32  # Warp size
        thread_idx = thread_idx or 0
        
        # Each thread loads a portion of the tile
        elements_per_thread = np.prod(self.window_lengths) // num_threads
        
        if self.trace_memory_access:
            print(f"Thread {thread_idx} loading {elements_per_thread} elements")
        
        # Record access pattern for visualization
        for elem in range(elements_per_thread):
            # Calculate which element this thread accesses
            linear_idx = thread_idx * elements_per_thread + elem
            
            # Convert to multi-dimensional index
            tile_coord = self._linear_to_coord(linear_idx, self.window_lengths)
            global_coord = [tile_offset[i] + tile_coord[i] 
                          for i in range(len(tile_offset))]
            
            access_pattern.append({
                'thread': thread_idx,
                'element': elem,
                'tile_coord': tile_coord,
                'global_coord': global_coord,
                'coalesced': self._check_coalescing(thread_idx, elem)
            })
        
        self.access_log.extend(access_pattern)
        
        # Create distributed tensor (simplified)
        data = np.zeros(self.window_lengths, dtype=self.bottom_tensor_view.dtype)
        dist_tensor = StaticDistributedTensor(
            data=data,
            distribution=self.tile_distribution
        )
        
        if self.verbose:
            print(f"Loaded tile with shape {self.window_lengths}")
            coalesced_count = sum(1 for a in access_pattern if a['coalesced'])
            print(f"Coalesced accesses: {coalesced_count}/{len(access_pattern)}")
        
        # Show C++ pattern
        if self.verbose:
            print("\nC++ Load Pattern:")
            print("```cpp")
            print("// Load tile using tile window")
            print("auto tile = make_static_distributed_tensor<TileShape>()")
            print("tile_window.load(tile, tensor_coordinate);")
            print("```")
        
        return dist_tensor
    
    def store_tile(self, tile_data: 'DistributedTensor', 
                   tile_offset: List[int],
                   thread_idx: Optional[int] = None):
        """
        Store a tile of data to global memory.
        
        Args:
            tile_data: Distributed tensor to store
            tile_offset: Offset in global tensor
            thread_idx: Thread index (for demonstration)
        """
        if self.verbose:
            print(f"\n=== Storing Tile at offset {tile_offset} ===")
        
        # Similar to load, but in reverse
        store_pattern = []
        num_threads = 32
        thread_idx = thread_idx or 0
        elements_per_thread = np.prod(self.window_lengths) // num_threads
        
        for elem in range(elements_per_thread):
            linear_idx = thread_idx * elements_per_thread + elem
            tile_coord = self._linear_to_coord(linear_idx, self.window_lengths)
            global_coord = [tile_offset[i] + tile_coord[i] 
                          for i in range(len(tile_offset))]
            
            store_pattern.append({
                'thread': thread_idx,
                'element': elem,
                'tile_coord': tile_coord,
                'global_coord': global_coord,
                'coalesced': self._check_coalescing(thread_idx, elem)
            })
        
        if self.verbose:
            print(f"Stored tile with shape {self.window_lengths}")
            print("\nC++ Store Pattern:")
            print("```cpp")
            print("// Store tile using tile window")
            print("tile_window.store(tile, tensor_coordinate);")
            print("__syncthreads(); // Ensure all threads complete")
            print("```")
    
    def _linear_to_coord(self, linear_idx: int, shape: List[int]) -> List[int]:
        """Convert linear index to multi-dimensional coordinate."""
        coord = []
        for dim in reversed(shape):
            coord.append(linear_idx % dim)
            linear_idx //= dim
        return list(reversed(coord))
    
    def _check_coalescing(self, thread_idx: int, element_idx: int) -> bool:
        """Check if memory access is coalesced."""
        # Simplified check: consecutive threads access consecutive memory
        return element_idx == 0  # First element per thread is coalesced
    
    def visualize_access_pattern(self, max_threads: int = 8):
        """
        Visualize memory access pattern for tile loading.
        """
        if not self.access_log:
            print("No access log available. Run load_tile() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Thread assignment in tile
        ax1.set_title('Thread Assignment in Tile', fontsize=14)
        tile_m, tile_n = self.window_lengths[:2]  # Assume 2D for visualization
        ax1.set_xlim(0, tile_n)
        ax1.set_ylim(0, tile_m)
        ax1.set_xlabel('Tile N')
        ax1.set_ylabel('Tile M')
        ax1.invert_yaxis()
        
        # Color by thread
        thread_colors = plt.cm.tab10(np.arange(max_threads) / max_threads)
        
        for access in self.access_log[:tile_m * tile_n]:
            if access['thread'] < max_threads:
                row, col = access['tile_coord'][:2]
                color = thread_colors[access['thread']]
                
                rect = Rectangle((col, row), 1, 1,
                               linewidth=1, edgecolor='black',
                               facecolor=color, alpha=0.7)
                ax1.add_patch(rect)
                ax1.text(col + 0.5, row + 0.5, f"T{access['thread']}",
                        ha='center', va='center', fontsize=8)
        
        # Right plot: Memory access pattern
        ax2.set_title('Memory Access Pattern\n(Green=Coalesced, Red=Uncoalesced)', 
                     fontsize=14)
        ax2.set_xlabel('Memory Address (relative)')
        ax2.set_ylabel('Time Step')
        
        # Group accesses by time step
        time_steps = {}
        for i, access in enumerate(self.access_log[:64]):  # First 64 accesses
            time_step = i // 32  # 32 threads per warp
            if time_step not in time_steps:
                time_steps[time_step] = []
            time_steps[time_step].append(access)
        
        # Plot access pattern
        for t, accesses in time_steps.items():
            for access in accesses:
                if access['thread'] < max_threads:
                    # Calculate memory address (simplified)
                    addr = access['global_coord'][0] * tile_n + access['global_coord'][1]
                    color = 'green' if access['coalesced'] else 'red'
                    ax2.scatter(addr, t, c=color, s=50, alpha=0.7)
                    ax2.annotate(f"T{access['thread']}", (addr, t),
                               fontsize=6, ha='center')
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# SECTION 2: Memory Access Patterns and Optimization
# =============================================================================

class MemoryAccessPatterns:
    """
    Demonstrate various memory access patterns and their optimization.
    """
    
    @staticmethod
    def demonstrate_coalescing():
        """
        Demonstrate coalesced vs uncoalesced memory access.
        
        C++ REFERENCE:
        Coalesced access is critical for GPU performance. CK ensures
        coalescing through careful tile and thread mapping.
        """
        print("=== Memory Coalescing Demo ===")
        print("Coalesced access: Adjacent threads access adjacent memory")
        print("Uncoalesced access: Threads access scattered memory locations")
        print()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Coalesced access pattern
        ax1.set_title('Coalesced Memory Access (Optimal)', fontsize=14)
        ax1.set_xlim(-1, 33)
        ax1.set_ylim(-1, 5)
        ax1.set_xlabel('Memory Address')
        ax1.set_ylabel('Access')
        
        # Draw memory blocks
        for i in range(32):
            rect = Rectangle((i, 0), 1, 3,
                           linewidth=1, edgecolor='black',
                           facecolor='lightblue', alpha=0.5)
            ax1.add_patch(rect)
            ax1.text(i + 0.5, 1.5, f'M{i}', ha='center', va='center', fontsize=6)
        
        # Draw thread accesses
        for tid in range(8):  # Show first 8 threads
            # Thread accesses consecutive address
            arrow = Arrow(tid + 0.5, 4, 0, -0.8, width=0.3,
                         color=plt.cm.viridis(tid/8))
            ax1.add_patch(arrow)
            ax1.text(tid + 0.5, 4.2, f'T{tid}', ha='center', va='center',
                    fontsize=8, color=plt.cm.viridis(tid/8))
        
        ax1.text(16, 4, 'One memory transaction\nfor all threads!',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        # Uncoalesced access pattern
        ax2.set_title('Uncoalesced Memory Access (Poor Performance)', fontsize=14)
        ax2.set_xlim(-1, 33)
        ax2.set_ylim(-1, 5)
        ax2.set_xlabel('Memory Address')
        ax2.set_ylabel('Access')
        
        # Draw memory blocks
        for i in range(32):
            rect = Rectangle((i, 0), 1, 3,
                           linewidth=1, edgecolor='black',
                           facecolor='lightcoral', alpha=0.5)
            ax2.add_patch(rect)
            ax2.text(i + 0.5, 1.5, f'M{i}', ha='center', va='center', fontsize=6)
        
        # Draw thread accesses (strided)
        stride = 4
        for tid in range(8):
            # Thread accesses strided address
            addr = tid * stride
            if addr < 32:
                arrow = Arrow(addr + 0.5, 4, 0, -0.8, width=0.3,
                             color=plt.cm.plasma(tid/8))
                ax2.add_patch(arrow)
                ax2.text(addr + 0.5, 4.2, f'T{tid}', ha='center', va='center',
                        fontsize=8, color=plt.cm.plasma(tid/8))
        
        ax2.text(16, 4, 'Multiple memory transactions\nneeded!',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        print("\nC++ Optimization Pattern:")
        print("```cpp")
        print("// Ensure coalesced access through proper indexing")
        print("template <typename TileWindow>")
        print("CK_TILE_DEVICE void load_with_coalescing(TileWindow& window) {")
        print("    // Good: consecutive threads access consecutive elements")
        print("    const index_t tid = threadIdx.x;")
        print("    const index_t coalesced_offset = tid;")
        print("    ")
        print("    // Bad: strided access")
        print("    // const index_t strided_offset = tid * stride;")
        print("}")
        print("```")
    
    @staticmethod
    def demonstrate_vectorization():
        """
        Demonstrate vectorized memory access.
        
        C++ REFERENCE:
        CK uses vectorized loads/stores to maximize memory bandwidth.
        """
        print("\n=== Vectorized Memory Access ===")
        print("Loading multiple elements per thread in one instruction")
        print()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scalar access
        ax1.set_title('Scalar Access (1 element/thread)', fontsize=14)
        ax1.set_xlim(0, 16)
        ax1.set_ylim(0, 4)
        ax1.set_xlabel('Memory')
        ax1.set_ylabel('Threads')
        
        # Show scalar loads
        for tid in range(4):
            for elem in range(1):
                rect = Rectangle((tid + elem, tid), 1, 0.8,
                               linewidth=1, edgecolor='black',
                               facecolor=plt.cm.Blues(0.5), alpha=0.7)
                ax1.add_patch(rect)
                ax1.text(tid + elem + 0.5, tid + 0.4, 
                        f'T{tid}', ha='center', va='center', fontsize=8)
        
        ax1.text(8, 2, '4 load instructions\nfor 4 elements',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Vector access
        ax2.set_title('Vector Access (4 elements/thread)', fontsize=14)
        ax2.set_xlim(0, 16)
        ax2.set_ylim(0, 4)
        ax2.set_xlabel('Memory')
        ax2.set_ylabel('Threads')
        
        # Show vector loads
        vector_size = 4
        for tid in range(4):
            rect = Rectangle((tid * vector_size, tid), vector_size, 0.8,
                           linewidth=2, edgecolor='black',
                           facecolor=plt.cm.Greens(0.5), alpha=0.7)
            ax2.add_patch(rect)
            
            # Show individual elements
            for v in range(vector_size):
                ax2.text(tid * vector_size + v + 0.5, tid + 0.4,
                        f'T{tid}', ha='center', va='center', fontsize=6)
            
            # Vector load annotation
            ax2.annotate('', xy=(tid * vector_size + vector_size, tid + 0.9),
                        xytext=(tid * vector_size, tid + 0.9),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax2.text(tid * vector_size + vector_size/2, tid + 1.1,
                    f'float4', ha='center', va='center', fontsize=8,
                    color='red', fontweight='bold')
        
        ax2.text(8, 2, '4 vector instructions\nfor 16 elements\n(4x speedup!)',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        plt.tight_layout()
        plt.show()
        
        print("\nC++ Vector Load Pattern:")
        print("```cpp")
        print("// Vectorized load using float4")
        print("template <typename T>")
        print("CK_TILE_DEVICE void load_vectorized(const T* src, T* dst) {")
        print("    // Scalar load (slow)")
        print("    // for(int i = 0; i < 4; ++i)")
        print("    //     dst[i] = src[i];")
        print("    ")
        print("    // Vector load (fast)")
        print("    using vec_t = typename vector_type<T, 4>::type;")
        print("    *reinterpret_cast<vec_t*>(dst) = ")
        print("        *reinterpret_cast<const vec_t*>(src);")
        print("}")
        print("```")


# =============================================================================
# SECTION 3: Advanced Tile Window Features
# =============================================================================

class AdvancedTileWindowFeatures:
    """
    Advanced features including boundary handling, padding, and data layout.
    """
    
    @staticmethod
    def demonstrate_boundary_handling():
        """
        Show how tile windows handle tensor boundaries.
        """
        print("=== Boundary Handling in Tile Windows ===")
        print("Tiles at tensor edges may need special handling")
        print()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Full tensor with tiles
        ax = axes[0, 0]
        ax.set_title('Tensor with Tile Grid', fontsize=12)
        tensor_m, tensor_n = 100, 100
        tile_m, tile_n = 32, 32
        
        ax.set_xlim(0, tensor_n)
        ax.set_ylim(0, tensor_m)
        ax.invert_yaxis()
        
        # Draw tiles
        for i in range(0, tensor_m, tile_m):
            for j in range(0, tensor_n, tile_n):
                # Check if tile extends beyond boundary
                actual_m = min(tile_m, tensor_m - i)
                actual_n = min(tile_n, tensor_n - j)
                
                if actual_m == tile_m and actual_n == tile_n:
                    color = 'lightblue'
                    label = 'Full'
                else:
                    color = 'lightcoral'
                    label = 'Partial'
                
                rect = Rectangle((j, i), actual_n, actual_m,
                               linewidth=1, edgecolor='black',
                               facecolor=color, alpha=0.5)
                ax.add_patch(rect)
                
                if i < tile_m and j < tile_n:
                    ax.text(j + actual_n/2, i + actual_m/2, label,
                           ha='center', va='center', fontsize=8)
        
        # Boundary tile detail
        ax = axes[0, 1]
        ax.set_title('Boundary Tile Detail', fontsize=12)
        ax.set_xlim(0, tile_n)
        ax.set_ylim(0, tile_m)
        ax.invert_yaxis()
        
        # Show partial tile
        partial_m, partial_n = 20, 20  # Actual data
        
        # Valid region
        valid_rect = Rectangle((0, 0), partial_n, partial_m,
                             linewidth=2, edgecolor='green',
                             facecolor='lightgreen', alpha=0.5,
                             label='Valid Data')
        ax.add_patch(valid_rect)
        
        # Padding region
        pad_rect1 = Rectangle((partial_n, 0), tile_n - partial_n, tile_m,
                            linewidth=2, edgecolor='red',
                            facecolor='pink', alpha=0.5,
                            label='Padding')
        ax.add_patch(pad_rect1)
        
        pad_rect2 = Rectangle((0, partial_m), partial_n, tile_m - partial_m,
                            linewidth=2, edgecolor='red',
                            facecolor='pink', alpha=0.5)
        ax.add_patch(pad_rect2)
        
        ax.legend()
        
        # Padding strategies
        ax = axes[1, 0]
        ax.set_title('Padding Strategies', fontsize=12)
        ax.text(0.5, 0.8, 'Common Padding Strategies:', ha='center', fontsize=12,
               transform=ax.transAxes, fontweight='bold')
        
        strategies = [
            '1. Zero Padding: Fill with zeros',
            '2. Replication: Repeat edge values',
            '3. Reflection: Mirror at boundaries',
            '4. Masking: Use validity mask'
        ]
        
        for i, strategy in enumerate(strategies):
            ax.text(0.1, 0.6 - i*0.15, strategy, fontsize=10,
                   transform=ax.transAxes)
        
        ax.axis('off')
        
        # C++ code example
        ax = axes[1, 1]
        ax.set_title('C++ Boundary Handling', fontsize=12)
        ax.text(0.05, 0.95, 
                'template <bool PadM, bool PadN>\n'
                'struct tile_window_with_padding {\n'
                '  CK_TILE_DEVICE auto load(coord) {\n'
                '    if constexpr(PadM || PadN) {\n'
                '      // Check boundaries\n'
                '      bool in_bounds = \n'
                '        coord.m < tensor_m &&\n'
                '        coord.n < tensor_n;\n'
                '      \n'
                '      return in_bounds ? \n'
                '        tensor[coord] : T{0};\n'
                '    } else {\n'
                '      return tensor[coord];\n'
                '    }\n'
                '  }\n'
                '};',
                transform=ax.transAxes, fontsize=8,
                fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def demonstrate_data_layout_impact():
        """
        Show impact of data layout on tile window performance.
        """
        print("\n=== Data Layout Impact on Tile Windows ===")
        print("Row-major vs Column-major layout affects access patterns")
        print()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Row-major layout
        ax = axes[0, 0]
        ax.set_title('Row-Major Layout (C-style)', fontsize=12)
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 5)
        ax.invert_yaxis()
        
        # Draw matrix elements with memory order
        memory_idx = 0
        for i in range(4):
            for j in range(8):
                rect = Rectangle((j, i), 1, 1,
                               linewidth=1, edgecolor='black',
                               facecolor='lightblue', alpha=0.5)
                ax.add_patch(rect)
                ax.text(j + 0.5, i + 0.5, f'{memory_idx}',
                       ha='center', va='center', fontsize=8)
                memory_idx += 1
        
        # Show access pattern for row
        for j in range(8):
            if j < 4:  # Highlight first 4
                arrow = Arrow(j + 0.5, -0.5, 0, 0.4, width=0.3,
                            color='green')
                ax.add_patch(arrow)
        
        ax.text(4, -0.7, 'Accessing row: Sequential memory',
               ha='center', fontsize=10, color='green')
        
        # Column-major layout
        ax = axes[0, 1]
        ax.set_title('Column-Major Layout (Fortran-style)', fontsize=12)
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 5)
        ax.invert_yaxis()
        
        # Draw matrix elements with memory order
        memory_idx = 0
        for j in range(8):
            for i in range(4):
                rect = Rectangle((j, i), 1, 1,
                               linewidth=1, edgecolor='black',
                               facecolor='lightcoral', alpha=0.5)
                ax.add_patch(rect)
                ax.text(j + 0.5, i + 0.5, f'{memory_idx}',
                       ha='center', va='center', fontsize=8)
                memory_idx += 1
        
        # Show access pattern for row
        for j in range(8):
            if j < 4:  # Highlight first 4
                arrow = Arrow(j + 0.5, -0.5, 0, 0.4, width=0.3,
                            color='red')
                ax.add_patch(arrow)
        
        ax.text(4, -0.7, 'Accessing row: Strided memory!',
               ha='center', fontsize=10, color='red')
        
        # Performance comparison
        ax = axes[1, 0]
        ax.set_title('Performance Impact', fontsize=12)
        
        layouts = ['Row-Major\n(row access)', 'Row-Major\n(col access)',
                  'Col-Major\n(row access)', 'Col-Major\n(col access)']
        performance = [100, 25, 25, 100]  # Relative performance
        colors = ['green', 'red', 'red', 'green']
        
        bars = ax.bar(layouts, performance, color=colors, alpha=0.7)
        ax.set_ylabel('Relative Performance (%)')
        ax.set_ylim(0, 120)
        
        for bar, perf in zip(bars, performance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{perf}%', ha='center', va='bottom')
        
        # Layout selection guide
        ax = axes[1, 1]
        ax.set_title('Layout Selection Guide', fontsize=12)
        ax.text(0.05, 0.9, 'Choose layout based on access pattern:',
               transform=ax.transAxes, fontsize=11, fontweight='bold')
        
        guide_text = """
Row-Major (CK default):
• Use when accessing rows frequently
• Natural for C/C++ arrays
• Good for GEMM with NN layout

Column-Major:
• Use when accessing columns frequently  
• Natural for Fortran/BLAS
• Good for GEMM with TN layout

CK Solution:
• Template on layout
• Optimize tile access accordingly
• Use transposition when needed
"""
        
        ax.text(0.05, 0.1, guide_text, transform=ax.transAxes,
               fontsize=9, fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# SECTION 4: Complete Tile Window Examples
# =============================================================================

class TileWindowExamples:
    """
    Complete examples showing tile window usage in real scenarios.
    """
    
    @staticmethod
    def matrix_multiplication_tiling():
        """
        Demonstrate tile window usage in matrix multiplication.
        """
        print("=== Matrix Multiplication with Tile Windows ===")
        print("Shows how A, B, and C matrices are accessed via tile windows")
        print()
        
        # Create figure with subplots for A, B, C matrices
        fig = plt.figure(figsize=(15, 10))
        
        # Matrix dimensions
        M, N, K = 128, 128, 64
        tile_m, tile_n, tile_k = 32, 32, 16
        
        # Create subplots
        ax_a = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=1)
        ax_b = plt.subplot2grid((3, 3), (0, 1), rowspan=1, colspan=2)
        ax_c = plt.subplot2grid((3, 3), (1, 1), rowspan=2, colspan=2)
        ax_code = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=3)
        
        # Matrix A (M x K)
        ax_a.set_title('Matrix A (M×K)', fontsize=12)
        ax_a.set_xlim(0, K)
        ax_a.set_ylim(0, M)
        ax_a.invert_yaxis()
        ax_a.set_xlabel('K')
        ax_a.set_ylabel('M')
        
        # Draw A tiles
        for i in range(0, M, tile_m):
            for j in range(0, K, tile_k):
                rect = Rectangle((j, i), tile_k, tile_m,
                               linewidth=1, edgecolor='blue',
                               facecolor='lightblue', alpha=0.3)
                ax_a.add_patch(rect)
        
        # Highlight current tile
        current_a = Rectangle((0, 0), tile_k, tile_m,
                            linewidth=3, edgecolor='blue',
                            facecolor='blue', alpha=0.5)
        ax_a.add_patch(current_a)
        ax_a.text(tile_k/2, tile_m/2, 'A-Tile',
                 ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Matrix B (K x N)
        ax_b.set_title('Matrix B (K×N)', fontsize=12)
        ax_b.set_xlim(0, N)
        ax_b.set_ylim(0, K)
        ax_b.invert_yaxis()
        ax_b.set_xlabel('N')
        ax_b.set_ylabel('K')
        
        # Draw B tiles
        for i in range(0, K, tile_k):
            for j in range(0, N, tile_n):
                rect = Rectangle((j, i), tile_n, tile_k,
                               linewidth=1, edgecolor='green',
                               facecolor='lightgreen', alpha=0.3)
                ax_b.add_patch(rect)
        
        # Highlight current tile
        current_b = Rectangle((0, 0), tile_n, tile_k,
                            linewidth=3, edgecolor='green',
                            facecolor='green', alpha=0.5)
        ax_b.add_patch(current_b)
        ax_b.text(tile_n/2, tile_k/2, 'B-Tile',
                 ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Matrix C (M x N)
        ax_c.set_title('Matrix C (M×N) = A × B', fontsize=12)
        ax_c.set_xlim(0, N)
        ax_c.set_ylim(0, M)
        ax_c.invert_yaxis()
        ax_c.set_xlabel('N')
        ax_c.set_ylabel('M')
        
        # Draw C tiles
        for i in range(0, M, tile_m):
            for j in range(0, N, tile_n):
                rect = Rectangle((j, i), tile_n, tile_m,
                               linewidth=1, edgecolor='red',
                               facecolor='lightcoral', alpha=0.3)
                ax_c.add_patch(rect)
        
        # Highlight current tile
        current_c = Rectangle((0, 0), tile_n, tile_m,
                            linewidth=3, edgecolor='red',
                            facecolor='red', alpha=0.5)
        ax_c.add_patch(current_c)
        ax_c.text(tile_n/2, tile_m/2, 'C-Tile\n(Output)',
                 ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Show computation arrow
        ax_c.annotate('', xy=(tile_n/2, -5), xytext=(tile_n/2, -15),
                     arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
        ax_c.text(tile_n/2, -20, 'Accumulate', ha='center', fontsize=10,
                 color='purple', fontweight='bold')
        
        # Code example
        ax_code.axis('off')
        code_text = """// GEMM Tile Window Usage (Simplified)
template <typename ALayout, typename BLayout, typename CLayout>
__global__ void gemm_kernel() {
    // Create tile windows for each matrix
    auto a_window = make_tile_window<TileM, TileK>(A_view, a_distribution);
    auto b_window = make_tile_window<TileK, TileN>(B_view, b_distribution);
    auto c_window = make_tile_window<TileM, TileN>(C_view, c_distribution);
    
    // Allocate register tiles
    auto a_tile = make_static_distributed_tensor<AType, TileM, TileK>();
    auto b_tile = make_static_distributed_tensor<BType, TileK, TileN>();
    auto c_tile = make_static_distributed_tensor<CType, TileM, TileN>();
    
    // Initialize C tile
    clear(c_tile);
    
    // Main GEMM loop over K dimension
    for(index_t k = 0; k < K; k += TileK) {
        // Load tiles from global memory
        a_window.load(a_tile, make_coord(tile_m, k));
        b_window.load(b_tile, make_coord(k, tile_n));
        
        // Perform tile multiplication
        tile_gemm(a_tile, b_tile, c_tile);
    }
    
    // Store result back to global memory
    c_window.store(c_tile, make_coord(tile_m, tile_n));
}"""
        
        ax_code.text(0.05, 0.95, code_text, transform=ax_code.transAxes,
                    fontsize=9, fontfamily='monospace', va='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def demonstrate_tile_window_lifecycle():
        """
        Show the complete lifecycle of tile window operations.
        """
        print("\n=== Tile Window Operation Lifecycle ===")
        print("Step-by-step visualization of tile window operations")
        print()
        
        # Create animation showing tile window operations
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        steps = [
            ("1. Create Window", "Initialize tile window\nwith tensor view"),
            ("2. Calculate Coords", "Map thread IDs to\ntensor coordinates"),
            ("3. Load Tile", "Load data from\nglobal to registers"),
            ("4. Process Tile", "Perform computations\non tile data"),
            ("5. Store Results", "Write results back\nto global memory"),
            ("6. Synchronize", "Ensure all threads\ncomplete operation")
        ]
        
        for idx, (title, desc) in enumerate(steps):
            ax = axes[idx]
            ax.set_title(f'Step {idx+1}: {title.split(".")[1].strip()}', 
                        fontsize=12)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Draw illustration for each step
            if idx == 0:  # Create window
                rect = Rectangle((2, 3), 6, 4, linewidth=2, 
                               edgecolor='blue', facecolor='lightblue',
                               alpha=0.5)
                ax.add_patch(rect)
                ax.text(5, 5, 'Tile\nWindow', ha='center', va='center',
                       fontsize=14, fontweight='bold')
                
            elif idx == 1:  # Calculate coords
                # Draw coordinate mapping
                for i in range(3):
                    ax.arrow(1 + i*3, 8, 0, -4, head_width=0.3,
                           head_length=0.3, fc='green', ec='green')
                    ax.text(1 + i*3, 8.5, f'T{i}', ha='center', fontsize=10)
                    ax.text(1 + i*3, 3, f'({i},0)', ha='center', fontsize=10)
                    
            elif idx == 2:  # Load tile
                # Memory to register
                mem_rect = Rectangle((1, 7), 8, 2, linewidth=1,
                                   edgecolor='black', facecolor='gray',
                                   alpha=0.3)
                ax.add_patch(mem_rect)
                ax.text(5, 8, 'Global Memory', ha='center', fontsize=10)
                
                reg_rect = Rectangle((3, 2), 4, 2, linewidth=1,
                                   edgecolor='black', facecolor='yellow',
                                   alpha=0.5)
                ax.add_patch(reg_rect)
                ax.text(5, 3, 'Registers', ha='center', fontsize=10)
                
                ax.arrow(5, 6.8, 0, -2.5, head_width=0.5,
                       head_length=0.3, fc='blue', ec='blue', lw=2)
                
            elif idx == 3:  # Process
                # Show computation
                ax.text(5, 8, 'A × B', ha='center', fontsize=20)
                ax.text(5, 6, '+', ha='center', fontsize=20)
                ax.text(5, 4, 'C', ha='center', fontsize=20)
                ax.text(5, 2, '↓', ha='center', fontsize=20)
                ax.text(5, 0.5, "C'", ha='center', fontsize=20)
                
            elif idx == 4:  # Store
                # Register to memory
                reg_rect = Rectangle((3, 7), 4, 2, linewidth=1,
                                   edgecolor='black', facecolor='yellow',
                                   alpha=0.5)
                ax.add_patch(reg_rect)
                ax.text(5, 8, 'Registers', ha='center', fontsize=10)
                
                mem_rect = Rectangle((1, 2), 8, 2, linewidth=1,
                                   edgecolor='black', facecolor='gray',
                                   alpha=0.3)
                ax.add_patch(mem_rect)
                ax.text(5, 3, 'Global Memory', ha='center', fontsize=10)
                
                ax.arrow(5, 6.8, 0, -2.5, head_width=0.5,
                       head_length=0.3, fc='red', ec='red', lw=2)
                
            elif idx == 5:  # Sync
                # Barrier
                ax.plot([1, 9], [5, 5], 'k-', lw=4)
                ax.text(5, 6, '__syncthreads()', ha='center', 
                       fontsize=12, fontweight='bold')
                ax.text(5, 3, 'All threads wait here', ha='center', 
                       fontsize=10, style='italic')
            
            # Add description
            ax.text(5, -0.5, desc, ha='center', va='top',
                   fontsize=9, style='italic', wrap=True)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# SECTION 5: Interactive Tutorial and Exercises
# =============================================================================

def run_tile_window_tutorial():
    """
    Run comprehensive tile window tutorial with examples and exercises.
    """
    print("=" * 60)
    print("CK TILE WINDOW INTERACTIVE TUTORIAL")
    print("=" * 60)
    print()
    
    # Introduction
    print("### Introduction to Tile Windows ###")
    print("Tile windows are the primary abstraction for accessing tensor")
    print("data in CK. They provide:")
    print("- Efficient mapping from thread IDs to tensor coordinates")
    print("- Optimized load/store operations")
    print("- Support for various data layouts and types")
    print()
    
    # Example 1: Basic tile window
    print("### Example 1: Creating a Basic Tile Window ###")
    
    from .tensor_view import make_naive_tensor_view
    from .tensor_descriptor import make_naive_tensor_descriptor
    from .tile_distribution import TileDistribution
    
    # Create tensor view (simplified)
    tensor_shape = [64, 64]
    tensor_view = make_naive_tensor_view(
        data=np.zeros(tensor_shape, dtype=np.float32),
        shape=tensor_shape,
        strides=[64, 1]  # Row-major
    )
    
    # Create distribution (simplified)
    adaptor = TensorAdaptor(
        top_dimensions=[32, 16, 16],  # [threads, tile_y, tile_x]
        bottom_dimensions=[64, 64]
    )
    
    descriptor = make_naive_tensor_descriptor([16, 16], [16, 1])
    
    encoding = TileDistributionEncoding(
        rs_lengths=[1],
        hs_lengths=[64, 64],
        ps_to_rs_adaptor=None,
        hs_to_ys_adaptor=None
    )
    
    distribution = TileDistribution(
        ps_ys_to_xs_adaptor=adaptor,
        ys_to_d_descriptor=descriptor,
        encoding=encoding
    )
    
    # Create tile window
    window = TileWindowTutorial(
        bottom_tensor_view=tensor_view,
        window_lengths=[16, 16],
        tile_distribution=distribution,
        verbose=True
    )
    
    # Example operations
    print("\n### Example Operations ###")
    
    # Load a tile
    print("\n1. Loading a tile:")
    tile_data = window.load_tile(tile_offset=[0, 0], thread_idx=0)
    
    # Visualize access pattern
    print("\n2. Visualizing access pattern:")
    window.visualize_access_pattern(max_threads=4)
    
    # Memory access patterns
    print("\n### Memory Access Optimization ###")
    MemoryAccessPatterns.demonstrate_coalescing()
    MemoryAccessPatterns.demonstrate_vectorization()
    
    # Advanced features
    print("\n### Advanced Features ###")
    AdvancedTileWindowFeatures.demonstrate_boundary_handling()
    AdvancedTileWindowFeatures.demonstrate_data_layout_impact()
    
    # Complete examples
    print("\n### Complete Examples ###")
    TileWindowExamples.matrix_multiplication_tiling()
    TileWindowExamples.demonstrate_tile_window_lifecycle()
    
    # Summary
    print("\n### Tutorial Summary ###")
    print("Key concepts covered:")
    print("1. Tile windows map threads to tensor data efficiently")
    print("2. Coalesced access is critical for performance")
    print("3. Vectorization improves memory bandwidth utilization")
    print("4. Boundary handling requires special consideration")
    print("5. Data layout affects access patterns significantly")
    print()
    print("Next steps: Explore tile operations (load_tile, store_tile, etc.)")
    print()
    
    # Exercise suggestions
    print("### Exercises ###")
    print("1. Modify vector size and observe performance impact")
    print("2. Implement different boundary padding strategies")
    print("3. Create a tile window for 3D tensors")
    print("4. Experiment with different thread-to-data mappings")
    print()
    print("Happy tiling!")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    run_tile_window_tutorial()