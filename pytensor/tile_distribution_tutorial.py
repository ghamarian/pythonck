"""
Enhanced tutorial implementation of CK Tile Distribution with C++ code correspondence.

This module provides a comprehensive tutorial on tile distribution - a fundamental
concept in GPU programming for distributing tensor data across processing elements
(threads, warps, blocks). It includes detailed explanations, visualizations, and
corresponding C++ code snippets from the CK library.

Key Concepts:
1. Tile Distribution: How tensor data is partitioned across GPU compute units
2. Coordinate Systems: X (tensor), Y (tile), P (processing), R (replication)
3. Tensor Adaptors: Transformations between coordinate systems
4. Distribution Encoding: Strategies for mapping threads to data

Author: CK Tutorial
License: MIT
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

# Import base implementations
from .tensor_descriptor import TensorAdaptor, TensorDescriptor
from .tensor_coordinate import MultiIndex, TensorAdaptorCoordinate
from .tile_distribution_encoding import TileDistributionEncoding


# =============================================================================
# SECTION 1: Core Concepts and C++ Correspondence
# =============================================================================

"""
C++ REFERENCE:
--------------
From ck_tile/core/tensor/tile_distribution.hpp:

template <typename PsYs2XsAdaptor_,
          typename Ys2DDescriptor_,
          typename StaticTileDistributionEncoding_,
          typename TileDistributionDetail_>
struct tile_distribution
{
    // Coordinate system dimensions:
    static constexpr index_t NDimX = PsYs2XsAdaptor::get_num_of_bottom_dimension();
    static constexpr index_t NDimY = Ys2DDescriptor::get_num_of_top_dimension();
    static constexpr index_t NDimP = PsYs2XsAdaptor::get_num_of_top_dimension() - NDimY;
    static constexpr index_t NDimR = StaticTileDistributionEncoding_::NDimR;
};

CONCEPT EXPLANATION:
-------------------
The tile distribution manages 4 coordinate systems:

1. X-coordinates: Original tensor dimensions (e.g., for a matrix: rows, columns)
2. Y-coordinates: Tile dimensions (how the tensor is divided into tiles)
3. P-coordinates: Processing dimensions (thread, warp, block hierarchy)
4. R-coordinates: Replication dimensions (for redundant computation/storage)

The transformation flow is:
P,Y -> X (via PsYs2XsAdaptor): Maps processing units and tile coords to tensor position
Y -> D (via Ys2DDescriptor): Maps tile coords to linearized storage
"""


@dataclass
class TileDistributionTutorial:
    """
    Enhanced tile distribution with tutorial features and visualizations.
    
    This class extends the basic TileDistribution with educational features
    including step-by-step computation traces, visualizations, and C++ code
    correspondence.
    """
    
    ps_ys_to_xs_adaptor: TensorAdaptor
    ys_to_d_descriptor: TensorDescriptor
    encoding: TileDistributionEncoding
    
    # Tutorial-specific attributes
    verbose: bool = True
    trace_computations: bool = True
    
    def __post_init__(self):
        """Validate and display distribution configuration."""
        if self.verbose:
            print("=== Tile Distribution Configuration ===")
            print(f"X dimensions (tensor): {self.ndim_x}")
            print(f"Y dimensions (tile): {self.ndim_y}")
            print(f"P dimensions (processing): {self.ndim_p}")
            print(f"R dimensions (replication): {self.ndim_r}")
            print()
            self._show_cpp_equivalent()
    
    def _show_cpp_equivalent(self):
        """Display equivalent C++ code for this configuration."""
        print("C++ Equivalent:")
        print("```cpp")
        print("// Define tile distribution types")
        print(f"using PsYs2XsAdaptor = /* adaptor with {self.ndim_x}D bottom, {self.ndim_p + self.ndim_y}D top */;")
        print(f"using Ys2DDescriptor = /* descriptor with {self.ndim_y}D top */;")
        print(f"using DstrEncode = /* encoding with {self.ndim_r} replication dims */;")
        print()
        print("// Create tile distribution")
        print("tile_distribution<PsYs2XsAdaptor, Ys2DDescriptor, DstrEncode, Detail> distribution{")
        print("    ps_ys_to_xs_adaptor,")
        print("    ys_to_d_descriptor")
        print("};")
        print("```")
        print()
    
    @property
    def ndim_x(self) -> int:
        """Number of X (tensor) dimensions."""
        return self.ps_ys_to_xs_adaptor.get_num_of_bottom_dimension()
    
    @property
    def ndim_y(self) -> int:
        """Number of Y (tile) dimensions."""
        return self.ys_to_d_descriptor.get_num_of_top_dimension()
    
    @property
    def ndim_p(self) -> int:
        """Number of P (processing) dimensions."""
        return self.ps_ys_to_xs_adaptor.get_num_of_top_dimension() - self.ndim_y
    
    @property
    def ndim_r(self) -> int:
        """Number of R (replication) dimensions."""
        return self.encoding.ndim_r
    
    def get_partition_index(self, thread_id: Optional[int] = None, 
                          warp_id: Optional[int] = None) -> List[int]:
        """
        Get partition index for current processing element.
        
        C++ REFERENCE:
        ```cpp
        CK_TILE_HOST_DEVICE static auto _get_partition_index()
        {
            if constexpr(NDimP == 1)
                return make_tuple(get_lane_id());
            else if constexpr(NDimP == 2)
                return make_tuple(get_warp_id(), get_lane_id());
        }
        ```
        
        Args:
            thread_id: Thread ID within warp (0-31 for AMD GPUs)
            warp_id: Warp ID within block
            
        Returns:
            Partition index [P0, P1, ...]
        """
        if self.ndim_p == 1:
            # Thread-level distribution
            partition_idx = [thread_id or 0]
            if self.verbose:
                print(f"Thread-level distribution: P = [{partition_idx[0]}]")
        elif self.ndim_p == 2:
            # Warp-thread distribution
            partition_idx = [warp_id or 0, thread_id or 0]
            if self.verbose:
                print(f"Warp-thread distribution: P = [{partition_idx[0]}, {partition_idx[1]}]")
        else:
            partition_idx = [0] * self.ndim_p
            
        return partition_idx
    
    def calculate_tensor_position(self, partition_index: List[int], 
                                tile_index: List[int]) -> Tuple[List[int], Dict[str, Any]]:
        """
        Calculate tensor position X from partition P and tile Y indices.
        
        This demonstrates the full transformation: (P,Y) -> X
        
        Args:
            partition_index: Processing element index [P0, P1, ...]
            tile_index: Tile coordinate [Y0, Y1, ...]
            
        Returns:
            Tuple of (tensor_position, trace_info)
        """
        trace = {}
        
        # Step 1: Combine P and Y indices
        ps_ys_idx = partition_index + tile_index
        trace['ps_ys_combined'] = ps_ys_idx
        
        if self.verbose:
            print(f"\n=== Calculating Tensor Position ===")
            print(f"Input P (partition): {partition_index}")
            print(f"Input Y (tile): {tile_index}")
            print(f"Combined (P,Y): {ps_ys_idx}")
        
        # Step 2: Apply adaptor transformation
        coord = TensorAdaptorCoordinate(
            self.ps_ys_to_xs_adaptor,
            MultiIndex(len(ps_ys_idx), ps_ys_idx)
        )
        
        # Step 3: Get bottom (X) index
        x_index = coord.get_bottom_index()
        trace['x_index'] = x_index.data
        
        if self.verbose:
            print(f"Output X (tensor): {x_index.data}")
            print("\nC++ equivalent:")
            print("```cpp")
            print(f"array<index_t, {len(ps_ys_idx)}> ps_ys_idx{{{', '.join(map(str, ps_ys_idx))}}};")
            print("auto coord = make_tensor_adaptor_coordinate(ps_ys_to_xs_adaptor, ps_ys_idx);")
            print("auto x_idx = coord.get_bottom_index();")
            print("```")
        
        return x_index.data, trace
    
    def visualize_distribution_2d(self, tensor_shape: Tuple[int, int], 
                                 tile_shape: Tuple[int, int],
                                 highlight_thread: Optional[int] = None):
        """
        Visualize 2D tile distribution with thread mapping.
        
        Args:
            tensor_shape: (M, N) dimensions of tensor
            tile_shape: (TileM, TileN) dimensions of each tile
            highlight_thread: Thread ID to highlight (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate tile grid
        tiles_m = tensor_shape[0] // tile_shape[0]
        tiles_n = tensor_shape[1] // tile_shape[1]
        
        # Left plot: Tensor divided into tiles
        ax1.set_title('Tensor Divided into Tiles', fontsize=14)
        ax1.set_xlim(0, tensor_shape[1])
        ax1.set_ylim(0, tensor_shape[0])
        ax1.set_xlabel('N (columns)')
        ax1.set_ylabel('M (rows)')
        ax1.invert_yaxis()
        
        # Draw tile boundaries
        for i in range(tiles_m + 1):
            ax1.axhline(y=i * tile_shape[0], color='black', linewidth=2)
        for j in range(tiles_n + 1):
            ax1.axvline(x=j * tile_shape[1], color='black', linewidth=2)
        
        # Label tiles
        for i in range(tiles_m):
            for j in range(tiles_n):
                tile_id = i * tiles_n + j
                cx = j * tile_shape[1] + tile_shape[1] / 2
                cy = i * tile_shape[0] + tile_shape[0] / 2
                ax1.text(cx, cy, f'Tile {tile_id}', ha='center', va='center', fontsize=12)
        
        # Right plot: Thread assignment within a tile
        ax2.set_title('Thread Assignment Within Tile', fontsize=14)
        ax2.set_xlim(0, tile_shape[1])
        ax2.set_ylim(0, tile_shape[0])
        ax2.set_xlabel('Tile N')
        ax2.set_ylabel('Tile M')
        ax2.invert_yaxis()
        
        # Assuming warp-level tiling for visualization
        warp_size = 32  # AMD wavefront size
        threads_per_tile = tile_shape[0] * tile_shape[1] // warp_size
        
        # Color map for threads
        colors = plt.cm.tab20(np.linspace(0, 1, min(threads_per_tile, 20)))
        
        # Draw thread assignments
        if self.ndim_p == 1:  # Thread-level
            # Simple row-major thread assignment
            for idx in range(tile_shape[0] * tile_shape[1]):
                row = idx // tile_shape[1]
                col = idx % tile_shape[1]
                thread_id = idx % warp_size
                
                color = colors[thread_id % len(colors)]
                if highlight_thread is not None and thread_id == highlight_thread:
                    color = 'red'
                    
                rect = Rectangle((col, row), 1, 1, 
                               linewidth=1, edgecolor='gray',
                               facecolor=color, alpha=0.7)
                ax2.add_patch(rect)
                ax2.text(col + 0.5, row + 0.5, f'{thread_id}', 
                        ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_coordinate_transform(self):
        """
        Interactive demonstration of coordinate transformations.
        """
        print("\n=== Coordinate Transformation Demo ===")
        print("This demonstrates how processing coordinates map to tensor positions.")
        print()
        
        # Example 1: Simple 1D case
        print("Example 1: 1D Thread Distribution")
        print("-" * 40)
        print("Tensor: 1D array of 64 elements")
        print("Tile size: 16 elements")
        print("Threads: 16 per tile")
        print()
        
        # Show mapping for first few threads
        for thread_id in range(4):
            for tile_id in range(2):
                partition_idx = [thread_id]
                tile_idx = [tile_id]
                
                # Calculate position (simplified for demo)
                tensor_pos = tile_id * 16 + thread_id
                
                print(f"Thread {thread_id}, Tile {tile_id} -> Position {tensor_pos}")
        
        print("\nC++ Pattern:")
        print("```cpp")
        print("// Each thread processes one element per tile")
        print("index_t tid = get_thread_id();")
        print("index_t tile_offset = tile_id * tile_size;")
        print("index_t global_pos = tile_offset + tid;")
        print("```")


# =============================================================================
# SECTION 2: Common Distribution Patterns
# =============================================================================

class CommonDistributionPatterns:
    """
    Library of common tile distribution patterns used in GPU kernels.
    
    These patterns correspond to common use cases in CK library:
    1. Row-major thread mapping
    2. Column-major thread mapping  
    3. Warp-tile distribution
    4. Block-tile distribution
    """
    
    @staticmethod
    def create_gemm_distribution(m_tile: int, n_tile: int, k_tile: int,
                               m_warp: int = 2, n_warp: int = 2) -> TileDistributionTutorial:
        """
        Create a tile distribution for GEMM operations.
        
        This corresponds to the C++ GEMM implementations in CK.
        
        C++ REFERENCE:
        ```cpp
        // From example/03_gemm/gemm_basic.cpp
        constexpr ck_tile::index_t M_Tile = 256;
        constexpr ck_tile::index_t N_Tile = 256;
        constexpr ck_tile::index_t K_Tile = 64;
        
        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        ```
        
        Args:
            m_tile: M dimension tile size
            n_tile: N dimension tile size
            k_tile: K dimension tile size (for reduction)
            m_warp: Number of warps in M dimension
            n_warp: Number of warps in N dimension
            
        Returns:
            Configured tile distribution for GEMM
        """
        print("=== Creating GEMM Tile Distribution ===")
        print(f"Tile shape: {m_tile} x {n_tile} x {k_tile}")
        print(f"Warp grid: {m_warp} x {n_warp}")
        print()
        
        # For GEMM, we typically have:
        # - 2D processing (warp, thread)
        # - 2D tile (M, N)  
        # - 3D tensor (M, N, K) but K is reduction dimension
        
        # Create simplified adaptor for demo
        # In real implementation, this would use proper tensor descriptors
        from .tensor_descriptor import make_naive_tensor_descriptor
        
        # P,Y -> X mapping (simplified)
        ps_ys_to_xs_adaptor = TensorAdaptor(
            top_dimensions=[m_warp * n_warp, 32, m_tile // (m_warp * 32), n_tile // (n_warp * 32)],
            bottom_dimensions=[m_tile, n_tile]
        )
        
        # Y -> D mapping
        ys_to_d_descriptor = make_naive_tensor_descriptor(
            [m_tile // (m_warp * 32), n_tile // (n_warp * 32)],
            [n_tile // (n_warp * 32), 1]
        )
        
        # Simple encoding
        encoding = TileDistributionEncoding(
            rs_lengths=[1],  # No replication
            hs_lengths=[m_tile, n_tile],
            ps_to_rs_adaptor=None,
            hs_to_ys_adaptor=None
        )
        
        return TileDistributionTutorial(
            ps_ys_to_xs_adaptor=ps_ys_to_xs_adaptor,
            ys_to_d_descriptor=ys_to_d_descriptor,
            encoding=encoding,
            verbose=True
        )
    
    @staticmethod
    def visualize_gemm_tiling():
        """
        Visualize GEMM tiling strategy with annotations.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Constants from C++ example
        m_total, n_total, k_total = 1024, 1024, 256
        m_tile, n_tile, k_tile = 256, 256, 64
        m_warp, n_warp = 2, 2
        warp_size = 32
        
        # Plot 1: Overall GEMM tiling
        ax = axes[0, 0]
        ax.set_title('GEMM Tiling Overview', fontsize=14)
        ax.set_xlim(0, n_total)
        ax.set_ylim(0, m_total)
        ax.set_xlabel('N')
        ax.set_ylabel('M')
        ax.invert_yaxis()
        
        # Draw tiles
        for i in range(0, m_total, m_tile):
            for j in range(0, n_total, n_tile):
                rect = Rectangle((j, i), n_tile, m_tile,
                               linewidth=2, edgecolor='blue',
                               facecolor='lightblue', alpha=0.3)
                ax.add_patch(rect)
                
                # Label tile
                tile_m = i // m_tile
                tile_n = j // n_tile
                ax.text(j + n_tile/2, i + m_tile/2, 
                       f'Tile\n({tile_m},{tile_n})',
                       ha='center', va='center', fontsize=10)
        
        # Plot 2: Single tile with warp assignment
        ax = axes[0, 1]
        ax.set_title('Warp Assignment in Tile', fontsize=14)
        ax.set_xlim(0, n_tile)
        ax.set_ylim(0, m_tile)
        ax.set_xlabel('Tile N')
        ax.set_ylabel('Tile M')
        ax.invert_yaxis()
        
        # Draw warp regions
        warp_m_size = m_tile // m_warp
        warp_n_size = n_tile // n_warp
        warp_colors = ['red', 'green', 'blue', 'orange']
        
        for i in range(m_warp):
            for j in range(n_warp):
                warp_id = i * n_warp + j
                rect = Rectangle((j * warp_n_size, i * warp_m_size), 
                               warp_n_size, warp_m_size,
                               linewidth=2, edgecolor='black',
                               facecolor=warp_colors[warp_id], alpha=0.3)
                ax.add_patch(rect)
                ax.text(j * warp_n_size + warp_n_size/2, 
                       i * warp_m_size + warp_m_size/2,
                       f'Warp {warp_id}',
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Plot 3: Thread assignment within warp
        ax = axes[1, 0]
        ax.set_title('Thread Assignment in Warp Region', fontsize=14)
        warp_region_m = warp_m_size
        warp_region_n = warp_n_size
        ax.set_xlim(0, warp_region_n)
        ax.set_ylim(0, warp_region_m)
        ax.set_xlabel('Warp Region N')
        ax.set_ylabel('Warp Region M')
        ax.invert_yaxis()
        
        # Show thread mapping pattern
        thread_m = 8  # Threads cover 8 rows
        thread_n = 4  # Threads cover 4 columns (8x4 = 32 threads)
        
        for tid in range(warp_size):
            t_row = tid // thread_n
            t_col = tid % thread_n
            
            # Each thread covers multiple elements
            elem_per_thread_m = warp_region_m // thread_m
            elem_per_thread_n = warp_region_n // thread_n
            
            rect = Rectangle((t_col * elem_per_thread_n, t_row * elem_per_thread_m),
                           elem_per_thread_n, elem_per_thread_m,
                           linewidth=1, edgecolor='gray',
                           facecolor=plt.cm.viridis(tid/warp_size), alpha=0.6)
            ax.add_patch(rect)
            ax.text(t_col * elem_per_thread_n + elem_per_thread_n/2,
                   t_row * elem_per_thread_m + elem_per_thread_m/2,
                   f'T{tid}', ha='center', va='center', fontsize=8)
        
        # Plot 4: K-dimension tiling
        ax = axes[1, 1]
        ax.set_title('K-Dimension Tiling for Reduction', fontsize=14)
        ax.set_xlim(0, k_total)
        ax.set_ylim(0, 2)
        ax.set_xlabel('K')
        ax.set_yticks([])
        
        # Show K tiles
        for k in range(0, k_total, k_tile):
            rect = Rectangle((k, 0.5), k_tile, 1,
                           linewidth=2, edgecolor='purple',
                           facecolor='lavender', alpha=0.5)
            ax.add_patch(rect)
            ax.text(k + k_tile/2, 1, f'K-tile {k//k_tile}',
                   ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print corresponding C++ code
        print("\nCorresponding C++ Code Pattern:")
        print("```cpp")
        print("// GEMM kernel launch configuration")
        print("template <typename TilePartitioner>")
        print("__global__ void gemm_kernel(const float* A, const float* B, float* C) {")
        print("    // Get tile and thread indices")
        print("    const index_t tile_m = blockIdx.y;")
        print("    const index_t tile_n = blockIdx.x;")
        print("    const index_t warp_id = threadIdx.y;") 
        print("    const index_t lane_id = threadIdx.x;")
        print("    ")
        print("    // Calculate global position using tile distribution")
        print("    auto partition_idx = make_tuple(warp_id, lane_id);")
        print("    auto tile_idx = make_tuple(tile_m, tile_n);")
        print("    ")
        print("    // Use distribution to map to tensor coordinates")
        print("    auto global_coord = distribution.calculate_index(partition_idx, tile_idx);")
        print("}")
        print("```")


# =============================================================================
# SECTION 3: Advanced Topics and Optimizations
# =============================================================================

class AdvancedDistributionConcepts:
    """
    Advanced tile distribution concepts including:
    - Swizzling for bank conflict avoidance
    - Replication for improved memory access
    - Hierarchical tiling (thread -> warp -> block)
    """
    
    @staticmethod
    def demonstrate_swizzling():
        """
        Demonstrate tile swizzling for avoiding shared memory bank conflicts.
        
        C++ REFERENCE:
        Space filling curves and swizzling patterns are implemented in:
        ck_tile/core/algorithm/space_filling_curve.hpp
        """
        print("=== Tile Swizzling for Bank Conflict Avoidance ===")
        print()
        print("Swizzling reorders thread-to-data mapping to avoid bank conflicts")
        print("in shared memory. CK uses space-filling curves for this.")
        print()
        
        # Visualize linear vs swizzled access patterns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear pattern
        ax1.set_title('Linear Thread Mapping\n(Causes Bank Conflicts)')
        ax1.set_xlim(0, 8)
        ax1.set_ylim(0, 8)
        ax1.invert_yaxis()
        
        # Show access pattern
        for tid in range(32):
            row = tid // 8
            col = tid % 8
            color = plt.cm.Reds(col / 7)  # Same column = same bank = conflict
            rect = Rectangle((col, row), 1, 1,
                           linewidth=1, edgecolor='black',
                           facecolor=color, alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(col + 0.5, row + 0.5, f'{tid}',
                    ha='center', va='center', fontsize=8)
        
        # Add bank labels
        for col in range(8):
            ax1.text(col + 0.5, -0.5, f'Bank {col}',
                    ha='center', va='center', fontsize=8, rotation=45)
        
        # Swizzled pattern (simplified)
        ax2.set_title('Swizzled Thread Mapping\n(Avoids Bank Conflicts)')
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 8)
        ax2.invert_yaxis()
        
        # Simple swizzle pattern
        swizzle_map = {}
        for tid in range(32):
            row = tid // 8
            col = tid % 8
            # XOR-based swizzle
            swizzled_col = col ^ (row % 4)
            swizzle_map[tid] = (row, swizzled_col)
            
            color = plt.cm.Blues(swizzled_col / 7)
            rect = Rectangle((swizzled_col, row), 1, 1,
                           linewidth=1, edgecolor='black',
                           facecolor=color, alpha=0.7)
            ax2.add_patch(rect)
            ax2.text(swizzled_col + 0.5, row + 0.5, f'{tid}',
                    ha='center', va='center', fontsize=8)
        
        # Add bank labels
        for col in range(8):
            ax2.text(col + 0.5, -0.5, f'Bank {col}',
                    ha='center', va='center', fontsize=8, rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("\nC++ Implementation Pattern:")
        print("```cpp")
        print("// Swizzled memory access using space-filling curve")
        print("template <typename SwizzlePattern>")
        print("struct swizzled_tile_window {")
        print("    template <typename Coord>")
        print("    CK_TILE_DEVICE auto get_swizzled_offset(const Coord& coord) {")
        print("        // Apply space-filling curve transformation")
        print("        return SwizzlePattern::transform(coord);")
        print("    }")
        print("};")
        print("```")
    
    @staticmethod
    def demonstrate_hierarchical_tiling():
        """
        Demonstrate hierarchical tiling: thread -> warp -> block.
        """
        print("=== Hierarchical Tiling ===")
        print("CK uses multi-level tiling for optimal resource utilization:")
        print("1. Thread-level: Each thread handles small tile")
        print("2. Warp-level: Warps cooperate on medium tile")
        print("3. Block-level: Thread blocks handle large tile")
        print()
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_title('Hierarchical Tiling Structure', fontsize=16)
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.invert_yaxis()
        
        # Block tile
        block_rect = Rectangle((0, 0), 256, 256,
                             linewidth=4, edgecolor='red',
                             facecolor='none', label='Block Tile (256x256)')
        ax.add_patch(block_rect)
        
        # Warp tiles (2x2 warps)
        warp_size = 128
        warp_colors = ['blue', 'green', 'orange', 'purple']
        for i in range(2):
            for j in range(2):
                warp_id = i * 2 + j
                warp_rect = Rectangle((j * warp_size, i * warp_size),
                                    warp_size, warp_size,
                                    linewidth=3, edgecolor=warp_colors[warp_id],
                                    facecolor='none', 
                                    label=f'Warp {warp_id} Tile' if i==0 and j==0 else "")
                ax.add_patch(warp_rect)
        
        # Thread tiles within one warp (simplified view)
        thread_tile_size = 16
        for i in range(0, 128, thread_tile_size):
            for j in range(0, 128, thread_tile_size):
                if i < 64 and j < 64:  # Show only in first quadrant
                    thread_rect = Rectangle((j, i), thread_tile_size, thread_tile_size,
                                          linewidth=1, edgecolor='gray',
                                          facecolor='lightgray', alpha=0.3)
                    ax.add_patch(thread_rect)
        
        # Add one detailed thread tile
        example_rect = Rectangle((0, 0), thread_tile_size, thread_tile_size,
                               linewidth=2, edgecolor='black',
                               facecolor='yellow', alpha=0.5,
                               label='Thread Tile (16x16)')
        ax.add_patch(example_rect)
        
        ax.legend(loc='upper right')
        ax.set_xlabel('N dimension')
        ax.set_ylabel('M dimension')
        
        plt.tight_layout()
        plt.show()
        
        print("\nC++ Code Structure:")
        print("```cpp")
        print("// Hierarchical tile configuration")
        print("using BlockGemmShape = TileGemmShape<")
        print("    sequence<256, 256, 64>,    // Block-level: MxNxK")
        print("    sequence<2, 2, 1>,         // Warp grid: 2x2 warps") 
        print("    sequence<32, 32, 16>       // Warp-level: MxNxK per warp")
        print(">;")
        print("")
        print("// Thread-level tile is derived:")
        print("// Thread tile M = Warp tile M / Threads per warp in M")
        print("// Thread tile N = Warp tile N / Threads per warp in N")
        print("```")


# =============================================================================
# SECTION 4: Tutorial Examples and Exercises
# =============================================================================

def run_interactive_tutorial():
    """
    Run an interactive tutorial demonstrating tile distribution concepts.
    """
    print("=" * 60)
    print("CK TILE DISTRIBUTION INTERACTIVE TUTORIAL")
    print("=" * 60)
    print()
    
    # Example 1: Basic 2D distribution
    print("### Example 1: Basic 2D Matrix Tiling ###")
    print()
    
    # Create a simple distribution
    from .tensor_descriptor import make_naive_tensor_descriptor
    
    # Matrix: 64x64, Tile: 16x16, Threads: 16x16 per tile
    adaptor = TensorAdaptor(
        top_dimensions=[16, 16],  # 16x16 threads
        bottom_dimensions=[64, 64]  # 64x64 matrix
    )
    
    descriptor = make_naive_tensor_descriptor([4, 4], [4, 1])  # 4x4 tiles
    
    encoding = TileDistributionEncoding(
        rs_lengths=[1],
        hs_lengths=[64, 64],
        ps_to_rs_adaptor=None,
        hs_to_ys_adaptor=None
    )
    
    dist = TileDistributionTutorial(
        ps_ys_to_xs_adaptor=adaptor,
        ys_to_d_descriptor=descriptor,
        encoding=encoding,
        verbose=True
    )
    
    # Demonstrate coordinate mapping
    print("\n### Coordinate Mapping Examples ###")
    test_cases = [
        ([0], [0, 0]),  # Thread 0, Tile (0,0)
        ([5], [0, 1]),  # Thread 5, Tile (0,1)
        ([15], [1, 0]), # Thread 15, Tile (1,0)
    ]
    
    for p_idx, y_idx in test_cases:
        x_idx, trace = dist.calculate_tensor_position(p_idx, y_idx)
        print()
    
    # Example 2: GEMM distribution
    print("\n### Example 2: GEMM Tiling Pattern ###")
    print()
    
    gemm_dist = CommonDistributionPatterns.create_gemm_distribution(
        m_tile=256, n_tile=256, k_tile=64
    )
    
    # Visualizations
    print("\n### Visualizations ###")
    print("Generating tiling visualizations...")
    
    # Show basic 2D distribution
    dist.visualize_distribution_2d(
        tensor_shape=(64, 64),
        tile_shape=(16, 16),
        highlight_thread=5
    )
    
    # Show GEMM tiling
    CommonDistributionPatterns.visualize_gemm_tiling()
    
    # Advanced concepts
    print("\n### Advanced Concepts ###")
    AdvancedDistributionConcepts.demonstrate_swizzling()
    AdvancedDistributionConcepts.demonstrate_hierarchical_tiling()
    
    print("\n### Tutorial Complete! ###")
    print("Key takeaways:")
    print("1. Tile distribution maps processing units to tensor data")
    print("2. Coordinate systems: X (tensor), Y (tile), P (processing), R (replication)")
    print("3. Hierarchical tiling improves resource utilization")
    print("4. Swizzling avoids memory bank conflicts")
    print()
    print("Next steps: Explore tile_window and tensor operations!")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    run_interactive_tutorial()