"""
GPU Hierarchy Visualizer - Interactive tool to visualize how work is distributed
across vectors, warps, blocks, and tiles on a GPU.
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

# Set page config
st.set_page_config(
    page_title="GPU Hierarchy Visualizer",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 GPU Work Distribution Visualizer")
st.markdown("""
This tool visualizes how work is hierarchically distributed on a GPU:
- **Vectors**: Individual elements processed together (SIMD)
- **Threads**: Basic execution units
- **Warps**: Groups of threads executing in lockstep (32 threads on AMD/NVIDIA)
- **Blocks**: Groups of warps that can share memory
- **Grid**: Complete set of blocks for the kernel
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Matrix dimensions
st.sidebar.subheader("📊 Matrix Dimensions")
matrix_m = st.sidebar.number_input("Matrix M dimension", min_value=16, max_value=4096, value=256, step=16)
matrix_n = st.sidebar.number_input("Matrix N dimension", min_value=16, max_value=4096, value=256, step=16)

# Hierarchical decomposition
st.sidebar.subheader("🏗️ Hierarchical Decomposition")

# Vector level
vector_m = st.sidebar.number_input("Vector size M", min_value=1, max_value=16, value=4)
vector_n = st.sidebar.number_input("Vector size N", min_value=1, max_value=16, value=4)

# Thread level (threads per warp)
threads_per_warp_m = st.sidebar.number_input("Threads per warp M", min_value=1, max_value=32, value=8)
threads_per_warp_n = st.sidebar.number_input("Threads per warp N", min_value=1, max_value=32, value=8)

# Warp level (warps per block)
warps_per_block_m = st.sidebar.number_input("Warps per block M", min_value=1, max_value=32, value=2)
warps_per_block_n = st.sidebar.number_input("Warps per block N", min_value=1, max_value=32, value=2)

# Repeat factor (how many times each thread repeats)
repeat_m = st.sidebar.number_input("Repeat factor M", min_value=1, max_value=16, value=4)
repeat_n = st.sidebar.number_input("Repeat factor N", min_value=1, max_value=16, value=4)

# Visualization options
st.sidebar.subheader("🎨 Visualization Options")
show_vectors = st.sidebar.checkbox("Show vectors", value=True)
show_threads = st.sidebar.checkbox("Show threads", value=True)
show_warps = st.sidebar.checkbox("Show warps", value=True)
show_blocks = st.sidebar.checkbox("Show blocks", value=True)
show_labels = st.sidebar.checkbox("Show labels", value=True)
color_scheme = st.sidebar.selectbox("Color scheme", ["Blues", "Viridis", "Plasma", "Cool"])

# Calculate derived values
def calculate_hierarchy():
    # Elements per thread
    elements_per_thread_m = vector_m * repeat_m
    elements_per_thread_n = vector_n * repeat_n
    
    # Elements per warp
    elements_per_warp_m = elements_per_thread_m * threads_per_warp_m
    elements_per_warp_n = elements_per_thread_n * threads_per_warp_n
    
    # Elements per block
    elements_per_block_m = elements_per_warp_m * warps_per_block_m
    elements_per_block_n = elements_per_warp_n * warps_per_block_n
    
    # Number of blocks needed
    blocks_m = (matrix_m + elements_per_block_m - 1) // elements_per_block_m
    blocks_n = (matrix_n + elements_per_block_n - 1) // elements_per_block_n
    
    # Total threads
    threads_per_block = threads_per_warp_m * threads_per_warp_n * warps_per_block_m * warps_per_block_n
    total_threads = threads_per_block * blocks_m * blocks_n
    
    return {
        'elements_per_thread': (elements_per_thread_m, elements_per_thread_n),
        'elements_per_warp': (elements_per_warp_m, elements_per_warp_n),
        'elements_per_block': (elements_per_block_m, elements_per_block_n),
        'blocks': (blocks_m, blocks_n),
        'threads_per_block': threads_per_block,
        'total_threads': total_threads,
        'total_elements': matrix_m * matrix_n
    }

hierarchy = calculate_hierarchy()

# Display statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Elements", f"{hierarchy['total_elements']:,}")
    st.metric("Elements/Thread", f"{hierarchy['elements_per_thread'][0]}×{hierarchy['elements_per_thread'][1]}")

with col2:
    st.metric("Total Blocks", f"{hierarchy['blocks'][0]}×{hierarchy['blocks'][1]}")
    st.metric("Elements/Warp", f"{hierarchy['elements_per_warp'][0]}×{hierarchy['elements_per_warp'][1]}")

with col3:
    st.metric("Threads/Block", hierarchy['threads_per_block'])
    st.metric("Elements/Block", f"{hierarchy['elements_per_block'][0]}×{hierarchy['elements_per_block'][1]}")

with col4:
    st.metric("Total Threads", f"{hierarchy['total_threads']:,}")
    coverage = (hierarchy['blocks'][0] * hierarchy['elements_per_block'][0] * 
                hierarchy['blocks'][1] * hierarchy['elements_per_block'][1]) / hierarchy['total_elements'] * 100
    st.metric("Coverage", f"{coverage:.1f}%")

# Create visualization
def create_visualization():
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Get colormap
    cmap = cm.get_cmap(color_scheme)
    
    # Draw the full matrix outline
    matrix_rect = Rectangle((0, 0), matrix_n, matrix_m, 
                          linewidth=3, edgecolor='black', facecolor='none')
    ax.add_patch(matrix_rect)
    
    # Draw blocks
    if show_blocks:
        for bi in range(hierarchy['blocks'][0]):
            for bj in range(hierarchy['blocks'][1]):
                block_x = bj * hierarchy['elements_per_block'][1]
                block_y = bi * hierarchy['elements_per_block'][0]
                
                # Clip to matrix bounds
                block_w = min(hierarchy['elements_per_block'][1], matrix_n - block_x)
                block_h = min(hierarchy['elements_per_block'][0], matrix_m - block_y)
                
                block_rect = Rectangle((block_x, block_y), block_w, block_h,
                                     linewidth=2, edgecolor='red', 
                                     facecolor=cmap(0.1), alpha=0.3)
                ax.add_patch(block_rect)
                
                if show_labels:
                    ax.text(block_x + block_w/2, block_y + block_h/2, 
                           f'Block\n({bi},{bj})', 
                           ha='center', va='center', fontsize=8, weight='bold')
    
    # Draw warps within first block (as example)
    if show_warps and hierarchy['blocks'][0] > 0 and hierarchy['blocks'][1] > 0:
        for wi in range(warps_per_block_m):
            for wj in range(warps_per_block_n):
                warp_x = wj * hierarchy['elements_per_warp'][1]
                warp_y = wi * hierarchy['elements_per_warp'][0]
                
                # Only draw if within matrix bounds
                if warp_x < matrix_n and warp_y < matrix_m:
                    warp_w = min(hierarchy['elements_per_warp'][1], matrix_n - warp_x)
                    warp_h = min(hierarchy['elements_per_warp'][0], matrix_m - warp_y)
                    
                    warp_rect = Rectangle((warp_x, warp_y), warp_w, warp_h,
                                        linewidth=1.5, edgecolor='green', 
                                        facecolor=cmap(0.3), alpha=0.3)
                    ax.add_patch(warp_rect)
                    
                    if show_labels and wi == 0 and wj == 0:
                        ax.text(warp_x + warp_w/2, warp_y + warp_h/2, 
                               f'Warp\n({wi},{wj})', 
                               ha='center', va='center', fontsize=7)
    
    # Draw threads within first warp (as example)
    if show_threads and hierarchy['blocks'][0] > 0 and hierarchy['blocks'][1] > 0:
        for ti in range(min(4, threads_per_warp_m)):  # Show first few threads
            for tj in range(min(4, threads_per_warp_n)):
                thread_x = tj * hierarchy['elements_per_thread'][1]
                thread_y = ti * hierarchy['elements_per_thread'][0]
                
                if thread_x < matrix_n and thread_y < matrix_m:
                    thread_w = min(hierarchy['elements_per_thread'][1], matrix_n - thread_x)
                    thread_h = min(hierarchy['elements_per_thread'][0], matrix_m - thread_y)
                    
                    thread_rect = Rectangle((thread_x, thread_y), thread_w, thread_h,
                                          linewidth=1, edgecolor='blue', 
                                          facecolor=cmap(0.5), alpha=0.3)
                    ax.add_patch(thread_rect)
                    
                    if show_labels and ti < 2 and tj < 2:
                        ax.text(thread_x + thread_w/2, thread_y + thread_h/2, 
                               f'T({ti},{tj})', 
                               ha='center', va='center', fontsize=6)
    
    # Draw vectors within first thread (as example)
    if show_vectors and hierarchy['blocks'][0] > 0 and hierarchy['blocks'][1] > 0:
        for ri in range(min(2, repeat_m)):  # Show first few repeats
            for rj in range(min(2, repeat_n)):
                for vi in range(1):  # Just show one vector as example
                    for vj in range(1):
                        vec_x = rj * vector_n
                        vec_y = ri * vector_m
                        
                        if vec_x < matrix_n and vec_y < matrix_m:
                            vec_rect = Rectangle((vec_x, vec_y), vector_n, vector_m,
                                               linewidth=0.5, edgecolor='purple', 
                                               facecolor=cmap(0.7), alpha=0.5)
                            ax.add_patch(vec_rect)
                            
                            if show_labels and ri == 0 and rj == 0:
                                ax.text(vec_x + vector_n/2, vec_y + vector_m/2, 
                                       f'Vec\n{vector_m}×{vector_n}', 
                                       ha='center', va='center', fontsize=5)
    
    # Set axis properties
    ax.set_xlim(0, matrix_n)
    ax.set_ylim(0, matrix_m)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Matrix convention: origin at top-left
    
    # Labels
    ax.set_xlabel('N dimension', fontsize=12)
    ax.set_ylabel('M dimension', fontsize=12)
    ax.set_title(f'GPU Work Distribution: {matrix_m}×{matrix_n} Matrix', fontsize=14, weight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    legend_elements = []
    if show_blocks:
        legend_elements.append(patches.Patch(facecolor=cmap(0.1), edgecolor='red', 
                                           alpha=0.3, label='Block'))
    if show_warps:
        legend_elements.append(patches.Patch(facecolor=cmap(0.3), edgecolor='green', 
                                           alpha=0.3, label='Warp'))
    if show_threads:
        legend_elements.append(patches.Patch(facecolor=cmap(0.5), edgecolor='blue', 
                                           alpha=0.3, label='Thread'))
    if show_vectors:
        legend_elements.append(patches.Patch(facecolor=cmap(0.7), edgecolor='purple', 
                                           alpha=0.5, label='Vector'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    return fig

# Display visualization
st.subheader("📊 Hierarchical Work Distribution")
fig = create_visualization()
st.pyplot(fig)

# Detailed breakdown
with st.expander("📋 Detailed Hierarchy Breakdown"):
    st.markdown(f"""
    ### Complete Decomposition
    
    **Matrix**: {matrix_m} × {matrix_n} = {hierarchy['total_elements']:,} elements
    
    **Per Block** ({hierarchy['blocks'][0]} × {hierarchy['blocks'][1]} blocks):
    - Size: {hierarchy['elements_per_block'][0]} × {hierarchy['elements_per_block'][1]} elements
    - Warps: {warps_per_block_m} × {warps_per_block_n} = {warps_per_block_m * warps_per_block_n} warps
    - Threads: {hierarchy['threads_per_block']} threads
    
    **Per Warp**:
    - Size: {hierarchy['elements_per_warp'][0]} × {hierarchy['elements_per_warp'][1]} elements
    - Threads: {threads_per_warp_m} × {threads_per_warp_n} = {threads_per_warp_m * threads_per_warp_n} threads
    
    **Per Thread**:
    - Size: {hierarchy['elements_per_thread'][0]} × {hierarchy['elements_per_thread'][1]} elements
    - Repeats: {repeat_m} × {repeat_n} = {repeat_m * repeat_n} iterations
    - Vectors per iteration: {vector_m} × {vector_n} elements
    
    **Memory Access Pattern**:
    - Each thread loads {vector_m * vector_n} elements per iteration
    - Total iterations per thread: {repeat_m * repeat_n}
    - Total elements per thread: {hierarchy['elements_per_thread'][0] * hierarchy['elements_per_thread'][1]}
    """)

# CK encoding representation
with st.expander("🔧 CK Tile Distribution Encoding"):
    st.code(f"""
# Equivalent CK encoding
encoding = make_tile_distribution_encoding(
    rs_lengths=[],  # No replication
    hs_lengthss=[
        [{repeat_m}, {warps_per_block_m}, {threads_per_warp_m}, {vector_m}],  # M dimension
        [{repeat_n}, {warps_per_block_n}, {threads_per_warp_n}, {vector_n}]   # N dimension
    ],
    ps_to_rhss_major=[[1, 2], [1, 2]],  # Standard mapping
    ps_to_rhss_minor=[[1, 1], [2, 2]],
    ys_to_rhs_major=[1, 1, 2, 2],
    ys_to_rhs_minor=[0, 3, 0, 3]
)

# Total tile size: {hierarchy['elements_per_block'][0]}×{hierarchy['elements_per_block'][1]}
# Threads per block: {hierarchy['threads_per_block']}
# Elements per thread: {hierarchy['elements_per_thread'][0]}×{hierarchy['elements_per_thread'][1]}
    """, language='python')

# Tips and insights
st.subheader("💡 Tips and Insights")
st.info("""
**Understanding the Hierarchy**:
- **Vectors**: Enable coalesced memory access and SIMD operations
- **Threads**: Individual execution units that process multiple vectors
- **Warps**: Execute in lockstep (SIMT), enabling efficient control flow
- **Blocks**: Share memory and synchronize, enabling cooperation

**Optimization Guidelines**:
- Vector size should match memory transaction size (typically 4 for float)
- Threads per warp should utilize available lanes (8×8 = 64 for 32-thread warps)
- Warps per block should balance occupancy and resource usage
- Repeat factor amortizes instruction overhead
""")

# Footer
st.markdown("---")
st.markdown("Built with CK (Composable Kernel) concepts for GPU optimization")