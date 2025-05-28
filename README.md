# Tile Distribution Visualizer

This tool provides an interactive visualization of the `tile_distribution_encoding` concepts from the Composable Kernels library. It allows you to explore the hierarchical structure of tile distributions and understand how threads are mapped to data elements.

## Components

The visualization system consists of three main components:

1. **Parser (`parser.py`)**: Parses C++ `tile_distribution_encoding` structures
2. **Tiler (`tiler.py`)**: Implements the tile distribution functionality 
3. **Visualizer (`visualizer.py`)**: Creates visual representations of the encoding structure and thread mappings
4. **Streamlit App (`app.py`)**: Provides an interactive web interface

## Installation

1. Make sure you have Python 3.8 or newer installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

To start the interactive visualization tool:

```bash
cd /main/ck_tile/visualisation
streamlit run app.py
```

### Using the Parser Directly

```python
from parser import TileDistributionParser

# Example C++ code
cpp_code = """
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
result = parser.parse_tile_distribution_encoding(cpp_code)
print(result)
```

### Creating Visualizations

```python
from parser import TileDistributionParser
from visualizer import visualize_encoding_structure
import matplotlib.pyplot as plt

# Parse encoding
parser = TileDistributionParser()
encoding = parser.parse_tile_distribution_encoding(cpp_code)

# Visualize encoding structure
fig = visualize_encoding_structure(encoding)
plt.savefig("encoding_structure.png")
plt.show()
```

## Understanding the Visualization

The tile distribution visualization shows three key levels:

1. **Top Level (P_dims and Y_dims)**: The top-level dimensions that organize the distribution
2. **Hidden Level**: The hierarchy numbers that connect top dimensions to bottom
3. **Bottom Level (X_dims)**: The tensor dimensions consisting of R_dims and H_dims

The connections between these levels show how threads are mapped to data elements.

## Features

- Parse C++ `tile_distribution_encoding` structures
- Visualize the encoding structure as a hierarchical diagram
- Show thread mappings to tile elements
- Animate thread access patterns
- Provide performance metrics (occupancy and utilization)
- Support variable adjustment for parameterized encodings

## License

This tool is provided under the same license as the Composable Kernels library.

## Acknowledgments

- AMD Composable Kernels team for the original concept
- Streamlit for the interactive app framework 

# Tile Distribution Visualization and Pedantic Implementation

This project visualizes tile distribution concepts from the Composable Kernel library and includes a pedantic Python reimplementation (`tiler_pedantic.py`) of the core C++ logic found in `tile_distribution.hpp` and `tile_distribution_encoding.hpp`.

## Hierarchical Parameters Calculation

The visualizer displays several hierarchical parameters that describe how a tile of computation is mapped to GPU execution resources. These include:

*   **Vector Size (K)**: The number of data elements processed per thread in its innermost loop (vectorized dimension).
*   **Thread/Warp (TPW)**: The 2D arrangement of threads within a warp (e.g., M rows x N columns of threads).
*   **Warps/Block (WPB)**: The 2D arrangement of warps within a thread block.
*   **Repeat**: Factors indicating if the entire block pattern is repeated.
*   **Block Size**: The total 2D dimensions of a thread block, derived from TPW, WPB, and Repeat.

### 1. C++ Ground Truth (Conceptual)

In a full Composable Kernel problem definition, these parameters are often explicitly specified or clearly derivable from types like `DistributionOrder` and `Problem::MakeBlockGemmProblem` or similar problem constructors. These C++ structures provide explicit template arguments or constructor parameters for:

*   `ThreadPerWarp_` (e.g., `ck_tile::sequence<threads_m_per_warp, threads_n_per_warp>`)
*   `WarpPerBlock_` (e.g., `ck_tile::sequence<warps_m_per_block, warps_n_per_block>`)
*   `VectorK_` (or similar, denoting the vector size along a K-like dimension)
*   `Repeat_` (e.g., `ck_tile::sequence<repeat_m, repeat_n>`)

The `tile_distribution_encoding` itself (the structure focused on by this visualizer and `tiler_pedantic.py`) *does not directly store these explicit distribution strategy parameters*. It only stores the mapping of P (problem-level thread distribution) and Y (problem-level spatial/output distribution) dimensions to R (reduction) and H (hardware/tile) dimensions.

### 2. `tiler.py` (Original Visualizer Logic)

The original `tiler.py` module attempts to *infer* these hierarchical parameters when they are not explicitly provided as "visualization hints" within the input encoding dictionary. Its goal is to provide a reasonable interpretation for visualization based on common Composable Kernel patterns.

**Assumptions and Heuristics in `tiler.py` (Conceptual):**

*   **Identifying M, N, K-like dimensions**: It analyzes the `HsLengthss` (H-dimension sequences) and how `Ps2RHssMajor/Minor` and `Ys2RHsMajor/Minor` map to them. It tries to identify which H-dimensions correspond to logical M (row-like), N (column-like), and K (reduction/vector-like) aspects of the problem.
    *   For example, H-dimensions mapped by P-dims that are part of a "thread tile" (e.g., `BlockTileM`, `BlockTileN` in some CK naming conventions) would inform `ThreadPerWarp` and `WarpPerBlock`.
    *   An H-dimension mapped by a Y-dim, often at the end of an H-sequence (e.g., `Kv`), is a strong candidate for `VectorK_`.
*   **Default Values**: If inference is difficult or ambiguous, `tiler.py` might fall back to common default values (e.g., 4 warps, 32 threads/warp in some configuration, vector size 4 or 8).
*   **`visualization_hints`**: `tiler.py` allows users to provide `visualization_hints` in the input dictionary to override inference for specific parameters (e.g., `'vector_dim_ys_index'`, `'thread_per_warp_p_indices'`).

The key is that `tiler.py` contains specific Python logic to look at the *structure* of the parsed `encoding_dict` and the *names/values* of H-dimensions (resolved from variables) to make educated guesses about the hierarchical parameters.

### 3. `tiler_pedantic.py` (Pedantic Reimplementation)

`tiler_pedantic.py` aims to be a close reimplementation of the C++ `tile_distribution.hpp` and `tile_distribution_encoding.hpp`.

**Current State and Differences from `tiler.py` regarding Hierarchical Parameters:**

*   **No C++ Equivalent for Inference**: The C++ `tile_distribution` classes *do not perform inference* of `ThreadPerWarp`, `WarpPerBlock`, etc. These are typically fixed by the template parameters of the specific `tile_distribution` instantiation (e.g., `ck_tile::tile_distribution<Problem::TileDesc::BlockWarps, ...>`) or the `DistributionOrder` it's constructed with. The `tile_distribution_encoding` itself doesn't hold enough info for this.
*   **`calculate_hierarchical_tile_structure` in `tiler_pedantic.py`**:
    *   This method in `tiler_pedantic.py` currently **relies almost entirely on `visualization_hints`** passed in the input `encoding_dict`.
    *   If these hints (e.g., `vector_dim_ys_index`, `thread_per_warp_p_indices`, `warp_per_block_p_indices`) are missing, it falls back to very basic defaults (e.g., TPW=[1,1], WPB=[1], VecK=1).
    *   It **does not (yet) replicate the sophisticated inference heuristics** present in the original `tiler.py`. This is the primary reason for the discrepancy you observed (1x1 TPW, etc.).
*   **Goal of `tiler_pedantic.py`**: Its main purpose was to accurately model the *data structures* within `tile_distribution_encoding` (the `detail` struct) and the *core calculation logic* (like `calculate_index`, `get_y_indices_from_p_indices`). Replicating the visualization-specific inference logic of `tiler.py` was a secondary goal, and is not yet complete.

**Path to Alignment (for Visualization):**

To make `tiler_pedantic.py` produce the same hierarchical visualization as `tiler.py` without explicit hints, its `calculate_hierarchical_tile_structure` method would need to be enhanced to include similar inference logic to what `tiler.py` employs. This would involve:

1.  Analyzing `self.DstrEncode.HsLengthss`, `self.DstrEncode.PsLengths`, `self.DstrEncode.detail['ys_lengths_']`.
2.  Examining the mappings (`Ps2RHssMajor/Minor`, `Ys2RHsMajor/Minor`).
3.  Making educated guesses based on common patterns (e.g., assuming the last element of an H-sequence mapped by a Y-dim is a vector dimension, or specific P-dim mappings contribute to warp/thread counts).
4.  This would make this specific part of `tiler_pedantic.py` less "pedantically C++-derived" (as C++ doesn't do this inference in `tile_distribution_encoding`) and more "visualizer-compatibility-driven."

The current discrepancy is expected given the different focuses of `tiler.py` (visualization with inference) and `tiler_pedantic.py` (pedantic C++ calculation core).

### Heuristics for Hierarchical Parameter Inference in `tiler_pedantic.py` (Work in Progress)

To improve compatibility with `app.py` when explicit `visualization_hints` are not provided, `tiler_pedantic.py`'s `calculate_hierarchical_tile_structure` method includes some initial heuristics to infer `VectorDimensions`, `ThreadPerWarp` (TPW), and `WarpPerBlock` (WPB). This is an ongoing effort and may not cover all Composable Kernel conventions perfectly.

**1. VectorDimensions (K):**
   - **Hint**: If `visualization_hints['vector_dim_ys_index']` is provided, the length of the corresponding Y-dimension (from `detail['ys_lengths_']`) is used.
   - **Inference (if no hint)**:
     - If there are Y-dimensions (`NDimYs > 0`) and `detail['ys_lengths_']` is populated:
       - The length of the *last* Y-dimension (`detail['ys_lengths_'][-1]`) is considered.
       - If this length is greater than 1 and less than or equal to 16 (a common range for vector sizes), it is used as the `VectorDimensions` (e.g., `[length]`).
     - Otherwise, it defaults to `[1]`.
   - **C++ Reference**: There's no direct C++ inference. `VectorK_` is usually an explicit template parameter in higher-level problem definitions or distribution strategies. `tile_distribution_encoding` itself (see `tile_distribution_encoding.hpp`) only provides `ys_lengths_` (L72) which are the lengths of the R/H components that Y-dimensions map to.

**2. ThreadPerWarp (TPW):**
   - **Hint**: If `visualization_hints['thread_per_warp_p_indices']` (e.g., `[p_idx_m, p_idx_n]`) is provided, the M and N components of TPW are derived. For each given P-dimension index, `_get_lengths_for_p_dim_component(p_idx)` is called. This helper function sums up the lengths of all R/H components that the specified P-dimension maps to (based on `Ps2RHssMajor` and `Ps2RHssMinor` from `tile_distribution_encoding.hpp`). The product of these lengths forms the TPW component.
   - **Inference (if no hint)**:
     - `tpw_m`: If P0 exists (`NDimP >= 1`), it's tentatively assigned to the M-dimension of TPW. The product of lengths from `_get_lengths_for_p_dim_component(0)` is used. Defaults to 1 if P0 doesn't contribute a valid length.
     - `tpw_n`: If P1 exists (`NDimP >= 2`), it's tentatively assigned to the N-dimension of TPW. The product of lengths from `_get_lengths_for_p_dim_component(1)` is used. Defaults to 1.
     - Result is `[max(1, tpw_m), max(1, tpw_n)]`.
   - **C++ Reference**: `ThreadPerWarp_` is typically an explicit `ck_tile::sequence` in C++ problem definitions. `Ps2RHssMajor` and `Ps2RHssMinor` in `tile_distribution_encoding.hpp` (L23-L24) define how P-dimensions map to R/H components, which `_get_lengths_for_p_dim_component` tries to interpret.

**3. WarpPerBlock (WPB):**
   - **Hint**: If `visualization_hints['warp_per_block_p_indices']` (e.g., `[p_idx_m]` or `[p_idx_m, p_idx_n]`) is provided, components are derived similarly to TPW hints using `_get_lengths_for_p_dim_component`.
   - **Inference (if no hint)**:
     - Assumes P-dimensions beyond those potentially used for TPW (i.e., P0, P1) might contribute to WPB.
     - `wpb_m`: If P2 exists (`NDimP >= 3`), product of lengths from `_get_lengths_for_p_dim_component(2)`. Defaults to 1.
     - `wpb_n`: If P3 exists (`NDimP >= 4`), product of lengths from `_get_lengths_for_p_dim_component(3)`. Defaults to 1.
     - Result is typically `[max(1, wpb_m)]` if `wpb_n` remains 1, or `[max(1, wpb_m), max(1, wpb_n)]`.
   - **C++ Reference**: `WarpPerBlock_` is also usually an explicit `ck_tile::sequence`. The inference logic here is a convention-based guess.

**4. Repeat:**
   - Currently defaults to `[1,1]` if no `visualization_hints['repeat_factor_ys_index']` is provided. Inference is not yet implemented due to its complexity without broader problem context.

**Important Note on Inference:**
The inference logic is based on common conventions where P0/P1 often define thread work within a warp, and P2/P3 might define warp arrangements. These are heuristics and may not hold for all `tile_distribution_encoding` schemes. The C++ `tile_distribution.hpp` and `tile_distribution_encoding.hpp` themselves do not perform such inference; they are parameterized by these strategic choices at a higher level. 

# PyTensor - Python Implementation of Composable Kernels Tensor Operations

This package provides educational Python implementations of tensor operations from the Composable Kernels library. The implementations focus on clarity and understanding rather than performance.

## Modules

### buffer_view.py
Memory buffer abstraction with different address spaces and access patterns.

**Key Classes:**
- `BufferView`: Generic buffer view with configurable memory operations
- `AddressSpace`: Enum for memory spaces (GENERIC, GLOBAL, LDS, VGPR)
- `MemoryOperation`: Enum for operations (SET, ADD, ATOMIC_ADD, ATOMIC_MAX)

**Key Functions:**
- `make_buffer_view()`: Create a buffer view

### tensor_coordinate.py
Multi-dimensional tensor coordinate handling and transformations.

**Key Classes:**
- `MultiIndex`: Multi-dimensional index representation
- `TensorAdaptorCoordinate`: Coordinate with transformation tracking
- `TensorCoordinate`: Extended adaptor coordinate

**Key Functions:**
- `make_tensor_adaptor_coordinate()`: Create adaptor coordinate
- `make_tensor_coordinate()`: Create tensor coordinate
- `move_tensor_adaptor_coordinate()`: Move coordinate by offset
- `is_valid_tensor_adaptor_coordinate()`: Validate coordinate

### tensor_descriptor.py
Tensor layout descriptions with transformation support.

**Key Classes:**
- `Transform`: Abstract base for coordinate transformations
- `EmbedTransform`: Strided tensor layout transform
- `UnmergeTransform`: Packed tensor layout transform
- `OffsetTransform`: Constant offset transform
- `TensorAdaptor`: Manages transformation chains
- `TensorDescriptor`: Complete tensor layout with element space

**Key Functions:**
- `make_naive_tensor_descriptor()`: Create strided tensor descriptor
- `make_naive_tensor_descriptor_packed()`: Create packed tensor descriptor
- `make_naive_tensor_descriptor_aligned()`: Create aligned tensor descriptor

### tensor_view.py
Unified tensor access interface combining buffers and descriptors.

**Key Classes:**
- `TensorView`: Provides array-style access to tensor data

**Key Functions:**
- `make_tensor_view()`: Create tensor view from buffer and descriptor
- `make_naive_tensor_view()`: Create view with strided layout
- `make_naive_tensor_view_packed()`: Create view with packed layout
- `transform_tensor_view()`: Apply transformation to view

### tile_distribution.py
Tile distribution for parallel processing across threads/warps/blocks.

**Key Classes:**
- `TileDistributedSpan`: Partial lengths in each dimension
- `TileDistributedIndex`: Partial indices in each dimension
- `TileDistributionEncoding`: Encodes distribution mapping
- `TileDistribution`: Manages data distribution across processing elements

**Key Functions:**
- `make_tile_distributed_span()`: Create distributed span
- `make_tile_distributed_index()`: Create distributed index
- `make_tile_distribution_encoding()`: Create distribution encoding
- `make_tile_distribution()`: Create tile distribution

## Usage Example

```python
import numpy as np
from pytensor import *

# Create a buffer
data = np.arange(24, dtype=np.float32).reshape(6, 4)
buffer = make_buffer_view(data)

# Create a tensor descriptor for 2D layout
descriptor = make_naive_tensor_descriptor([6, 4], [4, 1])

# Create a tensor view
tensor = make_tensor_view(buffer, descriptor)

# Access elements
print(tensor[2, 3])  # Access element at row 2, column 3

# Create a tile distribution for parallel processing
encoding = make_tile_distribution_encoding(
    rs_lengths=[],  # No replication
    hs_lengthss=[[2, 3], [4]],  # Hierarchical dimensions
    ps_to_rhss_major=[[1]],  # Partition mapping
    ps_to_rhss_minor=[[0]],
    ys_to_rhs_major=[1, 2],  # Tile mapping
    ys_to_rhs_minor=[0, 0]
)

# More examples in example_usage.py
```

## Testing

Run tests with pytest:

```bash
cd tests
python -m pytest -v
```

## Design Philosophy

These implementations prioritize:
1. **Clarity**: Easy to understand code structure
2. **Correctness**: Accurate implementation of concepts
3. **Education**: Suitable for learning complex tensor operations
4. **Pythonic**: Following Python idioms and conventions

The code adapts C++ template metaprogramming concepts to Python's dynamic nature while maintaining the mathematical rigor of the original implementations. 