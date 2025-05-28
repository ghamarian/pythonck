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
- `PassThroughTransform`: Identity transform
- `PadTransform`: Padding transform
- `MergeTransform`: Dimension merging transform
- `ReplicateTransform`: Broadcasting transform
- `TensorAdaptor`: Manages transformation chains
- `TensorDescriptor`: Complete tensor layout with element space

**Key Functions:**
- `make_naive_tensor_descriptor()`: Create strided tensor descriptor
- `make_naive_tensor_descriptor_packed()`: Create packed tensor descriptor
- `make_naive_tensor_descriptor_aligned()`: Create aligned tensor descriptor

### tensor_adaptor.py
Additional utilities for creating and manipulating tensor adaptors.

**Key Functions:**
- `make_single_stage_tensor_adaptor()`: Create single-stage adaptor
- `transform_tensor_adaptor()`: Add transformations to existing adaptor
- `chain_tensor_adaptors()`: Chain two adaptors together
- `chain_tensor_adaptors_multi()`: Chain multiple adaptors
- `make_identity_adaptor()`: Create identity adaptor
- `make_transpose_adaptor()`: Create transpose adaptor

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

### tile_distribution_encoding.py
Enhanced tile distribution encoding with detailed mapping information.

**Key Classes:**
- `TileDistributionEncodingDetail`: Computed detail information
- `TileDistributionEncodingEnhanced`: Enhanced encoding with detail computation

**Key Functions:**
- `make_embed_tile_distribution_encoding()`: Combine encodings
- `make_reduce_tile_distribution_encoding()`: Reduce dimensions

### static_distributed_tensor.py
Distributed tensor storage for parallel processing.

**Key Classes:**
- `StaticDistributedTensor`: Tensor distributed across processing elements

**Key Functions:**
- `make_static_distributed_tensor()`: Create distributed tensor

### tile_window.py
Windowed access to tensor data with distribution support.

**Key Classes:**
- `TileWindowWithStaticDistribution`: Distributed window access
- `TileWindowWithStaticLengths`: Simple window access

**Key Functions:**
- `make_tile_window()`: Create tile window
- `move_tile_window()`: Move window position

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

## Implementation Status

### Completed Modules (9/12)
- ✅ buffer_view.py
- ✅ tensor_coordinate.py
- ✅ tensor_descriptor.py
- ✅ tensor_view.py
- ✅ tile_distribution.py
- ✅ tile_distribution_encoding.py
- ✅ static_distributed_tensor.py
- ✅ tile_window.py (with enhanced functionality)
- ✅ tensor_adaptor.py
- ✅ store_tile.py
- ✅ sweep_tile.py

All implemented modules have comprehensive test coverage. The tile_window module has been enhanced with additional methods (load_raw, store_raw, update, async operations) to match the C++ implementation more closely. 