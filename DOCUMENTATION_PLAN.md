# PythonCK Documentation Plan

## Overview  
This document outlines the complete structure for PythonCK documentation, organized by learning phases and concepts. Edit this file to customize which concepts, classes, and methods to cover.

## 🎯 **Implementation Status Update**
Based on user feedback, the documentation has been **reorganized** to fix dependency issues and provide focused, single-concept pages.

### **✅ Completed Pages**
- [x] **Learning Path Restructured** - Fixed dependency ordering (descriptors before advanced coordinates)
- [x] **Tensor Descriptors** - Complete `to_upper`/`to_lower` examples for all transforms
- [x] **Advanced Coordinate Operations** - Descriptor-dependent functions separated out
- [x] **Tensor Views** - Unified memory access with coordinate integration and out-of-bounds handling
- [x] **Tensor Coordinate System** - Complete three-level hierarchy explanation with all coordinate functions
- [x] **Tensor Adaptor Coordinate** - Detailed TensorAdaptorCoordinate vs TensorCoordinate differences
- [x] **Buffer Views** - Memory abstraction layer with address spaces and operations
- [x] **Autorun Fixed** - Only installation cells run automatically, user clicks for examples

### **🔄 In Progress - Current Focus**
- [x] **Tile Distribution Encoding** - Mathematical encoding as graph structure with RMSNorm examples
- [x] **Tile Distribution** - Parallel processing coordination with adaptors and descriptors

### **📋 Remaining Pages to Create**
- [ ] **Static Distributed Tensors** - Thread-local tensor operations
- [ ] **Tile Windows Deep Dive** - `precompute` and `load` integration with descriptors  
- [ ] **Sweep Operations** - Tile-based computation patterns
- [ ] **Complete Thread Mapping** - Full pipeline documentation

---

## 📚 **Learning Path Structure**

### **Phase 1: Foundation (Memory & Indexing)**

#### 1. Buffer Views (`pytensor/buffer_view.py`)
**Concept**: Memory abstraction layer

**Key Classes:**
- [ ] `BufferView` - Main buffer abstraction
- [ ] `AddressSpace` - Memory address space enumeration  
- [ ] `MemoryOperation` - Memory operation types

**Key Functions:**
- [ ] `make_buffer_view()` - Factory function for buffer creation

**Key Methods:**
- [ ] `BufferView.get()` - Element access with bounds checking
- [ ] `BufferView.set()` - Element writing with validation
- [ ] `BufferView.update()` - Memory operation execution
- [ ] `BufferView.get_address_space()` - Address space retrieval

**Learning Objectives:**
- [ ] Understand different memory hierarchies (GLOBAL, LDS, VGPR)
- [ ] Learn vectorized memory access patterns
- [ ] Master bounds checking and invalid element handling
- [ ] Implement memory operations (SET, ADD, ATOMIC)

---

#### 2. Tensor Coordinates (`pytensor/tensor_coordinate.py`)
**Concept**: Multi-dimensional indexing system

**Key Classes:**
- [x] `MultiIndex` - Basic multi-dimensional index
- [x] `TensorAdaptorCoordinate` - Coordinate with transformation tracking
- [x] `TensorCoordinate` - Extended adaptor coordinate

**Key Functions (Basic):**
- [x] Direct coordinate construction - `TensorCoordinate()` and `TensorAdaptorCoordinate()`
- [x] Coordinate copying and movement concepts
- [x] Coordinate validation patterns

**Key Functions (Advanced - require descriptors):**
- [x] `make_tensor_adaptor_coordinate()` - Create adaptor coordinate from TensorAdaptor
- [x] `make_tensor_coordinate()` - Create tensor coordinate from TensorDescriptor
- [x] `move_tensor_adaptor_coordinate()` - Move coordinate by offset
- [x] `move_tensor_coordinate()` - Move tensor coordinate
- [x] `adaptor_coordinate_is_valid()` - Validate coordinate
- [x] `coordinate_has_valid_offset()` - Check offset validity

**Key Methods:**
- [x] `TensorCoordinate.get_index()` - Get underlying index
- [x] `TensorCoordinate.get_offset()` - Get linear offset
- [x] `TensorAdaptorCoordinate.get_top_index()` - Get top-level index
- [x] `TensorAdaptorCoordinate.get_bottom_index()` - Get bottom-level index
- [x] `TensorAdaptorCoordinate.get_hidden_index()` - Get hidden index

**Learning Objectives:**
- [ ] Understand the three-level coordinate system (top/hidden/bottom)
- [ ] Master the difference between general adaptor coordinates vs tensor-specific coordinates
- [ ] Learn how transformations are applied through hidden dimensions
- [ ] Understand coordinate movement and offset calculation
- [ ] Implement coordinate validation for complex transformation chains
- [ ] Use coordinates for memory addressing in tensor operations

---

#### 3. Tensor Descriptors (`pytensor/tensor_descriptor.py`)
**Concept**: Layout transformations and tensor structure

**Key Classes:**
- [ ] `Transform` - Abstract transformation base class
- [ ] `EmbedTransform` - Strided tensor layout transform
- [ ] `UnmergeTransform` - Packed tensor layout transform  
- [ ] `MergeTransform` - Dimension merging transform
- [ ] `OffsetTransform` - Constant offset transform
- [ ] `PassThroughTransform` - Identity transform
- [ ] `PadTransform` - Padding transform
- [ ] `ReplicateTransform` - Broadcasting transform
- [ ] `TensorAdaptor` - Transformation chain manager
- [ ] `TensorDescriptor` - Complete tensor layout description

**Key Functions:**
- [ ] `make_naive_tensor_descriptor()` - Create strided descriptor
- [ ] `make_naive_tensor_descriptor_packed()` - Create packed descriptor
- [ ] `make_naive_tensor_descriptor_aligned()` - Create aligned descriptor

**Key Methods:**
- [ ] `Transform.calculate_lower_index()` - Forward transformation
- [ ] `Transform.calculate_upper_index()` - Backward transformation
- [ ] `Transform.sympy_upper_to_lower()` - Symbolic forward transform
- [ ] `Transform.sympy_lower_to_upper()` - Symbolic backward transform
- [ ] `TensorDescriptor.get_lengths()` - Get dimension lengths
- [ ] `TensorDescriptor.get_element_space_size()` - Get memory size

**Learning Objectives:**
- [ ] Understand tensor layout transformations
- [ ] Master coordinate space mappings
- [ ] Learn symbolic transformation analysis
- [ ] Implement custom transforms

---

#### 4. Tensor Views (`pytensor/tensor_view.py`)
**Concept**: Unified tensor access interface

**Key Classes:**
- [ ] `TensorView` - Main tensor view class
- [ ] `NullTensorView` - Null pattern implementation

**Key Functions:**
- [ ] `make_tensor_view()` - Create tensor view from components
- [ ] `make_naive_tensor_view()` - Create simple tensor view
- [ ] `make_naive_tensor_view_packed()` - Create packed tensor view
- [ ] `transform_tensor_view()` - Apply transformations

**Key Methods:**
- [ ] `TensorView.get_tensor_descriptor()` - Get layout descriptor
- [ ] `TensorView.get_buffer_view()` - Get underlying buffer
- [ ] `TensorView.get_element_space_size()` - Get memory requirements

**Learning Objectives:**
- [ ] Combine buffers with descriptors
- [ ] Master unified tensor access
- [ ] Understand view transformations
- [ ] Implement tensor operations

---

### **Phase 2: Distribution (Parallel Processing)**

#### 5. Tile Distribution Encoding (`pytensor/tile_distribution_encoding.py`)
**Concept**: Mathematical encoding of data distribution

**Key Classes:**
- [ ] `TileDistributionEncoding` - Core encoding class
- [ ] `TileDistributionEncodingDetail` - Detailed encoding information

**Key Functions:**
- [ ] `make_embed_tile_distribution_encoding()` - Create embedding encoding
- [ ] `make_reduce_tile_distribution_encoding()` - Create reduction encoding

**Key Methods:**
- [ ] `TileDistributionEncoding.get_p_lengths()` - Get P dimension lengths
- [ ] `TileDistributionEncoding.get_y_lengths()` - Get Y dimension lengths
- [ ] `TileDistributionEncoding.get_r_lengths()` - Get R dimension lengths

**Learning Objectives:**
- [ ] Understand parallel data distribution mathematics
- [ ] Master P/Y/R/H coordinate spaces
- [ ] Learn embedding vs reduction patterns
- [ ] Implement custom encodings

---

#### 6. Tile Distribution (`pytensor/tile_distribution.py`)
**Concept**: Parallel processing coordination

**Key Classes:**
- [ ] `TileDistribution` - Main distribution coordinator
- [ ] `TileDistributedSpan` - Distributed span representation
- [ ] `TileDistributedIndex` - Distributed index representation
- [ ] `StaticDistributedTensor` - Thread-local tensor

**Key Functions:**
- [ ] `make_static_tile_distribution()` - Create static distribution
- [ ] `make_tile_distributed_span()` - Create distributed span
- [ ] `make_tile_distributed_index()` - Create distributed index
- [ ] `slice_distribution_from_x()` - Create distribution slice

**Key Methods:**
- [ ] `TileDistribution.get_partition_index()` - Get thread partition
- [ ] `TileDistribution.get_distributed_spans()` - Get distribution spans
- [ ] `TileDistribution.get_y_indices_from_distributed_indices()` - Index conversion

**Learning Objectives:**
- [ ] Coordinate parallel processing elements
- [ ] Master thread-to-data mapping
- [ ] Understand distributed spans
- [ ] Implement partition strategies

---

#### 7. Static Distributed Tensors (`pytensor/static_distributed_tensor.py`)
**Concept**: Thread-local data management

**Key Classes:**
- [ ] `StaticDistributedTensor` - Thread-local tensor container

**Key Functions:**
- [ ] `make_static_distributed_tensor()` - Factory function

**Key Methods:**
- [ ] `StaticDistributedTensor.get_element()` - Access thread-local element
- [ ] `StaticDistributedTensor.set_element()` - Set thread-local element
- [ ] `StaticDistributedTensor.get_num_of_elements()` - Get element count
- [ ] `StaticDistributedTensor.get_thread_buffer()` - Access raw buffer

**Learning Objectives:**
- [ ] Manage thread-local data efficiently
- [ ] Understand memory distribution
- [ ] Master element access patterns
- [ ] Implement thread coordination

---

### **Phase 3: Advanced Operations (GPU Coordination)**

#### 8. Tile Windows (`pytensor/tile_window.py`)
**Concept**: Windowed tensor access

**Key Classes:**
- [ ] `TileWindowWithStaticDistribution` - Main tile window class
- [ ] `TileWindowWithStaticLengths` - Fixed-size tile window

**Key Functions:**
- [ ] `make_tile_window()` - Create tile window
- [ ] `move_tile_window()` - Move window position

**Key Methods:**
- [ ] `TileWindow.load()` - Load data into thread buffer
- [ ] `TileWindow.store()` - Store data from thread buffer
- [ ] `TileWindow.update()` - Update window position

**Learning Objectives:**
- [ ] Access windowed tensor regions
- [ ] Master data loading patterns
- [ ] Understand window movement
- [ ] Implement streaming access

---

#### 9. Sweep Operations (`pytensor/sweep_tile.py`)
**Concept**: Coordinated iteration over distributed data

**Key Classes:**
- [ ] `TileSweeper` - Sweep operation coordinator

**Key Functions:**
- [ ] `sweep_tile()` - Main sweep function
- [ ] `sweep_tile_span()` - Sweep over spans
- [ ] `sweep_tile_uspan()` - Sweep over unrolled spans
- [ ] `make_tile_sweeper()` - Create sweeper

**Key Methods:**
- [ ] `TileSweeper.sweep()` - Execute sweep operation
- [ ] `TileSweeper.get_current_position()` - Get sweep position

**Learning Objectives:**
- [ ] Coordinate parallel iteration
- [ ] Master sweep patterns
- [ ] Understand data traversal
- [ ] Implement custom sweep operations

---

#### 10. Advanced Tile Operations

**Additional Classes to Cover:**
- [ ] `TileScatterGather` (`pytensor/tile_scatter_gather.py`) - Scatter/gather operations
- [ ] `TileWindowLinear` (`pytensor/tile_window_linear.py`) - Linear tile windows
- [ ] `ShuffleTile` (`pytensor/shuffle_tile.py`) - Data shuffling operations
- [ ] `StoreTile` (`pytensor/store_tile.py`) - Tile storage operations
- [ ] `UpdateTile` (`pytensor/update_tile.py`) - Tile update operations

---

## 🎯 **Final Integration: Understanding `tile_distr_thread_mapping.py`**

**Complete Pipeline Demonstration:**
- [ ] Real-world RMSNorm example setup
- [ ] Thread coordination patterns
- [ ] API comparison (old vs. clean)
- [ ] Performance implications
- [ ] Complete tile data visualization

---

## 📖 **Supporting Modules**

### **Utilities**
- [ ] `partition_simulation.py` - Thread simulation
- [ ] `space_filling_curve.py` - Space-filling curves
- [ ] `static_encoding_pattern.py` - Static encoding patterns

---

## 🎨 **Documentation Features**

### **Interactive Elements:**
- [ ] Live code execution for all examples
- [ ] Interactive parameter adjustment
- [ ] Real-time visualization
- [ ] Performance comparisons

### **Learning Aids:**
- [ ] Step-by-step progressions
- [ ] Concept dependency graphs
- [ ] Cross-references between concepts
- [ ] Practical exercises

### **Reference Materials:**
- [ ] Complete API documentation
- [ ] Usage examples for each method
- [ ] Performance benchmarks
- [ ] Troubleshooting guides

---

## ✅ **Customization Instructions**

1. **Check/uncheck concepts** you want to include
2. **Add/remove classes and methods** as needed
3. **Modify learning objectives** to match your goals
4. **Reorder concepts** if you prefer different progression
5. **Add custom examples** in the respective sections

This plan ensures comprehensive coverage while maintaining flexibility for customization! 