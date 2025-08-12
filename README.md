# PythonCK: Python Implementation of Composable Kernels Tensor Operations

This repository provides educational Python implementations of tensor operations and concepts from the AMD Composable Kernels library. The implementations focus on clarity and understanding rather than performance, making them ideal for learning GPU programming concepts.

## Core Components

### **PyTensor** - Core Library
Python implementation of CK Tile tensor primitives including:
- **Tensor Operations**: Multi-dimensional tensor handling with flexible layouts
- **Coordinate Systems**: Advanced indexing and transformation systems
- **Tile Distribution**: Parallel data distribution across GPU threads/warps/blocks
- **Buffer Management**: Memory abstraction with different address spaces
- **Transform Pipeline**: Tensor descriptor transformations and layout conversions

### **Interactive Applications**
Three Streamlit web applications for visualization and exploration:

1. **`app.py`** - **Tile Distribution Visualizer**
   - Interactive exploration of `tile_distribution_encoding` structures
   - Hierarchical visualization of thread-to-data mappings
   - Real-time parameter adjustment with visual feedback
   - Support for complex CK Tile distribution patterns

2. **`tensor_transform_app.py`** - **Tensor Transform Visualizer** 
   - Analysis and visualization of tensor descriptor transformations
   - Forward and backward transformation graph generation
   - C++ descriptor parsing with variable substitution
   - Interactive exploration of merge, unmerge, embed, and XOR transforms

3. **`thread_visualization_app.py`** - **Thread Coordinate Visualization**
   - Thread access pattern visualization as color-coded grids
   - Hierarchical tile structure analysis
   - Manual coordinate testing and debugging
   - Real-time thread-to-tensor coordinate mapping

### **Command Line Examples**
- **`tile_distr_thread_mapping.py`** - Comprehensive example demonstrating API usage patterns and thread coordinate mapping workflows

### **Educational Documentation**
Comprehensive Quarto-based documentation with interactive examples:
- **Foundation concepts**: Buffer views, tensor coordinates, descriptors
- **Distribution concepts**: Tile encoding, parallel coordination
- **Advanced operations**: Windowed access, sweep operations
- **Complete learning path**: Structured 10-step progression
- **Interactive code execution**: Run examples directly in browser

## Quick Start

### Running the Applications

```bash
# Tile Distribution Visualizer
streamlit run app.py

# Tensor Transform Visualizer  
streamlit run tensor_transform_app.py

# Thread Visualization
streamlit run thread_visualization_app.py
```

### Command Line Example
```bash
python tile_distr_thread_mapping.py
```

### Documentation
```bash
cd documentation
quarto serve
```

## Repository Structure

```
├── pytensor/                       # Core Python implementation
│   ├── buffer_view.py             # Memory buffer abstraction
│   ├── tensor_coordinate.py       # Multi-dimensional indexing
│   ├── tensor_descriptor.py       # Layout transformations
│   ├── tensor_view.py             # Unified tensor access
│   ├── tile_distribution.py       # Parallel data distribution
│   ├── tile_distribution_encoding.py # Distribution encoding
│   ├── tile_window.py             # Windowed tensor operations
│   ├── sweep_tile.py              # Coordinated iteration
│   └── ...                        # Additional tensor operations
├── tensor_transforms/              # Transform analysis tools
│   ├── parser.py                  # C++ descriptor parsing
│   ├── analyzer.py                # Transformation analysis
│   ├── examples.py                # Transform examples
│   └── extract_descriptors.py     # Descriptor extraction
├── tile_distribution/              # Tile distribution utilities
│   ├── parser.py                  # C++ encoding parsing
│   ├── visualizer.py              # Visualization functions
│   ├── examples.py                # Distribution examples
│   └── tiler_pedantic.py          # Reference implementation
├── cpp/                           # C++ header references
│   ├── tensor/                    # Tensor operation headers
│   └── algorithm/                 # Algorithm headers
├── documentation/                 # Quarto educational docs
│   ├── concepts/                  # Core concept explanations
│   ├── tutorials/                 # Interactive tutorials
│   ├── api/                       # API documentation
│   └── _site/                     # Generated documentation
├── tests/                         # Comprehensive test suite
├── app.py                         # Tile Distribution Visualizer
├── tensor_transform_app.py        # Tensor Transform Visualizer
├── thread_visualization_app.py    # Thread Visualization App
└── tile_distr_thread_mapping.py   # Command line example
```

## Learning Path

### 1. **Start with Documentation**
Visit the interactive documentation to understand core concepts:
- Buffer views and memory management
- Tensor coordinates and transformations
- Distribution encoding principles

### 2. **Explore Applications**
- **`app.py`**: Understand tile distribution patterns
- **`tensor_transform_app.py`**: Learn transformation graphs
- **`thread_visualization_app.py`**: See thread access patterns

### 3. **Study the Implementation**
- Review `pytensor/` modules for implementation details
- Examine test cases for usage patterns
- Analyze the command line example

## Key Concepts

### **Tensor Operations**
- **Buffer Views**: Abstract memory access with different address spaces (GLOBAL, LDS, VGPR)
- **Tensor Coordinates**: Multi-dimensional indexing with transformation support
- **Tensor Descriptors**: Layout definitions with transformation chains
- **Tensor Views**: Unified interface combining buffers and descriptors

### **Distribution & Parallelization**
- **Tile Distribution Encoding**: Defines how data is distributed across processing elements
- **Static Distributed Tensors**: Thread-local data views
- **Partition Simulation**: GPU thread positioning simulation
- **Coordinate Mapping**: P/Y coordinates → X tensor coordinates

### **Advanced Operations**
- **Tile Windows**: Windowed access patterns for efficient data movement
- **Sweep Operations**: Coordinated iteration over distributed data
- **Transform Pipeline**: Chain multiple tensor layout transformations
- **Thread Coordination**: Understanding GPU execution patterns

## Installation

### Using Poetry (Recommended)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/ROCm/composable_kernel.git
cd composable_kernel/pythonck

# Install dependencies
poetry install

# Install with visualization support
poetry install -E visualization

# Install with Streamlit apps support
poetry install -E streamlit
```

### Using pip

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install the package directly
pip install -e .
```

### Documentation

To build and serve the documentation locally:

```bash
# Install Quarto from https://quarto.org
cd tile_distribution_documentation
quarto serve
```

## Features

### **Interactive Visualization**
- Real-time parameter adjustment
- Color-coded thread access patterns  
- Hierarchical tile structure display
- Transformation graph generation

### **Educational Focus**
- Clear, readable Python implementations
- Comprehensive documentation with examples
- Step-by-step learning progression
- Interactive code execution

### **Comprehensive Coverage**
- Complete tensor operation suite
- Multiple visualization approaches
- Extensive test coverage
- Cross-referenced documentation

## Contributing

This project serves as an educational resource for understanding Composable Kernels concepts. Contributions that improve clarity, add examples, or enhance documentation are welcome.

## License

This project is provided under the same license as the Composable Kernels library.

## Acknowledgments

- AMD Composable Kernels team for the original concepts
- Streamlit for the interactive application framework
- Quarto for the documentation system

---

*Ready to explore GPU tensor operations? Start with the [interactive documentation](documentation/) or dive into the [applications](#interactive-applications)!* 