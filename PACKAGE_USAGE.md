# PythonCK Package Usage Guide

## Overview

**PythonCK** is a Python implementation of Composable Kernels tensor operations. The package is distributed as `pythonck` but provides three main modules that can be imported directly:

- **`pytensor`**: Core tensor operations, coordinates, descriptors, and views
- **`tensor_transforms`**: Tensor transformation parsing and analysis  
- **`tile_distribution`**: Tile distribution algorithms and visualization

## Installation

### From Wheel

```bash
pip install pythonck-0.1.0-py3-none-any.whl
```

### From Source

```bash
git clone https://github.com/amir/pythonck
cd pythonck
pip install .
```

## Basic Usage

### Import the Modules

```python
import pytensor
import tensor_transforms
import tile_distribution
```

### Core Tensor Operations

```python
# Create tensor descriptors
desc = pytensor.make_naive_tensor_descriptor([128, 256], [256, 1])

# Create tensor coordinates  
adaptor = pytensor.make_tensor_adaptor([64, 128])
coord = pytensor.make_tensor_coordinate(adaptor, [0, 0])

# Work with tensor views
view = pytensor.make_tensor_view(desc, [64, 64], [0, 0])
```

### Tensor Transforms

```python
from tensor_transforms import TensorTransformParser

# Parse tensor transformation expressions
parser = TensorTransformParser()
result = parser.parse_expression("Transform<Merge<1, 2>>")
```

### Tile Distribution

```python
from tile_distribution import get_default_variables, make_tile_distribution

# Get default configuration for specific examples
config = get_default_variables('rmsnorm')

# Create tile distributions
tile_dist = make_tile_distribution(
    lengths_x=[256, 256],
    lengths_y=[4, 2, 8, 4], 
    lengths_p=[2, 2]
)
```

## Usage in Different Environments

### 1. Regular Python Environment

```bash
# Install in virtual environment
python -m venv myproject
source myproject/bin/activate  # Windows: myproject\Scripts\activate
pip install pythonck-0.1.0-py3-none-any.whl

# Use in Python scripts
python my_tensor_script.py
```

### 2. Jupyter Notebooks

```python
# In a Jupyter cell
import pytensor
from tensor_transforms import TensorTransformParser

# Your tensor operations here...
```

### 3. Pyodide/Quarto (Web Environments)

```python
# Install in Pyodide
import micropip
await micropip.install("pythonck-0.1.0-py3-none-any.whl")

# Import and use
import pytensor
from tensor_transforms import TensorTransformParser
```

See [PYODIDE_USAGE.md](PYODIDE_USAGE.md) for detailed Pyodide/Quarto instructions.

### 4. Streamlit Applications

The repository also includes Streamlit applications for interactive exploration:

```bash
# These are NOT part of the wheel - run from source
streamlit run app.py                        # Main dashboard
streamlit run tensor_transform_app.py       # Transform analysis
streamlit run tensor_visualization_app.py   # Tensor visualization
streamlit run thread_visualization_app.py   # Thread mapping
```

## Package Structure

```
pythonck/                          # Repository root
├── pytensor/                      # Core tensor library
├── tensor_transforms/             # Transform parsing
├── tile_distribution/             # Tile distribution
├── app.py                        # Streamlit apps (not in wheel)
├── tensor_transform_app.py       
├── tensor_visualization_app.py    
├── thread_visualization_app.py    
├── tests/                        # Test suite
└── dist/                         # Built wheels
    └── pythonck-0.1.0-py3-none-any.whl
```

## What's Included in the Wheel

✅ **Library Code**:
- `pytensor/` - All tensor operations and utilities
- `tensor_transforms/` - Transform parsing and analysis
- `tile_distribution/` - Tile distribution algorithms

✅ **Documentation**:
- README files and implementation status
- Module docstrings and type hints

❌ **NOT Included**:
- Streamlit applications (these are demos, not library code)
- Test files
- Development tools

## Dependencies

- **Required**: `sympy>=1.9` (symbolic mathematics)
- **Optional**: `matplotlib`, `numpy`, `pandas` (for visualization features)
- **Development**: `pytest`, `black`, `isort`, `flake8`

## Examples

### Basic Tensor Coordinate Example

```python
import pytensor

# Create a tensor adaptor and coordinate
adaptor = pytensor.make_tensor_adaptor([64, 128]) 
coord = pytensor.make_tensor_coordinate(adaptor, [5, 10])

print(f"Coordinate offset: {coord.get_offset()}")
```

### Transform Parsing Example

```python
from tensor_transforms import TensorTransformParser

parser = TensorTransformParser()
transform = parser.parse_expression("Transform<PassThrough<0, 1>>")
print(f"Parsed: {transform}")
```

### Tile Distribution Example  

```python
from tile_distribution import make_tile_distribution

# Create a 2D tile distribution
dist = make_tile_distribution(
    lengths_x=[256, 256],    # Tensor dimensions
    lengths_y=[4, 2, 8, 4],  # Thread hierarchy  
    lengths_p=[2, 2]         # Partition dimensions
)

print(f"Distribution created with {dist.get_num_of_dimension_x()}D tensor")
```

## Building the Package

To build the wheel yourself:

```bash
# Install build tools
pip install build

# Build the wheel
python -m build --wheel

# Install the wheel
pip install dist/pythonck-0.1.0-py3-none-any.whl
```

## Support

- **Repository**: https://github.com/amir/pythonck
- **Issues**: https://github.com/amir/pythonck/issues
- **Documentation**: See module docstrings and README files 