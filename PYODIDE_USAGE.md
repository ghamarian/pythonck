# Using PythonCK in Pyodide/Quarto

This document explains how to use the `pythonck` package in Pyodide environments, including Quarto documents.

## Package Overview

The `pythonck` package provides Python implementations of Composable Kernels tensor operations, organized into three main modules:

- **`pytensor`**: Core tensor operations, coordinates, descriptors, and views
- **`tensor_transforms`**: Tensor transformation parsing and analysis
- **`tile_distribution`**: Tile distribution algorithms and visualization

## Installation in Pyodide

### Option 1: Install from Wheel

If you have uploaded the wheel to a web server or PyPI:

```python
import micropip
await micropip.install("pythonck")
```

### Option 2: Install from Local Wheel

If you have the wheel file locally:

```python
import micropip
await micropip.install("path/to/pythonck-0.1.0-py3-none-any.whl")
```

### Option 3: Install from URL

If the wheel is hosted online:

```python
import micropip
await micropip.install("https://example.com/path/to/pythonck-0.1.0-py3-none-any.whl")
```

## Usage in Quarto

### Basic Setup

```python
# In a Quarto Python cell
import micropip
await micropip.install("pythonck")  # or your wheel URL

# Import the modules
import pytensor
import tensor_transforms
import tile_distribution
```

### Example: Tensor Coordinates

```python
from pytensor import make_tensor_coordinate, make_tensor_adaptor_coordinate

# Create a simple tensor coordinate
coord = make_tensor_coordinate([64, 128])
print(f"Tensor shape: {coord.get_lengths()}")

# Create an adaptor coordinate for transformations
adaptor_coord = make_tensor_adaptor_coordinate([32, 64], [0, 0])
print(f"Adaptor coordinate at origin: {adaptor_coord.get_offset()}")
```

### Example: Tensor Transforms

```python
from tensor_transforms import TensorTransformParser

# Parse tensor transform expressions
parser = TensorTransformParser()
result = parser.parse_expression("Transform<Merge<1, 2>>")
print(f"Parsed transform: {result}")
```

### Example: Tile Distribution

```python
from tile_distribution import get_default_variables, TileDistributionVisualizer

# Get default configuration
config = get_default_variables()
print(f"Default tile configuration: {config}")

# Create visualizations (if matplotlib is available)
try:
    visualizer = TileDistributionVisualizer()
    # Use visualizer for tile distribution plots
except ImportError:
    print("Matplotlib not available for visualization")
```

## Dependencies

The package has minimal dependencies for Pyodide compatibility:

- **Required**: `sympy>=1.9` (for symbolic mathematics)
- **Optional**: `matplotlib`, `numpy`, `pandas` (for visualization features)

## Pyodide Compatibility Notes

1. **Pure Python**: The package is 100% pure Python, ensuring full Pyodide compatibility
2. **No Native Extensions**: No C extensions or compiled code
3. **Minimal Dependencies**: Only requires `sympy` for core functionality
4. **Wheel Format**: Distributed as a universal wheel (`py3-none-any`)

## Building the Wheel

To build the wheel yourself:

```bash
# Install build tools
pip install build

# Build the wheel
python -m build --wheel

# The wheel will be created in dist/pythonck-0.1.0-py3-none-any.whl
```

## Example Quarto Document

Here's a complete example for a Quarto document:

```markdown
---
title: "PythonCK in Quarto"
format: html
jupyter: python3
---

```{python}
import micropip
await micropip.install("pythonck")  # Replace with your wheel URL
```

```{python}
# Import and use pythonck
import pytensor
from tensor_transforms import TensorTransformParser

# Create a tensor coordinate
coord = pytensor.make_tensor_coordinate([128, 256])
print(f"Created tensor with shape: {coord.get_lengths()}")

# Parse a transform
parser = TensorTransformParser()
transform = parser.parse_expression("Transform<PassThrough<0, 1>>")
print(f"Parsed transform: {transform}")
```
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure:

1. The package was installed successfully
2. All dependencies are available in Pyodide
3. The correct module names are used

### SymPy Issues

If SymPy-related functions fail:

```python
# Check SymPy installation
import sympy
print(f"SymPy version: {sympy.__version__}")
```

### Memory Limitations

For large tensor operations in Pyodide:

- Use smaller tensor sizes for demonstrations
- Consider pagination for large visualizations
- Monitor browser memory usage

## API Reference

For detailed API documentation, see:

- `pytensor/README.md` - Core tensor operations
- `pytensor/IMPLEMENTATION_STATUS.md` - Implementation status
- Module docstrings and type hints throughout the codebase 