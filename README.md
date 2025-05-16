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