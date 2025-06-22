#!/usr/bin/env python3

from parser import TileDistributionParser
from visualizer import display_raw_encoding, visualize_encoding_structure
import matplotlib.pyplot as plt

# Example from the user
example_code = """
tile_distribution_encoding<
    sequence<1>,                            // 0 R
    tuple<sequence<4, 4>,                   // H (X0)
          sequence<4, 4>>,                  // H (X1)
    tuple<sequence<1>,                      // p major
          sequence<2>>,                     // p minor
    tuple<sequence<1>,                      // p minor
          sequence<0>>,                     // p minor
    sequence<1, 1>,                         // Y major
    sequence<0, 1>>{}                       // y minor
"""

# Parse the example
parser = TileDistributionParser()
encoding = parser.parse_tile_distribution_encoding(example_code)

# Display the raw encoding with interpretation
print(display_raw_encoding(encoding))

# Create and show the visualization
fig = visualize_encoding_structure(encoding)
plt.savefig('encoding_visualization.png')
print("\nVisualization saved to 'encoding_visualization.png'") 