"""
Examples of tile_distribution_encoding for the Composable Kernels Visualizer.

This module contains predefined examples of tile_distribution_encoding structures
that can be used with the visualizer.
"""

# Dictionary of all examples with descriptive names as keys and code snippets as values
EXAMPLES = {
    "Basic 16x4 Threads": """
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
    """,
    
    "Variable-Based Template": """
    tile_distribution_encoding<
        sequence<1>,                            // 0 R
        tuple<sequence<Nr_y, Nr_p, Nw>,         // H
              sequence<Kr_y, Kr_p, Kw, Kv>>,    // H
        tuple<sequence<1, 2>,                   // p major
              sequence<2, 1>>,                  // p minor
        tuple<sequence<1, 1>,                   // p minor
              sequence<2, 2>>,                  // p minor
        sequence<1, 2, 2>,                      // Y major
        sequence<0, 0, 3>>{}                    // y minor
    """,
    
    "Real-World Example (RMSNorm)": """
    // From include/ck_tile/ops/add_rmsnorm2d_rdquant/pipeline
    tile_distribution_encoding<
        sequence<>,                             // Empty R
        tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
              sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
        tuple<sequence<1, 2>, sequence<1, 2>>,
        tuple<sequence<1, 1>, sequence<2, 2>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 3, 0, 3>>{}
    """,
    
    "Complex Distribution": """
    tile_distribution_encoding<
        sequence<1>,                            // 0 R
        tuple<sequence<16, 4>,                  // H (X0)
              sequence<16, 4, 4>>,              // H (X1)
        tuple<sequence<1, 1>,                   // p major
              sequence<2, 2>>,                  // p minor
        tuple<sequence<0, 1>,                   // p minor
              sequence<0, 0>>,                  // p minor
        sequence<1, 1, 2>,                      // Y major
        sequence<1, 0, 1>>{}                    // y minor
    """,
    
    "Custom Y Dimension Mapping": """
    tile_distribution_encoding<
        sequence<1>,                            // 0 R
        tuple<sequence<M0, M1, M2>,             // H
              sequence<K0, K1>>,                // H
        tuple<sequence<1>,                      // p major
              sequence<1,2>>,                   // p minor
        tuple<sequence<1>,                      // p minor
              sequence<2, 0>>,                  // p minor
        sequence<1, 2>,                         // Y major
        sequence<0, 1>>{}                       // y minor
    """
}

# Default variable values for each example
DEFAULT_VARIABLES = {
    "Variable-Based Template": {
        "Nr_y": 4,
        "Nr_p": 4,
        "Nw": 8,
        "Kr_y": 4,
        "Kr_p": 8,
        "Kw": 8,
        "Kv": 4
    },
    "Real-World Example (RMSNorm)": {
        "S::Repeat_M": 4,
        "S::WarpPerBlock_M": 2,
        "S::ThreadPerWarp_M": 8,
        "S::Vector_M": 4,
        "S::Repeat_N": 4,
        "S::WarpPerBlock_N": 2,
        "S::ThreadPerWarp_N": 8,
        "S::Vector_N": 4
    },
    "Custom Y Dimension Mapping": {
        "M0": 2,
        "M1": 8,
        "M2": 4,
        "K0": 4,
        "K1": 4
    }
}

def get_examples():
    """Returns the dictionary of all examples."""
    return EXAMPLES

def get_default_variables(example_name):
    """
    Returns default variable values for a specific example.
    
    Args:
        example_name: Name of the example to get variables for
        
    Returns:
        Dictionary of variable names to values, or empty dict if none defined
    """
    return DEFAULT_VARIABLES.get(example_name, {}) 