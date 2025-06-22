#!/usr/bin/env python3
"""
Debug script to show what values are stored in window_adaptor_coord and bottom_tensor_coord
"""

import numpy as np
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_distribution import make_tile_distribution, make_tile_distribution_encoding
from pytensor.tensor_descriptor import TensorAdaptor, PassThroughTransform, make_naive_tensor_descriptor
from pytensor.tile_window import TileWindowWithStaticDistribution

def debug_coordinate_values():
    """Show what values are stored in the coordinate objects."""
    
    # Create a simple 2x2 tensor
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    tensor_view = make_naive_tensor_view(data, [2, 2], [2, 1])
    
    # Create a simple tile distribution
    encoding = make_tile_distribution_encoding(
        rs_lengths=[],
        hs_lengthss=[[2], [2]],  # 2 X dimensions
        ps_to_rhss_major=[[]],   # Trivial partition
        ps_to_rhss_minor=[[]],
        ys_to_rhs_major=[1, 2], # Y-dims map to H-dims
        ys_to_rhs_minor=[0, 0]
    )
    
    # Simple adaptor: Y(2) -> X(2)
    adaptor = TensorAdaptor(
        transforms=[PassThroughTransform(2)],
        lower_dimension_hidden_idss=[[0, 1]], # X-dims
        upper_dimension_hidden_idss=[[1, 2]], # Y-dims
        bottom_dimension_hidden_ids=[0, 1],
        top_dimension_hidden_ids=[3, 1, 2] # P-dim, Y-dims
    )
    
    descriptor = make_naive_tensor_descriptor([2, 2], [2, 1])
    
    dist = make_tile_distribution(
        ps_ys_to_xs_adaptor=adaptor,
        ys_to_d_descriptor=descriptor,
        encoding=encoding
    )
    
    # Create window
    window = TileWindowWithStaticDistribution(
        bottom_tensor_view=tensor_view,
        window_lengths=[2, 2],
        window_origin=[0, 0],
        tile_distribution=dist
    )
    
    print("=== COORDINATE DEBUG ===")
    print(f"Number of coordinate bundles: {len(window.pre_computed_coords)}")
    
    for i, (window_adaptor_coord, bottom_tensor_coord) in enumerate(window.pre_computed_coords):
        print(f"\n--- Coordinate Bundle {i} ---")
        
        print(f"window_adaptor_coord:")
        print(f"  Type: {type(window_adaptor_coord)}")
        print(f"  ndim_hidden: {window_adaptor_coord.ndim_hidden}")
        print(f"  bottom_dimension_hidden_ids: {window_adaptor_coord.bottom_dimension_hidden_ids}")
        print(f"  top_dimension_hidden_ids: {window_adaptor_coord.top_dimension_hidden_ids}")
        print(f"  idx_hidden: {window_adaptor_coord.idx_hidden.to_list()}")
        print(f"  get_top_index(): {window_adaptor_coord.get_top_index().to_list()}")
        print(f"  get_bottom_index(): {window_adaptor_coord.get_bottom_index().to_list()}")
        
        print(f"\nbottom_tensor_coord:")
        print(f"  Type: {type(bottom_tensor_coord)}")
        print(f"  ndim_hidden: {bottom_tensor_coord.ndim_hidden}")
        print(f"  bottom_dimension_hidden_ids: {bottom_tensor_coord.bottom_dimension_hidden_ids}")
        print(f"  top_dimension_hidden_ids: {bottom_tensor_coord.top_dimension_hidden_ids}")
        print(f"  idx_hidden: {bottom_tensor_coord.idx_hidden.to_list()}")
        print(f"  get_index(): {bottom_tensor_coord.get_index().to_list()}")
        print(f"  get_offset(): {bottom_tensor_coord.get_offset()}")
        
        # Show what this coordinate points to in the actual tensor
        tensor_idx = bottom_tensor_coord.get_index().to_list()
        print(f"  Points to tensor element at {tensor_idx}: {data[tuple(tensor_idx)]}")

if __name__ == "__main__":
    debug_coordinate_values() 