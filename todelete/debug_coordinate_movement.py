#!/usr/bin/env python3

import numpy as np
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_distribution import make_tile_distribution, make_tile_distribution_encoding
from pytensor.tensor_descriptor import TensorAdaptor, PassThroughTransform, make_naive_tensor_descriptor
from pytensor.tile_window import TileWindowWithStaticDistribution

def debug_coordinate_movement():
    """Debug coordinate movement in the C++ style implementation."""
    
    # Create the same setup as the test
    data = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]], dtype=np.float32)
    tensor_view = make_naive_tensor_view(data, [4, 4], [4, 1])

    encoding = make_tile_distribution_encoding(
        rs_lengths=[],
        hs_lengthss=[[2], [2]],
        ps_to_rhss_major=[[]],
        ps_to_rhss_minor=[[]],
        ys_to_rhs_major=[1, 2], 
        ys_to_rhs_minor=[0, 0]
    )

    adaptor = TensorAdaptor(
        transforms=[PassThroughTransform(2)],
        lower_dimension_hidden_idss=[[0, 1]],
        upper_dimension_hidden_idss=[[1, 2]],
        bottom_dimension_hidden_ids=[0, 1],
        top_dimension_hidden_ids=[3, 1, 2]
    )

    descriptor = make_naive_tensor_descriptor([2, 2], [2, 1])

    dist = make_tile_distribution(
        ps_ys_to_xs_adaptor=adaptor,
        ys_to_d_descriptor=descriptor,
        encoding=encoding
    )

    window = TileWindowWithStaticDistribution(
        bottom_tensor_view=tensor_view,
        window_lengths=[2, 2],
        window_origin=[1, 1],
        tile_distribution=dist
    )
    
    print("=== DEBUG: Y to D Mapping ===")
    ys_to_d_desc = window.tile_distribution.ys_to_d_descriptor
    print(f"Y-to-D descriptor lengths: {ys_to_d_desc.get_lengths()}")
    print(f"Y-to-D element space size: {ys_to_d_desc.get_element_space_size()}")
    
    print("\n=== Y to D Offset Mapping ===")
    for i in range(window.traits.sfc_ys.get_num_of_access()):
        y_indices = window.traits.sfc_ys.get_index(i)
        access_info = window.traits.get_vectorized_access_info(i)
        
        print(f"Access {i}: Y = {y_indices}")
        for j, vector_y in enumerate(access_info['vector_indices']):
            d_offset = ys_to_d_desc.calculate_offset(vector_y)
            print(f"  Vector element {j}: Y{vector_y} -> D{d_offset}")
    
    print("\n=== DEBUG: Space Filling Curve ===")
    print(f"Y tensor lengths: {window.traits.sfc_ys.tensor_lengths}")
    print(f"Access lengths: {window.traits.sfc_ys.access_lengths}")
    print(f"Dim access order: {window.traits.sfc_ys.dim_access_order}")
    print(f"Scalars per access: {window.traits.sfc_ys.scalars_per_access}")
    print(f"Total accesses: {window.traits.sfc_ys.get_num_of_access()}")
    
    print("\n=== All Y Indices ===")
    for i in range(window.traits.sfc_ys.get_num_of_access()):
        y_indices = window.traits.sfc_ys.get_index(i)
        print(f"Access {i}: Y = {y_indices}")
    
    print("\n=== Expected Pattern ===")
    print("For 2x2 tile, expected Y access pattern should be:")
    print("  Access 0: Y = [0, 0] -> X = [0, 0] -> tensor = [1, 1] ✓")
    print("  Access 1: Y = [0, 1] -> X = [0, 1] -> tensor = [1, 2] ✓")
    print("  Access 2: Y = [1, 0] -> X = [1, 0] -> tensor = [2, 1] ✓")
    print("  Access 3: Y = [1, 1] -> X = [1, 1] -> tensor = [2, 2] ✓")
    
    print("\n=== DEBUG: Coordinate Movement ===")
    print(f"Window origin: {window.window_origin}")
    print(f"Number of accesses: {window.traits.num_access}")
    print(f"Access per coord: {window.traits.num_access // window.num_coord}")
    
    # Debug precomputed coordinates
    print("\n=== Precomputed Coordinates ===")
    for i, (window_adaptor_coord, bottom_tensor_coord) in enumerate(window.pre_computed_coords):
        print(f"Coord bundle {i}:")
        print(f"  window_adaptor_coord.get_top_index(): {window_adaptor_coord.get_top_index()}")
        print(f"  window_adaptor_coord.get_bottom_index(): {window_adaptor_coord.get_bottom_index()}")
        print(f"  bottom_tensor_coord.get_index(): {bottom_tensor_coord.get_index().to_list()}")
    
    print("\n=== Access Pattern ===")
    # Manually trace through the traversal
    num_access_per_coord = window.traits.num_access // window.num_coord
    
    for i_coord in range(window.num_coord):
        print(f"\nCoordinate Bundle {i_coord}:")
        
        # Get fresh copies like the C++ code
        window_adaptor_coord = window.pre_computed_coords[i_coord][0].copy()
        bottom_tensor_coord = window.pre_computed_coords[i_coord][1].copy()
        
        print(f"  Initial: bottom_coord = {bottom_tensor_coord.get_index().to_list()}")
        
        for i_coord_access in range(num_access_per_coord):
            i_access = i_coord * num_access_per_coord + i_coord_access
            
            # Get access info
            access_info = window.traits.get_vectorized_access_info(i_access)
            print(f"  Access {i_access}: Y indices = {access_info['base_indices']}, "
                  f"bottom_coord = {bottom_tensor_coord.get_index().to_list()}")
            
            # Move to next (if not last)
            if i_coord_access != (num_access_per_coord - 1):
                idx_diff_ys = window.traits.sfc_ys.get_forward_step(i_access)
                ndim_p = window.tile_distribution.ndim_p
                idx_diff_ps_ys = [0] * ndim_p + idx_diff_ys
                
                print(f"    Forward step: idx_diff_ys = {idx_diff_ys}, idx_diff_ps_ys = {idx_diff_ps_ys}")
                
                # Move coordinates
                window._move_window_adaptor_and_bottom_tensor_coordinate(
                    window_adaptor_coord, bottom_tensor_coord, idx_diff_ps_ys
                )
                print(f"    After move: bottom_coord = {bottom_tensor_coord.get_index().to_list()}")

if __name__ == "__main__":
    debug_coordinate_movement() 