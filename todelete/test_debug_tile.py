import numpy as np
from pytensor.tile_window import make_tile_window, TileWindowWithStaticDistribution
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_distribution import make_tile_distribution, make_tile_distribution_encoding
from pytensor.tensor_descriptor import TensorAdaptor, PassThroughTransform, make_naive_tensor_descriptor
from pytensor.static_distributed_tensor import StaticDistributedTensor

def test_debug_tile_window():
    """Debug test for tile window with Y dimensions."""
    # Create a larger 4D tensor (4x4x2x2) to show multiple tiles
    tensor_shape = [4, 4, 2, 2]
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    print(f"Original tensor shape: {tensor_shape}")
    print(f"Original data shape: {data.shape}")
    
    # Create tensor view
    tensor_view = make_naive_tensor_view(data, tensor_shape, [16, 4, 2, 1])
    
    # Create distribution with P=2 (2 tiles) and Y=4
    encoding = make_tile_distribution_encoding(
        rs_lengths=[],
        hs_lengthss=[[2], [2], [1], [1]],  # 4 X-dims, with P=2 for first two dims
        ps_to_rhss_major=[[1], [1]],  # P dimensions
        ps_to_rhss_minor=[[0], [0]],
        ys_to_rhs_major=[1, 2, 3, 4],  # Y-dims map to H-dims (1-based)
        ys_to_rhs_minor=[0, 0, 0, 0]
    )
    
    # Adaptor: P(2) + Y(4) -> X(4)
    adaptor = TensorAdaptor(
        transforms=[PassThroughTransform(4)],  # Maps 4 Y-dims to 4 X-dims
        lower_dimension_hidden_idss=[[0, 1, 2, 3]],  # Hidden IDs for X-dims
        upper_dimension_hidden_idss=[[0, 1, 2, 3]],  # Hidden IDs for Y-dims
        bottom_dimension_hidden_ids=[0, 1, 2, 3],  # 4D bottom (X)
        top_dimension_hidden_ids=[4, 5, 0, 1, 2, 3]   # P-dims (4,5), Y-dims(0,1,2,3)
    )
    
    # Descriptor for D-dimensions (matches Y dimensions)
    ys_to_d_descriptor = make_naive_tensor_descriptor([2, 2, 2, 2], [8, 4, 2, 1])  # Each tile is 2x2x2x2
    
    dist = make_tile_distribution(
        ps_ys_to_xs_adaptor=adaptor,
        ys_to_d_descriptor=ys_to_d_descriptor,
        encoding=encoding
    )
    
    # Create windows for two different tiles
    print("\nCreating windows for two tiles:")
    
    # Tile 1: [0:2, 0:2, :, :]
    window1 = TileWindowWithStaticDistribution(
        bottom_tensor_view=tensor_view,
        window_lengths=[2, 2, 2, 2],  # Each tile is 2x2x2x2
        window_origin=[0, 0, 0, 0],
        tile_distribution=dist
    )
    
    # Tile 2: [2:4, 2:4, :, :]
    window2 = TileWindowWithStaticDistribution(
        bottom_tensor_view=tensor_view,
        window_lengths=[2, 2, 2, 2],
        window_origin=[2, 2, 0, 0],
        tile_distribution=dist
    )
    
    # Load data from both tiles
    print("\nLoading data from Tile 1:")
    distributed_tensor1 = window1.load()
    print(f"Number of accesses in Tile 1: {window1.get_num_of_access()}")
    
    print("\nLoading data from Tile 2:")
    distributed_tensor2 = window2.load()
    print(f"Number of accesses in Tile 2: {window2.get_num_of_access()}")
    
    # Print the traits information (same for both tiles)
    print("\nTraits information (same for both tiles):")
    print(f"Vector dimension (Y3): {window1.traits.vector_dim_y}")
    print(f"Scalars per vector: {window1.traits.scalar_per_vector}")
    print(f"Number of Y dimensions: {window1.traits.ndim_y}")
    
    # Print the space-filling curve information
    print("\nSpace-filling curve information:")
    print(f"Access lengths: {window1.traits.sfc_ys.access_lengths}")
    print(f"Dimension access order: {window1.traits.sfc_ys.dim_access_order}")
    
    # Print example access patterns for Tile 1
    print("\nExample access patterns for Tile 1:")
    for i in range(min(3, window1.get_num_of_access())):
        access_info = window1.traits.get_vectorized_access_info(i)
        print(f"\nAccess {i}:")
        print(f"Base indices: {access_info['base_indices']}")
        print(f"Vector indices: {access_info['vector_indices']}")
        print(f"Vector dimension: {access_info['vector_dim']}")
        print(f"Vector size: {access_info['vector_size']}")
    
    # Print example access patterns for Tile 2
    print("\nExample access patterns for Tile 2:")
    for i in range(min(3, window2.get_num_of_access())):
        access_info = window2.traits.get_vectorized_access_info(i)
        print(f"\nAccess {i}:")
        print(f"Base indices: {access_info['base_indices']}")
        print(f"Vector indices: {access_info['vector_indices']}")
        print(f"Vector dimension: {access_info['vector_dim']}")
        print(f"Vector size: {access_info['vector_size']}")

if __name__ == "__main__":
    test_debug_tile_window() 