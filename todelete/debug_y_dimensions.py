import numpy as np
from pytensor.tile_window import *
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_distribution import *
from pytensor.tensor_descriptor import *

# Create RMSNorm setup from test
tensor_shape = [2, 2, 2, 2]
data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
tensor_view = make_naive_tensor_view(data, tensor_shape, [8, 4, 2, 1])

encoding = make_tile_distribution_encoding(
    rs_lengths=[],
    hs_lengthss=[[1], [1], [1], [1]],
    ps_to_rhss_major=[[]],
    ps_to_rhss_minor=[[]],
    ys_to_rhs_major=[1, 2, 3, 4],
    ys_to_rhs_minor=[0, 0, 0, 0]
)

adaptor = TensorAdaptor(
    transforms=[PassThroughTransform(4)],
    lower_dimension_hidden_idss=[[0, 1, 2, 3]],
    upper_dimension_hidden_idss=[[0, 1, 2, 3]],
    bottom_dimension_hidden_ids=[0, 1, 2, 3],
    top_dimension_hidden_ids=[4, 0, 1, 2, 3]
)

ys_to_d_descriptor = make_naive_tensor_descriptor(tensor_shape, [8, 4, 2, 1])

dist = make_tile_distribution(
    ps_ys_to_xs_adaptor=adaptor,
    ys_to_d_descriptor=ys_to_d_descriptor,
    encoding=encoding
)

print(f"=== Y Dimension Analysis for RMSNorm Example ===")
print(f"ndim_y: {dist.ndim_y}")
print(f"Y dimension lengths: {dist.ys_to_d_descriptor.get_lengths()}")
print(f"Y vector lengths: {dist.get_y_vector_lengths()}")
print(f"Y vector strides: {dist.get_y_vector_strides()}")

# Create tile window and analyze traits
window = TileWindowWithStaticDistribution(
    bottom_tensor_view=tensor_view,
    window_lengths=tensor_shape,
    window_origin=[0, 0, 0, 0],
    tile_distribution=dist
)

print(f"\n=== LoadStoreTraits Y Usage ===")
print(f"LoadStoreTraits ndim_y: {window.traits.ndim_y}")
print(f"LoadStoreTraits vector_dim_y: {window.traits.vector_dim_y}")
print(f"LoadStoreTraits scalar_per_vector: {window.traits.scalar_per_vector}")
print(f"LoadStoreTraits scalars_per_access: {window.traits.scalars_per_access}")
print(f"LoadStoreTraits num_access: {window.traits.num_access}")

print(f"\n=== Space Filling Curve Y Usage ===")
sfc = window.traits.sfc_ys
print(f"SFC tensor_lengths (Y lengths): {sfc.tensor_lengths}")
print(f"SFC dim_access_order: {sfc.dim_access_order}")
print(f"SFC scalars_per_access: {sfc.scalars_per_access}")
print(f"SFC access_lengths: {sfc.access_lengths}")

print(f"\n=== Access Pattern Analysis (Y indices) ===")
# Show all access patterns for the 4D case
for i in range(window.traits.num_access):
    y_indices = window.traits.get_y_indices(i)
    access_info = window.traits.get_vectorized_access_info(i)
    print(f"Access {i}: Y_indices={y_indices}, base_indices={access_info['base_indices']}")
    print(f"  -> vector_indices={access_info['vector_indices']}")
    
    # Show how Y indices map to D (linearized) offset
    for j, idx_ys in enumerate(access_info['vector_indices']):
        d_offset = dist.ys_to_d_descriptor.calculate_offset(idx_ys)
        print(f"     vector_element_{j}: Y{idx_ys} -> D_offset={d_offset}")
    print()

print(f"\n=== Load/Store Y Usage ===")
# Show how load_into uses Y dimensions
distributed_tensor = window.load()
print(f"Distributed tensor size: {distributed_tensor.get_num_of_elements()}")
print(f"Distributed tensor data: {distributed_tensor.thread_buffer}")

# Create mapping from Y coordinates to tensor coordinates
print(f"\n=== Y to Tensor Coordinate Mapping ===")
for y0 in range(2):
    for y1 in range(2):
        for y2 in range(2):
            for y3 in range(2):
                y_indices = [y0, y1, y2, y3]
                d_offset = dist.ys_to_d_descriptor.calculate_offset(y_indices)
                tensor_coord = [window.window_origin[i] + y_indices[i] for i in range(4)]
                tensor_value = data[tuple(tensor_coord)]
                distributed_value = distributed_tensor.thread_buffer[d_offset]
                print(f"Y{y_indices} -> tensor{tensor_coord} = {tensor_value}, D[{d_offset}] = {distributed_value}") 