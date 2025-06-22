#!/usr/bin/env python3

import sys
sys.path.append('..')

from pytensor.tile_distribution import make_tile_distribution_encoding, make_tile_distribution
from pytensor.tensor_descriptor import TensorAdaptor, make_naive_tensor_descriptor, EmbedTransform, PassThroughTransform
from pytensor.tile_window import LoadStoreTraits

# Create the complex distribution
encoding = make_tile_distribution_encoding(
    rs_lengths=[2],
    hs_lengthss=[[4, 3], [2, 2]],
    ps_to_rhss_major=[[1, 1]],
    ps_to_rhss_minor=[[0, 1]],
    ys_to_rhs_major=[1, 1, 1, 1],
    ys_to_rhs_minor=[0, 1, 2, 3]
)

transform1 = PassThroughTransform(1)  # P dimension
transform2 = EmbedTransform([4], [1])  # Y1 -> X1
transform3 = EmbedTransform([3], [1])  # Y2 -> X2
transform4 = EmbedTransform([2], [1])  # Y3 -> X3
transform5 = EmbedTransform([2], [1])  # Y4 -> X4

adaptor = TensorAdaptor(
    transforms=[transform1, transform2, transform3, transform4, transform5],
    lower_dimension_hidden_idss=[[4], [0], [1], [2], [3]],
    upper_dimension_hidden_idss=[[4], [5], [6], [7], [8]],
    bottom_dimension_hidden_ids=[0, 1, 2, 3],
    top_dimension_hidden_ids=[4, 5, 6, 7, 8]
)

descriptor = make_naive_tensor_descriptor([4, 3, 2, 2], [6, 2, 1, 1])

tile_dist = make_tile_distribution(
    ps_ys_to_xs_adaptor=adaptor,
    ys_to_d_descriptor=descriptor,
    encoding=encoding
)

traits = LoadStoreTraits(tile_dist, 'float32')

print(f'Vector lengths: {tile_dist.get_y_vector_lengths()}')
print(f'Vector strides: {tile_dist.get_y_vector_strides()}')
print(f'Vector dim: {traits.vector_dim_y}')
print(f'Scalar per vector: {traits.scalar_per_vector}')
print(f'Num access: {traits.num_access}')

print('Checking each dimension:')
for i in range(4):
    stride = tile_dist.get_y_vector_strides()[i]
    length = tile_dist.get_y_vector_lengths()[i]
    print(f'  Dim {i}: stride={stride}, length={length}, has_stride_1={stride==1}')

# The algorithm should choose the dimension with stride 1 and maximum length
vector_dim_y = 0
scalar_per_vector = 1
for i in range(4):
    stride = tile_dist.get_y_vector_strides()[i]
    length = tile_dist.get_y_vector_lengths()[i]
    if stride == 1 and length > scalar_per_vector:
        scalar_per_vector = length
        vector_dim_y = i

print(f'Expected vector dim: {vector_dim_y}')
print(f'Expected scalar per vector: {scalar_per_vector}')

# Calculate expected num_access
# With scalars_per_access = [1, 1, 2, 1] (vector dim 2 has 2)
# Access lengths = [4/1, 3/1, 2/2, 2/1] = [4, 3, 1, 2]
# Total accesses = 4 * 3 * 1 * 2 = 24
scalars_per_access = [1, 1, 2, 1]  # vector dim 2 has scalar_per_vector=2
access_lengths = []
for i in range(4):
    length = tile_dist.get_y_vector_lengths()[i]
    scalar = scalars_per_access[i]
    access_count = (length + scalar - 1) // scalar
    access_lengths.append(access_count)

expected_num_access = 1
for length in access_lengths:
    expected_num_access *= length

print(f'Access lengths: {access_lengths}')
print(f'Expected num access: {expected_num_access}')
print(f'Actual num access: {traits.num_access}') 