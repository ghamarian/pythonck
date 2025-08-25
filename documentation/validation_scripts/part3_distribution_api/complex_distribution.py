from pytensor.tile_distribution import make_static_tile_distribution, make_tile_distribution_encoding
import numpy as np
from pytensor.tensor_view import make_naive_tensor_view_packed
from pytensor.sweep_tile import sweep_tile
from pytensor.partition_simulation import set_global_thread_position
from pytensor.tile_window import make_tile_window
import itertools

MAX_WARPS = 2
MAX_THREADS = 2


# Create a tile distribution encoding
# This defines how a tensor is distributed across threads
encoding = make_tile_distribution_encoding(
        rs_lengths=[],
        hs_lengthss=[
            [2, 2, 2, 2],
            [2, 2, 2, 2]
        ],
        ps_to_rhss_major=[[1, 2], [1, 2]],
        ps_to_rhss_minor=[[1, 1], [2, 2]],
        ys_to_rhs_major=[1, 1, 2, 2],
        ys_to_rhs_minor=[0, 3, 0, 3]
    )

# Create the tile distribution from the encoding
distribution = make_static_tile_distribution(encoding)

print(f"\n-Tile distribution created:")
print(f" X dimensions: {distribution.ndim_x}")
print(f"- Y dimensions: {distribution.ndim_y}")
print(f"- P dimensions: {distribution.ndim_p}")
print(f"- X lengths: {distribution.get_lengths()}")


x0_size = np.prod(distribution.encoding.hs_lengthss[0])  # 2 * 2
x1_size = np.prod(distribution.encoding.hs_lengthss[1])  # 2 * 2

global_shape = [x0_size + 5, x1_size + 5]  # Add margin for positioning
global_data = np.zeros(global_shape, dtype=np.int32)
visualization_canvas = np.zeros(global_shape, dtype=np.int32)

# Create a simple, readable pattern: value = row*100 + col
# Using 100 multiplier to make the row changes more visible
for i in range(global_shape[0]):
    for j in range(global_shape[1]):
          global_data[i, j] = float(i * 100 + j)


global_view = make_naive_tensor_view_packed(
        data=global_data.flatten(),
        lengths=global_shape
    )
    
window_lengths = [x0_size, x1_size]

print(f"Global data:\n")
for i in range(global_shape[0]):
    for j in range(global_shape[1]):
        print(f"{global_data[i, j]:.0f}", end="\t")
    print()
print()
    

def process_element(*y_indices):
    """Process a single element from the sweep."""
    global collected_values
    
    value = distributed_tensor[y_indices]
    collected_values.append(value)


for warp, thread in itertools.product(range(MAX_WARPS), range(MAX_THREADS)):

    set_global_thread_position(warp, thread)

    window_origin = [0, 0]  # Small offset from origin
    tile_window = make_tile_window(
            tensor_view=global_view,
            window_lengths=window_lengths,
            origin=window_origin,
            tile_distribution=distribution
        )

    distributed_tensor = tile_window.load(oob_conditional_check=True)

    collected_values = []
    warp_id, thread_id = distribution.get_partition_index()
    print(f"Partition index: (warp={warp_id}, thread={thread_id})")

    sweep_tile(distributed_tensor, process_element)
    for v in collected_values:
        print(f"{v:.0f}", end="\t")
    print()
    