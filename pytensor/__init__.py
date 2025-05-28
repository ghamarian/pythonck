"""
PyTensor - Python Implementation of Composable Kernels Tensor Operations

This package provides educational Python implementations of tensor operations
from the Composable Kernels library.
"""

# Buffer operations
from .buffer_view import (
    BufferView,
    AddressSpaceEnum,
    MemoryOperationEnum,
    AmdBufferCoherenceEnum,
    make_buffer_view
)

# Tensor coordinates
from .tensor_coordinate import (
    MultiIndex,
    TensorAdaptorCoordinate,
    TensorCoordinate,
    make_tensor_adaptor_coordinate,
    make_tensor_coordinate,
    move_tensor_adaptor_coordinate,
    move_tensor_coordinate,
    adaptor_coordinate_is_valid,
    adaptor_coordinate_is_valid_assuming_top_index_is_valid,
    coordinate_has_valid_offset,
    coordinate_has_valid_offset_assuming_top_index_is_valid
)

# Tensor descriptors and transforms
from .tensor_descriptor import (
    Transform,
    EmbedTransform,
    UnmergeTransform,
    OffsetTransform,
    PassThroughTransform,
    PadTransform,
    MergeTransform,
    ReplicateTransform,
    TensorAdaptor,
    TensorDescriptor,
    make_naive_tensor_descriptor,
    make_naive_tensor_descriptor_packed,
    make_naive_tensor_descriptor_aligned
)

# Tensor adaptors
from .tensor_adaptor import (
    make_single_stage_tensor_adaptor,
    transform_tensor_adaptor,
    chain_tensor_adaptors,
    chain_tensor_adaptors_multi,
    make_identity_adaptor,
    make_transpose_adaptor
)

# Tensor views
from .tensor_view import (
    TensorView,
    NullTensorView,
    make_tensor_view,
    make_naive_tensor_view,
    make_naive_tensor_view_packed,
    transform_tensor_view
)

# Tile distribution
from .tile_distribution import (
    TileDistributedSpan,
    TileDistributedIndex,
    TileDistributionEncoding,
    TileDistribution,
    make_tile_distributed_span,
    make_tile_distributed_index,
    make_tile_distribution_encoding,
    make_tile_distribution
)

# Tile distribution encoding
from .tile_distribution_encoding import (
    TileDistributionEncodingDetail,
    TileDistributionEncoding as TileDistributionEncodingEnhanced,
    make_embed_tile_distribution_encoding,
    make_reduce_tile_distribution_encoding
)

# Static distributed tensor
from .static_distributed_tensor import (
    StaticDistributedTensor,
    make_static_distributed_tensor
)

# Tile window
from .tile_window import (
    TileWindowWithStaticDistribution,
    TileWindowWithStaticLengths,
    make_tile_window,
    move_tile_window
)

# Store tile operations
from .store_tile import (
    store_tile,
    store_tile_raw
)

# Update tile operations
from .update_tile import (
    update_tile,
    update_tile_raw
)

# Sweep tile operations
from .sweep_tile import (
    sweep_tile_span,
    sweep_tile_uspan,
    sweep_tile,
    TileSweeper,
    make_tile_sweeper
)

# Shuffle tile operations
from .shuffle_tile import (
    shuffle_tile,
    shuffle_tile_in_thread
)

# Tile scatter/gather operations
from .tile_scatter_gather import (
    TileScatterGather,
    make_tile_scatter_gather
)

# Tile window linear
from .tile_window_linear import (
    TileWindowLinear,
    make_tile_window_linear
)

__all__ = [
    # Buffer operations
    'BufferView',
    'AddressSpaceEnum',
    'MemoryOperationEnum',
    'AmdBufferCoherenceEnum',
    'make_buffer_view',
    
    # Tensor coordinates
    'MultiIndex',
    'TensorAdaptorCoordinate',
    'TensorCoordinate',
    'make_tensor_adaptor_coordinate',
    'make_tensor_coordinate',
    'move_tensor_adaptor_coordinate',
    'move_tensor_coordinate',
    'adaptor_coordinate_is_valid',
    'adaptor_coordinate_is_valid_assuming_top_index_is_valid',
    'coordinate_has_valid_offset',
    'coordinate_has_valid_offset_assuming_top_index_is_valid',
    
    # Tensor descriptors and transforms
    'Transform',
    'EmbedTransform',
    'UnmergeTransform',
    'OffsetTransform',
    'PassThroughTransform',
    'PadTransform',
    'MergeTransform',
    'ReplicateTransform',
    'TensorAdaptor',
    'TensorDescriptor',
    'make_naive_tensor_descriptor',
    'make_naive_tensor_descriptor_packed',
    'make_naive_tensor_descriptor_aligned',
    
    # Tensor adaptors
    'make_single_stage_tensor_adaptor',
    'transform_tensor_adaptor',
    'chain_tensor_adaptors',
    'chain_tensor_adaptors_multi',
    'make_identity_adaptor',
    'make_transpose_adaptor',
    
    # Tensor views
    'TensorView',
    'NullTensorView',
    'make_tensor_view',
    'make_naive_tensor_view',
    'make_naive_tensor_view_packed',
    'transform_tensor_view',
    
    # Tile distribution
    'TileDistributedSpan',
    'TileDistributedIndex',
    'TileDistributionEncoding',
    'TileDistribution',
    'make_tile_distributed_span',
    'make_tile_distributed_index',
    'make_tile_distribution_encoding',
    'make_tile_distribution',
    
    # Tile distribution encoding
    'TileDistributionEncodingDetail',
    'TileDistributionEncodingEnhanced',
    'make_embed_tile_distribution_encoding',
    'make_reduce_tile_distribution_encoding',
    
    # Static distributed tensor
    'StaticDistributedTensor',
    'make_static_distributed_tensor',
    
    # Tile window
    'TileWindowWithStaticDistribution',
    'TileWindowWithStaticLengths',
    'make_tile_window',
    'move_tile_window',
    
    # Store tile operations
    'store_tile',
    'store_tile_raw',
    
    # Update tile operations
    'update_tile',
    'update_tile_raw',
    
    # Sweep tile operations
    'sweep_tile_span',
    'sweep_tile_uspan',
    'sweep_tile',
    'TileSweeper',
    'make_tile_sweeper',
    
    # Shuffle tile operations
    'shuffle_tile',
    'shuffle_tile_in_thread',
    
    # Tile scatter/gather operations
    'TileScatterGather',
    'make_tile_scatter_gather',

    # Tile window linear
    'TileWindowLinear',
    'make_tile_window_linear',
]

__version__ = '0.1.0' 