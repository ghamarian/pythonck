"""
Python implementation of shuffle_tile.hpp from Composable Kernels.

This module provides utilities for shuffling data between distributed tensors
with different layouts but compatible distributions.
"""

from typing import TypeVar
import numpy as np
from .static_distributed_tensor import StaticDistributedTensor
from .tile_distribution import TileDistribution


def shuffle_tile(out_tensor: StaticDistributedTensor, 
                 in_tensor: StaticDistributedTensor) -> None:
    """
    Shuffle data from input tensor to output tensor.
    
    This function transfers data between distributed tensors that may have
    different Y-dimension orderings but must have compatible distributions
    (same R, H, P dimensions).
    
    Args:
        out_tensor: Output distributed tensor
        in_tensor: Input distributed tensor
        
    Raises:
        ValueError: If tensors have incompatible distributions
    """
    # Get distributions
    in_dist = in_tensor.tile_distribution
    out_dist = out_tensor.tile_distribution
    
    # Get encoding details
    in_encoding = in_dist.encoding
    out_encoding = out_dist.encoding
    
    # Check compatibility - must have same R, H, P dimensions
    if (in_encoding.rs_lengths != out_encoding.rs_lengths or
        in_encoding.hs_lengthss != out_encoding.hs_lengthss or
        in_encoding.ps_to_rhss_major != out_encoding.ps_to_rhss_major or
        in_encoding.ps_to_rhss_minor != out_encoding.ps_to_rhss_minor):
        raise ValueError("Input and output tensors have incompatible distributions")
    
    # Type conversion if needed
    if in_tensor.data_type != out_tensor.data_type:
        # Convert input data to output type
        in_data = in_tensor.thread_buffer.astype(out_tensor.data_type)
    else:
        in_data = in_tensor.thread_buffer
    
    # Get Y-dimension descriptors
    y_in_desc = in_dist.ys_to_d_descriptor
    y_out_desc = out_dist.ys_to_d_descriptor
    
    # Get Y lengths
    y_lengths = y_in_desc.get_lengths()
    ndim_y = len(y_lengths)
    
    # For same layout, just copy data directly
    if (in_encoding.ys_to_rhs_major == out_encoding.ys_to_rhs_major and
        in_encoding.ys_to_rhs_minor == out_encoding.ys_to_rhs_minor):
        # Same layout, direct copy
        out_tensor.thread_buffer[:] = in_data[:]
        return
    
    # Build Y dimension mapping from output to input
    y_dim_out_to_in = {}
    for y_out in range(ndim_y):
        rh_major_out = out_encoding.ys_to_rhs_major[y_out]
        rh_minor_out = out_encoding.ys_to_rhs_minor[y_out]
        
        # Find corresponding input dimension
        for y_in in range(ndim_y):
            if (in_encoding.ys_to_rhs_major[y_in] == rh_major_out and
                in_encoding.ys_to_rhs_minor[y_in] == rh_minor_out):
                y_dim_out_to_in[y_out] = y_in
                break
    
    # Simple implementation: iterate over all elements
    total_elements = 1
    for length in y_lengths:
        total_elements *= length
    
    # For each element in the output
    for out_linear_idx in range(total_elements):
        # Convert linear index to multi-dimensional index for output
        out_idx = []
        temp = out_linear_idx
        for i in range(ndim_y - 1, -1, -1):
            out_idx.insert(0, temp % y_lengths[i])
            temp //= y_lengths[i]
        
        # Map to input indices
        in_idx = [0] * ndim_y
        for out_dim, in_dim in y_dim_out_to_in.items():
            in_idx[in_dim] = out_idx[out_dim]
        
        # Calculate offsets
        in_offset = y_in_desc.calculate_offset(in_idx)
        out_offset = y_out_desc.calculate_offset(out_idx)
        
        # Copy data
        out_tensor.thread_buffer[out_offset] = in_data[in_offset]


def shuffle_tile_in_thread(out_tensor: StaticDistributedTensor,
                          in_tensor: StaticDistributedTensor) -> None:
    """
    Thread-local implementation of shuffle_tile.
    
    This is an optimized version that processes data in a thread-local manner,
    suitable for GPU-style parallel execution.
    
    Args:
        out_tensor: Output distributed tensor
        in_tensor: Input distributed tensor
    """
    # For Python, this is the same as shuffle_tile since we don't have
    # actual thread-local optimizations
    shuffle_tile(out_tensor, in_tensor) 