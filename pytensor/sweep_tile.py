"""
Python implementation of sweep_tile.hpp from Composable Kernels.

This module provides utilities for iterating over distributed tensors
with various access patterns.
"""

from typing import Callable, List, Optional, Union, Any, TypeVar, Generic
from dataclasses import dataclass
import itertools
import numpy as np

from .tile_distribution import TileDistributedSpan, TileDistributedIndex, make_tile_distributed_index
from .static_distributed_tensor import StaticDistributedTensor


def sweep_tile_span(span: TileDistributedSpan, 
                   func: Callable[[TileDistributedIndex], None]) -> None:
    """
    Sweep over a span of a distributed tile and apply a function.
    
    Args:
        span: The distributed span to sweep over
        func: Function to apply to each distributed index
              Signature: func(tile_distributed_index) -> None
    """
    # Get the span lengths
    span_lengths = span.partial_lengths
    
    # Generate all possible indices within the span
    ranges = [range(length) for length in span_lengths]
    
    # Iterate over all combinations
    for indices in itertools.product(*ranges):
        # Create distributed index
        dstr_idx = make_tile_distributed_index(list(indices))
        # Apply function
        func(dstr_idx)


def sweep_tile_uspan(span: TileDistributedSpan,
                    func: Callable[..., None],
                    unpacks: Optional[List[int]] = None) -> None:
    """
    Sweep over a span with unpacking support.
    
    Args:
        span: The distributed span to sweep over
        func: Function to apply to unpacked distributed indices
        unpacks: Number of elements to unpack per call
    """
    if unpacks is None:
        unpacks = [1] * len(span.partial_lengths)
    
    # Check if this is the "no unpacking" case (all unpacks are 1)
    is_no_unpacking = all(unpack == 1 for unpack in unpacks)
    
    if is_no_unpacking:
        # Simple case: iterate over each individual index
        # This matches the test expectation for "no unpacking"
        span_lengths = span.partial_lengths
        ranges = [range(length) for length in span_lengths]
        
        for indices in itertools.product(*ranges):
            # Create a single distributed index with all dimensions
            dstr_idx = make_tile_distributed_index(list(indices))
            func(dstr_idx)
    else:
        # Complex unpacking case: group elements and unpack multiple indices
        span_lengths = span.partial_lengths
        
        # Calculate groups for each dimension
        dim_groups = []
        for length, unpack in zip(span_lengths, unpacks):
            groups = []
            for i in range(0, length, unpack):
                group = list(range(i, min(i + unpack, length)))
                groups.append(group)
            dim_groups.append(groups)
        
        # Iterate over all group combinations
        for group_combo in itertools.product(*dim_groups):
            # Flatten to get all indices
            all_indices = []
            for group in group_combo:
                for idx in group:
                    all_indices.append(idx)
            
            # Create distributed indices
            dstr_indices = []
            idx_ptr = 0
            for unpack in unpacks:
                indices = all_indices[idx_ptr:idx_ptr + unpack]
                if indices:  # Only create index if we have values
                    dstr_indices.append(make_tile_distributed_index(indices))
                idx_ptr += unpack
            
            # Apply function with unpacked indices
            func(*dstr_indices)


def sweep_tensor_direct(distributed_tensor: StaticDistributedTensor,
                       func: Callable[[List[int]], None]) -> None:
    """
    FIXED version of sweep_tile that works directly with StaticDistributedTensor instances.
    
    This matches the C++ pattern where sweep_tile<DistributedTensor>(func) uses the tensor
    directly without needing a separate template tensor.
    
    This eliminates the API mismatch by:
    1. Taking a StaticDistributedTensor instance (like C++ template)
    2. Generating Y indices directly that work with get_element()
    3. No conversion between different index formats needed
    
    Args:
        distributed_tensor: The distributed tensor to sweep over (e.g., from tile_window.load())
        func: Function that takes Y indices and processes them
              Signature: func(y_indices: List[int]) -> None
              
    Example:
        loaded = tile_window.load()
        sweep_tensor_direct(loaded, lambda y_indices: 
            print(f"Value at Y{y_indices}: {loaded.get_element(y_indices)}")
        )
    """
    # Get the tile distribution from the tensor
    tile_distribution = distributed_tensor.tile_distribution
    
    # Get all Y dimension lengths to iterate over
    y_lengths = tile_distribution.ys_to_d_descriptor.get_lengths()
    
    # Iterate through all Y indices (this matches C++ sweep_tile pattern)
    ranges = [range(length) for length in y_lengths]
    
    for y_indices in itertools.product(*ranges):
        # Convert to list for consistency and call function
        y_indices_list = list(y_indices)
        func(y_indices_list)


def sweep_tile(distributed_tensor_type: Union[type, Any],
               func: Callable[..., None],
               unpacks_per_x_dim: Optional[List[int]] = None) -> None:
    """
    Enhanced sweep-tile utility with control over unpacks along each X-dimension.
    
    The lambda function argument is the distributed index, which can be directly
    plugged into the distributed tensor as setter/getter.
    
    Examples:
        # Sweep tile 1 by 1
        sweep_tile(MyDistributedTensor, lambda idx: process(tensor[idx]))
        
        # Sweep tile with 2 pixels from last dim each function call
        sweep_tile(MyDistributedTensor, 
                  lambda idx0, idx1: process_pair(tensor[idx0], tensor[idx1]),
                  [1, 2])
        
        # Sweep tile with 2x2 pixel each function call
        sweep_tile(MyDistributedTensor,
                  lambda idx00, idx01, idx10, idx11: process_2x2(...),
                  [2, 2])
    
    Args:
        distributed_tensor_type: Type of the distributed tensor or instance
        func: Function to apply to distributed indices
        unpacks_per_x_dim: Number of elements to unpack per dimension
    """
    # FIXED API: If given a StaticDistributedTensor instance, use the direct API
    if isinstance(distributed_tensor_type, StaticDistributedTensor):
        # This is the FIXED pattern - use sweep_tensor_direct
        if unpacks_per_x_dim is None or all(x == 1 for x in unpacks_per_x_dim):
            # Simple case: use the clean direct API
            sweep_tensor_direct(distributed_tensor_type, func)
            return
        else:
            # Complex unpacking case - would need more implementation
            # For now, fall back to original logic
            pass
    
    # Get distributed spans from the tensor type or instance
    if hasattr(distributed_tensor_type, 'tile_distribution'):
        # It's an instance
        spans = distributed_tensor_type.tile_distribution.get_distributed_spans()
    elif hasattr(distributed_tensor_type, 'get_distributed_spans'):
        # It's a type with class method
        spans = distributed_tensor_type.get_distributed_spans()
    else:
        raise ValueError("Cannot get distributed spans from provided tensor")
    
    if unpacks_per_x_dim is None:
        unpacks_per_x_dim = [1] * len(spans)
    
    # Get all span lengths
    all_lengths = []
    for span in spans:
        all_lengths.extend(span.partial_lengths)
    
    # Calculate total unpacks
    total_unpacks = 1
    for unpack in unpacks_per_x_dim:
        total_unpacks *= unpack
    
    # If unpacking is 1 for all dimensions, sweep one by one
    if total_unpacks == 1:
        # Simple case: one index at a time
        ranges = [range(length) for length in all_lengths]
        for indices in itertools.product(*ranges):
            # Create single distributed index with all dimensions
            dstr_idx = make_tile_distributed_index(list(indices))
            func(dstr_idx)
    else:
        # Complex case: multiple indices based on unpacking
        # This is a simplified implementation
        # In practice, would need to handle space-filling curves
        dim_ranges = []
        dim_idx = 0
        for x_idx, span in enumerate(spans):
            unpack = unpacks_per_x_dim[x_idx]
            for length in span.partial_lengths:
                # Create groups for this dimension
                groups = []
                for i in range(0, length, unpack):
                    group = list(range(i, min(i + unpack, length)))
                    groups.append(group)
                dim_ranges.append(groups)
                dim_idx += 1
        
        # Iterate over all group combinations
        for group_combo in itertools.product(*dim_ranges):
            # Generate all indices in this group
            indices_list = list(itertools.product(*group_combo))
            
            # Create distributed indices
            dstr_indices = []
            for indices in indices_list:
                dstr_idx = make_tile_distributed_index(list(indices))
                dstr_indices.append(dstr_idx)
            
            # Apply function with all indices
            func(*dstr_indices)


@dataclass
class TileSweeper(Generic[TypeVar('T')]):
    """
    Construct a sweep tile instance which supports issuing the lambda one by one.
    
    This struct holds the lambda functor but does not hold the distributed tensor.
    The functionality is the same as sweep_tile().
    
    Attributes:
        distributed_tensor_type: Type of the distributed tensor or instance
        func: Function to apply to indices
        unpacks_per_x_dim: Unpacking configuration
    """
    
    distributed_tensor_type: Union[type, Any]
    func: Callable[..., None]
    unpacks_per_x_dim: Optional[List[int]] = None
    
    def __post_init__(self):
        """Initialize unpacks if not provided."""
        if self.unpacks_per_x_dim is None:
            # Determine number of dimensions
            if hasattr(self.distributed_tensor_type, 'tile_distribution'):
                # It's an instance
                spans = self.distributed_tensor_type.tile_distribution.get_distributed_spans()
                ndim = len(spans)
            elif hasattr(self.distributed_tensor_type, 'get_num_of_dimension'):
                ndim = self.distributed_tensor_type.get_num_of_dimension()
            else:
                # Default
                ndim = 1
            self.unpacks_per_x_dim = [1] * ndim
    
    def get_num_of_access(self) -> int:
        """
        Get the total number of accesses required for the sweep.
        
        Returns:
            Total number of function calls that will be made
        """
        # Get spans
        if hasattr(self.distributed_tensor_type, 'tile_distribution'):
            # It's an instance
            spans = self.distributed_tensor_type.tile_distribution.get_distributed_spans()
        elif hasattr(self.distributed_tensor_type, 'get_distributed_spans'):
            spans = self.distributed_tensor_type.get_distributed_spans()
        else:
            return 1
        
        # Calculate total accesses
        total = 1
        for span, unpack in zip(spans, self.unpacks_per_x_dim):
            span_length = span.partial_lengths[0] if span.partial_lengths else 1
            # Number of groups in this dimension
            num_groups = (span_length + unpack - 1) // unpack
            total *= num_groups
        
        return total
    
    def __call__(self, access_idx: Optional[int] = None) -> None:
        """
        Execute the sweep.
        
        Args:
            access_idx: Optional specific access index to execute.
                       If None, executes all accesses.
        """
        if access_idx is None:
            # Execute full sweep
            sweep_tile(self.distributed_tensor_type, self.func, self.unpacks_per_x_dim)
        else:
            # Execute specific access
            # This requires converting linear access index to multi-dimensional indices
            self._execute_access(access_idx)
    
    def _execute_access(self, access_idx: int) -> None:
        """Execute a specific access by index."""
        # Get spans
        if hasattr(self.distributed_tensor_type, 'tile_distribution'):
            # It's an instance
            spans = self.distributed_tensor_type.tile_distribution.get_distributed_spans()
        elif hasattr(self.distributed_tensor_type, 'get_distributed_spans'):
            spans = self.distributed_tensor_type.get_distributed_spans()
        else:
            return
        
        # Convert linear index to multi-dimensional group indices
        group_indices = []
        remaining = access_idx
        
        for i in range(len(spans) - 1, -1, -1):
            span = spans[i]
            unpack = self.unpacks_per_x_dim[i]
            span_length = span.partial_lengths[0] if span.partial_lengths else 1
            num_groups = (span_length + unpack - 1) // unpack
            
            group_idx = remaining % num_groups
            remaining //= num_groups
            group_indices.insert(0, group_idx)
        
        # Generate distributed indices for this access
        dstr_indices = []
        for dim_idx, (span, unpack, group_idx) in enumerate(zip(spans, self.unpacks_per_x_dim, group_indices)):
            span_length = span.partial_lengths[0] if span.partial_lengths else 1
            
            # Get indices for this group
            start_idx = group_idx * unpack
            for j in range(unpack):
                if start_idx + j < span_length:
                    idx = make_tile_distributed_index([start_idx + j])
                    dstr_indices.append(idx)
        
        # Apply function
        self.func(*dstr_indices)


def make_tile_sweeper(distributed_tensor_or_type: Union[type, Any],
                     func: Callable[..., None],
                     unpacks_per_x_dim: Optional[List[int]] = None) -> TileSweeper:
    """
    Create a tile sweeper instance.
    
    Args:
        distributed_tensor_or_type: Either a distributed tensor type or instance
        func: Function to apply during sweep
        unpacks_per_x_dim: Unpacking configuration
        
    Returns:
        TileSweeper instance
    """
    return TileSweeper(
        distributed_tensor_type=distributed_tensor_or_type,
        func=func,
        unpacks_per_x_dim=unpacks_per_x_dim
    ) 