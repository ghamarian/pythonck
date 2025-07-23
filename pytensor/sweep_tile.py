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
from .sequence_utils import get_y_unpacks_from_x_unpacks


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
    
    This function implements the C++ static_uford functionality with proper
    unpacking based on the get_y_unpacks_from_x_unpacks algorithm.
    
    Args:
        span: The distributed span to sweep over
        func: Function to apply to unpacked distributed indices
        unpacks: Y unpacks (calculated from X unpacks using get_y_unpacks_from_x_unpacks)
    """
    if unpacks is None:
        unpacks = [1] * len(span.partial_lengths)
    
    span_lengths = span.partial_lengths
    
    # Check if this is the "no unpacking" case (all unpacks are 1)
    is_no_unpacking = all(unpack == 1 for unpack in unpacks)
    
    if is_no_unpacking:
        # Simple case: iterate over each individual index
        ranges = [range(length) for length in span_lengths]
        
        for indices in itertools.product(*ranges):
            # Create a single distributed index with all dimensions
            dstr_idx = make_tile_distributed_index(list(indices))
            func(dstr_idx)
    else:
        # Complex unpacking case using proper C++ logic
        # Calculate how many groups we have in each dimension
        dim_groups = []
        total_calls = 1
        
        for length, unpack in zip(span_lengths, unpacks):
            num_groups = (length + unpack - 1) // unpack  # Ceiling division
            total_calls *= num_groups
            groups = []
            for group_idx in range(num_groups):
                start_idx = group_idx * unpack
                end_idx = min(start_idx + unpack, length)
                group = list(range(start_idx, end_idx))
                groups.append(group)
            dim_groups.append(groups)
        
        # Iterate over all group combinations
        for group_combo in itertools.product(*dim_groups):
            # Create distributed indices for this combination
            dstr_indices = []
            
            # For each position in the unpacking pattern
            max_group_size = max(len(group) for group in group_combo)
            for pos in range(max_group_size):
                indices = []
                for dim_idx, group in enumerate(group_combo):
                    if pos < len(group):
                        indices.append(group[pos])
                    else:
                        # Pad with the last valid index in this dimension
                        indices.append(group[-1])
                
                if indices:  # Only create index if we have values
                    dstr_indices.append(make_tile_distributed_index(indices))
            
            # Apply function with unpacked indices
            if dstr_indices:
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
    
    This now properly implements the C++ sweep_tile logic using get_y_unpacks_from_x_unpacks
    to convert X unpacks to Y unpacks for each span.
    
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
        unpacks_per_x_dim: Number of elements to unpack per X dimension
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
    
    if len(unpacks_per_x_dim) != len(spans):
        raise ValueError(f"unpacks_per_x_dim length {len(unpacks_per_x_dim)} must match number of spans {len(spans)}")
    
    # Process each span recursively like C++ sweep_tile_impl
    def process_spans(span_idx: int, current_span_indices: List[Any]):
        if span_idx >= len(spans):
            # Base case: apply function with all accumulated indices
            func(*current_span_indices)
            return
        
        # Get current span and its X unpack requirement
        current_span = spans[span_idx]
        x_unpacks = unpacks_per_x_dim[span_idx]
        
        # Convert X unpacks to Y unpacks using get_y_unpacks_from_x_unpacks
        y_lengths = current_span.partial_lengths
        try:
            y_unpacks = get_y_unpacks_from_x_unpacks(y_lengths, x_unpacks)
        except ValueError:
            # If unpacking is not possible, use default [1, 1, ...]
            y_unpacks = [1] * len(y_lengths)
        
        # Use sweep_tile_uspan for this span with proper Y unpacks
        def span_processor(*span_indices):
            # Add these indices to accumulated indices and recurse
            next_span_indices = current_span_indices + list(span_indices)
            process_spans(span_idx + 1, next_span_indices)
        
        sweep_tile_uspan(current_span, span_processor, y_unpacks)
    
    # Start the recursive processing
    process_spans(0, [])


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