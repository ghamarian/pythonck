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
from .functional_utils import StaticUford


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
    
    This function implements the C++ static_uford functionality using our
    StaticUford class for proper C++ compatibility.
    
    Args:
        span: The distributed span to sweep over
        func: Function to apply to unpacked distributed indices
        unpacks: Y unpacks (calculated from X unpacks using get_y_unpacks_from_x_unpacks)
    """
    if unpacks is None:
        unpacks = [1] * len(span.partial_lengths)
    
    span_lengths = span.partial_lengths
    
    # Handle empty spans gracefully
    if not span_lengths or any(length == 0 for length in span_lengths):
        return
    
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
        # Use StaticUford for proper C++ static_uford compatibility
        def uford_func(*index_lists):
            # Convert raw index lists to TileDistributedIndex objects
            dstr_indices = []
            for index_list in index_lists:
                dstr_indices.append(make_tile_distributed_index(index_list))
            
            # Call the original function with distributed indices
            func(*dstr_indices)
        
        # Create StaticUford with the span lengths and unpacks
        static_uford = StaticUford(span_lengths, unpacks)
        static_uford(uford_func)


def sweep_tile_old(distributed_tensor: StaticDistributedTensor,
               func: Callable[..., None],
               unpacks_per_x_dim: Optional[List[int]] = None) -> None:
    """
    Sweep-tile utility with control over unpacks along each X-dimension.
    
    This matches the C++ implementation exactly: always processes spans and
    passes TileDistributedIndex objects to the function, regardless of unpacking.
    
    The lambda function argument always receives TileDistributedIndex objects,
    just like in C++.
    
    Examples:
        # Sweep tile 1 by 1 - receives single TileDistributedIndex per call
        sweep_tile(tensor_instance, lambda idx: process(idx))
        
        # Sweep tile with 2 pixels from last dim - receives 2 TileDistributedIndex per call
        sweep_tile(tensor_instance, 
                  lambda idx0, idx1: process_pair(idx0, idx1),
                  [1, 2])
        
        # Sweep tile with 2x2 pixel - receives 4 TileDistributedIndex per call
        sweep_tile(tensor_instance,
                  lambda idx00, idx01, idx10, idx11: process_2x2(idx00, idx01, idx10, idx11),
                  [2, 2])
    
    Args:
        distributed_tensor: StaticDistributedTensor instance
        func: Function to apply to TileDistributedIndex objects
        unpacks_per_x_dim: Number of elements to unpack per X dimension
    """
    # Always use the span-based approach like C++
    spans = distributed_tensor.tile_distribution.get_distributed_spans()
    
    if unpacks_per_x_dim is None:
        unpacks_per_x_dim = [1] * len(spans)
    
    if len(unpacks_per_x_dim) != len(spans):
        raise ValueError(f"unpacks_per_x_dim length {len(unpacks_per_x_dim)} must match number of spans {len(spans)}")
    
    # Process each span recursively like C++ sweep_tile_impl
    def process_spans(span_idx: int, current_span_indices: List[Any]):
        if span_idx >= len(spans):
            # Base case: apply function with all accumulated indices
            # C++ does: unpack(f, span_idx) = f(*span_idx)
            # Always pass TileDistributedIndex objects, just like C++
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

def sweep_tile(distributed_tensor: StaticDistributedTensor,
               func: Callable[..., None],
               unpacks_per_x_dim: Optional[List[int]] = None) -> None:
    """
    Fixed sweep tile implementation: simplified but correct.
    
    Always calls func with one TileDistributedIndex per span,
    matching the C++ behavior exactly.
    """
    spans = distributed_tensor.tile_distribution.get_distributed_spans()
    
    if unpacks_per_x_dim is None:
        unpacks_per_x_dim = [1] * len(spans)
    
    # Validation: check that unpacks length matches spans
    if len(unpacks_per_x_dim) != len(spans):
        raise ValueError(f"unpacks_per_x_dim length {len(unpacks_per_x_dim)} must match number of spans {len(spans)}")
    
    # Check if this is the simple case (no unpacking)
    if all(unpack == 1 for unpack in unpacks_per_x_dim):
        # Simple case: each span contributes one TileDistributedIndex
        # Build all span combinations directly
        span_index_lists = []
        
        for span in spans:
            # For simple case, just iterate through span normally
            span_lengths = span.partial_lengths
            span_indices = []
            for indices in itertools.product(*[range(length) for length in span_lengths]):
                span_indices.append(make_tile_distributed_index(list(indices)))
            span_index_lists.append(span_indices)
        
        # Call function with one index per span
        for combination in itertools.product(*span_index_lists):
            func(*combination)
        
    else:
        # Complex case: use original logic but simplified
        # Pre-calculate all span groups
        all_span_groups = []
        
        for span_idx, span in enumerate(spans):
            x_unpacks = unpacks_per_x_dim[span_idx]
            y_lengths = span.partial_lengths
            
            # Convert X unpacks to Y unpacks
            try:
                y_unpacks = get_y_unpacks_from_x_unpacks(y_lengths, x_unpacks)
            except ValueError:
                # Fallback to no unpacking for this span
                y_unpacks = [1] * len(y_lengths)
            
            # Collect all index groups for this span
            span_groups = []
            def collect_span_group(*indices):
                # Convert to TileDistributedIndex objects
                dstr_indices = [make_tile_distributed_index(list(idx)) for idx in indices]
                span_groups.append(dstr_indices)
            
            StaticUford(y_lengths, y_unpacks)(collect_span_group)
            all_span_groups.append(span_groups)
        
        # Generate all combinations across spans
        for span_combination in itertools.product(*all_span_groups):
            # Each span_combination contains a list of indices for each span
            # Flatten to get all indices for this function call
            all_indices = []
            for span_group in span_combination:
                all_indices.extend(span_group)
            
            func(*all_indices)

@dataclass
class TileSweeper(Generic[TypeVar('T')]):
    """
    Construct a sweep tile instance which supports issuing the lambda one by one.
    
    This struct holds the lambda functor but does not hold the distributed tensor.
    The functionality is the same as sweep_tile().
    
    Attributes:
        distributed_tensor: StaticDistributedTensor instance
        func: Function to apply to indices
        unpacks_per_x_dim: Unpacking configuration
    """
    
    distributed_tensor: StaticDistributedTensor
    func: Callable[..., None]
    unpacks_per_x_dim: Optional[List[int]] = None
    
    def __post_init__(self):
        """Initialize unpacks if not provided."""
        if self.unpacks_per_x_dim is None:
            # Determine number of dimensions from the tensor instance
            spans = self.distributed_tensor.tile_distribution.get_distributed_spans()
            ndim = len(spans)
            self.unpacks_per_x_dim = [1] * ndim
    
    def get_num_of_access(self) -> int:
        """
        Get the total number of accesses required for the sweep.
        
        Returns:
            Total number of function calls that will be made
        """
        # Get spans from the tensor instance
        spans = self.distributed_tensor.tile_distribution.get_distributed_spans()
        
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
            sweep_tile(self.distributed_tensor, self.func, self.unpacks_per_x_dim)
        else:
            # Execute specific access
            # This requires converting linear access index to multi-dimensional indices
            self._execute_access(access_idx)
    
    def _execute_access(self, access_idx: int) -> None:
        """Execute a specific access by index."""
        # Get spans from the tensor instance
        spans = self.distributed_tensor.tile_distribution.get_distributed_spans()
        
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


def make_tile_sweeper(distributed_tensor: StaticDistributedTensor,
                     func: Callable[..., None],
                     unpacks_per_x_dim: Optional[List[int]] = None) -> TileSweeper:
    """
    Create a tile sweeper instance.
    
    Args:
        distributed_tensor: StaticDistributedTensor instance
        func: Function to apply during sweep
        unpacks_per_x_dim: Unpacking configuration
        
    Returns:
        TileSweeper instance
    """
    return TileSweeper(
        distributed_tensor=distributed_tensor,
        func=func,
        unpacks_per_x_dim=unpacks_per_x_dim
    ) 