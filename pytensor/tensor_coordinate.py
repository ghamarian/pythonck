"""
Python implementation of tensor_coordinate.hpp from Composable Kernels.

This module provides tensor coordinate functionality for navigating
multi-dimensional tensor spaces.
"""

from typing import List, Tuple, Optional, TypeVar, Generic, Union
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


class MultiIndex:
    """Multi-dimensional index for tensor coordinates."""
    
    def __init__(self, size: int, values: Optional[List[int]] = None):
        """
        Initialize multi-index.
        
        Args:
            size: Number of dimensions
            values: Initial values (defaults to zeros)
        """
        self.size = size
        self._values = values if values is not None else [0] * size
        
        if len(self._values) != size:
            raise ValueError(f"Values length {len(self._values)} doesn't match size {size}")
    
    def __getitem__(self, idx: int) -> int:
        """Get value at index."""
        return self._values[idx]
    
    def __setitem__(self, idx: int, value: int):
        """Set value at index."""
        self._values[idx] = value
    
    def __len__(self) -> int:
        """Get number of dimensions."""
        return self.size
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MultiIndex({self._values})"
    
    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, MultiIndex):
            return False
        return self._values == other._values
    
    def copy(self) -> 'MultiIndex':
        """Create a copy of this multi-index."""
        return MultiIndex(self.size, self._values.copy())
    
    def to_list(self) -> List[int]:
        """Convert to list."""
        return self._values.copy()
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self._values)


@dataclass
class TensorAdaptorCoordinate:
    """
    Coordinate for tensor adaptor transformations.
    
    This tracks coordinates through hidden dimensions during
    tensor transformations.
    """
    ndim_hidden: int
    bottom_dimension_hidden_ids: List[int]
    top_dimension_hidden_ids: List[int]
    idx_hidden: MultiIndex
    
    def __post_init__(self):
        """Validate dimensions after initialization."""
        if self.idx_hidden.size != self.ndim_hidden:
            raise ValueError(f"Hidden index size {self.idx_hidden.size} doesn't match ndim_hidden {self.ndim_hidden}")
    
    def get_top_index(self) -> MultiIndex:
        """Get top-level index from hidden index."""
        top_values = [self.idx_hidden[hid] for hid in self.top_dimension_hidden_ids]
        return MultiIndex(len(self.top_dimension_hidden_ids), top_values)
    
    def get_bottom_index(self) -> MultiIndex:
        """Get bottom-level index from hidden index."""
        bottom_values = [self.idx_hidden[hid] for hid in self.bottom_dimension_hidden_ids]
        return MultiIndex(len(self.bottom_dimension_hidden_ids), bottom_values)
    
    def get_hidden_index(self) -> MultiIndex:
        """Get the hidden index."""
        return self.idx_hidden
    
    def set_hidden_index(self, idx: MultiIndex):
        """Set the hidden index."""
        if idx.size != self.ndim_hidden:
            raise ValueError(f"Index size {idx.size} doesn't match ndim_hidden {self.ndim_hidden}")
        self.idx_hidden = idx

    def copy(self) -> 'TensorAdaptorCoordinate':
        """Create a copy of this TensorAdaptorCoordinate."""
        return TensorAdaptorCoordinate(
            ndim_hidden=self.ndim_hidden,
            bottom_dimension_hidden_ids=self.bottom_dimension_hidden_ids.copy(),
            top_dimension_hidden_ids=self.top_dimension_hidden_ids.copy(),
            idx_hidden=self.idx_hidden.copy()
        )


@dataclass
class TensorCoordinate(TensorAdaptorCoordinate):
    """
    Tensor coordinate that extends TensorAdaptorCoordinate.
    
    This represents a coordinate in a tensor with transformations applied.
    The bottom dimension is always a single offset dimension.
    """
    
    def __init__(self, ndim_hidden: int, top_dimension_hidden_ids: List[int], 
                 idx_hidden: Optional[MultiIndex] = None):
        """
        Initialize tensor coordinate.
        
        Args:
            ndim_hidden: Number of hidden dimensions
            top_dimension_hidden_ids: IDs of top dimensions in hidden space
            idx_hidden: Initial hidden index
        """
        # Bottom dimension is always [0] for tensor coordinate
        bottom_dimension_hidden_ids = [0]
        
        if idx_hidden is None:
            idx_hidden = MultiIndex(ndim_hidden)
            
        super().__init__(
            ndim_hidden=ndim_hidden,
            bottom_dimension_hidden_ids=bottom_dimension_hidden_ids,
            top_dimension_hidden_ids=top_dimension_hidden_ids,
            idx_hidden=idx_hidden
        )
    
    def get_index(self) -> MultiIndex:
        """Get the top-level tensor index."""
        return self.get_top_index()
    
    def get_offset(self) -> int:
        """Get the linear offset (bottom index)."""
        bottom_idx = self.get_bottom_index()
        return bottom_idx[0] if len(bottom_idx) > 0 else 0
    
    @classmethod
    def from_adaptor_coordinate(cls, adaptor_coord: TensorAdaptorCoordinate) -> 'TensorCoordinate':
        """Create tensor coordinate from adaptor coordinate."""
        return cls(
            ndim_hidden=adaptor_coord.ndim_hidden,
            top_dimension_hidden_ids=adaptor_coord.top_dimension_hidden_ids,
            idx_hidden=adaptor_coord.idx_hidden.copy()
        )

    def copy(self):
        """Create a copy of this TensorCoordinate."""
        return TensorCoordinate(
            ndim_hidden=self.ndim_hidden,
            top_dimension_hidden_ids=self.top_dimension_hidden_ids.copy(),
            idx_hidden=self.idx_hidden.copy()
        )


def make_tensor_coordinate(tensor_desc: 'TensorDescriptor', idx_top: Union[List[int], MultiIndex]) -> TensorCoordinate:
    """
    Create a tensor coordinate from a tensor descriptor and top index.
    
    Args:
        tensor_desc: Tensor descriptor defining the transformations
        idx_top: Top-level index
        
    Returns:
        TensorCoordinate instance
    """
    # Convert list to MultiIndex if needed
    if isinstance(idx_top, list):
        idx_top = MultiIndex(len(idx_top), idx_top)
    
    # Create adaptor coordinate first
    adaptor_coord = make_tensor_adaptor_coordinate(tensor_desc, idx_top)
    
    # Convert to tensor coordinate
    return TensorCoordinate.from_adaptor_coordinate(adaptor_coord)


def make_tensor_adaptor_coordinate(adaptor: 'TensorAdaptor', idx_top: MultiIndex) -> TensorAdaptorCoordinate:
    """
    Create a tensor adaptor coordinate by applying transformations.
    
    Args:
        adaptor: Tensor adaptor with transformations
        idx_top: Top-level index
        
    Returns:
        TensorAdaptorCoordinate instance
    """
    if adaptor.get_num_of_top_dimension() != len(idx_top):
        raise ValueError(f"Top index size {len(idx_top)} doesn't match adaptor top dimensions {adaptor.get_num_of_top_dimension()}")
    
    # Initialize hidden index
    ndim_hidden = adaptor.get_num_of_hidden_dimension()
    idx_hidden = MultiIndex(ndim_hidden)
    
    # Set top dimensions in hidden index
    top_dim_ids = adaptor.get_top_dimension_hidden_ids()
    for i, hid in enumerate(top_dim_ids):
        idx_hidden[hid] = idx_top[i]
    
    # Apply transformations in reverse order to calculate lower indices
    ntransform = adaptor.get_num_of_transform()
    for itran in range(ntransform - 1, -1, -1):
        transform = adaptor.get_transforms()[itran]
        dims_low = adaptor.get_lower_dimension_hidden_idss()[itran]
        dims_up = adaptor.get_upper_dimension_hidden_idss()[itran]
        
        # Get upper index values
        idx_up = MultiIndex(len(dims_up), [idx_hidden[hid] for hid in dims_up])
        
        # Calculate lower index
        idx_low = transform.calculate_lower_index(idx_up)
        
        # Set lower dimensions in hidden index
        for i, hid in enumerate(dims_low):
            idx_hidden[hid] = idx_low[i]
    
    return TensorAdaptorCoordinate(
        ndim_hidden=ndim_hidden,
        bottom_dimension_hidden_ids=adaptor.get_bottom_dimension_hidden_ids(),
        top_dimension_hidden_ids=adaptor.get_top_dimension_hidden_ids(),
        idx_hidden=idx_hidden
    )


def move_tensor_coordinate(tensor_desc: 'TensorDescriptor', coord: TensorCoordinate, 
                          coord_step: Union[List[int], MultiIndex], judge_do_transforms: bool = True):
    """
    Move tensor coordinate by a given step.
    
    Args:
        tensor_desc: Tensor descriptor
        coord: Coordinate to move
        coord_step: Step to move by
        judge_do_transforms: Whether to optimize by checking which transforms need updating
    """
    # Convert list to MultiIndex if needed
    if isinstance(coord_step, list):
        coord_step = MultiIndex(len(coord_step), coord_step)
        
    move_tensor_adaptor_coordinate(tensor_desc, coord, coord_step, judge_do_transforms)


def move_tensor_adaptor_coordinate(adaptor: 'TensorAdaptor', coord: TensorAdaptorCoordinate,
                                  idx_diff_top: MultiIndex, judge_do_transforms: bool = True) -> MultiIndex:
    """
    Move tensor adaptor coordinate and return bottom index difference.
    
    CURRENT IMPLEMENTATION: Recalculates everything from scratch.
    
    Args:
        adaptor: Tensor adaptor
        coord: Coordinate to move
        idx_diff_top: Top index difference
        judge_do_transforms: Whether to optimize transforms
        
    Returns:
        Bottom index difference
    """
    ndim_hidden = adaptor.get_num_of_hidden_dimension()
    ndim_top = adaptor.get_num_of_top_dimension()
    ndim_bottom = adaptor.get_num_of_bottom_dimension()
    ntransform = adaptor.get_num_of_transform()
    
    # Get current state
    idx_hidden = coord.get_hidden_index()
    top_dim_ids = adaptor.get_top_dimension_hidden_ids()
    bottom_dim_ids = adaptor.get_bottom_dimension_hidden_ids()
    
    # Calculate new top indices
    new_top_indices = []
    for i, hid in enumerate(top_dim_ids):
        if i < len(idx_diff_top):
            new_val = idx_hidden[hid] + idx_diff_top[i]
            new_top_indices.append(new_val)
        else:
            new_top_indices.append(idx_hidden[hid])
    
    # Get old bottom indices for difference calculation
    old_bottom_indices = [idx_hidden[hid] for hid in bottom_dim_ids]
    
    # Create a completely new coordinate with the new top indices
    # This ensures all transforms are applied correctly and consistently
    new_coord = make_tensor_adaptor_coordinate(adaptor, MultiIndex(len(new_top_indices), new_top_indices))
    
    # Get the new bottom indices from the correctly calculated coordinate
    new_bottom_indices = new_coord.get_bottom_index().to_list()
    
    # Calculate the bottom index difference
    idx_diff_bottom = MultiIndex(len(bottom_dim_ids))
    for i in range(len(bottom_dim_ids)):
        idx_diff_bottom[i] = new_bottom_indices[i] - old_bottom_indices[i]
    
    # Update the original coordinate to the new state
    coord.set_hidden_index(new_coord.get_hidden_index())
    
    return idx_diff_bottom


def move_tensor_adaptor_coordinate_efficient(adaptor: 'TensorAdaptor', coord: TensorAdaptorCoordinate,
                                           idx_diff_top: MultiIndex, judge_do_transforms: bool = True) -> MultiIndex:
    """
    Move tensor adaptor coordinate using efficient C++-like update_lower_index methods.
    
    NEW IMPLEMENTATION: Uses update_lower_index for efficiency, matching C++ exactly.
    
    Args:
        adaptor: Tensor adaptor
        coord: Coordinate to move (modified in place)
        idx_diff_top: Top index difference
        judge_do_transforms: Whether to optimize transforms (currently ignored)
        
    Returns:
        Bottom index difference
    """
    ndim_hidden = adaptor.get_num_of_hidden_dimension()
    ndim_bottom = adaptor.get_num_of_bottom_dimension()
    ntransform = adaptor.get_num_of_transform()
    
    # Get current state (work directly on the coordinate's hidden index)
    idx_hidden = coord.get_hidden_index()
    top_dim_ids = adaptor.get_top_dimension_hidden_ids()
    bottom_dim_ids = adaptor.get_bottom_dimension_hidden_ids()
    
    # Initialize hidden index differences (matches C++ idx_diff_hidden)
    idx_diff_hidden = MultiIndex(ndim_hidden, [0] * ndim_hidden)
    
    # Set top index difference in hidden space (matches C++)
    for i, hid in enumerate(top_dim_ids):
        if i < len(idx_diff_top):
            idx_diff_hidden[hid] = idx_diff_top[i]
    
    # Update top indices first (matches C++)
    for i, hid in enumerate(top_dim_ids):
        if i < len(idx_diff_top):
            idx_hidden[hid] += idx_diff_top[i]
    
    # Update rest through transforms (backwards iteration like C++)
    transforms = adaptor.get_transforms()
    lower_dim_idss = adaptor.get_lower_dimension_hidden_idss()
    upper_dim_idss = adaptor.get_upper_dimension_hidden_idss()
    
    for itran in reversed(range(ntransform)):
        transform = transforms[itran]
        dims_low = lower_dim_idss[itran]
        dims_up = upper_dim_idss[itran]
        
        # Skip transforms without update_lower_index method
        if not hasattr(transform, 'update_lower_index'):
            continue
        
        # Get current state for this transform (C++ pattern)
        idx_low_current = MultiIndex(len(dims_low), [idx_hidden[hid] for hid in dims_low])
        idx_up_new = MultiIndex(len(dims_up), [idx_hidden[hid] for hid in dims_up])
        idx_diff_up = MultiIndex(len(dims_up), [idx_diff_hidden[hid] for hid in dims_up])
        
        # Call update_lower_index (matches C++)
        idx_diff_low, idx_low_updated = transform.update_lower_index(
            idx_diff_up, idx_low_current, idx_up_new
        )
        
        # Update hidden index and differences (matches C++)
        for i, hid in enumerate(dims_low):
            idx_diff_hidden[hid] = idx_diff_low[i]
            idx_hidden[hid] = idx_low_updated[i]
    
    # The coordinate is already updated in place (no need to call set_hidden_index)
    
    # Return bottom index difference (matches C++)
    return MultiIndex(ndim_bottom, [idx_diff_hidden[hid] for hid in bottom_dim_ids])


def coordinate_has_valid_offset_assuming_top_index_is_valid(tensor_desc: 'TensorDescriptor', 
                                                           coord: TensorCoordinate) -> bool:
    """
    Check if coordinate has valid offset assuming top index is valid.
    
    Args:
        tensor_desc: Tensor descriptor
        coord: Coordinate to check
        
    Returns:
        True if offset is valid
    """
    return adaptor_coordinate_is_valid_assuming_top_index_is_valid(tensor_desc, coord)


def adaptor_coordinate_is_valid_assuming_top_index_is_valid(adaptor: 'TensorAdaptor',
                                                           coord: TensorAdaptorCoordinate) -> bool:
    """
    Check if adaptor coordinate is valid assuming top index is valid.
    
    Args:
        adaptor: Tensor adaptor
        coord: Coordinate to check
        
    Returns:
        True if coordinate is valid
    """
    valid = True
    idx_hidden = coord.get_hidden_index()
    
    # Check each transform
    for itran in range(adaptor.get_num_of_transform() - 1, -1, -1):
        transform = adaptor.get_transforms()[itran]
        
        # Only check if transform doesn't always map valid to valid
        if not transform.is_valid_upper_index_always_mapped_to_valid_lower_index():
            dims_up = adaptor.get_upper_dimension_hidden_idss()[itran]
            idx_up = MultiIndex(len(dims_up), [idx_hidden[hid] for hid in dims_up])
            
            if not transform.is_valid_upper_index_mapped_to_valid_lower_index(idx_up):
                valid = False
                break
    
    return valid


def coordinate_has_valid_offset(tensor_desc: 'TensorDescriptor', coord: TensorCoordinate) -> bool:
    """
    Check if coordinate has valid offset.
    
    Args:
        tensor_desc: Tensor descriptor
        coord: Coordinate to check
        
    Returns:
        True if coordinate is valid
    """
    return adaptor_coordinate_is_valid(tensor_desc, coord)


def adaptor_coordinate_is_valid(adaptor: 'TensorAdaptor', coord: TensorAdaptorCoordinate) -> bool:
    """
    Check if adaptor coordinate is valid.
    
    Args:
        adaptor: Tensor adaptor
        coord: Coordinate to check
        
    Returns:
        True if coordinate is valid
    """
    # First check top index
    idx_top = coord.get_top_index()
    
    # Check each dimension is within bounds
    for i in range(adaptor.get_num_of_top_dimension()):
        if idx_top[i] < 0 or idx_top[i] >= adaptor.get_length(i):
            return False
    
    # Then check validity through transforms
    return adaptor_coordinate_is_valid_assuming_top_index_is_valid(adaptor, coord) 