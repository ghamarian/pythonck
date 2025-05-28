"""
Python implementation of tile_window.hpp from Composable Kernels.

This module provides tile window functionality for accessing tensor data
in a windowed manner with support for distributed access patterns.
"""

from typing import List, Tuple, Optional, Union, Any, TypeVar, Generic
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .tensor_view import TensorView
from .tile_distribution import TileDistribution
from .static_distributed_tensor import StaticDistributedTensor
from .tensor_coordinate import (
    MultiIndex, TensorCoordinate, TensorAdaptorCoordinate,
    make_tensor_coordinate, move_tensor_coordinate, make_tensor_adaptor_coordinate,
    coordinate_has_valid_offset
)
from .tensor_descriptor import TensorAdaptor


@dataclass
class TileWindowWithStaticDistribution:
    """
    Tile window that provides distributed access to tensor data.
    
    This class manages a window view of a tensor with a specific distribution
    pattern across processing elements (threads/warps).
    
    Attributes:
        bottom_tensor_view: Underlying tensor view
        window_lengths: Size of the window in each dimension
        window_origin: Origin position of the window in the tensor
        tile_distribution: Distribution pattern for the tile
        num_coord: Number of coordinate bundles (default 1)
    """
    
    bottom_tensor_view: TensorView
    window_lengths: List[int]
    window_origin: List[int]
    tile_distribution: TileDistribution
    num_coord: int = 1
    
    def __post_init__(self):
        """Validate and initialize the tile window."""
        # Validate dimensions
        ndim_bottom = self.bottom_tensor_view.get_num_of_dimension()
        ndim_window = len(self.window_lengths)
        
        if ndim_bottom != ndim_window:
            raise ValueError(f"Window dimensions {ndim_window} must match tensor dimensions {ndim_bottom}")
        
        if len(self.window_origin) != ndim_bottom:
            raise ValueError(f"Window origin dimensions must match tensor dimensions {ndim_bottom}")
        
        # Check that window adaptor dimensions match
        window_adaptor = self.tile_distribution.ps_ys_to_xs_adaptor
        if window_adaptor.get_num_of_bottom_dimension() != ndim_bottom:
            raise ValueError("Window adaptor bottom dimensions must match tensor dimensions")
        
        # Pre-compute coordinates for efficient access
        self._precompute_coordinates()
    
    def _precompute_coordinates(self):
        """Pre-compute coordinate bundles for efficient load/store operations."""
        self.pre_computed_coords = []
        
        # Get partition index
        partition_idx = self.tile_distribution.get_partition_index()
        ndim_y = self.tile_distribution.ndim_y
        
        # Create initial adaptor coordinate
        # The adaptor expects P+Y dimensions, where P is partition dimensions
        ps_ys_idx = partition_idx + [0] * ndim_y
        window_adaptor_coord = make_tensor_adaptor_coordinate(
            self.tile_distribution.ps_ys_to_xs_adaptor,
            ps_ys_idx
        )
        
        # Create initial bottom tensor coordinate
        bottom_tensor_idx = [
            self.window_origin[i] + window_adaptor_coord.get_bottom_index()[i]
            for i in range(len(self.window_origin))
        ]
        bottom_tensor_coord = make_tensor_coordinate(
            self.bottom_tensor_view.tensor_desc,
            bottom_tensor_idx
        )
        
        # Store coordinate bundles
        for i_coord in range(self.num_coord):
            # For simplicity, we'll store the same coordinates for now
            # In practice, these would be offset based on access patterns
            self.pre_computed_coords.append((window_adaptor_coord, bottom_tensor_coord))
    
    def get_num_of_dimension(self) -> int:
        """Get number of dimensions."""
        return self.bottom_tensor_view.get_num_of_dimension()
    
    def get_window_lengths(self) -> List[int]:
        """Get window lengths."""
        return self.window_lengths
    
    def get_tile_distribution(self) -> TileDistribution:
        """Get tile distribution."""
        return self.tile_distribution
    
    def get_bottom_tensor_view(self) -> TensorView:
        """Get bottom tensor view."""
        return self.bottom_tensor_view
    
    def get_window_origin(self) -> List[int]:
        """Get window origin."""
        return self.window_origin
    
    def load(self, oob_conditional_check: bool = True) -> StaticDistributedTensor:
        """
        Load data from the window into a distributed tensor.
        
        Args:
            oob_conditional_check: Whether to check out-of-bounds access
            
        Returns:
            StaticDistributedTensor containing the loaded data
        """
        # Create distributed tensor
        dst_tensor = StaticDistributedTensor(
            data_type=self.bottom_tensor_view.dtype,
            tile_distribution=self.tile_distribution
        )
        
        # Load data into it
        self.load_into(dst_tensor, oob_conditional_check)
        
        return dst_tensor
    
    def load_into(self, dst_tensor: StaticDistributedTensor, 
                  oob_conditional_check: bool = True):
        """
        Load data from the window into an existing distributed tensor.
        
        Args:
            dst_tensor: Destination distributed tensor
            oob_conditional_check: Whether to check out-of-bounds access
        """
        # Get Y to D descriptor for mapping
        ys_to_d_desc = self.tile_distribution.ys_to_d_descriptor
        
        # For each coordinate bundle
        for i_coord in range(self.num_coord):
            window_adaptor_coord, bottom_tensor_coord = self.pre_computed_coords[i_coord]
            
            # In a real implementation, we would iterate over access patterns
            # For now, simplified version that loads one element
            if oob_conditional_check:
                # Check if coordinate is valid
                if not self._is_coordinate_valid(bottom_tensor_coord):
                    continue
            
            # Get element from bottom tensor
            offset = bottom_tensor_coord.get_offset()
            value = self.bottom_tensor_view.get_element_by_offset(offset)
            
            # Store in distributed tensor
            # This is simplified - in practice would use space-filling curves
            dst_tensor.set_thread_data(0, value)
    
    def load_raw(self, dst_tensor: StaticDistributedTensor,
                 oob_conditional_check: bool = True,
                 pre_nop: bool = False):
        """
        Load data using raw memory operations.
        
        This is a simplified version that mimics the C++ load_raw functionality.
        
        Args:
            dst_tensor: Destination distributed tensor
            oob_conditional_check: Whether to check out-of-bounds access
            pre_nop: Whether to insert a no-op before first access
        """
        # Similar to load_into but with raw access patterns
        # In practice, this would use vectorized loads and space-filling curves
        self.load_into(dst_tensor, oob_conditional_check)
    
    def store(self, src_tensor: StaticDistributedTensor,
              oob_conditional_check: bool = True):
        """
        Store data from a distributed tensor into the window.
        
        Args:
            src_tensor: Source distributed tensor
            oob_conditional_check: Whether to check out-of-bounds access
        """
        # For each coordinate bundle
        for i_coord in range(self.num_coord):
            window_adaptor_coord, bottom_tensor_coord = self.pre_computed_coords[i_coord]
            
            if oob_conditional_check:
                # Check if coordinate is valid
                if not self._is_coordinate_valid(bottom_tensor_coord):
                    continue
            
            # Get element from distributed tensor
            value = src_tensor.get_thread_data(0)
            
            # Store in bottom tensor
            offset = bottom_tensor_coord.get_offset()
            self.bottom_tensor_view.set_element_by_offset(offset, value)
    
    def store_raw(self, src_tensor: StaticDistributedTensor):
        """
        Store data using raw memory operations.
        
        Args:
            src_tensor: Source distributed tensor
        """
        # Similar to store but with raw access patterns
        self.store(src_tensor, oob_conditional_check=True)
    
    def update(self, src_tensor: StaticDistributedTensor,
               oob_conditional_check: bool = True):
        """
        Update (accumulate) data from a distributed tensor into the window.
        
        Args:
            src_tensor: Source distributed tensor
            oob_conditional_check: Whether to check out-of-bounds access
        """
        # For each coordinate bundle
        for i_coord in range(self.num_coord):
            window_adaptor_coord, bottom_tensor_coord = self.pre_computed_coords[i_coord]
            
            if oob_conditional_check:
                # Check if coordinate is valid
                if not self._is_coordinate_valid(bottom_tensor_coord):
                    continue
            
            # Get element from distributed tensor
            value = src_tensor.get_thread_data(0)
            
            # Update (accumulate) in bottom tensor
            offset = bottom_tensor_coord.get_offset()
            current_value = self.bottom_tensor_view.get_element_by_offset(offset)
            self.bottom_tensor_view.set_element_by_offset(offset, current_value + value)
    
    def update_raw(self, src_tensor: StaticDistributedTensor,
                   oob_conditional_check: bool = True,
                   pre_nop: bool = False):
        """
        Update data using raw memory operations.
        
        Args:
            src_tensor: Source distributed tensor
            oob_conditional_check: Whether to check out-of-bounds access
            pre_nop: Whether to insert a no-op before first access
        """
        # Similar to update but with raw access patterns
        self.update(src_tensor, oob_conditional_check)
    
    def async_load(self, lds_tile: 'TileWindowWithStaticDistribution',
                   oob_conditional_check: bool = True) -> None:
        """
        Asynchronously load data into LDS (Local Data Share) tile.
        
        This is a simplified version - in practice would use async memory operations.
        
        Args:
            lds_tile: LDS tile window to load into
            oob_conditional_check: Whether to check out-of-bounds access
        """
        # Simplified: just do a synchronous load
        data = self.load(oob_conditional_check)
        # In practice, would initiate async transfer to LDS
    
    def async_load_raw(self, lds_tile: 'TileWindowWithStaticDistribution',
                       oob_conditional_check: bool = True,
                       pre_nop: bool = False) -> None:
        """
        Asynchronously load data using raw operations.
        
        Args:
            lds_tile: LDS tile window to load into
            oob_conditional_check: Whether to check out-of-bounds access
            pre_nop: Whether to insert a no-op before first access
        """
        # Simplified version
        self.async_load(lds_tile, oob_conditional_check)
    
    def move(self, step: List[int]):
        """
        Move the window by the given step.
        
        Args:
            step: Step to move in each dimension
        """
        # Update window origin
        self.window_origin = [
            self.window_origin[i] + step[i] 
            for i in range(len(self.window_origin))
        ]
        
        # Update pre-computed coordinates
        for i_coord in range(self.num_coord):
            window_adaptor_coord, bottom_tensor_coord = self.pre_computed_coords[i_coord]
            
            # Move bottom tensor coordinate
            move_tensor_coordinate(
                self.bottom_tensor_view.tensor_desc,
                bottom_tensor_coord,
                step
            )
            
            self.pre_computed_coords[i_coord] = (window_adaptor_coord, bottom_tensor_coord)
    
    def set_window_origin(self, new_origin: List[int]):
        """
        Set a new window origin.
        
        Args:
            new_origin: New origin position
        """
        self.window_origin = list(new_origin)
        self._precompute_coordinates()
    
    def _is_coordinate_valid(self, coord: TensorCoordinate) -> bool:
        """Check if a coordinate is valid within the tensor bounds."""
        # Use the coordinate validation function
        return coordinate_has_valid_offset(
            self.bottom_tensor_view.tensor_desc,
            coord
        )
    
    def get_num_of_access(self) -> int:
        """Get the number of accesses required for the tile."""
        # Simplified: return based on tile size and vector width
        # In practice, would use space-filling curve calculations
        ys_to_d_desc = self.tile_distribution.ys_to_d_descriptor
        total_elements = ys_to_d_desc.get_element_space_size()
        vector_size = 1  # Simplified, would determine optimal vector size
        return (total_elements + vector_size - 1) // vector_size


@dataclass
class TileWindowWithStaticLengths:
    """
    Tile window with static (compile-time known) lengths.
    
    This is a simpler version without distribution, useful for
    single-threaded or uniform access patterns.
    
    Attributes:
        bottom_tensor_view: Underlying tensor view
        window_lengths: Size of the window in each dimension
        window_origin: Origin position of the window in the tensor
    """
    
    bottom_tensor_view: TensorView
    window_lengths: List[int]
    window_origin: List[int]
    
    def __post_init__(self):
        """Validate the tile window."""
        ndim_bottom = self.bottom_tensor_view.get_num_of_dimension()
        ndim_window = len(self.window_lengths)
        
        if ndim_bottom != ndim_window:
            raise ValueError(f"Window dimensions {ndim_window} must match tensor dimensions {ndim_bottom}")
        
        if len(self.window_origin) != ndim_bottom:
            raise ValueError(f"Window origin dimensions must match tensor dimensions {ndim_bottom}")
    
    def get_num_of_dimension(self) -> int:
        """Get number of dimensions."""
        return self.bottom_tensor_view.get_num_of_dimension()
    
    def get_window_lengths(self) -> List[int]:
        """Get window lengths."""
        return self.window_lengths
    
    def get_bottom_tensor_view(self) -> TensorView:
        """Get bottom tensor view."""
        return self.bottom_tensor_view
    
    def get_window_origin(self) -> List[int]:
        """Get window origin."""
        return self.window_origin
    
    def move(self, step: List[int]):
        """
        Move the window by the given step.
        
        Args:
            step: Step to move in each dimension
        """
        self.window_origin = [
            self.window_origin[i] + step[i] 
            for i in range(len(self.window_origin))
        ]
    
    def set_window_origin(self, new_origin: List[int]):
        """
        Set a new window origin.
        
        Args:
            new_origin: New origin position
        """
        self.window_origin = list(new_origin)
    
    def get_element(self, indices: List[int]) -> Any:
        """
        Get an element from the window.
        
        Args:
            indices: Indices within the window
            
        Returns:
            Element value
        """
        # Convert window indices to tensor indices
        tensor_indices = [
            self.window_origin[i] + indices[i]
            for i in range(len(indices))
        ]
        
        # Check bounds
        for i in range(len(indices)):
            if indices[i] < 0 or indices[i] >= self.window_lengths[i]:
                raise IndexError(f"Window index {indices[i]} out of bounds for dimension {i}")
        
        # Get element using tensor view's get_element method
        return self.bottom_tensor_view.get_element(tensor_indices)
    
    def set_element(self, indices: List[int], value: Any):
        """
        Set an element in the window.
        
        Args:
            indices: Indices within the window
            value: Value to set
        """
        # Convert window indices to tensor indices
        tensor_indices = [
            self.window_origin[i] + indices[i]
            for i in range(len(indices))
        ]
        
        # Check bounds
        for i in range(len(indices)):
            if indices[i] < 0 or indices[i] >= self.window_lengths[i]:
                raise IndexError(f"Window index {indices[i]} out of bounds for dimension {i}")
        
        self.bottom_tensor_view[tuple(tensor_indices)] = value


def make_tile_window(tensor_view: TensorView,
                    window_lengths: List[int],
                    origin: List[int],
                    tile_distribution: Optional[TileDistribution] = None,
                    num_coord: int = 1) -> Union[TileWindowWithStaticDistribution, 
                                                 TileWindowWithStaticLengths]:
    """
    Create a tile window.
    
    Args:
        tensor_view: Underlying tensor view
        window_lengths: Size of the window in each dimension
        origin: Origin position of the window
        tile_distribution: Optional tile distribution for distributed access
        num_coord: Number of coordinate bundles (only used with distribution)
        
    Returns:
        TileWindowWithStaticDistribution if distribution provided,
        TileWindowWithStaticLengths otherwise
    """
    if tile_distribution is not None:
        return TileWindowWithStaticDistribution(
            bottom_tensor_view=tensor_view,
            window_lengths=window_lengths,
            window_origin=origin,
            tile_distribution=tile_distribution,
            num_coord=num_coord
        )
    else:
        return TileWindowWithStaticLengths(
            bottom_tensor_view=tensor_view,
            window_lengths=window_lengths,
            window_origin=origin
        )


def move_tile_window(window: Union[TileWindowWithStaticDistribution, TileWindowWithStaticLengths],
                    step: List[int]):
    """
    Move a tile window by the given step.
    
    Args:
        window: Tile window to move
        step: Step to move in each dimension
    """
    window.move(step) 