"""
Python implementation of tile_scatter_gather.hpp from Composable Kernels.

This module provides scatter/gather operations for distributed tensors,
allowing indexed access patterns for reading and writing tensor data.
"""

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np

from .tensor_view import TensorView
from .tile_distribution import TileDistribution
from .static_distributed_tensor import StaticDistributedTensor, make_static_distributed_tensor
from .tensor_coordinate import (
    make_tensor_coordinate, move_tensor_coordinate,
    make_tensor_adaptor_coordinate, move_tensor_adaptor_coordinate
)
from .buffer_view import MemoryOperationEnum


@dataclass
class TileScatterGather:
    """
    Provides scatter/gather access to distributed tensors with page indexing.
    
    This class enables indexed memory access patterns where data can be
    scattered or gathered based on a page index array. This is useful for
    operations like attention mechanisms where access patterns are data-dependent.
    
    Attributes:
        bottom_tensor_view: The underlying tensor view
        window_lengths: Size of the access window
        window_origin: Origin of the window in the tensor
        tile_distribution: Distribution pattern for the tile
        page_idx_array: Array of page indices for scatter/gather
        hs_gather_dim: Dimension for gather operation (default 0)
        ys_gather_dim: Y-dimension for gather operation (default 0)
    """
    
    bottom_tensor_view: TensorView
    window_lengths: List[int]
    window_origin: List[int]
    tile_distribution: TileDistribution
    page_idx_array: np.ndarray
    hs_gather_dim: int = 0
    ys_gather_dim: int = 0
    
    def __post_init__(self):
        """Initialize pre-computed coordinates for efficient access."""
        self.ndim_bottom = len(self.window_lengths)
        self.ndim_p = self.tile_distribution.ndim_p
        self.ndim_y = self.tile_distribution.ndim_y
        
        # Get window adaptor
        self.window_adaptor = self.tile_distribution.ps_ys_to_xs_adaptor
        
        # Pre-compute coordinates
        self._precompute_coordinates()
    
    def _precompute_coordinates(self):
        """Pre-compute thread coordinates for efficient access."""
        # Get partition index (simulated for Python)
        partition_idx = [0] * self.ndim_p
        
        # Create initial adaptor coordinate
        adaptor_top_idx = partition_idx + [0] * self.ndim_y
        window_adaptor_coord = make_tensor_adaptor_coordinate(
            self.window_adaptor, adaptor_top_idx
        )
        
        # Create initial bottom tensor coordinate
        bottom_idx = window_adaptor_coord.get_bottom_index()
        bottom_origin = self.window_origin.copy()
        for i in range(len(bottom_origin)):
            bottom_origin[i] += bottom_idx[i]
        bottom_origin[self.hs_gather_dim] = 0
        
        bottom_tensor_coord = make_tensor_coordinate(
            self.bottom_tensor_view.get_tensor_descriptor(),
            bottom_origin
        )
        
        # Store pre-computed coordinates
        self.window_adaptor_coord = window_adaptor_coord
        self.bottom_tensor_coord = bottom_tensor_coord
    
    def get_num_of_access(self) -> int:
        """Get the number of memory accesses required."""
        # Calculate based on Y-dimension lengths
        y_desc = self.tile_distribution.ys_to_d_descriptor
        y_lengths = y_desc.get_lengths()
        
        num_access = 1
        for length in y_lengths:
            num_access *= length
        
        return num_access
    
    def load(self, oob_conditional_check: bool = True) -> StaticDistributedTensor:
        """
        Load data from tensor using scatter/gather pattern.
        
        Args:
            oob_conditional_check: Whether to check out-of-bounds access
            
        Returns:
            Loaded distributed tensor
        """
        # Create output tensor
        dst_tensor = make_static_distributed_tensor(
            self.bottom_tensor_view.dtype,
            self.tile_distribution
        )
        
        self.load_into(dst_tensor, oob_conditional_check)
        return dst_tensor
    
    def load_into(self, dst_tensor: StaticDistributedTensor,
                  oob_conditional_check: bool = True) -> None:
        """
        Load data into an existing distributed tensor.
        
        Args:
            dst_tensor: Destination distributed tensor
            oob_conditional_check: Whether to check out-of-bounds access
        """
        # Get Y-dimension info
        y_desc = self.tile_distribution.ys_to_d_descriptor
        y_lengths = y_desc.get_lengths()
        
        # Iterate over Y-space
        for y_indices in np.ndindex(*y_lengths):
            y_idx = list(y_indices)
            
            # Get gather index
            gather_idx = y_idx[self.ys_gather_dim]
            page_offset = self.page_idx_array[gather_idx]
            
            # Calculate coordinates
            ps_ys_idx = [0] * self.ndim_p + y_idx
            
            # Move to current position
            window_coord = make_tensor_adaptor_coordinate(
                self.window_adaptor, ps_ys_idx
            )
            
            bottom_idx = window_coord.get_bottom_index()
            tensor_idx = self.window_origin.copy()
            for i in range(len(tensor_idx)):
                if i == self.hs_gather_dim:
                    tensor_idx[i] = page_offset
                else:
                    tensor_idx[i] += bottom_idx[i]
            
            # Read value
            if oob_conditional_check and not self._is_valid_index(tensor_idx):
                value = 0  # Or use invalid element value
            else:
                value = self.bottom_tensor_view.get_element(tensor_idx)
            
            # Store in distributed tensor
            d_offset = y_desc.calculate_offset(y_idx)
            dst_tensor.thread_buffer[d_offset] = value
    
    def store(self, src_tensor: StaticDistributedTensor,
              oob_conditional_check: bool = True) -> None:
        """
        Store distributed tensor data using scatter pattern.
        
        Args:
            src_tensor: Source distributed tensor
            oob_conditional_check: Whether to check out-of-bounds access
        """
        # Get Y-dimension info
        y_desc = self.tile_distribution.ys_to_d_descriptor
        y_lengths = y_desc.get_lengths()
        
        # Iterate over Y-space
        for y_indices in np.ndindex(*y_lengths):
            y_idx = list(y_indices)
            
            # Get scatter index
            scatter_idx = y_idx[self.ys_gather_dim]
            page_offset = self.page_idx_array[scatter_idx]
            
            # Calculate coordinates
            ps_ys_idx = [0] * self.ndim_p + y_idx
            
            # Move to current position
            window_coord = make_tensor_adaptor_coordinate(
                self.window_adaptor, ps_ys_idx
            )
            
            bottom_idx = window_coord.get_bottom_index()
            tensor_idx = self.window_origin.copy()
            for i in range(len(tensor_idx)):
                if i == self.hs_gather_dim:
                    tensor_idx[i] = page_offset
                else:
                    tensor_idx[i] += bottom_idx[i]
            
            # Get value from distributed tensor
            d_offset = y_desc.calculate_offset(y_idx)
            value = src_tensor.thread_buffer[d_offset]
            
            # Write value
            if not oob_conditional_check or self._is_valid_index(tensor_idx):
                self.bottom_tensor_view.set_element(tensor_idx, value)
    
    def move(self, step: List[int]) -> None:
        """
        Move the window origin by the given step.
        
        Args:
            step: Step to move in each dimension
        """
        # Update window origin
        for i in range(len(self.window_origin)):
            if i != self.hs_gather_dim:
                self.window_origin[i] += step[i]
        
        # Re-compute coordinates
        self._precompute_coordinates()
    
    def update_page_idx(self, new_page_idx: np.ndarray) -> None:
        """
        Update the page index array.
        
        Args:
            new_page_idx: New page index array
        """
        self.page_idx_array = new_page_idx.copy()
    
    def set_window_origin(self, new_origin: List[int]) -> None:
        """
        Set a new window origin.
        
        Args:
            new_origin: New origin coordinates
        """
        self.window_origin = new_origin.copy()
        self._precompute_coordinates()
    
    def _is_valid_index(self, idx: List[int]) -> bool:
        """Check if index is valid for the tensor."""
        # Get shape from tensor descriptor
        desc = self.bottom_tensor_view.get_tensor_descriptor()
        shape = [desc.get_length(i) for i in range(desc.get_num_of_dimension())]
        for i, (index, size) in enumerate(zip(idx, shape)):
            if index < 0 or index >= size:
                return False
        return True
    
    def async_load_raw(self, lds_tile_window, 
                      oob_conditional_check: bool = True,
                      pre_nop: bool = False) -> None:
        """
        Async load operation (simulated in Python).
        
        In the C++ version, this performs asynchronous memory loads.
        In Python, we simulate this with regular loads.
        
        Args:
            lds_tile_window: LDS tile window (not used in Python)
            oob_conditional_check: Whether to check bounds
            pre_nop: Whether to insert no-op (not used in Python)
        """
        # In Python, just do a regular load
        # The async behavior is simulated
        dst_tensor = self.load(oob_conditional_check)
        # In real implementation, would copy to LDS window


def make_tile_scatter_gather(
    tensor_view: TensorView,
    window_lengths: List[int],
    origin: List[int],
    tile_distribution: TileDistribution,
    page_idx: np.ndarray,
    hs_gather_dim: int = 0,
    ys_gather_dim: int = 0
) -> TileScatterGather:
    """
    Create a tile scatter/gather accessor.
    
    Args:
        tensor_view: Underlying tensor view
        window_lengths: Size of access window
        origin: Window origin in tensor
        tile_distribution: Tile distribution pattern
        page_idx: Page index array for scatter/gather
        hs_gather_dim: H-space gather dimension
        ys_gather_dim: Y-space gather dimension
        
    Returns:
        TileScatterGather instance
    """
    return TileScatterGather(
        bottom_tensor_view=tensor_view,
        window_lengths=window_lengths,
        window_origin=origin,
        tile_distribution=tile_distribution,
        page_idx_array=page_idx,
        hs_gather_dim=hs_gather_dim,
        ys_gather_dim=ys_gather_dim
    ) 