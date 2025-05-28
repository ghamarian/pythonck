"""
Python implementation of static_distributed_tensor.hpp from Composable Kernels.

This module provides static distributed tensor functionality for managing
tensor data distributed across processing elements.
"""

from typing import List, Optional, Any, Union
import numpy as np
from dataclasses import dataclass, field

from .tile_distribution import TileDistribution


@dataclass
class StaticDistributedTensor:
    """
    Static distributed tensor that holds data distributed across processing elements.
    
    This class manages tensor data that is distributed according to a specific
    tile distribution pattern. Each processing element (thread) holds a portion
    of the tensor data.
    
    Attributes:
        data_type: Data type of tensor elements
        tile_distribution: Distribution pattern for the tensor
        thread_buffer: Local data buffer for this thread
    """
    
    data_type: type
    tile_distribution: TileDistribution
    thread_buffer: Optional[np.ndarray] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize the thread buffer."""
        # Get the size of data this thread holds
        ys_to_d_desc = self.tile_distribution.ys_to_d_descriptor
        element_space_size = ys_to_d_desc.get_element_space_size()
        
        # Create thread buffer
        self.thread_buffer = np.zeros(element_space_size, dtype=self.data_type)
    
    def get_thread_buffer(self) -> np.ndarray:
        """Get the thread's local data buffer."""
        return self.thread_buffer
    
    def get_thread_data(self, index: int) -> Any:
        """
        Get data at specified index in thread buffer.
        
        Args:
            index: Index in thread buffer
            
        Returns:
            Data value at index
        """
        return self.thread_buffer[index]
    
    def set_thread_data(self, index: int, value: Any):
        """
        Set data at specified index in thread buffer.
        
        Args:
            index: Index in thread buffer
            value: Value to set
        """
        self.thread_buffer[index] = value
    
    def get_element(self, y_indices: List[int]) -> Any:
        """
        Get element using Y-dimension indices.
        
        Args:
            y_indices: Indices in Y dimensions
            
        Returns:
            Element value
        """
        # Convert Y indices to D (linearized) index
        ys_to_d_desc = self.tile_distribution.ys_to_d_descriptor
        d_index = ys_to_d_desc.calculate_offset(y_indices)
        
        return self.thread_buffer[d_index]
    
    def set_element(self, y_indices: List[int], value: Any):
        """
        Set element using Y-dimension indices.
        
        Args:
            y_indices: Indices in Y dimensions
            value: Value to set
        """
        # Convert Y indices to D (linearized) index
        ys_to_d_desc = self.tile_distribution.ys_to_d_descriptor
        d_index = ys_to_d_desc.calculate_offset(y_indices)
        
        self.thread_buffer[d_index] = value
    
    def get_slice(self, y_slice: Union[slice, List[slice]]) -> np.ndarray:
        """
        Get a slice of the distributed tensor.
        
        Args:
            y_slice: Slice specification for Y dimensions
            
        Returns:
            Numpy array containing the slice
        """
        # This is a simplified implementation
        # In practice, would need to handle complex slicing across distributed data
        return self.thread_buffer[y_slice]
    
    def clear(self):
        """Clear the thread buffer (set to zeros)."""
        self.thread_buffer.fill(0)
    
    def fill(self, value: Any):
        """
        Fill the thread buffer with a value.
        
        Args:
            value: Value to fill with
        """
        self.thread_buffer.fill(value)
    
    def copy_from(self, other: 'StaticDistributedTensor'):
        """
        Copy data from another distributed tensor.
        
        Args:
            other: Source distributed tensor
        """
        if self.thread_buffer.shape != other.thread_buffer.shape:
            raise ValueError("Cannot copy between tensors with different shapes")
        
        np.copyto(self.thread_buffer, other.thread_buffer)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"StaticDistributedTensor(dtype={self.data_type.__name__}, "
                f"size={len(self.thread_buffer)})")


def make_static_distributed_tensor(data_type: type,
                                 tile_distribution: TileDistribution) -> StaticDistributedTensor:
    """
    Create a static distributed tensor.
    
    Args:
        data_type: Data type of tensor elements
        tile_distribution: Distribution pattern for the tensor
        
    Returns:
        StaticDistributedTensor instance
    """
    return StaticDistributedTensor(
        data_type=data_type,
        tile_distribution=tile_distribution
    ) 