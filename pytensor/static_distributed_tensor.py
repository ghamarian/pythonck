"""
Python implementation of static_distributed_tensor.hpp from Composable Kernels.

This module provides static distributed tensor functionality for managing
tensor data distributed across processing elements.
"""

from typing import List, Optional, Any, Union, Tuple
import numpy as np
from dataclasses import dataclass, field

from .tile_distribution import TileDistribution, TileDistributedIndex
from .tensor_coordinate import MultiIndex


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
    
    def get_num_of_elements(self) -> int:
        """Get the total number of elements in the thread's local buffer."""
        return len(self.thread_buffer)

    def __getitem__(self, key: Union[int, MultiIndex, TileDistributedIndex, Tuple[TileDistributedIndex, ...], List[TileDistributedIndex]]) -> Any:
        """Get element or slice from thread buffer using various index types."""
        if isinstance(key, int):
            return self.thread_buffer[key]
        elif isinstance(key, MultiIndex):
            # Assuming key is for D-dimensions, calculate flat offset
            offset = self.tile_distribution.ys_to_d_descriptor.calculate_offset(key)
            return self.thread_buffer[offset]
        elif isinstance(key, slice):
            # Allow basic slicing directly on the buffer for simplicity
            return self.thread_buffer[key]
        elif isinstance(key, TileDistributedIndex):
            # Single distributed index - convert to Y indices automatically
            y_indices = self.tile_distribution.get_y_indices_from_distributed_indices([key])
            offset = self.tile_distribution.ys_to_d_descriptor.calculate_offset(y_indices)
            return self.thread_buffer[offset]
        elif isinstance(key, (tuple, list)) and all(isinstance(idx, TileDistributedIndex) for idx in key):
            # Multiple distributed indices - convert to Y indices automatically
            y_indices = self.tile_distribution.get_y_indices_from_distributed_indices(list(key))
            offset = self.tile_distribution.ys_to_d_descriptor.calculate_offset(y_indices)
            return self.thread_buffer[offset]
        else:
            raise TypeError(f"Unsupported key type for __getitem__: {type(key)}")

    def __setitem__(self, key: Union[int, MultiIndex, TileDistributedIndex, Tuple[TileDistributedIndex, ...], List[TileDistributedIndex]], value: Any):
        """Set element in thread buffer using various index types."""
        if isinstance(key, int):
            if key < 0 or key >= len(self.thread_buffer):
                raise IndexError(f"Index {key} out of bounds for thread buffer size {len(self.thread_buffer)}")
            self.thread_buffer[key] = value
        elif isinstance(key, MultiIndex):
            # Assuming key is for D-dimensions, calculate flat offset
            offset = self.tile_distribution.ys_to_d_descriptor.calculate_offset(key)
            if offset < 0 or offset >= len(self.thread_buffer):
                raise IndexError(f"Calculated offset {offset} from MultiIndex {key} is out of bounds for thread buffer size {len(self.thread_buffer)}")
            self.thread_buffer[offset] = value
        elif isinstance(key, TileDistributedIndex):
            # Single distributed index - convert to Y indices automatically
            y_indices = self.tile_distribution.get_y_indices_from_distributed_indices([key])
            offset = self.tile_distribution.ys_to_d_descriptor.calculate_offset(y_indices)
            if offset < 0 or offset >= len(self.thread_buffer):
                raise IndexError(f"Calculated offset {offset} from distributed index {key} is out of bounds")
            self.thread_buffer[offset] = value
        elif isinstance(key, (tuple, list)) and all(isinstance(idx, TileDistributedIndex) for idx in key):
            # Multiple distributed indices - convert to Y indices automatically
            y_indices = self.tile_distribution.get_y_indices_from_distributed_indices(list(key))
            offset = self.tile_distribution.ys_to_d_descriptor.calculate_offset(y_indices)
            if offset < 0 or offset >= len(self.thread_buffer):
                raise IndexError(f"Calculated offset {offset} from distributed indices {key} is out of bounds")
            self.thread_buffer[offset] = value
        else:
            raise TypeError(f"Unsupported key type for __setitem__: {type(key)}")

    def __call__(self, key: Union[int, MultiIndex, TileDistributedIndex, Tuple[TileDistributedIndex, ...], List[TileDistributedIndex]]) -> Any:
        """
        Call-style access to tensor elements (matches C++ operator()).
        
        This provides the same functionality as __getitem__ but with call syntax
        to match the C++ operator() pattern used in sweep_tile examples.
        
        Args:
            key: Index specification (int, MultiIndex, or distributed indices)
            
        Returns:
            Element value at the specified location
        """
        return self.__getitem__(key)

    def get_y_sliced_thread_data(self, y_slice_origins: List[int], y_slice_lengths: List[int]) -> List[Any]:
        """
        Get a slice of the thread buffer based on Y slice origins and lengths.
        
        Args:
            y_slice_origins: List of origins for each Y dimension.
            y_slice_lengths: List of lengths for each Y dimension.
            
        Returns:
            A new list representing the sliced thread buffer.
        """
        # This is a simplified implementation. In practice, you would use a tensor descriptor
        # to calculate offsets and handle multi-dimensional slicing.
        # For now, we assume a flat buffer and simple slicing.
        start = sum(y_slice_origins)
        end = start + sum(y_slice_lengths)
        return self.thread_buffer[start:end].tolist()

    def set_y_sliced_thread_data(self, y_slice_origins: List[int], y_slice_lengths: List[int], sliced_data: List[Any]) -> None:
        """
        Set a slice of the thread buffer based on Y slice origins and lengths.
        
        Args:
            y_slice_origins: List of origins for each Y dimension.
            y_slice_lengths: List of lengths for each Y dimension.
            sliced_data: The data to set in the slice.
        """
        # Simplified implementation. In practice, use a tensor descriptor for offset calculation.
        start = sum(y_slice_origins)
        end = start + sum(y_slice_lengths)
        self.thread_buffer[start:end] = sliced_data

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