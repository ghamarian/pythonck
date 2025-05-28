"""
Python implementation of tensor_view.hpp from Composable Kernels.

This module provides tensor view functionality that combines buffer views
with tensor descriptors to provide unified access to tensor data.
"""

from typing import TypeVar, Generic, Optional, Union, Any, Tuple, List
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

from .buffer_view import BufferView, AddressSpaceEnum, MemoryOperationEnum, make_buffer_view
from .tensor_descriptor import TensorDescriptor, make_naive_tensor_descriptor, make_naive_tensor_descriptor_packed
from .tensor_coordinate import TensorCoordinate, make_tensor_coordinate, coordinate_has_valid_offset_assuming_top_index_is_valid, MultiIndex


@dataclass
class TensorView(Generic[TypeVar('T')]):
    """
    Tensor view that abstracts the underlying memory buffer and provides
    unified get/set functions for access.
    
    This combines a buffer view with a tensor descriptor to enable
    coordinate-based access to tensor data.
    
    Attributes:
        buffer_view: The underlying buffer view
        tensor_desc: The tensor descriptor defining the layout
        dst_in_mem_op: Default memory operation for updates
    """
    
    buffer_view: BufferView
    tensor_desc: TensorDescriptor
    dst_in_mem_op: MemoryOperationEnum = MemoryOperationEnum.SET
    
    def __post_init__(self):
        """Validate tensor view after initialization."""
        if self.buffer_view is None:
            raise ValueError("Buffer view cannot be None")
        if self.tensor_desc is None:
            raise ValueError("Tensor descriptor cannot be None")
    
    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Get the tensor descriptor."""
        return self.tensor_desc
    
    @property
    def dtype(self):
        """Get the data type of the tensor."""
        return self.buffer_view.dtype
    
    def get_num_of_dimension(self) -> int:
        """Get number of dimensions."""
        return self.tensor_desc.get_num_of_dimension()
    
    def get_buffer_view(self) -> BufferView:
        """Get the buffer view."""
        return self.buffer_view
    
    def get_vectorized_elements(self, 
                               coord: TensorCoordinate,
                               linear_offset: int = 0,
                               vector_size: int = 1,
                               oob_conditional_check: bool = True) -> Union[Any, np.ndarray]:
        """
        Get vectorized elements from tensor.
        
        Args:
            coord: Tensor coordinate
            linear_offset: Additional linear offset
            vector_size: Number of elements to read
            oob_conditional_check: Whether to check out-of-bounds
            
        Returns:
            Single element or array of elements
        """
        # Check if coordinate is valid
        is_valid = coordinate_has_valid_offset_assuming_top_index_is_valid(self.tensor_desc, coord)
        
        # Get the base offset from coordinate
        base_offset = coord.get_offset()
        
        # Get elements from buffer
        return self.buffer_view.get(
            base_offset,
            linear_offset,
            is_valid,
            vector_size=vector_size,
            oob_conditional_check=oob_conditional_check
        )
    
    def set_vectorized_elements(self,
                               coord: TensorCoordinate,
                               value: Union[Any, np.ndarray],
                               linear_offset: int = 0,
                               vector_size: int = 1,
                               oob_conditional_check: bool = True) -> None:
        """
        Set vectorized elements in tensor.
        
        Args:
            coord: Tensor coordinate
            value: Value(s) to set
            linear_offset: Additional linear offset
            vector_size: Number of elements to write
            oob_conditional_check: Whether to check out-of-bounds
        """
        # Check if coordinate is valid
        is_valid = coordinate_has_valid_offset_assuming_top_index_is_valid(self.tensor_desc, coord)
        
        # Get the base offset from coordinate
        base_offset = coord.get_offset()
        
        # Set elements in buffer
        self.buffer_view.set(
            base_offset,
            linear_offset,
            is_valid,
            value,
            vector_size=vector_size,
            oob_conditional_check=oob_conditional_check
        )
    
    def update_vectorized_elements(self,
                                  coord: TensorCoordinate,
                                  value: Union[Any, np.ndarray],
                                  linear_offset: int = 0,
                                  vector_size: int = 1,
                                  oob_conditional_check: bool = True) -> None:
        """
        Update vectorized elements in tensor using the configured operation.
        
        Args:
            coord: Tensor coordinate
            value: Value(s) to use in update
            linear_offset: Additional linear offset
            vector_size: Number of elements to update
            oob_conditional_check: Whether to check out-of-bounds
        """
        # Check if coordinate is valid
        is_valid = coordinate_has_valid_offset_assuming_top_index_is_valid(self.tensor_desc, coord)
        
        # Get the base offset from coordinate
        base_offset = coord.get_offset()
        
        # Update elements in buffer
        self.buffer_view.update(
            self.dst_in_mem_op,
            base_offset,
            linear_offset,
            is_valid,
            value,
            vector_size=vector_size,
            oob_conditional_check=oob_conditional_check
        )
    
    def get_element(self, idx_top: Union[List[int], MultiIndex, TensorCoordinate]) -> Any:
        """Get a single element from the tensor at the given top-level index."""
        _idx_top_multi_index: MultiIndex
        if isinstance(idx_top, list):
            _idx_top_multi_index = MultiIndex(len(idx_top), idx_top)
        elif isinstance(idx_top, MultiIndex):
            _idx_top_multi_index = idx_top
        elif isinstance(idx_top, TensorCoordinate):
            _idx_top_multi_index = idx_top.get_index()
        else:
            raise TypeError("idx_top must be a List[int], MultiIndex, or TensorCoordinate")

        # Create a new TensorCoordinate for internal use, from the resolved MultiIndex
        internal_coord = make_tensor_coordinate(self.tensor_desc, _idx_top_multi_index)
        
        # For a single element, vector_size is 1 and linear_offset is 0.
        vector_result = self.get_vectorized_elements(internal_coord, linear_offset=0, vector_size=1, oob_conditional_check=True)
        
        if isinstance(vector_result, (list, np.ndarray)):
            if hasattr(vector_result, '__len__') and len(vector_result) == 1:
                return vector_result[0]
            elif hasattr(vector_result, '__len__') and len(vector_result) == 0:
                 raise RuntimeError("get_vectorized_elements returned empty for single element access")
            # If it's a multi-element array but should be scalar (e.g. from vector_size=1 call)
            # and it's a numpy array with a single value, extract it.
            if isinstance(vector_result, np.ndarray) and vector_result.size == 1:
                return vector_result.item() # Safely extract scalar from numpy array
            raise RuntimeError(f"get_vectorized_elements returned unexpected result for single element: {vector_result}")
        return vector_result # Assumed to be scalar already

    def set_element(self, idx_top: Union[List[int], MultiIndex, TensorCoordinate], value: Any):
        """Set a single element in the tensor at the given top-level index."""
        _idx_top_multi_index: MultiIndex
        if isinstance(idx_top, list):
            _idx_top_multi_index = MultiIndex(len(idx_top), idx_top)
        elif isinstance(idx_top, MultiIndex):
            _idx_top_multi_index = idx_top
        elif isinstance(idx_top, TensorCoordinate):
            _idx_top_multi_index = idx_top.get_index()
        else:
            raise TypeError("idx_top must be a List[int], MultiIndex, or TensorCoordinate")

        # Create a new TensorCoordinate for internal use, from the resolved MultiIndex
        internal_coord = make_tensor_coordinate(self.tensor_desc, _idx_top_multi_index)

        self.set_vectorized_elements(internal_coord, value, linear_offset=0, vector_size=1, oob_conditional_check=True)
    
    def __getitem__(self, indices: Union[int, Tuple[int, ...], List[int]]) -> Any:
        """
        Get element using array indexing syntax.
        
        Args:
            indices: Single index, tuple of indices, or list of indices
            
        Returns:
            Element value
        """
        _indices_list: List[int]
        if isinstance(indices, int):
            _indices_list = [indices]
        elif isinstance(indices, tuple):
            _indices_list = list(indices)
        elif isinstance(indices, list):
            _indices_list = indices
        else:
            raise TypeError("Indices for __getitem__ must be int, Tuple[int, ...], or List[int]")
            
        return self.get_element(_indices_list)
    
    def __setitem__(self, indices: Union[int, Tuple[int, ...], List[int]], value: Any) -> None:
        """
        Set element using array indexing syntax.
        
        Args:
            indices: Single index, tuple of indices, or list of indices
            value: Value to set
        """
        _indices_list: List[int]
        if isinstance(indices, int):
            _indices_list = [indices]
        elif isinstance(indices, tuple):
            _indices_list = list(indices)
        elif isinstance(indices, list):
            _indices_list = indices
        else:
            raise TypeError("Indices for __setitem__ must be int, Tuple[int, ...], or List[int]")
            
        self.set_element(_indices_list, value)
    
    def get_element_by_offset(self, offset: int) -> Any:
        """
        Get element by linear offset.
        
        Args:
            offset: Linear offset into the buffer
            
        Returns:
            Element value
        """
        return self.buffer_view.get(offset, 0, True, 1)
    
    def set_element_by_offset(self, offset: int, value: Any) -> None:
        """
        Set element by linear offset.
        
        Args:
            offset: Linear offset into the buffer
            value: Value to set
        """
        self.buffer_view.set(offset, 0, True, value, 1)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"TensorView(ndim={self.get_num_of_dimension()}, "
                f"buffer_size={self.buffer_view.buffer_size}, "
                f"element_space_size={self.tensor_desc.get_element_space_size()})")


class NullTensorView:
    """Placeholder type for opting out a tensor view parameter."""
    pass


def make_tensor_view(data: np.ndarray,
                    tensor_desc: TensorDescriptor,
                    address_space: AddressSpaceEnum = AddressSpaceEnum.GENERIC,
                    dst_in_mem_op: MemoryOperationEnum = MemoryOperationEnum.SET) -> TensorView:
    """
    Create a tensor view from data and tensor descriptor.
    
    Args:
        data: Numpy array containing the data
        tensor_desc: Tensor descriptor defining the layout
        address_space: Memory address space
        dst_in_mem_op: Default memory operation for updates
        
    Returns:
        TensorView instance
    """
    # Create buffer view
    buffer_view = make_buffer_view(
        data=data,
        buffer_size=tensor_desc.get_element_space_size(),
        address_space=address_space
    )
    
    return TensorView(
        buffer_view=buffer_view,
        tensor_desc=tensor_desc,
        dst_in_mem_op=dst_in_mem_op
    )


def make_naive_tensor_view(data: np.ndarray,
                          lengths: List[int],
                          strides: List[int],
                          address_space: AddressSpaceEnum = AddressSpaceEnum.GENERIC,
                          dst_in_mem_op: MemoryOperationEnum = MemoryOperationEnum.SET,
                          guaranteed_last_dim_vector_length: int = -1,
                          guaranteed_last_dim_vector_stride: int = -1) -> TensorView:
    """
    Create a naive tensor view with strided layout.
    
    Args:
        data: Numpy array containing the data
        lengths: Dimension lengths
        strides: Dimension strides
        address_space: Memory address space
        dst_in_mem_op: Default memory operation for updates
        guaranteed_last_dim_vector_length: Guaranteed vector length for last dim
        guaranteed_last_dim_vector_stride: Guaranteed vector stride for last dim
        
    Returns:
        TensorView instance
    """
    # Create tensor descriptor
    tensor_desc = make_naive_tensor_descriptor(
        lengths, strides,
        guaranteed_last_dim_vector_length,
        guaranteed_last_dim_vector_stride
    )
    
    return make_tensor_view(data, tensor_desc, address_space, dst_in_mem_op)


def make_naive_tensor_view_packed(data: np.ndarray,
                                 lengths: List[int],
                                 address_space: AddressSpaceEnum = AddressSpaceEnum.GENERIC,
                                 dst_in_mem_op: MemoryOperationEnum = MemoryOperationEnum.SET,
                                 guaranteed_last_dim_vector_length: int = -1) -> TensorView:
    """
    Create a naive tensor view with packed layout.
    
    Args:
        data: Numpy array containing the data
        lengths: Dimension lengths
        address_space: Memory address space
        dst_in_mem_op: Default memory operation for updates
        guaranteed_last_dim_vector_length: Guaranteed vector length for last dim
        
    Returns:
        TensorView instance
    """
    # Create tensor descriptor
    tensor_desc = make_naive_tensor_descriptor_packed(
        lengths,
        guaranteed_last_dim_vector_length
    )
    
    return make_tensor_view(data, tensor_desc, address_space, dst_in_mem_op)


def transform_tensor_view(old_tensor_view: TensorView,
                         new_transforms: List[Any],
                         new_lower_dimension_old_visible_idss: List[List[int]],
                         new_upper_dimension_new_visible_idss: List[List[int]]) -> TensorView:
    """
    Transform a tensor view by applying new transformations.
    
    Args:
        old_tensor_view: Original tensor view
        new_transforms: New transformations to apply
        new_lower_dimension_old_visible_idss: Lower dimension mappings
        new_upper_dimension_new_visible_idss: Upper dimension mappings
        
    Returns:
        New TensorView with transformed descriptor
    """
    # Transform the tensor descriptor
    from .tensor_descriptor import transform_tensor_descriptor
    
    new_desc = transform_tensor_descriptor(
        old_tensor_view.tensor_desc,
        new_transforms,
        new_lower_dimension_old_visible_idss,
        new_upper_dimension_new_visible_idss
    )
    
    # Create new tensor view with same buffer but new descriptor
    return TensorView(
        buffer_view=old_tensor_view.buffer_view,
        tensor_desc=new_desc,
        dst_in_mem_op=old_tensor_view.dst_in_mem_op
    )


def pad_tensor_view(tensor_view: TensorView,
                   tile_lengths: List[int],
                   do_pads: List[bool]) -> TensorView:
    """
    Pad a tensor view to tile boundaries.
    
    Args:
        tensor_view: Original tensor view
        tile_lengths: Tile dimensions
        do_pads: Which dimensions to pad
        
    Returns:
        New TensorView with padding
    """
    # This is a simplified implementation
    # In practice, this would create appropriate padding transformations
    raise NotImplementedError("pad_tensor_view not yet implemented") 