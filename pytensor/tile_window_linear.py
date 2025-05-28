"""
Python implementation of tile_window_linear.hpp from Composable Kernels.

This module provides an optimized tile window implementation that uses linear
addressing for improved performance. It pre-caches offsets and flags based on
access patterns to minimize register usage.
"""

from typing import List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

from .tensor_view import TensorView
from .tile_distribution import TileDistribution
from .static_distributed_tensor import StaticDistributedTensor, make_static_distributed_tensor
from .tensor_coordinate import (
    make_tensor_coordinate, move_tensor_coordinate,
    make_tensor_adaptor_coordinate, move_tensor_adaptor_coordinate,
    coordinate_has_valid_offset_assuming_top_index_is_valid,
    MultiIndex, TensorCoordinate
)
from .buffer_view import AddressSpaceEnum
from .tile_window_utils import get_warp_id, get_lane_id, m0_set_with_memory, m0_inc_with_memory


class SpaceFillingCurve:
    """
    Space-filling curve for iterating over multi-dimensional tensor space.
    
    This class generates access patterns for efficient memory access,
    supporting vectorized operations and dimension ordering.
    """
    
    def __init__(self, tensor_lengths: List[int], dim_access_order: List[int],
                 scalars_per_access: List[int], snaked: bool = False):
        """
        Initialize space-filling curve.
        
        Args:
            tensor_lengths: Length in each dimension
            dim_access_order: Order to traverse dimensions
            scalars_per_access: Number of scalars accessed per dimension
            snaked: Whether to use snaked traversal (not supported in linear window)
        """
        self.tensor_lengths = tensor_lengths
        self.dim_access_order = dim_access_order
        self.scalars_per_access = scalars_per_access
        self.snaked = snaked
        self.ndim = len(tensor_lengths)
        
        # Calculate access lengths for each dimension
        self.access_lengths = [
            (length + scalars - 1) // scalars
            for length, scalars in zip(tensor_lengths, scalars_per_access)
        ]
        
        # Calculate total number of accesses
        self.num_access = 1
        for length in self.access_lengths:
            self.num_access *= length
    
    def get_num_of_access(self) -> int:
        """Get total number of accesses."""
        return self.num_access
    
    def get_index(self, access_id: int) -> List[int]:
        """
        Get multi-dimensional index for given access ID.
        
        Args:
            access_id: Linear access ID
            
        Returns:
            Multi-dimensional index
        """
        index = [0] * self.ndim
        remaining = access_id
        
        # Convert linear index to multi-dimensional
        for i in range(self.ndim - 1, -1, -1):
            dim = self.dim_access_order[i]
            index[dim] = remaining % self.access_lengths[dim]
            remaining //= self.access_lengths[dim]
        
        return index
    
    def get_forward_step(self, access_id: int) -> List[int]:
        """
        Get step to next access position.
        
        Args:
            access_id: Current access ID
            
        Returns:
            Step in each dimension
        """
        if access_id >= self.num_access - 1:
            return [0] * self.ndim
        
        current_idx = self.get_index(access_id)
        next_idx = self.get_index(access_id + 1)
        
        return [next_idx[i] - current_idx[i] for i in range(self.ndim)]


@dataclass
class TileWindowLinearTraits:
    """Traits and compile-time constants for linear tile window."""
    
    packed_size: int
    vector_dim_y: int
    scalar_per_vector: int
    num_access: int
    num_access_non_linear: int
    access_map_non_linear: List[int]
    access_histogram_non_linear: List[int]
    access_prefix_sum_non_linear: List[int]
    sfc_ys: SpaceFillingCurve


class TileWindowLinear:
    """
    Optimized tile window with linear addressing.
    
    This implementation pre-caches coordinates and flags for efficient access,
    and uses immediate offsets where possible to save registers.
    """
    
    def __init__(self, bottom_tensor_view: TensorView, window_lengths: List[int],
                 window_origin: List[int], tile_distribution: TileDistribution,
                 linear_bottom_dims: Optional[List[int]] = None):
        """
        Initialize linear tile window.
        
        Args:
            bottom_tensor_view: Underlying tensor view
            window_lengths: Size of the window
            window_origin: Origin of window in tensor
            tile_distribution: Tile distribution pattern
            linear_bottom_dims: Which dimensions use linear addressing (1) vs non-linear (0)
        """
        self.bottom_tensor_view = bottom_tensor_view
        self.window_lengths = window_lengths
        self.window_origin = window_origin.copy()
        self.tile_distribution = tile_distribution
        
        # Set default linear dims based on address space
        if linear_bottom_dims is None:
            ndim = len(window_lengths)
            if bottom_tensor_view.buffer_view.address_space == AddressSpaceEnum.GLOBAL:
                # Global memory: last dimension is linear
                self.linear_bottom_dims = [0] * (ndim - 1) + [1]
            else:
                # LDS: all dimensions are linear
                self.linear_bottom_dims = [1] * ndim
        else:
            self.linear_bottom_dims = linear_bottom_dims
        
        # Initialize traits
        self._init_traits()
        
        # Pre-compute coordinates and flags
        self._precompute_coordinates()
    
    def _init_traits(self):
        """Initialize compile-time traits."""
        # Get dimensions
        self.ndim_bottom = len(self.window_lengths)
        self.ndim_p = self.tile_distribution.ndim_p
        self.ndim_y = self.tile_distribution.ndim_y
        
        # Get adaptors
        self.window_adaptor = self.tile_distribution.ps_ys_to_xs_adaptor
        
        # Calculate vector dimensions and sizes
        self._calculate_vector_info()
        
        # Create space-filling curve
        self._create_space_filling_curve()
        
        # Calculate non-linear access patterns
        self._calculate_non_linear_access_patterns()
    
    def _calculate_vector_info(self):
        """Calculate vectorization information."""
        # For simplicity, use scalar access in Python
        self.packed_size = 1
        self.vector_dim_y = 0
        self.scalar_per_vector = 1
    
    def _create_space_filling_curve(self):
        """Create space-filling curve for access pattern."""
        encoding = self.tile_distribution.encoding
        y_tile_lengths = []
        for y_idx in range(self.ndim_y):
            map_to_major = encoding.ys_to_rhs_major[y_idx]
            map_to_minor = encoding.ys_to_rhs_minor[y_idx]
            if map_to_major > 0:  # Maps to H-dimension map_to_major-1
                # Assuming hs_lengthss[map_to_major-1] is like [length]
                y_tile_lengths.append(encoding.hs_lengthss[map_to_major - 1][0])
            else:  # Maps to R-dimension (map_to_major == 0), R-dim index is map_to_minor
                y_tile_lengths.append(encoding.rs_lengths[map_to_minor])
        
        # Simple dimension order
        dim_access_order = list(range(self.ndim_y))
        
        # Scalars per access (simplified for Python)
        scalars_per_access = [1] * self.ndim_y
        if self.ndim_y > 0: # Protect against empty y_tile_lengths if ndim_y is 0
            scalars_per_access[self.vector_dim_y % self.ndim_y] = self.scalar_per_vector
        else:
            # Handle cases where ndim_y might be 0, though sfc_ys might not be used then
            pass # scalars_per_access remains empty or default handling by SFC
        
        # Create space-filling curve
        self.sfc_ys = SpaceFillingCurve(
            y_tile_lengths, dim_access_order, scalars_per_access, snaked=False
        )
        
        self.num_access = self.sfc_ys.get_num_of_access()
    
    def _calculate_non_linear_access_patterns(self):
        """Calculate access patterns for non-linear dimensions."""
        # Get encoding info
        encoding = self.tile_distribution.encoding
        ys_to_rhs_major = encoding.ys_to_rhs_major
        
        # Build non-linear access map
        access_map = []
        for i in range(self.num_access):
            # Calculate which non-linear coordinate this access uses
            y_indices = self.sfc_ys.get_index(i)
            non_linear_id = 0
            multiplier = 1
            
            for dim_y in range(self.ndim_y - 1, -1, -1):
                rhs_major = ys_to_rhs_major[dim_y]
                target_h_dim = rhs_major - 1  # No R dimension
                
                if target_h_dim < len(self.linear_bottom_dims):
                    if self.linear_bottom_dims[target_h_dim] == 0:  # Non-linear
                        non_linear_id += y_indices[dim_y] * multiplier
                        multiplier *= self.sfc_ys.access_lengths[dim_y]
            
            access_map.append(non_linear_id)
        
        self.access_map_non_linear = access_map
        
        # Calculate histogram and prefix sum
        unique_ids = sorted(set(access_map))
        self.num_access_non_linear = len(unique_ids)
        
        # Build histogram
        histogram = []
        for uid in unique_ids:
            histogram.append(access_map.count(uid))
        self.access_histogram_non_linear = histogram
        
        # Build prefix sum
        prefix_sum = [0]
        for h in histogram:
            prefix_sum.append(prefix_sum[-1] + h)
        self.access_prefix_sum_non_linear = prefix_sum
    
    def _precompute_coordinates(self):
        """Pre-compute coordinates and flags for efficient access."""
        # Get partition index values (P-dimensions)
        idx_p_values = self.tile_distribution.get_partition_index()

        # Sanity check for TileDistribution consistency (optional, but good practice)
        if len(idx_p_values) != self.ndim_p:
            raise RuntimeError(
                f"TileDistribution inconsistency: get_partition_index() returned {len(idx_p_values)} "
                f"elements, but self.ndim_p is {self.ndim_p}."
            )

        # Initial Y-dimension values (all zeros)
        initial_y_values = [0] * self.ndim_y
        
        # Combined (P,Y) initial index values
        combined_py_values = idx_p_values + initial_y_values
        
        # Create the MultiIndex for the adaptor's top dimensions.
        # Its length is self.ndim_p + self.ndim_y.
        idx_top_for_adaptor = MultiIndex(self.ndim_p + self.ndim_y, combined_py_values)

        # Create initial adaptor coordinate.
        # This will raise an error if self.window_adaptor.get_num_of_top_dimension()
        # does not match (self.ndim_p + self.ndim_y), highlighting the
        # inconsistency in TileDistribution setup from the tests.
        window_adaptor_thread_coord = make_tensor_adaptor_coordinate(
            self.window_adaptor, # This is ps_ys_to_xs_adaptor
            idx_top_for_adaptor
        )
        
        # Calculate bottom tensor thread origin
        bottom_tensor_thread_origin_idx_values = [
            self.window_origin[i] + window_adaptor_thread_coord.get_bottom_index()[i]
            for i in range(self.ndim_bottom)
        ]
        idx_top_for_bottom_tensor = MultiIndex(self.ndim_bottom, bottom_tensor_thread_origin_idx_values)

        # Create bottom tensor coordinate using the standard factory function
        # This ensures ndim_hidden and idx_hidden are correctly initialized based on its descriptor
        bottom_tensor_thread_coord = make_tensor_coordinate(
            self.bottom_tensor_view.get_tensor_descriptor(),
            idx_top_for_bottom_tensor
        )
        
        # Initialize cached coordinates and flags
        self.cached_coords = [None] * self.num_access_non_linear
        self.cached_flags = [False] * self.num_access
        
        # Process each access
        for i_access in range(self.num_access):
            # Get non-linear ID
            non_linear_id = self.access_map_non_linear[i_access]
            
            # Save coordinate if needed
            if self.access_prefix_sum_non_linear[non_linear_id] == i_access:
                self.cached_coords[non_linear_id] = bottom_tensor_thread_coord.copy()
            
            # Calculate flag
            self.cached_flags[i_access] = coordinate_has_valid_offset_assuming_top_index_is_valid(
                self.bottom_tensor_view.get_tensor_descriptor(),
                bottom_tensor_thread_coord
            )
            
            # Move to next position if not last access
            if i_access < self.num_access - 1:
                # Get step in Y dimensions
                idx_diff_ys = self.sfc_ys.get_forward_step(i_access)
                
                # Create step in P+Y dimensions
                idx_diff_ps_ys = [0] * self.ndim_p + idx_diff_ys
                
                # Move coordinates
                move_tensor_adaptor_coordinate(
                    self.window_adaptor,
                    window_adaptor_thread_coord,
                    MultiIndex(len(idx_diff_ps_ys), idx_diff_ps_ys)
                )
                
                # Update bottom tensor coordinate
                move_tensor_coordinate(
                    self.bottom_tensor_view.get_tensor_descriptor(),
                    bottom_tensor_thread_coord,
                    MultiIndex(len(idx_diff_ys), idx_diff_ys)
                )
    
    def get_num_of_dimension(self) -> int:
        """Get number of dimensions."""
        return self.ndim_bottom
    
    def get_num_of_access(self) -> int:
        """Get number of memory accesses."""
        return self.num_access
    
    def get_bottom_linear_offset(self, access_id: int) -> int:
        """
        Get linear offset for given access.
        
        Args:
            access_id: Access ID
            
        Returns:
            Linear memory offset
        """
        # Get Y indices for this access
        y_indices = self.sfc_ys.get_index(access_id)
        
        # Calculate linear offset
        # This is simplified - actual implementation would use
        # compile-time optimizations
        offset = 0
        stride = 1
        
        for i in range(self.ndim_y - 1, -1, -1):
            offset += y_indices[i] * stride
            stride *= self.tile_distribution.ys_to_d_descriptor.get_lengths()[i]
        
        return offset
    
    def load(self, oob_conditional_check: bool = True) -> StaticDistributedTensor:
        """
        Load data from tensor into distributed tensor.
        
        Args:
            oob_conditional_check: Whether to check out-of-bounds access
            
        Returns:
            Loaded distributed tensor
        """
        # Create output tensor
        dst_tensor = make_static_distributed_tensor(
            self.bottom_tensor_view.dtype, self.tile_distribution
        )
        
        # Load into tensor
        self.load_into(dst_tensor, oob_conditional_check)
        return dst_tensor
    
    def load_into(self, dst_tensor: StaticDistributedTensor,
                  oob_conditional_check: bool = True):
        """
        Load data into existing distributed tensor.
        
        Args:
            dst_tensor: Destination tensor
            oob_conditional_check: Whether to check bounds
        """
        # Get Y descriptor
        y_desc = self.tile_distribution.ys_to_d_descriptor
        
        # Process each access
        for i_access in range(self.num_access):
            # Get cached coordinate
            non_linear_id = self.access_map_non_linear[i_access]
            coord = self.cached_coords[non_linear_id]
            flag = self.cached_flags[i_access]
            
            # Get linear offset
            linear_offset = self.get_bottom_linear_offset(i_access)
            
            # Read value
            if oob_conditional_check and not flag:
                value = 0  # Invalid access
            else:
                # In real implementation, would use vectorized access
                # Here we simulate scalar access
                idx = coord.get_index()
                value = self.bottom_tensor_view.get_element(idx)
            
            # Store in distributed tensor
            y_indices = self.sfc_ys.get_index(i_access)
            d_offset = y_desc.calculate_offset(y_indices)
            dst_tensor.thread_buffer[d_offset] = value
    
    def store(self, src_tensor: StaticDistributedTensor,
              oob_conditional_check: bool = True):
        """
        Store distributed tensor to memory.
        
        Args:
            src_tensor: Source distributed tensor
            oob_conditional_check: Whether to check bounds
        """
        # Get Y descriptor
        y_desc = self.tile_distribution.ys_to_d_descriptor
        
        # Process each access
        for i_access in range(self.num_access):
            # Get cached coordinate
            non_linear_id = self.access_map_non_linear[i_access]
            coord = self.cached_coords[non_linear_id]
            flag = self.cached_flags[i_access]
            
            # Get value from distributed tensor
            y_indices = self.sfc_ys.get_index(i_access)
            d_offset = y_desc.calculate_offset(y_indices)
            value = src_tensor.thread_buffer[d_offset]
            
            # Write value
            if not oob_conditional_check or flag:
                idx = coord.get_index()
                self.bottom_tensor_view.set_element(idx, value)
    
    def move(self, step: List[int]):
        """
        Move window by given step.
        
        Args:
            step: Step in each dimension
        """
        # Update origin
        for i in range(len(self.window_origin)):
            self.window_origin[i] += step[i]
        
        # Update cached coordinates
        for i in range(self.num_access_non_linear):
            if self.cached_coords[i] is not None:
                move_tensor_coordinate(
                    self.bottom_tensor_view.get_tensor_descriptor(),
                    self.cached_coords[i],
                    MultiIndex(len(step), step)
                )
        
        # Update flags
        for i_access in range(self.num_access):
            non_linear_id = self.access_map_non_linear[i_access]
            coord = self.cached_coords[non_linear_id]
            
            # Calculate flag
            self.cached_flags[i_access] = coordinate_has_valid_offset_assuming_top_index_is_valid(
                self.bottom_tensor_view.get_tensor_descriptor(),
                coord
            )
    
    def set_window_origin(self, new_origin: List[int]):
        """
        Set new window origin.
        
        Args:
            new_origin: New origin coordinates
        """
        self.window_origin = new_origin.copy()
        self._precompute_coordinates()
    
    def load_raw(self, dst_tensor: StaticDistributedTensor,
                 oob_conditional_check: bool = True, pre_nop: bool = False):
        """Raw load operation (same as load in Python)."""
        self.load_into(dst_tensor, oob_conditional_check)
    
    def store_raw(self, src_tensor: StaticDistributedTensor):
        """Raw store operation (same as store in Python)."""
        self.store(src_tensor, oob_conditional_check=True)
    
    def update(self, src_tensor: StaticDistributedTensor,
               oob_conditional_check: bool = True):
        """Update (accumulate) values."""
        # Get Y descriptor
        y_desc = self.tile_distribution.ys_to_d_descriptor
        
        # Process each access
        for i_access in range(self.num_access):
            # Get cached coordinate
            non_linear_id = self.access_map_non_linear[i_access]
            coord = self.cached_coords[non_linear_id]
            flag = self.cached_flags[i_access]
            
            # Get value from distributed tensor
            y_indices = self.sfc_ys.get_index(i_access)
            d_offset = y_desc.calculate_offset(y_indices)
            value = src_tensor.thread_buffer[d_offset]
            
            # Update value
            if not oob_conditional_check or flag:
                idx = coord.get_index()
                current = self.bottom_tensor_view.get_element(idx)
                self.bottom_tensor_view.set_element(idx, current + value)
    
    def update_raw(self, src_tensor: StaticDistributedTensor,
                   oob_conditional_check: bool = True, pre_nop: bool = False):
        """Raw update operation (same as update in Python)."""
        self.update(src_tensor, oob_conditional_check)
    
    def async_load_raw(self, lds_tile_window, oob_conditional_check: bool = True,
                       pre_nop: bool = False):
        """
        Async load operation (simulated).
        
        In GPU code, this would perform asynchronous loads.
        In Python, we simulate with regular loads.
        """
        # Set up m0 register (simulated)
        size_per_buf = self.bottom_tensor_view.get_tensor_descriptor().calculate_offset([0, 0, 0])
        size_per_wave = self.bottom_tensor_view.get_tensor_descriptor().calculate_offset([0, 1, 0]) - size_per_buf
        size_per_issue = self.bottom_tensor_view.get_tensor_descriptor().calculate_offset([1, 0, 0]) - size_per_buf
        
        m0_init = size_per_buf + size_per_wave * get_warp_id()
        m0_set_with_memory(m0_init)
        
        # Simulate async loads
        for i_access in range(self.num_access):
            # In real implementation, would issue async load
            # Here we just increment m0
            if i_access < self.num_access - 1:
                m0_inc_with_memory(size_per_issue)


def make_tile_window_linear(tensor_view: TensorView, window_lengths: List[int],
                           origin: List[int], tile_distribution: TileDistribution,
                           linear_bottom_dims: Optional[List[int]] = None) -> TileWindowLinear:
    """
    Create a linear tile window.
    
    Args:
        tensor_view: Underlying tensor view
        window_lengths: Window size
        origin: Window origin
        tile_distribution: Tile distribution
        linear_bottom_dims: Which dimensions use linear addressing
        
    Returns:
        Linear tile window
    """
    return TileWindowLinear(tensor_view, window_lengths, origin,
                           tile_distribution, linear_bottom_dims)


def is_tile_window_linear(obj: Any) -> bool:
    """Check if object is a linear tile window."""
    return isinstance(obj, TileWindowLinear) 