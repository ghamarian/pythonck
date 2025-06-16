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
from .tile_distribution import (
    TileDistribution, 
    TileDistributedIndex
)
from .static_distributed_tensor import StaticDistributedTensor
from .tensor_coordinate import (
    MultiIndex, TensorCoordinate, TensorAdaptorCoordinate,
    make_tensor_coordinate, move_tensor_coordinate, make_tensor_adaptor_coordinate,
    coordinate_has_valid_offset, move_tensor_adaptor_coordinate
)
from .tensor_descriptor import TensorAdaptor
from .space_filling_curve import SpaceFillingCurve


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
        
        # Initialize load/store traits
        self.traits = LoadStoreTraits(
            tile_distribution=self.tile_distribution,
            data_type=self.bottom_tensor_view.dtype
        )
        
        # Pre-compute coordinates for efficient access
        self._precompute_coordinates()
    
    def _precompute_coordinates(self):
        """Pre-compute coordinate bundles for efficient load/store operations."""
        self.pre_computed_coords = []
        
        # Get partition index and Y dimensions
        partition_idx = self.tile_distribution.get_partition_index()
        ndim_y = self.tile_distribution.ndim_y
        
        # Create initial adaptor coordinate
        # The adaptor expects P+Y dimensions, where P is partition dimensions
        adaptor = self.tile_distribution.ps_ys_to_xs_adaptor
        ndim_top = adaptor.get_num_of_top_dimension()
        
        # Create base coordinates (matches C++ window_adaptor_thread_coord_tmp)
        idx_top_base = [0] * ndim_top
        # Set partition dimensions (P dimensions) - matches C++ detail::get_partition_index()
        for i, p_idx in enumerate(partition_idx):
            idx_top_base[i] = p_idx
        # Y dimensions start at 0 (matches C++ array<index_t, NDimY>{0})
        
        window_adaptor_coord_base = make_tensor_adaptor_coordinate(
            adaptor,
            idx_top_base
        )
        
        # Create base bottom tensor coordinate
        bottom_tensor_idx_base = [
            self.window_origin[i] + window_adaptor_coord_base.get_bottom_index()[i]
            for i in range(len(self.window_origin))
        ]
        bottom_tensor_coord_base = make_tensor_coordinate(
            self.bottom_tensor_view.tensor_desc,
            bottom_tensor_idx_base
        )
        
        # Calculate accesses per coordinate bundle
        num_access_per_coord = self.traits.num_access // self.num_coord
        
        # For each coordinate bundle (matches C++ static_for<0, NumCoord, 1>)
        for i_coord in range(self.num_coord):
            # Copy base coordinates (matches C++ pattern)
            window_adaptor_coord = window_adaptor_coord_base.copy() if hasattr(window_adaptor_coord_base, 'copy') else window_adaptor_coord_base
            bottom_tensor_coord = bottom_tensor_coord_base.copy() if hasattr(bottom_tensor_coord_base, 'copy') else bottom_tensor_coord_base
            
            # Calculate step offset for this coordinate bundle
            # This matches C++: SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{})
            start_access = i_coord * num_access_per_coord
            
            if start_access > 0:
                # Get Y indices for this access
                start_idx_ys = self.traits.get_y_indices(start_access)
                
                # Create step for P+Y dimensions (P dimensions get 0 step)
                ndim_p = len(partition_idx)
                idx_diff_ps_ys = [0] * ndim_p + start_idx_ys
                
                # Move coordinates by the calculated step
                idx_diff_adaptor_bottom = [0] * self.bottom_tensor_view.get_num_of_dimension()
                
                move_tensor_adaptor_coordinate(
                    adaptor,
                    window_adaptor_coord,
                    idx_diff_ps_ys,
                    idx_diff_adaptor_bottom
                )
                
                move_tensor_coordinate(
                    self.bottom_tensor_view.tensor_desc,
                    bottom_tensor_coord,
                    idx_diff_adaptor_bottom
                )
            
            # Store coordinate bundle
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
    
    def _traverse_window(self, process_access, oob_conditional_check: bool = True):
        """
        Generic traversal of the window, applying a processing function at each access.
        
        Args:
            process_access: A function to call for each access. It will receive
                            access_info, bottom_tensor_coord, and ys_to_d_desc.
            oob_conditional_check: Whether to check out-of-bounds access.
        """
        ys_to_d_desc = self.tile_distribution.ys_to_d_descriptor
        num_access_per_coord = self.traits.num_access // self.num_coord
        
        for i_coord in range(self.num_coord):
            window_adaptor_coord = self.pre_computed_coords[i_coord][0].copy()
            bottom_tensor_coord = self.pre_computed_coords[i_coord][1].copy()
            
            for i_coord_access in range(num_access_per_coord):
                i_access = i_coord * num_access_per_coord + i_coord_access
                access_info = self.traits.get_vectorized_access_info(i_access)
                
                if oob_conditional_check and not self._is_coordinate_valid(bottom_tensor_coord):
                    if i_coord_access < num_access_per_coord - 1:
                        self._move_coordinates_for_next_access(
                            window_adaptor_coord, bottom_tensor_coord, i_access,
                            access_info['vector_indices'], access_info['vector_dim']
                        )
                    continue
                
                process_access(access_info, bottom_tensor_coord, ys_to_d_desc)
                
                if i_coord_access < num_access_per_coord - 1:
                    self._move_coordinates_for_next_access(
                        window_adaptor_coord, bottom_tensor_coord, i_access,
                        access_info['vector_indices'], access_info['vector_dim']
                    )

    def _analyze_y_dimensions(self):
        """
        Analyze Y dimensions to determine optimal access patterns.
        Matches C++ load_store_traits::get_vector_dim_y_scalar_per_vector().
        """
        # Get vector lengths and strides for each Y dimension
        ys_vector_lengths = self.tile_distribution.get_y_vector_lengths()
        ys_vector_strides = self.tile_distribution.get_y_vector_strides()
        
        # Find dimension with stride 1 and maximum length (like C++ implementation)
        vector_dim_y = 0
        scalar_per_vector = 1
        
        for i in range(self.tile_distribution.ndim_y):
            if ys_vector_strides[i] == 1 and ys_vector_lengths[i] > scalar_per_vector:
                scalar_per_vector = ys_vector_lengths[i]
                vector_dim_y = i
        
        return {
            'vector_dim_y': vector_dim_y,
            'scalar_per_vector': scalar_per_vector,
            'vector_strides': ys_vector_strides
        }
    
    def load_into(self, dst_tensor: StaticDistributedTensor, 
                  oob_conditional_check: bool = True):
        """
        Load data from the window into an existing distributed tensor.
        
        Args:
            dst_tensor: Destination distributed tensor
            oob_conditional_check: Whether to check out-of-bounds access
        """
        def process_load(access_info, bottom_tensor_coord, ys_to_d_desc):
            offset = bottom_tensor_coord.get_offset()
            values = [self.bottom_tensor_view.get_element_by_offset(offset + j) for j in range(access_info['vector_size'])]
            
            for j, idx_ys in enumerate(access_info['vector_indices']):
                d_offset = ys_to_d_desc.calculate_offset(idx_ys)
                dst_tensor.set_thread_data(d_offset, values[j])

        self._traverse_window(process_load, oob_conditional_check)
    
    def _move_coordinates_for_next_access(self, window_adaptor_coord, bottom_tensor_coord, 
                                        i_access, vector_indices, vector_dim):
        """
        Move coordinates to the next access position using space-filling curve step.
        """
        # Get next access indices
        next_access_info = self.traits.get_vectorized_access_info(i_access + 1)
        
        # Calculate step difference in Y dimensions
        # This logic now works for both single and multi-Y-dimension cases
        idx_diff_ys = [
            next_access_info['base_indices'][i] - vector_indices[0][i]
            for i in range(self.tile_distribution.ndim_y)
        ]
        
        # Create step for P+Y dimensions (P dimensions get 0 step)
        partition_idx = self.tile_distribution.get_partition_index()
        ndim_p = len(partition_idx)
        idx_diff_ps_ys = [0] * ndim_p + idx_diff_ys
        
        # Move window adaptor coordinate
        idx_diff_adaptor_bottom = [0] * self.bottom_tensor_view.get_num_of_dimension()
        
        move_tensor_adaptor_coordinate(
            self.tile_distribution.ps_ys_to_xs_adaptor,
            window_adaptor_coord,
            idx_diff_ps_ys,
            idx_diff_adaptor_bottom
        )
        
        # Move bottom tensor coordinate
        move_tensor_coordinate(
            self.bottom_tensor_view.tensor_desc,
            bottom_tensor_coord,
            idx_diff_adaptor_bottom
        )
    
    def _gather_values_from_src(self, src_tensor, access_info, ys_to_d_desc):
        """Gathers values from the source distributed tensor for a vectorized access."""
        values = []
        for j, idx_ys in enumerate(access_info['vector_indices']):
            d_offset = ys_to_d_desc.calculate_offset(idx_ys)
            value = src_tensor.get_thread_data(d_offset)
            values.append(value)
        return values

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
        def process_store(access_info, bottom_tensor_coord, ys_to_d_desc):
            values = self._gather_values_from_src(src_tensor, access_info, ys_to_d_desc)
            offset = bottom_tensor_coord.get_offset()
            for j in range(access_info['vector_size']):
                self.bottom_tensor_view.set_element_by_offset(offset + j, values[j])

        self._traverse_window(process_store, oob_conditional_check)
    
    def store_raw(self, src_tensor: StaticDistributedTensor, oob_conditional_check: bool = True):
        """
        Store data using raw memory operations.
        
        Args:
            src_tensor: Source distributed tensor
            oob_conditional_check: Whether to check out-of-bounds access
        """
        # Similar to store but with raw access patterns
        self.store(src_tensor, oob_conditional_check)
    
    def update(self, src_tensor: StaticDistributedTensor,
               oob_conditional_check: bool = True):
        """
        Update (accumulate) data from a distributed tensor into the window.
        
        Args:
            src_tensor: Source distributed tensor
            oob_conditional_check: Whether to check out-of-bounds access
        """
        def process_update(access_info, bottom_tensor_coord, ys_to_d_desc):
            values = self._gather_values_from_src(src_tensor, access_info, ys_to_d_desc)
            offset = bottom_tensor_coord.get_offset()
            for j in range(access_info['vector_size']):
                current_value = self.bottom_tensor_view.get_element_by_offset(offset + j)
                self.bottom_tensor_view.set_element_by_offset(offset + j, current_value + values[j])

        self._traverse_window(process_update, oob_conditional_check)
    
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
        # Use d_length directly from the ys_to_d_descriptor
        # This matches the C++ implementation which uses the d_length
        # calculated during tile distribution creation
        ys_to_d_desc = self.tile_distribution.ys_to_d_descriptor
        
        # Both approaches are equivalent:
        # 1. Use get_element_space_size() which returns the stored d_length
        # 2. Calculate d_length directly from Y lengths (product of all Y dimensions)
        # 
        # The C++ code calculates d_length as: d_length *= y_length for each Y dimension
        # and stores it in the tensor descriptor's element_space_size field
        
        # Method 1: Direct calculation (matches C++ calculation logic)
        y_lengths = ys_to_d_desc.get_lengths()
        d_length = 1
        for length in y_lengths:
            d_length *= length
        
        # Method 2: Use stored value (equivalent result)
        # d_length = ys_to_d_desc.get_element_space_size()
        
        return d_length


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


class LoadStoreTraits:
    """
    Python equivalent of C++ load_store_traits.
    Handles vectorization and access patterns for tile windows.
    """
    
    def __init__(self, tile_distribution, data_type):
        self.tile_distribution = tile_distribution
        self.data_type = data_type
        self.ndim_y = tile_distribution.ndim_y
        
        # Initialize vectorization info
        vector_info = self._get_vector_dim_y_scalar_per_vector()
        self.vector_dim_y = vector_info[0]
        self.scalar_per_vector = vector_info[1]
        
        # PackedSize is always 1 in Python (no SIMD)
        self.packed_size = 1
        
        # Create space-filling curve
        self.scalars_per_access = self._get_scalars_per_access()
        self.sfc_ys = self._get_space_filling_curve()
        
        # Calculate number of accesses
        self.num_access = self._calculate_num_access()
    
    def _get_vector_dim_y_scalar_per_vector(self):
        """
        Find best Y dimension for vectorization.
        Matches C++ get_vector_dim_y_scalar_per_vector().
        """
        # Get vector lengths and strides
        ys_vector_lengths = self.tile_distribution.get_y_vector_lengths()
        ys_vector_strides = self.tile_distribution.get_y_vector_strides()
        
        # Find dimension with stride 1 and maximum length
        vector_dim_y = 0
        scalar_per_vector = 1
        
        for i in range(self.ndim_y):
            if ys_vector_strides[i] == 1 and ys_vector_lengths[i] > scalar_per_vector:
                scalar_per_vector = ys_vector_lengths[i]
                vector_dim_y = i
        
        return (vector_dim_y, scalar_per_vector)
    
    def _get_scalars_per_access(self):
        """
        Get number of scalars to access per dimension.
        Matches C++ scalars_per_access_.
        """
        if self.ndim_y == 1:
            return [self.scalar_per_vector]
        else:
            scalars_per_access = [1] * self.ndim_y
            scalars_per_access[self.vector_dim_y] = self.scalar_per_vector
            return scalars_per_access
    
    def _get_space_filling_curve(self):
        """
        Create space-filling curve for memory access pattern.
        Matches C++ get_space_filling_curve().
        """
        # Get Y dimension lengths
        y_lengths = self.tile_distribution.ys_to_d_descriptor.get_lengths()
        
        # Create dimension access order (simple sequential for now)
        dim_access_order = list(range(self.ndim_y))
        
        # Create space-filling curve
        from .space_filling_curve import SpaceFillingCurve
        return SpaceFillingCurve(
            tensor_lengths=y_lengths,
            dim_access_order=dim_access_order,
            scalars_per_access=self.scalars_per_access
        )
    
    def _calculate_num_access(self):
        """
        Calculate total number of accesses needed.
        Matches C++ NumAccess calculation.
        """
        return self.sfc_ys.get_num_of_access()
    
    def get_y_indices(self, i_access):
        """
        Get Y indices for a given access index.
        Matches C++ SFC_Ys::get_index().
        """
        return self.sfc_ys.get_index(i_access)
    
    def get_vectorized_access_info(self, i_access):
        """
        Get vectorized access information for a given access index.
        This combines several C++ operations into one Python-friendly interface.
        """
        # Get base Y indices
        idx_ys_start = self.get_y_indices(i_access)
        
        # Ensure idx_ys_start is a list
        if not isinstance(idx_ys_start, list):
            idx_ys_start = [idx_ys_start]
            
        if self.ndim_y == 1:
            # Single Y dimension case
            base_indices = idx_ys_start
            vector_indices = [[base_indices[0] + j] for j in range(self.scalar_per_vector)]
            return {
                'base_indices': base_indices,
                'vector_indices': vector_indices,
                'vector_dim': 0,
                'vector_size': self.scalar_per_vector
            }
        else:
            # Multiple Y dimensions case
            vector_indices = []
            for j in range(self.scalar_per_vector):
                idx_ys = list(idx_ys_start)  # Copy base indices
                # Modify vector dimension index
                idx_ys[self.vector_dim_y] = idx_ys_start[self.vector_dim_y] + j
                vector_indices.append(idx_ys)
            
            return {
                'base_indices': idx_ys_start,
                'vector_indices': vector_indices,
                'vector_dim': self.vector_dim_y,
                'vector_size': self.scalar_per_vector
            }


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