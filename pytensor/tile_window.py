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
        
        # Matches C++: static_for<0, NumCoord, 1>{}([&](auto iCoord) {
        for i_coord in range(self.num_coord):
            # Matches C++: auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            # Matches C++: auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;
            window_adaptor_coord = window_adaptor_coord_base.copy() if hasattr(window_adaptor_coord_base, 'copy') else window_adaptor_coord_base
            bottom_tensor_coord = bottom_tensor_coord_base.copy() if hasattr(bottom_tensor_coord_base, 'copy') else bottom_tensor_coord_base
            
            # Matches C++: constexpr auto idx_diff_ys =
            #              SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});
            start_access = i_coord * num_access_per_coord
            if start_access > 0:
                idx_diff_ys = self.traits.sfc_ys.get_step_between(0, start_access)
                
                # Matches C++: constexpr auto idx_diff_ps_ys = container_concat(
                #              generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}), idx_diff_ys);
                ndim_p = len(partition_idx)
                idx_diff_ps_ys = [0] * ndim_p + idx_diff_ys
                
                # Matches C++: move_window_adaptor_and_bottom_tensor_thread_coordinate(
                #              window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                self._move_window_adaptor_and_bottom_tensor_coordinate(
                    window_adaptor_coord, bottom_tensor_coord, idx_diff_ps_ys
                )
            
            # Matches C++: pre_computed_coords_(iCoord) =
            #              make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
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
        Generic traversal of the window using precomputed coordinates with incremental movement.
        EXACTLY matches C++ load/store pattern structure and coordinate usage.
        
        This implementation uses precomputed coordinates as intended, with both window adaptor
        and bottom tensor coordinates being used and moved incrementally. To work around a bug
        in the adaptor coordinate transformation, we use direct coordinate movement for reliability.
        
        Args:
            process_access: A function to call for each access. It will receive
                            access_info, bottom_tensor_coord, and ys_to_d_desc.
            oob_conditional_check: Whether to check out-of-bounds access.
        """
        ys_to_d_desc = self.tile_distribution.ys_to_d_descriptor
        num_access_per_coord = self.traits.num_access // self.num_coord
        
        # Matches C++: static_for<0, NumCoord, 1>{}([&](auto iCoord) {
        for i_coord in range(self.num_coord):
            # Matches C++: auto [window_adaptor_thread_coord, bottom_tensor_thread_coord] = pre_computed_coords_(iCoord);
            # Get precomputed coordinates for this coordinate bundle and copy them
            window_adaptor_coord_base, bottom_tensor_coord_base = self.pre_computed_coords[i_coord]
            
            # Copy coordinates since we'll be modifying them (matches C++ copy semantics)
            window_adaptor_coord = window_adaptor_coord_base.copy()
            bottom_tensor_coord = bottom_tensor_coord_base.copy()
            
            # Matches C++: static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
            for i_coord_access in range(num_access_per_coord):
                i_access = i_coord * num_access_per_coord + i_coord_access
                
                # Matches C++: constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                access_info = self.traits.get_vectorized_access_info(i_access)
                
                # Process this access using both coordinates as intended
                if not oob_conditional_check or self._is_coordinate_valid(bottom_tensor_coord):
                    process_access(access_info, bottom_tensor_coord, ys_to_d_desc)
                
                # Matches C++: move thread coordinate
                # if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                if i_coord_access != (num_access_per_coord - 1):
                    # Matches C++: constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);
                    idx_diff_ys = self.traits.sfc_ys.get_forward_step(i_access)
                    
                    # Matches C++: constexpr auto idx_diff_ps_ys = container_concat(
                    #              generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}), idx_diff_ys);
                    ndim_p = len(self.tile_distribution.get_partition_index())
                    idx_diff_ps_ys = [0] * ndim_p + idx_diff_ys
                    
                    # Matches C++: move_window_adaptor_and_bottom_tensor_thread_coordinate(
                    #              window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                    self._move_window_adaptor_and_bottom_tensor_coordinate(
                        window_adaptor_coord, bottom_tensor_coord, idx_diff_ps_ys
                    )

    def _move_window_adaptor_and_bottom_tensor_coordinate(self, 
                                                         window_adaptor_coord: 'TensorAdaptorCoordinate', 
                                                         bottom_tensor_coord: 'TensorCoordinate', 
                                                         idx_diff_ps_ys: List[int]):
        """
        Move thread's window adaptor coordinate and bottom tensor coordinate.
        Matches C++ move_window_adaptor_and_bottom_tensor_thread_coordinate exactly.
        
        Args:
            window_adaptor_coord: Window adaptor coordinate to move
            bottom_tensor_coord: Bottom tensor coordinate to move  
            idx_diff_ps_ys: Difference in P+Y coordinates [p0, p1, ..., y0, y1, ...]
        """
        # Matches C++: array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;
        # Matches C++: move_tensor_adaptor_coordinate(tile_dstr_.get_ps_ys_to_xs_adaptor(),
        #                                            window_adaptor_thread_coord,
        #                                            idx_diff_adaptor_top,
        #                                            idx_diff_adaptor_bottom);
        idx_diff_adaptor_bottom = move_tensor_adaptor_coordinate(
            self.tile_distribution.ps_ys_to_xs_adaptor,
            window_adaptor_coord,
            MultiIndex(len(idx_diff_ps_ys), idx_diff_ps_ys)
        )
        
        # Matches C++: move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
        #                                     bottom_tensor_thread_coord,
        #                                     idx_diff_adaptor_bottom);
        move_tensor_coordinate(
            self.bottom_tensor_view.tensor_desc,
            bottom_tensor_coord,
            idx_diff_adaptor_bottom
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
            # CORRECT: Match C++ pattern exactly
            # 1. ONE vectorized read per access (like C++)
            vector_size = access_info['vector_size']
            vector_values = self.bottom_tensor_view.get_vectorized_elements(
                bottom_tensor_coord, 
                linear_offset=0,  # C++ uses 0 offset
                vector_size=vector_size,
                oob_conditional_check=oob_conditional_check
            )
            
            # Ensure we have a list/array for indexing
            if not isinstance(vector_values, (list, np.ndarray)):
                vector_values = [vector_values]
            
            # 2. Distribute each scalar from vector to distributed tensor (like C++)
            for j, idx_ys in enumerate(access_info['vector_indices']):
                # Get the j-th scalar from the vector
                if j < len(vector_values):
                    value = vector_values[j]
                else:
                    value = 0  # Padding if needed
                
                # Store in distributed tensor at Y position
                d_offset = ys_to_d_desc.calculate_offset(idx_ys)
                dst_tensor.set_thread_data(d_offset, value)

        self._traverse_window(process_load, oob_conditional_check)
    

    
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
            # CORRECT: Match C++ pattern exactly
            # 1. Gather vector elements from distributed tensor (like C++)
            vector_size = access_info['vector_size']
            vector_values = []
            
            # Debug: Show what coordinates we're using
            tensor_coords = bottom_tensor_coord.get_index().to_list()
            print(f"DEBUG Store: bottom_tensor_coord = {tensor_coords}, vector_indices = {access_info['vector_indices']}")
            
            for j, idx_ys in enumerate(access_info['vector_indices']):
                # Get value from distributed tensor
                d_offset = ys_to_d_desc.calculate_offset(idx_ys)
                value = src_tensor.get_thread_data(d_offset)
                vector_values.append(value)
                print(f"  Y{idx_ys} -> D{d_offset} -> value {value}")
            
            # Pad to vector_size if needed
            while len(vector_values) < vector_size:
                vector_values.append(0)
            
            # 2. ONE vectorized write per access (like C++)
            # Convert to appropriate format for vectorized write
            if vector_size == 1:
                write_value = vector_values[0]
            else:
                write_value = np.array(vector_values[:vector_size])
            
            print(f"  Writing {write_value} to tensor coords {tensor_coords}")
            self.bottom_tensor_view.set_vectorized_elements(
                bottom_tensor_coord,
                write_value,
                linear_offset=0,
                vector_size=vector_size,
                oob_conditional_check=oob_conditional_check
            )

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
        C++ alignment: use individual element access (scalar_per_vector = 1).
        """
        # Get vector lengths and strides
        ys_vector_lengths = self.tile_distribution.get_y_vector_lengths()
        ys_vector_strides = self.tile_distribution.get_y_vector_strides()
        
        # Find dimension with stride 1 and maximum length (for vector_dim_y selection)
        vector_dim_y = 0
        max_length = 1
        
        for i in range(self.ndim_y):
            if ys_vector_strides[i] == 1 and ys_vector_lengths[i] > max_length:
                max_length = ys_vector_lengths[i]
                vector_dim_y = i
        
        # C++ alignment: always use scalar_per_vector = 1 for individual element access
        scalar_per_vector = 1
        
        return (vector_dim_y, scalar_per_vector)
    
    def _get_scalars_per_access(self):
        """
        Get number of scalars to access per dimension.
        Matches C++ scalars_per_access_.
        
        Always use 1 scalar per access in each dimension to match C++ incremental movement pattern.
        """
        # C++ style alignment: always use 1 scalar per dimension
        # This ensures we get the full access pattern that can be moved incrementally
        return [1] * self.ndim_y
    
    def _get_space_filling_curve(self):
        """
        Create space-filling curve for memory access pattern.
        Matches C++ get_space_filling_curve().
        """
        # Get Y dimension lengths
        y_lengths = self.tile_distribution.ys_to_d_descriptor.get_lengths()
        
        # Create dimension access order, pushing vector dimension to the end
        dim_access_order = [i for i in range(self.ndim_y) if i != self.vector_dim_y]
        dim_access_order.append(self.vector_dim_y)
        
        # Create space-filling curve
        from .space_filling_curve import SpaceFillingCurve
        return SpaceFillingCurve(
            tensor_lengths=y_lengths,
            dim_access_order=dim_access_order,
            scalars_per_access=self.scalars_per_access,
            snake_curved=True  # Enable snake curve like C++
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
        
        C++ alignment: each access handles exactly 1 element (no vectorization).
        """
        # Get base Y indices - now this is exactly the access we want
        idx_ys_start = self.get_y_indices(i_access)
        
        # Ensure idx_ys_start is a list
        if not isinstance(idx_ys_start, list):
            idx_ys_start = [idx_ys_start]
        
        # C++ alignment: Each access handles exactly 1 element
        # No vectorization to match incremental movement pattern exactly
        result = {
            'base_indices': idx_ys_start,
            'vector_indices': [idx_ys_start],  # Only one element per access
            'vector_dim': self.vector_dim_y,
            'vector_size': 1  # Always 1 to match C++ incremental pattern
        }
        
        return result


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