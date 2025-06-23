"""
Python implementation of tile_distribution.hpp from Composable Kernels.

This module provides tile distribution functionality for distributing
tensor data across processing elements (threads, warps, blocks).
"""

from typing import List, Tuple, Optional, Union, Any, Dict
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .tensor_descriptor import TensorAdaptor, TensorDescriptor
from .tensor_coordinate import MultiIndex, TensorAdaptorCoordinate, make_tensor_adaptor_coordinate
from .tile_distribution_encoding import TileDistributionEncoding


@dataclass
class TileDistributedSpan:
    """
    Distributed span representing partial lengths in each dimension.
    
    Attributes:
        partial_lengths: Partial lengths for each dimension
    """
    partial_lengths: List[int]
    
    def is_static(self) -> bool:
        """Check if span is static (compile-time known)."""
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TileDistributedSpan({self.partial_lengths})"


@dataclass
class TileDistributedIndex:
    """
    Distributed index representing partial indices in each dimension.
    
    Attributes:
        partial_indices: Partial indices for each dimension
    """
    partial_indices: List[int]
    
    def is_static(self) -> bool:
        """Check if index is static (compile-time known)."""
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TileDistributedIndex({self.partial_indices})"


@dataclass
class TileDistribution:
    """
    Tile distribution that manages how tensor data is distributed across
    processing elements.
    
    Attributes:
        ps_ys_to_xs_adaptor: Adaptor mapping from (P,Y) to X dimensions
        ys_to_d_descriptor: Descriptor mapping from Y to linearized D dimension
        encoding: Static tile distribution encoding
    """
    
    ps_ys_to_xs_adaptor: TensorAdaptor
    ys_to_d_descriptor: TensorDescriptor
    encoding: TileDistributionEncoding
    
    @property
    def ndim_x(self) -> int:
        """Number of X dimensions."""
        return self.ps_ys_to_xs_adaptor.get_num_of_bottom_dimension()
    
    @property
    def ndim_y(self) -> int:
        """Number of Y dimensions."""
        return self.ys_to_d_descriptor.get_num_of_top_dimension()
    
    @property
    def ndim_p(self) -> int:
        """Number of P dimensions."""
        return self.ps_ys_to_xs_adaptor.get_num_of_top_dimension() - self.ndim_y
    
    @property
    def ndim_r(self) -> int:
        """Number of R dimensions."""
        return self.encoding.ndim_r
    
    def get_partition_index(self) -> List[int]:
        """
        Get current partition index.
        
        This matches the C++ implementation which directly calls hardware intrinsics:
        - For NDimP == 1: returns [get_lane_id()]
        - For NDimP == 2: returns [get_warp_id(), get_lane_id()]
        """
        # Import here to avoid circular imports
        try:
            from .tile_window_utils import get_warp_id, get_lane_id
            
            if self.ndim_p == 1:
                return [get_lane_id()]
            elif self.ndim_p == 2:
                return [get_warp_id(), get_lane_id()]
            else:
                # For other cases, return zeros (fallback)
                return [0] * self.ndim_p
        except ImportError:
            # Fallback if tile_window_utils is not available
            return [0] * self.ndim_p
    
    def calculate_index(self, partition_index: Optional[List[int]] = None) -> MultiIndex:
        """
        Calculate X index from partition index.
        
        Args:
            partition_index: Partition index (uses current if None)
            
        Returns:
            X dimension index
        """
        if partition_index is None:
            partition_index = self.get_partition_index()
        
        # Concatenate partition index with zeros for Y dimensions
        ps_ys_idx = partition_index + [0] * self.ndim_y
        
        # Create coordinate and get bottom index
        coord = make_tensor_adaptor_coordinate(
            self.ps_ys_to_xs_adaptor,
            MultiIndex(len(ps_ys_idx), ps_ys_idx)
        )
        
        return coord.get_bottom_index()
    
    def calculate_rs_index_from_ps_index(self, partition_index: Optional[List[int]] = None) -> List[int]:
        """
        Calculate replication index from partition index.
        
        Args:
            partition_index: Partition index (uses current if None)
            
        Returns:
            Replication index
        """
        if partition_index is None:
            partition_index = self.get_partition_index()
        
        # Create dummy coordinate to extract R indices
        ps_ys_idx = partition_index + [0] * self.ndim_y
        coord = make_tensor_adaptor_coordinate(
            self.ps_ys_to_xs_adaptor,
            MultiIndex(len(ps_ys_idx), ps_ys_idx)
        )
        
        # Extract R indices from hidden dimensions
        rs_idx = [0] * self.ndim_r
        
        # This is a simplified version - in practice would need full mapping
        # from encoding to extract R indices from hidden dimensions
        
        return rs_idx
    
    def get_lengths(self) -> List[int]:
        """Get X dimension lengths."""
        lengths = []
        for x_idx in range(self.ndim_x):
            # Calculate product of H lengths for this X dimension
            h_lengths = self.encoding.hs_lengthss[x_idx]
            x_length = 1
            for h_len in h_lengths:
                x_length *= h_len
            lengths.append(x_length)
        return lengths
    
    def get_distributed_spans(self) -> List[TileDistributedSpan]:
        """Get distributed spans for each X dimension."""
        return self.encoding.get_distributed_spans()
    
    def get_y_indices_from_distributed_indices(self, distributed_indices: Union[int, List[TileDistributedIndex]]) -> List[int]:
        """
        Get Y indices from either an access index or distributed indices.
        This matches C++ get_y_indices_from_distributed_indices().
        
        Args:
            distributed_indices: Either:
                - An integer access index for direct Y calculation
                - List of distributed indices for each X dimension
            
        Returns:
            List of Y dimension indices
        """
        y_indices = [0] * self.ndim_y
        
        if isinstance(distributed_indices, int):
            # Handle integer access index case (from load_into/store/update)
            i_access = distributed_indices
            y_lengths = self.ys_to_d_descriptor.get_lengths()
            
            # Calculate Y indices directly from access index
            remaining = i_access
            for y_idx in range(self.ndim_y - 1, -1, -1):
                y_indices[y_idx] = remaining % y_lengths[y_idx]
                remaining //= y_lengths[y_idx]
        else:
            # Handle distributed indices case (from direct calls/tests)
            # FIXED: Use span mappings from detail structure, not raw RH mappings
            for y_idx in range(self.ndim_y):
                span_major = self.encoding.detail.ys_to_span_major[y_idx]  # Use pre-computed span_major
                span_minor = self.encoding.detail.ys_to_span_minor[y_idx]  # Use pre-computed span_minor
                
                # Get the distributed index for this span
                if 0 <= span_major < len(distributed_indices):
                    dstr_index = distributed_indices[span_major]
                    # Get the specific component's index
                    if span_minor < len(dstr_index.partial_indices):
                        y_indices[y_idx] = dstr_index.partial_indices[span_minor]
        
        return y_indices
    
    def is_static(self) -> bool:
        """Check if distribution is static."""
        return (self.ps_ys_to_xs_adaptor.is_static() and 
                self.ys_to_d_descriptor.is_static())
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"TileDistribution(ndim_x={self.ndim_x}, ndim_y={self.ndim_y}, "
                f"ndim_p={self.ndim_p}, ndim_r={self.ndim_r})")

    def get_y_vector_lengths(self) -> List[int]:
        """
        Get vector lengths for each Y dimension.
        Matches C++ get_window_adaptor_ys_safe_vector_length_strides().
        """
        return [self.ys_to_d_descriptor.get_lengths()[i] for i in range(self.ndim_y)]

    def get_y_vector_strides(self) -> List[int]:
        """
        Get vector strides for each Y dimension.
        Matches C++ get_window_adaptor_ys_safe_vector_length_strides().
        """
        # Extract the actual strides from the tensor descriptor's EmbedTransform
        # instead of calculating our own (which was wrong)
        transforms = self.ys_to_d_descriptor.get_transforms()
        
        # For naive tensor descriptors, the first transform is always an EmbedTransform
        if len(transforms) > 0 and hasattr(transforms[0], 'strides'):
            return transforms[0].strides
        
        # Fallback to calculated strides if no EmbedTransform found
        strides = [1]  # First dimension has stride 1
        lengths = self.get_y_vector_lengths()
        for i in range(1, self.ndim_y):
            stride = strides[-1] * lengths[i-1]
            strides.append(stride)
        return strides


def make_tile_distributed_span(partial_lengths: List[int]) -> TileDistributedSpan:
    """
    Create a tile distributed span.
    
    Args:
        partial_lengths: Partial lengths for each dimension
        
    Returns:
        TileDistributedSpan instance
    """
    return TileDistributedSpan(partial_lengths)


def make_tile_distributed_index(partial_indices: List[int]) -> TileDistributedIndex:
    """
    Create a tile distributed index.
    
    Args:
        partial_indices: Partial indices for each dimension
        
    Returns:
        TileDistributedIndex instance
    """
    return TileDistributedIndex(partial_indices)


def make_tile_distribution_encoding(rs_lengths: List[int],
                                  hs_lengthss: List[List[int]],
                                  ps_to_rhss_major: List[List[int]],
                                  ps_to_rhss_minor: List[List[int]],
                                  ys_to_rhs_major: List[int],
                                  ys_to_rhs_minor: List[int]) -> TileDistributionEncoding:
    """
    Create a tile distribution encoding.
    
    Args:
        rs_lengths: Replication dimension lengths
        hs_lengthss: Hierarchical dimension lengths for each X
        ps_to_rhss_major: P to RH major dimension mapping
        ps_to_rhss_minor: P to RH minor dimension mapping
        ys_to_rhs_major: Y to RH major dimension mapping
        ys_to_rhs_minor: Y to RH minor dimension mapping
        
    Returns:
        TileDistributionEncoding instance
    """
    return TileDistributionEncoding(
        rs_lengths=rs_lengths,
        hs_lengthss=hs_lengthss,
        ps_to_rhss_major=ps_to_rhss_major,
        ps_to_rhss_minor=ps_to_rhss_minor,
        ys_to_rhs_major=ys_to_rhs_major,
        ys_to_rhs_minor=ys_to_rhs_minor
    )


def make_tile_distribution(ps_ys_to_xs_adaptor: TensorAdaptor,
                         ys_to_d_descriptor: TensorDescriptor,
                         encoding: TileDistributionEncoding) -> TileDistribution:
    """
    Create a tile distribution.
    
    Args:
        ps_ys_to_xs_adaptor: Adaptor mapping from (P,Y) to X dimensions
        ys_to_d_descriptor: Descriptor mapping from Y to linearized D dimension
        encoding: Static tile distribution encoding
        
    Returns:
        TileDistribution instance
    """
    return TileDistribution(
        ps_ys_to_xs_adaptor=ps_ys_to_xs_adaptor,
        ys_to_d_descriptor=ys_to_d_descriptor,
        encoding=encoding
    )


def make_static_tile_distribution(encoding: TileDistributionEncoding) -> TileDistribution:
    """
    Create a static tile distribution from encoding.
    
    Args:
        encoding: Static tile distribution encoding
        
    Returns:
        TileDistribution instance with static properties
    """
    # Create adaptor encoding for tile distribution
    adaptor_impl = _make_adaptor_encoding_for_tile_distribution(encoding)
    
    ps_ys_to_xs_adaptor_impl = adaptor_impl[0]
    ys_to_d_adaptor_impl = adaptor_impl[1]
    d_length = adaptor_impl[2]
    rh_major_minor_to_hidden_ids_impl = adaptor_impl[3]
    
    # Construct static tensor adaptors
    ps_ys_to_xs_adaptor = _construct_static_tensor_adaptor_from_encoding(ps_ys_to_xs_adaptor_impl)
    ys_to_d_adaptor = _construct_static_tensor_adaptor_from_encoding(ys_to_d_adaptor_impl)
    
    # Create descriptor from adaptor
    ys_to_d_descriptor = make_tensor_descriptor_from_adaptor(ys_to_d_adaptor, d_length)
    
    # Get dimensions
    ndim_rh_major = len(encoding.hs_lengthss) + 1  # +1 for R
    ndims_rhs_minor = [len(encoding.rs_lengths)] + [len(hs) for hs in encoding.hs_lengthss]
    
    # Convert hidden IDs to tuple of sequences
    rh_major_minor_to_hidden_ids = _to_tuple_of_sequence(
        rh_major_minor_to_hidden_ids_impl, 
        ndim_rh_major, 
        ndims_rhs_minor
    )
    
    # Create distribution detail
    distribution_detail = TileDistributionDetail(rh_major_minor_to_hidden_ids)
    
    return TileDistribution(
        ps_ys_to_xs_adaptor=ps_ys_to_xs_adaptor,
        ys_to_d_descriptor=ys_to_d_descriptor,
        encoding=encoding
    )

def _make_adaptor_encoding_for_tile_distribution(encoding: TileDistributionEncoding) -> Tuple:
    """Create adaptor encoding for tile distribution."""
    # Get encoding components
    rs_lengths = encoding.rs_lengths
    hs_lengthss = encoding.hs_lengthss
    ps_to_rhss_major = encoding.ps_to_rhss_major
    ps_to_rhss_minor = encoding.ps_to_rhss_minor
    ys_to_rhs_major = encoding.ys_to_rhs_major
    ys_to_rhs_minor = encoding.ys_to_rhs_minor
    
    # Constants
    MAX_NUM_TRANSFORMS = 20
    MAX_META_DATA_SIZE = 128
    MAX_NUM_DIM = 10
    
    # Get dimensions
    ndim_x = len(hs_lengthss)
    
    # Initialize arrays for hidden dimensions
    rh_major_minor_to_hidden_ids = [[0] * MAX_NUM_DIM for _ in range(ndim_x + 1)]
    rh_major_minor_to_hidden_lengths = [[0] * MAX_NUM_DIM for _ in range(ndim_x + 1)]
    
    # Initialize transforms array
    transforms = []
    num_trans = 0
    hidden_dim_cnt = ndim_x
    
    # Add replicate transform
    ndim_r_minor = len(rs_lengths)
    transforms.append({
        'name': 'replicate',
        'meta_data': rs_lengths,
        'num_dim': 0,
        'dims': [],
        'num_dim_out': ndim_r_minor,
        'dims_out': list(range(hidden_dim_cnt, hidden_dim_cnt + ndim_r_minor))
    })
    
    # Update hidden dimension mappings
    for i in range(ndim_r_minor):
        rh_major_minor_to_hidden_ids[0][i] = hidden_dim_cnt
        rh_major_minor_to_hidden_lengths[0][i] = rs_lengths[i]
        hidden_dim_cnt += 1
    
    # Add unmerge transforms for X dimensions
    for idim_x in range(ndim_x):
        h_minor_lengths = hs_lengthss[idim_x]
        ndim_h_minor = len(h_minor_lengths)
        
        transforms.append({
            'name': 'unmerge',
            'meta_data': h_minor_lengths,
            'num_dim': 1,
            'dims': [idim_x],
            'num_dim_out': ndim_h_minor,
            'dims_out': list(range(hidden_dim_cnt, hidden_dim_cnt + ndim_h_minor))
        })
        
        # Update hidden dimension mappings
        for i in range(ndim_h_minor):
            rh_major_minor_to_hidden_ids[idim_x + 1][i] = hidden_dim_cnt
            rh_major_minor_to_hidden_lengths[idim_x + 1][i] = h_minor_lengths[i]
            hidden_dim_cnt += 1
    
    # Add P dimension transforms
    ndim_p = len(ps_to_rhss_major)
    hidden_dim_id_ps = [0] * ndim_p
    
    for i_dim_p in range(ndim_p):
        hidden_dim_id_p = hidden_dim_cnt
        hidden_dim_cnt += 1
        hidden_dim_id_ps[i_dim_p] = hidden_dim_id_p
        
        p2RHsMajor = ps_to_rhss_major[i_dim_p]
        p2RHsMinor = ps_to_rhss_minor[i_dim_p]
        
        assert len(p2RHsMajor) == len(p2RHsMinor), "wrong!"
        
        ndim_low = len(p2RHsMajor)
        low_dims = [0] * ndim_low
        low_lengths = [0] * ndim_low
        
        for i in range(ndim_low):
            rh_major = p2RHsMajor[i]
            rh_minor = p2RHsMinor[i]
            low_dims[i] = rh_major_minor_to_hidden_ids[rh_major][rh_minor]
            low_lengths[i] = rh_major_minor_to_hidden_lengths[rh_major][rh_minor]
        
        transforms.append({
            'name': 'merge',
            'meta_data': low_lengths,
            'num_dim': ndim_low,
            'dims': low_dims,
            'num_dim_out': 1,
            'dims_out': [hidden_dim_id_p]
        })
    
    # Create bottom and top dimension IDs
    bottom_dim_ids = list(range(ndim_x))
    top_dim_ids = hidden_dim_id_ps.copy()
    
    # Add Y dimensions to top_dim_ids
    for i in range(len(ys_to_rhs_major)):
        rh_major = ys_to_rhs_major[i]
        rh_minor = ys_to_rhs_minor[i]
        top_dim_ids.append(rh_major_minor_to_hidden_ids[rh_major][rh_minor])
    
    # Create adaptor encodings
    ps_ys_to_xs_adaptor_encoding = (
        transforms,
        len(transforms),
        bottom_dim_ids,
        len(bottom_dim_ids),
        top_dim_ids,
        len(top_dim_ids)
    )
    
    # Create Y to D descriptor encoding
    y_lengths = [0] * len(ys_to_rhs_major)
    d_length = 1
    
    for i in range(len(ys_to_rhs_major)):
        rh_major = ys_to_rhs_major[i]
        rh_minor = ys_to_rhs_minor[i]
        y_length = rh_major_minor_to_hidden_lengths[rh_major][rh_minor]
        y_lengths[i] = y_length
        d_length *= y_length
    
    ys_to_d_adaptor_encoding = (
        [{
            'name': 'unmerge',
            'meta_data': y_lengths,
            'num_dim': 1,
            'dims': [0],
            'num_dim_out': len(y_lengths),
            'dims_out': list(range(1, len(y_lengths) + 1))
        }],
        1,
        [0],
        1,
        list(range(1, len(y_lengths) + 1)),
        len(y_lengths)
    )
    
    return (
        ps_ys_to_xs_adaptor_encoding,
        ys_to_d_adaptor_encoding,
        d_length,
        rh_major_minor_to_hidden_ids
    )

def _construct_static_tensor_adaptor_from_encoding(encoding: Tuple) -> TensorAdaptor:
    """Construct static tensor adaptor from encoding."""
    transforms, num_trans, bottom_dim_ids, ndim_bottom, top_dim_ids, ndim_top = encoding
    
    # Create lower and upper dimension hidden IDs for each transform
    lower_dimension_hidden_idss = []
    upper_dimension_hidden_idss = []
    
    for transform in transforms:
        # Get dimensions from transform
        dims = transform['dims']
        dims_out = transform['dims_out']
        
        # For lower dimensions, use the input dimensions directly
        lower_ids = dims
        # For upper dimensions, use the output dimensions directly
        upper_ids = dims_out
        lower_dimension_hidden_idss.append(lower_ids)
        upper_dimension_hidden_idss.append(upper_ids)
    
    # Validation: number of transforms must match number of lower/upper dimension ID lists
    if not (len(transforms) == len(lower_dimension_hidden_idss) == len(upper_dimension_hidden_idss)):
        raise ValueError("Number of transforms must match lower/upper dimension IDs")
    
    # Create adaptor with static properties
    adaptor = TensorAdaptor(
        transforms=transforms,
        lower_dimension_hidden_idss=lower_dimension_hidden_idss,
        upper_dimension_hidden_idss=upper_dimension_hidden_idss,
        bottom_dimension_hidden_ids=bottom_dim_ids,
        top_dimension_hidden_ids=top_dim_ids
    )
    
    # Mark as static
    adaptor._is_static = True
    
    return adaptor

def _to_tuple_of_sequence(arr: List[List[int]], ndim_rh_major: int, ndims_rhs_minor: List[int]) -> Tuple:
    """Convert array to tuple of sequences."""
    result = []
    for i in range(ndim_rh_major):
        result.append(tuple(arr[i][:ndims_rhs_minor[i]]))
    return tuple(result)

@dataclass
class TileDistributionDetail:
    """Detail information for tile distribution."""
    rh_major_minor_to_adaptor_hidden_idss: Tuple[Tuple[int, ...], ...]

def make_tensor_descriptor_from_adaptor(adaptor, element_space_size):
    """Create a TensorDescriptor from a TensorAdaptor and element space size."""
    from .tensor_descriptor import TensorDescriptor
    desc = TensorDescriptor(
        transforms=adaptor.transforms,
        lower_dimension_hidden_idss=adaptor.lower_dimension_hidden_idss,
        upper_dimension_hidden_idss=adaptor.upper_dimension_hidden_idss,
        top_dimension_hidden_ids=adaptor.top_dimension_hidden_ids,
        element_space_size=element_space_size
    )
    desc._is_static = True
    return desc



def slice_distribution_from_x(distribution: TileDistribution, x_slice_begins: List[int], x_slice_ends: List[int]) -> Tuple[TileDistribution, List[int], List[int]]:
    """
    Slice a tile distribution along X dimensions.
    
    Args:
        distribution: The tile distribution to slice.
        x_slice_begins: List of begin indices for each X dimension.
        x_slice_ends: List of end indices for each X dimension.
        
    Returns:
        A tuple containing:
        - A new TileDistribution representing the sliced distribution.
        - A list of Y slice origins.
        - A list of Y slice lengths.
    """
    # Validate slice parameters
    if len(x_slice_begins) != len(x_slice_ends):
        raise ValueError("x_slice_begins and x_slice_ends must have the same length")
    if len(x_slice_begins) != distribution.ndim_x:
        raise ValueError("Slice parameters must match the number of X dimensions")

    # Calculate slice lengths
    x_slice_lengths = [end - begin for begin, end in zip(x_slice_begins, x_slice_ends)]

    # Get the encoding from the distribution
    encoding = distribution.encoding

    # Create a new encoding for the sliced distribution
    # For simplicity, we assume slicing only affects H dimensions
    # In practice, you would need to adjust R, P, and Y mappings as well
    new_hs_lengthss = []
    for x_idx, h_lengths in enumerate(encoding.hs_lengthss):
        new_h_lengths = h_lengths.copy()
        # Adjust H lengths based on slice
        new_h_lengths[0] = x_slice_lengths[x_idx]  # Simplified: assume first H dimension is sliced
        new_hs_lengthss.append(new_h_lengths)

    new_encoding = make_tile_distribution_encoding(
        rs_lengths=encoding.rs_lengths,
        hs_lengthss=new_hs_lengthss,
        ps_to_rhss_major=encoding.ps_to_rhss_major,
        ps_to_rhss_minor=encoding.ps_to_rhss_minor,
        ys_to_rhs_major=encoding.ys_to_rhs_major,
        ys_to_rhs_minor=encoding.ys_to_rhs_minor
    )

    # Create a new distribution from the sliced encoding
    new_distribution = make_static_tile_distribution(new_encoding)

    # Calculate Y slice origins and lengths
    # For simplicity, we assume Y origins are derived from X slice begins
    y_slice_origins = [begin for begin in x_slice_begins]
    y_slice_lengths = [length for length in x_slice_lengths]

    return new_distribution, y_slice_origins, y_slice_lengths 

def make_embedding_tile_distribution(
    base_distribution: TileDistribution,
    embedding_dims: List[int],
    embedding_lengths: List[int]
) -> TileDistribution:
    """
    Create a tile distribution by embedding additional dimensions into an existing distribution.
    
    Args:
        base_distribution: The base tile distribution to embed into.
        embedding_dims: List of X dimension indices where to embed new dimensions.
        embedding_lengths: List of lengths for each embedded dimension.
        
    Returns:
        A new TileDistribution with embedded dimensions.
    """
    # Validate embedding parameters
    if len(embedding_dims) != len(embedding_lengths):
        raise ValueError("Number of embedding dimensions must match number of embedding lengths")
    if not all(0 <= d <= base_distribution.ndim_x for d in embedding_dims):
        raise ValueError("Invalid embedding dimension index")
    
    # Get base encoding
    base_encoding = base_distribution.encoding
    
    # Create new encoding with embedded dimensions
    new_hs_lengthss = []
    # Build mapping from old to new X dimension indices
    old_to_new_x_idx = {}
    new_x_idx = 0
    
    for x_idx in range(base_distribution.ndim_x):
        if x_idx in embedding_dims:
            # Add embedded dimension first
            new_hs_lengthss.append([embedding_lengths[embedding_dims.index(x_idx)]])
            new_x_idx += 1
        
        # Map old index to new index
        old_to_new_x_idx[x_idx] = new_x_idx
        # Add original dimension
        new_hs_lengthss.append(base_encoding.hs_lengthss[x_idx])
        new_x_idx += 1
    
    # Adjust Y dimension mappings for new X indices
    new_ys_to_rhs_major = []
    new_ys_to_rhs_minor = []
    
    for old_major, old_minor in zip(base_encoding.ys_to_rhs_major, base_encoding.ys_to_rhs_minor):
        if old_major == 0:  # R dimension - no change
            new_ys_to_rhs_major.append(old_major)
            new_ys_to_rhs_minor.append(old_minor)
        else:  # X dimension - adjust for embedded dimensions
            old_x_idx = old_major - 1  # Convert from RH space to X space
            new_x_idx = old_to_new_x_idx[old_x_idx]
            new_ys_to_rhs_major.append(new_x_idx + 1)  # Convert back to RH space
            new_ys_to_rhs_minor.append(old_minor)
    
    # Adjust P dimension mappings for new X indices
    new_ps_to_rhss_major = []
    new_ps_to_rhss_minor = []
    
    for p_major, p_minor in zip(base_encoding.ps_to_rhss_major, base_encoding.ps_to_rhss_minor):
        new_major = []
        new_minor = []
        for old_major, old_minor in zip(p_major, p_minor):
            if old_major == 0:  # R dimension - no change
                new_major.append(old_major)
                new_minor.append(old_minor)
            else:  # X dimension - adjust for embedded dimensions
                old_x_idx = old_major - 1  # Convert from RH space to X space
                new_x_idx = old_to_new_x_idx[old_x_idx]
                new_major.append(new_x_idx + 1)  # Convert back to RH space
                new_minor.append(old_minor)
        new_ps_to_rhss_major.append(new_major)
        new_ps_to_rhss_minor.append(new_minor)
    
    # Create new encoding
    new_encoding = make_tile_distribution_encoding(
        rs_lengths=base_encoding.rs_lengths,
        hs_lengthss=new_hs_lengthss,
        ps_to_rhss_major=new_ps_to_rhss_major,
        ps_to_rhss_minor=new_ps_to_rhss_minor,
        ys_to_rhs_major=new_ys_to_rhs_major,
        ys_to_rhs_minor=new_ys_to_rhs_minor
    )
    
    return make_static_tile_distribution(new_encoding)

def make_reduction_tile_distribution(
    base_distribution: TileDistribution,
    reduction_dims: List[int]
) -> TileDistribution:
    """
    Create a tile distribution by reducing dimensions from an existing distribution.
    
    Args:
        base_distribution: The base tile distribution to reduce.
        reduction_dims: List of X dimension indices to reduce.
        
    Returns:
        A new TileDistribution with reduced dimensions.
    """
    # Validate reduction dimensions
    if not all(0 <= d < base_distribution.ndim_x for d in reduction_dims):
        raise ValueError("Invalid reduction dimension index")
    
    # Get base encoding
    base_encoding = base_distribution.encoding
    
    # Create new encoding without reduced dimensions
    new_hs_lengthss = [
        lengths for i, lengths in enumerate(base_encoding.hs_lengthss)
        if i not in reduction_dims
    ]
    
    # Adjust P dimension mappings
    new_ps_to_rhss_major = []
    new_ps_to_rhss_minor = []
    for p_major, p_minor in zip(base_encoding.ps_to_rhss_major, base_encoding.ps_to_rhss_minor):
        new_major = []
        new_minor = []
        for m, n in zip(p_major, p_minor):
            if m == 0:  # R dimension
                new_major.append(m)
                new_minor.append(n)
            elif m not in reduction_dims:
                # Adjust major index for removed dimensions
                new_major.append(m - sum(1 for d in reduction_dims if d < m))
                new_minor.append(n)
        if new_major:  # Only add if there are remaining dimensions
            new_ps_to_rhss_major.append(new_major)
            new_ps_to_rhss_minor.append(new_minor)
    
    # Adjust Y dimension mappings
    new_ys_to_rhs_major = []
    new_ys_to_rhs_minor = []
    for m, n in zip(base_encoding.ys_to_rhs_major, base_encoding.ys_to_rhs_minor):
        if m not in reduction_dims:
            # Adjust major index for removed dimensions
            new_ys_to_rhs_major.append(m - sum(1 for d in reduction_dims if d < m))
            new_ys_to_rhs_minor.append(n)
    
    # Create new encoding
    new_encoding = make_tile_distribution_encoding(
        rs_lengths=base_encoding.rs_lengths,
        hs_lengthss=new_hs_lengthss,
        ps_to_rhss_major=new_ps_to_rhss_major,
        ps_to_rhss_minor=new_ps_to_rhss_minor,
        ys_to_rhs_major=new_ys_to_rhs_major,
        ys_to_rhs_minor=new_ys_to_rhs_minor
    )
    
    return make_static_tile_distribution(new_encoding)

def compose_tile_distributions(
    outer_distribution: TileDistribution,
    inner_distribution: TileDistribution,
    composition_dims: List[int]
) -> TileDistribution:
    """
    Create a tile distribution by composing two distributions.
    
    Args:
        outer_distribution: The outer tile distribution.
        inner_distribution: The inner tile distribution to compose with.
        composition_dims: List of X dimension indices in outer distribution to compose with inner dimensions.
        
    Returns:
        A new TileDistribution representing the composition.
    """
    # Validate composition dimensions
    if not all(0 <= d < outer_distribution.ndim_x for d in composition_dims):
        raise ValueError("Invalid composition dimension index")
    
    # Get encodings
    outer_encoding = outer_distribution.encoding
    inner_encoding = inner_distribution.encoding
    
    # Create new encoding by composing dimensions
    new_hs_lengthss = []
    for x_idx in range(outer_distribution.ndim_x):
        if x_idx in composition_dims:
            # Add inner dimensions
            new_hs_lengthss.extend(inner_encoding.hs_lengthss)
        else:
            new_hs_lengthss.append(outer_encoding.hs_lengthss[x_idx])
    
    # Calculate mapping from old to new dimension indices
    new_dim_idx = 0
    old_to_new_major = {}  # Maps old major IDs to new major IDs
    
    for x_idx in range(outer_distribution.ndim_x):
        if x_idx in composition_dims:
            # This X dimension is replaced by inner dimensions
            for inner_x_idx in range(inner_distribution.ndim_x):
                old_to_new_major[x_idx + 1] = new_dim_idx + 1  # +1 for R dimension
                new_dim_idx += 1
        else:
            # This X dimension is preserved
            old_to_new_major[x_idx + 1] = new_dim_idx + 1  # +1 for R dimension
            new_dim_idx += 1
    
    # Adjust P dimension mappings
    new_ps_to_rhss_major = []
    new_ps_to_rhss_minor = []
    for p_major, p_minor in zip(outer_encoding.ps_to_rhss_major, outer_encoding.ps_to_rhss_minor):
        new_major = []
        new_minor = []
        for m, n in zip(p_major, p_minor):
            if m == 0:  # R dimension
                new_major.append(m)
                new_minor.append(n)
            else:
                # Use the mapping to get the new major dimension ID
                if m in old_to_new_major:
                    new_major.append(old_to_new_major[m])
                    new_minor.append(n)
        if new_major:  # Only add if there are remaining dimensions
            new_ps_to_rhss_major.append(new_major)
            new_ps_to_rhss_minor.append(new_minor)
    
    # Adjust Y dimension mappings
    new_ys_to_rhs_major = []
    new_ys_to_rhs_minor = []
    for m, n in zip(outer_encoding.ys_to_rhs_major, outer_encoding.ys_to_rhs_minor):
        if m in old_to_new_major:
            new_ys_to_rhs_major.append(old_to_new_major[m])
            new_ys_to_rhs_minor.append(n)
    
    # Create new encoding
    new_encoding = make_tile_distribution_encoding(
        rs_lengths=outer_encoding.rs_lengths,
        hs_lengthss=new_hs_lengthss,
        ps_to_rhss_major=new_ps_to_rhss_major,
        ps_to_rhss_minor=new_ps_to_rhss_minor,
        ys_to_rhs_major=new_ys_to_rhs_major,
        ys_to_rhs_minor=new_ys_to_rhs_minor
    )
    
    return make_static_tile_distribution(new_encoding) 