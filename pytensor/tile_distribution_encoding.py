"""
Python implementation of tile_distribution_encoding.hpp from Composable Kernels.

This module provides tile distribution encoding functionality that describes
how tensor data is distributed across processing elements with detailed
mapping information.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass, field
from functools import reduce
import operator


@dataclass
class TileDistributionEncodingDetail:
    """
    Detailed information computed from tile distribution encoding.
    
    This corresponds to the detail struct in the C++ implementation
    and contains derived information for efficient access.
    """
    
    # Basic dimensions
    ndim_rh_major: int
    ndim_span_major: int
    ndims_rhs_minor: List[int]
    max_ndim_rh_minor: int
    
    # RHS (R and H) lengths
    rhs_lengthss: List[List[int]]
    
    # Y dimension info
    ys_lengths: List[int]
    
    # Mapping tables
    rhs_major_minor_to_ys: List[List[int]]
    ndims_span_minor: List[int]
    max_ndim_span_minor: int
    rhs_major_minor_to_span_minor: List[List[int]]
    
    # Y to span mappings
    ys_to_span_major: List[int]
    ys_to_span_minor: List[int]
    
    # Distributed span info
    distributed_spans_lengthss: List[List[int]]
    ndims_distributed_spans_minor: List[int]
    
    # P ownership info
    does_p_own_r: List[List[bool]]
    ps_over_rs_derivative: List[List[int]]
    
    @classmethod
    def from_encoding(cls, 
                     rs_lengths: List[int],
                     hs_lengthss: List[List[int]], 
                     ps_to_rhss_major: List[List[int]],
                     ps_to_rhss_minor: List[List[int]],
                     ys_to_rhs_major: List[int],
                     ys_to_rhs_minor: List[int]) -> 'TileDistributionEncodingDetail':
        """
        Create detail information from encoding parameters.
        
        This implements the logic from the C++ detail struct constructor.
        """
        ndim_x = len(hs_lengthss)
        ndim_p = len(ps_to_rhss_major)
        ndim_y = len(ys_to_rhs_major)
        ndim_r = len(rs_lengths)
        
        # Basic dimensions
        ndim_rh_major = ndim_x + 1
        ndim_span_major = ndim_x
        
        # ndims_rhs_minor
        ndims_rhs_minor = [ndim_r] + [len(hs) for hs in hs_lengthss]
        max_ndim_rh_minor = max(ndims_rhs_minor) if ndims_rhs_minor else 0
        
        # rhs_lengthss - combine R and H lengths
        rhs_lengthss = [[0] * max_ndim_rh_minor for _ in range(ndim_rh_major)]
        # R lengths
        for i, r_len in enumerate(rs_lengths):
            rhs_lengthss[0][i] = r_len
        # H lengths
        for i, h_lengths in enumerate(hs_lengthss):
            for j, h_len in enumerate(h_lengths):
                rhs_lengthss[i + 1][j] = h_len
        
        # ys_lengths
        ys_lengths = []
        for i in range(ndim_y):
            rh_major = ys_to_rhs_major[i]
            rh_minor = ys_to_rhs_minor[i]
            # Bounds check before accessing
            if 0 <= rh_major < len(rhs_lengthss) and 0 <= rh_minor < len(rhs_lengthss[rh_major]):
                ys_lengths.append(rhs_lengthss[rh_major][rh_minor])
            else:
                ys_lengths.append(0)  # Default value for invalid indices
        
        # rhs_major_minor_to_ys
        rhs_major_minor_to_ys = [[-1] * max_ndim_rh_minor for _ in range(ndim_rh_major)]
        for i in range(ndim_y):
            rh_major = ys_to_rhs_major[i]
            rh_minor = ys_to_rhs_minor[i]
            # Bounds check before assignment
            if (0 <= rh_major < len(rhs_major_minor_to_ys) and 
                0 <= rh_minor < len(rhs_major_minor_to_ys[rh_major])):
                rhs_major_minor_to_ys[rh_major][rh_minor] = i
        
        # ndims_span_minor
        ndims_span_minor = [0] * ndim_x
        for i in range(ndim_y):
            span_major = ys_to_rhs_major[i] - 1
            if 0 <= span_major < ndim_x:
                ndims_span_minor[span_major] += 1
        
        max_ndim_span_minor = max(ndims_span_minor) if ndims_span_minor else 0
        
        # rhs_major_minor_to_span_minor
        rhs_major_minor_to_span_minor = [[-1] * max_ndim_rh_minor for _ in range(ndim_rh_major)]
        span_minor_counts = [0] * ndim_rh_major
        
        for rh_major in range(ndim_rh_major):
            ndim_rh_minor = ndims_rhs_minor[rh_major]
            for rh_minor in range(ndim_rh_minor):
                idim_y = rhs_major_minor_to_ys[rh_major][rh_minor]
                if idim_y >= 0:
                    rhs_major_minor_to_span_minor[rh_major][rh_minor] = span_minor_counts[rh_major]
                    span_minor_counts[rh_major] += 1
        
        # ys_to_span_major and ys_to_span_minor
        ys_to_span_major = []
        ys_to_span_minor = []
        for i in range(ndim_y):
            span_major = ys_to_rhs_major[i] - 1
            ys_to_span_major.append(span_major)
            
            rh_major = ys_to_rhs_major[i]
            rh_minor = ys_to_rhs_minor[i]
            # Bounds check before accessing span_minor
            if (0 <= rh_major < len(rhs_major_minor_to_span_minor) and 
                0 <= rh_minor < len(rhs_major_minor_to_span_minor[rh_major])):
                span_minor = rhs_major_minor_to_span_minor[rh_major][rh_minor]
            else:
                span_minor = -1  # Default for invalid indices
            ys_to_span_minor.append(span_minor)
        
        # distributed_spans_lengthss
        distributed_spans_lengthss = [[-1] * max_ndim_span_minor for _ in range(ndim_span_major)]
        for i in range(ndim_y):
            rh_major = ys_to_rhs_major[i]
            rh_minor = ys_to_rhs_minor[i]
            
            if rh_major > 0:  # Skip R dimension
                h_major_idx = rh_major - 1
                # Bounds check before accessing H lengths
                if (0 <= h_major_idx < len(hs_lengthss) and 
                    0 <= rh_minor < len(hs_lengthss[h_major_idx])):
                    h_length = hs_lengthss[h_major_idx][rh_minor]
                    span_major = rh_major - 1
                    span_minor = rhs_major_minor_to_span_minor[rh_major][rh_minor]
                    if (0 <= span_major < len(distributed_spans_lengthss) and 
                        0 <= span_minor < len(distributed_spans_lengthss[span_major])):
                        distributed_spans_lengthss[span_major][span_minor] = h_length
        
        # ndims_distributed_spans_minor
        ndims_distributed_spans_minor = [0] * ndim_span_major
        for i in range(ndim_y):
            span_major = ys_to_rhs_major[i] - 1
            if 0 <= span_major < ndim_span_major:
                ndims_distributed_spans_minor[span_major] += 1
        
        # does_p_own_r
        does_p_own_r = [[False] * ndim_r for _ in range(ndim_p)]
        for idim_p in range(ndim_p):
            for idim_low in range(len(ps_to_rhss_major[idim_p])):
                rh_major = ps_to_rhss_major[idim_p][idim_low]
                rh_minor = ps_to_rhss_minor[idim_p][idim_low]
                if rh_major == 0 and 0 <= rh_minor < ndim_r:  # R dimension with bounds check
                    does_p_own_r[idim_p][rh_minor] = True
        
        # ps_over_rs_derivative
        ps_over_rs_derivative = [[0] * ndim_r for _ in range(ndim_p)]
        for idim_p in range(ndim_p):
            p_over_rh_derivative = 1
            # Process in reverse order
            for idim_low in range(len(ps_to_rhss_major[idim_p]) - 1, -1, -1):
                rh_major = ps_to_rhss_major[idim_p][idim_low]
                rh_minor = ps_to_rhss_minor[idim_p][idim_low]
                
                # Skip invalid indices
                if rh_minor < 0:
                    continue
                
                # Bounds check before accessing
                if (0 <= rh_major < len(rhs_lengthss) and 
                    0 <= rh_minor < len(rhs_lengthss[rh_major])):
                    rh_length = rhs_lengthss[rh_major][rh_minor]
                    
                    if rh_major == 0 and 0 <= rh_minor < ndim_r:  # R dimension
                        ps_over_rs_derivative[idim_p][rh_minor] = p_over_rh_derivative
                    
                    p_over_rh_derivative *= rh_length
        
        return cls(
            ndim_rh_major=ndim_rh_major,
            ndim_span_major=ndim_span_major,
            ndims_rhs_minor=ndims_rhs_minor,
            max_ndim_rh_minor=max_ndim_rh_minor,
            rhs_lengthss=rhs_lengthss,
            ys_lengths=ys_lengths,
            rhs_major_minor_to_ys=rhs_major_minor_to_ys,
            ndims_span_minor=ndims_span_minor,
            max_ndim_span_minor=max_ndim_span_minor,
            rhs_major_minor_to_span_minor=rhs_major_minor_to_span_minor,
            ys_to_span_major=ys_to_span_major,
            ys_to_span_minor=ys_to_span_minor,
            distributed_spans_lengthss=distributed_spans_lengthss,
            ndims_distributed_spans_minor=ndims_distributed_spans_minor,
            does_p_own_r=does_p_own_r,
            ps_over_rs_derivative=ps_over_rs_derivative
        )


@dataclass
class TileDistributionEncoding:
    """
    Complete tile distribution encoding with all mapping information.
    
    This is an enhanced version that includes the detail computation
    from the C++ implementation.
    """
    
    rs_lengths: List[int]
    hs_lengthss: List[List[int]]
    ps_to_rhss_major: List[List[int]]
    ps_to_rhss_minor: List[List[int]]
    ys_to_rhs_major: List[int]
    ys_to_rhs_minor: List[int]
    detail: TileDistributionEncodingDetail = field(init=False)
    
    def __post_init__(self):
        """Compute detail information after initialization."""
        # Validate encoding before computing detail
        # Validate dimensions match
        if len(self.ps_to_rhss_major) != len(self.ps_to_rhss_minor):
            raise ValueError("P dimension mappings must have same length")
        if len(self.ys_to_rhs_major) != len(self.ys_to_rhs_minor):
            raise ValueError("Y dimension mappings must have same length")
            
        # Validate P dimension mappings
        for i, (major_ids, minor_ids) in enumerate(zip(self.ps_to_rhss_major, self.ps_to_rhss_minor)):
            if len(major_ids) != len(minor_ids):
                raise ValueError(f"P dimension {i} major and minor mappings must have same length")
            
            # Check that major IDs are valid (0 for R, 1+ for X dimensions)
            for major_id in major_ids:
                if major_id < 0 or major_id > self.ndim_x:
                    raise ValueError(f"Invalid major dimension ID {major_id} in P dimension {i}")
            
            # Check that minor IDs are valid for each major dimension
            # Note: -1 is allowed as a sentinel value
            for major_id, minor_id in zip(major_ids, minor_ids):
                if minor_id >= 0:  # Only validate non-sentinel values
                    if major_id == 0:  # R dimension
                        if minor_id >= len(self.rs_lengths):
                            raise ValueError(f"Invalid minor dimension ID {minor_id} for R dimension in P dimension {i}")
                    else:  # X dimension
                        x_idx = major_id - 1
                        if x_idx < len(self.hs_lengthss) and minor_id >= len(self.hs_lengthss[x_idx]):
                            raise ValueError(f"Invalid minor dimension ID {minor_id} for X dimension {x_idx} in P dimension {i}")

        self.detail = TileDistributionEncodingDetail.from_encoding(
            self.rs_lengths,
            self.hs_lengthss,
            self.ps_to_rhss_major,
            self.ps_to_rhss_minor,
            self.ys_to_rhs_major,
            self.ys_to_rhs_minor
        )
    
    @property
    def ndim_x(self) -> int:
        """Number of X dimensions."""
        return len(self.hs_lengthss)
    
    @property
    def ndim_p(self) -> int:
        """Number of P dimensions."""
        return len(self.ps_to_rhss_major)
    
    @property
    def ndim_y(self) -> int:
        """Number of Y dimensions."""
        return len(self.ys_to_rhs_major)
    
    @property
    def ndim_r(self) -> int:
        """Number of R dimensions."""
        return len(self.rs_lengths)
    
    def get_h_dim_lengths_prefix_sum(self) -> List[int]:
        """
        Get prefix sum of H dimension lengths.
        
        Returns [0, len_d0, len_d0+len_d1, ...]
        """
        h_dim_lengths = [len(hs) for hs in self.hs_lengthss]
        prefix_sum = [0]
        for length in h_dim_lengths:
            prefix_sum.append(prefix_sum[-1] + length)
        return prefix_sum
    
    def get_uniformed_idx_y_to_h(self) -> List[int]:
        """
        Get uniformed Y to H index mapping.
        
        Maps Y indices to a flattened H index space.
        """
        # Get prefix sum including R dimension at position 0
        h_dim_lengths = [len(self.rs_lengths)] + [len(hs) for hs in self.hs_lengthss]
        x_dim_prefix_sum = [0]
        for length in h_dim_lengths:
            x_dim_prefix_sum.append(x_dim_prefix_sum[-1] + length)
        
        uniformed_indices = []
        for i in range(self.ndim_y):
            major = self.ys_to_rhs_major[i]
            minor = self.ys_to_rhs_minor[i]
            uniformed_idx = x_dim_prefix_sum[major] + minor
            uniformed_indices.append(uniformed_idx)
        
        return uniformed_indices
    
    def get_distributed_spans(self) -> List['TileDistributedSpan']:
        """
        Get distributed spans for each X dimension.
        
        Returns:
            List of TileDistributedSpan objects, one for each X dimension
        """
        from .tile_distribution import TileDistributedSpan
        
        spans = []
        for x_idx in range(self.ndim_x):
            # Get the distributed span lengths for this X dimension
            span_lengths = []
            for span_minor in range(self.detail.ndims_distributed_spans_minor[x_idx]):
                length = self.detail.distributed_spans_lengthss[x_idx][span_minor]
                if length != -1:
                    span_lengths.append(length)
            
            spans.append(TileDistributedSpan(span_lengths))
        
        return spans
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"TileDistributionEncoding(ndim_x={self.ndim_x}, "
                f"ndim_p={self.ndim_p}, ndim_y={self.ndim_y}, "
                f"ndim_r={self.ndim_r})")


def make_embed_tile_distribution_encoding(outer_encoding: TileDistributionEncoding,
                                        inner_encoding: TileDistributionEncoding) -> TileDistributionEncoding:
    """
    Create an embedded tile distribution encoding by combining outer and inner encodings.
    
    This implements the make_embed_tile_distribution_encoding function from C++.
    
    Args:
        outer_encoding: Outer distribution encoding
        inner_encoding: Inner distribution encoding
        
    Returns:
        Combined TileDistributionEncoding
    """
    if outer_encoding.ndim_x != inner_encoding.ndim_x:
        raise ValueError("Outer and inner encodings must have same number of X dimensions")
    
    ndim_h_major = outer_encoding.ndim_x
    
    # Merge R lengths - only keep unique R dimensions
    rs_lengths = list(outer_encoding.rs_lengths)
    for r in inner_encoding.rs_lengths:
        if r not in rs_lengths:
            rs_lengths.append(r)
    
    # Merge H lengths for each X dimension
    hs_lengthss = []
    for i in range(ndim_h_major):
        merged_hs = outer_encoding.hs_lengthss[i] + inner_encoding.hs_lengthss[i]
        hs_lengthss.append(merged_hs)
    
    # Calculate number of outer RH minor dimensions for each major
    rhs_major_to_ndim_outer_rhs_minor = [len(outer_encoding.rs_lengths)]
    for i in range(ndim_h_major):
        rhs_major_to_ndim_outer_rhs_minor.append(len(outer_encoding.hs_lengthss[i]))
    
    # Update inner P mappings
    updated_inner_ps_to_rhss_minor = []
    for p in range(inner_encoding.ndim_p):
        inner_p_to_rhss_major = inner_encoding.ps_to_rhss_major[p]
        inner_p_to_rhss_minor = inner_encoding.ps_to_rhss_minor[p]
        
        updated_minor = []
        for i in range(len(inner_p_to_rhss_minor)):
            rh_major = inner_p_to_rhss_major[i]
            ndim_outer_h_minor = rhs_major_to_ndim_outer_rhs_minor[rh_major]
            updated_minor.append(inner_p_to_rhss_minor[i] + ndim_outer_h_minor)
        
        updated_inner_ps_to_rhss_minor.append(updated_minor)
    
    # Update inner Y mappings
    updated_inner_ys_to_rhs_minor = []
    for i in range(inner_encoding.ndim_y):
        rh_major = inner_encoding.ys_to_rhs_major[i]
        ndim_outer_h_minor = rhs_major_to_ndim_outer_rhs_minor[rh_major]
        updated_inner_ys_to_rhs_minor.append(
            inner_encoding.ys_to_rhs_minor[i] + ndim_outer_h_minor
        )
    
    # Combine P mappings
    ps_to_rhss_major = outer_encoding.ps_to_rhss_major + inner_encoding.ps_to_rhss_major
    ps_to_rhss_minor = outer_encoding.ps_to_rhss_minor + updated_inner_ps_to_rhss_minor
    
    # Combine Y mappings
    ys_to_rhs_major = outer_encoding.ys_to_rhs_major + inner_encoding.ys_to_rhs_major
    ys_to_rhs_minor = outer_encoding.ys_to_rhs_minor + updated_inner_ys_to_rhs_minor
    
    return TileDistributionEncoding(
        rs_lengths=rs_lengths,
        hs_lengthss=hs_lengthss,
        ps_to_rhss_major=ps_to_rhss_major,
        ps_to_rhss_minor=ps_to_rhss_minor,
        ys_to_rhs_major=ys_to_rhs_major,
        ys_to_rhs_minor=ys_to_rhs_minor
    )


def make_reduce_tile_distribution_encoding(in_encoding: TileDistributionEncoding,
                                         reduce_dim_xs: List[int]) -> TileDistributionEncoding:
    """
    Create a reduced tile distribution encoding by removing specified X dimensions.
    
    This implements the make_reduce_tile_distribution_encoding function from C++.
    
    Args:
        in_encoding: Input distribution encoding
        reduce_dim_xs: X dimension indices to reduce
        
    Returns:
        Reduced TileDistributionEncoding
    """
    ndim_p = in_encoding.ndim_p
    ndim_x_in = in_encoding.ndim_x
    ndim_y_in = in_encoding.ndim_y
    ndim_rh_major_in = in_encoding.ndim_x + 1
    ndim_x_out = ndim_x_in - len(reduce_dim_xs)
    
    # Determine which RH major dimensions are for reduction
    is_rh_major_in_for_reduce = [False] * ndim_rh_major_in
    for x_idx in reduce_dim_xs:
        rh_major = x_idx + 1  # +1 because R is at index 0
        is_rh_major_in_for_reduce[rh_major] = True
    
    # Determine which Y dimensions are for reduction
    is_y_in_for_reduce = [False] * ndim_y_in
    for i in range(ndim_y_in):
        rh_major = in_encoding.ys_to_rhs_major[i]
        if is_rh_major_in_for_reduce[rh_major]:
            is_y_in_for_reduce[i] = True
    
    # Determine which RH minor dimensions are for Y reduction
    max_ndim_rh_minor_in = in_encoding.detail.max_ndim_rh_minor
    is_rh_minor_in_for_y_reduce = [[False] * max_ndim_rh_minor_in for _ in range(ndim_rh_major_in)]
    
    for i in range(ndim_y_in):
        if is_y_in_for_reduce[i]:
            rh_major = in_encoding.ys_to_rhs_major[i]
            rh_minor = in_encoding.ys_to_rhs_minor[i]
            is_rh_minor_in_for_y_reduce[rh_major][rh_minor] = True
    
    # Create input to output RH major mapping
    in2out_rh_major = [-1] * ndim_rh_major_in
    cnt_ndim_rh_major_out = 0
    
    for i in range(ndim_rh_major_in):
        if is_rh_major_in_for_reduce[i]:
            in2out_rh_major[i] = 0  # Maps to R dimension
        else:
            in2out_rh_major[i] = cnt_ndim_rh_major_out
            cnt_ndim_rh_major_out += 1
    
    # Build output R lengths and RH minor mappings
    rs_lengths_out = list(in_encoding.rs_lengths)  # Start with input R lengths
    in2out_rh_minor = [[-1] * max_ndim_rh_minor_in for _ in range(ndim_rh_major_in)]
    
    # Map input R dimensions
    for i in range(len(in_encoding.rs_lengths)):
        in2out_rh_minor[0][i] = i
    
    # Process H dimensions
    cnt_ndim_r_out = len(in_encoding.rs_lengths)
    
    for rh_major_in in range(1, ndim_rh_major_in):
        h_major_in = rh_major_in - 1
        
        if h_major_in < len(in_encoding.hs_lengthss):
            ndim_rh_minor_in = len(in_encoding.hs_lengthss[h_major_in])
            
            if is_rh_major_in_for_reduce[rh_major_in]:
                # Move non-Y-reduced H dimensions to R
                for rh_minor_in in range(ndim_rh_minor_in):
                    if not is_rh_minor_in_for_y_reduce[rh_major_in][rh_minor_in]:
                        rs_lengths_out.append(in_encoding.hs_lengthss[h_major_in][rh_minor_in])
                        in2out_rh_minor[rh_major_in][rh_minor_in] = cnt_ndim_r_out
                        cnt_ndim_r_out += 1
            else:
                # Keep as H dimension
                for rh_minor_in in range(ndim_rh_minor_in):
                    in2out_rh_minor[rh_major_in][rh_minor_in] = rh_minor_in
    
    # Build output H lengths
    hs_lengthss_out = []
    for i in range(ndim_x_in):
        if i + 1 not in [x + 1 for x in reduce_dim_xs]:  # Not reduced
            hs_lengthss_out.append(in_encoding.hs_lengthss[i])
    
    # Update P mappings
    ps_to_rhss_major_out = []
    ps_to_rhss_minor_out = []
    
    for idim_p in range(ndim_p):
        major_out = []
        minor_out = []
        
        for idim_low in range(len(in_encoding.ps_to_rhss_major[idim_p])):
            rh_major_in = in_encoding.ps_to_rhss_major[idim_p][idim_low]
            rh_minor_in = in_encoding.ps_to_rhss_minor[idim_p][idim_low]
            
            major_out.append(in2out_rh_major[rh_major_in])
            minor_out.append(in2out_rh_minor[rh_major_in][rh_minor_in])
        
        ps_to_rhss_major_out.append(major_out)
        ps_to_rhss_minor_out.append(minor_out)
    
    # Update Y mappings (only non-reduced Y dimensions)
    ys_to_rhs_major_out = []
    ys_to_rhs_minor_out = []
    
    for i in range(ndim_y_in):
        if not is_y_in_for_reduce[i]:
            rh_major_in = in_encoding.ys_to_rhs_major[i]
            rh_minor_in = in_encoding.ys_to_rhs_minor[i]
            
            ys_to_rhs_major_out.append(in2out_rh_major[rh_major_in])
            ys_to_rhs_minor_out.append(in2out_rh_minor[rh_major_in][rh_minor_in])
    
    return TileDistributionEncoding(
        rs_lengths=rs_lengths_out[:cnt_ndim_r_out],
        hs_lengthss=hs_lengthss_out,
        ps_to_rhss_major=ps_to_rhss_major_out,
        ps_to_rhss_minor=ps_to_rhss_minor_out,
        ys_to_rhs_major=ys_to_rhs_major_out,
        ys_to_rhs_minor=ys_to_rhs_minor_out
    )

def make_slice_tile_distribution_encoding(in_encoding: TileDistributionEncoding,
                                        slices: Dict[int, int]) -> Tuple[TileDistributionEncoding, List[int], List[int]]:
    """
    Creates a sliced tile distribution by taking a subset of H components.
    `slices` is a dict mapping x_dim index to the number of components to keep.
    Returns the new encoding, plus the y_origins and y_lengths for the slice.
    """
    hs_lengthss_out = []
    for i, h_lengths in enumerate(in_encoding.hs_lengthss):
        if i in slices:
            hs_lengthss_out.append(h_lengths[:slices[i]])
        else:
            hs_lengthss_out.append(list(h_lengths))

    rs_lengths_out = list(in_encoding.rs_lengths)

    ps_to_rhss_major_out = []
    ps_to_rhss_minor_out = []
    for p_majors, p_minors in zip(in_encoding.ps_to_rhss_major, in_encoding.ps_to_rhss_minor):
        new_p_majors, new_p_minors = [], []
        for rh_major, rh_minor in zip(p_majors, p_minors):
            if rh_major == 0:
                new_p_majors.append(rh_major)
                new_p_minors.append(rh_minor)
            else:
                x_dim = rh_major - 1
                if x_dim < len(hs_lengthss_out) and rh_minor < len(hs_lengthss_out[x_dim]):
                    new_p_majors.append(rh_major)
                    new_p_minors.append(rh_minor)
        if new_p_majors:
            ps_to_rhss_major_out.append(new_p_majors)
            ps_to_rhss_minor_out.append(new_p_minors)
    
    ys_to_rhs_major_out = []
    ys_to_rhs_minor_out = []
    for rh_major, rh_minor in zip(in_encoding.ys_to_rhs_major, in_encoding.ys_to_rhs_minor):
        if rh_major == 0:
            ys_to_rhs_major_out.append(rh_major)
            ys_to_rhs_minor_out.append(rh_minor)
        else:
            x_dim = rh_major - 1
            if x_dim < len(hs_lengthss_out) and rh_minor < len(hs_lengthss_out[x_dim]):
                ys_to_rhs_major_out.append(rh_major)
                ys_to_rhs_minor_out.append(rh_minor)

    final_encoding = TileDistributionEncoding(
        rs_lengths=rs_lengths_out,
        hs_lengthss=hs_lengthss_out,
        ps_to_rhss_major=ps_to_rhss_major_out,
        ps_to_rhss_minor=ps_to_rhss_minor_out,
        ys_to_rhs_major=ys_to_rhs_major_out,
        ys_to_rhs_minor=ys_to_rhs_minor_out
    )

    # For this simplified visualization, origins are all 0 as we slice from the beginning.
    sliced_y_origins = [0] * final_encoding.ndim_y
    sliced_y_lengths = final_encoding.detail.ys_lengths
    
    return final_encoding, sliced_y_origins, sliced_y_lengths 