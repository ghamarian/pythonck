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
class TileDistributionEncoding:
    """
    Encoding for tile distribution that describes how data is distributed.
    
    This encodes the mapping between:
    - R dimensions: Replication dimensions
    - H dimensions: Hierarchical dimensions (for each X dimension)
    - P dimensions: Partition dimensions (thread, warp, block)
    - Y dimensions: Tile dimensions
    - X dimensions: Tensor dimensions
    """
    
    rs_lengths: List[int]  # Replication dimension lengths
    hs_lengthss: List[List[int]]  # Hierarchical dimension lengths for each X
    ps_to_rhss_major: List[List[int]]  # P to RH major dimension mapping
    ps_to_rhss_minor: List[List[int]]  # P to RH minor dimension mapping
    ys_to_rhs_major: List[int]  # Y to RH major dimension mapping
    ys_to_rhs_minor: List[int]  # Y to RH minor dimension mapping
    
    def __post_init__(self):
        """Validate encoding after initialization."""
        # Validate dimensions match
        if len(self.ps_to_rhss_major) != len(self.ps_to_rhss_minor):
            raise ValueError("P dimension mappings must have same length")
        if len(self.ys_to_rhs_major) != len(self.ys_to_rhs_minor):
            raise ValueError("Y dimension mappings must have same length")
    
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
    
    def get_distributed_spans(self) -> List[TileDistributedSpan]:
        """Get distributed spans for each X dimension."""
        spans = []
        
        for x_idx in range(self.ndim_x):
            # Get H lengths for this X dimension
            h_lengths = self.hs_lengthss[x_idx]
            
            # Find which H dimensions are distributed (referenced by Y)
            distributed_h_indices = []
            for y_idx in range(self.ndim_y):
                if self.ys_to_rhs_major[y_idx] == x_idx + 1:  # +1 for R dimension
                    h_idx = self.ys_to_rhs_minor[y_idx]
                    if h_idx not in distributed_h_indices:
                        distributed_h_indices.append(h_idx)
            
            # Create span with distributed H lengths
            distributed_lengths = [h_lengths[i] for i in sorted(distributed_h_indices)]
            spans.append(TileDistributedSpan(distributed_lengths))
        
        return spans


@dataclass
class TileDistribution:
    """
    Tile distribution that manages how tensor data is distributed across
    processing elements.
    
    Attributes:
        ps_ys_to_xs_adaptor: Adaptor mapping from (P,Y) to X dimensions
        ys_to_d_descriptor: Descriptor mapping from Y to linearized D dimension
        encoding: Static tile distribution encoding
        partition_index_func: Function to get current partition index
    """
    
    ps_ys_to_xs_adaptor: TensorAdaptor
    ys_to_d_descriptor: TensorDescriptor
    encoding: TileDistributionEncoding
    partition_index_func: Optional[callable] = None
    
    def __post_init__(self):
        """Validate distribution after initialization."""
        # Set default partition index function
        if self.partition_index_func is None:
            # Default: single thread (partition index = [0])
            self.partition_index_func = lambda: [0] * self.encoding.ndim_p
    
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
        """Get current partition index."""
        return self.partition_index_func()
    
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
    
    def get_y_indices_from_distributed_indices(self, distributed_indices: List[TileDistributedIndex]) -> List[int]:
        """
        Get Y indices from distributed indices.
        
        Args:
            distributed_indices: Distributed indices for each X dimension
            
        Returns:
            Y dimension indices
        """
        y_indices = [0] * self.ndim_y
        
        # Map from distributed indices to Y indices using encoding
        for y_idx in range(self.ndim_y):
            span_major = self.encoding.ys_to_rhs_major[y_idx] - 1  # -1 to convert to X index
            span_minor = self.encoding.ys_to_rhs_minor[y_idx]
            
            if 0 <= span_major < len(distributed_indices):
                dstr_index = distributed_indices[span_major]
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
                         encoding: TileDistributionEncoding,
                         partition_index_func: Optional[callable] = None) -> TileDistribution:
    """
    Create a tile distribution.
    
    Args:
        ps_ys_to_xs_adaptor: Adaptor mapping from (P,Y) to X dimensions
        ys_to_d_descriptor: Descriptor mapping from Y to linearized D dimension
        encoding: Static tile distribution encoding
        partition_index_func: Function to get current partition index
        
    Returns:
        TileDistribution instance
    """
    return TileDistribution(
        ps_ys_to_xs_adaptor=ps_ys_to_xs_adaptor,
        ys_to_d_descriptor=ys_to_d_descriptor,
        encoding=encoding,
        partition_index_func=partition_index_func
    ) 