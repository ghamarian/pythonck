"""
Python implementation of tensor_descriptor.hpp from Composable Kernels.

This module provides tensor descriptor functionality for describing
multi-dimensional tensor layouts with transformations.
"""

from typing import List, Tuple, Optional, Union, Any, TypeVar, Generic, Dict
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import sympy as sp

from .tensor_coordinate import MultiIndex


# Define a custom XOR function for symbolic representation
class XorFunction(sp.Function):
    """Custom SymPy function to represent XOR operation."""
    
    @classmethod
    def eval(cls, x, y):
        """Evaluate XOR if both arguments are integers."""
        # Only evaluate if both arguments are concrete integers
        if x.is_integer and y.is_integer and x.is_number and y.is_number:
            try:
                return int(x) ^ int(y)
            except (TypeError, ValueError):
                pass
        # For complex expressions (like floor operations), return None to keep symbolic
        return None
    
    def _latex(self, printer):
        """LaTeX representation."""
        args = [printer._print(arg) for arg in self.args]
        return f"{args[0]} \\oplus {args[1]}"
    
    def _pretty(self, printer):
        """Pretty printing representation."""
        from sympy.printing.pretty.stringpict import prettyForm
        args = [printer._print(arg) for arg in self.args]
        return prettyForm(f"{args[0]} ⊕ {args[1]}")
    
    def __str__(self):
        """String representation."""
        return f"{self.args[0]} ⊕ {self.args[1]}"

# Create the XOR function instance
Xor = XorFunction


class Transform(ABC):
    """Abstract base class for coordinate transformations."""
    
    @abstractmethod
    def calculate_lower_index(self, idx_upper: MultiIndex) -> MultiIndex:
        """Calculate lower index from upper index."""
        pass
    
    @abstractmethod
    def calculate_upper_index(self, idx_lower: MultiIndex) -> MultiIndex:
        """Calculate upper index from lower index."""
        pass
    
    @abstractmethod
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self) -> bool:
        """Check if all valid upper indices map to valid lower indices."""
        pass
    
    @abstractmethod
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_upper: MultiIndex) -> bool:
        """Check if specific upper index maps to valid lower index."""
        pass
    
    @abstractmethod
    def sympy_calculate_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Calculate lower symbols from upper symbols using SymPy expressions.
        
        Corresponds to calculate_lower_index() - computes lower (physical) from upper (logical).
        """
        pass
    
    @abstractmethod
    def sympy_calculate_upper(self, lower_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Calculate upper symbols from lower symbols using SymPy expressions.
        
        Corresponds to calculate_upper_index() - computes upper (logical) from lower (physical).
        """
        pass
    
    @abstractmethod
    def update_lower_index(self, idx_upper_diff: MultiIndex, idx_lower_current: MultiIndex, 
                          idx_upper_new: MultiIndex) -> Tuple[MultiIndex, MultiIndex]:
        """Update lower index using difference in upper indices.
        
        This is more efficient than recalculating from scratch and matches the C++ 
        update_lower_index implementation pattern.
        
        Args:
            idx_upper_diff: Difference in upper indices (new - current)
            idx_lower_current: Current lower index to update
            idx_upper_new: New upper index (for reference/validation)
            
        Returns:
            Tuple of (idx_lower_diff, idx_lower_updated) where:
            - idx_lower_diff: Difference in lower indices
            - idx_lower_updated: Updated lower index (idx_lower_current + idx_lower_diff)
        """
        pass
    
    # Backward compatibility aliases for the old directional naming
    def sympy_upper_to_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Deprecated: Use sympy_calculate_lower instead (computes lower from upper)."""
        return self.sympy_calculate_lower(upper_symbols)
    
    def sympy_lower_to_upper(self, lower_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Deprecated: Use sympy_calculate_upper instead (computes upper from lower)."""
        return self.sympy_calculate_upper(lower_symbols)



class EmbedTransform(Transform):
    """
    Embed transform that maps multi-dimensional indices to linear offset.
    
    This is the most common transform used for strided tensor layouts.
    """
    
    def __init__(self, lengths: List[int], strides: List[int]):
        """
        Initialize embed transform.
        
        Args:
            lengths: Dimension lengths
            strides: Dimension strides
        """
        if len(lengths) != len(strides):
            raise ValueError(f"Lengths and strides must have same size: {len(lengths)} vs {len(strides)}")
        
        self.lengths = lengths
        self.strides = strides
        self.ndim = len(lengths)
    
    def get_num_of_lower_dimension(self) -> int:
        """Get number of lower dimensions (1)."""
        return 1
    
    def get_num_of_upper_dimension(self) -> int:
        """Get number of upper dimensions."""
        return self.ndim
    
    def get_upper_lengths(self) -> List[int]:
        """Get upper dimension lengths."""
        return self.lengths
    
    def calculate_lower_index(self, idx_upper: MultiIndex) -> MultiIndex:
        """Calculate linear offset from multi-dimensional index."""
        if len(idx_upper) != self.ndim:
            raise ValueError(f"Index dimension {len(idx_upper)} doesn't match transform dimension {self.ndim}")
        
        offset = 0
        for i in range(self.ndim):
            offset += idx_upper[i] * self.strides[i]
        
        return MultiIndex(1, [offset])
    
    def calculate_upper_index(self, idx_lower: MultiIndex) -> MultiIndex:
        """Calculate multi-dimensional index from linear offset."""
        if len(idx_lower) != 1:
            raise ValueError("Lower index must be 1D for embed transform")
        
        offset = idx_lower[0]
        upper_values = []
        
        # Decompose offset using strides (assuming row-major order)
        for i in range(self.ndim):
            if self.strides[i] > 0:
                idx = offset // self.strides[i]
                offset = offset % self.strides[i]
                upper_values.append(idx)
            else:
                upper_values.append(0)
        
        return MultiIndex(self.ndim, upper_values)
    
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self) -> bool:
        """Embed transform always maps valid indices."""
        return True
    
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_upper: MultiIndex) -> bool:
        """Check if upper index is within bounds."""
        for i in range(self.ndim):
            if idx_upper[i] < 0 or idx_upper[i] >= self.lengths[i]:
                return False
        return True
    
    def sympy_calculate_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Calculate linear offset from multi-dimensional symbols (computes lower from upper)."""
        if len(upper_symbols) != self.ndim:
            raise ValueError(f"Upper symbols {len(upper_symbols)} doesn't match transform dimension {self.ndim}")
        
        offset = 0
        for i in range(self.ndim):
            offset += upper_symbols[i] * self.strides[i]
        
        return [offset]
    
    def sympy_calculate_upper(self, lower_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Calculate multi-dimensional symbols from linear offset symbol (computes upper from lower)."""
        if len(lower_symbols) != 1:
            raise ValueError("Lower symbols must be 1D for embed transform")
        
        offset = lower_symbols[0]
        upper_symbols = []
        
        # Decompose offset using strides
        for i in range(self.ndim):
            if self.strides[i] > 0:
                idx = offset // self.strides[i]
                offset = offset % self.strides[i]
                upper_symbols.append(idx)
            else:
                upper_symbols.append(sp.Integer(0))
        
        return upper_symbols
    
    def update_lower_index(self, idx_upper_diff: MultiIndex, idx_lower_current: MultiIndex, 
                          idx_upper_new: MultiIndex) -> Tuple[MultiIndex, MultiIndex]:
        """Update lower index using stride multiplication."""
        if len(idx_upper_diff) != self.ndim or len(idx_lower_current) != 1:
            raise ValueError(f"EmbedTransform expects {self.ndim}D upper and 1D lower indices")
        
        # Calculate lower difference: sum of (upper_diff[i] * stride[i])
        lower_diff_value = 0
        for i in range(self.ndim):
            lower_diff_value += idx_upper_diff[i] * self.strides[i]
        
        idx_lower_diff = MultiIndex(1, [lower_diff_value])
        
        # Update: lower_new = lower_current + lower_diff
        idx_lower_updated = MultiIndex(1, [idx_lower_current[0] + idx_lower_diff[0]])
        
        return idx_lower_diff, idx_lower_updated


class UnmergeTransform(Transform):
    """
    Unmerge transform that decomposes a linear index into multi-dimensional indices.
    
    This is used for packed tensor layouts.
    """
    
    def __init__(self, lengths: List[int]):
        """
        Initialize unmerge transform.
        
        Args:
            lengths: Dimension lengths
        """
        self.lengths = lengths
        self.ndim = len(lengths)
        
        # Calculate strides for packed layout (row-major)
        self.strides = [1]
        for i in range(self.ndim - 1, 0, -1):
            self.strides.insert(0, self.strides[0] * self.lengths[i])
    
    def get_num_of_lower_dimension(self) -> int:
        """Get number of lower dimensions (1)."""
        return 1
    
    def get_num_of_upper_dimension(self) -> int:
        """Get number of upper dimensions."""
        return self.ndim
    
    def get_upper_lengths(self) -> List[int]:
        """Get upper dimension lengths."""
        return self.lengths
    
    def calculate_lower_index(self, idx_upper: MultiIndex) -> MultiIndex:
        """Calculate linear offset from multi-dimensional index."""
        if len(idx_upper) != self.ndim:
            raise ValueError(f"Index dimension {len(idx_upper)} doesn't match transform dimension {self.ndim}")
        
        offset = 0
        for i in range(self.ndim):
            offset += idx_upper[i] * self.strides[i]
        # offset = np.array(idx_upper) @ np.array(self.strides)  
        
        return MultiIndex(1, [offset])
    
    def calculate_upper_index(self, idx_lower: MultiIndex) -> MultiIndex:
        """Calculate multi-dimensional index from linear offset."""
        if len(idx_lower) != 1:
            raise ValueError("Lower index must be 1D for unmerge transform")
        
        offset = idx_lower[0]
        upper_values = []
        
        for i in range(self.ndim):
            idx = offset // self.strides[i]
            offset = offset % self.strides[i]
            upper_values.append(idx)
        
        return MultiIndex(self.ndim, upper_values)
    
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self) -> bool:
        """Unmerge transform always maps valid indices."""
        return True
    
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_upper: MultiIndex) -> bool:
        """Check if upper index is within bounds."""
        for i in range(self.ndim):
            if idx_upper[i] < 0 or idx_upper[i] >= self.lengths[i]:
                return False
        return True
    
    def sympy_calculate_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Calculate linear offset from multi-dimensional symbols (computes lower from upper)."""
        if len(upper_symbols) != self.ndim:
            raise ValueError(f"Upper symbols {len(upper_symbols)} doesn't match transform dimension {self.ndim}")
        
        offset = 0
        for i in range(self.ndim):
            offset += upper_symbols[i] * self.strides[i]
        
        return [offset]
    
    def sympy_calculate_upper(self, lower_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Calculate multi-dimensional symbols from linear offset symbol (computes upper from lower)."""
        if len(lower_symbols) != 1:
            raise ValueError("Lower symbols must be 1D for unmerge transform")
        
        offset = lower_symbols[0]
        upper_symbols = []
        
        # Decompose offset using strides
        remaining_offset = offset
        for i in range(self.ndim):
            idx = remaining_offset // self.strides[i]
            remaining_offset = remaining_offset % self.strides[i]
            upper_symbols.append(idx)
        
        return upper_symbols
    
    def update_lower_index(self, idx_upper_diff: MultiIndex, idx_lower_current: MultiIndex, 
                          idx_upper_new: MultiIndex) -> Tuple[MultiIndex, MultiIndex]:
        """Update lower index using incremental calculation.
        
        Following C++ unmerge pattern: calculate lower diff from upper diff,
        then add to current lower index.
        """
        if len(idx_upper_diff) != self.ndim or len(idx_lower_current) != 1:
            raise ValueError(f"UnmergeTransform expects {self.ndim}D upper and 1D lower indices")
        
        # Match C++ unmerge update_lower_index: calculate_lower_index(idx_diff_low, idx_diff_up)
        # Then: idx_low += idx_diff_low
        
        # Calculate lower diff from upper diff using the same logic as calculate_lower_index
        lower_diff = 0
        for i in range(self.ndim):
            lower_diff += idx_upper_diff[i] * self.strides[i]
        
        idx_lower_diff = MultiIndex(1, [lower_diff])
        
        # Update: lower_new = lower_current + lower_diff
        idx_lower_updated = MultiIndex(1, [idx_lower_current[0] + idx_lower_diff[0]])
        
        return idx_lower_diff, idx_lower_updated


class OffsetTransform(Transform):
    """
    Offset transform that adds a constant offset.
    """
    
    def __init__(self, element_space_size: int, offset: int):
        """
        Initialize offset transform.
        
        Args:
            element_space_size: Size of element space
            offset: Constant offset to add
        """
        self.element_space_size = element_space_size
        self.offset = offset
    
    def get_num_of_lower_dimension(self) -> int:
        """Get number of lower dimensions (1)."""
        return 1
    
    def get_num_of_upper_dimension(self) -> int:
        """Get number of upper dimensions (1)."""
        return 1
    
    def calculate_lower_index(self, idx_upper: MultiIndex) -> MultiIndex:
        """Add offset to index."""
        if len(idx_upper) != 1:
            raise ValueError("Offset transform expects 1D upper index")
        
        return MultiIndex(1, [idx_upper[0] + self.offset])
    
    def calculate_upper_index(self, idx_lower: MultiIndex) -> MultiIndex:
        """Subtract offset from index."""
        if len(idx_lower) != 1:
            raise ValueError("Offset transform expects 1D lower index")
        
        return MultiIndex(1, [idx_lower[0] - self.offset])
    
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self) -> bool:
        """Offset transform preserves validity."""
        return True
    
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_upper: MultiIndex) -> bool:
        """Check if index is within element space."""
        return 0 <= idx_upper[0] < self.element_space_size

    def sympy_calculate_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Add offset to symbol (computes lower from upper)."""
        if len(upper_symbols) != 1:
            raise ValueError("Offset transform expects 1D upper symbols")
        
        return [upper_symbols[0] + self.offset]
    
    def sympy_calculate_upper(self, lower_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Subtract offset from symbol (computes upper from lower)."""
        if len(lower_symbols) != 1:
            raise ValueError("Offset transform expects 1D lower symbols")
        
        return [lower_symbols[0] - self.offset]
    
    def update_lower_index(self, idx_upper_diff: MultiIndex, idx_lower_current: MultiIndex, 
                          idx_upper_new: MultiIndex) -> Tuple[MultiIndex, MultiIndex]:
        """Update lower index (add same offset to difference)."""
        if len(idx_upper_diff) != 1 or len(idx_lower_current) != 1:
            raise ValueError("Offset transform expects 1D indices")
        
        # For offset: lower_diff = upper_diff (offset cancels out in difference)
        idx_lower_diff = MultiIndex(1, [idx_upper_diff[0]])
        
        # Update: lower_new = lower_current + lower_diff
        idx_lower_updated = MultiIndex(1, [idx_lower_current[0] + idx_lower_diff[0]])
        
        return idx_lower_diff, idx_lower_updated
    
    def __repr__(self) -> str:
        """String representation."""
        return f"OffsetTransform(offset={self.offset})"


@dataclass
class PassThroughTransform(Transform):
    """
    Pass-through transform that leaves dimension unchanged.
    
    This is the identity transform for a single dimension.
    
    Attributes:
        length: Length of the dimension
    """
    
    length: int
    
    def get_num_of_lower_dimension(self) -> int:
        """Get number of lower dimensions (1)."""
        return 1
    
    def get_num_of_upper_dimension(self) -> int:
        """Get number of upper dimensions (1)."""
        return 1
    
    def get_upper_lengths(self) -> List[int]:
        """Get upper dimension lengths."""
        return [self.length]
    
    def calculate_lower_index(self, idx_upper: MultiIndex) -> MultiIndex:
        """Calculate lower index (same as upper)."""
        return idx_upper
    
    def calculate_upper_index(self, idx_lower: MultiIndex) -> MultiIndex:
        """Calculate upper index (same as lower)."""
        return idx_lower
    
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self) -> bool:
        """Pass-through always maps valid indices."""
        return True
    
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_upper: MultiIndex) -> bool:
        """Check if index is within bounds."""
        return 0 <= idx_upper[0] < self.length
    
    def sympy_calculate_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Pass through symbols unchanged (computes lower from upper)."""
        if len(upper_symbols) != 1:
            raise ValueError("PassThrough transform expects 1D upper symbols")
        return upper_symbols
    
    def sympy_calculate_upper(self, lower_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Pass through symbols unchanged (computes upper from lower)."""
        if len(lower_symbols) != 1:
            raise ValueError("PassThrough transform expects 1D lower symbols")
        return lower_symbols
    
    def update_lower_index(self, idx_upper_diff: MultiIndex, idx_lower_current: MultiIndex, 
                          idx_upper_new: MultiIndex) -> Tuple[MultiIndex, MultiIndex]:
        """Update lower index (pass through for PassThroughTransform)."""
        if len(idx_upper_diff) != 1 or len(idx_lower_current) != 1:
            raise ValueError("PassThrough transform expects 1D indices")
        
        # For pass-through: lower_diff = upper_diff
        idx_lower_diff = MultiIndex(1, [idx_upper_diff[0]])
        
        # Update: lower_new = lower_current + lower_diff
        idx_lower_updated = MultiIndex(1, [idx_lower_current[0] + idx_lower_diff[0]])
        
        return idx_lower_diff, idx_lower_updated
    
    def calculate_upper_dimension_safe_vector_length_strides(
        self,
        vector_lengths_lower: List[int],
        vector_strides_lower: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Calculate safe vector lengths and strides (pass through)."""
        return vector_lengths_lower, vector_strides_lower
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PassThroughTransform(length={self.length})"


@dataclass
class PadTransform(Transform):
    """
    Padding transform that adds padding to a dimension.
    
    Attributes:
        lower_length: Original dimension length
        left_pad: Left padding size
        right_pad: Right padding size
    """
    
    lower_length: int
    left_pad: int
    right_pad: int
    
    def get_num_of_lower_dimension(self) -> int:
        """Get number of lower dimensions (1)."""
        return 1
    
    def get_num_of_upper_dimension(self) -> int:
        """Get number of upper dimensions (1)."""
        return 1
    
    def get_upper_lengths(self) -> List[int]:
        """Get upper dimension lengths (with padding)."""
        return [self.lower_length + self.left_pad + self.right_pad]
    
    def calculate_lower_index(self, idx_upper: MultiIndex) -> MultiIndex:
        """Calculate lower index (adjust for padding)."""
        idx = idx_upper[0] - self.left_pad
        # Match C++: NO clamping, just subtract left padding
        return MultiIndex(1, [idx])
    
    def calculate_upper_index(self, idx_lower: MultiIndex) -> MultiIndex:
        """Calculate upper index (add padding offset)."""
        return MultiIndex(1, [idx_lower[0] + self.left_pad])
    
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self) -> bool:
        """Padding with no clamping doesn't guarantee valid lower indices."""
        return False
    
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_upper: MultiIndex) -> bool:
        """Check if index is within padded bounds."""
        return 0 <= idx_upper[0] < self.get_upper_lengths()[0]
    
    def sympy_calculate_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Adjust symbol for padding - subtract left pad (computes lower from upper)."""
        if len(upper_symbols) != 1:
            raise ValueError("Pad transform expects 1D upper symbols")
        
        # For upper -> lower: subtract left padding (match C++ - no clamping)
        adjusted = upper_symbols[0] - self.left_pad
        return [adjusted]
    
    def sympy_calculate_upper(self, lower_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Add padding offset to symbol (computes upper from lower)."""
        if len(lower_symbols) != 1:
            raise ValueError("Pad transform expects 1D lower symbols")
        
        # For lower -> upper: add left padding
        return [lower_symbols[0] + self.left_pad]
    
    def calculate_upper_dimension_safe_vector_length_strides(
        self,
        vector_lengths_lower: List[int],
        vector_strides_lower: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Calculate safe vector lengths and strides."""
        # Padding can break vectorization at boundaries
        if self.left_pad > 0 or self.right_pad > 0:
            return [1], [1]
        return vector_lengths_lower, vector_strides_lower
    
    def update_lower_index(self, idx_upper_diff: MultiIndex, idx_lower_current: MultiIndex, 
                          idx_upper_new: MultiIndex) -> Tuple[MultiIndex, MultiIndex]:
        """Update lower index with direct pass-through.
        
        Match C++ behavior: idx_diff_low[I0] = idx_diff_up[I0]; idx_low += idx_diff_low;
        Since padding is just an offset, the difference passes through unchanged.
        """
        if len(idx_upper_diff) != 1 or len(idx_lower_current) != 1:
            raise ValueError("Pad transform expects 1D indices")
        
        # Match C++: Direct pass-through of differences
        idx_lower_diff = MultiIndex(1, [idx_upper_diff[0]])
        
        # Update: lower_new = lower_current + lower_diff
        idx_lower_updated = MultiIndex(1, [idx_lower_current[0] + idx_lower_diff[0]])
        
        return idx_lower_diff, idx_lower_updated
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PadTransform(length={self.lower_length}, left={self.left_pad}, right={self.right_pad})"


@dataclass
class MergeTransform(Transform):
    """
    Merge transform that combines multiple dimensions into one.
    
    This is the inverse of UnmergeTransform.
    
    Attributes:
        lengths: Lengths of dimensions to merge
    """
    
    lengths: List[int]
    
    def get_num_of_lower_dimension(self) -> int:
        """Get number of lower dimensions."""
        return len(self.lengths)
    
    def get_num_of_upper_dimension(self) -> int:
        """Get number of upper dimensions (1)."""
        return 1
    
    def get_upper_lengths(self) -> List[int]:
        """Get upper dimension lengths (product of lower)."""
        return [math.prod(self.lengths)]
    
    def calculate_lower_index(self, idx_upper: MultiIndex) -> MultiIndex:
        """Calculate lower indices from merged index."""
        idx = idx_upper[0]
        lower_indices = []
        
        # Convert linear index to multi-dimensional indices
        for i in range(len(self.lengths) - 1, -1, -1):
            lower_indices.append(idx % self.lengths[i])
            idx //= self.lengths[i]
        
        return MultiIndex(len(self.lengths), list(reversed(lower_indices)))
    
    def calculate_upper_index(self, idx_lower: MultiIndex) -> MultiIndex:
        """Calculate merged index from lower indices."""
        if len(idx_lower) != len(self.lengths):
            raise ValueError(f"Index dimension {len(idx_lower)} doesn't match transform dimension {len(self.lengths)}")
        
        # Check bounds for each dimension
        for i, idx in enumerate(idx_lower):
            if idx < 0 or idx >= self.lengths[i]:
                raise ValueError("Index out of bounds")
        
        idx = 0
        stride = 1
        for i in range(len(self.lengths) - 1, -1, -1):
            idx += idx_lower[i] * stride
            stride *= self.lengths[i]
        return MultiIndex(1, [idx])
    
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self) -> bool:
        """Merge transform always maps valid indices."""
        return True
    
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_upper: MultiIndex) -> bool:
        """Check if index is within merged bounds."""
        return 0 <= idx_upper[0] < self.get_upper_lengths()[0]
    
    def sympy_calculate_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Decompose one symbol into multiple (computes lower from upper).
        
        This matches calculate_lower_index: takes 1 upper symbol and produces 
        multiple lower symbols (decomposition).
        
        Corresponds to C++ calculate_lower_index: upper (single merged) → lower (multiple separate)
        """
        if len(upper_symbols) != 1:
            raise ValueError("Merge transform expects 1D upper symbols for upper->lower")
        
        merged_symbol = upper_symbols[0]
        lower_symbols = []
        
        # Calculate strides for packed layout (row-major)
        strides = [1]
        for i in range(len(self.lengths) - 1, 0, -1):
            strides.insert(0, strides[0] * self.lengths[i])
        
        # Extract each dimension using division and modulo
        # This matches the C++ merge calculate_lower_index implementation
        remaining = merged_symbol
        for i in range(len(self.lengths)):
            if i < len(self.lengths) - 1:
                lower_symbols.append(remaining // strides[i])
                remaining = remaining % strides[i]
            else:
                lower_symbols.append(remaining)
        
        return lower_symbols
    
    def sympy_calculate_upper(self, lower_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Merge multiple symbols into one (computes upper from lower).
        
        This matches calculate_upper_index: takes multiple lower symbols and produces
        1 upper symbol (composition).
        
        Corresponds to C++ calculate_upper_index: lower (multiple separate) → upper (single merged)
        """
        if len(lower_symbols) != len(self.lengths):
            raise ValueError(f"Lower symbols {len(lower_symbols)} doesn't match lengths {len(self.lengths)}")
        
        # Compose multiple lower symbols into single upper symbol
        # Lower level: multiple separate dimensions  
        # Upper level: single merged dimension
        result = 0
        stride = 1
        for i in range(len(self.lengths) - 1, -1, -1):
            result += lower_symbols[i] * stride
            stride *= self.lengths[i]
        
        return [result]
    
    def calculate_upper_dimension_safe_vector_length_strides(
        self,
        vector_lengths_lower: List[int],
        vector_strides_lower: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Calculate safe vector lengths and strides."""
        # Only the innermost dimension can be vectorized after merge
        if vector_strides_lower[-1] == 1:
            return [vector_lengths_lower[-1]], [1]
        return [1], [1]
    
    def update_lower_index(self, idx_upper_diff: MultiIndex, idx_lower_current: MultiIndex, 
                          idx_upper_new: MultiIndex) -> Tuple[MultiIndex, MultiIndex]:
        """Update lower index using incremental calculation.
        
        Match C++ merge_v3_division_mod update_lower_index pattern:
        - Calculate new lower indices from new upper index using division/mod
        - Compute differences incrementally
        """
        if len(idx_upper_diff) != 1 or len(idx_lower_current) != len(self.lengths):
            raise ValueError(f"MergeTransform expects 1D upper and {len(self.lengths)}D lower indices")
        
        # Match C++ merge_v3_division_mod update_lower_index implementation
        tmp = idx_upper_new[0]
        lower_diff_values = []
        lower_updated_values = []
        
        # Calculate strides for packed layout (row-major)
        strides = [1]
        for i in range(len(self.lengths) - 1, 0, -1):
            strides.insert(0, strides[0] * self.lengths[i])
        
        # Incremental update using division and mod (matches C++ pattern)
        for i in range(len(self.lengths) - 1):
            tmp2 = idx_lower_current[i]
            new_val = tmp // strides[i]
            lower_updated_values.append(new_val)
            lower_diff_values.append(new_val - tmp2)
            tmp %= strides[i]
        
        # Handle last dimension
        tmp2 = idx_lower_current[len(self.lengths) - 1]
        lower_updated_values.append(tmp)
        lower_diff_values.append(tmp - tmp2)
        
        idx_lower_diff = MultiIndex(len(self.lengths), lower_diff_values)
        idx_lower_updated = MultiIndex(len(self.lengths), lower_updated_values)
        
        return idx_lower_diff, idx_lower_updated
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MergeTransform(lengths={self.lengths})"


@dataclass
class XorTransform(Transform):
    """
    XOR transform that applies XOR operation for address scrambling.
    
    This transform takes two dimensions and applies XOR to create
    a scrambled memory access pattern.
    
    Attributes:
        lengths: Lengths of the two dimensions to XOR
        apply_modulo: Whether to apply modulo operation (matches C++ ApplyModulo template parameter)
    """
    
    lengths: List[int]
    apply_modulo: bool = True
    
    def __post_init__(self):
        """Validate XOR transform has exactly 2 dimensions."""
        if len(self.lengths) != 2:
            raise ValueError("XOR transform requires exactly 2 dimensions")
    
    def get_num_of_lower_dimension(self) -> int:
        """Get number of lower dimensions (2)."""
        return 2
    
    def get_num_of_upper_dimension(self) -> int:
        """Get number of upper dimensions (2)."""
        return 2
    
    def get_upper_lengths(self) -> List[int]:
        """Get upper dimension lengths (same as lower)."""
        return self.lengths
    
    def calculate_lower_index(self, idx_upper: MultiIndex) -> MultiIndex:
        """Calculate lower index by applying XOR (upper -> lower direction)."""
        if len(idx_upper) != 2:
            raise ValueError("XOR transform expects 2D upper index")
        
        # Match C++ CalculateLowerIndex: lower[0] = upper[0], lower[1] = upper[1] ^ f(upper[0])
        x, y = idx_upper[0], idx_upper[1]
        if self.apply_modulo:
            xor_value = x % self.lengths[1]
        else:
            xor_value = x
        return MultiIndex(2, [x, y ^ xor_value])
    
    def calculate_upper_index(self, idx_lower: MultiIndex) -> MultiIndex:
        """Calculate upper index by applying inverse XOR (lower -> upper direction)."""
        if len(idx_lower) != 2:
            raise ValueError("XOR transform expects 2D lower index")
        
        # Inverse operation: since XOR is self-inverse, use same formula with lower indices
        # upper[0] = lower[0], upper[1] = lower[1] ^ f(lower[0])
        x, y = idx_lower[0], idx_lower[1]
        if self.apply_modulo:
            xor_value = x % self.lengths[1]
        else:
            xor_value = x
        return MultiIndex(2, [x, y ^ xor_value])
    
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self) -> bool:
        """XOR transform preserves validity."""
        return True
    
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_upper: MultiIndex) -> bool:
        """Check if indices are within bounds."""
        for i, idx in enumerate(idx_upper):
            if idx < 0 or idx >= self.lengths[i]:
                return False
        return True
    
    def sympy_calculate_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Apply XOR transform to compute lower symbols from upper symbols.
        
        This mirrors calculate_lower_index and exactly matches the C++ 
        CalculateLowerIndex implementation.
        """
        if len(upper_symbols) != 2:
            raise ValueError("XOR transform expects 2D upper symbols")
        
        # Computes lower symbols from upper symbols
        # upper_symbols[0], upper_symbols[1] are UPPER coordinate symbols
        # We want to compute LOWER coordinate symbols
        upper_x, upper_y = upper_symbols[0], upper_symbols[1]
        
        # From C++ CalculateLowerIndex: 
        # lower[0] = upper[0]
        # lower[1] = upper[1] ^ f(upper[0])
        lower_x = upper_x  # First dimension passes through
        if self.apply_modulo:
            xor_expr = upper_x % self.lengths[1]
        else:
            xor_expr = upper_x
        lower_y = Xor(upper_y, xor_expr)
        
        return [lower_x, lower_y]
    
    def sympy_calculate_upper(self, lower_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Apply XOR transform to compute upper symbols from lower symbols.
        
        This mirrors calculate_upper_index: takes lower coordinate symbols 
        and produces upper coordinate symbols.
        """
        if len(lower_symbols) != 2:
            raise ValueError("XOR transform expects 2D lower symbols")
        
        # Computes upper symbols from lower symbols
        # lower_symbols[0], lower_symbols[1] are LOWER coordinate symbols
        # We want to compute UPPER coordinate symbols
        lower_x, lower_y = lower_symbols[0], lower_symbols[1]
        
        # From C++ CalculateLowerIndex: lower[1] = upper[1] ^ f(upper[0]) 
        # So to get upper from lower: upper[1] = lower[1] ^ f(upper[0])
        # But since upper[0] = lower[0], we have: upper[1] = lower[1] ^ f(lower[0])
        upper_x = lower_x  # First dimension passes through
        if self.apply_modulo:
            xor_expr = lower_x % self.lengths[1]
        else:
            xor_expr = lower_x
        upper_y = Xor(lower_y, xor_expr)
        
        return [upper_x, upper_y]
    
    def calculate_upper_dimension_safe_vector_length_strides(
        self,
        vector_lengths_lower: List[int],
        vector_strides_lower: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Calculate safe vector lengths and strides."""
        # XOR can break vectorization patterns, so be conservative
        return [1, 1], [1, 1]
    
    def update_lower_index(self, idx_upper_diff: MultiIndex, idx_lower_current: MultiIndex, 
                          idx_upper_new: MultiIndex) -> Tuple[MultiIndex, MultiIndex]:
        """Update lower index using incremental calculation.
        
        Match C++ xor_t update_lower_index pattern:
        - Calculate new lower index from new upper index
        - Compute difference incrementally
        """
        if len(idx_upper_diff) != 2 or len(idx_lower_current) != 2:
            raise ValueError("XorTransform expects 2D indices")
        
        # Match C++ xor_t update_lower_index: calculate_lower_index(idx_low, idx_up)
        # Then: idx_diff_low = idx_low - idx_low_old
        
        # Calculate new lower index from new upper index
        idx_lower_new = self.calculate_lower_index(idx_upper_new)
        
        # Calculate difference incrementally
        lower_diff_values = []
        for i in range(2):
            diff = idx_lower_new[i] - idx_lower_current[i]
            lower_diff_values.append(diff)
        
        idx_lower_diff = MultiIndex(2, lower_diff_values)
        
        return idx_lower_diff, idx_lower_new
    
    def __repr__(self) -> str:
        """String representation."""
        return f"XorTransform(lengths={self.lengths}, apply_modulo={self.apply_modulo})"


@dataclass
class ReplicateTransform(Transform):
    """
    Replicate transform that broadcasts a single value to multiple dimensions.
    
    This transform has no lower dimensions and creates upper dimensions
    that all map to the same value.
    
    Attributes:
        upper_lengths: Lengths of replicated dimensions
    """
    
    upper_lengths: List[int]
    
    def get_num_of_lower_dimension(self) -> int:
        """Get number of lower dimensions (0)."""
        return 0
    
    def get_num_of_upper_dimension(self) -> int:
        """Get number of upper dimensions."""
        return len(self.upper_lengths)
    
    def get_upper_lengths(self) -> List[int]:
        """Get upper dimension lengths."""
        return self.upper_lengths
    
    def calculate_lower_index(self, idx_upper: MultiIndex) -> MultiIndex:
        """Calculate lower index (empty for replicate)."""
        return MultiIndex(0, [])
    
    def calculate_upper_index(self, idx_lower: MultiIndex) -> MultiIndex:
        """Calculate upper index (zeros for replicate)."""
        return MultiIndex(len(self.upper_lengths), [0] * len(self.upper_lengths))
    
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self) -> bool:
        """Replicate always maps to valid (empty) lower index."""
        return True
    
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_upper: MultiIndex) -> bool:
        """Check if indices are within bounds."""
        for i, idx in enumerate(idx_upper):
            if idx < 0 or idx >= self.upper_lengths[i]:
                return False
        return True
    
    def sympy_calculate_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Replicate computes lower symbols (creates empty list).
        
        ReplicateTransform is special: it can handle both cases:
        - Proper computation: N symbols → 0 symbols (empty list)
        - Legacy case: 0 symbols → 0 symbols (for backward compatibility)
        """
        # Accept either the correct number of upper symbols or empty list for compatibility
        if len(upper_symbols) != 0 and len(upper_symbols) != len(self.upper_lengths):
            raise ValueError(f"Upper symbols {len(upper_symbols)} doesn't match upper lengths {len(self.upper_lengths)}")
        return []
    
    def sympy_calculate_upper(self, lower_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Replicate computes upper symbols (creates zeros for all dimensions).
        
        ReplicateTransform is special: it can handle both cases:
        - Proper computation: 0 symbols → N symbols (zeros)  
        - Legacy case: 1 symbol → 0 symbols (for backward compatibility)
        """
        # Accept either empty list or single symbol for compatibility
        if len(lower_symbols) != 0 and len(lower_symbols) != 1:
            raise ValueError(f"Replicate transform expects 0 or 1 lower symbols, got {len(lower_symbols)}")
        
        if len(lower_symbols) == 0:
            # Proper computation: 0 inputs → N outputs (zeros)
            return [sp.Integer(0)] * len(self.upper_lengths)
        else:
            # Legacy case: 1 input → 0 outputs (for backward compatibility)
            return []
    
    def calculate_upper_dimension_safe_vector_length_strides(
        self,
        vector_lengths_lower: List[int],
        vector_strides_lower: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Calculate safe vector lengths and strides."""
        # All dimensions can be vectorized since they're replicated
        return [length for length in self.upper_lengths], [0 for _ in self.upper_lengths]  # Stride 0 for broadcast
    
    def update_lower_index(self, idx_upper_diff: MultiIndex, idx_lower_current: MultiIndex, 
                          idx_upper_new: MultiIndex) -> Tuple[MultiIndex, MultiIndex]:
        """Update lower index (no change for ReplicateTransform).
        
        ReplicateTransform has no lower dimensions, so the difference and
        current index are both empty.
        """
        if len(idx_upper_diff) != len(self.upper_lengths) or len(idx_lower_current) != 0:
            raise ValueError(f"ReplicateTransform expects {len(self.upper_lengths)}D upper and 0D lower indices")
        
        # No lower dimensions, so both diff and updated are empty
        idx_lower_diff = MultiIndex(0, [])
        idx_lower_updated = MultiIndex(0, [])
        
        return idx_lower_diff, idx_lower_updated
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ReplicateTransform(lengths={self.upper_lengths})"


class TensorAdaptor:
    """
    Tensor adaptor that manages coordinate transformations.
    
    This class tracks transformations between different index spaces
    through a hierarchy of hidden dimensions.
    """
    
    def __init__(self,
                 transforms: List[Union[Transform, Dict]],
                 lower_dimension_hidden_idss: List[List[int]],
                 upper_dimension_hidden_idss: List[List[int]],
                 bottom_dimension_hidden_ids: List[int],
                 top_dimension_hidden_ids: List[int]):
        """
        Initialize tensor adaptor.
        
        Args:
            transforms: List of transformations (objects or dictionaries)
            lower_dimension_hidden_idss: Lower dimension IDs for each transform
            upper_dimension_hidden_idss: Upper dimension IDs for each transform
            bottom_dimension_hidden_ids: Bottom dimension IDs
            top_dimension_hidden_ids: Top dimension IDs
        """
        # Convert dictionary transforms to actual transform objects
        actual_transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                actual_transforms.append(self._convert_dict_transform_to_object(transform))
            else:
                actual_transforms.append(transform)
        
        self.transforms = actual_transforms
        self.lower_dimension_hidden_idss = lower_dimension_hidden_idss
        self.upper_dimension_hidden_idss = upper_dimension_hidden_idss
        self.bottom_dimension_hidden_ids = bottom_dimension_hidden_ids
        self.top_dimension_hidden_ids = top_dimension_hidden_ids
        
        # Calculate derived properties
        self.ndim_hidden = self._calculate_ndim_hidden()
        
        # Validate dimensions
        if not (len(self.transforms) == len(self.lower_dimension_hidden_idss) == len(self.upper_dimension_hidden_idss)):
            raise ValueError("Number of transforms must match lower/upper dimension IDs")
    
    def _convert_dict_transform_to_object(self, dict_transform: Dict) -> Transform:
        """Convert dictionary transform to actual transform object."""
        name = dict_transform['name']
        meta_data = dict_transform['meta_data']
        
        if name == 'merge':
            return MergeTransform(meta_data)
        elif name == 'unmerge':
            return UnmergeTransform(meta_data)
        elif name == 'replicate':
            return ReplicateTransform(meta_data)
        elif name == 'pass_through':
            # meta_data should be the length for pass_through
            return PassThroughTransform(meta_data)
        elif name == 'pad':
            # meta_data should be [lower_length, left_pad, right_pad]
            return PadTransform(meta_data[0], meta_data[1], meta_data[2])
        elif name == 'embed':
            # meta_data should be [lengths, strides]
            return EmbedTransform(meta_data[0], meta_data[1])
        elif name == 'xor':
            # meta_data should be lengths for XOR
            return XorTransform(meta_data)
        else:
            raise ValueError(f"Unknown transform name: {name}")
    
    def _calculate_ndim_hidden(self) -> int:
        """Calculate number of hidden dimensions."""
        all_hidden_ids = set()
        all_hidden_ids.update(self.bottom_dimension_hidden_ids)
        all_hidden_ids.update(self.top_dimension_hidden_ids)
        for ids in self.lower_dimension_hidden_idss:
            all_hidden_ids.update(ids)
        for ids in self.upper_dimension_hidden_idss:
            all_hidden_ids.update(ids)
        return len(all_hidden_ids)
    
    def get_num_of_transform(self) -> int:
        """Get number of transformations."""
        return len(self.transforms)
    
    def get_num_of_hidden_dimension(self) -> int:
        """Get number of hidden dimensions."""
        return self.ndim_hidden
    
    def get_num_of_top_dimension(self) -> int:
        """Get number of top dimensions."""
        return len(self.top_dimension_hidden_ids)
    
    def get_num_of_bottom_dimension(self) -> int:
        """Get number of bottom dimensions."""
        return len(self.bottom_dimension_hidden_ids)
    
    def get_transforms(self) -> List[Transform]:
        """Get list of transforms."""
        return self.transforms
    
    def get_lower_dimension_hidden_idss(self) -> List[List[int]]:
        """Get lower dimension hidden IDs."""
        return self.lower_dimension_hidden_idss
    
    def get_upper_dimension_hidden_idss(self) -> List[List[int]]:
        """Get upper dimension hidden IDs."""
        return self.upper_dimension_hidden_idss
    
    def get_bottom_dimension_hidden_ids(self) -> List[int]:
        """Get bottom dimension hidden IDs."""
        return self.bottom_dimension_hidden_ids
    
    def get_top_dimension_hidden_ids(self) -> List[int]:
        """Get top dimension hidden IDs."""
        return self.top_dimension_hidden_ids
    
    def calculate_bottom_index(self, idx_top: MultiIndex) -> MultiIndex:
        """Calculate bottom index from top index."""
        # Initialize hidden index
        idx_hidden = MultiIndex(self.ndim_hidden)
        
        # Set top dimensions
        for i, hid in enumerate(self.top_dimension_hidden_ids):
            idx_hidden[hid] = idx_top[i]
        
        # Apply transforms in reverse order
        for itran in range(len(self.transforms) - 1, -1, -1):
            transform = self.transforms[itran]
            dims_low = self.lower_dimension_hidden_idss[itran]
            dims_up = self.upper_dimension_hidden_idss[itran]
            
            # Get upper index
            idx_up = MultiIndex(len(dims_up), [idx_hidden[hid] for hid in dims_up])
            
            # Calculate lower index
            idx_low = transform.calculate_lower_index(idx_up)
            
            # Set lower dimensions
            for i, hid in enumerate(dims_low):
                idx_hidden[hid] = idx_low[i]
        
        # Extract bottom index
        return MultiIndex(len(self.bottom_dimension_hidden_ids),
                         [idx_hidden[hid] for hid in self.bottom_dimension_hidden_ids])
    
    def is_static(self) -> bool:
        """Check if adaptor is static (compile-time known)."""
        return getattr(self, '_is_static', False)


@dataclass
class TensorDescriptor(TensorAdaptor):
    """
    Tensor descriptor that extends TensorAdaptor with element space information.
    
    This class represents a complete tensor layout description including
    transformations and memory space size.
    """
    
    def __init__(self,
                 transforms: List[Union[Transform, Dict]],
                 lower_dimension_hidden_idss: List[List[int]],
                 upper_dimension_hidden_idss: List[List[int]],
                 top_dimension_hidden_ids: List[int],
                 element_space_size: int,
                 guaranteed_vector_lengths: Optional[List[int]] = None,
                 guaranteed_vector_strides: Optional[List[int]] = None):
        """
        Initialize tensor descriptor.
        
        Args:
            transforms: List of transformations (objects or dictionaries)
            lower_dimension_hidden_idss: Lower dimension IDs for each transform
            upper_dimension_hidden_idss: Upper dimension IDs for each transform
            top_dimension_hidden_ids: Top dimension IDs
            element_space_size: Total element space size
            guaranteed_vector_lengths: Guaranteed vector lengths for each hidden dim
            guaranteed_vector_strides: Guaranteed vector strides for each hidden dim
        """
        # Bottom dimension is always [0] for tensor descriptor
        bottom_dimension_hidden_ids = [0]
        
        super().__init__(
            transforms=transforms,
            lower_dimension_hidden_idss=lower_dimension_hidden_idss,
            upper_dimension_hidden_idss=upper_dimension_hidden_idss,
            bottom_dimension_hidden_ids=bottom_dimension_hidden_ids,
            top_dimension_hidden_ids=top_dimension_hidden_ids
        )
        
        self.element_space_size = element_space_size
        
        # Set default guaranteed vector info if not provided
        if guaranteed_vector_lengths is None:
            guaranteed_vector_lengths = [-1] * self.ndim_hidden
        if guaranteed_vector_strides is None:
            guaranteed_vector_strides = [-1] * self.ndim_hidden
            
        self.guaranteed_vector_lengths = guaranteed_vector_lengths
        self.guaranteed_vector_strides = guaranteed_vector_strides
    
    def get_num_of_dimension(self) -> int:
        """Get number of visible dimensions."""
        return self.get_num_of_top_dimension()
    
    def get_length(self, dim: int) -> int:
        """Get length of a dimension."""
        # Find which hidden dimension corresponds to this top dimension
        if dim >= len(self.top_dimension_hidden_ids):
            raise IndexError(f"Dimension {dim} out of range")
        
        hidden_id = self.top_dimension_hidden_ids[dim]
        
        # Find which transform produces this hidden dimension
        for i, upper_ids in enumerate(self.upper_dimension_hidden_idss):
            if hidden_id in upper_ids:
                transform = self.transforms[i]
                # Get the position within the transform's upper dimensions
                local_idx = upper_ids.index(hidden_id)
                
                # Get length from transform
                if hasattr(transform, 'get_upper_lengths'):
                    lengths = transform.get_upper_lengths()
                    return lengths[local_idx]
                elif hasattr(transform, 'lengths'):
                    return transform.lengths[local_idx]
                elif hasattr(transform, 'upper_lengths'):
                    return transform.upper_lengths[local_idx]
                else:
                    # Default for simple transforms
                    return 1
        
        # If not found, return 1 as default
        return 1
    
    def get_lengths(self) -> List[int]:
        """Get all dimension lengths."""
        return [self.get_length(i) for i in range(self.get_num_of_dimension())]
    
    def get_element_space_size(self) -> int:
        """Get total element space size."""
        return self.element_space_size
    
    def calculate_offset(self, idx: Union[List[int], MultiIndex]) -> int:
        """Calculate linear offset from multi-dimensional index."""
        if isinstance(idx, list):
            idx = MultiIndex(len(idx), idx)
        
        bottom_idx = self.calculate_bottom_index(idx)
        return bottom_idx[0] if len(bottom_idx) > 0 else 0
    
    def is_static(self) -> bool:
        """Check if descriptor is static (compile-time known)."""
        return getattr(self, '_is_static', False)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"TensorDescriptor(ndim={self.get_num_of_dimension()}, "
                f"element_space_size={self.element_space_size})")


def make_naive_tensor_descriptor(lengths: List[int], 
                               strides: List[int],
                               guaranteed_last_dim_vector_length: int = -1,
                               guaranteed_last_dim_vector_stride: int = -1) -> TensorDescriptor:
    """
    Create a naive tensor descriptor with simple strided layout.
    
    Args:
        lengths: Dimension lengths
        strides: Dimension strides
        guaranteed_last_dim_vector_length: Guaranteed vector length for last dim
        guaranteed_last_dim_vector_stride: Guaranteed vector stride for last dim
        
    Returns:
        TensorDescriptor instance
    """
    if len(lengths) != len(strides):
        raise ValueError(f"Lengths and strides must have same size: {len(lengths)} vs {len(strides)}")
    
    ndim = len(lengths)
    
    # Create embed transform
    transform = EmbedTransform(lengths, strides)
    transforms = [transform]
    
    # Set up dimension mappings
    # Hidden dimension 0 is the bottom (offset)
    # Hidden dimensions 1..ndim are the top dimensions
    lower_dimension_hidden_idss = [[0]]  # Transform outputs to hidden dim 0
    upper_dimension_hidden_idss = [list(range(1, ndim + 1))]  # Transform takes hidden dims 1..ndim
    top_dimension_hidden_ids = list(range(1, ndim + 1))
    
    # Calculate element space size
    # element_space_size = 1
    # for i in range(ndim):
    #     element_space_size += (lengths[i] - 1) * strides[i]
    element_space_size = (np.array(lengths) - 1) @ np.array(strides) + 1 
    
    # Set up guaranteed vector info
    guaranteed_vector_lengths = [-1] * (ndim + 1)  # +1 for bottom dimension
    guaranteed_vector_strides = [-1] * (ndim + 1)
    
    if guaranteed_last_dim_vector_length != -1:
        guaranteed_vector_lengths[-1] = guaranteed_last_dim_vector_length
    if guaranteed_last_dim_vector_stride != -1:
        guaranteed_vector_strides[-1] = guaranteed_last_dim_vector_stride
    
    return TensorDescriptor(
        transforms=transforms,
        lower_dimension_hidden_idss=lower_dimension_hidden_idss,
        upper_dimension_hidden_idss=upper_dimension_hidden_idss,
        top_dimension_hidden_ids=top_dimension_hidden_ids,
        element_space_size=element_space_size,
        guaranteed_vector_lengths=guaranteed_vector_lengths,
        guaranteed_vector_strides=guaranteed_vector_strides
    )


def make_naive_tensor_descriptor_packed(lengths: List[int],
                                      guaranteed_last_dim_vector_length: int = -1) -> TensorDescriptor:
    """
    Create a naive tensor descriptor with packed layout (no padding).
    
    Args:
        lengths: Dimension lengths
        guaranteed_last_dim_vector_length: Guaranteed vector length for last dim
        
    Returns:
        TensorDescriptor instance
    """
    ndim = len(lengths)
    
    # Create unmerge transform
    transform = UnmergeTransform(lengths)
    transforms = [transform]
    
    # Set up dimension mappings
    lower_dimension_hidden_idss = [[0]]
    upper_dimension_hidden_idss = [list(range(1, ndim + 1))]
    top_dimension_hidden_ids = list(range(1, ndim + 1))
    
    # Calculate element space size (product of lengths)
    element_space_size = math.prod(lengths) 
    
    # Set up guaranteed vector info
    guaranteed_vector_lengths = [-1] * (ndim + 1)
    guaranteed_vector_strides = [-1] * (ndim + 1)
    
    if guaranteed_last_dim_vector_length != -1:
        guaranteed_vector_lengths[-1] = guaranteed_last_dim_vector_length
    guaranteed_vector_strides[-1] = 1  # Packed layout has stride 1 for last dim
    
    return TensorDescriptor(
        transforms=transforms,
        lower_dimension_hidden_idss=lower_dimension_hidden_idss,
        upper_dimension_hidden_idss=upper_dimension_hidden_idss,
        top_dimension_hidden_ids=top_dimension_hidden_ids,
        element_space_size=element_space_size,
        guaranteed_vector_lengths=guaranteed_vector_lengths,
        guaranteed_vector_strides=guaranteed_vector_strides
    )


def make_naive_tensor_descriptor_aligned(lengths: List[int], align: int) -> TensorDescriptor:
    """
    Create a naive tensor descriptor with aligned layout.
    
    The second-to-last dimension stride is aligned to the specified alignment.
    
    Args:
        lengths: Dimension lengths
        align: Alignment requirement
        
    Returns:
        TensorDescriptor instance
    """
    ndim = len(lengths)
    if ndim < 2:
        raise ValueError("Aligned descriptor requires at least 2 dimensions")
    
    # Calculate strides with alignment
    strides = [1] * ndim
    
    # Last dimension has stride 1
    strides[-1] = 1
    
    # Second-to-last dimension has aligned stride
    stride_n_minus_2 = ((lengths[-1] + align - 1) // align) * align
    strides[-2] = stride_n_minus_2
    
    # Calculate remaining strides
    for i in range(ndim - 3, -1, -1):
        strides[i] = strides[i + 1] * lengths[i + 1]
    
    return make_naive_tensor_descriptor(lengths, strides)





def create_single_stage_descriptor(input_desc, transforms, lower_dimension_old_top_idss, upper_dimension_new_top_idss):
    """
    Create a single-stage tensor descriptor using C++-like simple approach.
    
    This matches the C++ make_single_stage_tensor_adaptor implementation:
    - Simple sequential ID assignment
    - No complex collision avoidance
    - Direct dimension mapping
    - Automatic flattening of nested transforms
    - Combines old and new transforms (does not replace)
    """
    # Use transforms directly (no nested transforms exist in this simplified version)
    flattened_transforms = transforms
    
    # Convert tuples to lists for consistency
    if isinstance(lower_dimension_old_top_idss, tuple):
        lower_dimension_old_top_idss = list(lower_dimension_old_top_idss)
    if isinstance(upper_dimension_new_top_idss, tuple):
        upper_dimension_new_top_idss = list(upper_dimension_new_top_idss)
    
    # Get input dimension count
    ndim_old_top = input_desc.get_num_of_top_dimension()
    
    # Calculate new dimension count
    all_new_top_ids = []
    for ids in upper_dimension_new_top_idss:
        all_new_top_ids.extend(ids)
    ndim_new_top = len(all_new_top_ids)
    
    # IMPORTANT: Combine old and new transforms (don't replace)
    # This matches C++ transform_tensor_descriptor behavior
    combined_transforms = input_desc.get_transforms() + flattened_transforms
    
    # Combine lower dimension mappings
    old_lower_idss = input_desc.get_lower_dimension_hidden_idss()
    combined_lower_idss = old_lower_idss + lower_dimension_old_top_idss
    
    # Combine upper dimension mappings (offset new ones by total hidden dimensions)
    old_upper_idss = input_desc.get_upper_dimension_hidden_idss()
    old_num_hidden = input_desc.get_num_of_hidden_dimension()
    new_upper_idss = []
    for ids in upper_dimension_new_top_idss:
        new_upper_idss.append([id + old_num_hidden for id in ids])
    combined_upper_idss = old_upper_idss + new_upper_idss
    
    # Bottom dimension stays the same
    bottom_dim_hidden_ids = [0]
    
    # Top dimension IDs: sequential from old_num_hidden
    top_dim_hidden_ids = list(range(old_num_hidden, old_num_hidden + ndim_new_top))
    
    # Create new tensor descriptor with combined transforms
    return TensorDescriptor(
        transforms=combined_transforms,
        lower_dimension_hidden_idss=combined_lower_idss,
        upper_dimension_hidden_idss=combined_upper_idss,
        top_dimension_hidden_ids=top_dim_hidden_ids,
        element_space_size=input_desc.get_element_space_size()
    )


def make_merge_transform(lengths: Union[List[int], Tuple]) -> Transform:
    """
    Create a merge transform - C++ equivalent with immediate flattening.
    
    Args:
        lengths: List or tuple of integers (dimension lengths) or Transform objects
        
    Returns:
        MergeTransform with flattened lengths (C++ equivalent behavior)
    """
    # Convert tuple to list if needed (for make_tuple compatibility)
    if isinstance(lengths, tuple):
        lengths = list(lengths)
    
    return MergeTransform(lengths)


def make_merge_transform_for_visualization(lengths: Union[List[int], Tuple]) -> Transform:
    """
    Create a merge transform - SIMPLIFIED VERSION.
    
    This function now creates simple MergeTransform objects by flattening any 
    nested Transform objects to their lengths. The previous hierarchical merge
    complexity was based on invalid C++ syntax patterns that don't exist.
    
    Used by: tensor_transform_app.py, parser, and other visualization components.
    
    Args:
        lengths: List or tuple of integers (dimension lengths) or Transform objects
        
    Returns:
        MergeTransform (always flattened)
    """
    # Convert tuple to list if needed (for make_tuple compatibility)
    if isinstance(lengths, tuple):
        lengths = list(lengths)
    
    # Flatten any Transform objects to their lengths
    flattened_lengths = []
    for item in lengths:
        if isinstance(item, Transform):
            # Extract length from Transform object
            if hasattr(item, 'length'):
                flattened_lengths.append(item.length)
            elif hasattr(item, 'lengths') and item.lengths:
                # For MergeTransform, use the product of its lengths
                flattened_lengths.append(math.prod(item.lengths))
            else:
                flattened_lengths.append(1)  # Default fallback
        else:
            flattened_lengths.append(int(item))
    
    # Always create a simple MergeTransform
    return MergeTransform(lengths=flattened_lengths)


def transform_tensor_descriptor(
    input_descriptor: TensorDescriptor,
    transforms: Union[List[Transform], Tuple[Transform, ...]],
    lower_dimension_hidden_idss: Union[List[List[int]], Tuple],
    upper_dimension_hidden_idss: Union[List[List[int]], Tuple]
) -> TensorDescriptor:
    """
    Create a new tensor descriptor by applying transforms - TRUE C++ EQUIVALENT VERSION.
    
    This version properly matches C++ transform_tensor_descriptor behavior:
    1. First use transform_tensor_adaptor to create new adaptor
    2. Then combine adaptor with element space size to create descriptor
    
    Used by: Normal Python API, tests, and C++ equivalent code.
    
    Args:
        input_descriptor: The input tensor descriptor
        transforms: List or tuple of transforms to apply
        lower_dimension_hidden_idss: Lower dimension indices for each transform
        upper_dimension_hidden_idss: Upper dimension indices for each transform
        
    Returns:
        A new TensorDescriptor with the transforms applied (true C++ equivalent behavior)
    """
    # Import here to avoid circular imports
    from .tensor_adaptor import transform_tensor_adaptor
    
    # Convert tuples to lists for consistency
    if isinstance(transforms, tuple):
        transforms = list(transforms)
    if isinstance(lower_dimension_hidden_idss, tuple):
        lower_dimension_hidden_idss = list(lower_dimension_hidden_idss)
    if isinstance(upper_dimension_hidden_idss, tuple):
        upper_dimension_hidden_idss = list(upper_dimension_hidden_idss)
    
    # Step 1: Use transform_tensor_adaptor to create new adaptor (C++ equivalent)
    new_adaptor = transform_tensor_adaptor(
        old_adaptor=input_descriptor,  # TensorDescriptor inherits from TensorAdaptor
        new_transforms=transforms,
        new_lower_dimension_old_top_idss=lower_dimension_hidden_idss,
        new_upper_dimension_new_top_idss=upper_dimension_hidden_idss
    )
    
    # Step 2: Create descriptor from adaptor + element space size (C++ equivalent)
    # Element space size remains the same - transforms don't change memory layout
    return TensorDescriptor(
        transforms=new_adaptor.transforms,
        lower_dimension_hidden_idss=new_adaptor.lower_dimension_hidden_idss,
        upper_dimension_hidden_idss=new_adaptor.upper_dimension_hidden_idss,
        top_dimension_hidden_ids=new_adaptor.top_dimension_hidden_ids,
        element_space_size=input_descriptor.get_element_space_size()
    )


def transform_tensor_descriptor_for_visualization(
    input_descriptor: TensorDescriptor,
    transforms: Union[List[Transform], Tuple[Transform, ...]],
    lower_dimension_hidden_idss: Union[List[List[int]], Tuple],
    upper_dimension_hidden_idss: Union[List[List[int]], Tuple]
) -> TensorDescriptor:
    """
    Create a new tensor descriptor by applying transforms - VISUALIZATION VERSION.
    
    This version preserves nested transform structures and uses multi-stage approach
    to maintain hierarchical information needed for proper graph visualization.
    
    Used by: tensor_transform_app.py, parser, and other visualization components.
    
    Args:
        input_descriptor: The input tensor descriptor
        transforms: List or tuple of transforms to apply
        lower_dimension_hidden_idss: Lower dimension indices for each transform
        upper_dimension_hidden_idss: Upper dimension indices for each transform
        
    Returns:
        A new TensorDescriptor with the transforms applied (preserving nested structures)
    """
    # Convert tuples to lists for consistency
    if isinstance(transforms, tuple):
        transforms = list(transforms)
    if isinstance(lower_dimension_hidden_idss, tuple):
        lower_dimension_hidden_idss = list(lower_dimension_hidden_idss)
    if isinstance(upper_dimension_hidden_idss, tuple):
        upper_dimension_hidden_idss = list(upper_dimension_hidden_idss)
    
    # Validate input
    if len(transforms) != len(lower_dimension_hidden_idss) or len(transforms) != len(upper_dimension_hidden_idss):
        raise ValueError("Number of transforms must match number of dimension index lists")
    
    # Use transforms directly (no nested transforms exist in this simplified version)
    flattened_transforms = transforms
    
    # Combine transforms from input descriptor and new transforms
    all_transforms = input_descriptor.get_transforms() + flattened_transforms
    
    # Combine dimension mappings
    all_lower_idss = input_descriptor.get_lower_dimension_hidden_idss() + lower_dimension_hidden_idss
    all_upper_idss = input_descriptor.get_upper_dimension_hidden_idss() + upper_dimension_hidden_idss
    
    # Collect all unique output dimension indices from all transforms
    all_output_indices = set()
    for upper_ids in upper_dimension_hidden_idss:
        all_output_indices.update(upper_ids)
    
    # Sort them to create the final top dimension IDs
    top_dimension_hidden_ids = sorted(list(all_output_indices)) if all_output_indices else input_descriptor.get_top_dimension_hidden_ids()
    
    # Create new tensor descriptor
    return TensorDescriptor(
        transforms=all_transforms,
        lower_dimension_hidden_idss=all_lower_idss,
        upper_dimension_hidden_idss=all_upper_idss,
        top_dimension_hidden_ids=top_dimension_hidden_ids,
        element_space_size=input_descriptor.get_element_space_size(),
        guaranteed_vector_lengths=input_descriptor.guaranteed_vector_lengths,
        guaranteed_vector_strides=input_descriptor.guaranteed_vector_strides
    )


def make_pass_through_transform(length: int) -> PassThroughTransform:
    """
    Create a pass-through transform.
    
    Args:
        length: Dimension length
        
    Returns:
        PassThroughTransform
    """
    return PassThroughTransform(length)


def make_unmerge_transform(lengths: List[int]) -> UnmergeTransform:
    """
    Create an unmerge transform.
    
    Args:
        lengths: Dimension lengths
        
    Returns:
        UnmergeTransform
    """
    return UnmergeTransform(lengths)


def make_embed_transform(lengths: List[int], strides: List[int]) -> EmbedTransform:
    """
    Create an embed transform.
    
    Args:
        lengths: Dimension lengths
        strides: Dimension strides
        
    Returns:
        EmbedTransform
    """
    return EmbedTransform(lengths, strides)


def make_pad_transform(lower_length: int, left_pad: int, right_pad: int) -> PadTransform:
    """
    Create a pad transform.
    
    Args:
        lower_length: Original dimension length
        left_pad: Left padding size
        right_pad: Right padding size
        
    Returns:
        PadTransform
    """
    return PadTransform(lower_length, left_pad, right_pad)


def make_replicate_transform(lengths: List[int]) -> ReplicateTransform:
    """
    Create a replicate transform.
    
    Args:
        lengths: Dimension lengths for replication
        
    Returns:
        ReplicateTransform
    """
    return ReplicateTransform(lengths)


def make_xor_transform(lengths: List[int], apply_modulo: bool = True) -> XorTransform:
    """
    Create an XOR transform.
    
    Args:
        lengths: Dimension lengths (must be exactly 2)
        apply_modulo: Whether to apply modulo operation
        
    Returns:
        XorTransform
    """
    return XorTransform(lengths, apply_modulo)


def make_offset_transform(element_space_size: int, offset: int) -> OffsetTransform:
    """
    Create an offset transform.
    
    Args:
        element_space_size: Size of element space
        offset: Constant offset to add
        
    Returns:
        OffsetTransform
    """
    return OffsetTransform(element_space_size, offset)


# Utility functions for C++ compatibility

def make_tuple(*args):
    """
    Create a tuple from arguments (C++ compatibility).
    
    Args:
        *args: Variable arguments
        
    Returns:
        Tuple of arguments
    """
    return tuple(args)


def number(value: int):
    """
    Create a number wrapper (C++ number<> compatibility).
    
    Args:
        value: Integer value
        
    Returns:
        The integer value (no wrapper needed in Python)
    """
    return value


def sequence(*values):
    """
    Create a sequence (C++ sequence<> compatibility).
    
    Args:
        *values: Variable arguments
        
    Returns:
        List of values
    """
    return list(values) 