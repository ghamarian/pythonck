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
        # Clamp to valid range
        idx = max(0, min(idx, self.lower_length - 1))
        return MultiIndex(1, [idx])
    
    def calculate_upper_index(self, idx_lower: MultiIndex) -> MultiIndex:
        """Calculate upper index (add padding offset)."""
        return MultiIndex(1, [idx_lower[0] + self.left_pad])
    
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self) -> bool:
        """Padding maps all indices to valid lower indices (clamped)."""
        return True
    
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_upper: MultiIndex) -> bool:
        """Check if index is within padded bounds."""
        return 0 <= idx_upper[0] < self.get_upper_lengths()[0]
    
    def sympy_calculate_lower(self, upper_symbols: List[sp.Expr]) -> List[sp.Expr]:
        """Adjust symbol for padding - subtract left pad (computes lower from upper)."""
        if len(upper_symbols) != 1:
            raise ValueError("Pad transform expects 1D upper symbols")
        
        # For upper -> lower: subtract left padding
        # But clamp to valid range [0, lower_length-1]
        adjusted = upper_symbols[0] - self.left_pad
        clamped = sp.Max(0, sp.Min(adjusted, self.lower_length - 1))
        return [clamped]
    
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
                    return transform.get_upper_lengths()[local_idx]
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


def transform_tensor_descriptor(
    input_descriptor: TensorDescriptor,
    transforms: List[Transform],
    lower_dimension_hidden_idss: List[List[int]],
    upper_dimension_hidden_idss: List[List[int]]
) -> TensorDescriptor:
    """
    Create a new tensor descriptor by applying a sequence of transforms to an input descriptor.
    
    Args:
        input_descriptor: The input tensor descriptor
        transforms: List of transforms to apply
        lower_dimension_hidden_idss: Lower dimension indices for each transform
        upper_dimension_hidden_idss: Upper dimension indices for each transform
        
    Returns:
        A new TensorDescriptor with the transforms applied
    """
    # Validate input
    if len(transforms) != len(lower_dimension_hidden_idss) or len(transforms) != len(upper_dimension_hidden_idss):
        raise ValueError("Number of transforms must match number of dimension index lists")
    
    # Combine transforms from input descriptor and new transforms
    all_transforms = input_descriptor.get_transforms() + transforms
    
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