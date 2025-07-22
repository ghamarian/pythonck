"""
Python implementation of sequence utility functions from C++ sequence.hpp.

This module provides utilities for sequence operations equivalent to
the C++ ck_tile sequence functionality.
"""

from typing import List, Callable, Any, Tuple
from functools import reduce
import math


def gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


class Multiplies:
    """Function object for multiplication, equivalent to C++ multiplies."""
    def __call__(self, a: int, b: int) -> int:
        return a * b


class Plus:
    """Function object for addition, equivalent to C++ plus."""
    def __call__(self, a: int, b: int) -> int:
        return a + b


def reduce_on_sequence(seq: List[int], reduce_func: Callable[[int, int], int], init: int) -> int:
    """
    Reduce a sequence using a reduction function with an initial value.
    
    Equivalent to C++ reduce_on_sequence.
    
    Args:
        seq: List of integers to reduce
        reduce_func: Function that takes two integers and returns one
        init: Initial value for the reduction
        
    Returns:
        Reduced value
        
    Example:
        reduce_on_sequence([2, 3, 4], Multiplies(), 1) -> 24
        reduce_on_sequence([1, 2, 3], Plus(), 0) -> 6
    """
    result = init
    for value in seq:
        result = reduce_func(result, value)
    return result


def reverse_slice_sequence(seq: List[int], slice_size: int, mask: List[int] = None) -> Tuple[List[int], List[int], int]:
    """
    Reverse slice sequence implementation that matches C++ algorithm.
    
    This processes the sequence from right to left, consuming the slice_size
    by taking as much as possible from each dimension.
    
    Args:
        seq: Sequence to slice
        slice_size: Size per slice
        mask: Optional mask (1 means sliceable, 0 means not)
        
    Returns:
        Tuple of (slice_lengths, slice_nums, split_idx)
    """
    if mask is None:
        mask = [1] * len(seq)
    
    if len(seq) != len(mask):
        raise ValueError("Sequence and mask must have same length")
    
    if len(seq) == 0:
        return [], [], 0
    
    # Process from right to left, consuming slice_size
    dim_lengths = []
    dim_slices = []
    split_idx = 0
    split_flag = False
    
    remaining = slice_size
    
    for i in range(len(seq) - 1, -1, -1):
        x = seq[i]
        m = mask[i]
        
        if m:  # This dimension is sliceable
            if remaining >= x:
                # We can take the full dimension, but only if it divides evenly
                if remaining % x != 0:
                    raise ValueError(f"Cannot evenly divide sequence {seq} with slice size {slice_size}")
                slice_length = x
                remaining = remaining // x
            else:
                # We can only take part of the dimension
                if x % remaining != 0:
                    raise ValueError(f"Cannot evenly divide sequence {seq} with slice size {slice_size}")
                slice_length = remaining
                remaining = 1
        else:  # This dimension is not sliceable
            slice_length = x
            # remaining stays the same for non-sliceable dimensions
        
        # Calculate number of slices for this dimension
        num_slices = x // slice_length
        
        # Insert at beginning to build left-to-right result
        dim_lengths.insert(0, slice_length)
        dim_slices.insert(0, num_slices)
        
        # Check split condition - first index where slicing differs from original
        if m and slice_length != x and remaining == 1 and not split_flag:
            split_flag = True
            split_idx = i
    
    # Verify we can evenly divide
    if remaining != 1:
        raise ValueError(f"Cannot evenly divide sequence {seq} with slice size {slice_size}")
    
    return dim_lengths, dim_slices, split_idx


def slice_sequence(seq: List[int], slice_size: int, mask: List[int] = None) -> Tuple[List[int], List[int], int]:
    """
    Slice sequence using the direct "consume from right to left" algorithm.
    
    Equivalent to C++ slice_sequence.
    
    Args:
        seq: Sequence to slice  
        slice_size: Size per slice
        mask: Optional mask
        
    Returns:
        Tuple of (slice_lengths, slice_nums, split_idx)
        
    Examples:
        slice_sequence([2, 1, 4, 2], 8) -> ([1, 1, 4, 2], [2, 1, 1, 1], 0)
        slice_sequence([4, 2, 4, 1, 2], 4) -> ([1, 1, 2, 1, 2], [4, 2, 2, 1, 1], 2)
        slice_sequence([4, 2, 8], 64) -> ([4, 2, 8], [1, 1, 1], 0)
        slice_sequence([4, 2, 8], 32) -> ([2, 2, 8], [2, 1, 1], 0) 
        slice_sequence([4, 2, 8], 8) -> ([1, 1, 8], [4, 2, 1], 1)
        slice_sequence([4, 2, 8], 4) -> ([1, 1, 4], [4, 2, 2], 2)
    """
    if mask is None:
        mask = [1] * len(seq)
    
    if len(seq) != len(mask):
        raise ValueError("Sequence and mask must have same length")
    
    if len(seq) == 0:
        return [], [], 0
    
    # Direct algorithm: process from right to left, consuming slice_size
    dim_lengths = []
    dim_slices = []
    remaining = slice_size
    split_idx = 0
    split_flag = False
    
    for i in range(len(seq) - 1, -1, -1):
        x = seq[i]
        m = mask[i]
        
        if m:  # This dimension is sliceable
            if remaining >= x:
                # We can take the full dimension, but only if it divides evenly
                if remaining % x != 0:
                    raise ValueError(f"Cannot evenly divide sequence {seq} with slice size {slice_size}")
                slice_length = x
                remaining = remaining // x
            else:
                # We can only take part of the dimension
                if x % remaining != 0:
                    raise ValueError(f"Cannot evenly divide sequence {seq} with slice size {slice_size}")
                slice_length = remaining
                remaining = 1
        else:  # This dimension is not sliceable
            slice_length = x
            # remaining stays the same for non-sliceable dimensions
        
        num_slices = x // slice_length
        
        # Insert at beginning to build left-to-right result
        dim_lengths.insert(0, slice_length)
        dim_slices.insert(0, num_slices)
        
        # Check split condition - first index where slicing differs from original
        if m and slice_length != x and remaining == 1 and not split_flag:
            split_flag = True
            split_idx = i
    
    # Verify we can evenly divide
    if remaining != 1:
        raise ValueError(f"Cannot evenly divide sequence {seq} with slice size {slice_size}")
    
    return dim_lengths, dim_slices, split_idx


def get_y_unpacks_from_x_unpacks(y_lengths: List[int], x_unpacks: int) -> List[int]:
    """
    Get Y unpacks from X unpacks, equivalent to C++ get_y_unpacks_from_x_unpacks.
    
    This function calculates how to unpack Y dimensions based on X unpacking requirements.
    
    Args:
        y_lengths: List of Y dimension lengths
        x_unpacks: Number of X unpacks
        
    Returns:
        List of Y unpacks
        
    Example:
        get_y_unpacks_from_x_unpacks([2, 4], 2) -> [1, 2]
    """
    # Calculate total Y size by multiplying all Y lengths
    y_size = reduce_on_sequence(y_lengths, multiplies, 1)
    y_packs = x_unpacks
    
    # Verify that y_size is divisible by y_packs
    if y_size % y_packs != 0:
        raise ValueError(f"Y size {y_size} is not divisible by Y packs {y_packs}")
    
    # Calculate slice size per pack
    y_slice_size = y_size // y_packs
    
    # Use slice_sequence to determine unpacking pattern
    slice_lengths, slice_nums, split_idx = slice_sequence(y_lengths, y_slice_size)
    
    # The unpacks are the slice_nums (number of slices per dimension)
    return slice_nums


# Constants for common use
multiplies = Multiplies()
plus = Plus() 