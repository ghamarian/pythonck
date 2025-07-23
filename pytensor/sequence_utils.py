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
    Reverse slice sequence implementation that matches C++ algorithm using GCD.
    
    This processes the sequence from left to right (in the C++ "reverse" order),
    consuming the slice_size using GCD for sliceable dimensions.
    
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
    
    # C++ algorithm: process from left to right, using GCD
    dim_lengths = []
    dim_slices = []
    remaining_slice_sizes = [slice_size]
    split_idx = len(seq)  # Initialize to "no split" value
    split_flag = False
    
    for i, (x, m) in enumerate(zip(seq, mask)):
        current_slice_size = remaining_slice_sizes[-1]
        
        if m:  # This dimension is sliceable (mask = 1)
            # Use GCD like C++ implementation
            slice_length = gcd(x, current_slice_size)
        else:  # This dimension is not sliceable (mask = 0)
            slice_length = x
        
        num_slices = x // slice_length
        
        dim_lengths.append(slice_length)
        dim_slices.append(num_slices)
        
        # Update remaining slice sizes for next iteration
        if m:  # Only update for sliceable dimensions
            next_slice_size = current_slice_size // slice_length
        else:
            next_slice_size = current_slice_size
        remaining_slice_sizes.append(next_slice_size)
        
        # Check split condition - first index where slicing differs from original
        # AND remaining slice size becomes 1 (indicating we're done slicing)
        if m and slice_length != x and next_slice_size == 1 and not split_flag:
            split_flag = True
            split_idx = i
    
    # Verify we can evenly divide (final remaining should be 1)
    if remaining_slice_sizes[-1] != 1:
        raise ValueError(f"Cannot evenly divide sequence {seq} with slice size {slice_size}")
    
    return dim_lengths, dim_slices, split_idx


def slice_sequence(seq: List[int], slice_size: int, mask: List[int] = None) -> Tuple[List[int], List[int], int]:
    """
    Slice sequence implementing the true C++ pattern.
    
    C++ pattern: reverses inputs, calls reverse_slice_sequence, then reverses outputs.
    
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
    
    # Special case: if slice_size equals total size, no slicing occurs
    total_size = reduce_on_sequence(seq, multiplies, 1)
    if slice_size == total_size:
        return seq[:], [1] * len(seq), 0
    
    # C++ pattern: reverse inputs, call reverse_slice_sequence, then transform outputs
    reversed_seq = list(reversed(seq))
    reversed_mask = list(reversed(mask))
    
    # Call reverse_slice_sequence with reversed inputs
    r_lengths, r_nums, r_split_idx = reverse_slice_sequence(reversed_seq, slice_size, reversed_mask)
    
    # Transform outputs following C++ pattern:
    # 1. Reverse the lengths and nums
    # 2. Adjust split_idx as: len(seq) - r_split_idx - 1
    slice_lengths = list(reversed(r_lengths))
    slice_nums = list(reversed(r_nums))
    split_idx = len(seq) - r_split_idx - 1
    
    return slice_lengths, slice_nums, split_idx


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