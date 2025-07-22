"""
Python implementation of C++ functional utilities.

This module provides utilities equivalent to the C++ functional.hpp and
functional_with_tuple.hpp functionality.
"""

from typing import Callable, List, Any, Tuple, Optional, Union
import itertools


class Identity:
    """Identity function, equivalent to C++ identity."""
    
    def __call__(self, arg: Any) -> Any:
        return arg


def unpack(func: Callable, container: List[Any]) -> Any:
    """
    Unpack container elements as function arguments.
    
    Equivalent to C++ unpack function.
    
    Args:
        func: Function to call
        container: Container whose elements will be unpacked as arguments
        
    Returns:
        Result of calling func with unpacked arguments
        
    Example:
        unpack(lambda a, b, c: a + b + c, [1, 2, 3]) -> 6
    """
    return func(*container)


def unpack2(func: Callable, container1: List[Any], container2: List[Any]) -> Any:
    """
    Unpack two containers' elements as function arguments.
    
    Equivalent to C++ unpack2 function.
    
    Args:
        func: Function to call
        container1: First container to unpack
        container2: Second container to unpack
        
    Returns:
        Result of calling func with unpacked arguments from both containers
        
    Example:
        unpack2(lambda a, b, c, d: a + b + c + d, [1, 2], [3, 4]) -> 10
    """
    return func(*(container1 + container2))


class StaticFor:
    """
    Static for loop equivalent to C++ static_for.
    
    Provides compile-time-like iteration over a range of indices.
    """
    
    def __init__(self, begin: int, end: int, increment: int = 1):
        """
        Initialize static for loop.
        
        Args:
            begin: Starting index
            end: Ending index (exclusive)
            increment: Step size
        """
        if increment == 0:
            raise ValueError("Increment cannot be zero")
        if (end - begin) % increment != 0:
            raise ValueError("(end - begin) must be divisible by increment")
        if increment > 0 and begin > end:
            raise ValueError("For positive increment, begin must be <= end")
        if increment < 0 and begin < end:
            raise ValueError("For negative increment, begin must be >= end")
            
        self.begin = begin
        self.end = end
        self.increment = increment
    
    def __call__(self, func: Callable[[int], None]) -> None:
        """
        Execute the loop by calling func with each index.
        
        Args:
            func: Function to call with each index
        """
        current = self.begin
        while (self.increment > 0 and current < self.end) or (self.increment < 0 and current > self.end):
            func(current)
            current += self.increment


class StaticFord:
    """
    Multi-dimensional for loop equivalent to C++ static_ford.
    
    Loops over N-dimensional space with configurable ordering.
    """
    
    def __init__(self, lengths: List[int], orders: Optional[List[int]] = None):
        """
        Initialize multi-dimensional loop.
        
        Args:
            lengths: Length of each dimension
            orders: Order in which to iterate dimensions (default: [0, 1, 2, ...])
        """
        if not lengths:
            raise ValueError("Lengths cannot be empty")
        
        if orders is None:
            orders = list(range(len(lengths)))
        
        if len(orders) != len(lengths):
            raise ValueError("Orders and lengths must have same size")
        
        # Validate that orders is a valid permutation
        if sorted(orders) != list(range(len(lengths))):
            raise ValueError("Orders must be a valid permutation of [0, 1, ..., n-1]")
        
        self.lengths = lengths
        self.orders = orders
    
    def __call__(self, func: Callable[[List[int]], None]) -> None:
        """
        Execute the multi-dimensional loop.
        
        Args:
            func: Function to call with multi-dimensional index
        """
        # Reorder lengths according to orders for iteration
        ordered_lengths = [self.lengths[i] for i in self.orders]
        
        # Generate all combinations in the ordered space
        ranges = [range(length) for length in ordered_lengths]
        
        for ordered_indices in itertools.product(*ranges):
            # Convert back to unordered indices
            unordered_indices = [0] * len(self.lengths)
            for i, order_idx in enumerate(self.orders):
                unordered_indices[order_idx] = ordered_indices[i]
            
            func(unordered_indices)


class StaticUford:
    """
    Multi-dimensional for loop with unpacking, equivalent to C++ static_uford.
    
    Loops over N-dimensional space with unpacking - calls function with
    multiple indices at once based on unpacking configuration.
    """
    
    def __init__(self, lengths: List[int], unpacks: Optional[List[int]] = None, orders: Optional[List[int]] = None):
        """
        Initialize multi-dimensional loop with unpacking.
        
        Args:
            lengths: Length of each dimension
            unpacks: Number of elements to unpack per dimension (default: all 1s)
            orders: Order in which to iterate dimensions (default: [0, 1, 2, ...])
        """
        if not lengths:
            raise ValueError("Lengths cannot be empty")
        
        if unpacks is None:
            unpacks = [1] * len(lengths)
        
        if orders is None:
            orders = list(range(len(lengths)))
        
        if len(unpacks) != len(lengths):
            raise ValueError("Unpacks and lengths must have same size")
        
        if len(orders) != len(lengths):
            raise ValueError("Orders and lengths must have same size")
        
        # Validate that each length is divisible by its unpack
        for i, (length, unpack) in enumerate(zip(lengths, unpacks)):
            if length % unpack != 0:
                raise ValueError(f"Length {length} at dimension {i} must be divisible by unpack {unpack}")
        
        # Validate that orders is a valid permutation
        if sorted(orders) != list(range(len(lengths))):
            raise ValueError("Orders must be a valid permutation of [0, 1, ..., n-1]")
        
        self.lengths = lengths
        self.unpacks = unpacks
        self.orders = orders
    
    def get_num_of_access(self) -> int:
        """
        Get the total number of function calls that will be made.
        
        Returns:
            Number of function calls
        """
        total = 1
        for length, unpack in zip(self.lengths, self.unpacks):
            total *= length // unpack
        return total
    
    def __call__(self, func: Callable[..., None], access_index: Optional[int] = None) -> None:
        """
        Execute the multi-dimensional loop with unpacking.
        
        Args:
            func: Function to call with unpacked indices
            access_index: If specified, execute only this specific access
        """
        if access_index is not None:
            self._execute_single_access(func, access_index)
        else:
            self._execute_all_accesses(func)
    
    def _execute_all_accesses(self, func: Callable[..., None]) -> None:
        """Execute all accesses."""
        # Calculate how many groups we have in each dimension
        group_counts = [length // unpack for length, unpack in zip(self.lengths, self.unpacks)]
        
        # Reorder according to orders
        ordered_group_counts = [group_counts[i] for i in self.orders]
        ordered_unpacks = [self.unpacks[i] for i in self.orders]
        ordered_lengths = [self.lengths[i] for i in self.orders]
        
        # Generate all group combinations in ordered space
        group_ranges = [range(count) for count in ordered_group_counts]
        
        for ordered_group_indices in itertools.product(*group_ranges):
            # Generate the unpacked indices for this group combination
            unpacked_indices_list = []
            
            # Calculate total number of unpacked indices
            total_unpacked = 1
            for unpack in ordered_unpacks:
                total_unpacked *= unpack
            
            # Generate all combinations of unpacked indices within the groups
            unpack_ranges = []
            for i, (group_idx, unpack) in enumerate(zip(ordered_group_indices, ordered_unpacks)):
                start_idx = group_idx * unpack
                unpack_ranges.append(range(start_idx, start_idx + unpack))
            
            for unpacked_combo in itertools.product(*unpack_ranges):
                # Convert back to unordered indices
                unordered_indices = [0] * len(self.lengths)
                for i, order_idx in enumerate(self.orders):
                    unordered_indices[order_idx] = unpacked_combo[i]
                
                unpacked_indices_list.append(unordered_indices)
            
            # Call function with all unpacked indices for this group
            func(*unpacked_indices_list)
    
    def _execute_single_access(self, func: Callable[..., None], access_index: int) -> None:
        """Execute a single access by index."""
        total_accesses = self.get_num_of_access()
        if access_index >= total_accesses:
            raise ValueError(f"Access index {access_index} out of range [0, {total_accesses})")
        
        # Calculate which group combination this access corresponds to
        group_counts = [length // unpack for length, unpack in zip(self.lengths, self.unpacks)]
        
        # Convert linear access index to multi-dimensional group indices
        remaining = access_index
        group_indices = []
        
        for i in range(len(group_counts) - 1, -1, -1):
            count = group_counts[i]
            group_indices.insert(0, remaining % count)
            remaining //= count
        
        # Generate unpacked indices for this specific group combination
        unpacked_indices_list = []
        
        for dim_idx, (group_idx, unpack) in enumerate(zip(group_indices, self.unpacks)):
            start_idx = group_idx * unpack
            for unpack_offset in range(unpack):
                indices = [0] * len(self.lengths)
                indices[dim_idx] = start_idx + unpack_offset
                unpacked_indices_list.append(indices)
        
        # Call function with unpacked indices
        func(*unpacked_indices_list)


# Global instances for convenience
identity = Identity() 