"""
Python implementation of space_filling_curve.hpp from Composable Kernels.

This module provides space-filling curve functionality for memory access patterns.
"""

from typing import List, Tuple
import numpy as np

class SpaceFillingCurve:
    """
    Space-filling curve for memory access patterns.
    Matches C++ space_filling_curve template.
    """
    
    def __init__(self, tensor_lengths: List[int], 
                 dim_access_order: List[int],
                 scalars_per_access: List[int],
                 snake_curved: bool = True):
        """
        Initialize space-filling curve.
        
        Args:
            tensor_lengths: Length of each dimension
            dim_access_order: Order to traverse dimensions
            scalars_per_access: Number of scalars to access per dimension
            snake_curved: Whether to use snake-like pattern
        """
        self.tensor_lengths = tensor_lengths
        self.dim_access_order = dim_access_order
        self.scalars_per_access = scalars_per_access
        self.snake_curved = snake_curved
        
        # Calculate total size and validate
        self.tensor_size = 1
        for length in tensor_lengths:
            self.tensor_size *= length
        
        if self.tensor_size <= 0:
            raise ValueError("Space-filling curve requires non-empty tensor")
        
        # Calculate number of dimensions
        self.ndim = len(tensor_lengths)
        
        # Calculate scalar per vector (product of scalars_per_access)
        self.scalar_per_vector = 1
        for scalar in scalars_per_access:
            self.scalar_per_vector *= scalar
        
        # Calculate access lengths (tensor_lengths / scalars_per_access)
        self.access_lengths = []
        for i in range(self.ndim):
            # Calculate how many accesses are needed in this dimension
            # If we access scalars_per_access[i] elements at a time, 
            # we need ceil(tensor_lengths[i] / scalars_per_access[i]) accesses
            access_count = (tensor_lengths[i] + scalars_per_access[i] - 1) // scalars_per_access[i]
            self.access_lengths.append(access_count)
        
        # Reorder access lengths by dimension order
        self.ordered_access_lengths = [
            self.access_lengths[i] for i in dim_access_order
        ]
    
    def get_num_of_access(self) -> int:
        """
        Get total number of accesses needed.
        Matches C++ get_num_of_access().
        """
        num_access = 1
        for length in self.access_lengths:
            num_access *= length
        return num_access
    
    def get_index(self, i_access: int) -> List[int]:
        """
        Get indices for a given access number.
        Matches C++ get_index().
        
        Args:
            i_access: Access number
            
        Returns:
            List of indices for each dimension
        """
        # Calculate indices in ordered dimensions first
        ordered_indices = []
        remaining = i_access
        
        # Calculate indices in the order of ordered_access_lengths
        for i in range(len(self.ordered_access_lengths) - 1, -1, -1):
            length = self.ordered_access_lengths[i]
            idx = remaining % length
            ordered_indices.insert(0, idx)
            remaining //= length
        
        # Now map back to original dimension order
        indices = [0] * self.ndim
        for i, ordered_idx in enumerate(ordered_indices):
            original_dim = self.dim_access_order[i]
            # Apply the scalar multiplication AFTER getting the access index
            indices[original_dim] = ordered_idx * self.scalars_per_access[original_dim]
        
        return indices
    
    def get_step_between(self, start_access: int, end_access: int) -> List[int]:
        """
        Get step between two access indices.
        Matches C++ get_step_between().
        
        Args:
            start_access: Starting access number
            end_access: Ending access number
            
        Returns:
            List of step sizes for each dimension
        """
        start_indices = self.get_index(start_access)
        end_indices = self.get_index(end_access)
        
        return [
            end_indices[i] - start_indices[i]
            for i in range(self.ndim)
        ]
    
    def get_forward_step(self, i_access: int) -> List[int]:
        """
        Get step to next access position.
        Matches C++ get_forward_step().
        
        Args:
            i_access: Current access number
            
        Returns:
            List of step sizes for each dimension
        """
        return self.get_step_between(i_access, i_access + 1) 