"""
Python implementation of additional tensor_adaptor utilities from Composable Kernels.

This module provides additional utilities for creating and manipulating tensor adaptors
beyond what's in tensor_descriptor.py.
"""

from typing import List, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass

from .tensor_descriptor import (
    Transform, EmbedTransform, UnmergeTransform, OffsetTransform,
    TensorAdaptor, PassThroughTransform, PadTransform, MergeTransform,
    ReplicateTransform
)


def make_single_stage_tensor_adaptor(
    transforms: List[Transform],
    lower_dimension_old_top_idss: List[List[int]],
    upper_dimension_new_top_idss: List[List[int]]
) -> TensorAdaptor:
    """
    Create a single-stage tensor adaptor.
    
    This is a convenience function for creating tensor adaptors with a single
    stage of transformations.
    
    Args:
        transforms: List of transforms to apply
        lower_dimension_old_top_idss: Lower dimension mappings for each transform
        upper_dimension_new_top_idss: Upper dimension mappings for each transform
        
    Returns:
        TensorAdaptor instance
    """
    num_transforms = len(transforms)
    
    if len(lower_dimension_old_top_idss) != num_transforms:
        raise ValueError("Number of lower dimension mappings must match number of transforms")
    
    if len(upper_dimension_new_top_idss) != num_transforms:
        raise ValueError("Number of upper dimension mappings must match number of transforms")
    
    # Flatten all lower dimension IDs to get old top dimension count
    all_lower_ids = []
    for ids in lower_dimension_old_top_idss:
        all_lower_ids.extend(ids)
    
    ndim_old_top = len(set(all_lower_ids))
    
    # Lower dimension hidden IDs are the same as old top IDs
    lower_dimension_hidden_idss = lower_dimension_old_top_idss
    
    # Upper dimension hidden IDs are shifted by ndim_old_top
    upper_dimension_hidden_idss = []
    for ids in upper_dimension_new_top_idss:
        shifted_ids = [id + ndim_old_top for id in ids]
        upper_dimension_hidden_idss.append(shifted_ids)
    
    # Bottom dimension hidden IDs are [0, 1, ..., ndim_old_top-1]
    bottom_dimension_hidden_ids = list(range(ndim_old_top))
    
    # Top dimension hidden IDs start from ndim_old_top
    all_upper_ids = []
    for ids in upper_dimension_new_top_idss:
        all_upper_ids.extend(ids)
    
    ndim_new_top = len(set(all_upper_ids))
    top_dimension_hidden_ids = list(range(ndim_old_top, ndim_old_top + ndim_new_top))
    
    return TensorAdaptor(
        transforms=transforms,
        lower_dimension_hidden_idss=lower_dimension_hidden_idss,
        upper_dimension_hidden_idss=upper_dimension_hidden_idss,
        bottom_dimension_hidden_ids=bottom_dimension_hidden_ids,
        top_dimension_hidden_ids=top_dimension_hidden_ids
    )


def transform_tensor_adaptor(
    old_adaptor: TensorAdaptor,
    new_transforms: List[Transform],
    new_lower_dimension_old_top_idss: List[List[int]],
    new_upper_dimension_new_top_idss: List[List[int]]
) -> TensorAdaptor:
    """
    Transform an existing tensor adaptor by adding new transformations.
    
    Args:
        old_adaptor: Existing tensor adaptor
        new_transforms: New transforms to add
        new_lower_dimension_old_top_idss: Lower dimension mappings for new transforms
        new_upper_dimension_new_top_idss: Upper dimension mappings for new transforms
        
    Returns:
        New TensorAdaptor with combined transformations
    """
    # Convert old top IDs to hidden IDs
    old_top_hidden_ids = old_adaptor.top_dimension_hidden_ids
    
    # Convert lower dimension top IDs to hidden IDs
    lower_hidden_idss = []
    for top_ids in new_lower_dimension_old_top_idss:
        hidden_ids = [old_top_hidden_ids[id] for id in top_ids]
        lower_hidden_idss.append(hidden_ids)
    
    # Calculate new hidden dimension offset
    old_num_hidden = old_adaptor.get_num_of_hidden_dimension()
    
    # Create upper dimension hidden IDs
    upper_hidden_idss = []
    offset = old_num_hidden
    for i, transform in enumerate(new_transforms):
        num_upper = transform.get_num_of_upper_dimension()
        upper_ids = list(range(offset, offset + num_upper))
        upper_hidden_idss.append(upper_ids)
        offset += num_upper
    
    # Combine transforms
    all_transforms = old_adaptor.transforms + new_transforms
    
    # Combine dimension mappings
    all_lower_hidden_idss = old_adaptor.lower_dimension_hidden_idss + lower_hidden_idss
    all_upper_hidden_idss = old_adaptor.upper_dimension_hidden_idss + upper_hidden_idss
    
    # Bottom dimensions remain the same
    bottom_hidden_ids = old_adaptor.bottom_dimension_hidden_ids
    
    # Create new top dimension hidden IDs
    new_top_hidden_ids = []
    for i, ids in enumerate(new_upper_dimension_new_top_idss):
        for id in ids:
            if id not in new_top_hidden_ids:
                new_top_hidden_ids.append(upper_hidden_idss[i][ids.index(id)])
    
    return TensorAdaptor(
        transforms=all_transforms,
        lower_dimension_hidden_idss=all_lower_hidden_idss,
        upper_dimension_hidden_idss=all_upper_hidden_idss,
        bottom_dimension_hidden_ids=bottom_hidden_ids,
        top_dimension_hidden_ids=sorted(new_top_hidden_ids)
    )


def chain_tensor_adaptors(adaptor0: TensorAdaptor, adaptor1: TensorAdaptor) -> TensorAdaptor:
    """
    Chain two tensor adaptors together.
    
    The top dimensions of adaptor0 must match the bottom dimensions of adaptor1.
    
    Args:
        adaptor0: First adaptor (bottom)
        adaptor1: Second adaptor (top)
        
    Returns:
        Combined TensorAdaptor
    """
    if adaptor0.get_num_of_top_dimension() != adaptor1.get_num_of_bottom_dimension():
        raise ValueError("Top dimensions of first adaptor must match bottom dimensions of second")
    
    # Combine transforms
    all_transforms = adaptor0.transforms + adaptor1.transforms
    
    # Calculate hidden ID shift for adaptor1
    adaptor0_max_hidden = max(
        max([max(ids) if ids else -1 for ids in adaptor0.lower_dimension_hidden_idss], default=-1),
        max([max(ids) if ids else -1 for ids in adaptor0.upper_dimension_hidden_idss], default=-1)
    )
    
    # Find minimum non-bottom hidden ID in adaptor1
    adaptor1_min_hidden = float('inf')
    for i, transform in enumerate(adaptor1.transforms):
        for id in adaptor1.lower_dimension_hidden_idss[i]:
            if id not in adaptor1.bottom_dimension_hidden_ids:
                adaptor1_min_hidden = min(adaptor1_min_hidden, id)
        for id in adaptor1.upper_dimension_hidden_idss[i]:
            adaptor1_min_hidden = min(adaptor1_min_hidden, id)
    
    if adaptor1_min_hidden == float('inf'):
        adaptor1_min_hidden = 0
    
    hidden_id_shift = adaptor0_max_hidden + 1 - adaptor1_min_hidden
    
    # Shift and match adaptor1's lower dimension hidden IDs
    lower_hidden_idss_1 = []
    for i, ids in enumerate(adaptor1.lower_dimension_hidden_idss):
        new_ids = []
        for id in ids:
            # Check if this is a bottom dimension that needs matching
            if id in adaptor1.bottom_dimension_hidden_ids:
                # Find corresponding position and use adaptor0's top hidden ID
                bottom_pos = adaptor1.bottom_dimension_hidden_ids.index(id)
                new_ids.append(adaptor0.top_dimension_hidden_ids[bottom_pos])
            else:
                # Shift the hidden ID
                new_ids.append(id + hidden_id_shift)
        lower_hidden_idss_1.append(new_ids)
    
    # Shift adaptor1's upper dimension hidden IDs
    upper_hidden_idss_1 = []
    for ids in adaptor1.upper_dimension_hidden_idss:
        new_ids = [id + hidden_id_shift for id in ids]
        upper_hidden_idss_1.append(new_ids)
    
    # Combine dimension mappings
    all_lower_hidden_idss = adaptor0.lower_dimension_hidden_idss + lower_hidden_idss_1
    all_upper_hidden_idss = adaptor0.upper_dimension_hidden_idss + upper_hidden_idss_1
    
    # Bottom dimensions from adaptor0
    bottom_hidden_ids = adaptor0.bottom_dimension_hidden_ids
    
    # Top dimensions from adaptor1 (shifted)
    top_hidden_ids = [id + hidden_id_shift for id in adaptor1.top_dimension_hidden_ids]
    
    return TensorAdaptor(
        transforms=all_transforms,
        lower_dimension_hidden_idss=all_lower_hidden_idss,
        upper_dimension_hidden_idss=all_upper_hidden_idss,
        bottom_dimension_hidden_ids=bottom_hidden_ids,
        top_dimension_hidden_ids=top_hidden_ids
    )


def chain_tensor_adaptors_multi(*adaptors: TensorAdaptor) -> TensorAdaptor:
    """
    Chain multiple tensor adaptors together.
    
    Args:
        *adaptors: Variable number of tensor adaptors to chain
        
    Returns:
        Combined TensorAdaptor
    """
    if len(adaptors) == 0:
        raise ValueError("At least one adaptor required")
    
    if len(adaptors) == 1:
        return adaptors[0]
    
    result = adaptors[0]
    for adaptor in adaptors[1:]:
        result = chain_tensor_adaptors(result, adaptor)
    
    return result


# Factory functions for common adaptor patterns

def make_identity_adaptor(ndim: int) -> TensorAdaptor:
    """
    Create an identity adaptor that passes through all dimensions unchanged.
    
    Args:
        ndim: Number of dimensions
        
    Returns:
        Identity TensorAdaptor
    """
    transforms = [PassThroughTransform(1) for _ in range(ndim)]
    lower_idss = [[i] for i in range(ndim)]
    upper_idss = [[i] for i in range(ndim)]
    
    return TensorAdaptor(
        transforms=transforms,
        lower_dimension_hidden_idss=lower_idss,
        upper_dimension_hidden_idss=upper_idss,
        bottom_dimension_hidden_ids=list(range(ndim)),
        top_dimension_hidden_ids=list(range(ndim))
    )


def make_transpose_adaptor(ndim: int, permutation: List[int]) -> TensorAdaptor:
    """
    Create an adaptor that transposes dimensions according to permutation.
    
    Args:
        ndim: Number of dimensions
        permutation: New order of dimensions
        
    Returns:
        Transpose TensorAdaptor
    """
    if len(permutation) != ndim:
        raise ValueError("Permutation must have same length as number of dimensions")
    
    if sorted(permutation) != list(range(ndim)):
        raise ValueError("Permutation must be a valid permutation of dimensions")
    
    transforms = [PassThroughTransform(1) for _ in range(ndim)]
    lower_idss = [[i] for i in range(ndim)]
    upper_idss = [[i + ndim] for i in range(ndim)]
    
    bottom_ids = list(range(ndim))
    top_ids = [permutation[i] + ndim for i in range(ndim)]
    
    return TensorAdaptor(
        transforms=transforms,
        lower_dimension_hidden_idss=lower_idss,
        upper_dimension_hidden_idss=upper_idss,
        bottom_dimension_hidden_ids=bottom_ids,
        top_dimension_hidden_ids=top_ids
    ) 