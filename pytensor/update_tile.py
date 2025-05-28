"""
Python implementation of update_tile.hpp from Composable Kernels.

This module provides convenience functions for updating (accumulating) distributed tensors
into tile windows. This is similar to store_tile but performs accumulation instead of overwrite.
"""

from typing import Union
from .tile_window import TileWindowWithStaticDistribution, TileWindowWithStaticLengths, make_tile_window
from .static_distributed_tensor import StaticDistributedTensor


def update_tile(tile_window: Union[TileWindowWithStaticLengths, TileWindowWithStaticDistribution],
                distributed_tensor: StaticDistributedTensor,
                oob_conditional_check: bool = True) -> None:
    """
    Update (accumulate) a distributed tensor into a tile window.
    
    This is a convenience function that handles both static lengths and
    static distribution tile windows. The values are added to existing values
    rather than overwriting them.
    
    Args:
        tile_window: The tile window to update
        distributed_tensor: The distributed tensor to add
        oob_conditional_check: Whether to check out-of-bounds access
    """
    if isinstance(tile_window, TileWindowWithStaticLengths):
        # For static lengths window, we need to create a distributed version
        # using the distributed tensor's tile distribution
        distributed_window = make_tile_window(
            tensor_view=tile_window.bottom_tensor_view,
            window_lengths=tile_window.window_lengths,
            origin=tile_window.window_origin,
            tile_distribution=distributed_tensor.tile_distribution
        )
        distributed_window.update(distributed_tensor, oob_conditional_check)
    else:
        # Already a distributed window, update directly
        tile_window.update(distributed_tensor, oob_conditional_check)


def update_tile_raw(tile_window: Union[TileWindowWithStaticLengths, TileWindowWithStaticDistribution],
                    distributed_tensor: StaticDistributedTensor,
                    oob_conditional_check: bool = True,
                    pre_nop: bool = False) -> None:
    """
    Update (accumulate) a distributed tensor into a tile window using raw memory operations.
    
    This is similar to update_tile but uses raw memory operations which
    may be more efficient in some cases.
    
    Args:
        tile_window: The tile window to update
        distributed_tensor: The distributed tensor to add
        oob_conditional_check: Whether to check out-of-bounds access
        pre_nop: Whether to insert a no-op before first access
    """
    if isinstance(tile_window, TileWindowWithStaticLengths):
        # For static lengths window, we need to create a distributed version
        distributed_window = make_tile_window(
            tensor_view=tile_window.bottom_tensor_view,
            window_lengths=tile_window.window_lengths,
            origin=tile_window.window_origin,
            tile_distribution=distributed_tensor.tile_distribution
        )
        distributed_window.update_raw(distributed_tensor, oob_conditional_check, pre_nop)
    else:
        # Already a distributed window, update directly
        tile_window.update_raw(distributed_tensor, oob_conditional_check, pre_nop) 