"""
Python implementation of algorithm/static_encoding_pattern.hpp
"""
from enum import Enum, auto
from typing import TYPE_CHECKING

from .partition_simulation import get_warp_size, get_num_warps
from .tile_distribution import make_static_tile_distribution, make_tile_distribution_encoding

if TYPE_CHECKING:
    from .tile_distribution import TileDistribution


class TileDistributionPattern(Enum):
    """
    Enumeration describing static tile distribution patterns.
    """
    THREAD_RAKED = auto()
    WARP_RAKED = auto()
    BLOCK_RAKED = auto()


def make_2d_static_tile_distribution(
    pattern: TileDistributionPattern,
    y_per_tile: int,
    x_per_tile: int,
    vec_size: int,
    block_size: int,
    shuffled: bool = False
) -> "TileDistribution":
    """
    Creates a 2D static tile distribution with different load/store patterns.

    This is a Python implementation of the C++ TileDistributionEncodingPattern2D.

    We always assume that the Tile is YPerTile x XPerTile where the X dim (rightmost)
    is contiguous and we can do vector load on this dimension.

    Args:
        pattern: The distribution pattern (thread_raked, warp_raked, block_raked).
        y_per_tile: The tile size of outer/leftmost dimension.
        x_per_tile: The tile size of inner/rightmost dimension (contiguous).
        vec_size: The vector access size.
        block_size: Number of threads in a workgroup.
        shuffled: Whether to create a shuffled distribution (only for thread_raked).
    """
    warp_size = get_warp_size()
    num_warps = block_size // warp_size

    if x_per_tile % vec_size != 0:
        raise ValueError(f"x_per_tile ({x_per_tile}) must be a multiple of vec_size ({vec_size})")

    if pattern == TileDistributionPattern.THREAD_RAKED:
        return _make_thread_raked_distribution(
            y_per_tile, x_per_tile, vec_size, block_size, warp_size, num_warps, shuffled
        )
    elif pattern == TileDistributionPattern.WARP_RAKED:
        return _make_warp_raked_distribution(
            y_per_tile, x_per_tile, vec_size, block_size, warp_size, num_warps
        )
    elif pattern == TileDistributionPattern.BLOCK_RAKED:
        return _make_block_raked_distribution(
            y_per_tile, x_per_tile, vec_size, block_size, warp_size, num_warps
        )
    else:
        raise ValueError(f"Unknown tile distribution pattern: {pattern}")


def _make_thread_raked_distribution(y_per_tile, x_per_tile, vec_size, block_size, warp_size, num_warps, shuffled):
    """Creates a thread-raked distribution."""
    elements_per_thread = (x_per_tile * y_per_tile) / block_size
    x1 = min(vec_size, elements_per_thread)

    if x_per_tile % x1 != 0:
        if x_per_tile % vec_size == 0:
            x1 = vec_size
        else:
            candidates = [i for i in range(1, int(x1) + 1) if x_per_tile % i == 0]
            if not candidates:
                raise ValueError(f"Cannot find a suitable vector size for thread_raked with x_per_tile={x_per_tile}, vec_size={vec_size}")
            x1 = max(candidates)

    x0 = x_per_tile // x1

    if warp_size % x0 != 0:
        raise ValueError(f"warp_size ({warp_size}) must be a multiple of x0 ({x0}) for thread_raked pattern")
    y1 = warp_size // x0
    y0 = num_warps

    if y_per_tile % (y1 * y0) != 0:
        raise ValueError(f"y_per_tile ({y_per_tile}) must be multiple of y1*y0 ({y1*y0}) for thread_raked")
    y2 = y_per_tile // (y1 * y0)

    if not shuffled:
        encoding = make_tile_distribution_encoding(
            rs_lengths=[1],
            hs_lengthss=[[y0, y1, y2], [x0, x1]],
            ps_to_rhss_major=[[1], [1, 2]],
            ps_to_rhss_minor=[[0], [1, 0]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[2, 1]
        )
    else:
        encoding = make_tile_distribution_encoding(
            rs_lengths=[1],
            hs_lengthss=[[x0, x1], [y0, y1, y2]],
            ps_to_rhss_major=[[2], [2, 1]],
            ps_to_rhss_minor=[[0], [1, 0]],
            ys_to_rhs_major=[1, 2],
            ys_to_rhs_minor=[1, 2]
        )
    return make_static_tile_distribution(encoding)


def _make_warp_raked_distribution(y_per_tile, x_per_tile, vec_size, block_size, warp_size, num_warps):
    """Creates a warp-raked distribution."""
    elements_per_thread = (x_per_tile * y_per_tile) / block_size
    x1 = min(vec_size, elements_per_thread)
    x0 = x_per_tile // x1

    if warp_size % x0 != 0:
        raise ValueError(f"warp_size ({warp_size}) must be a multiple of x0 ({x0}) for warp_raked pattern")
    y2 = warp_size // x0
    y0 = num_warps

    if y_per_tile % (y2 * y0) != 0:
        raise ValueError(f"y_per_tile ({y_per_tile}) must be a multiple of y2*y0 ({y2*y0}) for warp_raked pattern")
    y1 = y_per_tile // (y2 * y0)

    encoding = make_tile_distribution_encoding(
        rs_lengths=[1],
        hs_lengthss=[[y0, y1, y2], [x0, x1]],
        ps_to_rhss_major=[[1], [1, 2]],
        ps_to_rhss_minor=[[0], [2, 0]],
        ys_to_rhs_major=[1, 2],
        ys_to_rhs_minor=[1, 1]
    )
    return make_static_tile_distribution(encoding)


def _make_block_raked_distribution(y_per_tile, x_per_tile, vec_size, block_size, warp_size, num_warps):
    """Creates a block-raked distribution."""
    elements_per_thread = (x_per_tile * y_per_tile) / block_size
    x1 = min(vec_size, elements_per_thread)
    x0 = x_per_tile // x1

    if warp_size % x0 != 0:
        raise ValueError(f"warp_size ({warp_size}) must be a multiple of x0 ({x0}) for block_raked pattern")
    y2 = warp_size // x0
    y1 = num_warps

    if y_per_tile % (y2 * y1) != 0:
        raise ValueError(f"y_per_tile ({y_per_tile}) must be a multiple of y2*y1 ({y2*y1}) for block_raked pattern")
    y0 = y_per_tile // (y2 * y1)

    encoding = make_tile_distribution_encoding(
        rs_lengths=[1],
        hs_lengthss=[[y0, y1, y2], [x0, x1]],
        ps_to_rhss_major=[[1], [1, 2]],
        ps_to_rhss_minor=[[1], [2, 0]],
        ys_to_rhs_major=[1, 2],
        ys_to_rhs_minor=[0, 1]
    )
    return make_static_tile_distribution(encoding) 