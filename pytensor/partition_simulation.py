"""
Partition index simulation for testing distributed tensor operations.

This module provides utilities to simulate the hardware partition indices
that would be returned by get_lane_id() and get_warp_id() in the C++ implementation.
"""

from typing import List, Callable, Optional
from dataclasses import dataclass
import threading


@dataclass
class PartitionConfig:
    """
    Configuration for partition simulation.
    
    Attributes:
        warp_size: Number of lanes per warp (e.g., 64 for AMD, 32 for NVIDIA)
        num_warps: Number of warps in the block
        block_size: Total number of threads (warp_size * num_warps)
    """
    warp_size: int = 64  # Default to AMD warp size
    num_warps: int = 1
    
    @property
    def block_size(self) -> int:
        """Total number of threads in the block."""
        return self.warp_size * self.num_warps


class PartitionSimulator:
    """
    Simulates hardware partition indices for testing.
    
    This class provides thread-local storage for simulated warp_id and lane_id
    values, allowing tests to simulate different thread positions.
    """
    
    def __init__(self, config: PartitionConfig):
        """
        Initialize the partition simulator.
        
        Args:
            config: Partition configuration
        """
        self.config = config
        self._thread_local = threading.local()
    
    def set_thread_position(self, warp_id: int, lane_id: int):
        """
        Set the simulated position for the current thread.
        
        Args:
            warp_id: Simulated warp ID (0 to num_warps-1)
            lane_id: Simulated lane ID (0 to warp_size-1)
        """
        if warp_id < 0 or warp_id >= self.config.num_warps:
            raise ValueError(f"warp_id {warp_id} out of range [0, {self.config.num_warps})")
        if lane_id < 0 or lane_id >= self.config.warp_size:
            raise ValueError(f"lane_id {lane_id} out of range [0, {self.config.warp_size})")
        
        self._thread_local.warp_id = warp_id
        self._thread_local.lane_id = lane_id
    
    def set_thread_position_from_global_id(self, global_thread_id: int):
        """
        Set thread position from a global thread ID.
        
        Args:
            global_thread_id: Global thread ID (0 to block_size-1)
        """
        if global_thread_id < 0 or global_thread_id >= self.config.block_size:
            raise ValueError(f"global_thread_id {global_thread_id} out of range [0, {self.config.block_size})")
        
        warp_id = global_thread_id // self.config.warp_size
        lane_id = global_thread_id % self.config.warp_size
        self.set_thread_position(warp_id, lane_id)
    
    def get_warp_id(self) -> int:
        """Get the simulated warp ID for the current thread."""
        return getattr(self._thread_local, 'warp_id', 0)
    
    def get_lane_id(self) -> int:
        """Get the simulated lane ID for the current thread."""
        return getattr(self._thread_local, 'lane_id', 0)
    
    def get_partition_index_1d(self) -> List[int]:
        """
        Get 1D partition index (NDimP = 1).
        
        Returns:
            [lane_id] - matches C++ behavior for NDimP = 1
        """
        return [self.get_lane_id()]
    
    def get_partition_index_2d(self) -> List[int]:
        """
        Get 2D partition index (NDimP = 2).
        
        Returns:
            [warp_id, lane_id] - matches C++ behavior for NDimP = 2
        """
        return [self.get_warp_id(), self.get_lane_id()]
    
    def create_partition_index_func(self, ndim_p: int) -> Callable[[], List[int]]:
        """
        Create a partition index function for the given number of P dimensions.
        
        Args:
            ndim_p: Number of partition dimensions (1 or 2)
            
        Returns:
            Function that returns partition index for current thread
        """
        if ndim_p == 1:
            return self.get_partition_index_1d
        elif ndim_p == 2:
            return self.get_partition_index_2d
        else:
            raise ValueError(f"Unsupported ndim_p: {ndim_p}. Only 1 and 2 are supported.")


# Global simulator instance for convenience
_default_simulator = PartitionSimulator(PartitionConfig(warp_size=64, num_warps=4))


def set_global_thread_position(warp_id: int, lane_id: int):
    """
    Set the global simulated thread position.
    
    Args:
        warp_id: Simulated warp ID
        lane_id: Simulated lane ID
    """
    _default_simulator.set_thread_position(warp_id, lane_id)


def set_global_thread_position_from_id(global_thread_id: int):
    """
    Set the global simulated thread position from global thread ID.
    
    Args:
        global_thread_id: Global thread ID
    """
    _default_simulator.set_thread_position_from_global_id(global_thread_id)


def get_simulated_warp_id() -> int:
    """Get the global simulated warp ID."""
    return _default_simulator.get_warp_id()


def get_simulated_lane_id() -> int:
    """Get the global simulated lane ID."""
    return _default_simulator.get_lane_id()


def create_partition_index_func(ndim_p: int, 
                               warp_id: int = 0, 
                               lane_id: int = 0) -> Callable[[], List[int]]:
    """
    Create a partition index function for testing.
    
    Args:
        ndim_p: Number of partition dimensions
        warp_id: Simulated warp ID
        lane_id: Simulated lane ID
        
    Returns:
        Function that returns the specified partition index
    """
    if ndim_p == 1:
        return lambda: [lane_id]
    elif ndim_p == 2:
        return lambda: [warp_id, lane_id]
    else:
        raise ValueError(f"Unsupported ndim_p: {ndim_p}")


def create_multi_thread_partition_funcs(ndim_p: int, 
                                       num_threads: int,
                                       warp_size: int = 64) -> List[Callable[[], List[int]]]:
    """
    Create partition index functions for multiple simulated threads.
    
    Args:
        ndim_p: Number of partition dimensions
        num_threads: Number of threads to simulate
        warp_size: Size of each warp
        
    Returns:
        List of partition index functions, one per thread
    """
    funcs = []
    for thread_id in range(num_threads):
        warp_id = thread_id // warp_size
        lane_id = thread_id % warp_size
        funcs.append(create_partition_index_func(ndim_p, warp_id, lane_id))
    
    return funcs


# Context manager for temporary thread position
class ThreadPositionContext:
    """Context manager for temporarily setting thread position."""
    
    def __init__(self, warp_id: int, lane_id: int, simulator: Optional[PartitionSimulator] = None):
        """
        Initialize context manager.
        
        Args:
            warp_id: Warp ID to set
            lane_id: Lane ID to set
            simulator: Simulator to use (default: global simulator)
        """
        self.warp_id = warp_id
        self.lane_id = lane_id
        self.simulator = simulator or _default_simulator
        self.old_warp_id = None
        self.old_lane_id = None
    
    def __enter__(self):
        """Save current position and set new position."""
        self.old_warp_id = self.simulator.get_warp_id()
        self.old_lane_id = self.simulator.get_lane_id()
        self.simulator.set_thread_position(self.warp_id, self.lane_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore old position."""
        self.simulator.set_thread_position(self.old_warp_id, self.old_lane_id)


def with_thread_position(warp_id: int, lane_id: int):
    """
    Context manager for temporarily setting thread position.
    
    Args:
        warp_id: Warp ID to set
        lane_id: Lane ID to set
        
    Returns:
        Context manager
    """
    return ThreadPositionContext(warp_id, lane_id) 