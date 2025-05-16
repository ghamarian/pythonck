"""
Tile distribution module for the Composable Kernels visualizer.

This module implements the tile distribution functionality for visualizing
thread to data mappings in the Composable Kernels library.
"""

from typing import Dict, List, Any, Tuple, Union
import numpy as np

class TileDistribution:
    """
    Class representing a tile distribution for visualizing thread layouts.
    
    This class simulates the behavior of a real tile distribution by mapping
    threads to data elements based on the parsed encoding structure.
    """
    
    def __init__(self, encoding: Dict[str, Any], variables: Dict[str, int] = None):
        """
        Initialize a tile distribution from the parsed encoding.
        
        Args:
            encoding: Dictionary containing the parsed tile_distribution_encoding
            variables: Dictionary mapping variable names to their values
        """
        self.encoding = encoding
        self.variables = variables or {}
        
        # Extract key dimensions
        self.rs_lengths = self._resolve_values(encoding.get('RsLengths', []))
        self.hs_lengthss = [
            self._resolve_values(h_list) 
            for h_list in encoding.get('HsLengthss', [])
        ]
        self.ps_to_rhs_major = encoding.get('Ps2RHssMajor', [])
        self.ps_to_rhs_minor = encoding.get('Ps2RHssMinor', [])
        self.ys_to_rhs_major = encoding.get('Ys2RHsMajor', [])
        self.ys_to_rhs_minor = encoding.get('Ys2RHsMinor', [])
        
        # Calculate dimensions
        self.p_dims = self._calculate_p_dims()
        self.y_dims = self._calculate_y_dims()
        self.x_dims = self._calculate_x_dims()
        
        # Calculate tile shape and thread mapping
        self.tile_shape = self._calculate_tile_shape()
        self.thread_mapping = self._calculate_thread_mapping()
        
    def _resolve_values(self, values: List[Any]) -> List[int]:
        """
        Resolve variable names to their values.
        
        Args:
            values: List of values or variable names
            
        Returns:
            List of resolved integer values
        """
        resolved = []
        for val in values:
            if isinstance(val, str) and val in self.variables:
                resolved.append(self.variables[val])
            elif isinstance(val, (int, float)):
                resolved.append(int(val))
            else:
                # Default value for unresolved variables
                resolved.append(4)
        return resolved
    
    def _calculate_p_dims(self) -> List[int]:
        """Calculate the P dimensions based on the encoding."""
        # In a real implementation, this would analyze the actual
        # tile_distribution_encoding structure more precisely
        dims = []
        for majors, minors in zip(self.ps_to_rhs_major, self.ps_to_rhs_minor):
            if isinstance(majors, list) and isinstance(minors, list):
                dims.append(len(majors))
        return dims or [1]
    
    def _calculate_y_dims(self) -> List[int]:
        """Calculate the Y dimensions based on the encoding."""
        # In a real implementation, this would analyze the actual
        # tile_distribution_encoding structure more precisely
        if self.ys_to_rhs_major and self.ys_to_rhs_minor:
            return [len(self.ys_to_rhs_major)]
        return [1]
    
    def _calculate_x_dims(self) -> List[int]:
        """Calculate the X dimensions based on the encoding."""
        # In a real implementation, this would calculate the actual
        # tensor dimensions from the encoding
        dims = []
        
        # Add dimensions from R
        for r_len in self.rs_lengths:
            dims.append(r_len)
            
        # Add dimensions from H
        for h_list in self.hs_lengthss:
            if h_list:
                product = 1
                for val in h_list:
                    product *= val
                dims.append(product)
        
        return dims or [1, 1]
    
    def _calculate_tile_shape(self) -> List[int]:
        """Calculate the shape of the tile based on the encoding."""
        # In a simple case, use the X dimensions as the tile shape
        # In a real implementation, this would be more sophisticated
        return self.x_dims if len(self.x_dims) >= 2 else [4, 4]
    
    def _calculate_thread_mapping(self) -> Dict[str, int]:
        """
        Calculate the mapping of threads to tile elements.
        
        Returns:
            Dictionary mapping position strings (e.g., "0,1") to thread IDs
        """
        # In a real implementation, this would use the actual encoding
        # to determine how threads map to tile elements
        mapping = {}
        rows, cols = self.tile_shape[:2] if len(self.tile_shape) >= 2 else (4, 4)
        
        # Create a simple thread mapping based on the encoding
        # This is a placeholder implementation
        thread_idx = 0
        for i in range(rows):
            for j in range(cols):
                mapping[f"{i},{j}"] = thread_idx
                thread_idx += 1
        
        return mapping
    
    def get_occupancy(self) -> float:
        """
        Calculate the occupancy of the tile distribution.
        
        Returns:
            Occupancy as a value between 0 and 1
        """
        # In a real implementation, this would calculate the actual occupancy
        # based on the thread mapping and hardware constraints
        rows, cols = self.tile_shape[:2] if len(self.tile_shape) >= 2 else (4, 4)
        total_elements = rows * cols
        mapped_elements = len(self.thread_mapping)
        
        if total_elements == 0:
            return 0.0
            
        return mapped_elements / total_elements
    
    def get_utilization(self) -> float:
        """
        Calculate the utilization of the tile distribution.
        
        Returns:
            Utilization as a value between 0 and 1
        """
        # In a real implementation, this would calculate the actual utilization
        # based on thread divergence, memory access patterns, etc.
        occupancy = self.get_occupancy()
        
        # Simple model: utilization is occupancy * efficiency factor
        efficiency = 0.95  # Example efficiency factor
        return occupancy * efficiency
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data needed for visualization.
        
        Returns:
            Dictionary with visualization data
        """
        return {
            'tile_shape': self.tile_shape,
            'thread_mapping': self.thread_mapping,
            'dimensions': {
                'p_dims': self.p_dims,
                'y_dims': self.y_dims,
                'x_dims': self.x_dims
            },
            'occupancy': self.get_occupancy(),
            'utilization': self.get_utilization()
        }

def interleave_bits(x: int, y: int, n: int) -> List[int]:
    """Interleave the bits of x and y to create a z-order curve.
    
    Args:
        x: First coordinate
        y: Second coordinate
        n: Number of bits per coordinate
        
    Returns:
        Array of interleaved coordinates
    """
    # Implementation of interleaved bits (Z-order curve)
    # Used for mapping 2D thread ids to 1D memory indices
    result = []
    for i in range(n):
        bit_x = (x >> i) & 1
        bit_y = (y >> i) & 1
        result.append(bit_x | (bit_y << 1))
    return result

def generate_tile_layout(shape: List[int], interleaved: bool = False) -> np.ndarray:
    """Generate a layout for a tile with given shape.
    
    Args:
        shape: Tile shape
        interleaved: Whether to use Z-order interleaving (default: False)
        
    Returns:
        numpy array with tile layout indices
    """
    if len(shape) != 2:
        # Only supporting 2D for now
        if len(shape) == 1:
            shape = [shape[0], 1]
        else:
            shape = shape[:2]
    
    # Create tile with sequential indices
    tile = np.arange(np.prod(shape)).reshape(shape)
    
    if interleaved:
        # Apply Z-order curve (Morton order)
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Number of bits needed
                n_bits = max(shape[0].bit_length(), shape[1].bit_length())
                interleaved_bits = interleave_bits(i, j, n_bits)
                # Convert back to single index
                idx = 0
                for k, bit in enumerate(interleaved_bits):
                    idx |= bit << k
                tile[i, j] = idx
    
    return tile

# Example usage
if __name__ == "__main__":
    # Example usage
    example_encoding = {
        "RsLengths": [1],
        "HsLengthss": [
            ["Nr_y", "Nr_p", "Nw"],
            ["Kr_y", "Kr_p", "Kw", "Kv"]
        ],
        "Ps2RHssMajor": [[1, 2], [2, 1]],
        "Ps2RHssMinor": [[1, 1], [2, 2]],
        "Ys2RHsMajor": [1, 2, 2],
        "Ys2RHsMinor": [0, 0, 3]
    }
    
    example_variables = {
        "Nr_y": 4,
        "Nr_p": 8,
        "Nw": 2,
        "Kr_y": 8,
        "Kr_p": 4,
        "Kw": 2,
        "Kv": 4
    }
    
    tile_dist = TileDistribution(example_encoding, example_variables)
    print("Tile Shape:", tile_dist.tile_shape)
    print("Thread Mapping:", tile_dist.thread_mapping)
    print("Occupancy:", tile_dist.get_occupancy())
    print("Utilization:", tile_dist.get_utilization())
    
    viz_data = tile_dist.get_visualization_data()
    print("Visualization Data:", viz_data) 