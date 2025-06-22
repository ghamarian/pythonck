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
        self.source_code = None
        
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
    
    def set_tile_name(self, name: str):
        """
        Set a custom name for this tile distribution.
        
        Args:
            name: The name to display in the visualization
        """
        self.encoding['_tile_name'] = name
        
    def set_source_code(self, code: str):
        """
        Set the source code for this tile distribution to show in the visualization.
        
        Args:
            code: Source code string
        """
        self.source_code = code
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data needed for visualization.
        
        Returns:
            Dictionary with visualization data
        """
        # Calculate hierarchical tile structure
        hierarchical_structure = self.calculate_hierarchical_tile_structure()
        
        return {
            'tile_shape': self.tile_shape,
            'thread_mapping': self.thread_mapping,
            'dimensions': {
                'p_dims': self.p_dims,
                'y_dims': self.y_dims,
                'x_dims': self.x_dims
            },
            'occupancy': self.get_occupancy(),
            'utilization': self.get_utilization(),
            'hierarchical_structure': hierarchical_structure,
            'source_code': self.source_code
        }
        
    def calculate_hierarchical_tile_structure(self) -> Dict[str, Any]:
        """
        Calculate the hierarchical structure of tiles for visualization.
        
        This calculates the detailed structure of:
        - BlockSize (M, N) dimensions  
        - ThreadsPerWarp
        - WarpsPerBlock
        - VectorDimensions
        - Repeat factors
        
        Returns:
            Dictionary with hierarchical tile information
        """
        # Extract and resolve key dimensions from encoding
        hierarchical_info = {
            'BlockSize': [],            # Overall block dimensions
            'ThreadPerWarp': [],        # ThreadPerWarp dimensions (typically 16x4 in example)
            'WarpPerBlock': [],         # WarpsPerBlock dimensions (typically 4 in example) 
            'VectorDimensions': [],     # Vector dimensions (typically 8 in example)
            'Repeat': [],               # Repeat factors
            'ThreadBlocks': {},         # Thread layout in blocks with IDs
            'TileName': self.encoding.get('_tile_name', "Tile Distribution"),  # Name of the tile distribution
            'DimensionValues': []       # Values for dimensions (for code annotations)
        }
        
        # Get the sequences, P mappings, and Y mappings
        h_sequences = self.hs_lengthss if self.hs_lengthss else []
        p_major = self.ps_to_rhs_major
        p_minor = self.ps_to_rhs_minor
        y_major = self.ys_to_rhs_major
        y_minor = self.ys_to_rhs_minor
        
        # Ensure hierarchical_info has some default values if necessary data is missing
        if not h_sequences or not p_major or not p_minor:
            # Set sensible defaults if no proper data is available
            hierarchical_info['ThreadPerWarp'] = [16, 4]  # Default thread dimensions
            hierarchical_info['WarpPerBlock'] = [4]       # Default warps per block
            hierarchical_info['VectorDimensions'] = [8]   # Default vector size
            hierarchical_info['Repeat'] = [4]            # Default repeat factor
        else:
            # Correctly map the P and Y indices to H sequence values
            
            # 1. Process P1 for ThreadPerWarp (if exists)
            # P1 maps from highest index to lowest for ThreadPerWarp
            if len(p_major) > 1 and len(p_minor) > 1:
                p1_values = []
                for maj, min_idx in zip(p_major[1], p_minor[1]):
                    # Major index determines which H sequence (0 = R, 1 = H0, 2 = H1, etc.)
                    if maj == 0:  # R sequence
                        if min_idx < len(self.rs_lengths):
                            p1_values.append(self.rs_lengths[min_idx])
                    elif maj > 0 and maj-1 < len(h_sequences):  # H sequences
                        h_idx = maj - 1
                        if min_idx < len(h_sequences[h_idx]):
                            p1_values.append(h_sequences[h_idx][min_idx])
                
                # Set ThreadPerWarp from P1 values if available
                # Reverse the order since the rightmost (column) dimension changes fastest
                if p1_values:
                    # Ensure the column dimension is second in our representation
                    # This means rows x columns format
                    if len(p1_values) >= 2:
                        hierarchical_info['ThreadPerWarp'] = p1_values
                    elif len(p1_values) == 1:
                        # If only one dimension, create a row x column format with 1 column
                        hierarchical_info['ThreadPerWarp'] = [p1_values[0], 1]
                else:
                    hierarchical_info['ThreadPerWarp'] = [16, 4]  # Default if mapping failed
            else:
                hierarchical_info['ThreadPerWarp'] = [16, 4]  # Default if P1 doesn't exist
            
            # 2. Process P0 for WarpPerBlock (if exists)
            if len(p_major) > 0 and len(p_minor) > 0 and len(p_major[0]) > 0 and len(p_minor[0]) > 0:
                p0_values = []
                for maj, min_idx in zip(p_major[0], p_minor[0]):
                    if maj == 0:  # R sequence
                        if min_idx < len(self.rs_lengths):
                            p0_values.append(self.rs_lengths[min_idx])
                    elif maj > 0 and maj-1 < len(h_sequences):  # H sequences
                        h_idx = maj - 1
                        if min_idx < len(h_sequences[h_idx]):
                            p0_values.append(h_sequences[h_idx][min_idx])
                
                # Set WarpPerBlock from P0 values if available
                if p0_values:
                    hierarchical_info['WarpPerBlock'] = p0_values
                else:
                    hierarchical_info['WarpPerBlock'] = [4]  # Default if mapping failed
            else:
                hierarchical_info['WarpPerBlock'] = [4]  # Default if P0 doesn't exist
            
            # 3. Use the highest indexed Y for Vector dimensions
            # (Previously was using Y1 specifically)
            if y_major and y_minor and len(y_major) > 0 and len(y_minor) > 0:
                # Get the highest Y index (last Y dimension)
                highest_y_idx = len(y_major) - 1
                highest_y_major = y_major[highest_y_idx]
                highest_y_minor = y_minor[highest_y_idx]
                
                vector_dim = None
                if highest_y_major == 0:  # Highest Y maps to R
                    if highest_y_minor < len(self.rs_lengths):
                        vector_dim = self.rs_lengths[highest_y_minor]
                elif highest_y_major > 0 and highest_y_major-1 < len(h_sequences):  # Highest Y maps to H
                    h_idx = highest_y_major - 1
                    if highest_y_minor < len(h_sequences[h_idx]):
                        vector_dim = h_sequences[h_idx][highest_y_minor]
                
                if vector_dim is not None:
                    hierarchical_info['VectorDimensions'] = [vector_dim]
                else:
                    hierarchical_info['VectorDimensions'] = [8]  # Default if mapping is invalid
            else:
                # Fall back to using the last element of H0 if no Y dimensions exist
                if len(h_sequences) > 0 and len(h_sequences[0]) > 2:
                    hierarchical_info['VectorDimensions'] = [h_sequences[0][-1]]
                else:
                    hierarchical_info['VectorDimensions'] = [8]  # Default vector size
            
            # 4. Process Y0 for Repeat factor (if Y mappings exist)
            if len(y_major) > 0 and len(y_minor) > 0:
                y0_major = y_major[0]
                y0_minor = y_minor[0]
                
                if y0_major == 0:  # Y0 maps to R
                    if y0_minor < len(self.rs_lengths):
                        hierarchical_info['Repeat'] = [self.rs_lengths[y0_minor]]
                elif y0_major > 0 and y0_major-1 < len(h_sequences):  # Y0 maps to H
                    h_idx = y0_major - 1
                    if y0_minor < len(h_sequences[h_idx]):
                        hierarchical_info['Repeat'] = [h_sequences[h_idx][y0_minor]]
                else:
                    hierarchical_info['Repeat'] = [4]  # Default if mapping is invalid
            else:
                hierarchical_info['Repeat'] = [4]  # Default if Y mappings don't exist
            
            # Store dimension values for visualization annotations
            for seq in h_sequences:
                for val in seq:
                    if isinstance(val, int) or (isinstance(val, str) and val in self.variables):
                        resolved_val = self.variables.get(val, val) if isinstance(val, str) else val
                        hierarchical_info['DimensionValues'].append(resolved_val)
        
        # Calculate block size
        threads_per_warp = hierarchical_info.get('ThreadPerWarp', [16, 4])
        warps_per_block = hierarchical_info.get('WarpPerBlock', [4])
        
        if len(threads_per_warp) >= 2 and len(warps_per_block) >= 1:
            # Calculate block dimensions
            hierarchical_info['BlockSize'] = [
                threads_per_warp[0] * warps_per_block[0],
                threads_per_warp[1]
            ]
        
        # Generate thread block structure - columns change fastest (rightmost dimension)
        threads_per_warp_m = threads_per_warp[0] if len(threads_per_warp) > 0 else 16  # Rows
        threads_per_warp_n = threads_per_warp[1] if len(threads_per_warp) > 1 else 4   # Columns
        warps_per_block_m = warps_per_block[0] if len(warps_per_block) > 0 else 4
        
        # Create thread blocks with IDs
        thread_blocks = {}
        
        # Warp level organization
        for warp_idx in range(warps_per_block_m):
            warp_key = f"Warp{warp_idx}"
            thread_blocks[warp_key] = {}
            
            # Thread level within warp - column-major order (column changes fastest)
            for thread_m in range(threads_per_warp_m):  # Rows
                for thread_n in range(threads_per_warp_n):  # Columns - inner loop for fastest changing
                    # Calculate thread ID in column-major order
                    thread_id = thread_n + thread_m * threads_per_warp_n + warp_idx * threads_per_warp_m * threads_per_warp_n
                    thread_blocks[warp_key][f"T{thread_id}"] = {
                        "position": [thread_m, thread_n],  # [row, column]
                        "global_id": thread_id
                    }
        
        hierarchical_info['ThreadBlocks'] = thread_blocks
        
        # Get information about the overall calculation structure
        if 'VectorDimensions' in hierarchical_info and len(hierarchical_info['VectorDimensions']) > 0:
            hierarchical_info['VectorK'] = hierarchical_info['VectorDimensions'][0]
        
        return hierarchical_info

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