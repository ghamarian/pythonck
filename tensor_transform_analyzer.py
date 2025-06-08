"""
Analyzer for tensor descriptor transformations.

This module provides functionality to analyze tensor descriptor transformations
and build a graph of transformations.
"""

from typing import List, Dict, Any, Set
import sympy as sp
from tensor_transform_parser import TensorTransformParser

class TensorTransformAnalyzer:
    """Analyzer for tensor descriptor transformations."""
    
    def __init__(self, descriptors: List[Dict[str, Any]], initial_dims: List[str]):
        """Initialize the analyzer with a list of descriptors and initial dimension names."""
        self.descriptors = descriptors
        self.initial_dims = initial_dims
        self.current_dims = initial_dims.copy()
        self.transformations = []
        
        # Find max dimension index across all descriptors
        max_dim = -1
        for desc in descriptors:
            if desc['type'] == 'naive':
                # For naive descriptors, use the number of dimensions
                max_dim = max(max_dim, len(desc['dimensions']) - 1)
            else:
                # For transform descriptors, check both lower and upper dimensions
                for dims in desc.get('lower_dimensions', []):
                    if isinstance(dims, list):
                        max_dim = max(max_dim, max(dims, default=-1))
                    else:
                        max_dim = max(max_dim, dims)
                for dims in desc.get('upper_dimensions', []):
                    if isinstance(dims, list):
                        max_dim = max(max_dim, max(dims, default=-1))
                    else:
                        max_dim = max(max_dim, dims)
        
        # Ensure we have enough dimension names
        while len(self.current_dims) <= max_dim:
            self.current_dims.append(f"dim_{len(self.current_dims)}")
    
    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze the descriptors and return a list of transformations."""
        for desc in self.descriptors:
            if desc['type'] == 'naive':
                # For naive descriptors, just store the dimensions
                self.transformations.append({
                    'type': 'naive',
                    'dimensions': desc['dimensions'],
                    'dim_names': self.current_dims[:len(desc['dimensions'])]
                })
            else:
                # For transform descriptors, process the transforms
                transforms = desc['transforms']
                lower_dims = desc['lower_dimensions']
                upper_dims = desc['upper_dimensions']
                
                # Map lower dimensions to current dimension names
                lower_dim_names = []
                for dims in lower_dims:
                    if isinstance(dims, list):
                        lower_dim_names.append([self.current_dims[idx] for idx in dims])
                    else:
                        lower_dim_names.append(self.current_dims[dims])
                
                # Process each transform
                for i, transform in enumerate(transforms):
                    if transform['type'] == 'pass_through':
                        # Pass-through transform (value is a size, not an index)
                        dim_size = transform['value']
                        self.transformations.append({
                            'type': 'pass_through',
                            'size': dim_size
                        })
                    elif transform['type'] == 'merge':
                        # Merge transform
                        input_dims = lower_dim_names[i]
                        # Handle both single and list upper_dims
                        if isinstance(upper_dims[i], list):
                            output_dim = [self.current_dims[idx] for idx in upper_dims[i]]
                        else:
                            output_dim = self.current_dims[upper_dims[i]]
                        self.transformations.append({
                            'type': 'merge',
                            'input_dims': input_dims,
                            'output_dim': output_dim,
                            'values': transform['values']
                        })
                    elif transform['type'] == 'xor':
                        # XOR transform
                        input_dims = lower_dim_names[i]
                        if isinstance(upper_dims[i], list):
                            output_dim = [self.current_dims[idx] for idx in upper_dims[i]]
                        else:
                            output_dim = self.current_dims[upper_dims[i]]
                        self.transformations.append({
                            'type': 'xor',
                            'input_dims': input_dims,
                            'output_dim': output_dim,
                            'values': transform['values']
                        })
                    elif transform['type'] == 'unmerge':
                        # Unmerge transform
                        if isinstance(lower_dims[i], list):
                            input_dim = [self.current_dims[idx] for idx in lower_dims[i]]
                        else:
                            input_dim = self.current_dims[lower_dims[i]]
                        if isinstance(upper_dims[i], list):
                            output_dims = [self.current_dims[idx] for idx in upper_dims[i]]
                        else:
                            output_dims = [self.current_dims[upper_dims[i]]]
                        self.transformations.append({
                            'type': 'unmerge',
                            'input_dim': input_dim,
                            'output_dims': output_dims,
                            'values': transform['values']
                        })
        
        return self.transformations

# Simple CLI/test function
def demo():
    from tensor_transform_parser import TensorTransformParser
    # Example: chain of two descriptors (can be replaced with any list)
    descriptor1 = """
    transform_tensor_descriptor(
        input_tensor,
        make_tuple(make_pass_through_transform(number<0>{}), make_merge_transform(make_tuple(number<1>{}, number<2>{}))),
        make_tuple(sequence<0>{}, sequence<1, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{})
    )
    """
    descriptor2 = """
    transform_tensor_descriptor(
        prev_tensor,
        make_tuple(make_merge_transform(make_tuple(number<0>{}, number<1>{}))),
        make_tuple(sequence<0, 1>{}),
        make_tuple(sequence<0>{})
    )
    """
    parser = TensorTransformParser()
    parsed1 = parser.parse_tensor_descriptor(descriptor1)
    parsed2 = parser.parse_tensor_descriptor(descriptor2)
    initial_dims = ["n", "c", "hi", "wi"]
    analyzer = TensorTransformAnalyzer([parsed1, parsed2], initial_dims)
    stages = analyzer.analyze()
    for stage in stages:
        print(stage)

if __name__ == "__main__":
    demo() 
 