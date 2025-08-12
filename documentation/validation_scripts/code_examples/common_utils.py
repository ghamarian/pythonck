#!/usr/bin/env python3
"""
Common utilities for tile distribution documentation code examples.

This module provides helper functions for demonstrating pytensor concepts
in a clean, focused way without educational fluff.
"""

import sys
import os
import numpy as np
from typing import List, Any, Tuple, Optional

# pytensor is now installed in development mode, no path manipulation needed

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_step(step_num: int, description: str):
    """Print a step header."""
    print(f"\n{step_num}️⃣ {description}")
    print("-" * 50)

def show_result(label: str, result: Any, note: str = ""):
    """Show a result with optional note."""
    print(f"  {label}: {result}")
    if note:
        print(f"    💡 {note}")

def validate_example(name: str, test_func) -> bool:
    """Validate an example and return success."""
    try:
        success = test_func()
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
        return success
    except Exception as e:
        print(f"  ❌ ERROR: {name} - {str(e)}")
        return False

def show_tensor_info(tensor_desc, name: str):
    """Show tensor descriptor information."""
    print(f"  {name}:")
    print(f"    Dimensions: {tensor_desc.get_num_of_dimension()}")
    print(f"    Lengths: {tensor_desc.get_lengths()}")
    print(f"    Element space size: {tensor_desc.get_element_space_size()}")
    print(f"    Is static: {tensor_desc.is_static()}")

def show_adaptor_info(adaptor, name: str):
    """Show tensor adaptor information."""
    print(f"  {name}:")
    print(f"    Top dimensions: {adaptor.get_num_of_top_dimension()}")
    print(f"    Bottom dimensions: {adaptor.get_num_of_bottom_dimension()}")
    print(f"    Hidden dimensions: {adaptor.get_num_of_hidden_dimension()}")
    print(f"    Is static: {adaptor.is_static()}")

def show_coordinate_info(coord, name: str):
    """Show coordinate information."""
    print(f"  {name}:")
    print(f"    Indices: {coord.to_list()}")
    print(f"    Size: {coord.size}")

def show_transform_info(transform, name: str):
    """Show transform information."""
    print(f"  {name}:")
    print(f"    Type: {type(transform).__name__}")
    
    # Handle transforms that might not have standard methods
    if hasattr(transform, 'get_num_of_upper_dimension'):
        print(f"    Upper dimensions: {transform.get_num_of_upper_dimension()}")
    else:
        print(f"    Upper dimensions: N/A")
        
    if hasattr(transform, 'get_num_of_lower_dimension'):
        print(f"    Lower dimensions: {transform.get_num_of_lower_dimension()}")
    else:
        print(f"    Lower dimensions: N/A")
        
    # Handle is_static method that might not exist
    if hasattr(transform, 'is_static'):
        print(f"    Is static: {transform.is_static()}")
    else:
        print(f"    Is static: N/A (method not available)")

def run_script_safely(script_func, script_name: str) -> bool:
    """Run a script function safely with error handling."""
    try:
        print_section(f"Running {script_name}")
        result = script_func()
        print(f"\n✅ {script_name} completed successfully")
        return True
    except Exception as e:
        print(f"\n❌ {script_name} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_imports():
    """Check if pytensor imports work correctly."""
    try:
        import pytensor
        print("✅ pytensor imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import pytensor: {str(e)}")
        return False

def demo_coordinate_calculation(transform, coords: List[List[int]], direction: str = "lower"):
    """Demonstrate coordinate calculations for a transform."""
    print(f"    {direction.title()} calculations:")
    for coord_list in coords:
        from pytensor.tensor_coordinate import MultiIndex
        
        if direction == "lower":
            upper_coord = MultiIndex(len(coord_list), coord_list)
            result = transform.calculate_lower_index(upper_coord)
            print(f"      {coord_list} → {result.to_list()}")
        else:
            lower_coord = MultiIndex(len(coord_list), coord_list)
            result = transform.calculate_upper_index(lower_coord)
            print(f"      {coord_list} → {result.to_list()}")

def demo_distribution_info(distribution, name: str):
    """Show tile distribution information."""
    print(f"  {name}:")
    print(f"    X dimensions: {distribution.ndim_x}")
    print(f"    Y dimensions: {distribution.ndim_y}")
    print(f"    P dimensions: {distribution.ndim_p}")
    print(f"    R dimensions: {distribution.ndim_r}")
    print(f"    X lengths: {distribution.get_lengths()}")
    print(f"    Is static: {distribution.is_static()}") 