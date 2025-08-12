#!/usr/bin/env python3
"""
Test suite for Part 1: Foundation

This script validates that all our foundation concepts work correctly.
It runs both buffer basics and tensor view examples to ensure they're
ready for inclusion in the documentation.
"""

import sys
import os
import subprocess

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tile_distribution_documentation.validation_scripts.common import (
    print_section, print_step, show_result, validate_example,
    run_script_safely, check_imports
)


def test_buffer_basics():
    """Test that the buffer basics script runs successfully."""
    def run_buffer_script():
        script_path = os.path.join(os.path.dirname(__file__), 'buffer_view_basics.py')
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Buffer basics script failed with output:")
            print(result.stdout)
            print(result.stderr)
            return False
        
        # Check for success message
        return "completed successfully" in result.stdout
    
    return validate_example("Buffer basics script", run_buffer_script)


def test_tensor_view():
    """Test that the tensor view script runs successfully."""
    def run_tensor_script():
        script_path = os.path.join(os.path.dirname(__file__), 'tensor_view_basics.py')
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Tensor view script failed with output:")
            print(result.stdout)
            print(result.stderr)
            return False
        
        # Check for success message
        return "completed successfully" in result.stdout
    
    return validate_example("Tensor view script", run_tensor_script)


def test_core_concepts():
    """Test that we can import and use the core concepts."""
    def test_numpy_operations():
        import numpy as np
        
        # Test buffer creation
        buffer = np.array([1, 2, 3, 4], dtype=np.float32)
        
        # Test tensor view creation
        tensor = buffer.reshape(2, 2)
        
        # Test memory sharing
        tensor[0, 0] = 999
        
        return buffer[0] == 999 and tensor.shape == (2, 2)
    
    return validate_example("Core NumPy operations", test_numpy_operations)


def test_pytensor_import():
    """Test that we can import the pytensor library."""
    def import_pytensor():
        try:
            import pytensor
            return True
        except ImportError:
            return False
    
    return validate_example("PyTensor import", import_pytensor)


def main():
    """Run all Part 1 tests."""
    print("🧪 Testing Part 1: Foundation")
    print("Making sure all our foundation concepts work correctly...")
    
    # Test individual components
    tests = [
        ("PyTensor import", test_pytensor_import),
        ("Core concepts", test_core_concepts),
        ("Buffer basics script", test_buffer_basics),
        ("Tensor view script", test_tensor_view),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed_tests += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # Summary
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All Part 1 tests passed!")
        print("The foundation is solid - ready for Part 2!")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False


if __name__ == "__main__":
    run_script_safely("Part 1 Test Suite", main) 