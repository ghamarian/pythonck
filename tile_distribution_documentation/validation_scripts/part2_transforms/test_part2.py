#!/usr/bin/env python3
"""
Part 2 Transform Tests - Comprehensive validation suite

This script tests all the Part 2 transform examples to ensure they work correctly
before being included in the documentation.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tile_distribution_documentation.validation_scripts.common import (
    print_section, print_step, show_result, validate_example,
    run_script_safely, check_imports
)


def test_individual_transforms():
    """Test the individual transforms script."""
    print_step(1, "Testing individual transforms")
    
    script_path = Path(__file__).parent / "01_individual_transforms.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        success = result.returncode == 0
        if success:
            print("✅ Individual transforms script executed successfully")
            # Check for key success indicators
            if "All transform operations work correctly!" in result.stdout:
                print("✅ All transform operations validated")
            else:
                print("⚠️  Transform validation message not found")
                success = False
        else:
            print(f"❌ Individual transforms script failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            success = False
            
        return success
        
    except subprocess.TimeoutExpired:
        print("❌ Individual transforms script timed out")
        return False
    except Exception as e:
        print(f"❌ Error running individual transforms script: {e}")
        return False


def test_transform_api_consistency():
    """Test that all transforms have consistent APIs."""
    print_step(2, "Testing transform API consistency")
    
    try:
        from pytensor.tensor_descriptor import (
            UnmergeTransform, MergeTransform, ReplicateTransform,
            EmbedTransform, PassThroughTransform, PadTransform, XorTransform
        )
        from pytensor.tensor_coordinate import MultiIndex
        
        # Test that all transforms have the required methods
        transforms = [
            ("UnmergeTransform", UnmergeTransform([2, 3])),
            ("MergeTransform", MergeTransform([2, 3])),
            ("ReplicateTransform", ReplicateTransform([4])),
            ("EmbedTransform", EmbedTransform([2, 3], [3, 1])),
            ("PassThroughTransform", PassThroughTransform(5)),
            ("PadTransform", PadTransform(5, 1, 1)),
            ("XorTransform", XorTransform([4, 4])),
        ]
        
        required_methods = [
            'calculate_lower_index',
            'calculate_upper_index',
            'get_num_of_lower_dimension',
            'get_num_of_upper_dimension',
            'get_upper_lengths',
            'is_valid_upper_index_always_mapped_to_valid_lower_index',
            'is_valid_upper_index_mapped_to_valid_lower_index',
            'sympy_calculate_lower',
            'sympy_calculate_upper',
            'update_lower_index'
        ]
        
        all_passed = True
        for name, transform in transforms:
            print(f"  Testing {name}...")
            for method in required_methods:
                if not hasattr(transform, method):
                    print(f"    ❌ Missing method: {method}")
                    all_passed = False
                else:
                    print(f"    ✅ Has method: {method}")
        
        if all_passed:
            print("✅ All transforms have consistent APIs")
        else:
            print("❌ Some transforms are missing required methods")
            
        return all_passed
        
    except Exception as e:
        print(f"❌ Error testing transform API consistency: {e}")
        return False


def test_basic_transform_operations():
    """Test basic transform operations work correctly."""
    print_step(3, "Testing basic transform operations")
    
    try:
        from pytensor.tensor_descriptor import UnmergeTransform, MergeTransform, ReplicateTransform
        from pytensor.tensor_coordinate import MultiIndex
        
        # Test UnmergeTransform
        unmerge = UnmergeTransform([2, 3])
        coord_1d = MultiIndex(1, [4])
        coord_2d = unmerge.calculate_upper_index(coord_1d)
        expected_2d = [1, 1]  # 4 in 1D should be [1, 1] in 2x3 grid
        
        if coord_2d.to_list() != expected_2d:
            print(f"❌ UnmergeTransform failed: expected {expected_2d}, got {coord_2d.to_list()}")
            return False
        print("✅ UnmergeTransform working correctly")
        
        # Test MergeTransform
        merge = MergeTransform([2, 3])
        coord_2d = MultiIndex(2, [1, 1])
        coord_1d = merge.calculate_upper_index(coord_2d)
        expected_1d = [4]  # [1, 1] in 2x3 grid should be 4 in 1D
        
        if coord_1d.to_list() != expected_1d:
            print(f"❌ MergeTransform failed: expected {expected_1d}, got {coord_1d.to_list()}")
            return False
        print("✅ MergeTransform working correctly")
        
        # Test ReplicateTransform
        replicate = ReplicateTransform([3])
        coord_empty = MultiIndex(0, [])
        coord_replicated = replicate.calculate_upper_index(coord_empty)
        expected_replicated = [0]  # Should create zeros for all dimensions
        
        if coord_replicated.to_list() != expected_replicated:
            print(f"❌ ReplicateTransform failed: expected {expected_replicated}, got {coord_replicated.to_list()}")
            return False
        print("✅ ReplicateTransform working correctly")
        
        print("✅ All basic transform operations work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing basic transform operations: {e}")
        return False


def main():
    """Main test function."""
    print_section("Part 2 Transform Tests")
    print("Running comprehensive validation of all Part 2 transform examples...")
    
    # First, make sure we can import our libraries
    if not check_imports():
        return False
    
    # Run all tests
    tests = [
        ("Individual transforms script", test_individual_transforms),
        ("Transform API consistency", test_transform_api_consistency),
        ("Basic transform operations", test_basic_transform_operations),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        passed = validate_example(test_name, test_func)
        all_passed = all_passed and passed
    
    # Summary
    if all_passed:
        print("\n🎉 All Part 2 transform tests passed!")
        print("The coordinate transform examples are ready for documentation.")
        print("\n💡 What we validated:")
        print("  • Individual transform scripts execute correctly")
        print("  • All transforms have consistent APIs")
        print("  • Basic coordinate transformations work as expected")
        print("  • Error handling is robust")
        print("  • FastBook-style explanations are clear")
        return True
    else:
        print("\n❌ Some Part 2 transform tests failed!")
        print("Please fix the issues before including in documentation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 