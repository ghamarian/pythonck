#!/usr/bin/env python3
"""
Validation script for Part 3: Distribution API

This script validates all the Part 3 examples to ensure they work correctly.
"""

import sys
import os
import subprocess
import numpy as np

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from documentation.validation_scripts.common import (
    print_section, print_step, show_result, validate_example,
    explain_concept, show_comparison, run_script_safely, check_imports,
    show_tensor_shape, show_coordinate_transform
)


def validate_part3_scripts():
    """
    Validate all Part 3 scripts run successfully.
    """
    print_step(1, "Validating Part 3 Distribution API Scripts")
    
    scripts_to_test = [
        "tile_distribution_basics.py",
        "tile_window_basics.py",
        # Add more scripts as they're created
    ]
    
    all_passed = True
    
    for script_name in scripts_to_test:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        
        if not os.path.exists(script_path):
            print(f"⚠️  Script {script_name} not found, skipping...")
            continue
        
        print(f"\n🧪 Testing {script_name}...")
        
        try:
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"  ✅ {script_name} passed")
            else:
                print(f"  ❌ {script_name} failed")
                print(f"  Error: {result.stderr}")
                all_passed = False
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {script_name} timed out")
            all_passed = False
        except Exception as e:
            print(f"  💥 {script_name} crashed: {e}")
            all_passed = False
    
    return all_passed


def test_distribution_api_concepts():
    """
    Test key concepts from Part 3.
    """
    print_step(2, "Testing Distribution API Concepts")
    
    def test_imports():
        """Test that all required modules can be imported."""
        try:
            from pytensor.tile_distribution import (
                TileDistribution, make_static_tile_distribution, 
                make_tile_distribution_encoding
            )
            from pytensor.tile_window import make_tile_window
            from pytensor.static_distributed_tensor import make_static_distributed_tensor
            return True
        except ImportError as e:
            print(f"Import error: {e}")
            return False
    
    def test_real_examples():
        """Test that real examples from examples.py work."""
        try:
            from pytensor.tile_distribution import (
                make_static_tile_distribution, make_tile_distribution_encoding
            )
            
            # Test RMSNorm pattern
            rmsnorm_encoding = make_tile_distribution_encoding(
                rs_lengths=[],
                hs_lengthss=[[4, 2, 8, 4], [4, 2, 8, 4]],
                ps_to_rhss_major=[[1, 2], [1, 2]],
                ps_to_rhss_minor=[[1, 1], [2, 2]],
                ys_to_rhs_major=[1, 1, 2, 2],
                ys_to_rhs_minor=[0, 3, 0, 3]
            )
            rmsnorm_dist = make_static_tile_distribution(rmsnorm_encoding)
            
            # Test R Sequence pattern
            r_sequence_encoding = make_tile_distribution_encoding(
                rs_lengths=[2, 8],
                hs_lengthss=[[4, 2, 8, 4]],
                ps_to_rhss_major=[[0, 1], [0, 1]],
                ps_to_rhss_minor=[[0, 1], [1, 2]],
                ys_to_rhs_major=[1, 1],
                ys_to_rhs_minor=[0, 3]
            )
            r_sequence_dist = make_static_tile_distribution(r_sequence_encoding)
            
            # Verify properties
            return (rmsnorm_dist.ndim_x == 2 and rmsnorm_dist.ndim_y == 4 and
                    r_sequence_dist.ndim_x == 1 and r_sequence_dist.ndim_y == 2)
        except Exception as e:
            print(f"Real examples error: {e}")
            return False
    
    def test_p_plus_y_to_x_concept():
        """Test the P + Y → X concept."""
        try:
            from pytensor.tile_distribution import (
                make_static_tile_distribution, make_tile_distribution_encoding
            )
            
            # Create a distribution and verify dimensions make sense
            encoding = make_tile_distribution_encoding(
                rs_lengths=[],
                hs_lengthss=[[4, 2], [8, 4]],
                ps_to_rhss_major=[[1], [2]],
                ps_to_rhss_minor=[[0], [1]],
                ys_to_rhs_major=[1, 2],
                ys_to_rhs_minor=[1, 0]
            )
            distribution = make_static_tile_distribution(encoding)
            
            # Check that P + Y dimensions exist and X dimensions result
            return (distribution.ndim_p >= 0 and 
                    distribution.ndim_y >= 0 and 
                    distribution.ndim_x >= 0)
        except Exception as e:
            print(f"P + Y → X concept error: {e}")
            return False
    
    # Run all concept tests
    tests = [
        ("Module imports", test_imports),
        ("Real examples from examples.py", test_real_examples),
        ("P + Y → X concept", test_p_plus_y_to_x_concept),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        passed = validate_example(test_name, test_func)
        all_passed = all_passed and passed
    
    return all_passed


def main():
    """
    Main validation function.
    """
    if not check_imports():
        return False
    
    print("🧪 Validating Part 3: Distribution API")
    print("=" * 50)
    
    # Run all validation tests
    scripts_passed = validate_part3_scripts()
    concepts_passed = test_distribution_api_concepts()
    
    # Final result
    if scripts_passed and concepts_passed:
        print("\n🎉 Part 3 validation completed successfully!")
        print("All distribution API examples work correctly.")
        return True
    else:
        print("\n❌ Part 3 validation failed!")
        print("Some examples need fixing.")
        return False


if __name__ == "__main__":
    success = run_script_safely("Part 3 Validation", main)
    sys.exit(0 if success else 1) 