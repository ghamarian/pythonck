"""
Comprehensive test suite for move_tensor_adaptor_coordinate methods.

This module provides extensive testing to ensure the original move_tensor_adaptor_coordinate
and the efficient move_tensor_adaptor_coordinate_efficient produce identical results across
all scenarios and edge cases.
"""

import pytest
from typing import List, Tuple, Any
import itertools
import random

from pytensor.tensor_coordinate import (
    MultiIndex, TensorAdaptorCoordinate, move_tensor_adaptor_coordinate,
    move_tensor_adaptor_coordinate_efficient, make_tensor_adaptor_coordinate
)
from pytensor.tensor_descriptor import (
    PassThroughTransform, EmbedTransform, UnmergeTransform, PadTransform,
    MergeTransform, XorTransform, OffsetTransform, ReplicateTransform
)
from pytensor.tensor_adaptor import make_single_stage_tensor_adaptor


class ComprehensiveTestSuite:
    """Comprehensive test suite for coordinate movement comparison."""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_details = []
    
    def compare_methods(self, adaptor, initial_coords: List[int], diff_coords: List[int], test_name: str) -> bool:
        """Compare both methods on a single test case."""
        try:
            # Method 1: Original approach
            coord1 = make_tensor_adaptor_coordinate(adaptor, MultiIndex(len(initial_coords), initial_coords))
            initial_hidden1 = coord1.get_hidden_index().copy()
            diff_bottom1 = move_tensor_adaptor_coordinate(
                adaptor, coord1, MultiIndex(len(diff_coords), diff_coords)
            )
            final_hidden1 = coord1.get_hidden_index()
            
            # Method 2: Efficient approach  
            coord2 = make_tensor_adaptor_coordinate(adaptor, MultiIndex(len(initial_coords), initial_coords))
            initial_hidden2 = coord2.get_hidden_index().copy()
            diff_bottom2 = move_tensor_adaptor_coordinate_efficient(
                adaptor, coord2, MultiIndex(len(diff_coords), diff_coords)
            )
            final_hidden2 = coord2.get_hidden_index()
            
            # Compare results
            hidden_match = (final_hidden1.to_list() == final_hidden2.to_list())
            diff_match = (diff_bottom1.to_list() == diff_bottom2.to_list())
            
            success = hidden_match and diff_match
            
            self.test_details.append({
                'name': test_name,
                'initial': initial_coords,
                'diff': diff_coords,
                'success': success,
                'final_hidden1': final_hidden1.to_list(),
                'final_hidden2': final_hidden2.to_list(),
                'diff_bottom1': diff_bottom1.to_list(),
                'diff_bottom2': diff_bottom2.to_list(),
            })
            
            if success:
                self.passed_tests += 1
            else:
                self.failed_tests += 1
                print(f"❌ MISMATCH in {test_name}")
                print(f"   Initial: {initial_coords}, Diff: {diff_coords}")
                print(f"   Hidden1: {final_hidden1.to_list()}")
                print(f"   Hidden2: {final_hidden2.to_list()}")
                print(f"   DiffBottom1: {diff_bottom1.to_list()}")
                print(f"   DiffBottom2: {diff_bottom2.to_list()}")
            
            return success
            
        except Exception as e:
            self.failed_tests += 1
            print(f"❌ ERROR in {test_name}: {e}")
            return False
    
    def run_test_suite(self, adaptor, test_cases: List[Tuple[List[int], List[int]]], suite_name: str):
        """Run a suite of test cases."""
        print(f"\n=== {suite_name} ===")
        suite_passed = 0
        suite_total = len(test_cases)
        
        for i, (initial, diff) in enumerate(test_cases):
            test_name = f"{suite_name}_case_{i}"
            if self.compare_methods(adaptor, initial, diff, test_name):
                suite_passed += 1
        
        print(f"   Results: {suite_passed}/{suite_total} passed")
        return suite_passed == suite_total


def generate_single_transform_adaptors():
    """Generate adaptors with single transforms for systematic testing."""
    adaptors = []
    
    # PassThrough transforms
    for size in [1, 8, 16, 64]:
        transform = PassThroughTransform(size)
        adaptor = make_single_stage_tensor_adaptor([transform], [[0]], [[0]])
        adaptors.append((adaptor, f"PassThrough_{size}"))
    
    # Embed transforms
    embed_configs = [
        ([2, 4], [4, 1]),
        ([4, 8, 16], [128, 16, 1]),
        ([2, 2, 2, 2], [8, 4, 2, 1]),
    ]
    for lengths, strides in embed_configs:
        transform = EmbedTransform(lengths, strides)
        adaptor = make_single_stage_tensor_adaptor([transform], [[0]], [list(range(len(lengths)))])
        adaptors.append((adaptor, f"Embed_{lengths}_{strides}"))
    
    # Unmerge transforms
    unmerge_configs = [
        [2, 4],
        [4, 8, 16], 
        [2, 2, 2, 2],
        [8, 4],
    ]
    for lengths in unmerge_configs:
        transform = UnmergeTransform(lengths)
        adaptor = make_single_stage_tensor_adaptor([transform], [[0]], [list(range(len(lengths)))])
        adaptors.append((adaptor, f"Unmerge_{lengths}"))
    
    # Pad transforms
    pad_configs = [
        (8, 0, 2),    # Left pad
        (8, 2, 0),    # Right pad  
        (16, 3, 5),   # Both pads
        (4, 1, 1),    # Small pads
    ]
    for size, left, right in pad_configs:
        transform = PadTransform(size, left, right)
        adaptor = make_single_stage_tensor_adaptor([transform], [[0]], [[0]])
        adaptors.append((adaptor, f"Pad_{size}_{left}_{right}"))
    
    # Merge transforms
    merge_configs = [
        [2, 4],
        [4, 8],
        [2, 2, 2],
        [8, 4, 2],
    ]
    for lengths in merge_configs:
        transform = MergeTransform(lengths)
        adaptor = make_single_stage_tensor_adaptor([transform], [list(range(len(lengths)))], [[0]])
        adaptors.append((adaptor, f"Merge_{lengths}"))
    
    # Xor transforms (requires exactly 2 dimensions)
    xor_configs = [
        [4, 4],
        [8, 8], 
        [16, 8],
    ]
    for lengths in xor_configs:
        transform = XorTransform(lengths)
        adaptor = make_single_stage_tensor_adaptor([transform], [[0, 1]], [[0, 1]])
        adaptors.append((adaptor, f"Xor_{lengths}"))
    
    # Offset transforms
    offset_configs = [
        (8, 0),     # No offset
        (16, 5),    # Positive offset
        (32, 10),   # Larger offset
    ]
    for size, offset in offset_configs:
        transform = OffsetTransform(size, offset)
        adaptor = make_single_stage_tensor_adaptor([transform], [[0]], [[0]])
        adaptors.append((adaptor, f"Offset_{size}_{offset}"))
    
    # Replicate transforms
    replicate_configs = [
        [2, 4],
        [4, 8],
        [2, 2, 2],
    ]
    for lengths in replicate_configs:
        transform = ReplicateTransform(lengths)
        adaptor = make_single_stage_tensor_adaptor([transform], [[]], [list(range(len(lengths)))])
        adaptors.append((adaptor, f"Replicate_{lengths}"))
    
    return adaptors


def generate_comprehensive_test_cases(ndim_top: int, max_coord: int = 8, max_diff: int = 4):
    """Generate comprehensive test cases for given dimensionality."""
    cases = []
    
    # Zero movements
    for coord in range(min(max_coord, 4)):
        initial = [coord] * ndim_top
        diff = [0] * ndim_top
        cases.append((initial, diff))
    
    # Single dimension movements
    for dim in range(ndim_top):
        for coord_val in range(min(max_coord, 4)):
            for diff_val in [-2, -1, 1, 2]:
                initial = [coord_val] * ndim_top
                diff = [0] * ndim_top
                diff[dim] = diff_val
                cases.append((initial, diff))
    
    # Boundary cases
    boundary_coords = [0, min(max_coord-1, 3)]
    boundary_diffs = [-2, 0, 2]
    
    for coord_val in boundary_coords:
        for diff_val in boundary_diffs:
            initial = [coord_val] * ndim_top
            diff = [diff_val] * ndim_top
            cases.append((initial, diff))
    
    return cases


def test_all_single_transforms():
    """Test all single transform types comprehensively."""
    print("🧪 Testing All Single Transform Types")
    print("="*60)
    
    suite = ComprehensiveTestSuite()
    adaptors = generate_single_transform_adaptors()
    
    for adaptor, name in adaptors:
        ndim_top = adaptor.get_num_of_top_dimension()
        test_cases = generate_comprehensive_test_cases(ndim_top, max_coord=6, max_diff=3)
        suite.run_test_suite(adaptor, test_cases, name)
    
    print(f"\n📊 Overall Results: {suite.passed_tests}/{suite.passed_tests + suite.failed_tests} tests passed")
    return suite.failed_tests == 0


def test_real_world_patterns():
    """Test patterns that occur in real applications like tile_distr_thread_mapping.py."""
    print("\n🧪 Testing Real-World Patterns")
    print("="*60)
    
    suite = ComprehensiveTestSuite()
    
    # Tile distribution-like pattern: P+Y -> X mapping
    unmerge1 = UnmergeTransform([2, 2, 2, 4])  # repeat, warp, thread, vector
    unmerge2 = UnmergeTransform([2, 2, 2, 4])  # repeat, warp, thread, vector for second tensor
    
    adaptor = make_single_stage_tensor_adaptor(
        [unmerge1, unmerge2],
        [[0], [1]], 
        [[0, 1, 2, 3], [4, 5, 6, 7]]
    )
    
    # Patterns typical in GPU thread access
    tile_patterns = [
        # Sequential access in innermost dimension
        ([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]),
        ([0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]),
        
        # Warp-level access
        ([0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]),
        
        # Block-level access  
        ([0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
        
        # Cross-tensor access
        ([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0]),
        
        # Larger jumps
        ([0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]),
        ([1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, 0, 0, 0, 0]),
    ]
    
    suite.run_test_suite(adaptor, tile_patterns, "TileDistribution_GPU")
    
    print(f"\n📊 Real-World Results: {suite.passed_tests}/{suite.passed_tests + suite.failed_tests} tests passed")
    return suite.failed_tests == 0


def test_stress_random():
    """Stress test with random coordinates and differences."""
    print("\n🧪 Stress Testing with Random Cases")
    print("="*60)
    
    suite = ComprehensiveTestSuite()
    
    # Test various adaptor types with random inputs
    adaptors_to_stress = [
        (make_single_stage_tensor_adaptor([UnmergeTransform([4, 8])], [[0]], [[0, 1]]), "Unmerge_4_8"),
        (make_single_stage_tensor_adaptor([EmbedTransform([4, 8], [8, 1])], [[0]], [[0, 1]]), "Embed_4_8"),
        (make_single_stage_tensor_adaptor([MergeTransform([4, 8])], [[0, 1]], [[0]]), "Merge_4_8"),
    ]
    
    random.seed(12345)
    for adaptor, name in adaptors_to_stress:
        ndim_top = adaptor.get_num_of_top_dimension()
        
        # Generate many random test cases
        random_cases = []
        for _ in range(30):
            initial = [random.randint(0, 7) for _ in range(ndim_top)]
            diff = [random.randint(-3, 3) for _ in range(ndim_top)]
            random_cases.append((initial, diff))
        
        suite.run_test_suite(adaptor, random_cases, f"Stress_{name}")
    
    print(f"\n📊 Stress Test Results: {suite.passed_tests}/{suite.passed_tests + suite.failed_tests} tests passed")
    return suite.failed_tests == 0


def run_comprehensive_comparison_tests():
    """Run all comprehensive comparison tests."""
    print("🚀 COMPREHENSIVE MOVE COORDINATE COMPARISON TESTS")
    print("="*80)
    
    tests = [
        test_all_single_transforms,
        test_real_world_patterns,
        test_stress_random,
    ]
    
    all_passed = True
    for test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("✅ Both move_tensor_adaptor_coordinate methods produce identical results")
        print("✅ The efficient implementation is ready for production use")
    else:
        print("❌ Some comprehensive tests failed")
        print("🔍 Need to investigate differences between implementations")
    
    return all_passed


if __name__ == "__main__":
    run_comprehensive_comparison_tests() 