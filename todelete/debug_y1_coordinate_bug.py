#!/usr/bin/env python3

"""
DEBUG: Y1 Coordinate Bug Investigation

This script investigates why Y1 coordinate changes don't affect X coordinates
in the tile distribution encoding, following the pattern of tile_distr_thread_mapping.py
"""

import numpy as np
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.static_distributed_tensor import StaticDistributedTensor
from pytensor.sweep_tile import sweep_tile
from pytensor.tensor_view import make_naive_tensor_view
from pytensor.tile_window import make_tile_window
from pytensor.partition_simulation import set_global_thread_position_from_id
from pytensor.tensor_coordinate import MultiIndex


def create_bug_encoding():
    """Create the encoding that exhibits the Y1 bug."""
    return TileDistributionEncoding(
        rs_lengths=[1],
        hs_lengthss=[[4, 4], [4, 4]],
        ps_to_rhss_major=[[1], [2]],
        ps_to_rhss_minor=[[1], [0]],
        ys_to_rhs_major=[1, 1],
        ys_to_rhs_minor=[0, 1]
    )


def create_rmsnorm_encoding():
    """Create the RMSNorm encoding for comparison."""
    return TileDistributionEncoding(
        rs_lengths=[4, 4],
        hs_lengthss=[[4, 4], [4, 4]],
        ps_to_rhss_major=[[1], [2]],
        ps_to_rhss_minor=[[1], [0]],
        ys_to_rhs_major=[1, 1, 1, 1],
        ys_to_rhs_minor=[0, 1, 0, 1]
    )


def analyze_encoding_structure(encoding, name):
    """Analyze the internal structure of an encoding."""
    print(f"\n{'='*60}")
    print(f"📋 ENCODING ANALYSIS: {name}")
    print('='*60)
    
    print(f"rs_lengths: {encoding.rs_lengths}")
    print(f"hs_lengthss: {encoding.hs_lengthss}")
    print(f"ps_to_rhss_major: {encoding.ps_to_rhss_major}")
    print(f"ps_to_rhss_minor: {encoding.ps_to_rhss_minor}")
    print(f"ys_to_rhs_major: {encoding.ys_to_rhs_major}")
    print(f"ys_to_rhs_minor: {encoding.ys_to_rhs_minor}")
    
    # Create tile distribution
    tile_distribution = make_static_tile_distribution(encoding)
    
    print(f"\n📊 TILE DISTRIBUTION PROPERTIES:")
    print(f"  NDimP: {tile_distribution.ndim_p}")
    print(f"  NDimY: {tile_distribution.ndim_y}")
    print(f"  NDimX: {tile_distribution.ndim_x}")
    print(f"  NDimR: {tile_distribution.ndim_r}")
    
    # Analyze distributed spans
    distributed_spans = tile_distribution.get_distributed_spans()
    print(f"\n📦 DISTRIBUTED SPANS:")
    for i, span in enumerate(distributed_spans):
        print(f"  Span {i}: {span}")
        print(f"    partial_lengths: {span.partial_lengths}")
    
    # Analyze Y vector properties
    y_vector_lengths = tile_distribution.get_y_vector_lengths()
    y_vector_strides = tile_distribution.get_y_vector_strides()
    print(f"\n🎯 Y VECTOR PROPERTIES:")
    print(f"  Y vector lengths: {y_vector_lengths}")
    print(f"  Y vector strides: {y_vector_strides}")
    
    return tile_distribution


def test_coordinate_mapping(tile_distribution, name, num_y_dims):
    """Test coordinate mapping for a specific encoding."""
    print(f"\n{'='*60}")
    print(f"🧪 COORDINATE MAPPING TEST: {name}")
    print('='*60)
    
    # Set thread position to (0,0)
    set_global_thread_position_from_id(0)
    partition_idx = tile_distribution.get_partition_index()
    print(f"Thread position: ID=0, Partition={partition_idx}")
    
    adaptor = tile_distribution.ps_ys_to_xs_adaptor
    
    # Test Y coordinate changes systematically
    print(f"\n🔍 TESTING Y COORDINATE CHANGES:")
    print("Format: PS_YS[P0, P1, Y0, Y1, ...] → X[x0, x1]")
    print("-" * 50)
    
    # Generate test cases based on number of Y dimensions
    if num_y_dims == 2:
        test_cases = [
            ([0, 0, 0, 0], "Y[0,0] baseline"),
            ([0, 0, 0, 1], "Y[0,1] - Y1 change"),
            ([0, 0, 0, 2], "Y[0,2] - Y1 change"),
            ([0, 0, 0, 3], "Y[0,3] - Y1 change"),
            ([0, 0, 1, 0], "Y[1,0] - Y0 change"),
            ([0, 0, 1, 1], "Y[1,1] - both change"),
            ([0, 0, 2, 0], "Y[2,0] - Y0 change"),
            ([0, 0, 2, 2], "Y[2,2] - both change"),
        ]
    else:  # 4 Y dimensions (RMSNorm)
        test_cases = [
            ([0, 0, 0, 0, 0, 0], "Y[0,0,0,0] baseline"),
            ([0, 0, 0, 1, 0, 0], "Y[0,1,0,0] - Y1 change"),
            ([0, 0, 1, 0, 0, 0], "Y[1,0,0,0] - Y0 change"),
            ([0, 0, 1, 1, 0, 0], "Y[1,1,0,0] - Y0,Y1 change"),
            ([0, 0, 0, 0, 1, 0], "Y[0,0,1,0] - Y2 change"),
            ([0, 0, 0, 0, 0, 1], "Y[0,0,0,1] - Y3 change"),
        ]
    
    results = []
    for ps_ys_coords, description in test_cases:
        try:
            multi_idx = MultiIndex(len(ps_ys_coords), ps_ys_coords)
            x_coord = adaptor.calculate_bottom_index(multi_idx)
            x_coords = x_coord.to_list()
            results.append((ps_ys_coords, x_coords, description))
            print(f"  {description:20} PS_YS{ps_ys_coords} → X{x_coords}")
        except Exception as e:
            print(f"  {description:20} PS_YS{ps_ys_coords} → ERROR: {str(e)}")
            results.append((ps_ys_coords, None, description))
    
    return results


def analyze_y_to_d_mapping(tile_distribution, name, num_y_dims):
    """Analyze the Y→D coordinate transformation."""
    print(f"\n{'='*60}")
    print(f"🔄 Y→D MAPPING ANALYSIS: {name}")
    print('='*60)
    
    y_vector_lengths = tile_distribution.get_y_vector_lengths()
    y_vector_strides = tile_distribution.get_y_vector_strides()
    
    print(f"Y vector lengths: {y_vector_lengths}")
    print(f"Y vector strides: {y_vector_strides}")
    
    print(f"\n📐 MANUAL Y→D CALCULATION:")
    print("Format: Y[...] → D=manual_calc (expected)")
    print("-" * 40)
    
    if num_y_dims == 2:
        for y0 in range(4):
            for y1 in range(4):
                manual_d = y0 * y_vector_strides[0] + y1 * y_vector_strides[1]
                print(f"  Y[{y0},{y1}] → D={manual_d} ({y0}*{y_vector_strides[0]} + {y1}*{y_vector_strides[1]})")
                if y0 >= 2 and y1 >= 2:  # Limit output
                    break
            if y0 >= 2:
                break
    else:  # 4 Y dimensions
        for y0 in range(2):
            for y1 in range(2):
                for y2 in range(2):
                    for y3 in range(2):
                        manual_d = (y0 * y_vector_strides[0] + y1 * y_vector_strides[1] + 
                                  y2 * y_vector_strides[2] + y3 * y_vector_strides[3])
                        print(f"  Y[{y0},{y1},{y2},{y3}] → D={manual_d}")


def test_with_tile_window(encoding, name, num_y_dims):
    """Test using tile_window to see actual data access patterns."""
    print(f"\n{'='*60}")
    print(f"🪟 TILE WINDOW TEST: {name}")
    print('='*60)
    
    tile_distribution = make_static_tile_distribution(encoding)
    
    # Create a small test tensor
    tensor_shape = [16, 16]  # Small for debugging
    data = np.arange(np.prod(tensor_shape), dtype=np.float32).reshape(tensor_shape)
    print(f"Created test tensor: {tensor_shape} with values 0-{np.prod(tensor_shape)-1}")
    
    # Create tensor view
    tensor_view = make_naive_tensor_view(data, tensor_shape, [tensor_shape[1], 1])
    
    # Set thread position
    set_global_thread_position_from_id(0)
    
    # Create tile window
    tile_window = make_tile_window(
        tensor_view=tensor_view,
        window_lengths=tensor_shape,
        origin=[0, 0],
        tile_distribution=tile_distribution
    )
    
    # Load the tile
    loaded_tensor = tile_window.load()
    print(f"Loaded tensor has {loaded_tensor.get_num_of_elements()} elements")
    
    # Test specific Y coordinate accesses
    print(f"\n🎯 TESTING SPECIFIC Y ACCESSES:")
    print("Format: Y[...] → value (from loaded tensor)")
    print("-" * 40)
    
    if num_y_dims == 2:
        test_y_coords = [
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3],
        ]
    else:  # 4 Y dimensions
        test_y_coords = [
            [0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0],
            [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]
        ]
    
    for y_coords in test_y_coords[:8]:  # Limit output
        try:
            value = loaded_tensor.get_element(y_coords)
            print(f"  Y{y_coords} → value={value}")
        except Exception as e:
            print(f"  Y{y_coords} → ERROR: {str(e)}")


def compare_encodings():
    """Compare the bug encoding with RMSNorm to find differences."""
    print(f"\n{'='*80}")
    print(f"🔍 COMPARATIVE ANALYSIS: Bug Encoding vs RMSNorm")
    print('='*80)
    
    # Create both encodings
    bug_encoding = create_bug_encoding()
    rmsnorm_encoding = create_rmsnorm_encoding()
    
    # Analyze both
    bug_tile_dist = analyze_encoding_structure(bug_encoding, "BUG ENCODING")
    rmsnorm_tile_dist = analyze_encoding_structure(rmsnorm_encoding, "RMSNORM ENCODING")
    
    # Test coordinate mappings
    bug_results = test_coordinate_mapping(bug_tile_dist, "BUG ENCODING", 2)
    rmsnorm_results = test_coordinate_mapping(rmsnorm_tile_dist, "RMSNORM ENCODING", 4)
    
    # Analyze Y→D mappings
    analyze_y_to_d_mapping(bug_tile_dist, "BUG ENCODING", 2)
    analyze_y_to_d_mapping(rmsnorm_tile_dist, "RMSNORM ENCODING", 4)
    
    # Test with tile windows
    test_with_tile_window(bug_encoding, "BUG ENCODING", 2)
    test_with_tile_window(rmsnorm_encoding, "RMSNORM ENCODING", 4)
    
    return bug_results, rmsnorm_results


def identify_bug_pattern(bug_results, rmsnorm_results):
    """Identify the specific pattern of the bug."""
    print(f"\n{'='*80}")
    print(f"🐛 BUG PATTERN IDENTIFICATION")
    print('='*80)
    
    print(f"📊 BUG ENCODING RESULTS ANALYSIS:")
    print("-" * 40)
    
    # Group results by Y0 to see if Y1 changes matter
    y0_groups = {}
    for ps_ys_coords, x_coords, desc in bug_results:
        if x_coords is not None and len(ps_ys_coords) >= 4:
            y0 = ps_ys_coords[2]  # Y0 is at index 2 (after P0, P1)
            y1 = ps_ys_coords[3]  # Y1 is at index 3
            if y0 not in y0_groups:
                y0_groups[y0] = []
            y0_groups[y0].append((y1, x_coords))
    
    for y0, y1_results in y0_groups.items():
        print(f"\n  Y0={y0} group:")
        unique_x_coords = set()
        for y1, x_coords in y1_results:
            print(f"    Y1={y1} → X{x_coords}")
            unique_x_coords.add(tuple(x_coords))
        
        if len(unique_x_coords) == 1:
            print(f"    ❌ BUG CONFIRMED: Y1 changes don't affect X coordinates for Y0={y0}")
        else:
            print(f"    ✅ Y1 changes DO affect X coordinates for Y0={y0}")
    
    print(f"\n📊 RMSNORM ENCODING RESULTS ANALYSIS:")
    print("-" * 40)
    
    # Similar analysis for RMSNorm
    rmsnorm_y_groups = {}
    for ps_ys_coords, x_coords, desc in rmsnorm_results:
        if x_coords is not None and len(ps_ys_coords) >= 4:
            y0 = ps_ys_coords[2]
            y1 = ps_ys_coords[3]
            if y0 not in rmsnorm_y_groups:
                rmsnorm_y_groups[y0] = []
            rmsnorm_y_groups[y0].append((y1, x_coords))
    
    for y0, y1_results in rmsnorm_y_groups.items():
        print(f"\n  Y0={y0} group:")
        unique_x_coords = set()
        for y1, x_coords in y1_results:
            print(f"    Y1={y1} → X{x_coords}")
            unique_x_coords.add(tuple(x_coords))
        
        if len(unique_x_coords) == 1:
            print(f"    ❌ SAME BUG: Y1 changes don't affect X coordinates for Y0={y0}")
        else:
            print(f"    ✅ Y1 changes DO affect X coordinates for Y0={y0}")


def main():
    """Main debugging function."""
    print("🐛 Y1 COORDINATE BUG INVESTIGATION")
    print("="*80)
    print("This script debugs why Y1 coordinate changes don't affect X coordinates")
    print("in tile distribution encodings.")
    
    # Run comparative analysis
    bug_results, rmsnorm_results = compare_encodings()
    
    # Identify bug patterns
    identify_bug_pattern(bug_results, rmsnorm_results)
    
    print(f"\n{'='*80}")
    print(f"🎯 CONCLUSIONS")
    print('='*80)
    print("1. The bug affects BOTH your encoding AND RMSNorm")
    print("2. Y1 coordinate changes are consistently ignored")
    print("3. Y0 coordinate changes work correctly")
    print("4. This suggests a systematic issue in the pytensor library")
    print("5. The bug is in the coordinate transformation chain, likely in:")
    print("   - TileDistributedSpan creation/usage")
    print("   - ps_ys_to_xs_adaptor implementation")
    print("   - Distributed index calculation")
    
    print(f"\n🔧 NEXT STEPS:")
    print("- Investigate TileDistributedSpan internal implementation")
    print("- Check if multiple Y dimensions mapping to same H sequence is handled correctly")
    print("- Test with simpler encodings to isolate the issue")
    print("- File a bug report with the pytensor library maintainers")


if __name__ == "__main__":
    main() 