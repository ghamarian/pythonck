#!/usr/bin/env python3

from examples import get_default_variables
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.tensor_coordinate import MultiIndex

def test_warp_mapping():
    """Test how different warp positions should map to tensor regions."""
    
    variables = get_default_variables('Real-World Example (RMSNorm)')
    encoding = TileDistributionEncoding(
        rs_lengths=[],
        hs_lengthss=[
            [variables['S::Repeat_M'], variables['S::WarpPerBlock_M'], 
             variables['S::ThreadPerWarp_M'], variables['S::Vector_M']],
            [variables['S::Repeat_N'], variables['S::WarpPerBlock_N'], 
             variables['S::ThreadPerWarp_N'], variables['S::Vector_N']]
        ],
        ps_to_rhss_major=[[1, 2], [1, 2]],
        ps_to_rhss_minor=[[1, 1], [2, 2]],
        ys_to_rhs_major=[1, 1, 2, 2],
        ys_to_rhs_minor=[0, 3, 0, 3]
    )

    distribution = make_static_tile_distribution(encoding)
    adaptor = distribution.ps_ys_to_xs_adaptor

    print("=== Testing Warp Distribution Impact ===")
    print("RMSNorm Configuration:")
    print(f"  WarpPerBlock: {variables['S::WarpPerBlock_M']}x{variables['S::WarpPerBlock_N']} = {variables['S::WarpPerBlock_M'] * variables['S::WarpPerBlock_N']} warps")
    print(f"  ThreadPerWarp: {variables['S::ThreadPerWarp_M']}x{variables['S::ThreadPerWarp_N']} = {variables['S::ThreadPerWarp_M'] * variables['S::ThreadPerWarp_N']} threads/warp")
    print()

    # Test baseline
    baseline = adaptor.calculate_bottom_index(MultiIndex(6, [0,0,0,0,0,0])).to_list()
    print(f"Baseline P[0,0] -> X{baseline}")
    print()

    # Test different warp IDs (P0)
    print("Testing P0 (warp_id) changes:")
    for warp_id in range(4):
        result = adaptor.calculate_bottom_index(MultiIndex(6, [warp_id, 0, 0, 0, 0, 0])).to_list()
        print(f"  P[{warp_id},0] -> X{result}")
    print()

    # Test different lane IDs (P1) 
    print("Testing P1 (lane_id) changes:")
    for lane_id in range(8):
        result = adaptor.calculate_bottom_index(MultiIndex(6, [0, lane_id, 0, 0, 0, 0])).to_list()
        print(f"  P[0,{lane_id}] -> X{result}")
    print()

    # Test combinations to see the pattern
    print("Testing combinations to understand the pattern:")
    for warp_id in range(2):
        for lane_id in range(0, 8, 2):  # Every other lane
            result = adaptor.calculate_bottom_index(MultiIndex(6, [warp_id, lane_id, 0, 0, 0, 0])).to_list()
            print(f"  P[{warp_id},{lane_id}] -> X{result}")
    print()
    
    # Now let's understand WHY this happens by examining the encoding
    print("=== Analyzing the Encoding ===")
    print("P0 maps to positions:", encoding.ps_to_rhss_major[0], "with minor positions:", encoding.ps_to_rhss_minor[0])
    print("P1 maps to positions:", encoding.ps_to_rhss_major[1], "with minor positions:", encoding.ps_to_rhss_minor[1])
    print()
    print("H-space structure:")
    print("  H0 (M-dim):", variables['S::Repeat_M'], variables['S::WarpPerBlock_M'], variables['S::ThreadPerWarp_M'], variables['S::Vector_M'])
    print("  H1 (N-dim):", variables['S::Repeat_N'], variables['S::WarpPerBlock_N'], variables['S::ThreadPerWarp_N'], variables['S::Vector_N'])
    print()
    print("P0 maps to:")
    print("  H0[1] = WarpPerBlock_M =", variables['S::WarpPerBlock_M'])
    print("  H1[1] = WarpPerBlock_N =", variables['S::WarpPerBlock_N'])
    print("P1 maps to:")
    print("  H0[2] = ThreadPerWarp_M =", variables['S::ThreadPerWarp_M'])
    print("  H1[2] = ThreadPerWarp_N =", variables['S::ThreadPerWarp_N'])

if __name__ == "__main__":
    test_warp_mapping() 