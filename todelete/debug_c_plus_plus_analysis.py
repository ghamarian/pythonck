#!/usr/bin/env python3

from examples import get_default_variables
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.tensor_coordinate import MultiIndex

def debug_encoding_behavior():
    """Compare the current behavior vs expected behavior based on C++ logic."""
    
    # Create RMSNorm encoding
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
    
    print("=== C++ ENCODING ANALYSIS ===")
    print(f"H0 = {encoding.hs_lengthss[0]} (M dimension)")
    print(f"H1 = {encoding.hs_lengthss[1]} (N dimension)")
    print()
    
    print("=== P DIMENSION MAPPINGS ===")
    print("P0 maps to:")
    print(f"  - H0[1] = {encoding.hs_lengthss[0][1]} (WarpPerBlock_M)")
    print(f"  - H1[1] = {encoding.hs_lengthss[1][1]} (WarpPerBlock_N)")
    print("P1 maps to:")
    print(f"  - H0[2] = {encoding.hs_lengthss[0][2]} (ThreadPerWarp_M)")  
    print(f"  - H1[2] = {encoding.hs_lengthss[1][2]} (ThreadPerWarp_N)")
    print()
    
    print("=== EXPECTED BEHAVIOR PER C++ ===")
    print("Since P0 maps to both H0[1] and H1[1], changing P0 should affect BOTH X0 and X1")
    print("Since P1 maps to both H0[2] and H1[2], changing P1 should affect BOTH X0 and X1")
    print()
    
    # Test the actual behavior
    distribution = make_static_tile_distribution(encoding)
    adaptor = distribution.ps_ys_to_xs_adaptor
    
    print("=== ACTUAL BEHAVIOR ===")
    base_coord = MultiIndex(6, [0, 0, 0, 0, 0, 0])
    p0_coord = MultiIndex(6, [1, 0, 0, 0, 0, 0])  
    p1_coord = MultiIndex(6, [0, 1, 0, 0, 0, 0])
    both_coord = MultiIndex(6, [1, 1, 0, 0, 0, 0])
    
    base_x = adaptor.calculate_bottom_index(base_coord).to_list()
    p0_x = adaptor.calculate_bottom_index(p0_coord).to_list()
    p1_x = adaptor.calculate_bottom_index(p1_coord).to_list()
    both_x = adaptor.calculate_bottom_index(both_coord).to_list()
    
    print(f"Base  [0,0,0,0,0,0] -> X{base_x}")
    print(f"P0=1  [1,0,0,0,0,0] -> X{p0_x}")
    print(f"P1=1  [0,1,0,0,0,0] -> X{p1_x}")
    print(f"Both  [1,1,0,0,0,0] -> X{both_x}")
    print()
    
    # Calculate differences
    p0_diff = [p0_x[i] - base_x[i] for i in range(len(base_x))]
    p1_diff = [p1_x[i] - base_x[i] for i in range(len(base_x))]
    
    print("=== DIFFERENCE ANALYSIS ===")
    print(f"P0 effect: {p0_diff} (should be non-zero for both X0 and X1)")
    print(f"P1 effect: {p1_diff} (should be non-zero for both X0 and X1)")
    print()
    
    if p0_diff[0] == 0:
        print("❌ BUG: P0 does not affect X0 (row dimension)")
    else:
        print("✅ P0 correctly affects X0 (row dimension)")
        
    if p0_diff[1] == 0:
        print("❌ BUG: P0 does not affect X1 (column dimension)")
    else:
        print("✅ P0 correctly affects X1 (column dimension)")
        
    if p1_diff[0] == 0:
        print("❌ BUG: P1 does not affect X0 (row dimension)")
    else:
        print("✅ P1 correctly affects X0 (row dimension)")
        
    if p1_diff[1] == 0:
        print("❌ BUG: P1 does not affect X1 (column dimension)")
    else:
        print("✅ P1 correctly affects X1 (column dimension)")
        
    print()
    
    # Check if this matches the RMSNorm configuration intent
    print("=== RMSNORM INTENT CHECK ===")
    print("RMSNorm should distribute threads across BOTH row and column for optimal memory access")
    if p0_diff[0] != 0 and p1_diff[0] != 0:
        print("✅ Threads are distributed across rows (X0)")
    else:
        print("❌ Threads are NOT distributed across rows (X0)")
        
    if p0_diff[1] != 0 and p1_diff[1] != 0:
        print("✅ Threads are distributed across columns (X1)")
    else:
        print("❌ Threads are NOT distributed across columns (X1)")

if __name__ == "__main__":
    debug_encoding_behavior() 