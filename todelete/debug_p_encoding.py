#!/usr/bin/env python3

from examples import get_default_variables
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import _make_adaptor_encoding_for_tile_distribution

def trace_p_dimension_encoding():
    """Trace exactly how P dimensions are encoded to find the bug."""
    
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
    
    print('=== Input Encoding ===')
    print(f'hs_lengthss: {encoding.hs_lengthss}')
    print(f'ps_to_rhss_major: {encoding.ps_to_rhss_major}')
    print(f'ps_to_rhss_minor: {encoding.ps_to_rhss_minor}')
    print()
    
    print('=== Expected P Dimension Mapping ===')
    print('P0 should map to: H0[1] (WarpPerBlock_M=2) and H1[1] (WarpPerBlock_N=2)')
    print('P1 should map to: H0[2] (ThreadPerWarp_M=8) and H1[2] (ThreadPerWarp_N=8)')
    print('Both should affect BOTH X0 and X1 dimensions')
    print()
    
    # Trace the encoding creation step by step
    print('=== Tracing _make_adaptor_encoding_for_tile_distribution ===')
    
    # Get encoding components
    rs_lengths = encoding.rs_lengths
    hs_lengthss = encoding.hs_lengthss
    ps_to_rhss_major = encoding.ps_to_rhss_major
    ps_to_rhss_minor = encoding.ps_to_rhss_minor
    ys_to_rhs_major = encoding.ys_to_rhs_major
    ys_to_rhs_minor = encoding.ys_to_rhs_minor
    
    # Constants
    MAX_NUM_DIM = 10
    ndim_x = len(hs_lengthss)
    
    print(f'ndim_x = {ndim_x}')
    
    # Initialize arrays for hidden dimensions
    rh_major_minor_to_hidden_ids = [[0] * MAX_NUM_DIM for _ in range(ndim_x + 1)]
    rh_major_minor_to_hidden_lengths = [[0] * MAX_NUM_DIM for _ in range(ndim_x + 1)]
    
    transforms = []
    hidden_dim_cnt = ndim_x
    
    print(f'Initial hidden_dim_cnt = {hidden_dim_cnt}')
    
    # Add replicate transform
    ndim_r_minor = len(rs_lengths)
    print(f'\\nAdding replicate transform for {ndim_r_minor} R dimensions')
    
    # Update hidden dimension mappings for R
    for i in range(ndim_r_minor):
        rh_major_minor_to_hidden_ids[0][i] = hidden_dim_cnt
        rh_major_minor_to_hidden_lengths[0][i] = rs_lengths[i]
        print(f'  R[{i}] -> hidden_id {hidden_dim_cnt}, length {rs_lengths[i]}')
        hidden_dim_cnt += 1
    
    # Add unmerge transforms for X dimensions
    print(f'\\nAdding unmerge transforms for X dimensions:')
    for idim_x in range(ndim_x):
        h_minor_lengths = hs_lengthss[idim_x]
        ndim_h_minor = len(h_minor_lengths)
        
        print(f'  X{idim_x}: unmerging into {ndim_h_minor} H dimensions {h_minor_lengths}')
        
        # Update hidden dimension mappings
        for i in range(ndim_h_minor):
            rh_major_minor_to_hidden_ids[idim_x + 1][i] = hidden_dim_cnt
            rh_major_minor_to_hidden_lengths[idim_x + 1][i] = h_minor_lengths[i]
            print(f'    H{idim_x}[{i}] -> hidden_id {hidden_dim_cnt}, length {h_minor_lengths[i]}')
            hidden_dim_cnt += 1
    
    print(f'\\nHidden dimension mapping after unmerge transforms:')
    for rh_major in range(ndim_x + 1):
        for rh_minor in range(MAX_NUM_DIM):
            if rh_major_minor_to_hidden_lengths[rh_major][rh_minor] > 0:
                hidden_id = rh_major_minor_to_hidden_ids[rh_major][rh_minor]
                length = rh_major_minor_to_hidden_lengths[rh_major][rh_minor]
                if rh_major == 0:
                    print(f'  R[{rh_minor}] -> hidden_id {hidden_id}, length {length}')
                else:
                    print(f'  H{rh_major-1}[{rh_minor}] -> hidden_id {hidden_id}, length {length}')
    
    # Add P dimension transforms - THIS IS WHERE THE BUG LIKELY IS
    ndim_p = len(ps_to_rhss_major)
    hidden_dim_id_ps = [0] * ndim_p
    
    print(f'\\n=== CRITICAL: Adding P dimension transforms ===')
    for i_dim_p in range(ndim_p):
        hidden_dim_id_p = hidden_dim_cnt
        hidden_dim_cnt += 1
        hidden_dim_id_ps[i_dim_p] = hidden_dim_id_p
        
        p2RHsMajor = ps_to_rhss_major[i_dim_p]
        p2RHsMinor = ps_to_rhss_minor[i_dim_p]
        
        print(f'\\nP{i_dim_p}:')
        print(f'  p2RHsMajor = {p2RHsMajor}')
        print(f'  p2RHsMinor = {p2RHsMinor}')
        print(f'  Will create hidden_id {hidden_dim_id_p}')
        
        ndim_low = len(p2RHsMajor)
        low_dims = [0] * ndim_low
        low_lengths = [0] * ndim_low
        
        print(f'  Merge transform will combine {ndim_low} dimensions:')
        for i in range(ndim_low):
            rh_major = p2RHsMajor[i]
            rh_minor = p2RHsMinor[i]
            low_dims[i] = rh_major_minor_to_hidden_ids[rh_major][rh_minor]
            low_lengths[i] = rh_major_minor_to_hidden_lengths[rh_major][rh_minor]
            print(f'    Input {i}: RH[{rh_major}][{rh_minor}] -> hidden_id {low_dims[i]}, length {low_lengths[i]}')
        
        print(f'  Merge transform: dims {low_dims} -> {hidden_dim_id_p}')
        print(f'  Merge lengths: {low_lengths}')
    
    print(f'\\n=== ANALYSIS OF THE PROBLEM ===')
    print('From the debug output, we can see:')
    print('- X0 unmerged to hidden IDs [2, 3, 4, 5] (H0[0] through H0[3])')
    print('- X1 unmerged to hidden IDs [6, 7, 8, 9] (H1[0] through H1[3])')
    print()
    print('P0 merge should combine:')
    print('  - H0[1] (WarpPerBlock_M) -> hidden_id 3')
    print('  - H1[1] (WarpPerBlock_N) -> hidden_id 7')
    print('P1 merge should combine:')
    print('  - H0[2] (ThreadPerWarp_M) -> hidden_id 4') 
    print('  - H1[2] (ThreadPerWarp_N) -> hidden_id 8')
    print()
    print('THE BUG: The merge transforms combine contributions from both H-spaces')
    print('but the resulting P dimensions are treated as separate coordinates,')
    print('not as contributions to BOTH X dimensions!')
    print()
    print('The issue is that merge transforms create NEW hidden dimensions')
    print('that become top-level P coordinates, but they dont CONTRIBUTE BACK')
    print('to both X0 and X1 during coordinate calculation.')

if __name__ == '__main__':
    trace_p_dimension_encoding() 