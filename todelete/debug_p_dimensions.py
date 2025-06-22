#!/usr/bin/env python3

from examples import get_default_variables
from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.tensor_coordinate import MultiIndex

def test_p_dimension_mapping():
    """Test P dimension coordinate mapping to understand the bug."""
    
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
    
    # Create tile distribution
    distribution = make_static_tile_distribution(encoding)
    adaptor = distribution.ps_ys_to_xs_adaptor
    
    print('=== Encoding Analysis ===')
    print(f'ps_to_rhss_major: {encoding.ps_to_rhss_major}')
    print(f'ps_to_rhss_minor: {encoding.ps_to_rhss_minor}')
    print(f'H-spaces: {encoding.hs_lengthss}')
    print()
    
    print('According to encoding:')
    print('  P0 maps to: H0[1] (WarpPerBlock_M) and H1[1] (WarpPerBlock_N)')
    print('  P1 maps to: H0[2] (ThreadPerWarp_M) and H1[2] (ThreadPerWarp_N)')
    print('  Both should affect BOTH X0 and X1 dimensions')
    print()
    
    print('=== Transform Analysis ===')
    transforms = adaptor.get_transforms()
    lower_dims = adaptor.get_lower_dimension_hidden_idss()
    upper_dims = adaptor.get_upper_dimension_hidden_idss()
    
    for i, transform in enumerate(transforms):
        print(f'Transform {i}: {transform}')
        print(f'  Lower dims: {lower_dims[i]}')
        print(f'  Upper dims: {upper_dims[i]}')
    print()
    
    print(f'Top dimension IDs: {adaptor.get_top_dimension_hidden_ids()}')
    print(f'Bottom dimension IDs: {adaptor.get_bottom_dimension_hidden_ids()}')
    print()
    
    print('=== P Dimension Coordinate Test ===')
    test_coords = [
        ([0, 0, 0, 0, 0, 0], 'Baseline'),
        ([1, 0, 0, 0, 0, 0], 'P0=1 (should affect BOTH X0 and X1)'),
        ([0, 1, 0, 0, 0, 0], 'P1=1 (should affect BOTH X0 and X1)'),
        ([1, 1, 0, 0, 0, 0], 'P0=1, P1=1 (should affect BOTH X0 and X1)'),
        ([0, 0, 1, 0, 0, 0], 'Y0=1 (Repeat_M - should affect X0)'),
        ([0, 0, 0, 0, 1, 0], 'Y2=1 (Repeat_N - should affect X1)'),
    ]
    
    for coords, description in test_coords:
        multi_idx = MultiIndex(len(coords), coords)
        x_coord = adaptor.calculate_bottom_index(multi_idx)
        print(f'PS_YS{coords} -> X{x_coord.to_list()} ({description})')
    
    print()
    print('=== Bug Analysis ===')
    print('BUG: P0 and P1 only affect X1, not X0!')
    print('Expected: Both P0 and P1 should affect both X0 and X1')
    print('Actual: P0 and P1 only affect X1 (column dimension)')
    print()
    print('This suggests the merge transforms for P dimensions are')
    print('not correctly mapping to both H-spaces as encoded.')

if __name__ == '__main__':
    test_p_dimension_mapping() 