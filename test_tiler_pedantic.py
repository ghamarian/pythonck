import pytest
from tiler_pedantic import TileDistributionEncodingPedantic, TileDistributionPedantic

# Sample encoding and variables for testing (based on tiler_pedantic.py example)
SAMPLE_ENCODING_1 = {
    "RsLengths": [1],
    "HsLengthss": [
        ["Nr_y", "Nr_p", "Nw"],
        ["Kr_y", "Kr_p", "Kw", "Kv"]
    ],
    "Ps2RHssMajor": [[1, 2], [2, 1]],
    "Ps2RHssMinor": [[1, 1], [2, 2]],
    "Ys2RHsMajor": [1, 2, 2],
    "Ys2RHsMinor": [0, 0, 3],
    "PsLengths": [32, 4]  # ADDED for NDimP=2 and for get_lengths()
}

SAMPLE_VARIABLES_1 = {
    "Nr_y": 4, "Nr_p": 8, "Nw": 2,
    "Kr_y": 8, "Kr_p": 4, "Kw": 2, "Kv": 4
}

@pytest.fixture
def sample_encoding_data():
    """Provides a copy of the sample encoding and variables."""
    return SAMPLE_ENCODING_1.copy(), SAMPLE_VARIABLES_1.copy()

def test_tile_distribution_encoding_pedantic_init(sample_encoding_data):
    """Test initialization of TileDistributionEncodingPedantic and basic NDim attributes."""
    encoding_dict, variables = sample_encoding_data
    td_encode = TileDistributionEncodingPedantic(encoding_dict, variables)

    assert td_encode.NDimX == 2  # HsLengthss has 2 entries
    assert td_encode.NDimP == 2  # Ps2RHssMajor has 2 entries
    assert td_encode.NDimY == 3  # Ys2RHsMajor has 3 entries
    assert td_encode.NDimR == 1  # RsLengths has 1 entry

    # Check resolved lengths
    assert td_encode.RsLengths == [1]
    assert td_encode.HsLengthss == [[4, 8, 2], [8, 4, 2, 4]]

def test_tile_distribution_encoding_pedantic_detail_ys_lengths(sample_encoding_data):
    """Test calculation of detail.ys_lengths_."""
    encoding_dict, variables = sample_encoding_data
    td_encode = TileDistributionEncodingPedantic(encoding_dict, variables)
    # Y0 -> H0[0] (Nr_y) = 4
    # Y1 -> H1[0] (Kr_y) = 8
    # Y2 -> H1[3] (Kv)   = 4
    assert td_encode.detail['ys_lengths_'] == [4, 8, 4]

def test_tile_distribution_encoding_pedantic_detail_rhs_major_minor_to_ys(sample_encoding_data):
    """Test calculation of detail.rhs_major_minor_to_ys_."""
    encoding_dict, variables = sample_encoding_data
    td_encode = TileDistributionEncodingPedantic(encoding_dict, variables)
    
    # Expected: (max_rh_minor is 4 for this encoding)
    # R (major 0): [-1, -1, -1, -1]
    # H0 (major 1): [0 (Y0->H0[0]), -1, -1, -1]
    # H1 (major 2): [1 (Y1->H1[0]), -1, -1, 2 (Y2->H1[3])]
    expected_mapping = [
        [-1, -1, -1, -1], # R
        [0, -1, -1, -1],  # H0
        [1, -1, -1, 2]   # H1
    ]
    assert td_encode.detail['rhs_major_minor_to_ys_'] == expected_mapping
    assert td_encode.detail['max_ndim_rh_minor_'] == 4
    assert td_encode.detail['ndims_rhs_minor_'] == [1, 3, 4] # [len(R), len(H0), len(H1)]

def test_tile_distribution_pedantic_init(sample_encoding_data):
    """Test initialization of TileDistributionPedantic."""
    encoding_dict, variables = sample_encoding_data
    td = TileDistributionPedantic(encoding_dict, variables)

    assert td.NDimX == 2
    assert td.NDimPs == 2
    assert td.NDimYs == 3
    assert td.NDimRs == 1
    
    # Check if adaptors are created (basic check, not their internal structure yet)
    assert 'Transformations' in td.PsYs2XsAdaptor
    assert 'd_length' in td.Ys2DDescriptor
    assert td.Ys2DDescriptor['d_length'] == 4 * 8 * 4 # Product of ys_lengths_

def test_calculate_rs_index_from_ps_index(sample_encoding_data):
    """Test the calculate_rs_index_from_ps_index method."""
    encoding_dict, variables = sample_encoding_data
    td = TileDistributionPedantic(encoding_dict, variables)

    # For SAMPLE_ENCODING_1:
    # NDimP = 2, NDimR = 1
    # Ps2RHssMajor = [[1,2], [2,1]] (P0->H0,H1; P1->H1,H0)
    # Ps2RHssMinor = [[1,1], [2,2]] (P0->H0[1],H1[1]; P1->H1[2],H0[2])
    # RsLengths = [1] (R0_len=1)
    # HsLengthss = [[4,8,2], [8,4,2,4]] (H0:[Nr_y,Nr_p,Nw], H1:[Kr_y,Kr_p,Kw,Kv])
    #
    # detail['ps_over_rs_derivative_'] calculation:
    # P0: No mapping to R. So ps_over_rs_derivative_[0][0] = 0
    # P1: No mapping to R. So ps_over_rs_derivative_[1][0] = 0
    # This example does not have P->R mappings, so derivatives for R will be 0.
    assert td.DstrEncode.detail['ps_over_rs_derivative_'] == [[0], [0]]
    
    ps_idx = [1, 1] # Example P-coordinates
    rs_idx = td.calculate_rs_index_from_ps_index(ps_idx)
    assert rs_idx == [0] # Since derivatives are all 0 for R

    # Create a new encoding with P->R mapping for a better test
    encoding_pr = {
        "RsLengths": [10, 20], # R0, R1
        "HsLengthss": [],
        "Ps2RHssMajor": [[0], [0]],    # P0 -> R, P1 -> R
        "Ps2RHssMinor": [[0], [1]],    # P0 -> R[0], P1 -> R[1]
        "Ys2RHsMajor": [], "Ys2RHsMinor": []
    }
    td_pr = TileDistributionPedantic(encoding_pr, {})
    # P0 maps to R[0] (length 10), P1 maps to R[1] (length 20)
    # ps_over_rs_derivative:
    # P0 contributes to R0 with derivative 1. P0 does not contribute to R1. -> [1, 0]
    # P1 contributes to R1 with derivative 1. P1 does not contribute to R0. -> [0, 1]
    # So, detail['ps_over_rs_derivative_'] should be [[1,0], [0,1]]
    assert td_pr.DstrEncode.detail['ps_over_rs_derivative_'] == [[1, 0], [0, 1]]
    
    ps_idx_pr = [2, 3] # p0=2, p1=3
    rs_idx_pr = td_pr.calculate_rs_index_from_ps_index(ps_idx_pr)
    # rs_idx[0] = ps_idx[0]*deriv[0][0] + ps_idx[1]*deriv[1][0] = 2*1 + 3*0 = 2
    # rs_idx[1] = ps_idx[0]*deriv[0][1] + ps_idx[1]*deriv[1][1] = 2*0 + 3*1 = 3
    assert rs_idx_pr == [2, 3]

# TODO: Add tests for calculate_index - requires known good output for a given input
# TODO: Add tests for get_y_indices_from_distributed_indices
# TODO: Add tests for get_visualization_data (structure and placeholder values)
# TODO: Add tests for _calculate_tile_shape

def test_calculate_tile_shape(sample_encoding_data):
    encoding_dict, variables = sample_encoding_data
    td = TileDistributionPedantic(encoding_dict, variables)
    # RsLengths = [1] -> sum = 1
    # HsLengthss[0] = [4, 8, 2] -> sum = 14
    assert td.tile_shape == [1, 14]

def test_get_visualization_data_structure(sample_encoding_data):
    """Test the basic structure of get_visualization_data output."""
    encoding_dict, variables = sample_encoding_data
    td = TileDistributionPedantic(encoding_dict, variables)
    vis_data = td.get_visualization_data()

    assert isinstance(vis_data, dict)
    
    # Check for P-coords keys
    # PsLengths = [ product(H0[1],H1[1]), product(H1[2],H0[2]) ] ? No, PsLengths is from encoding.
    # PsLengths is not directly in SAMPLE_ENCODING_1, need to derive or add for a full test
    # For now, check if at least one p_coord key exists if NDimP > 0 or a default key if NDimP=0
    
    # Using the specific SAMPLE_ENCODING_1 and VARIABLES_1:
    # Ps2RHssMajor/Minor define P-dims. NDimP = 2.
    # Lengths of P-dims are not explicitly in encoding dict, TileDistributionEncodingPedantic
    # does not directly compute PsLengths. It's usually an input from a higher level
    # or assumed based on context (like block iterators).
    # The get_visualization_data() uses self.DstrEncode.PsLengths.
    # Let's modify the fixture to include PsLengths in the encoding_dict for this test.
    
    encoding_dict_with_ps = encoding_dict.copy()
    # P0 maps to H0[1] (Nr_p=8), H1[1] (Kr_p=4).  Effective length for P0 could be 8*4=32.
    # P1 maps to H1[2] (Kw=2), H0[2] (Nw=2). Effective length for P1 could be 2*2=4.
    # This interpretation of PsLengths is complex and depends on how P dims are defined.
    # The C++ code often gets PsLengths from template arguments separate from the mapping.
    # For testing get_visualization_data, we need a concrete PsLengths.
    # Let's assume PsLengths = [2, 2] for iteration.
    encoding_dict_with_ps['PsLengths'] = [2,1] # Example P0_len=2, P1_len=1
    
    # Re-init DstrEncode with PsLengths if it were to use it directly (it doesn't for NDimP calc)
    # TileDistributionPedantic will pick up self.DstrEncode.PsLengths if it's set after init.
    # It's better if PsLengths is part of the initial encoding_dict for TileDistributionEncodingPedantic
    # if it's meant to be a primary dimension length.
    # However, PsLengths is NOT part of the TileDistributionEncodingPedantic constructor's direct fields.
    # It's used by TileDistributionPedantic.get_visualization_data() by accessing self.DstrEncode.PsLengths
    # So, we can inject it into DstrEncode instance for the test.
    
    td_for_vis = TileDistributionPedantic(encoding_dict_with_ps, variables) 
    # The following line is not strictly necessary if PsLengths is in encoding_dict_with_ps from the start
    # td_for_vis.DstrEncode.PsLengths = encoding_dict_with_ps['PsLengths'] 

    vis_data = td_for_vis.get_visualization_data()

    assert isinstance(vis_data, dict)
    # Check for expected top-level keys
    expected_keys = [
        'tile_shape', 'dimensions', 'hierarchical_structure', 
        'occupancy', 'utilization', 'source_code'
    ]
    for key in expected_keys:
        assert key in vis_data

    # Further structural checks can be added, e.g., type of vis_data['tile_shape']
    assert isinstance(vis_data['tile_shape'], list) # or typing.Sequence
    assert isinstance(vis_data['dimensions'], dict)
    assert isinstance(vis_data['hierarchical_structure'], dict)

    # Remove the old incorrect assertion
    # if td_for_vis.DstrEncode.PsLengths:
    #     num_p_iterations = 1
    #     for length in td_for_vis.DstrEncode.PsLengths:
    #         num_p_iterations *= length
    #     assert len(vis_data) == num_p_iterations # This was incorrect
    # elif td_for_vis.NDimP == 0:
    #     assert len(vis_data) == 1 
    #     assert tuple() in vis_data


# Placeholder for a more complex test case
COMPLEX_ENCODING = {
    "RsLengths": [4, 4], # R0, R1
    "HsLengthss": [
        [2, 2],    # H0: H00, H01
        [3, 3]     # H1: H10, H11
    ],
    "Ps2RHssMajor": [ # P0, P1
        [0, 1],      # P0 -> R0, H0
        [0, 2]       # P1 -> R1, H1 (using 1-based for H index in major, so 1=H0, 2=H1)
    ],
    "Ps2RHssMinor": [
        [0, 0],      # P0 -> R0[0], H0[0]
        [1, 1]       # P1 -> R1[1], H1[1]
    ],
    "Ys2RHsMajor": [0, 1, 2], # Y0->R, Y1->H0, Y2->H1
    "Ys2RHsMinor": [1, 1, 0], # Y0->R[1], Y1->H0[1], Y2->H1[0]
    "PsLengths": [2,3] # Explicit P dimension lengths for iteration in get_visualization_data
}
COMPLEX_VARIABLES = {}


@pytest.fixture
def complex_encoding_data():
    return COMPLEX_ENCODING.copy(), COMPLEX_VARIABLES.copy()

# --- Tests for Tensor Adaptor Transformation Helpers ---

def test_apply_tensor_adaptor_passthrough():
    adaptor = {
        'top_tensor_view': {'dims': [['A', 0, 10]]},
        'bottom_tensor_view': {'dims': [['B', 0, 10]]},
        'transforms': [{'type': 'PassThrough', 'top_ids_':['A'], 'bottom_ids_':['B'], 'parameters': {}}]
    }
    td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1) # Dummy
    assert td._apply_tensor_adaptor_transformations([5], adaptor) == [5]

def test_invert_tensor_adaptor_passthrough():
    adaptor = {
        'top_tensor_view': {'dims': [['A', 0, 10]]},
        'bottom_tensor_view': {'dims': [['B', 0, 10]]},
        'transforms': [{'type': 'PassThrough', 'top_ids_':['A'], 'bottom_ids_':['B'], 'parameters': {}}]
    }
    td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1) # Dummy
    assert td._invert_tensor_adaptor_transformations([5], adaptor) == [5]

def test_apply_tensor_adaptor_embed():
    adaptor = {
        'top_tensor_view': {'dims': [['Orig', 0, 5]]},
        'bottom_tensor_view': {'dims': [['Padded', 0, 10]]},
        'transforms': [
            {'type': 'Embed', 'top_ids_':['Orig'], 'bottom_ids_':['Padded'], 
             'parameters': {'dim_to_pad_id':'Orig', 'padded_dim_id':'Padded', 'pre_padding':2, 'post_padding':3, 'pad_value':0}}
        ]
    }
    td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1)
    input_coords = [3] # Original coord
    # Expected padded_coord = 3 + pre_padding (2) = 5
    output_coords = td._apply_tensor_adaptor_transformations(input_coords, adaptor)
    assert output_coords == [5]

def test_invert_tensor_adaptor_embed():
    adaptor = {
        'top_tensor_view': {'dims': [['Orig', 0, 5]]},
        'bottom_tensor_view': {'dims': [['Padded', 0, 10]]},
        'transforms': [
            {'type': 'Embed', 'top_ids_':['Orig'], 'bottom_ids_':['Padded'], 
             'parameters': {'dim_to_pad_id':'Orig', 'padded_dim_id':'Padded', 'pre_padding':2, 'post_padding':3, 'pad_value':0}}
        ]
    }
    td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1)
    target_coords = [5] # Padded coord
    # Expected original_coord = 5 - pre_padding (2) = 3
    inverted_coords = td._invert_tensor_adaptor_transformations(target_coords, adaptor)
    assert inverted_coords == [3]

def test_apply_tensor_adaptor_split():
    adaptor = {
        'top_tensor_view': {'dims': [['ToSplit', 0, 12]]}, # Length 12
        'bottom_tensor_view': {'dims': [['Major', 0, 3], ['Minor', 0, 4]]}, # L_minor=4
        'transforms': [
            {'type': 'Split', 'top_ids_':['ToSplit'], 'bottom_ids_':['Major','Minor'], 
             'parameters': {'split_dim_id':'ToSplit', 'lengths':[3, 4]}}
        ]
    }
    td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1)
    input_coords = [7] # Coord in ToSplit (0-11)
    # Major = 7 // 4 = 1
    # Minor = 7 % 4 = 3
    output_coords = td._apply_tensor_adaptor_transformations(input_coords, adaptor)
    assert output_coords == [1, 3]

def test_invert_tensor_adaptor_split():
    adaptor = {
        'top_tensor_view': {'dims': [['ToSplit', 0, 12]]},
        'bottom_tensor_view': {'dims': [['Major', 0, 3], ['Minor', 0, 4]]}, # L_minor_output=4
        'transforms': [
            {'type': 'Split', 'top_ids_':['ToSplit'], 'bottom_ids_':['Major','Minor'], 
             'parameters': {'split_dim_id':'ToSplit', 'lengths':[3, 4]}}
        ]
    }
    td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1)
    target_coords = [1, 3] # Major=1, Minor=3
    # Original = Major * L_minor_output + Minor = 1 * 4 + 3 = 7
    inverted_coords = td._invert_tensor_adaptor_transformations(target_coords, adaptor)
    assert inverted_coords == [7]

def test_apply_tensor_adaptor_merge():
    adaptor = {
        'top_tensor_view': {'dims': [['M_In', 0, 3], ['N_In', 0, 4]]}, # L_N_In = 4
        'bottom_tensor_view': {'dims': [['Merged', 0, 12]]},
        'transforms': [
            {'type': 'Merge', 'top_ids_':['M_In', 'N_In'], 'bottom_ids_':['Merged'], 
             'parameters': {'dims_to_merge_ids':['M_In','N_In'], 'lengths_of_inputs':[3,4], 'merged_dim_id':'Merged'}}
        ]
    }
    td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1)
    input_coords = [1, 3] # M_In=1, N_In=3
    # Merged = M_In * L_N_In + N_In = 1 * 4 + 3 = 7
    output_coords = td._apply_tensor_adaptor_transformations(input_coords, adaptor)
    assert output_coords == [7]

def test_invert_tensor_adaptor_merge():
    adaptor = {
        'top_tensor_view': {'dims': [['M_In', 0, 3], ['N_In', 0, 4]]},
        'bottom_tensor_view': {'dims': [['Merged', 0, 12]]}, # L_N_In = 4 (from params)
        'transforms': [
            {'type': 'Merge', 'top_ids_':['M_In', 'N_In'], 'bottom_ids_':['Merged'], 
             'parameters': {'dims_to_merge_ids':['M_In','N_In'], 'lengths_of_inputs':[3,4], 'merged_dim_id':'Merged'}}
        ]
    }
    td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1)
    target_coords = [7] # Merged_coord = 7
    # N_In = Merged % L_N_In = 7 % 4 = 3
    # M_In = Merged // L_N_In = 7 // 4 = 1
    inverted_coords = td._invert_tensor_adaptor_transformations(target_coords, adaptor)
    assert inverted_coords == [1, 3]

# Add more tests using complex_encoding_data later, especially for calculate_index 

# --- Tests for Core Calculation Methods (with mocked/simplified adaptors) ---

@pytest.mark.skip(reason="Relies on old mocked adaptor format, incompatible with new calculate_index.")
def test_calculate_index_simple_merge_passthrough(sample_encoding_data):
    """Test calculate_index with a simple, manually defined adaptor structure."""
    # Use a dummy TileDistributionPedantic instance primarily to call the method.
    # We will override its NDim attributes and provide mock adaptors.
    encoding_dict, variables = sample_encoding_data # Base for dummy init
    td = TileDistributionPedantic(encoding_dict, variables)

    Lp0, Ly0 = 2, 3
    td.NDimPs = 1
    td.NDimYs = 1

    # Mock PsYs2XsAdaptor: Merges P0 and Y0 into X0
    # X0 = P0 * Ly0 + Y0
    td.PsYs2XsAdaptor = {
        'top_tensor_view': {'dims': [['P0_abs', 0, Lp0], ['Y0_abs', 0, Ly0]]},
        'bottom_tensor_view': {'dims': [['X0_abs', 0, Lp0 * Ly0]]},
        'transforms': [
            {'type': 'Merge', 'top_ids_':['P0_abs', 'Y0_abs'], 'bottom_ids_':['X0_abs'],
             'parameters': {'lengths_of_inputs':[Lp0, Ly0], 'merged_dim_id': 'X0_abs', 
                            'dims_to_merge_ids':['P0_abs','Y0_abs']}}
        ]
    }

    # Mock Ys2DDescriptor: Passes Y0 through to D
    # D = Y0
    td.Ys2DDescriptor = {
        'adaptor_encoding': {
            'top_tensor_view': {'dims': [['Y0_d_abs', 0, Ly0]]},
            'bottom_tensor_view': {'dims': [['D_abs', 0, Ly0]]},
            'transforms': [
                {'type': 'PassThrough', 'top_ids_':['Y0_d_abs'], 'bottom_ids_':['D_abs'], 'parameters': {}}
            ]
        },
        'd_length': Ly0
    }

    idx_p = [1]  # p0 = 1
    idx_y = [2]  # y0 = 2

    # Expected calculations:
    # input_coords_psys for PsYs2XsAdaptor = [p0, y0] = [1, 2]
    # x_idx = [p0 * Ly0 + y0] = [1 * 3 + 2] = [5]
    # x_lengths = [Lp0 * Ly0] = [6]
    # flattened_x_offset = 5
    # product_of_x_lengths = 6

    # input_coords_ys_d for Ys2DAdaptor = [y0] = [2]
    # d_idx_scalar = y0 = 2

    # final_index = d_idx_scalar * product_of_x_lengths + flattened_x_offset
    #             = 2 * 6 + 5 = 12 + 5 = 17
    final_index = td.calculate_index(idx_p, idx_y)
    assert final_index == 17

@pytest.mark.skip(reason="Relies on old mocked adaptor format, incompatible with new get_y_indices.")
def test_get_y_indices_from_distributed_indices_simple_passthrough(sample_encoding_data):
    """Test get_y_indices_from_distributed_indices with a simple passthrough."""
    encoding_dict, variables = sample_encoding_data
    td = TileDistributionPedantic(encoding_dict, variables)

    Ly0 = 3
    td.NDimYs = 1
    # Mock Ys2DDescriptor: Passes Y0 through to D (and D to Y0 for inverse)
    td.Ys2DDescriptor = {
        'adaptor_encoding': {
            'top_tensor_view': {'dims': [['Y0_d_abs', 0, Ly0]]}, # This is the YS coord we want
            'bottom_tensor_view': {'dims': [['D_abs', 0, Ly0]]}, # This is the D coord input
            'transforms': [
                {'type': 'PassThrough', 'top_ids_':['Y0_d_abs'], 'bottom_ids_':['D_abs'], 'parameters': {}}
            ]
        },
        'd_length': Ly0
    }

    d_scalar_idx = 2
    # Expected: Since it's a passthrough from Y to D, inverse D to Y should give same coord.
    # target_coords for inversion is [d_scalar_idx] = [2]
    # inverted_coords should be [y0] = [2]
    ys_idx = td.get_y_indices_from_distributed_indices(d_scalar_idx)
    assert ys_idx == [2]

@pytest.mark.skip(reason="Relies on old mocked adaptor format, incompatible with new get_y_indices.")
def test_get_y_indices_from_distributed_indices_simple_merge(sample_encoding_data):
    """Test get_y_indices_from_distributed_indices with an inverted merge."""
    encoding_dict, variables = sample_encoding_data
    td = TileDistributionPedantic(encoding_dict, variables)

    Ly0, Ly1 = 2, 3 # Lengths of Y0, Y1
    td.NDimYs = 2

    # Adaptor merges Y0, Y1 into D.  D = Y0 * Ly1 + Y1
    td.Ys2DDescriptor = {
        'adaptor_encoding': {
            'top_tensor_view': {'dims': [['Y0abs', 0, Ly0], ['Y1abs', 0, Ly1]]},
            'bottom_tensor_view': {'dims': [['Dabs', 0, Ly0 * Ly1]]},
            'transforms': [
                {'type': 'Merge', 'top_ids_':['Y0abs', 'Y1abs'], 'bottom_ids_':['Dabs'],
                 'parameters': {'lengths_of_inputs':[Ly0, Ly1], 'merged_dim_id': 'Dabs', 
                                'dims_to_merge_ids':['Y0abs','Y1abs']}}
            ]
        },
        'd_length': Ly0 * Ly1
    }
    
    # If D = Y0 * Ly1 + Y1. Example: Y0=1, Y1=2. Ly1=3. D = 1*3 + 2 = 5.
    d_scalar_idx = 5 
    
    # Expected inverse:
    # target_coords = [5]
    # Y1 = D % Ly1 = 5 % 3 = 2
    # Y0 = D // Ly1 = 5 // 3 = 1
    # ys_idx = [Y0, Y1] = [1, 2]
    ys_idx = td.get_y_indices_from_distributed_indices(d_scalar_idx)
    assert ys_idx == [1, 2]

# Add more tests using complex_encoding_data later 

# --- Tests for TileDistributionPedantic ---

def test_get_lengths(sample_encoding_data):
    """Test get_lengths method."""
    encoding_dict, variables = sample_encoding_data
    td = TileDistributionPedantic(encoding_dict, variables)
    assert td.get_lengths() == [32, 4]

def test_calculate_rs_index_from_ps_index(sample_encoding_data):
    encoding_dict, variables = sample_encoding_data
    td = TileDistributionPedantic(encoding_dict, variables)
    ps_idx = [2, 3] # Arbitrary P-indices
    rs_idx = td.calculate_rs_index_from_ps_index(ps_idx)
    assert rs_idx == [0] # NDimRs=1, sum(ps_idx[i] * 0) = 0

@pytest.mark.skip(reason="Relies on placeholder adaptors, calculate_index now expects full structure.")
def test_calculate_index_placeholder_adaptors():
    enc_p1y1 = {"RsLengths": [2], "HsLengthss": [["H0"]], 
                  "PsLengths": [2], 
                  "Ps2RHssMajor": [[0]], "Ps2RHssMinor": [[0]], 
                  "Ys2RHsMajor": [1], "Ys2RHsMinor": [0]}
    var_p1y1 = {"H0": 3}
    td_p1y1 = TileDistributionPedantic(enc_p1y1, var_p1y1)
    idx_p1y1 = td_p1y1.calculate_index([1], [1])
    assert isinstance(idx_p1y1, int)

    enc_p1y0 = {"RsLengths": [2], "HsLengthss": [],
                  "PsLengths": [2], 
                  "Ps2RHssMajor": [[0]], "Ps2RHssMinor": [[0]],
                  "Ys2RHsMajor": [], "Ys2RHsMinor": []}
    var_p1y0 = {}
    td_p1y0 = TileDistributionPedantic(enc_p1y0, var_p1y0)
    idx_p1y0 = td_p1y0.calculate_index([1], [])
    assert isinstance(idx_p1y0, int)

    enc_p0y1 = {"Ys2RHsMajor": [1], "Ys2RHsMinor": [0], "HsLengthss": [["Y0_H"]]}
    var_p0y1 = {"Y0_H": 5}
    td_p0y1 = TileDistributionPedantic(enc_p0y1, var_p0y1)
    idx_p0y1 = td_p0y1.calculate_index([], [2])
    assert isinstance(idx_p0y1, int)

@pytest.mark.skip(reason="Relies on placeholder adaptors, get_y_indices now expects full structure.")
def test_get_y_indices_from_distributed_indices_placeholder():
    enc_p0y1 = {"Ys2RHsMajor": [1], "Ys2RHsMinor": [0], "HsLengthss": [["Y0_H"]]}
    var_p0y1 = {"Y0_H": 5}
    td_p0y1 = TileDistributionPedantic(enc_p0y1, var_p0y1)
    ys_coords = td_p0y1.get_y_indices_from_distributed_indices(3)
    assert ys_coords == [3]

def test_calculate_hierarchical_structure_no_hints(sample_encoding_data):
    encoding_dict, variables = sample_encoding_data
    td = TileDistributionPedantic(encoding_dict, variables)
    hier_info = td.calculate_hierarchical_tile_structure()
    assert hier_info['TileName'] == "Pedantic Tile Distribution" # Default name
    assert hier_info['ThreadPerWarp'] == [1,1] # Default
    assert hier_info['WarpPerBlock'] == [1]    # Default
    assert hier_info['VectorDimensions'] == [1] # Default
    assert hier_info['Repeat'] == [1]           # Default
    assert hier_info['BlockSize'] == [1,1]      # 1*1, 1
    assert 'Warp0' in hier_info['ThreadBlocks']
    assert 'T0' in hier_info['ThreadBlocks']['Warp0']

def test_calculate_hierarchical_structure_with_hints():
    encoding_dict, variables = SAMPLE_ENCODING_1.copy(), SAMPLE_VARIABLES_1.copy()
    encoding_dict["visualization_hints"] = {
        "thread_per_warp_p_indices": [0],  # Use P0 for TPW. P0 maps to H0[1]=8, H1[1]=4. Product = 32. So TPW=[32,1]
        "warp_per_block_p_indices": 1,   # Use P1 for WPB. P1 maps to H1[2]=2, H0[2]=2. Product = 4. So WPB=[4]
        "vector_dim_ys_index": 2,          # Use Y2 for Vec. Y2 maps to H1[3]=Kv=4. So Vec=[4]
        "repeat_factor_ys_index": 0        # Use Y0 for Rep. Y0 maps to H0[0]=Nr_y=4. So Rep=[4]
    }
    encoding_dict["_tile_name"] = "Hinted Tile"

    td_hints = TileDistributionPedantic(encoding_dict, variables)
    hier_info = td_hints.calculate_hierarchical_tile_structure()

    assert hier_info['TileName'] == "Hinted Tile"
    # P0 maps to H0[1] (len 8) and H1[1] (len 4). get_lengths_for_p_indices([0]) gives [8*4=32]
    assert hier_info['ThreadPerWarp'] == [32, 1]
    # P1 maps to H1[2] (len 2) and H0[2] (len 2). get_lengths_for_p_indices([1]) gives [2*2=4]
    assert hier_info['WarpPerBlock'] == [4]
    # YS[2] (Kv) has length 4
    assert hier_info['VectorDimensions'] == [4]
    # YS[0] (Nr_y) has length 4
    assert hier_info['Repeat'] == [4]
    # BlockSize = (32*4, 1) = [128,1]
    assert hier_info['BlockSize'] == [128, 1]
    assert len(hier_info['ThreadBlocks']) == 4 # 4 warps
    assert len(hier_info['ThreadBlocks']['Warp0']) == 32 # 32 threads in warp0

# Placeholder tests for tensor adaptor helpers from original test_tiler_pedantic
# These are more for structure and can be expanded/refined when adaptors are fully implemented.
# def test_apply_tensor_adaptor_passthrough(): // DUPLICATE START - REMOVING
#     adaptor = {
#         'top_tensor_view': {'dims': [['A', 0, 10]]},
#         'bottom_tensor_view': {'dims': [['B', 0, 10]]},
#         'transforms': [{'type': 'PassThrough', 'top_ids_':['A'], 'bottom_ids_':['B'], 'parameters': {}}]
#     }
#     td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1) # Dummy
#     assert td._apply_tensor_adaptor_transformations([5], adaptor) == [5]

# def test_invert_tensor_adaptor_passthrough():
#     adaptor = {
#         'top_tensor_view': {'dims': [['A', 0, 10]]},
#         'bottom_tensor_view': {'dims': [['B', 0, 10]]},
#         'transforms': [{'type': 'PassThrough', 'top_ids_':['A'], 'bottom_ids_':['B'], 'parameters': {}}]
#     }
#     td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1) # Dummy
#     assert td._invert_tensor_adaptor_transformations([5], adaptor) == [5]

# def test_apply_tensor_adaptor_merge():
#     adaptor = {
#         'top_tensor_view': {'dims': [['A',0,2], ['B',0,3]]}, # A len 2, B len 3
#         'bottom_tensor_view': {'dims': [['C',0,6]]},        # C len 6
#         'transforms': [{'type': 'Merge', 'top_ids_':['A','B'], 'bottom_ids_':['C'], 
#                         'parameters': {'lengths_of_inputs':[2,3], 'merged_dim_id':'C', 'dims_to_merge_ids':['A','B']}}]
#     }
#     td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1) # Dummy
#     # C = A * len(B) + B = A * 3 + B
#     # Input A=1, B=1 => C = 1*3 + 1 = 4
#     assert td._apply_tensor_adaptor_transformations([1,1], adaptor) == [4]

# def test_invert_tensor_adaptor_merge():
#     adaptor = {
#         'top_tensor_view': {'dims': [['A',0,2], ['B',0,3]]}, 
#         'bottom_tensor_view': {'dims': [['C',0,6]]},
#         'transforms': [{'type': 'Merge', 'top_ids_':['A','B'], 'bottom_ids_':['C'], 
#                         'parameters': {'lengths_of_inputs':[2,3], 'merged_dim_id':'C', 'dims_to_merge_ids':['A','B']}}]
#     }
#     td = TileDistributionPedantic(SAMPLE_ENCODING_1, SAMPLE_VARIABLES_1) # Dummy
#     # Input C=4. len(B)=3
#     # B = C % len(B) = 4 % 3 = 1
#     # A = C // len(B) = 4 // 3 = 1
#     assert td._invert_tensor_adaptor_transformations([4], adaptor) == [1,1]
# // DUPLICATE END - REMOVED THE ABOVE BLOCK (lines 623-724)

# Minimal tests for Split and Embed can be added similarly if needed for basic coverage
# For now, focusing on the main TileDistributionPedantic methods.

# Add more tests using complex_encoding_data later, especially for calculate_index 

# --- Tests for Core Calculation Methods (with mocked/simplified adaptors) ---

# def test_calculate_index_simple_merge_passthrough(sample_encoding_data): // DUPLICATE START - REMOVING
#     """Test calculate_index with a simple, manually defined adaptor structure."""
#     # Use a dummy TileDistributionPedantic instance primarily to call the method.
#     # We will override its NDim attributes and provide mock adaptors.
#     encoding_dict, variables = sample_encoding_data # Base for dummy init
#     td = TileDistributionPedantic(encoding_dict, variables)

#     Lp0, Ly0 = 2, 3
#     td.NDimPs = 1
#     td.NDimYs = 1

#     # Mock PsYs2XsAdaptor: Merges P0 and Y0 into X0
#     # X0 = P0 * Ly0 + Y0
#     td.PsYs2XsAdaptor = {
#         'top_tensor_view': {'dims': [['P0_abs', 0, Lp0], ['Y0_abs', 0, Ly0]]},
#         'bottom_tensor_view': {'dims': [['X0_abs', 0, Lp0 * Ly0]]},
#         'transforms': [
#             {'type': 'Merge', 'top_ids_':['P0_abs', 'Y0_abs'], 'bottom_ids_':['X0_abs'],
#              'parameters': {'lengths_of_inputs':[Lp0, Ly0], 'merged_dim_id': 'X0_abs', 
#                             'dims_to_merge_ids':['P0_abs','Y0_abs']}}
#         ]
#     }

#     # Mock Ys2DDescriptor: Passes Y0 through to D
#     # D = Y0
#     td.Ys2DDescriptor = {
#         'adaptor_encoding': {
#             'top_tensor_view': {'dims': [['Y0_d_abs', 0, Ly0]]},
#             'bottom_tensor_view': {'dims': [['D_abs', 0, Ly0]]},
#             'transforms': [
#                 {'type': 'PassThrough', 'top_ids_':['Y0_d_abs'], 'bottom_ids_':['D_abs'], 'parameters': {}}
#             ]
#         },
#         'd_length': Ly0
#     }

#     idx_p = [1]  # p0 = 1
#     idx_y = [2]  # y0 = 2

#     # Expected calculations:
#     # input_coords_psys for PsYs2XsAdaptor = [p0, y0] = [1, 2]
#     # x_idx = [p0 * Ly0 + y0] = [1 * 3 + 2] = [5]
#     # x_lengths = [Lp0 * Ly0] = [6]
#     # flattened_x_offset = 5
#     # product_of_x_lengths = 6

#     # input_coords_ys_d for Ys2DAdaptor = [y0] = [2]
#     # d_idx_scalar = y0 = 2

#     # final_index = d_idx_scalar * product_of_x_lengths + flattened_x_offset
#     #             = 2 * 6 + 5 = 12 + 5 = 17
#     final_index = td.calculate_index(idx_p, idx_y)
#     assert final_index == 17

# def test_get_y_indices_from_distributed_indices_simple_passthrough(sample_encoding_data):
#     """Test get_y_indices_from_distributed_indices with a simple passthrough."""
#     encoding_dict, variables = sample_encoding_data
#     td = TileDistributionPedantic(encoding_dict, variables)

#     Ly0 = 3
#     td.NDimYs = 1
#     # Mock Ys2DDescriptor: Passes Y0 through to D (and D to Y0 for inverse)
#     td.Ys2DDescriptor = {
#         'adaptor_encoding': {
#             'top_tensor_view': {'dims': [['Y0_d_abs', 0, Ly0]]}, # This is the YS coord we want
#             'bottom_tensor_view': {'dims': [['D_abs', 0, Ly0]]}, # This is the D coord input
#             'transforms': [
#                 {'type': 'PassThrough', 'top_ids_':['Y0_d_abs'], 'bottom_ids_':['D_abs'], 'parameters': {}}
#             ]
#         },
#         'd_length': Ly0
#     }

#     d_scalar_idx = 2
#     # Expected: Since it's a passthrough from Y to D, inverse D to Y should give same coord.
#     # target_coords for inversion is [d_scalar_idx] = [2]
#     # inverted_coords should be [y0] = [2]
#     ys_idx = td.get_y_indices_from_distributed_indices(d_scalar_idx)
#     assert ys_idx == [2]

# def test_get_y_indices_from_distributed_indices_simple_merge(sample_encoding_data):
#     """Test get_y_indices_from_distributed_indices with an inverted merge."""
#     encoding_dict, variables = sample_encoding_data
#     td = TileDistributionPedantic(encoding_dict, variables)

#     Ly0, Ly1 = 2, 3 # Lengths of Y0, Y1
#     td.NDimYs = 2

#     # Adaptor merges Y0, Y1 into D.  D = Y0 * Ly1 + Y1
#     td.Ys2DDescriptor = {
#         'adaptor_encoding': {
#             'top_tensor_view': {'dims': [['Y0abs', 0, Ly0], ['Y1abs', 0, Ly1]]},
#             'bottom_tensor_view': {'dims': [['Dabs', 0, Ly0 * Ly1]]},
#             'transforms': [
#                 {'type': 'Merge', 'top_ids_':['Y0abs', 'Y1abs'], 'bottom_ids_':['Dabs'],
#                  'parameters': {'lengths_of_inputs':[Ly0, Ly1], 'merged_dim_id': 'Dabs', 
#                                 'dims_to_merge_ids':['Y0abs','Y1abs']}}
#             ]
#         },
#         'd_length': Ly0 * Ly1
#     }
    
#     # If D = Y0 * Ly1 + Y1. Example: Y0=1, Y1=2. Ly1=3. D = 1*3 + 2 = 5.
#     d_scalar_idx = 5 
    
#     # Expected inverse:
#     # target_coords = [5]
#     # Y1 = D % Ly1 = 5 % 3 = 2
#     # Y0 = D // Ly1 = 5 // 3 = 1
#     # ys_idx = [Y0, Y1] = [1, 2]
#     ys_idx = td.get_y_indices_from_distributed_indices(d_scalar_idx)
#     assert ys_idx == [1, 2]
# // DUPLICATE END - REMOVED THE ABOVE BLOCK (lines 623-724)

# Add more tests using complex_encoding_data later 

# --- Start: New Fixture and Test for Adaptor Generation ---
ADAPTOR_TEST_ENCODING_1 = {
    "RsLengths": [2],             # R0_len = 2
    "HsLengthss": [               # H0, H1
        [4, 8],                   # H0_0=4, H0_1=8  (X0 len 32)
        [3]                       # H1_0=3          (X1 len 3)
    ],
    "PsLengths": [16, 7],         # P0_len=16, P1_len=7
    "Ps2RHssMajor": [             # P0, P1
        [1],                      # P0 -> H0
        [2, 0]                    # P1 -> H1, R0 (Test P mapping to R)
    ],
    "Ps2RHssMinor": [
        [0],                      # P0 -> H0_0 (length 4)
        [0, 0]                    # P1 -> H1_0 (len 3), R0_0 (len 2)
    ],
    "Ys2RHsMajor": [0, 1],        # Y0 -> R0, Y1 -> H0
    "Ys2RHsMinor": [0, 1]         # Y0 -> R0_0 (len 2), Y1 -> H0_1 (len 8)
}
ADAPTOR_TEST_VARIABLES_1 = {}

@pytest.fixture
def adaptor_test_data_1():
    """Provides a specific encoding for testing adaptor generation."""
    return ADAPTOR_TEST_ENCODING_1.copy(), ADAPTOR_TEST_VARIABLES_1.copy()

def test_create_adaptor_encodings_pedantic(adaptor_test_data_1):
    """Test the _create_adaptor_encodings_json_style method output."""
    encoding_dict, variables = adaptor_test_data_1
    td = TileDistributionPedantic(encoding_dict, variables)

    # 1. Verify rh_map (DstrDetail)
    # Expected rh_map:
    # R0_0 (major 0, minor 0): id=0, length=2
    # H0_0 (major 1, minor 0): id=1, length=4
    # H0_1 (major 1, minor 1): id=2, length=8
    # H1_0 (major 2, minor 0): id=3, length=3
    # next_available_hidden_id should be 4 after these.
    expected_rh_map = {
        (0, 0): {'id': 0, 'length': 2}, # R0_0
        (1, 0): {'id': 1, 'length': 4}, # H0_0
        (1, 1): {'id': 2, 'length': 8}, # H0_1
        (2, 0): {'id': 3, 'length': 3}, # H1_0
    }
    assert td.DstrDetail['rh_map'] == expected_rh_map

    # 2. Verify PsYs2XsAdaptor
    psys_adaptor = td.PsYs2XsAdaptor
    
    # BottomView
    assert psys_adaptor['BottomView']['BottomDimensionIdToName'] == {0: "X0", 1: "X1"}
    assert psys_adaptor['BottomView']['BottomDimensionNameLengths'] == {"X0": 32, "X1": 3} # 4*8, 3

    # TopView _effective_display_order_ids_ (P0_hid, P1_hid, Y0_hid, Y1_hid)
    # P0 merges H0_0 (id=1) -> P0_hid=4
    # P1 merges H1_0 (id=3), R0_0 (id=0) -> P1_hid=5
    # Y0 is R0_0 (id=0)
    # Y1 is H0_1 (id=2)
    expected_effective_ids = [4, 5, 0, 2] 
    assert psys_adaptor['TopView']['_effective_display_order_ids_'] == expected_effective_ids
    
    # TopView Names and Lengths (maps hidden_id to Name, and Name to Length)
    # P0 (hid=4), len=16 (from PsLengths[0])
    # P1 (hid=5), len=7  (from PsLengths[1])
    # Y0 (hid=0), len=2  (from R0_0)
    # Y1 (hid=2), len=8  (from H0_1)
    assert psys_adaptor['TopView']['TopDimensionIdToName'] == {4: "P0", 5: "P1", 0: "Y0", 2: "Y1"}
    assert psys_adaptor['TopView']['TopDimensionNameLengths'] == {"P0": 16, "P1": 7, "Y0": 2, "Y1": 8}

    # Transformations
    transforms = psys_adaptor['Transformations']
    assert len(transforms) == 5 # Replicate, Unmerge H0, Unmerge H1, Merge P0, Merge P1

    # T0: Replicate R0
    assert transforms[0]['Name'] == "Replicate"
    assert transforms[0]['MetaData'] == [2] # RsLengths
    assert transforms[0]['SrcDimIds'] == []
    assert transforms[0]['DstDimIds'] == [0] # hid for R0_0

    # T1: Unmerge X0 from H0
    assert transforms[1]['Name'] == "Unmerge"
    assert transforms[1]['MetaData'] == [4, 8] # HsLengthss[0]
    assert transforms[1]['SrcDimIds'] == [0]    # X0
    assert transforms[1]['DstDimIds'] == [1, 2] # hids for H0_0, H0_1

    # T2: Unmerge X1 from H1
    assert transforms[2]['Name'] == "Unmerge"
    assert transforms[2]['MetaData'] == [3]    # HsLengthss[1]
    assert transforms[2]['SrcDimIds'] == [1]    # X1
    assert transforms[2]['DstDimIds'] == [3]    # hid for H1_0

    # T3: Merge P0 from H0_0
    # P0 -> H0_0 (id=1, len=4)
    assert transforms[3]['Name'] == "Merge"
    assert transforms[3]['MetaData'] == [4]
    assert transforms[3]['SrcDimIds'] == [1] # hid of H0_0
    assert transforms[3]['DstDimIds'] == [4] # new hid for P0

    # T4: Merge P1 from H1_0, R0_0
    # P1 -> H1_0 (id=3, len=3), R0_0 (id=0, len=2)
    assert transforms[4]['Name'] == "Merge"
    assert transforms[4]['MetaData'] == [3, 2] # Lengths of H1_0, R0_0
    assert transforms[4]['SrcDimIds'] == [3, 0] # hids of H1_0, R0_0
    assert transforms[4]['DstDimIds'] == [5] # new hid for P1

    # 3. Verify Ys2DDescriptor
    ys_d_desc = td.Ys2DDescriptor
    # Y0 (from R0_0, len 2), Y1 (from H0_1, len 8)
    assert ys_d_desc['d_length'] == 2 * 8 # 16

    # BottomView
    assert ys_d_desc['BottomView']['BottomDimensionIdToName'] == {"0": "D_flat"}
    assert ys_d_desc['BottomView']['BottomDimensionNameLengths'] == {"D_flat": 16}

    # TopView
    assert ys_d_desc['TopView']['TopDimensionIdToName'] == {0: "Y0", 1: "Y1"} # Logical YS IDs
    assert ys_d_desc['TopView']['TopDimensionNameLengths'] == {"Y0": 2, "Y1": 8}

    # Transformation (single Unmerge)
    ys_d_transforms = ys_d_desc['Transformations']
    assert len(ys_d_transforms) == 1
    assert ys_d_transforms[0]['Name'] == "Unmerge"
    assert ys_d_transforms[0]['MetaData'] == [2, 8] # Lengths of Y0, Y1
    assert ys_d_transforms[0]['SrcDimIds'] == [0]    # D_flat
    assert ys_d_transforms[0]['DstDimIds'] == [0, 1] # Logical Y0, Y1

# --- End: New Fixture and Test for Adaptor Generation ---

# def test_tile_distribution_pedantic_init(sample_encoding_data): // DUPLICATE START - REMOVING
#     """Test initialization of TileDistributionPedantic."""
#     encoding_dict, variables = sample_encoding_data
#     td = TileDistributionPedantic(encoding_dict, variables)

#     assert td.NDimX == 2
#     assert td.NDimPs == 2
#     assert td.NDimYs == 3
#     assert td.NDimRs == 1
    
#     # Check if adaptors are created (basic check, not their internal structure yet)
#     assert 'Transformations' in td.PsYs2XsAdaptor
#     assert 'd_length' in td.Ys2DDescriptor
#     assert td.Ys2DDescriptor['d_length'] == 4 * 8 * 4 # Product of ys_lengths_
# // DUPLICATE END - REMOVED THE ABOVE BLOCK (lines 851-863)

# Add more tests using complex_encoding_data later 

# --- Tests for new calculate_index and get_y_indices_from_distributed_indices ---

def test_calculate_index_with_adaptor_test_data_1(adaptor_test_data_1):
    """Test calculate_index using the ADAPTOR_TEST_ENCODING_1."""
    encoding_dict, variables = adaptor_test_data_1
    td = TileDistributionPedantic(encoding_dict, variables)

    # Based on manual calculation:
    # idx_p = [1, 2] (p0=1, p1=2)
    # idx_y = [1, 3] (y0=1, y1=3)
    # Expected final_index = 1090
    idx_p = [1, 2]
    idx_y = [1, 3]
    final_index = td.calculate_index(idx_p, idx_y)
    assert final_index == 1090

    # Another case: idx_p = [0,0], idx_y = [0,0]
    # P0(hid=4, coord=0) -> H0_0(hid=1) = 0 % 4 = 0
    # P1(hid=5, coord=0) -> H1_0(hid=3)=0, R0_0(hid=0)=0
    # Y0(R0_0 hid=0) = 0
    # Y1(H0_1 hid=2) = 0
    # all_hid_coords: {0:0, 1:0, 2:0, 3:0} (relevant ones)
    # X0 from H0_0(0), H0_1(0) -> 0*8+0 = 0
    # X1 from H1_0(0) -> 0
    # xs_coords = [0,0]
    # flattened_x_offset = 0*3+0 = 0
    # product_of_x_lengths = 32*3 = 96
    # d_scalar_idx from Ys=[0,0] with L=[2,8] -> 0*8+0 = 0
    # final_index = 0 * 96 + 0 = 0
    idx_p_zero = [0,0]
    idx_y_zero = [0,0]
    final_index_zero = td.calculate_index(idx_p_zero, idx_y_zero)
    assert final_index_zero == 0

def test_get_y_indices_from_distributed_indices_with_adaptor_test_data_1(adaptor_test_data_1):
    """Test get_y_indices_from_distributed_indices using ADAPTOR_TEST_ENCODING_1."""
    encoding_dict, variables = adaptor_test_data_1
    td = TileDistributionPedantic(encoding_dict, variables)

    # Based on manual calculation:
    # d_scalar_idx = 11 should yield ys_coords = [1, 3]
    # Ys2DAdaptor Unmerge MetaData (lengths of Ys) is [L_Y0=2, L_Y1=8]
    d_scalar_idx = 11
    expected_ys_coords = [1, 3] # y0=1, y1=3
    ys_coords = td.get_y_indices_from_distributed_indices(d_scalar_idx)
    assert ys_coords == expected_ys_coords

    # Test with d_scalar_idx = 0
    # temp_d = 0
    # Y1 = 0 % 8 = 0
    # temp_d = 0 // 8 = 0
    # Y0 = 0 % 2 = 0
    # temp_d = 0 // 2 = 0
    # expected_ys_coords = [0,0]
    d_scalar_idx_zero = 0
    expected_ys_coords_zero = [0, 0]
    ys_coords_zero = td.get_y_indices_from_distributed_indices(d_scalar_idx_zero)
    assert ys_coords_zero == expected_ys_coords_zero

# Add more tests using complex_encoding_data later 