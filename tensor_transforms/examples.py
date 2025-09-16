"""
Provides example tensor descriptor transformation strings.
"""

from typing import Dict

def get_transform_examples() -> Dict[str, str]:
    """Returns a dictionary of example tensor descriptor transformations."""
    examples = {
        "Simple Pass-Through & Merge": """
transform_tensor_descriptor(
    k_lds_block_desc_0,
    make_tuple(
        make_pass_through_transform(number<kNPerBlock>{}),
        make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))
    ),
    make_tuple(sequence<1>{}, sequence<0, 2>{}),
    make_tuple(sequence<0>{}, sequence<1>{})
)
""",
        "All Pass-Through": """
transform_tensor_descriptor(
    input_desc,
    make_tuple(
        make_pass_through_transform(number<X>{}),
        make_pass_through_transform(number<Y>{}),
        make_pass_through_transform(number<Z>{})
    ),
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{})
)
""",
        "K LDS Block Desc": """
transform_tensor_descriptor(
    k_lds_block_desc_0,
    make_tuple(
        make_merge_transform(make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
        make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                        number<kKVector / kKPack>{},
                                        number<kKPack>{}))),
    make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
    make_tuple(sequence<0>{}, sequence<1>{})
)""",
        "V LDS Block Desc": """
transform_tensor_descriptor(
    v_lds_block_desc_0,
    make_tuple(
        make_merge_transform(make_tuple(
            number<NumVLdsBuffers>{}, number<kNPerBlock / NPerRow>{}, number<NPerRow>{})),
        make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
    make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
    make_tuple(sequence<0>{}, sequence<1>{})
)""",
        "B LDS Block Desc (Raw Vars)": """
transform_tensor_descriptor(
    b_lds_block_desc_0,
    make_tuple(make_pass_through_transform(kNPerBlock),
               make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
    make_tuple(sequence<1>{}, sequence<0, 2>{}),
    make_tuple(sequence<0>{}, sequence<1>{})
)""",
        "Realistic Multi-Descriptor Example": """
constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
    make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
               number<K0PerThreadWrite>{},
               number<KThreadReadPerm * N1>{},
               number<kfold * N0 / npair>{},
               number<npair>{},
               BK1));

constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
    b_lds_block_desc,
    make_tuple(
        make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
        make_pass_through_transform(number<K0PerThreadWrite>{}),
        make_xor_transform(
            make_tuple(number<KThreadReadPerm * N1>{}, number<kfold * N0 / npair>{})),
        make_pass_through_transform(number<npair>{}),
        make_pass_through_transform(BK1)),
    make_tuple(
        sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
    make_tuple(
        sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
    b_lds_block_desc_permuted,
    make_tuple(
        make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
        make_pass_through_transform(number<K0PerThreadWrite>{}),
        make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<N1>{})),
        make_unmerge_transform(make_tuple(number<kfold>{}, number<N0 / npair>{})),
        make_pass_through_transform(number<npair>{}),
        make_pass_through_transform(BK1)),
    make_tuple(sequence<0>{},
               sequence<1>{},
               sequence<2>{},
               sequence<3>{},
               sequence<4>{},
               sequence<5>{}),
    make_tuple(sequence<1>{},
               sequence<2>{},
               sequence<0, 3>{},
               sequence<4, 5>{},
               sequence<6>{},
               sequence<7>{}));

constexpr auto b_lds_block_desc_kn = transform_tensor_descriptor(
    b_lds_block_desc_unmerged,
    make_tuple(make_merge_transform_v3_division_mod(
                   make_tuple(number<KThreadReadPerm>{},
                              number<KThreadWrite / kfold / KThreadReadPerm>{},
                              number<kfold>{},
                              number<K0PerThreadWrite>{},
                              BK1)),
               make_merge_transform_v3_division_mod(
                   make_tuple(number<N0 / npair>{}, number<npair>{}, number<N1>{}))),
    make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
    make_tuple(sequence<1>{}, sequence<0>{}));
""",
        "Arithmetic Sequence Transform": """
transform_tensor_descriptor(
    desc_0,
    make_tuple(make_unmerge_transform(lengths)),
    make_tuple(sequence<0>{}),
    make_tuple(typename arithmetic_sequence_gen<0, N, 1>::type{}))
""",
        "A LDS Block Desc Example": """
constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
    make_tuple(number<kKPerBlock / kKPack * MLdsLayer>{},
               number<kMPerBlock / MLdsLayer>{},
               number<kKPack>{}),
    make_tuple(number<kKPack>{}, number<kKPerBlock * MLdsLayer>{}, number<1>{}),
    number<kKPack>{},
    number<1>{});

constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
    a_lds_block_desc_0,
    make_tuple(make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},
                                            number<kKPerBlock / kKPack * MLdsLayer>{})),
               make_pass_through_transform(number<kKPack>{})),
    make_tuple(sequence<1, 0>{}, sequence<2>{}),
    make_tuple(sequence<1, 0>{}, sequence<2>{}));

constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
    a_lds_block_desc_permuted,
    make_tuple(make_unmerge_transform(
                   make_tuple(number<MLdsLayer>{}, number<kKPerBlock / kKPack>{})),
               make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),
               make_pass_through_transform(number<kKPack>{})),
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

constexpr auto a_lds_block_desc = transform_tensor_descriptor(
    a_lds_block_desc_xk0_mnldslayer_mn_xk1,
    make_tuple(make_merge_transform(
                   make_tuple(number<kMPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
               make_merge_transform(
                   make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
    make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
    make_tuple(sequence<0>{}, sequence<1>{}));
""",
        "X LDS Block Desc Example": """
constexpr auto x_lds_block_desc_0 = make_naive_tensor_descriptor(
    make_tuple(number<KPerBlock / KPack * MNLdsLayer>{},
               number<MNPerBlock / MNLdsLayer>{},
               number<KPack>{}),
    make_tuple(number<KPack>{}, number<KPerBlock * MNLdsLayer>{}, number<1>{}),
    number<KPack>{},
    number<1>{});

constexpr auto x_lds_block_desc_permuted = transform_tensor_descriptor(
    x_lds_block_desc_0,
    make_tuple(make_xor_transform(make_tuple(number<MNPerBlock / MNLdsLayer>{},
                                            number<KPerBlock / KPack * MNLdsLayer>{})),
               make_pass_through_transform(number<KPack>{})),
    make_tuple(sequence<1, 0>{}, sequence<2>{}),
    make_tuple(sequence<1, 0>{}, sequence<2>{}));

constexpr auto x_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
    x_lds_block_desc_permuted,
    make_tuple(make_unmerge_transform(
                   make_tuple(number<KPerBlock / KPack>{}, number<MNLdsLayer>{})),
               make_pass_through_transform(number<MNPerBlock / MNLdsLayer>{}),
               make_pass_through_transform(number<KPack>{})),
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

constexpr auto x_lds_block_desc = transform_tensor_descriptor(
    x_lds_block_desc_xk0_mnldslayer_mn_xk1,
    make_tuple(make_merge_transform_v3_division_mod(
                   make_tuple(number<MNPerBlock / MNLdsLayer>{}, number<MNLdsLayer>{})),
               make_merge_transform_v3_division_mod(
                   make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
    make_tuple(sequence<1, 2>{}, sequence<0, 3>{}),
    make_tuple(sequence<0>{}, sequence<1>{}));
""",
        "XT LDS Block Desc Example": """
constexpr auto xt_lds_block_desc_raw = make_naive_tensor_descriptor(
    make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
               number<K0PerThreadWrite>{},
               number<KThreadReadPerm * MN1>{},
               number<kfold * MN0 / mnpair>{},
               number<mnpair>{},
               KPackT),
    make_tuple(number<KPackT * kfold * MN0 * KThreadReadPerm * MN1 * K0PerThreadWrite>{},
               number<KPackT * kfold * MN0 * KThreadReadPerm * MN1>{},
               number<KPackT * kfold * MN0>{},
               number<KPackT * mnpair>{},
               number<KPackT>{},
               number<1>{}),
    number<KPackT>{},
    number<1>{});

constexpr auto xt_lds_block_desc_permuted = transform_tensor_descriptor(
    xt_lds_block_desc_raw,
    make_tuple(
        make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
        make_pass_through_transform(number<K0PerThreadWrite>{}),
        make_xor_transform(
            make_tuple(number<KThreadReadPerm * MN1>{}, number<kfold * MN0 / mnpair>{})),
        make_pass_through_transform(number<mnpair>{}),
        make_pass_through_transform(KPackT)),
    make_tuple(
        sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
    make_tuple(
        sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

constexpr auto xt_lds_block_desc_unmerged = transform_tensor_descriptor(
    xt_lds_block_desc_permuted,
    make_tuple(
        make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
        make_pass_through_transform(number<K0PerThreadWrite>{}),
        make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<MN1>{})),
        make_unmerge_transform(make_tuple(number<kfold>{}, number<MN0 / mnpair>{})),
        make_pass_through_transform(number<mnpair>{}),
        make_pass_through_transform(KPackT)),
    make_tuple(sequence<0>{},
               sequence<1>{},
               sequence<2>{},
               sequence<3>{},
               sequence<4>{},
               sequence<5>{}),
    make_tuple(sequence<1>{},
               sequence<2>{},
               sequence<0, 3>{},
               sequence<4, 5>{},
               sequence<6>{},
               sequence<7>{}));

constexpr auto xt_lds_block_desc = transform_tensor_descriptor(
    xt_lds_block_desc_unmerged,
    make_tuple(make_merge_transform_v3_division_mod(
                   make_tuple(number<KThreadReadPerm>{},
                              number<KThreadWrite / kfold / KThreadReadPerm>{},
                              number<kfold>{},
                              number<K0PerThreadWrite>{},
                              number<KPackT>{})),
               make_merge_transform_v3_division_mod(
                   make_tuple(number<MN0 / mnpair>{}, number<mnpair>{}, number<MN1>{}))),
    make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
    make_tuple(sequence<0>{}, sequence<1>{}));
""",
        "B LDS Block Desc Example": """
constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
    make_tuple(
        BK0 * number<NLdsLayer>{}, number<NPerBlock / NLdsLayer>{}, number<KPack>{}),
    make_tuple(number<KPack>{}, number<KPerBlock * NLdsLayer>{}, number<1>{}),
    number<KPack>{},
    number<1>{});

constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
    b_lds_block_desc_0,
    make_tuple(make_xor_transform(make_tuple(number<NPerBlock / NLdsLayer>{},
                                            BK0 * number<NLdsLayer>{})),
               make_pass_through_transform(number<KPack>{})),
    make_tuple(sequence<1, 0>{}, sequence<2>{}),
    make_tuple(sequence<1, 0>{}, sequence<2>{}));

constexpr auto b_lds_block_desc_bk0_nldslayer_n_bk1 = transform_tensor_descriptor(
    b_lds_block_desc_permuted,
    make_tuple(make_unmerge_transform(make_tuple(number<NLdsLayer>{}, BK0)),
               make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
               make_pass_through_transform(number<KPack>{})),
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

constexpr auto b_lds_block_desc = transform_tensor_descriptor(
    b_lds_block_desc_bk0_nldslayer_n_bk1,
    make_tuple(make_merge_transform_v3_division_mod(
                   make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
               make_merge_transform_v3_division_mod(make_tuple(BK0, number<KPack>{}))),
    make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
    make_tuple(sequence<0>{}, sequence<1>{}));
""",
        "A Grid Desc Multi-Transform Example": """
const auto a_grid_desc_mraw_kraw = make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));

const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
    a_grid_desc_mraw_kraw,
    make_tuple(make_unmerge_transform(make_tuple(K / KPerBlock, AK0Number, AK1Value)),
               make_pass_through_transform(M)),
    make_tuple(Sequence<1>{}, Sequence<0>{}),
    make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

const auto a_grid_desc_permuted = transform_tensor_descriptor(
    a_grid_desc_ak0_m_ak1,
    make_tuple(make_pass_through_transform(K / KPerBlock),
               make_xor_with_modulo_transform(make_tuple(M, AK0Number)),
               make_pass_through_transform(AK1Value)),
    make_tuple(Sequence<0>{}, Sequence<2, 1>{}, Sequence<3>{}),
    make_tuple(Sequence<0>{}, Sequence<2, 1>{}, Sequence<3>{}));

const auto a_grid_desc = transform_tensor_descriptor(
    a_grid_desc_permuted,
    make_tuple(
        make_merge_transform_v3_division_mod(make_tuple(K / KPerBlock, AK0Number)),
        make_pass_through_transform(M),
        make_pass_through_transform(AK1Value)),
    make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}),
    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
""",
        "Arithmetic Sequence with Explicit Lengths": """
transform_tensor_descriptor(
    desc_0,
    make_tuple(make_unmerge_transform(make_tuple(A, B, C))),
    make_tuple(sequence<0>{}),
    make_tuple(typename arithmetic_sequence_gen<0, N, 1>::type{}))
""",
        "Arithmetic Sequence with List Variable": """
transform_tensor_descriptor(
    desc_0,
    make_tuple(make_unmerge_transform(lengths)),
    make_tuple(sequence<0>{}),
    make_tuple(typename arithmetic_sequence_gen<0, N, 1>::type{}))
""",
        "Simple Naive Packed Only": """
constexpr auto simple_packed_desc = make_naive_tensor_descriptor_packed(
    make_tuple(number<A>{}, number<B>{}, number<C>{}, number<D>{})
);
""",
        "Merge Consecutive Dimensions (6D → 3D)": """
constexpr auto packed_6d_desc = make_naive_tensor_descriptor_packed(
    make_tuple(number<D0>{}, number<D1>{}, number<D2>{}, number<D3>{}, number<D4>{}, number<D5>{})
);

constexpr auto merged_3d_desc = transform_tensor_descriptor(
    packed_6d_desc,
    make_tuple(
        make_merge_transform(make_tuple(number<D0>{}, number<D1>{})),
        make_merge_transform(make_tuple(number<D2>{}, number<D3>{})),
        make_merge_transform(make_tuple(number<D4>{}, number<D5>{}))
    ),
    make_tuple(sequence<0, 1>{}, sequence<2, 3>{}, sequence<4, 5>{}),
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{})
);
""",
        "Simple Naive Regular (with Strides)": """
constexpr auto simple_strided_desc = make_naive_tensor_descriptor(
    make_tuple(number<M>{}, number<N>{}, number<K>{}),
    make_tuple(number<StrideM>{}, number<StrideN>{}, number<StrideK>{}),
    number<VectorSize>{},
    number<Offset>{});
"""
    }
    return examples

def get_default_variables() -> Dict[str, Dict[str, int]]:
    """Returns a dictionary of default variables for each example."""
    return {
        "Simple Pass-Through & Merge": {
            'kNPerBlock': 32,
            'kKPerBlock': 64,
            'kKPack': 8,
        },
        "Complex Nested Merge": {
            'A': 2,
            'B': 4,
            'C': 8,
            'D': 16,
        },
        "All Pass-Through": {
            'X': 16,
            'Y': 32,
            'Z': 64,
        },
        "K LDS Block Desc": {
            'NumKLdsBuffers': 2,
            'kNPerBlock': 128,
            'kKPerBlock': 32,
            'kKVector': 8,
            'kKPack': 4,
        },
        "V LDS Block Desc": {
            'NumVLdsBuffers': 2,
            'kNPerBlock': 128,
            'NPerRow': 4,
            'kKPerBlock': 32,
            'kKPack': 4,
        },
        "B LDS Block Desc (Raw Vars)": {
            'kNPerBlock': 128,
            'kKPerBlock': 64,
        },
        "Realistic Multi-Descriptor Example": {
            'KThreadWrite': 8,
            'kfold': 2,
            'KThreadReadPerm': 4,
            'K0PerThreadWrite': 2,
            'N1': 2,
            'N0': 8,
            'npair': 2,
            'BK1': 1,
        },
        "Arithmetic Sequence Transform": {
            'N': 2,
            'lengths': [2, 2],
        },
        "A LDS Block Desc Example": {
            'kKPerBlock': 32,
            'kKPack': 4,
            'MLdsLayer': 2,
            'kMPerBlock': 64,
        },
        "X LDS Block Desc Example": {
            'KPerBlock': 32,
            'KPack': 4,
            'MNLdsLayer': 2,
            'MNPerBlock': 64,
        },
        "XT LDS Block Desc Example": {
            'KThreadWrite': 8,
            'kfold': 2,
            'KThreadReadPerm': 4,
            'K0PerThreadWrite': 2,
            'MN1': 2,
            'MN0': 8,
            'mnpair': 2,
            'KPackT': 1,
        },
        "B LDS Block Desc Example": {
            'BK0': 2,
            'NLdsLayer': 2,
            'NPerBlock': 64,
            'KPack': 4,
            'KPerBlock': 32,
        },
        "A Grid Desc Multi-Transform Example": {
            'M': 1024,
            'K': 512,
            'StrideA': 512,
            'I1': 1,
            'KPerBlock': 64,
            'AK0Number': 8,
            'AK1Value': 4,
        },
        "Arithmetic Sequence with Explicit Lengths": {
            'A': 2,
            'B': 4,
            'C': 8,
            'N': 3,
        },
        "Arithmetic Sequence with List Variable": {
            # Demo of list variable - 'lengths' will show as list control in webapp
            'N': 3,
            'lengths': [2, 4, 8],  # This will be editable as a list in the UI
        },
        "Simple Naive Packed Only": {
            'A': 2,
            'B': 3,
            'C': 4,
            'D': 5,
        },
        "Merge Consecutive Dimensions (6D → 3D)": {
            'D0': 2,
            'D1': 3,
            'D2': 4,
            'D3': 5,
            'D4': 6,
            'D5': 7,
        },
        "Simple Naive Regular (with Strides)": {
            'M': 4,
            'N': 8,
            'K': 16,
            'StrideM': 128,  # N * K
            'StrideN': 16,   # K
            'StrideK': 1,    # Column-major layout
            'VectorSize': 1,
            'Offset': 0,
        },
    } 