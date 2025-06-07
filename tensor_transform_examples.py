"""
Provides example tensor descriptor transformation strings.
"""

def get_transform_examples() -> dict[str, str]:
    """Returns a dictionary of example tensor descriptor transformations."""
    return {
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
        "Complex Nested Merge": """
transform_tensor_descriptor(
    input_desc,
    make_tuple(
        make_merge_transform(
            make_tuple(
                make_pass_through_transform(number<A>{}),
                make_merge_transform(make_tuple(number<B>{}, number<C>{}))
            )
        ),
        make_pass_through_transform(number<D>{})
    ),
    make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
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
"""
    }

def get_default_variables() -> dict[str, dict[str, int]]:
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
    } 