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
)"""
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
        }
    } 