import pytest
from extract_descriptors import extract_descriptors_from_text

def test_extract_descriptors_from_realistic_code():
    code = '''
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
'''
    descriptors = extract_descriptors_from_text(code)
    print("Extracted descriptors:")
    for i, desc in enumerate(descriptors):
        print(f"--- Descriptor {i+1} ---\n{desc}\n")
    assert len(descriptors) == 4, f"Expected 4 descriptors, got {len(descriptors)}"
    expected_types = [
        "make_naive_tensor_descriptor_packed",
        "transform_tensor_descriptor",
        "transform_tensor_descriptor",
        "transform_tensor_descriptor",
    ]
    for desc, expected in zip(descriptors, expected_types):
        assert desc.strip().startswith(expected), f"Descriptor does not start with {expected}: {desc[:40]}..." 
 