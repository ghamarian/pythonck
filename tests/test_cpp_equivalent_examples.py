#!/usr/bin/env python3
"""
Comprehensive tests for C++ equivalent behavior using examples from examples.py.

These tests verify that our Python implementation now behaves the same way as the C++ version,
particularly with nested merge transforms and pass-through transforms.
"""

import pytest
import math
from pytensor.tensor_descriptor import (
    make_naive_tensor_descriptor_packed,
    make_naive_tensor_descriptor,
    make_merge_transform,
    make_pass_through_transform,
    make_unmerge_transform,
    make_xor_transform,
    transform_tensor_descriptor,
    make_tuple,
    number,
    sequence,
    MergeTransform,
    PassThroughTransform,
    UnmergeTransform,
    XorTransform,
    EmbedTransform
)


class TestCppEquivalentExamples:
    """Test various examples from examples.py to verify C++ equivalent behavior."""

    def test_simple_pass_through_and_merge(self):
        """Test: Simple Pass-Through & Merge example"""
        # Parameters from examples.py
        kNPerBlock = 32
        kKPerBlock = 64
        kKPack = 8
        
        # Create input descriptor
        input_desc = make_naive_tensor_descriptor_packed([kNPerBlock, kKPerBlock])
        
        # Create transforms - this should match the C++ behavior exactly
        transforms = make_tuple(
            make_pass_through_transform(number(kNPerBlock)),
            make_merge_transform(make_tuple(number(kKPerBlock // kKPack), number(kKPack)))
        )
        
        # Apply transformation
        result = transform_tensor_descriptor(
            input_desc,
            transforms,
            make_tuple(sequence(0), sequence(1)),
            make_tuple(sequence(0), sequence(1))
        )
        
        # Verify the result has correct lengths (C++ equivalent behavior)
        expected_lengths = [kNPerBlock, kKPerBlock]
        assert result.get_lengths() == expected_lengths
        
        # Verify the result has 3 transforms: original UnmergeTransform + 2 new transforms
        assert len(result.transforms) == 3
        assert isinstance(result.transforms[0], UnmergeTransform)
        assert isinstance(result.transforms[1], PassThroughTransform)
        assert isinstance(result.transforms[2], MergeTransform)
        
        # Verify the PassThroughTransform has correct length
        pass_through_transform = result.transforms[1]
        assert pass_through_transform.length == kNPerBlock
        
        # Verify the MergeTransform has correct lengths
        merge_transform = result.transforms[2]
        assert merge_transform.lengths == [kKPerBlock // kKPack, kKPack]


    def test_all_pass_through(self):
        """Test: All Pass-Through example"""
        # Parameters from examples.py
        X, Y, Z = 16, 32, 64
        
        # Create input descriptor
        input_desc = make_naive_tensor_descriptor_packed([X, Y, Z])
        
        # Create transforms
        transforms = make_tuple(
            make_pass_through_transform(number(X)),
            make_pass_through_transform(number(Y)),
            make_pass_through_transform(number(Z))
        )
        
        # Apply transformation
        result = transform_tensor_descriptor(
            input_desc,
            transforms,
            make_tuple(sequence(0), sequence(1), sequence(2)),
            make_tuple(sequence(0), sequence(1), sequence(2))
        )
        
        # Verify C++ equivalent behavior
        expected_lengths = [X, Y, Z]
        assert result.get_lengths() == expected_lengths
        
        # Should have original transform plus 3 new pass-through transforms
        assert len(result.transforms) == 4
        assert isinstance(result.transforms[0], UnmergeTransform)
        assert isinstance(result.transforms[1], PassThroughTransform)
        assert isinstance(result.transforms[2], PassThroughTransform)
        assert isinstance(result.transforms[3], PassThroughTransform)

    def test_k_lds_block_desc(self):
        """Test: K LDS Block Desc example"""
        # Parameters from examples.py
        NumKLdsBuffers = 2
        kNPerBlock = 128
        kKPerBlock = 32
        kKVector = 8
        kKPack = 4
        
        # Create input descriptor
        input_desc = make_naive_tensor_descriptor_packed([
            NumKLdsBuffers, kKPerBlock // kKVector, kKVector // kKPack, kNPerBlock, kKPack
        ])
        
        # Create transforms
        transforms = make_tuple(
            make_merge_transform(make_tuple(number(NumKLdsBuffers), number(kNPerBlock))),
            make_merge_transform(make_tuple(
                number(kKPerBlock // kKVector),
                number(kKVector // kKPack),
                number(kKPack)
            ))
        )
        
        # Apply transformation
        result = transform_tensor_descriptor(
            input_desc,
            transforms,
            make_tuple(sequence(0, 3), sequence(1, 2, 4)),
            make_tuple(sequence(0), sequence(1))
        )
        
        # Verify C++ equivalent behavior
        expected_lengths = [
            NumKLdsBuffers * kNPerBlock,
            (kKPerBlock // kKVector) * (kKVector // kKPack) * kKPack
        ]
        assert result.get_lengths() == expected_lengths
        
        # Should have 3 total transforms (original + 2 new)
        assert len(result.transforms) == 3
        assert isinstance(result.transforms[0], UnmergeTransform)
        assert isinstance(result.transforms[1], MergeTransform)
        assert isinstance(result.transforms[2], MergeTransform)

    def test_v_lds_block_desc(self):
        """Test: V LDS Block Desc example"""
        # Parameters from examples.py
        NumVLdsBuffers = 2
        kNPerBlock = 128
        NPerRow = 4
        kKPerBlock = 32
        kKPack = 4
        
        # Create input descriptor
        input_desc = make_naive_tensor_descriptor_packed([
            NumVLdsBuffers, kKPerBlock // kKPack, kNPerBlock // NPerRow, NPerRow, kKPack
        ])
        
        # Create transforms
        transforms = make_tuple(
            make_merge_transform(make_tuple(
                number(NumVLdsBuffers),
                number(kNPerBlock // NPerRow),
                number(NPerRow)
            )),
            make_merge_transform(make_tuple(
                number(kKPerBlock // kKPack),
                number(kKPack)
            ))
        )
        
        # Apply transformation
        result = transform_tensor_descriptor(
            input_desc,
            transforms,
            make_tuple(sequence(0, 2, 3), sequence(1, 4)),
            make_tuple(sequence(0), sequence(1))
        )
        
        # Verify C++ equivalent behavior
        expected_lengths = [
            NumVLdsBuffers * (kNPerBlock // NPerRow) * NPerRow,
            (kKPerBlock // kKPack) * kKPack
        ]
        assert result.get_lengths() == expected_lengths

    def test_b_lds_block_desc_raw_vars(self):
        """Test: B LDS Block Desc (Raw Vars) example"""
        # Parameters from examples.py
        kNPerBlock = 128
        kKPerBlock = 64
        
        # Create input descriptor (assuming b_lds_block_desc_0 structure)
        input_desc = make_naive_tensor_descriptor_packed([kNPerBlock, kKPerBlock])
        
        # Create transforms
        transforms = make_tuple(
            make_pass_through_transform(number(kNPerBlock)),
            make_merge_transform(make_tuple(number(kKPerBlock // 8), number(8)))
        )
        
        # Apply transformation
        result = transform_tensor_descriptor(
            input_desc,
            transforms,
            make_tuple(sequence(0), sequence(1)),
            make_tuple(sequence(0), sequence(1))
        )
        
        # Verify C++ equivalent behavior
        expected_lengths = [kNPerBlock, kKPerBlock]
        assert result.get_lengths() == expected_lengths
        
        # Should have 3 transforms: original + 2 new
        assert len(result.transforms) == 3
        assert isinstance(result.transforms[0], UnmergeTransform)
        assert isinstance(result.transforms[1], PassThroughTransform)
        assert isinstance(result.transforms[2], MergeTransform)

    def test_realistic_multi_descriptor_example(self):
        """Test: Realistic Multi-Descriptor Example from examples.py"""
        # Parameters from examples.py
        KThreadWrite = 8
        kfold = 2
        KThreadReadPerm = 4
        K0PerThreadWrite = 2
        N1 = 2
        N0 = 8
        npair = 2
        BK1 = 1
        
        # Create the base descriptor
        base_desc = make_naive_tensor_descriptor_packed([
            KThreadWrite // kfold // KThreadReadPerm,
            K0PerThreadWrite,
            KThreadReadPerm * N1,
            kfold * N0 // npair,
            npair,
            BK1
        ])
        
        # First transformation - permuted
        permuted_desc = transform_tensor_descriptor(
            base_desc,
            make_tuple(
                make_pass_through_transform(number(KThreadWrite // kfold // KThreadReadPerm)),
                make_pass_through_transform(number(K0PerThreadWrite)),
                make_xor_transform(make_tuple(
                    number(KThreadReadPerm * N1),
                    number(kfold * N0 // npair)
                )),
                make_pass_through_transform(number(npair)),
                make_pass_through_transform(number(BK1))
            ),
            make_tuple(
                sequence(0), sequence(1), sequence(2, 3), sequence(4), sequence(5)
            ),
            make_tuple(
                sequence(0), sequence(1), sequence(2, 3), sequence(4), sequence(5)
            )
        )
        
        # Second transformation - unmerged
        unmerged_desc = transform_tensor_descriptor(
            permuted_desc,
            make_tuple(
                make_pass_through_transform(number(KThreadWrite // kfold // KThreadReadPerm)),
                make_pass_through_transform(number(K0PerThreadWrite)),
                make_unmerge_transform(make_tuple(number(KThreadReadPerm), number(N1))),
                make_unmerge_transform(make_tuple(number(kfold), number(N0 // npair))),
                make_pass_through_transform(number(npair)),
                make_pass_through_transform(number(BK1))
            ),
            make_tuple(
                sequence(0), sequence(1), sequence(2), sequence(3), sequence(4), sequence(5)
            ),
            make_tuple(
                sequence(1), sequence(2), sequence(0, 3), sequence(4, 5), sequence(6), sequence(7)
            )
        )
        
        # Verify the multi-stage transformation worked
        assert len(unmerged_desc.transforms) >= 6
        assert unmerged_desc.get_lengths() == [
            KThreadReadPerm,  # 4
            KThreadWrite // kfold // KThreadReadPerm,  # 1
            K0PerThreadWrite,  # 2
            N1,  # 2
            kfold,  # 2
            N0 // npair,  # 4
            npair,  # 2
            BK1  # 1
        ]

    def test_a_lds_block_desc_example(self):
        """Test: A LDS Block Desc Example from examples.py"""
        # Parameters from examples.py
        kKPerBlock = 32
        kKPack = 4
        MLdsLayer = 2
        kMPerBlock = 64
        
        # Create initial descriptor with strides
        initial_desc = make_naive_tensor_descriptor(
            [kKPerBlock // kKPack * MLdsLayer, kMPerBlock // MLdsLayer, kKPack],
            [kKPack, kKPerBlock * MLdsLayer, 1]
        )
        
        # First transformation - permuted
        permuted_desc = transform_tensor_descriptor(
            initial_desc,
            make_tuple(
                make_xor_transform(make_tuple(
                    number(kMPerBlock // MLdsLayer),
                    number(kKPerBlock // kKPack * MLdsLayer)
                )),
                make_pass_through_transform(number(kKPack))
            ),
            make_tuple(sequence(1, 0), sequence(2)),
            make_tuple(sequence(1, 0), sequence(2))
        )
        
        # Verify the transformation worked
        assert len(permuted_desc.transforms) >= 2
        expected_lengths = [kKPerBlock // kKPack * MLdsLayer, kMPerBlock // MLdsLayer, kKPack]
        assert permuted_desc.get_lengths() == expected_lengths

    def test_arithmetic_sequence_transform(self):
        """Test: Arithmetic Sequence Transform example"""
        # Parameters from examples.py
        lengths = [2, 4, 8]
        
        # Create input descriptor
        input_desc = make_naive_tensor_descriptor_packed([math.prod(lengths)])
        
        # Create unmerge transform
        transforms = make_tuple(
            make_unmerge_transform(make_tuple(number(lengths[0]), number(lengths[1]), number(lengths[2])))
        )
        
        # Apply transformation
        result = transform_tensor_descriptor(
            input_desc,
            transforms,
            make_tuple(sequence(0)),
            make_tuple(sequence(0, 1, 2))
        )
        
        # Verify the result
        expected_lengths = lengths
        assert result.get_lengths() == expected_lengths
        
        # Should have 2 transforms: original + new unmerge
        assert len(result.transforms) == 2
        assert isinstance(result.transforms[0], UnmergeTransform)
        assert isinstance(result.transforms[1], UnmergeTransform)

    def test_mixed_transform_types(self):
        """Test mixing pass-through and merge transforms"""
        A, B, C, D = 2, 3, 4, 5
        
        # Create input descriptor
        input_desc = make_naive_tensor_descriptor_packed([A, B, C, D])
        
        # Create mixed transforms
        transforms = make_tuple(
            make_pass_through_transform(number(A)),
            make_merge_transform(make_tuple(number(B), number(C))),
            make_pass_through_transform(number(D))
        )
        
        # Apply transformation
        result = transform_tensor_descriptor(
            input_desc,
            transforms,
            make_tuple(sequence(0), sequence(1, 2), sequence(3)),
            make_tuple(sequence(0), sequence(1), sequence(2))
        )
        
        # Verify C++ equivalent behavior
        expected_lengths = [A, B * C, D]
        assert result.get_lengths() == expected_lengths
        
        # Should have 4 transforms: original UnmergeTransform + 3 new transforms
        assert len(result.transforms) == 4
        assert isinstance(result.transforms[0], UnmergeTransform)
        assert isinstance(result.transforms[1], PassThroughTransform)
        assert isinstance(result.transforms[2], MergeTransform)
        assert isinstance(result.transforms[3], PassThroughTransform)

    def test_unmerge_transform_basic(self):
        """Test basic unmerge transform"""
        # Create input descriptor with merged dimension
        input_desc = make_naive_tensor_descriptor_packed([24])  # 24 = 2 * 3 * 4
        
        # Create unmerge transform
        transforms = make_tuple(
            make_unmerge_transform(make_tuple(number(2), number(3), number(4)))
        )
        
        # Apply transformation
        result = transform_tensor_descriptor(
            input_desc,
            transforms,
            make_tuple(sequence(0)),
            make_tuple(sequence(0, 1, 2))
        )
        
        # Verify C++ equivalent behavior
        expected_lengths = [2, 3, 4]
        assert result.get_lengths() == expected_lengths
        
        # Should have 2 transforms: original + new unmerge
        assert len(result.transforms) == 2
        assert isinstance(result.transforms[0], UnmergeTransform)
        assert isinstance(result.transforms[1], UnmergeTransform)

    def test_merge_then_unmerge(self):
        """Test merge followed by unmerge"""
        A, B, C = 2, 3, 4
        
        # Create input descriptor
        input_desc = make_naive_tensor_descriptor_packed([
            A, B, C
        ])
        
        # First merge B and C
        intermediate = transform_tensor_descriptor(
            input_desc,
            make_tuple(
                make_pass_through_transform(number(A)),
                make_merge_transform(make_tuple(number(B), number(C)))
            ),
            make_tuple(sequence(0), sequence(1, 2)),
            make_tuple(sequence(0), sequence(1))
        )
        
        # Then unmerge the merged dimension
        final_result = transform_tensor_descriptor(
            intermediate,
            make_tuple(
                make_pass_through_transform(number(A)),
                make_unmerge_transform(make_tuple(number(B), number(C)))
            ),
            make_tuple(sequence(0), sequence(1)),
            make_tuple(sequence(0), sequence(1, 2))
        )
        
        # Should be back to original dimensions
        expected_lengths = [A, B, C]
        assert final_result.get_lengths() == expected_lengths

    def test_complex_realistic_example(self):
        """Test a more realistic complex example"""
        # Parameters similar to realistic examples
        KThreadWrite = 8
        kfold = 2
        KThreadReadPerm = 4
        K0PerThreadWrite = 2
        N1 = 2
        N0 = 8
        npair = 2
        BK1 = 1
        
        # Create input descriptor
        input_desc = make_naive_tensor_descriptor_packed([
            KThreadWrite // kfold // KThreadReadPerm,
            K0PerThreadWrite,
            KThreadReadPerm * N1,
            kfold * N0 // npair,
            npair,
            BK1
        ])
        
        # Create complex transforms
        transforms = make_tuple(
            make_pass_through_transform(number(KThreadWrite // kfold // KThreadReadPerm)),
            make_pass_through_transform(number(K0PerThreadWrite)),
            make_merge_transform(make_tuple(
                number(KThreadReadPerm * N1),
                number(kfold * N0 // npair)
            )),
            make_pass_through_transform(number(npair)),
            make_pass_through_transform(number(BK1))
        )
        
        # Apply transformation
        result = transform_tensor_descriptor(
            input_desc,
            transforms,
            make_tuple(
                sequence(0), sequence(1), sequence(2, 3), sequence(4), sequence(5)
            ),
            make_tuple(
                sequence(0), sequence(1), sequence(2), sequence(3), sequence(4)
            )
        )
        
        # Verify the result has correct structure
        # Should have 6 transforms: original + 5 new
        assert len(result.transforms) == 6
        assert isinstance(result.transforms[0], UnmergeTransform)
        assert isinstance(result.transforms[1], PassThroughTransform)
        assert isinstance(result.transforms[2], PassThroughTransform)
        assert isinstance(result.transforms[3], MergeTransform)
        assert isinstance(result.transforms[4], PassThroughTransform)
        assert isinstance(result.transforms[5], PassThroughTransform)
        
        # Verify dimensions
        expected_lengths = [
            KThreadWrite // kfold // KThreadReadPerm,
            K0PerThreadWrite,
            (KThreadReadPerm * N1) * (kfold * N0 // npair),
            npair,
            BK1
        ]
        assert result.get_lengths() == expected_lengths

    def test_xor_transform_functionality(self):
        """Test XOR transform basic functionality"""
        # Parameters for XOR transform
        dim1, dim2 = 4, 8
        
        # Create input descriptor
        input_desc = make_naive_tensor_descriptor_packed([dim1, dim2])
        
        # Create XOR transform
        transforms = make_tuple(
            make_xor_transform(make_tuple(number(dim1), number(dim2)))
        )
        
        # Apply transformation
        result = transform_tensor_descriptor(
            input_desc,
            transforms,
            make_tuple(sequence(0, 1)),
            make_tuple(sequence(0, 1))
        )
        
        # Verify the result has correct structure
        expected_lengths = [dim1, dim2]
        assert result.get_lengths() == expected_lengths
        
        # Should have 2 transforms: original + new XOR
        assert len(result.transforms) == 2
        assert isinstance(result.transforms[0], UnmergeTransform)
        assert isinstance(result.transforms[1], XorTransform)

    def test_embed_transform_basic(self):
        """Test basic EmbedTransform functionality"""
        # Parameters for EmbedTransform
        lengths = [4, 6]
        strides = [6, 1]
        
        # Create input descriptor with EmbedTransform
        input_desc = make_naive_tensor_descriptor(lengths, strides)
        
        # Verify the descriptor was created correctly
        assert input_desc.get_lengths() == lengths
        
        # Should have 1 transform (EmbedTransform)
        assert len(input_desc.transforms) == 1
        assert isinstance(input_desc.transforms[0], EmbedTransform)
        
        # Verify the EmbedTransform has correct properties
        embed_transform = input_desc.transforms[0]
        assert embed_transform.lengths == lengths
        assert embed_transform.strides == strides


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 