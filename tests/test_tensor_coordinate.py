"""
Tests for tensor_coordinate module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.tensor_coordinate import (
    MultiIndex, TensorAdaptorCoordinate, TensorCoordinate,
    make_tensor_coordinate, make_tensor_adaptor_coordinate,
    move_tensor_coordinate, move_tensor_adaptor_coordinate,
    coordinate_has_valid_offset, coordinate_has_valid_offset_assuming_top_index_is_valid,
    adaptor_coordinate_is_valid, adaptor_coordinate_is_valid_assuming_top_index_is_valid
)
from pytensor.tensor_descriptor import (
    EmbedTransform, MergeTransform, UnmergeTransform, PadTransform
)
from pytensor.tensor_adaptor import make_single_stage_tensor_adaptor


class TestMultiIndex:
    """Test cases for MultiIndex class."""
    
    def test_multi_index_creation(self):
        """Test basic multi-index creation."""
        # Default initialization
        idx = MultiIndex(3)
        assert len(idx) == 3
        assert idx[0] == 0
        assert idx[1] == 0
        assert idx[2] == 0
        
        # With initial values
        idx2 = MultiIndex(4, [1, 2, 3, 4])
        assert len(idx2) == 4
        assert idx2[0] == 1
        assert idx2[1] == 2
        assert idx2[2] == 3
        assert idx2[3] == 4
    
    def test_multi_index_invalid_creation(self):
        """Test invalid multi-index creation."""
        with pytest.raises(ValueError, match="Values length"):
            MultiIndex(3, [1, 2])  # Wrong length
    
    def test_multi_index_access(self):
        """Test multi-index element access."""
        idx = MultiIndex(3, [10, 20, 30])
        
        # Get
        assert idx[0] == 10
        assert idx[1] == 20
        assert idx[2] == 30
        
        # Set
        idx[1] = 25
        assert idx[1] == 25
    
    def test_multi_index_equality(self):
        """Test multi-index equality comparison."""
        idx1 = MultiIndex(3, [1, 2, 3])
        idx2 = MultiIndex(3, [1, 2, 3])
        idx3 = MultiIndex(3, [1, 2, 4])
        idx4 = MultiIndex(2, [1, 2])
        
        assert idx1 == idx2
        assert idx1 != idx3
        assert idx1 != idx4
        assert idx1 != "not a multi-index"
    
    def test_multi_index_copy(self):
        """Test multi-index copying."""
        idx1 = MultiIndex(3, [1, 2, 3])
        idx2 = idx1.copy()
        
        # Should be equal but different objects
        assert idx1 == idx2
        assert idx1 is not idx2
        
        # Modifying copy shouldn't affect original
        idx2[0] = 10
        assert idx1[0] == 1
        assert idx2[0] == 10
    
    def test_multi_index_conversions(self):
        """Test multi-index conversions."""
        idx = MultiIndex(3, [1, 2, 3])
        
        # To list
        lst = idx.to_list()
        assert lst == [1, 2, 3]
        assert isinstance(lst, list)
        
        # To numpy array
        arr = idx.to_array()
        assert np.array_equal(arr, np.array([1, 2, 3]))
        assert isinstance(arr, np.ndarray)
    
    def test_multi_index_repr(self):
        """Test multi-index string representation."""
        idx = MultiIndex(3, [1, 2, 3])
        assert repr(idx) == "MultiIndex([1, 2, 3])"


class TestTensorAdaptorCoordinate:
    """Test cases for TensorAdaptorCoordinate class."""
    
    def test_tensor_adaptor_coordinate_creation(self):
        """Test basic tensor adaptor coordinate creation."""
        idx_hidden = MultiIndex(5, [0, 1, 2, 3, 4])
        coord = TensorAdaptorCoordinate(
            ndim_hidden=5,
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[3, 4],
            idx_hidden=idx_hidden
        )
        
        assert coord.ndim_hidden == 5
        assert coord.bottom_dimension_hidden_ids == [0, 1]
        assert coord.top_dimension_hidden_ids == [3, 4]
        assert coord.idx_hidden == idx_hidden
    
    def test_tensor_adaptor_coordinate_invalid_creation(self):
        """Test invalid tensor adaptor coordinate creation."""
        idx_hidden = MultiIndex(3, [0, 1, 2])
        
        with pytest.raises(ValueError, match="Hidden index size"):
            TensorAdaptorCoordinate(
                ndim_hidden=5,  # Mismatch with idx_hidden size
                bottom_dimension_hidden_ids=[0],
                top_dimension_hidden_ids=[1, 2],
                idx_hidden=idx_hidden
            )
    
    def test_get_top_index(self):
        """Test getting top index from adaptor coordinate."""
        idx_hidden = MultiIndex(5, [10, 20, 30, 40, 50])
        coord = TensorAdaptorCoordinate(
            ndim_hidden=5,
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 4],  # Indices 2 and 4
            idx_hidden=idx_hidden
        )
        
        top_idx = coord.get_top_index()
        assert len(top_idx) == 2
        assert top_idx[0] == 30  # Hidden index 2
        assert top_idx[1] == 50  # Hidden index 4
    
    def test_get_bottom_index(self):
        """Test getting bottom index from adaptor coordinate."""
        idx_hidden = MultiIndex(5, [10, 20, 30, 40, 50])
        coord = TensorAdaptorCoordinate(
            ndim_hidden=5,
            bottom_dimension_hidden_ids=[1, 3],  # Indices 1 and 3
            top_dimension_hidden_ids=[2, 4],
            idx_hidden=idx_hidden
        )
        
        bottom_idx = coord.get_bottom_index()
        assert len(bottom_idx) == 2
        assert bottom_idx[0] == 20  # Hidden index 1
        assert bottom_idx[1] == 40  # Hidden index 3
    
    def test_get_set_hidden_index(self):
        """Test getting and setting hidden index."""
        idx_hidden = MultiIndex(3, [1, 2, 3])
        coord = TensorAdaptorCoordinate(
            ndim_hidden=3,
            bottom_dimension_hidden_ids=[0],
            top_dimension_hidden_ids=[1, 2],
            idx_hidden=idx_hidden
        )
        
        # Get
        assert coord.get_hidden_index() == idx_hidden
        
        # Set valid
        new_idx = MultiIndex(3, [4, 5, 6])
        coord.set_hidden_index(new_idx)
        assert coord.get_hidden_index() == new_idx
        
        # Set invalid
        with pytest.raises(ValueError, match="Index size"):
            coord.set_hidden_index(MultiIndex(2, [1, 2]))


class TestTensorCoordinate:
    """Test cases for TensorCoordinate class."""
    
    def test_tensor_coordinate_creation(self):
        """Test basic tensor coordinate creation."""
        coord = TensorCoordinate(
            ndim_hidden=3,
            top_dimension_hidden_ids=[1, 2]
        )
        
        assert coord.ndim_hidden == 3
        assert coord.bottom_dimension_hidden_ids == [0]  # Always [0]
        assert coord.top_dimension_hidden_ids == [1, 2]
        assert len(coord.idx_hidden) == 3
    
    def test_tensor_coordinate_with_initial_index(self):
        """Test tensor coordinate with initial hidden index."""
        idx_hidden = MultiIndex(4, [10, 20, 30, 40])
        coord = TensorCoordinate(
            ndim_hidden=4,
            top_dimension_hidden_ids=[1, 2, 3],
            idx_hidden=idx_hidden
        )
        
        assert coord.idx_hidden == idx_hidden
    
    def test_get_index(self):
        """Test getting tensor index (same as top index)."""
        idx_hidden = MultiIndex(4, [10, 20, 30, 40])
        coord = TensorCoordinate(
            ndim_hidden=4,
            top_dimension_hidden_ids=[1, 3],  # Indices 1 and 3
            idx_hidden=idx_hidden
        )
        
        idx = coord.get_index()
        assert len(idx) == 2
        assert idx[0] == 20  # Hidden index 1
        assert idx[1] == 40  # Hidden index 3
    
    def test_get_offset(self):
        """Test getting linear offset (bottom index[0])."""
        idx_hidden = MultiIndex(3, [100, 20, 30])
        coord = TensorCoordinate(
            ndim_hidden=3,
            top_dimension_hidden_ids=[1, 2],
            idx_hidden=idx_hidden
        )
        
        # Bottom dimension is always [0], so offset is idx_hidden[0]
        assert coord.get_offset() == 100
    
    def test_from_adaptor_coordinate(self):
        """Test creating tensor coordinate from adaptor coordinate."""
        idx_hidden = MultiIndex(4, [10, 20, 30, 40])
        adaptor_coord = TensorAdaptorCoordinate(
            ndim_hidden=4,
            bottom_dimension_hidden_ids=[0, 1],
            top_dimension_hidden_ids=[2, 3],
            idx_hidden=idx_hidden
        )
        
        tensor_coord = TensorCoordinate.from_adaptor_coordinate(adaptor_coord)
        
        assert tensor_coord.ndim_hidden == 4
        assert tensor_coord.top_dimension_hidden_ids == [2, 3]
        assert tensor_coord.bottom_dimension_hidden_ids == [0]  # Always [0]
        assert tensor_coord.idx_hidden == idx_hidden


# Mock classes for testing coordinate creation and movement
class MockTransform:
    """Mock transform for testing."""
    
    def __init__(self, always_valid=True):
        self.always_valid = always_valid
    
    def calculate_lower_index(self, idx_up):
        """Simple mock: lower = upper * 2."""
        return MultiIndex(len(idx_up), [val * 2 for val in idx_up.to_list()])
    
    def is_valid_upper_index_always_mapped_to_valid_lower_index(self):
        """Check if transform always maps valid to valid."""
        return self.always_valid
    
    def is_valid_upper_index_mapped_to_valid_lower_index(self, idx_up):
        """Check if specific upper index maps to valid lower."""
        # Mock: invalid if any index is negative
        return all(val >= 0 for val in idx_up.to_list())


class MockTensorAdaptor:
    """Mock tensor adaptor for testing."""
    
    def __init__(self):
        self.transforms = [MockTransform()]
        self.lower_dimension_hidden_idss = [[0]]  # Transform 0: lower dim is hidden 0
        self.upper_dimension_hidden_idss = [[1]]  # Transform 0: upper dim is hidden 1
        self.bottom_dimension_hidden_ids = [0]
        self.top_dimension_hidden_ids = [1]
        self.top_lengths = [10]  # Top dimension has length 10
    
    def get_num_of_transform(self):
        return len(self.transforms)
    
    def get_num_of_hidden_dimension(self):
        return 2  # Simple case: 2 hidden dimensions
    
    def get_num_of_top_dimension(self):
        return len(self.top_dimension_hidden_ids)
    
    def get_num_of_bottom_dimension(self):
        return len(self.bottom_dimension_hidden_ids)
    
    def get_transforms(self):
        return self.transforms
    
    def get_lower_dimension_hidden_idss(self):
        return self.lower_dimension_hidden_idss
    
    def get_upper_dimension_hidden_idss(self):
        return self.upper_dimension_hidden_idss
    
    def get_bottom_dimension_hidden_ids(self):
        return self.bottom_dimension_hidden_ids
    
    def get_top_dimension_hidden_ids(self):
        return self.top_dimension_hidden_ids
    
    def get_length(self, dim):
        return self.top_lengths[dim]


class TestCoordinateCreation:
    """Test coordinate creation functions."""
    
    def test_make_tensor_adaptor_coordinate(self):
        """Test creating tensor adaptor coordinate."""
        adaptor = MockTensorAdaptor()
        idx_top = MultiIndex(1, [5])
        
        coord = make_tensor_adaptor_coordinate(adaptor, idx_top)
        
        assert coord.ndim_hidden == 2
        assert coord.bottom_dimension_hidden_ids == [0]
        assert coord.top_dimension_hidden_ids == [1]
        
        # Check hidden index
        # Top dimension 1 should have value 5
        assert coord.idx_hidden[1] == 5
        # Bottom dimension 0 should have value 10 (5 * 2 from mock transform)
        assert coord.idx_hidden[0] == 10
    
    def test_make_tensor_coordinate(self):
        """Test creating tensor coordinate."""
        adaptor = MockTensorAdaptor()
        
        # Test with list
        coord1 = make_tensor_coordinate(adaptor, [3])
        assert coord1.get_index()[0] == 3
        assert coord1.get_offset() == 6  # 3 * 2 from mock transform
        
        # Test with MultiIndex
        idx_top = MultiIndex(1, [4])
        coord2 = make_tensor_coordinate(adaptor, idx_top)
        assert coord2.get_index()[0] == 4
        assert coord2.get_offset() == 8  # 4 * 2 from mock transform


class TestCoordinateMovement:
    """Test coordinate movement functions."""
    
    def test_move_tensor_adaptor_coordinate(self):
        """Test moving tensor adaptor coordinate."""
        adaptor = MockTensorAdaptor()
        
        # Create initial coordinate
        idx_top = MultiIndex(1, [2])
        coord = make_tensor_adaptor_coordinate(adaptor, idx_top)
        
        # Initial state
        assert coord.idx_hidden[1] == 2  # Top
        assert coord.idx_hidden[0] == 4  # Bottom (2 * 2)
        
        # Move by [3]
        idx_diff = MultiIndex(1, [3])
        idx_diff_bottom = move_tensor_adaptor_coordinate(adaptor, coord, idx_diff)
        
        # After move
        assert coord.idx_hidden[1] == 5  # 2 + 3
        assert coord.idx_hidden[0] == 10  # Updated by transform
        
        # Bottom difference
        assert idx_diff_bottom[0] == 6  # 3 * 2 from mock transform
    
    def test_move_tensor_coordinate(self):
        """Test moving tensor coordinate."""
        adaptor = MockTensorAdaptor()
        
        # Create initial coordinate
        coord = make_tensor_coordinate(adaptor, [1])
        
        # Initial state
        assert coord.get_index()[0] == 1
        assert coord.get_offset() == 2  # 1 * 2
        
        # Move with list
        move_tensor_coordinate(adaptor, coord, [2])
        assert coord.get_index()[0] == 3  # 1 + 2
        assert coord.get_offset() == 6  # Updated
        
        # Move with MultiIndex
        move_tensor_coordinate(adaptor, coord, MultiIndex(1, [1]))
        assert coord.get_index()[0] == 4  # 3 + 1
        assert coord.get_offset() == 8  # Updated

    def test_embed_transform_movement(self):
        """Test moving coordinates with EmbedTransform."""
        # Create a 2D to 1D embedding transform
        transform = UnmergeTransform([4, 3])  # 4x3 2D space
        adaptor = make_single_stage_tensor_adaptor(
            [transform],
            [[0]],  # Bottom dimension
            [[0, 1]]  # Top dimensions
        )
        
        # Create initial coordinate at (1,1)
        coord = make_tensor_coordinate(adaptor, [1, 1])
        assert coord.get_offset() == 4  # 1 * 3 + 1
        
        # Move in first dimension
        move_tensor_coordinate(adaptor, coord, [1, 0])
        assert coord.get_offset() == 7  # 2 * 3 + 1
        
        # Move in second dimension
        move_tensor_coordinate(adaptor, coord, [0, 1])
        assert coord.get_offset() == 8  # 2 * 3 + 2

    def test_merge_transform_movement(self):
        """Test moving coordinates with MergeTransform using TensorAdaptorCoordinate."""
        # Create a merge transform combining two dimensions
        transform = MergeTransform([3, 4])  # Merge 3x4 into 12
        adaptor = make_single_stage_tensor_adaptor(
            [transform],
            [[0, 1]],  # LowerDimensionOldTopIdss - two dimensions to merge
            [[0]]  # UpperDimensionNewTopIdss - one dimension after merge
        )
        
        # Create initial coordinate at [1, 2] in the original 3x4 space
        # Use TensorAdaptorCoordinate directly since we need multiple bottom dimensions
        # coord = make_tensor_adaptor_coordinate(adaptor, MultiIndex(1, [6]))  # 6 in merged space
        coord = make_tensor_adaptor_coordinate(adaptor, [6])  # 6 in merged space
        
        # Check that the bottom index correctly represents [1, 2] in original space
        bottom_idx = coord.get_bottom_index()
        assert bottom_idx.to_list() == [1, 2]  # Should be [1, 2] since 6 = 1 * 4 + 2
        
        # Move in merged space
        move_tensor_adaptor_coordinate(adaptor, coord,  [1])
        bottom_idx = coord.get_bottom_index()
        assert bottom_idx.to_list() == [1, 3]  # Should be [1, 3] since 7 = 1 * 4 + 3
        
        # Move again to cross dimension boundary
        move_tensor_adaptor_coordinate(adaptor, coord, MultiIndex(1, [1]))
        bottom_idx = coord.get_bottom_index()
        assert bottom_idx.to_list() == [2, 0]  # Should be [2, 0] since 8 = 2 * 4 + 0

    def test_chain_of_real_transforms(self):
        """Test moving coordinates through a chain of real transforms."""
        # Create a chain that goes: 1D -> 2D -> 1D
        # 1. UnmergeTransform: Split 1D into 2D (12 -> 3x4)  
        # 2. MergeTransform: Merge 2D back to 1D (3x4 -> 12)
        transforms = [
            UnmergeTransform([3, 4]),  # Split 12 into 3x4
            MergeTransform([3, 4])     # Merge 3x4 back to 12
        ]
        
        adaptor = make_single_stage_tensor_adaptor(
            transforms,
            [[0], [0, 1]],     # Lower dimensions for each transform
            [[0, 1], [0]]      # Upper dimensions for each transform  
        )
        
        # The final result has 2 top dimensions from the UnmergeTransform
        # Create initial coordinate at (1, 2) in the 3x4 space
        coord = make_tensor_coordinate(adaptor, [1, 2])
        initial_offset = coord.get_offset()
        assert initial_offset == 6  # 1 * 4 + 2 = 6
        
        # Move in first dimension
        move_tensor_coordinate(adaptor, coord, [1, 0])
        assert coord.get_offset() == 10  # 2 * 4 + 2 = 10
        
        # Move in second dimension  
        move_tensor_coordinate(adaptor, coord, [0, 1])
        assert coord.get_offset() == 11  # 2 * 4 + 3 = 11

    def test_pad_transform_movement(self):
        """Test moving coordinates with PadTransform."""
        # Create a pad transform: lower_length=2, left_pad=1, right_pad=0
        # This means: original space [0,1] -> padded space [0,1,2] (length 3)
        transform = PadTransform(2, 1, 0)
        adaptor = make_single_stage_tensor_adaptor(
            [transform],
            [[0]],  # Lower dimension (input to transform)
            [[0]]   # Upper dimension (output from transform)
        )
        
        # Test that the transform works correctly by checking hidden indices
        # Create coordinate at padded index 1
        coord = make_tensor_coordinate(adaptor, [1])
        
        # Check that the transform correctly maps padded index 1 to original index 0
        # (1 - 1 left_pad = 0, clamped to [0,1])
        hidden_idx = coord.get_hidden_index()
        assert hidden_idx[0] == 0  # Should be 0 after transform
        assert hidden_idx[1] == 1  # Top index should be 1
        
        # Move to padded index 2
        move_tensor_coordinate(adaptor, coord, [1])
        hidden_idx = coord.get_hidden_index()
        assert hidden_idx[0] == 1  # Should be 1 after transform (2 - 1 = 1)
        assert hidden_idx[1] == 2  # Top index should be 2
        
        # Move to padded index 3 (out of bounds)
        move_tensor_coordinate(adaptor, coord, [1])
        hidden_idx = coord.get_hidden_index()
        assert hidden_idx[0] == 1  # Should be clamped to 1 (3 - 1 = 2, clamped to max 1)
        assert hidden_idx[1] == 3  # Top index should be 3


class TestCoordinateValidation:
    """Test coordinate validation functions."""
    
    def test_adaptor_coordinate_is_valid(self):
        """Test full coordinate validation."""
        adaptor = MockTensorAdaptor()
        
        # Valid coordinate
        coord1 = make_tensor_adaptor_coordinate(adaptor, MultiIndex(1, [5]))
        assert adaptor_coordinate_is_valid(adaptor, coord1) == True
        
        # Invalid: out of bounds
        coord2 = make_tensor_adaptor_coordinate(adaptor, MultiIndex(1, [5]))
        coord2.idx_hidden[1] = 15  # Exceeds length 10
        assert adaptor_coordinate_is_valid(adaptor, coord2) == False
        
        # Invalid: negative index
        coord3 = make_tensor_adaptor_coordinate(adaptor, MultiIndex(1, [5]))
        coord3.idx_hidden[1] = -1
        assert adaptor_coordinate_is_valid(adaptor, coord3) == False
    
    def test_adaptor_coordinate_is_valid_assuming_top_valid(self):
        """Test coordinate validation assuming top is valid."""
        # Create adaptor with transform that can be invalid
        adaptor = MockTensorAdaptor()
        adaptor.transforms = [MockTransform(always_valid=False)]
        
        # Valid coordinate
        coord1 = make_tensor_adaptor_coordinate(adaptor, MultiIndex(1, [5]))
        assert adaptor_coordinate_is_valid_assuming_top_index_is_valid(adaptor, coord1) == True
        
        # Invalid: negative in upper dimension
        coord2 = make_tensor_adaptor_coordinate(adaptor, MultiIndex(1, [5]))
        coord2.idx_hidden[1] = -1  # Makes transform invalid
        assert adaptor_coordinate_is_valid_assuming_top_index_is_valid(adaptor, coord2) == False
    
    def test_coordinate_has_valid_offset(self):
        """Test tensor coordinate offset validation."""
        adaptor = MockTensorAdaptor()
        
        # Valid
        coord1 = make_tensor_coordinate(adaptor, [5])
        assert coordinate_has_valid_offset(adaptor, coord1) == True
        
        # Invalid
        coord2 = make_tensor_coordinate(adaptor, [5])
        coord2.idx_hidden[1] = 20  # Out of bounds
        assert coordinate_has_valid_offset(adaptor, coord2) == False
    
    def test_coordinate_has_valid_offset_assuming_top_valid(self):
        """Test tensor coordinate offset validation assuming top valid."""
        adaptor = MockTensorAdaptor()
        adaptor.transforms = [MockTransform(always_valid=False)]
        
        coord = make_tensor_coordinate(adaptor, [5])
        assert coordinate_has_valid_offset_assuming_top_index_is_valid(adaptor, coord) == True
        
        # Make transform invalid
        coord.idx_hidden[1] = -1
        assert coordinate_has_valid_offset_assuming_top_index_is_valid(adaptor, coord) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 