
import numpy as np
from typing import List

from pytensor.tensor_coordinate import (
    move_tensor_adaptor_coordinate,
    make_tensor_adaptor_coordinate,
    TensorAdaptorCoordinate
)
from pytensor.tensor_descriptor import (
    MergeTransform, UnmergeTransform, PadTransform, EmbedTransform
)
from pytensor.tensor_adaptor import make_single_stage_tensor_adaptor


class TestTensorAdaptorCoordinateMovement:

    def test_move_with_merge_transform(self):
        """Test coordinate movement with a MergeTransform."""
        transform = MergeTransform([4, 8])  # 4×8 grid merged to linear
        adaptor = make_single_stage_tensor_adaptor([transform], [[0, 1]], [[0]])
        
        # Start at position 10 (1×8 + 2) -> (1,2)
        initial_coord = make_tensor_adaptor_coordinate(adaptor, [10])
        # Move -3 steps to position 7 (0×8 + 7) -> (0,7)
        move_tensor_adaptor_coordinate(adaptor, initial_coord, [-3])
        
        assert initial_coord.get_top_index().to_list() == [7]
        # Bottom indices are the unmerged coordinates
        assert initial_coord.get_bottom_index().to_list() == [0, 7]

    def test_move_with_unmerge_transform(self):
        """Test coordinate movement with an UnmergeTransform."""
        transform = UnmergeTransform([6, 6])  # Linear to 6×6 grid
        adaptor = make_single_stage_tensor_adaptor([transform], [[0]], [[0, 1]])
        
        # Start at (2,3)
        initial_coord = make_tensor_adaptor_coordinate(adaptor, [2, 3])
        # Move to (3,2)
        move_tensor_adaptor_coordinate(adaptor, initial_coord, [1, -1])
        
        assert initial_coord.get_top_index().to_list() == [3, 2]
        # Bottom index is the merged linear position
        # (3,2) -> 3×6 + 2 = 20
        assert initial_coord.get_bottom_index().to_list() == [20]

    def test_move_with_pad_transform(self):
        """Test coordinate movement with a PadTransform."""
        transform = PadTransform(20, 5, 5)  # 20 elements with 5 padding each side
        adaptor = make_single_stage_tensor_adaptor([transform], [[0]], [[0]])
        
        # Start at position 15 (10 in original space)
        initial_coord = make_tensor_adaptor_coordinate(adaptor, [15])
        # Move 5 steps to position 20 (15 in original space)
        move_tensor_adaptor_coordinate(adaptor, initial_coord, [5])
        
        assert initial_coord.get_top_index().to_list() == [20]
        # Bottom index is in the original unpadded space
        assert initial_coord.get_bottom_index().to_list() == [15]
        
    def test_move_with_embed_transform(self):
        """Test coordinate movement with an EmbedTransform."""
        # EmbedTransform maps a single lower dimension to multiple upper dimensions
        # Here we map a linear index to a 10×20 grid with strides [20,1]
        transform = EmbedTransform([10, 20], [20, 1])
        adaptor = make_single_stage_tensor_adaptor([transform], [[0]], [[0, 1]])

        # Start at position [0,5] in the 10×20 grid (maps to linear position 5)
        initial_coord = make_tensor_adaptor_coordinate(adaptor, [0, 5])
        # Move to position [0,3] in the grid (maps to linear position 3)
        move_tensor_adaptor_coordinate(adaptor, initial_coord, [0, -2])
        
        assert initial_coord.get_top_index().to_list() == [0, 3]
        # Bottom index is the linear position
        assert initial_coord.get_bottom_index().to_list() == [3] 