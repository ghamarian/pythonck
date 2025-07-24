
import pytest
from pytensor.space_filling_curve import SpaceFillingCurve

class TestSpaceFillingCurve:

    def test_initialization(self):
        """Test basic initialization of the space-filling curve."""
        sfc = SpaceFillingCurve(
            tensor_lengths=[10, 20],
            dim_access_order=[0, 1],
            scalars_per_access=[1, 1],
            snake_curved=False
        )
        assert sfc.tensor_size == 200
        assert sfc.ndim == 2
        assert sfc.scalar_per_vector == 1

    def test_snake_curve(self):
        """Test snake curve behavior."""
        sfc = SpaceFillingCurve(
            tensor_lengths=[4, 4],
            dim_access_order=[0, 1],
            scalars_per_access=[1, 1],
            snake_curved=True
        )
        assert sfc.snake_curved

    def test_vectorized_access_lengths(self):
        """Test access length calculation for vectorized dimensions."""
        sfc = SpaceFillingCurve(
            tensor_lengths=[10, 8],
            dim_access_order=[0, 1],
            scalars_per_access=[1, 4], # Vector of 4 on second dim
            snake_curved=False
        )
        # For dim 0 (length 10, scalar access), access_length should be 10
        # For dim 1 (length 8, vector 4), access_length should be ceil(8/4) = 2
        assert sfc.access_lengths == [10, 2]
        assert sfc.ordered_access_lengths == [10, 2]

    def test_invalid_initialization(self):
        """Test that initialization with empty tensor raises ValueError."""
        with pytest.raises(ValueError):
            SpaceFillingCurve(
                tensor_lengths=[10, 0],
                dim_access_order=[0, 1],
                scalars_per_access=[1, 1]
            )

    def test_get_index_simple(self):
        """Test index calculation for different access positions."""
        sfc = SpaceFillingCurve(
            tensor_lengths=[2, 2],
            dim_access_order=[0, 1],
            scalars_per_access=[1, 1]
        )
        assert sfc.get_index(0) == [0, 0]
        assert sfc.get_index(1) == [0, 1]
        assert sfc.get_index(2) == [1, 0]
        assert sfc.get_index(3) == [1, 1]
        
    def test_get_step_between(self):
        """Test step calculation between access positions."""
        sfc = SpaceFillingCurve(
            tensor_lengths=[2, 2],
            dim_access_order=[0, 1],
            scalars_per_access=[1, 1]
        )
        # Step from [0,0] to [0,1]
        assert sfc.get_step_between(0, 1) == [0, 1]
        # Step from [0,1] to [1,0]
        assert sfc.get_step_between(1, 2) == [1, -1]
        # Step from [1,0] to [1,1]
        assert sfc.get_step_between(2, 3) == [0, 1] 