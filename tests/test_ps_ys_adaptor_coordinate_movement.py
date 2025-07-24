"""
Test ps_ys_to_xs_adaptor coordinate movement behavior.

This test file specifically focuses on testing the ps_ys_to_xs_adaptor behavior
for coordinate movement in tile window operations.
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytensor.tile_distribution import make_tile_distribution_encoding, make_static_tile_distribution
from pytensor.tensor_coordinate import (
    make_tensor_adaptor_coordinate, move_tensor_adaptor_coordinate, MultiIndex
)


class TestPsYsAdaptorCoordinateMovement:
    """Test ps_ys_to_xs_adaptor coordinate movement behavior."""
    
    @pytest.fixture
    def adaptor(self):
        """Create a standard test adaptor with 2×2 tile configuration."""
        encoding = make_tile_distribution_encoding(
            rs_lengths=[],  # Empty R (no replication)
            hs_lengthss=[
                [2, 2, 2, 2],  # X0: 2*2*2*2 = 16 elements
                [2, 2, 2, 2]   # X1: 2*2*2*2 = 16 elements
            ],
            ps_to_rhss_major=[[1, 2], [1, 2]],  # P dimensions map to major indices [1,2]
            ps_to_rhss_minor=[[1, 1], [2, 2]],  # P dimensions use minor indices [1,1] and [2,2]
            ys_to_rhs_major=[1, 1, 2, 2],       # Y0,Y1 from X0; Y2,Y3 from X1
            ys_to_rhs_minor=[0, 3, 0, 3]        # Y dimensions use minor indices [0,3,0,3]
        )
        distribution = make_static_tile_distribution(encoding)
        return distribution.ps_ys_to_xs_adaptor
    
    def test_adaptor_dimensions(self, adaptor):
        """Test that adaptor dimensions are correct."""
        assert adaptor.get_num_of_top_dimension() == 6  # P0, P1, Y0, Y1, Y2, Y3
        assert adaptor.get_num_of_bottom_dimension() == 2  # X0, X1
        assert adaptor.get_num_of_hidden_dimension() == 12  # 6 from each X hierarchy (2 P + 4 Y)

    def test_origin_coordinate(self, adaptor):
        """Test coordinate at origin maps correctly."""
        coord = make_tensor_adaptor_coordinate(adaptor, MultiIndex(6, [0, 0, 0, 0, 0, 0]))
        assert coord.get_top_index().to_list() == [0, 0, 0, 0, 0, 0]
        assert coord.get_bottom_index().to_list() == [0, 0]

    def test_y_dimension_movement(self, adaptor):
        """Test that Y dimension movements affect bottom coordinates correctly."""
        # Test each Y dimension independently
        y_movements = [
            ([0, 0, 1, 0, 0, 0], "Y0"),  # Move Y0
            ([0, 0, 0, 1, 0, 0], "Y1"),  # Move Y1
            ([0, 0, 0, 0, 1, 0], "Y2"),  # Move Y2
            ([0, 0, 0, 0, 0, 1], "Y3"),  # Move Y3
        ]

        for movement, y_dim in y_movements:
            # Start at origin
            coord = make_tensor_adaptor_coordinate(adaptor, MultiIndex(6, [0, 0, 0, 0, 0, 0]))
            # Move in Y dimension
            bottom_diff = move_tensor_adaptor_coordinate(adaptor, coord, MultiIndex(6, movement))
            
            # Movement in any Y dimension should affect bottom coordinates
            assert any(diff != 0 for diff in bottom_diff.to_list()), \
                f"Movement in {y_dim} should change bottom coordinates"

    def test_p_dimension_movement(self, adaptor):
        """Test that P dimension movements affect partitioning correctly."""
        # Test each P dimension independently
        p_movements = [
            ([1, 0, 0, 0, 0, 0], "P0"),  # Move P0
            ([0, 1, 0, 0, 0, 0], "P1"),  # Move P1
        ]

        for movement, p_dim in p_movements:
            # Start at origin
            coord = make_tensor_adaptor_coordinate(adaptor, MultiIndex(6, [0, 0, 0, 0, 0, 0]))
            # Get initial bottom position
            initial_bottom = coord.get_bottom_index().to_list()
            # Move in P dimension
            move_tensor_adaptor_coordinate(adaptor, coord, MultiIndex(6, movement))
            # Get new bottom position
            new_bottom = coord.get_bottom_index().to_list()
            
            # P dimension movement should affect partitioning
            assert new_bottom != initial_bottom, \
                f"Movement in {p_dim} should affect partitioning"

    def test_combined_movement(self, adaptor):
        """Test combined P and Y dimension movements."""
        # Test combinations of P and Y movements
        test_cases = [
            ([1, 0, 1, 0, 0, 0], "P0+Y0"),  # Move P0 and Y0
            ([0, 1, 0, 1, 0, 0], "P1+Y1"),  # Move P1 and Y1
            ([1, 1, 0, 0, 1, 1], "P0+P1+Y2+Y3"),  # Move multiple dimensions
        ]

        for movement, description in test_cases:
            # Start at origin
            coord = make_tensor_adaptor_coordinate(adaptor, MultiIndex(6, [0, 0, 0, 0, 0, 0]))
            # Get initial state
            initial_bottom = coord.get_bottom_index().to_list()
            # Apply combined movement
            bottom_diff = move_tensor_adaptor_coordinate(adaptor, coord, MultiIndex(6, movement))
            # Get new state
            new_bottom = coord.get_bottom_index().to_list()
            
            # Combined movements should produce valid coordinate changes
            assert new_bottom != initial_bottom, \
                f"Combined movement {description} should change bottom coordinates"
            assert any(diff != 0 for diff in bottom_diff.to_list()), \
                f"Combined movement {description} should produce non-zero bottom difference"

    def test_boundary_conditions(self, adaptor):
        """Test coordinate movement at boundaries."""
        # Test movements at boundaries
        test_cases = [
            # Start at max position and try to move further
            ([1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0], "Max position + Y0"),
            ([1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0], "Max position + Y1"),
            # Start at origin and try to move back
            ([0, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0], "Origin - Y0"),
            ([0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0], "Origin - Y1"),
        ]

        for start_pos, movement, description in test_cases:
            coord = make_tensor_adaptor_coordinate(adaptor, MultiIndex(6, start_pos))
            # Get initial position
            initial_bottom = coord.get_bottom_index().to_list()
            # Apply movement
            bottom_diff = move_tensor_adaptor_coordinate(adaptor, coord, MultiIndex(6, movement))
            # Get new position
            new_bottom = coord.get_bottom_index().to_list()
            
            # Verify that the bottom difference matches the actual change
            for i in range(len(bottom_diff)):
                assert new_bottom[i] - initial_bottom[i] == bottom_diff[i], \
                    f"Movement {description} produced inconsistent bottom difference"
            
            # Verify that the coordinate movement is valid
            # For Y dimensions, movement should affect bottom coordinates
            # even at boundaries, as long as the transform allows it
            if any(movement[2:]):  # Y dimension movement
                assert bottom_diff.to_list() != [0] * len(bottom_diff), \
                    f"Movement {description} should affect bottom coordinates" 