"""
Tests for tensor_view module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.tensor_view import (
    TensorView, NullTensorView,
    make_tensor_view, make_naive_tensor_view, make_naive_tensor_view_packed,
    transform_tensor_view
)
from pytensor.buffer_view import AddressSpaceEnum, MemoryOperationEnum
from pytensor.tensor_descriptor import (
    make_naive_tensor_descriptor, make_naive_tensor_descriptor_packed,
    EmbedTransform, OffsetTransform
)
from pytensor.tensor_coordinate import make_tensor_coordinate, move_tensor_coordinate


class TestTensorView:
    """Test cases for TensorView class."""
    
    def test_tensor_view_creation(self):
        """Test basic tensor view creation."""
        # Create data and descriptor
        data = np.arange(12, dtype=np.float32)
        desc = make_naive_tensor_descriptor([3, 4], [4, 1])
        
        # Create tensor view
        view = make_tensor_view(data, desc)
        
        assert view.get_num_of_dimension() == 2
        assert view.get_tensor_descriptor() == desc
        assert view.dst_in_mem_op == MemoryOperationEnum.SET
    
    def test_tensor_view_invalid_creation(self):
        """Test invalid tensor view creation."""
        data = np.arange(12, dtype=np.float32)
        desc = make_naive_tensor_descriptor([3, 4], [4, 1])
        
        # Test with None buffer view
        with pytest.raises(ValueError, match="Buffer view cannot be None"):
            TensorView(buffer_view=None, tensor_desc=desc)
        
        # Test with None tensor descriptor
        from pytensor.buffer_view import make_buffer_view
        buf = make_buffer_view(data, 12)
        with pytest.raises(ValueError, match="Tensor descriptor cannot be None"):
            TensorView(buffer_view=buf, tensor_desc=None)
    
    def test_get_element(self):
        """Test getting single elements."""
        # Create 3x4 matrix
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        flat_data = data.flatten()
        
        view = make_naive_tensor_view(flat_data, [3, 4], [4, 1])
        
        # Test getting elements
        assert view.get_element([0, 0]) == 0.0
        assert view.get_element([0, 3]) == 3.0
        assert view.get_element([1, 2]) == 6.0
        assert view.get_element([2, 1]) == 9.0
    
    def test_set_element(self):
        """Test setting single elements."""
        data = np.zeros(12, dtype=np.float32)
        view = make_naive_tensor_view(data, [3, 4], [4, 1])
        
        # Set some elements
        view.set_element([0, 0], 10.0)
        view.set_element([1, 2], 20.0)
        view.set_element([2, 3], 30.0)
        
        # Check values
        assert data[0] == 10.0
        assert data[6] == 20.0
        assert data[11] == 30.0
    
    def test_array_indexing(self):
        """Test array-style indexing with __getitem__ and __setitem__."""
        data = np.arange(12, dtype=np.float32)
        view = make_naive_tensor_view(data, [3, 4], [4, 1])
        
        # Test __getitem__
        assert view[0, 0] == 0.0
        assert view[1, 1] == 5.0
        assert view[2, 3] == 11.0
        
        # Test __setitem__
        view[0, 1] = 100.0
        view[2, 0] = 200.0
        
        assert data[1] == 100.0
        assert data[8] == 200.0
    
    def test_get_vectorized_elements(self):
        """Test getting vectorized elements."""
        data = np.arange(16, dtype=np.float32)
        view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create coordinate
        coord = make_tensor_coordinate(view.tensor_desc, [1, 0])
        
        # Get single element
        val = view.get_vectorized_elements(coord, 0, 1)
        assert val == 4.0
        
        # Get vector of 4 elements
        vec = view.get_vectorized_elements(coord, 0, 4)
        assert np.array_equal(vec, [4.0, 5.0, 6.0, 7.0])
    
    def test_set_vectorized_elements(self):
        """Test setting vectorized elements."""
        data = np.zeros(16, dtype=np.float32)
        view = make_naive_tensor_view(data, [4, 4], [4, 1])
        
        # Create coordinate
        coord = make_tensor_coordinate(view.tensor_desc, [2, 0])
        
        # Set vector of elements
        vec = np.array([10.0, 11.0, 12.0, 13.0])
        view.set_vectorized_elements(coord, vec, 0, 4)
        
        # Check values
        assert np.array_equal(data[8:12], vec)
    
    def test_update_vectorized_elements(self):
        """Test updating vectorized elements."""
        data = np.ones(16, dtype=np.float32) * 5.0
        view = make_naive_tensor_view(data, [4, 4], [4, 1], 
                                     dst_in_mem_op=MemoryOperationEnum.ADD)
        
        # Create coordinate
        coord = make_tensor_coordinate(view.tensor_desc, [1, 0])
        
        # Update with ADD operation
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        view.update_vectorized_elements(coord, vec, 0, 4)
        
        # Check values (should be original + vec)
        assert np.array_equal(data[4:8], [6.0, 7.0, 8.0, 9.0])
    
    def test_linear_offset(self):
        """Test using linear offset parameter."""
        data = np.arange(20, dtype=np.float32)
        view = make_naive_tensor_view(data, [4, 5], [5, 1])
        
        # Create coordinate at [1, 0]
        coord = make_tensor_coordinate(view.tensor_desc, [1, 0])
        
        # Get element with linear offset
        val = view.get_vectorized_elements(coord, linear_offset=2)
        assert val == 7.0  # [1,0] is offset 5, plus 2 = 7
        
        # Set element with linear offset
        view.set_vectorized_elements(coord, 100.0, linear_offset=3)
        assert data[8] == 100.0  # [1,0] is offset 5, plus 3 = 8


class TestFactoryFunctions:
    """Test cases for tensor view factory functions."""
    
    def test_make_naive_tensor_view(self):
        """Test creating naive tensor view."""
        data = np.arange(12, dtype=np.float32)
        view = make_naive_tensor_view(data, [3, 4], [4, 1])
        
        assert view.get_num_of_dimension() == 2
        assert view[0, 0] == 0.0
        assert view[2, 3] == 11.0
    
    def test_make_naive_tensor_view_with_options(self):
        """Test creating tensor view with various options."""
        data = np.zeros(12, dtype=np.float32)
        
        view = make_naive_tensor_view(
            data, [3, 4], [4, 1],
            address_space=AddressSpaceEnum.LDS,
            dst_in_mem_op=MemoryOperationEnum.ATOMIC_ADD,
            guaranteed_last_dim_vector_length=4,
            guaranteed_last_dim_vector_stride=1
        )
        
        assert view.buffer_view.address_space == AddressSpaceEnum.LDS
        assert view.dst_in_mem_op == MemoryOperationEnum.ATOMIC_ADD
    
    def test_make_naive_tensor_view_packed(self):
        """Test creating packed tensor view."""
        data = np.arange(24, dtype=np.float32)
        view = make_naive_tensor_view_packed(data, [2, 3, 4])
        
        assert view.get_num_of_dimension() == 3
        
        # Test packed layout
        assert view[0, 0, 0] == 0.0
        assert view[0, 0, 1] == 1.0
        assert view[0, 1, 0] == 4.0
        assert view[1, 0, 0] == 12.0


class TestTensorViewWithCoordinates:
    """Test tensor view operations with coordinates."""
    
    def test_coordinate_movement(self):
        """Test using coordinates with tensor view."""
        data = np.arange(20, dtype=np.float32)
        view = make_naive_tensor_view(data, [4, 5], [5, 1])
        
        # Create and move coordinate
        coord = make_tensor_coordinate(view.tensor_desc, [0, 0])
        
        # Check initial position
        assert view.get_element(coord) == 0.0
        
        # Move coordinate
        move_tensor_coordinate(view.tensor_desc, coord, [1, 2])
        assert view.get_element(coord) == 7.0  # [1,2]
        
        # Move again
        move_tensor_coordinate(view.tensor_desc, coord, [1, 1])
        assert view.get_element(coord) == 13.0  # [2,3]
    
    def test_invalid_coordinate_access(self):
        """Test accessing with invalid coordinates."""
        data = np.arange(12, dtype=np.float32)
        view = make_naive_tensor_view(data, [3, 4], [4, 1])
        
        # Create out-of-bounds coordinate
        coord = make_tensor_coordinate(view.tensor_desc, [2, 3])
        
        # Move it out of bounds
        move_tensor_coordinate(view.tensor_desc, coord, [1, 0])
        
        # This should handle gracefully (return 0 or skip)
        val = view.get_vectorized_elements(coord, 0, 1, oob_conditional_check=True)
        # The exact behavior depends on buffer view implementation


class TestNullTensorView:
    """Test NullTensorView placeholder."""
    
    def test_null_tensor_view(self):
        """Test that NullTensorView can be instantiated."""
        null_view = NullTensorView()
        assert null_view is not None


class TestTensorViewRepr:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        data = np.arange(12, dtype=np.float32)
        view = make_naive_tensor_view(data, [3, 4], [4, 1])
        
        repr_str = repr(view)
        assert "TensorView" in repr_str
        assert "ndim=2" in repr_str
        assert "buffer_size=12" in repr_str
        assert "element_space_size=12" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 