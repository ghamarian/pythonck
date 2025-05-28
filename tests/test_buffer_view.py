"""
Tests for buffer_view module.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from pytensor.buffer_view import (
    BufferView, AddressSpaceEnum, MemoryOperationEnum, 
    AmdBufferCoherenceEnum, make_buffer_view
)


class TestBufferView:
    """Test cases for BufferView class."""
    
    def test_buffer_view_creation(self):
        """Test basic buffer view creation."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        buf = BufferView(
            address_space=AddressSpaceEnum.GENERIC,
            data=data,
            buffer_size=5
        )
        
        assert buf.address_space == AddressSpaceEnum.GENERIC
        assert buf.buffer_size == 5
        assert np.array_equal(buf.data, data)
        assert buf.invalid_element_use_numerical_zero == True
        assert buf.invalid_element_value == 0
    
    def test_buffer_view_auto_size(self):
        """Test automatic buffer size calculation."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        buf = BufferView(
            address_space=AddressSpaceEnum.GLOBAL,
            data=data
        )
        
        assert buf.buffer_size == 5
    
    def test_get_address_space(self):
        """Test get_address_space method."""
        for space in AddressSpaceEnum:
            buf = BufferView(address_space=space)
            assert buf.get_address_space() == space
    
    def test_indexing(self):
        """Test __getitem__ and __call__ methods."""
        data = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        buf = BufferView(
            address_space=AddressSpaceEnum.LDS,
            data=data
        )
        
        # Test __getitem__
        assert buf[0] == 10
        assert buf[2] == 30
        assert buf[4] == 50
        
        # Test __call__
        assert buf(1) == 20
        assert buf(3) == 40
    
    def test_get_single_element(self):
        """Test get method for single elements."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        buf = BufferView(
            address_space=AddressSpaceEnum.VGPR,
            data=data
        )
        
        # Valid element access
        assert buf.get(0, 0, True) == 1.0
        assert buf.get(2, 1, True) == 4.0  # index=2, offset=1 -> 3rd element
        
        # Invalid element access with numerical zero
        assert buf.get(0, 0, False) == 0
        
        # Invalid element access with custom value
        buf.invalid_element_use_numerical_zero = False
        buf.invalid_element_value = -999
        assert buf.get(0, 0, False) == -999
    
    def test_get_vectorized(self):
        """Test vectorized get access."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        buf = BufferView(
            address_space=AddressSpaceEnum.GENERIC,
            data=data
        )
        
        # Get vector of 4 elements starting at index 0
        vec = buf.get(0, 0, True, vector_size=4)
        assert np.array_equal(vec, [1, 2, 3, 4])
        
        # Get vector with offset
        vec = buf.get(2, 1, True, vector_size=3)
        assert np.array_equal(vec, [4, 5, 6])
        
        # Get vector that goes out of bounds
        vec = buf.get(6, 0, True, vector_size=4)
        assert vec.shape == (4,)
        assert np.array_equal(vec[:2], [7, 8])
        assert np.array_equal(vec[2:], [0, 0])
    
    def test_set_single_element(self):
        """Test set method for single elements."""
        data = np.zeros(5, dtype=np.float32)
        buf = BufferView(
            address_space=AddressSpaceEnum.GENERIC,
            data=data
        )
        
        # Set valid elements
        buf.set(0, 0, True, 10.0)
        buf.set(2, 1, True, 30.0)  # index=2, offset=1 -> 3rd element
        
        assert data[0] == 10.0
        assert data[3] == 30.0
        
        # Try to set invalid element (should not change)
        buf.set(1, 0, False, 99.0)
        assert data[1] == 0.0
    
    def test_set_vectorized(self):
        """Test vectorized set access."""
        data = np.zeros(8, dtype=np.int32)
        buf = BufferView(
            address_space=AddressSpaceEnum.GENERIC,
            data=data
        )
        
        # Set vector of values
        vec = np.array([10, 20, 30, 40])
        buf.set(0, 0, True, vec, vector_size=4)
        assert np.array_equal(data[:4], vec)
        
        # Set with offset
        vec2 = np.array([50, 60, 70])
        buf.set(4, 1, True, vec2, vector_size=3)
        assert np.array_equal(data[5:8], vec2)
    
    def test_update_operations(self):
        """Test update method with different operations."""
        data = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        buf = BufferView(
            address_space=AddressSpaceEnum.GENERIC,
            data=data
        )
        
        # Test SET operation
        buf.update(MemoryOperationEnum.SET, 0, 0, True, 100)
        assert data[0] == 100
        
        # Test ADD operation
        buf.update(MemoryOperationEnum.ADD, 1, 0, True, 5)
        assert data[1] == 25  # 20 + 5
        
        # Test ATOMIC_ADD (simulated)
        buf.update(MemoryOperationEnum.ATOMIC_ADD, 2, 0, True, 10)
        assert data[2] == 40  # 30 + 10
        
        # Test ATOMIC_MAX
        buf.update(MemoryOperationEnum.ATOMIC_MAX, 3, 0, True, 35)
        assert data[3] == 40  # max(40, 35)
        
        buf.update(MemoryOperationEnum.ATOMIC_MAX, 3, 0, True, 45)
        assert data[3] == 45  # max(40, 45)
    
    def test_update_vectorized(self):
        """Test vectorized update operations."""
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        buf = BufferView(
            address_space=AddressSpaceEnum.GENERIC,
            data=data
        )
        
        # Vectorized ADD
        add_vec = np.array([10.0, 20.0, 30.0])
        buf.update(MemoryOperationEnum.ADD, 0, 0, True, add_vec, vector_size=3)
        assert np.array_equal(data[:3], [11.0, 22.0, 33.0])
        
        # Vectorized MAX
        max_vec = np.array([40.0, 2.0, 50.0])
        buf.update(MemoryOperationEnum.ATOMIC_MAX, 3, 0, True, max_vec, vector_size=3)
        assert np.array_equal(data[3:6], [40.0, 5.0, 50.0])
    
    def test_raw_operations(self):
        """Test raw memory operations."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        buf = BufferView(
            address_space=AddressSpaceEnum.GLOBAL,
            data=data
        )
        
        # Test init_raw
        buf.init_raw()
        assert buf.cached_buf_res is not None
        assert "buffer_resource_" in buf.cached_buf_res
        
        # Test get_raw
        dst = np.zeros(3, dtype=np.float32)
        buf.get_raw(dst, 1, 0, True)
        assert np.array_equal(dst, [2, 3, 4])
        
        # Test set_raw
        buf.set_raw(0, 0, True, 99.0)
        assert data[0] == 99.0
    
    def test_out_of_bounds_handling(self):
        """Test out-of-bounds access handling."""
        data = np.array([1, 2, 3], dtype=np.int32)
        buf = BufferView(
            address_space=AddressSpaceEnum.GENERIC,
            data=data
        )
        
        # Out of bounds read should return invalid value
        with pytest.warns(UserWarning, match="Out of bounds access"):
            val = buf.get(5, 0, True, oob_conditional_check=True)
        
        # Out of bounds write should warn but not crash
        with pytest.warns(UserWarning, match="Out of bounds write"):
            buf.set(5, 0, True, 99, oob_conditional_check=True)
    
    def test_make_buffer_view_factory(self):
        """Test make_buffer_view factory function."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        
        buf = make_buffer_view(
            data=data,
            buffer_size=4,
            address_space=AddressSpaceEnum.LDS,
            invalid_element_value=-1.0,
            invalid_element_use_numerical_zero=False
        )
        
        assert buf.address_space == AddressSpaceEnum.LDS
        assert buf.buffer_size == 4
        assert buf.invalid_element_value == -1.0
        assert buf.invalid_element_use_numerical_zero == False
        assert np.array_equal(buf.data, data)
    
    def test_buffer_properties(self):
        """Test buffer property methods."""
        buf = BufferView(address_space=AddressSpaceEnum.GENERIC)
        
        assert buf.is_static_buffer() == False
        assert buf.is_dynamic_buffer() == True
    
    def test_repr(self):
        """Test string representation."""
        data = np.array([1, 2, 3], dtype=np.int32)
        buf = BufferView(
            address_space=AddressSpaceEnum.VGPR,
            data=data,
            buffer_size=3,
            invalid_element_value=42
        )
        
        repr_str = repr(buf)
        assert "VGPR" in repr_str
        assert "buffer_size=3" in repr_str
        assert "invalid_element_value=42" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 