"""
Tests for functional utility functions.
"""

import pytest
import sys
sys.path.append('..')

from pytensor.functional_utils import (
    Identity, identity, unpack, unpack2, StaticFor, StaticFord, StaticUford
)


class TestIdentity:
    """Test Identity function."""
    
    def test_identity_function(self):
        """Test identity function returns input unchanged."""
        ident = Identity()
        
        assert ident(42) == 42
        assert ident("hello") == "hello"
        assert ident([1, 2, 3]) == [1, 2, 3]
        assert ident(None) is None
        
        # Test global instance
        assert identity(42) == 42


class TestUnpack:
    """Test unpack functions."""
    
    def test_unpack_basic(self):
        """Test basic unpack functionality."""
        result = unpack(lambda a, b, c: a + b + c, [1, 2, 3])
        assert result == 6
        
        result = unpack(lambda x, y: x * y, [4, 5])
        assert result == 20
        
        result = unpack(lambda: "no args", [])
        assert result == "no args"
    
    def test_unpack2_basic(self):
        """Test unpack2 functionality."""
        result = unpack2(lambda a, b, c, d: a + b + c + d, [1, 2], [3, 4])
        assert result == 10
        
        result = unpack2(lambda x, y, z: x * y + z, [2, 3], [4])
        assert result == 10
        
        result = unpack2(lambda a, b: a - b, [10], [3])
        assert result == 7
    
    def test_unpack_with_different_functions(self):
        """Test unpack with various function types."""
        # String concatenation
        result = unpack(lambda a, b, c: f"{a}-{b}-{c}", ["hello", "world", "test"])
        assert result == "hello-world-test"
        
        # List operations
        result = unpack(lambda *args: list(args), [1, 2, 3, 4])
        assert result == [1, 2, 3, 4]


class TestStaticFor:
    """Test StaticFor class."""
    
    def test_static_for_basic(self):
        """Test basic static for functionality."""
        values = []
        
        loop = StaticFor(0, 5, 1)
        loop(lambda i: values.append(i))
        
        assert values == [0, 1, 2, 3, 4]
    
    def test_static_for_with_increment(self):
        """Test static for with different increments."""
        values = []
        
        loop = StaticFor(0, 10, 2)
        loop(lambda i: values.append(i))
        
        assert values == [0, 2, 4, 6, 8]
    
    def test_static_for_negative_increment(self):
        """Test static for with negative increment."""
        values = []
        
        loop = StaticFor(10, 0, -2)
        loop(lambda i: values.append(i))
        
        assert values == [10, 8, 6, 4, 2]
    
    def test_static_for_single_iteration(self):
        """Test static for with single iteration."""
        values = []
        
        loop = StaticFor(5, 6, 1)
        loop(lambda i: values.append(i))
        
        assert values == [5]
    
    def test_static_for_no_iterations(self):
        """Test static for with no iterations."""
        values = []
        
        loop = StaticFor(5, 5, 1)
        loop(lambda i: values.append(i))
        
        assert values == []
    
    def test_static_for_validation(self):
        """Test static for input validation."""
        
        # Zero increment
        with pytest.raises(ValueError, match="Increment cannot be zero"):
            StaticFor(0, 5, 0)
        
        # Non-divisible range
        with pytest.raises(ValueError, match="must be divisible by increment"):
            StaticFor(0, 5, 2)
        
        # Invalid positive increment
        with pytest.raises(ValueError, match="begin must be <= end"):
            StaticFor(5, 0, 1)
        
        # Invalid negative increment
        with pytest.raises(ValueError, match="begin must be >= end"):
            StaticFor(0, 5, -1)


class TestStaticFord:
    """Test StaticFord class."""
    
    def test_static_ford_basic(self):
        """Test basic static ford functionality."""
        indices = []
        
        loop = StaticFord([2, 3])
        loop(lambda idx: indices.append(idx.copy()))
        
        expected = [
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2]
        ]
        assert indices == expected
    
    def test_static_ford_with_orders(self):
        """Test static ford with custom ordering."""
        indices = []
        
        # Reverse order: iterate second dimension first, then first
        loop = StaticFord([2, 3], orders=[1, 0])
        loop(lambda idx: indices.append(idx.copy()))
        
        expected = [
            [0, 0], [1, 0],
            [0, 1], [1, 1],
            [0, 2], [1, 2]
        ]
        assert indices == expected
    
    def test_static_ford_3d(self):
        """Test static ford with 3 dimensions."""
        indices = []
        
        loop = StaticFord([2, 2, 2])
        loop(lambda idx: indices.append(idx.copy()))
        
        # Should visit all 8 combinations
        assert len(indices) == 8
        
        # Check first and last
        assert indices[0] == [0, 0, 0]
        assert indices[-1] == [1, 1, 1]
    
    def test_static_ford_validation(self):
        """Test static ford input validation."""
        
        # Empty lengths
        with pytest.raises(ValueError, match="Lengths cannot be empty"):
            StaticFord([])
        
        # Mismatched orders length
        with pytest.raises(ValueError, match="Orders and lengths must have same size"):
            StaticFord([2, 3], orders=[0])
        
        # Invalid orders
        with pytest.raises(ValueError, match="Orders must be a valid permutation"):
            StaticFord([2, 3], orders=[0, 2])
        
        with pytest.raises(ValueError, match="Orders must be a valid permutation"):
            StaticFord([2, 3], orders=[0, 0])


class TestStaticUford:
    """Test StaticUford class."""
    
    def test_static_uford_no_unpacking(self):
        """Test static uford with no unpacking (all 1s)."""
        calls = []
        
        loop = StaticUford([2, 3], unpacks=[1, 1])
        loop(lambda *indices: calls.append(len(indices)))
        
        # Should make 2*3 = 6 calls, each with 1 index
        assert len(calls) == 6
        assert all(call_size == 1 for call_size in calls)
    
    def test_static_uford_with_unpacking(self):
        """Test static uford with unpacking."""
        calls = []
        indices_received = []
        
        def collector(*indices):
            calls.append(len(indices))
            indices_received.append([idx.copy() for idx in indices])
        
        # Unpack 2 elements from last dimension
        loop = StaticUford([2, 4], unpacks=[1, 2])
        loop(collector)
        
        # Should make (2/1) * (4/2) = 2 * 2 = 4 calls
        assert len(calls) == 4
        
        # Each call should receive 1*2 = 2 indices
        assert all(call_size == 2 for call_size in calls)
    
    def test_static_uford_get_num_of_access(self):
        """Test get_num_of_access calculation."""
        
        loop1 = StaticUford([4, 6], unpacks=[2, 3])
        assert loop1.get_num_of_access() == (4//2) * (6//3)  # 2 * 2 = 4
        
        loop2 = StaticUford([8], unpacks=[4])
        assert loop2.get_num_of_access() == 8//4  # 2
        
        loop3 = StaticUford([2, 3, 4], unpacks=[1, 1, 2])
        assert loop3.get_num_of_access() == (2//1) * (3//1) * (4//2)  # 2 * 3 * 2 = 12
    
    def test_static_uford_single_access(self):
        """Test executing single access by index."""
        calls = []
        
        loop = StaticUford([4], unpacks=[2])
        
        # Execute each access individually
        for i in range(loop.get_num_of_access()):
            calls.clear()
            loop(lambda *indices: calls.append(len(indices)), access_index=i)
            assert len(calls) == 1
    
    def test_static_uford_validation(self):
        """Test static uford input validation."""
        
        # Empty lengths
        with pytest.raises(ValueError, match="Lengths cannot be empty"):
            StaticUford([])
        
        # Mismatched unpacks length
        with pytest.raises(ValueError, match="Unpacks and lengths must have same size"):
            StaticUford([2, 3], unpacks=[1])
        
        # Mismatched orders length
        with pytest.raises(ValueError, match="Orders and lengths must have same size"):
            StaticUford([2, 3], orders=[0])
        
        # Non-divisible length/unpack
        with pytest.raises(ValueError, match="must be divisible by unpack"):
            StaticUford([5], unpacks=[2])
        
        # Invalid orders
        with pytest.raises(ValueError, match="Orders must be a valid permutation"):
            StaticUford([2, 3], orders=[0, 2])
    
    def test_static_uford_complex_example(self):
        """Test static uford with complex unpacking pattern."""
        calls = []
        all_indices = []
        
        def collector(*indices):
            calls.append(len(indices))
            all_indices.extend(indices)
        
        # Example: [2, 3, 4] with unpacks [1, 1, 2]
        # Should generate pairs of indices where last dimension is unpacked
        loop = StaticUford([2, 3, 4], unpacks=[1, 1, 2])
        loop(collector)
        
        # Should make (2/1) * (3/1) * (4/2) = 12 calls
        assert len(calls) == 12
        
        # Each call should receive 1*1*2 = 2 indices
        assert all(call_size == 2 for call_size in calls)
        
        # Total indices received should be 12 * 2 = 24
        assert len(all_indices) == 24
    
    def test_static_uford_with_orders(self):
        """Test static uford with custom ordering."""
        calls = []
        
        loop = StaticUford([2, 4], unpacks=[1, 2], orders=[1, 0])
        loop(lambda *indices: calls.append(len(indices)))
        
        # Should still make same number of calls
        assert len(calls) == (2//1) * (4//2)  # 4 calls
        assert all(call_size == 2 for call_size in calls)


class TestIntegrationExamples:
    """Integration tests with examples that match C++ behavior."""
    
    def test_cpp_static_ford_example(self):
        """Test example that matches C++ static_ford behavior."""
        indices = []
        
        # Simulate: static_ford<sequence<2, 3>>{}([&](auto idx) { ... })
        loop = StaticFord([2, 3])
        loop(lambda idx: indices.append(tuple(idx)))
        
        expected = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2)
        ]
        assert indices == expected
    
    def test_cpp_static_uford_example(self):
        """Test example that matches C++ static_uford behavior."""
        call_data = []
        
        def func(idx1, idx2):
            call_data.append((tuple(idx1), tuple(idx2)))
        
        # Simulate: static_uford<sequence<2, 3, 4>, sequence<1, 1, 2>>{}([&](auto i_0, auto i_1) { ... })
        loop = StaticUford([2, 3, 4], unpacks=[1, 1, 2])
        loop(func)
        
        # Should make 12 calls, each with 2 indices
        assert len(call_data) == 12
        
        # Check first few calls match expected pattern
        expected_first_calls = [
            ((0, 0, 0), (0, 0, 1)),  # loop #0
            ((0, 0, 2), (0, 0, 3)),  # loop #1
            ((0, 1, 0), (0, 1, 1)),  # loop #2
            ((0, 1, 2), (0, 1, 3)),  # loop #3
        ]
        
        for i, expected in enumerate(expected_first_calls):
            if i < len(call_data):
                assert call_data[i] == expected, f"Call {i}: expected {expected}, got {call_data[i]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 