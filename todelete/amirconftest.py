"""
Common test fixtures for tensor transform tests.
"""

import pytest
from tensor_transforms import TensorTransformParser

@pytest.fixture
def parser():
    """Create a fresh parser instance for each test."""
    return TensorTransformParser()

@pytest.fixture
def example_variables():
    """Common example variables for tests."""
    return {
        'kNPerBlock': 32,
        'kKPerBlock': 64,
        'kKPack': 8,
        'NumIssues': 4,
        'NumWarps': 8,
        'LaneGroups': 2,
        'LanesPerK': 4,
        'KVector': 2
    } 
 