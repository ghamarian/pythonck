"""
Common test fixtures and configuration for pythonck tests.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import modules for fixtures
from tensor_transforms import TensorTransformParser, extract_descriptors
from tile_distribution import get_default_variables


@pytest.fixture
def parser():
    """Create a fresh tensor transform parser instance for each test."""
    return TensorTransformParser()


@pytest.fixture
def example_variables():
    """Common example variables for tensor transform tests."""
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


@pytest.fixture
def tile_distribution_variables():
    """Default variables for tile distribution tests."""
    return get_default_variables()


@pytest.fixture
def sample_tensor_shapes():
    """Common tensor shapes for testing."""
    return {
        'small_2d': [32, 64],
        'medium_2d': [128, 256],
        'large_2d': [512, 1024],
        'small_3d': [16, 32, 64],
        'medium_3d': [64, 128, 256],
        'vector_1d': [1024],
        'batch_2d': [8, 256, 256],
    }


@pytest.fixture
def rmsnorm_config():
    """RMSNorm configuration for testing."""
    return {
        'Repeat_M': 4,
        'WarpPerBlock_M': 2,
        'ThreadPerWarp_M': 8,
        'Vector_M': 4,
        'Repeat_N': 4,
        'WarpPerBlock_N': 2,
        'ThreadPerWarp_N': 8,
        'Vector_N': 4,
    }


@pytest.fixture
def gemm_config():
    """GEMM configuration for testing."""
    return {
        'BlockSize_M': 128,
        'BlockSize_N': 128,
        'BlockSize_K': 32,
        'ThreadCluster_M': 4,
        'ThreadCluster_N': 4,
        'WarpPerBlock_M': 2,
        'WarpPerBlock_N': 2,
        'Vector_M': 4,
        'Vector_N': 4,
    }


@pytest.fixture
def coordinate_test_cases():
    """Test cases for coordinate transformations."""
    return [
        {
            'name': 'simple_2d',
            'shape': [4, 8],
            'coordinates': [(0, 0), (1, 2), (3, 7)],
        },
        {
            'name': 'medium_3d',
            'shape': [8, 16, 32],
            'coordinates': [(0, 0, 0), (2, 4, 8), (7, 15, 31)],
        },
        {
            'name': 'large_4d',
            'shape': [4, 8, 16, 32],
            'coordinates': [(0, 0, 0, 0), (1, 2, 4, 8), (3, 7, 15, 31)],
        },
    ]


@pytest.fixture
def transform_test_cases():
    """Test cases for tensor transforms."""
    return [
        {
            'name': 'passthrough',
            'input_shape': [64, 128],
            'transform_type': 'PassThrough',
            'expected_output_shape': [64, 128],
        },
        {
            'name': 'merge',
            'input_shape': [8, 16, 32],
            'transform_type': 'Merge',
            'merge_dims': [0, 1],
            'expected_output_shape': [128, 32],
        },
        {
            'name': 'unmerge',
            'input_shape': [128, 32],
            'transform_type': 'Unmerge',
            'unmerge_dim': 0,
            'unmerge_lengths': [8, 16],
            'expected_output_shape': [8, 16, 32],
        },
    ]


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data files."""
    data_dir = Path(__file__).parent / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (skip in CI)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if any(keyword in item.nodeid for keyword in ["large", "performance", "benchmark"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if any(keyword in item.nodeid for keyword in ["gpu", "cuda", "hip"]):
            item.add_marker(pytest.mark.gpu)


# Skip GPU tests if not available
def pytest_runtest_setup(item):
    """Skip GPU tests if GPU is not available."""
    if "gpu" in item.keywords:
        pytest.skip("GPU tests require CUDA/HIP runtime") 