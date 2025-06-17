"""Shared pytest fixtures and configuration for all tests."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "model": {
            "name": "test_model",
            "params": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "num_epochs": 10,
            }
        },
        "data": {
            "train_path": "/tmp/train",
            "val_path": "/tmp/val",
            "image_size": 256,
        },
        "trainer": {
            "gpus": 0,
            "max_epochs": 10,
            "check_val_every_n_epoch": 1,
        }
    }


@pytest.fixture
def mock_omega_config(mock_config: Dict[str, Any]) -> Dict[str, Any]:
    """Provide a configuration object for testing."""
    # Return as dict for now, can be converted to OmegaConf when available
    return mock_config


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample image for testing."""
    img_path = temp_dir / "test_image.png"
    # Create a dummy file to simulate an image
    img_path.write_text("dummy image data")
    return img_path


@pytest.fixture
def sample_numpy_image() -> Dict[str, Any]:
    """Create a mock numpy array image for testing."""
    return {
        "shape": (256, 256, 3),
        "dtype": "float32",
        "type": "numpy.ndarray"
    }


@pytest.fixture
def sample_batch_images() -> Dict[str, Any]:
    """Create a mock batch of sample images."""
    return {
        "shape": (4, 3, 256, 256),
        "dtype": "float32",
        "type": "numpy.ndarray"
    }


@pytest.fixture
def mock_model_checkpoint(temp_dir: Path) -> Path:
    """Create a mock model checkpoint file."""
    checkpoint_path = temp_dir / "model_checkpoint.ckpt"
    # Create a dummy file to simulate checkpoint
    checkpoint_path.write_text("dummy checkpoint data")
    return checkpoint_path


@pytest.fixture
def mock_dataset_dir(temp_dir: Path) -> Path:
    """Create a mock dataset directory structure."""
    dataset_dir = temp_dir / "dataset"
    
    # Create train and val directories
    (dataset_dir / "train").mkdir(parents=True)
    (dataset_dir / "val").mkdir(parents=True)
    
    # Create some sample image files
    for split in ["train", "val"]:
        for i in range(5):
            img_path = dataset_dir / split / f"image_{i}.png"
            img_path.write_text(f"dummy image {i}")
    
    return dataset_dir


@pytest.fixture
def mock_3d_data() -> Dict[str, Any]:
    """Create mock 3D data for testing renderer and raymarching modules."""
    return {
        "vertices": {"shape": (100, 3), "dtype": "float32"},
        "faces": {"shape": (50, 3), "dtype": "int32"},
        "normals": {"shape": (100, 3), "dtype": "float32"},
        "camera_matrix": {"shape": (4, 4), "dtype": "float32", "type": "identity"},
    }


@pytest.fixture
def mock_camera_params() -> Dict[str, Any]:
    """Create mock camera parameters for testing."""
    return {
        "intrinsics": {
            "fx": 525.0,
            "fy": 525.0,
            "cx": 320.0,
            "cy": 240.0,
        },
        "extrinsics": {"shape": (4, 4), "dtype": "float32", "type": "identity"},
        "width": 640,
        "height": 480,
        "near": 0.1,
        "far": 100.0,
    }


@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
    """Set up mock environment variables for testing."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TEST_MODE"] = "1"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducible tests."""
    import random
    
    random.seed(42)
    # NumPy and PyTorch seed setting will be done when modules are available


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    return captured_output


@pytest.fixture
def mock_gpu_available(monkeypatch):
    """Mock GPU availability for testing."""
    # This will be implemented when torch is available
    pass
    

@pytest.fixture
def cleanup_gpu():
    """Clean up GPU memory after tests."""
    yield
    # GPU cleanup will be done when torch is available


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers documentation
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker for specific test patterns
        if "slow" in item.name or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)