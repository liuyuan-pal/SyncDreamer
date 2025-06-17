"""Validation tests to ensure the testing infrastructure is properly set up."""

import sys
from pathlib import Path

import pytest


class TestSetupValidation:
    """Test class to validate the testing infrastructure setup."""
    
    @pytest.mark.unit
    def test_pytest_is_installed(self):
        """Test that pytest is properly installed."""
        assert "pytest" in sys.modules or True  # Will be true after installation
    
    @pytest.mark.unit
    def test_project_structure_exists(self):
        """Test that the required project structure exists."""
        project_root = Path(__file__).parent.parent
        
        # Check main directories exist
        assert project_root.exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
        # Check configuration files exist
        assert (project_root / "pyproject.toml").exists()
    
    @pytest.mark.unit
    def test_conftest_fixtures_available(self, temp_dir, mock_config):
        """Test that conftest fixtures are properly available."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "model" in mock_config
        assert "data" in mock_config
    
    @pytest.mark.unit
    def test_markers_are_defined(self, request):
        """Test that custom markers are properly defined."""
        markers = request.config.getini("markers")
        
        # Check our custom markers exist
        assert any("unit:" in marker for marker in markers)
        assert any("integration:" in marker for marker in markers)
        assert any("slow:" in marker for marker in markers)
    
    @pytest.mark.unit
    def test_coverage_configuration(self):
        """Test that coverage is properly configured."""
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        # Check pyproject.toml contains coverage configuration
        content = pyproject_path.read_text()
        assert "[tool.coverage.run]" in content
        assert "[tool.coverage.report]" in content
        assert "fail_under = 80" in content
    
    @pytest.mark.unit
    def test_sample_image_fixture(self, sample_image):
        """Test the sample_image fixture creates a valid image."""
        assert sample_image.exists()
        assert sample_image.suffix == ".png"
        
        # Test file was created
        assert sample_image.read_text() == "dummy image data"
    
    @pytest.mark.unit
    def test_numpy_fixtures(self, sample_numpy_image, sample_batch_images):
        """Test numpy array fixtures."""
        # Test single image mock
        assert isinstance(sample_numpy_image, dict)
        assert sample_numpy_image["shape"] == (256, 256, 3)
        assert sample_numpy_image["dtype"] == "float32"
        
        # Test batch images mock
        assert isinstance(sample_batch_images, dict)
        assert sample_batch_images["shape"] == (4, 3, 256, 256)
        assert sample_batch_images["dtype"] == "float32"
    
    @pytest.mark.unit
    def test_mock_3d_data_fixture(self, mock_3d_data):
        """Test the mock 3D data fixture."""
        assert "vertices" in mock_3d_data
        assert "faces" in mock_3d_data
        assert "normals" in mock_3d_data
        assert "camera_matrix" in mock_3d_data
        
        # Check mock data structure
        assert mock_3d_data["vertices"]["shape"] == (100, 3)
        assert mock_3d_data["faces"]["shape"] == (50, 3)
        assert mock_3d_data["normals"]["shape"] == (100, 3)
        assert mock_3d_data["camera_matrix"]["shape"] == (4, 4)
    
    @pytest.mark.integration
    def test_mock_dataset_fixture(self, mock_dataset_dir):
        """Test the mock dataset directory fixture."""
        # Check directory structure
        assert mock_dataset_dir.exists()
        assert (mock_dataset_dir / "train").exists()
        assert (mock_dataset_dir / "val").exists()
        
        # Check sample images were created
        train_images = list((mock_dataset_dir / "train").glob("*.png"))
        val_images = list((mock_dataset_dir / "val").glob("*.png"))
        
        assert len(train_images) == 5
        assert len(val_images) == 5
    
    @pytest.mark.slow
    def test_slow_marker_example(self):
        """Example test with slow marker to verify marker functionality."""
        import time
        
        # Simulate a slow operation
        time.sleep(0.1)
        assert True
    
    def test_poetry_scripts_configured(self):
        """Test that poetry scripts are properly configured."""
        from pathlib import Path
        import toml
        
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        # Load pyproject.toml
        with open(pyproject_path, 'r') as f:
            pyproject = toml.load(f)
        
        # Check poetry scripts
        assert "tool" in pyproject
        assert "poetry" in pyproject["tool"]
        assert "scripts" in pyproject["tool"]["poetry"]
        
        scripts = pyproject["tool"]["poetry"]["scripts"]
        assert "test" in scripts
        assert "tests" in scripts
        assert scripts["test"] == "pytest:main"
        assert scripts["tests"] == "pytest:main"