import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def sample_raster_data():
    """Create sample raster data for testing."""
    return {
        "blue": np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        "green": np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        "red": np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        "nir": np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        "red_edge": np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    }

@pytest.fixture(scope="session")
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        "camera": {
            "make": "MicaSense",
            "model": "RedEdge-M",
            "serial_number": "TEST123"
        },
        "capture": {
            "timestamp": "2024-01-01T12:00:00Z",
            "altitude": 100.0,
            "pitch": 0.0,
            "roll": 0.0,
            "yaw": 0.0
        },
        "bands": {
            "blue": {"wavelength": 475},
            "green": {"wavelength": 560},
            "red": {"wavelength": 668},
            "nir": {"wavelength": 840},
            "red_edge": {"wavelength": 717}
        }
    } 