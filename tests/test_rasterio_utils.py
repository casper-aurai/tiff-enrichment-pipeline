"""
Tests for Rasterio Utilities
"""

import unittest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import tempfile
import shutil

from pipeline.utils.rasterio_utils import RasterioManager

class TestRasterioManager(unittest.TestCase):
    """Test cases for RasterioManager"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.manager = RasterioManager()
        
        # Create test data
        cls._create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_data(cls):
        """Create test TIFF files"""
        # Create a simple test image
        data = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        
        # Save as TIFF
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'count': 1,
            'height': 100,
            'width': 100,
            'transform': rasterio.transform.from_origin(0, 0, 1, 1),
            'crs': 'EPSG:4326'
        }
        
        test_file = cls.test_dir / "test.tif"
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(data, 1)
        
        # Create a multi-band test image
        data = np.random.randint(0, 65535, (3, 100, 100), dtype=np.uint16)
        profile.update({'count': 3})
        
        test_file = cls.test_dir / "test_multi.tif"
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(data)
    
    def test_safe_open(self):
        """Test safe file opening"""
        # Test valid file
        test_file = self.test_dir / "test.tif"
        with self.manager.safe_open(test_file) as src:
            self.assertEqual(src.count, 1)
            self.assertEqual(src.dtypes[0], 'uint16')
        
        # Test invalid file
        invalid_file = self.test_dir / "nonexistent.tif"
        with self.assertRaises(RasterioIOError):
            with self.manager.safe_open(invalid_file) as src:
                pass
    
    def test_get_metadata(self):
        """Test metadata caching"""
        test_file = self.test_dir / "test.tif"
        
        # First call should read from file
        metadata1 = self.manager.get_metadata(str(test_file))
        self.assertIn('crs', metadata1)
        self.assertIn('transform', metadata1)
        
        # Second call should use cache
        metadata2 = self.manager.get_metadata(str(test_file))
        self.assertEqual(metadata1, metadata2)
    
    def test_validate_dataset(self):
        """Test dataset validation"""
        test_file = self.test_dir / "test.tif"
        
        with self.manager.safe_open(test_file) as src:
            is_valid, error_msg = self.manager.validate_dataset(src)
            self.assertTrue(is_valid)
            self.assertIsNone(error_msg)
    
    def test_process_in_chunks(self):
        """Test chunked processing"""
        test_file = self.test_dir / "test.tif"
        
        # Process in chunks
        chunks = list(self.manager.process_in_chunks(test_file, chunk_size=50))
        self.assertGreater(len(chunks), 0)
        
        # Verify chunk shapes
        for chunk in chunks:
            self.assertLessEqual(chunk.shape[0], 50)
            self.assertLessEqual(chunk.shape[1], 50)
    
    def test_safe_write(self):
        """Test safe file writing"""
        test_file = self.test_dir / "test_write.tif"
        data = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'count': 1,
            'height': 100,
            'width': 100,
            'transform': rasterio.transform.from_origin(0, 0, 1, 1),
            'crs': 'EPSG:4326'
        }
        
        # Test successful write
        success = self.manager.safe_write(test_file, data, profile)
        self.assertTrue(success)
        self.assertTrue(test_file.exists())
        
        # Test invalid write
        invalid_profile = profile.copy()
        invalid_profile['driver'] = 'invalid'
        success = self.manager.safe_write(test_file, data, invalid_profile)
        self.assertFalse(success)
    
    def test_get_statistics(self):
        """Test statistics calculation"""
        test_file = self.test_dir / "test_multi.tif"
        
        stats = self.manager.get_statistics(test_file)
        self.assertEqual(len(stats), 3)  # Three bands
        
        for band_stats in stats.values():
            self.assertIn('min', band_stats)
            self.assertIn('max', band_stats)
            self.assertIn('mean', band_stats)
            self.assertIn('std', band_stats)
            self.assertIn('p25', band_stats)
            self.assertIn('median', band_stats)
            self.assertIn('p75', band_stats)
    
    def test_metrics(self):
        """Test metrics collection"""
        test_file = self.test_dir / "test.tif"
        
        # Perform some operations
        with self.manager.safe_open(test_file) as src:
            src.read(1)
        
        # Check metrics
        self.assertIn('file_open', self.manager.metrics.operation_times)
        self.assertGreater(len(self.manager.metrics.operation_times['file_open']), 0) 