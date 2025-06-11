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
import os
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
import threading

from pipeline.utils.rasterio_utils import RasterioManager

class TestRasterioManager(unittest.TestCase):
    """Test cases for RasterioManager"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.manager = RasterioManager(max_memory_mb=512)  # Set lower memory limit for testing
        
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
        
        # Create a large test image
        data = np.random.randint(0, 65535, (1000, 1000), dtype=np.uint16)
        profile.update({'count': 1, 'height': 1000, 'width': 1000})
        
        test_file = cls.test_dir / "test_large.tif"
        with rasterio.open(test_file, 'w', **profile) as dst:
            dst.write(data, 1)
    
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
    
    def test_safe_open_network(self):
        """Test network file opening with retry"""
        test_file = self.test_dir / "test.tif"
        
        # Test successful open
        with self.manager.safe_open_network(test_file) as src:
            self.assertEqual(src.count, 1)
        
        # Test retry on temporary failure
        def mock_open(*args, **kwargs):
            if not hasattr(mock_open, 'attempts'):
                mock_open.attempts = 0
            mock_open.attempts += 1
            if mock_open.attempts < 2:
                raise RasterioIOError("Temporary network error")
            return rasterio.open(*args, **kwargs)
        
        # Replace rasterio.open with mock
        original_open = rasterio.open
        rasterio.open = mock_open
        
        try:
            with self.manager.safe_open_network(test_file) as src:
                self.assertEqual(src.count, 1)
        finally:
            rasterio.open = original_open
    
    def test_file_integrity(self):
        """Test file integrity validation"""
        test_file = self.test_dir / "test.tif"
        
        # Test valid file
        is_valid, error_msg = self.manager.validate_file_integrity(test_file)
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)
        
        # Test non-existent file
        invalid_file = self.test_dir / "nonexistent.tif"
        is_valid, error_msg = self.manager.validate_file_integrity(invalid_file)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)
        
        # Test empty file
        empty_file = self.test_dir / "empty.tif"
        empty_file.touch()
        is_valid, error_msg = self.manager.validate_file_integrity(empty_file)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)
    
    def test_memory_management(self):
        """Test memory management"""
        # Test memory limit
        self.assertTrue(self.manager.check_memory_usage())
        
        # Test memory monitoring
        test_file = self.test_dir / "test.tif"
        with self.manager.safe_open(test_file) as src:
            src.read(1)
        
        metrics = self.manager.get_operation_metrics()
        self.assertIn('file_open', metrics['memory_usage'])
        self.assertGreater(len(metrics['memory_usage']['file_open']), 0)
    
    def test_concurrent_access(self):
        """Test concurrent file access"""
        test_file = self.test_dir / "test.tif"
        
        def read_file():
            with self.manager.safe_open(test_file) as src:
                return src.read(1)
        
        # Test concurrent reads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read_file) for _ in range(4)]
            results = [f.result() for f in futures]
        
        self.assertEqual(len(results), 4)
        for result in results:
            self.assertEqual(result.shape, (1, 100, 100))
    
    def test_large_file_handling(self):
        """Test handling of large files"""
        test_file = self.test_dir / "test_large.tif"
        
        # Test chunked processing
        chunks = list(self.manager.process_in_chunks(test_file, chunk_size=100))
        self.assertGreater(len(chunks), 0)
        
        # Verify chunk shapes
        for chunk in chunks:
            self.assertLessEqual(chunk.shape[0], 100)
            self.assertLessEqual(chunk.shape[1], 100)
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted files"""
        test_file = self.test_dir / "test.tif"
        
        # Corrupt the file
        with open(test_file, 'ab') as f:
            f.write(b'corrupted data')
        
        # Test validation
        is_valid, error_msg = self.manager.validate_file_integrity(test_file)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        test_file = self.test_dir / "test.tif"
        
        # Perform some operations
        with self.manager.safe_open(test_file) as src:
            src.read(1)
        
        metrics = self.manager.get_operation_metrics()
        
        # Check operation times
        self.assertIn('file_open', metrics['operation_times'])
        self.assertGreater(len(metrics['operation_times']['file_open']), 0)
        
        # Check memory usage
        self.assertIn('file_open', metrics['memory_usage'])
        self.assertGreater(len(metrics['memory_usage']['file_open']), 0)
        
        # Check error counts
        self.assertIn('file_open', metrics['error_counts'])
    
    def test_statistics_calculation(self):
        """Test statistics calculation with chunked processing"""
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
    
    def test_file_locking(self):
        """Test file locking mechanism"""
        test_file = self.test_dir / "test.tif"
        lock_file = test_file.with_suffix('.lock')
        
        # Test lock creation and cleanup
        with self.manager.safe_open(test_file) as src:
            self.assertTrue(lock_file.exists())
            src.read(1)
        
        self.assertFalse(lock_file.exists())
        
        # Test concurrent access with locks
        def read_with_delay():
            with self.manager.safe_open(test_file) as src:
                time.sleep(0.1)  # Simulate processing
                return src.read(1)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(read_with_delay) for _ in range(2)]
            results = [f.result() for f in futures]
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result.shape, (1, 100, 100)) 