"""
Performance tests for the TIFF Enrichment Pipeline
Measures processing speed and resource usage
"""

import unittest
from pathlib import Path
import shutil
import tempfile
import time
import psutil
import os
import numpy as np
from datetime import datetime
import logging

from src.pipeline.main import TIFFPipelineMain

class TestPipelinePerformance(unittest.TestCase):
    """Performance tests for the pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create temporary directories
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.input_dir = cls.test_dir / "input"
        cls.output_dir = cls.test_dir / "output"
        cls.failed_dir = cls.test_dir / "failed"
        
        # Create directories
        for dir_path in [cls.input_dir, cls.output_dir, cls.failed_dir]:
            dir_path.mkdir(parents=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=str(cls.test_dir / "performance.log")
        )
        cls.logger = logging.getLogger(__name__)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)
    
    def _create_large_tiff(self, size_mb: int, filename: str):
        """Create a large TIFF file of specified size"""
        # Calculate dimensions to achieve desired file size
        # Assuming 16-bit depth (2 bytes per pixel)
        pixels = (size_mb * 1024 * 1024) // 2
        side_length = int(np.sqrt(pixels))
        
        # Create synthetic image data
        data = np.random.randint(0, 65535, (side_length, side_length), dtype=np.uint16)
        
        # Save as TIFF
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        
        transform = from_origin(0, 0, 1, 1)
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'count': 1,
            'height': side_length,
            'width': side_length,
            'transform': transform,
            'crs': 'EPSG:4326'
        }
        
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(data, 1)
                dst.update_tags(
                    EXIF_DateTime=datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    EXIF_GPS_Lat=40.7128,
                    EXIF_GPS_Lon=-74.0060
                )
            
            # Save to file
            output_path = self.input_dir / filename
            with open(output_path, 'wb') as f:
                f.write(memfile.read())
    
    def _measure_memory_usage(self):
        """Measure current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def test_small_file_performance(self):
        """Test performance with small files (1MB)"""
        self.logger.info("Testing small file performance")
        
        # Create test file
        self._create_large_tiff(1, "small.tif")
        
        # Configure pipeline
        config = {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "failed_dir": str(self.failed_dir),
            "max_workers": 1,
            "batch_size": 1
        }
        
        # Measure performance
        start_time = time.time()
        start_memory = self._measure_memory_usage()
        
        pipeline = TIFFPipelineMain(config)
        result = pipeline.run()
        
        end_time = time.time()
        end_memory = self._measure_memory_usage()
        
        # Log results
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        self.logger.info(f"Small file processing:")
        self.logger.info(f"  Duration: {duration:.2f} seconds")
        self.logger.info(f"  Memory used: {memory_used:.2f} MB")
        
        # Assert reasonable performance
        self.assertLess(duration, 5.0)  # Should process in under 5 seconds
        self.assertLess(memory_used, 500)  # Should use less than 500MB
    
    def test_large_file_performance(self):
        """Test performance with large files (100MB)"""
        self.logger.info("Testing large file performance")
        
        # Create test file
        self._create_large_tiff(100, "large.tif")
        
        # Configure pipeline
        config = {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "failed_dir": str(self.failed_dir),
            "max_workers": 1,
            "batch_size": 1
        }
        
        # Measure performance
        start_time = time.time()
        start_memory = self._measure_memory_usage()
        
        pipeline = TIFFPipelineMain(config)
        result = pipeline.run()
        
        end_time = time.time()
        end_memory = self._measure_memory_usage()
        
        # Log results
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        self.logger.info(f"Large file processing:")
        self.logger.info(f"  Duration: {duration:.2f} seconds")
        self.logger.info(f"  Memory used: {memory_used:.2f} MB")
        
        # Assert reasonable performance
        self.assertLess(duration, 60.0)  # Should process in under 60 seconds
        self.assertLess(memory_used, 2000)  # Should use less than 2GB
    
    def test_parallel_processing(self):
        """Test performance with parallel processing"""
        self.logger.info("Testing parallel processing performance")
        
        # Create multiple test files
        for i in range(4):
            self._create_large_tiff(10, f"parallel_{i}.tif")
        
        # Test different worker configurations
        worker_configs = [1, 2, 4]
        
        for workers in worker_configs:
            config = {
                "input_dir": str(self.input_dir),
                "output_dir": str(self.output_dir),
                "failed_dir": str(self.failed_dir),
                "max_workers": workers,
                "batch_size": 2
            }
            
            # Measure performance
            start_time = time.time()
            start_memory = self._measure_memory_usage()
            
            pipeline = TIFFPipelineMain(config)
            result = pipeline.run()
            
            end_time = time.time()
            end_memory = self._measure_memory_usage()
            
            # Log results
            duration = end_time - start_time
            memory_used = end_memory - start_memory
            
            self.logger.info(f"Parallel processing with {workers} workers:")
            self.logger.info(f"  Duration: {duration:.2f} seconds")
            self.logger.info(f"  Memory used: {memory_used:.2f} MB")
            
            # Assert reasonable performance
            self.assertLess(duration, 30.0)  # Should process in under 30 seconds
            self.assertLess(memory_used, 2000)  # Should use less than 2GB

if __name__ == '__main__':
    unittest.main() 