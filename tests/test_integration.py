"""
Integration tests for the TIFF Enrichment Pipeline
Tests the complete pipeline workflow with sample data
"""

import unittest
from pathlib import Path
import shutil
import tempfile
import json
import numpy as np
from datetime import datetime
import os

from src.pipeline.main import TIFFPipelineMain
from src.pipeline.micasense.core.processor import MicaSenseProcessor
from src.pipeline.health import health_check

class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline workflow"""
    
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
        
        # Create test configuration
        cls.config = {
            "input_dir": str(cls.input_dir),
            "output_dir": str(cls.output_dir),
            "failed_dir": str(cls.failed_dir),
            "max_workers": 2,
            "batch_size": 1,
            "logging": {
                "level": "DEBUG",
                "file": str(cls.test_dir / "test.log")
            }
        }
        
        # Create sample data
        cls._create_sample_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_sample_data(cls):
        """Create sample TIFF files for testing"""
        # Create a MicaSense set
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bands = ["Blue", "Green", "Red", "NIR", "RedEdge"]
        
        for i, band in enumerate(bands, 1):
            # Create synthetic image data
            data = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
            
            # Save as TIFF with metadata
            from rasterio.io import MemoryFile
            from rasterio.transform import from_origin
            
            transform = from_origin(0, 0, 1, 1)
            profile = {
                'driver': 'GTiff',
                'dtype': 'uint16',
                'count': 1,
                'height': 100,
                'width': 100,
                'transform': transform,
                'crs': 'EPSG:4326'
            }
            
            with MemoryFile() as memfile:
                with memfile.open(**profile) as dst:
                    dst.write(data, 1)
                    dst.set_band_description(1, band)
                    # Add metadata
                    dst.update_tags(
                        EXIF_DateTime=datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                        EXIF_GPS_Lat=40.7128,
                        EXIF_GPS_Lon=-74.0060
                    )
                
                # Save to file
                output_path = cls.input_dir / f"IMG_{timestamp}_{i}.tif"
                with open(output_path, 'wb') as f:
                    f.write(memfile.read())
        
        # Create a regular TIFF
        data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        from PIL import Image
        img = Image.fromarray(data)
        img.save(cls.input_dir / "regular_image.tif")
    
    def test_health_check(self):
        """Test health check functionality"""
        # Set up test environment variables
        os.environ['INPUT_DIR'] = str(self.input_dir)
        os.environ['OUTPUT_DIR'] = str(self.output_dir)
        
        # Run health check
        result = health_check()
        self.assertTrue(result)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = TIFFPipelineMain(self.config)
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.input_dir, self.input_dir)
        self.assertEqual(pipeline.output_dir, self.output_dir)
    
    def test_file_type_detection(self):
        """Test file type detection"""
        pipeline = TIFFPipelineMain(self.config)
        file_types = pipeline.detect_file_types()
        
        # Should detect both MicaSense and regular TIFFs
        self.assertGreater(len(file_types['micasense']), 0)
        self.assertGreater(len(file_types['regular_tiff']), 0)
    
    def test_complete_pipeline(self):
        """Test complete pipeline execution"""
        pipeline = TIFFPipelineMain(self.config)
        result = pipeline.run()
        
        # Check results
        self.assertIn('status', result)
        self.assertIn('processing_results', result)
        
        # Check output files
        micasense_output = self.output_dir / "micasense"
        self.assertTrue(micasense_output.exists())
        
        # Check for processed files
        processed_files = list(micasense_output.glob("**/*.tif"))
        self.assertGreater(len(processed_files), 0)
        
        # Check for metadata
        metadata_files = list(micasense_output.glob("**/*.json"))
        self.assertGreater(len(metadata_files), 0)
    
    def test_error_handling(self):
        """Test pipeline error handling"""
        # Create an invalid file
        invalid_file = self.input_dir / "invalid.tif"
        with open(invalid_file, 'w') as f:
            f.write("This is not a valid TIFF file")
        
        pipeline = TIFFPipelineMain(self.config)
        result = pipeline.run()
        
        # Check that invalid file was moved to failed directory
        self.assertTrue((self.failed_dir / "invalid.tif").exists())
        self.assertFalse(invalid_file.exists())
        
        # Check that other files were still processed
        self.assertIn('processing_results', result)

if __name__ == '__main__':
    unittest.main() 