"""
Test script for MicaSense processor
"""

import unittest
from pathlib import Path
import shutil
import tempfile
import json
import numpy as np
from datetime import datetime

from src.pipeline.micasense.core.processor import MicaSenseProcessor
from src.pipeline.micasense.core.config import load_config, validate_config

class TestMicaSenseProcessor(unittest.TestCase):
    """Test cases for MicaSense processor"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create temporary directories
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.input_dir = cls.test_dir / "input"
        cls.output_dir = cls.test_dir / "output"
        cls.input_dir.mkdir()
        cls.output_dir.mkdir()
        
        # Create test configuration
        cls.config = {
            "quality_control": {
                "generate_histograms": True,
                "validate_dimensions": True,
                "validate_reflectance_range": True
            },
            "vegetation_indices": {
                "ndvi": True,
                "ndre": True,
                "gndvi": True,
                "savi": True,
                "msavi": True,
                "evi": True
            },
            "output": {
                "directory": str(cls.output_dir),
                "save_individual_bands": True,
                "generate_thumbnails": True,
                "generate_visualizations": True
            },
            "processing": {
                "max_workers": 2,
                "batch_size": 1
            },
            "logging": {
                "level": "INFO",
                "file": str(cls.test_dir / "test.log")
            }
        }
        
        # Save configuration
        config_path = cls.test_dir / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(cls.config, f)
        
        # Create test image sets
        cls._create_test_images()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_images(cls):
        """Create synthetic test images"""
        # Create a test image set
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bands = ["Blue", "Green", "Red", "NIR", "RedEdge"]
        
        for i, band in enumerate(bands):
            # Create synthetic image data
            data = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
            
            # Save as TIFF
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
                
                # Save to file
                output_path = cls.input_dir / f"{timestamp}_{band}.tif"
                with open(output_path, 'wb') as f:
                    f.write(memfile.read())
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        processor = MicaSenseProcessor(self.config)
        self.assertIsNotNone(processor)
        self.assertEqual(processor.config, self.config)
    
    def test_image_set_discovery(self):
        """Test finding image sets"""
        from src.pipeline.micasense.__main__ import find_image_sets
        
        image_sets = find_image_sets(self.input_dir)
        self.assertGreater(len(image_sets), 0)
        
        # Verify image set structure
        image_set = image_sets[0]
        self.assertIn("name", image_set)
        self.assertIn("band_files", image_set)
        self.assertEqual(len(image_set["band_files"]), 5)
    
    def test_processing_pipeline(self):
        """Test complete processing pipeline"""
        processor = MicaSenseProcessor(self.config)
        
        # Find image sets
        from src.pipeline.micasense.__main__ import find_image_sets
        image_sets = find_image_sets(self.input_dir)
        
        # Process images
        results = processor.process_all(image_sets)
        
        # Verify results
        self.assertGreater(len(results), 0)
        result = results[0]
        
        # Check output files
        self.assertTrue(Path(result["aligned_image"]).exists())
        self.assertTrue(Path(result["calibrated_image"]).exists())
        
        # Check indices
        self.assertIn("indices", result)
        for index_path in result["indices"].values():
            self.assertTrue(Path(index_path).exists())
        
        # Check thumbnails
        self.assertIn("thumbnails", result)
        for thumb_path in result["thumbnails"].values():
            self.assertTrue(Path(thumb_path).exists())
        
        # Check metadata
        self.assertIn("metadata", result)
        self.assertTrue(Path(result["metadata"]).exists())
        
        # Verify metadata content
        with open(result["metadata"]) as f:
            metadata = json.load(f)
            self.assertIn("image_set", metadata)
            self.assertIn("processing", metadata)
            self.assertIn("outputs", metadata)

if __name__ == '__main__':
    unittest.main() 