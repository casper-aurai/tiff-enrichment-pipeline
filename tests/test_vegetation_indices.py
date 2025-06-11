import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from src.pipeline.micasense.processing.indices import VegetationIndices

class TestVegetationIndices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.output_dir = cls.test_dir / "output"
        cls.output_dir.mkdir()
        
        # Test configuration
        cls.config = {
            'vegetation_indices': {
                'ndvi': True,
                'ndre': True,
                'gndvi': True,
                'savi': True,
                'msavi': True,
                'evi': True
            }
        }
        
        # Create test data
        cls.nir = np.array([[0.8, 0.7], [0.6, 0.5]], dtype='float32')
        cls.red = np.array([[0.2, 0.3], [0.4, 0.5]], dtype='float32')
        cls.green = np.array([[0.3, 0.4], [0.5, 0.6]], dtype='float32')
        cls.red_edge = np.array([[0.4, 0.5], [0.6, 0.7]], dtype='float32')
        cls.blue = np.array([[0.1, 0.2], [0.3, 0.4]], dtype='float32')
        
        # Initialize processor
        cls.processor = VegetationIndices(cls.config, cls.output_dir)
    
    def test_ndvi_calculation(self):
        """Test NDVI calculation"""
        ndvi = self.processor._calculate_ndvi(self.nir, self.red)
        expected = np.array([[0.6, 0.4], [0.2, 0.0]], dtype='float32')
        np.testing.assert_array_almost_equal(ndvi, expected, decimal=2)
    
    def test_ndre_calculation(self):
        """Test NDRE calculation"""
        ndre = self.processor._calculate_ndre(self.nir, self.red_edge)
        expected = np.array([[0.333, 0.167], [0.0, -0.167]], dtype='float32')
        np.testing.assert_array_almost_equal(ndre, expected, decimal=2)
    
    def test_gndvi_calculation(self):
        """Test GNDVI calculation"""
        gndvi = self.processor._calculate_gndvi(self.nir, self.green)
        expected = np.array([[0.455, 0.273], [0.091, -0.091]], dtype='float32')
        np.testing.assert_array_almost_equal(gndvi, expected, decimal=2)
    
    def test_savi_calculation(self):
        """Test SAVI calculation"""
        savi = self.processor._calculate_savi(self.nir, self.red)
        expected = np.array([[0.6, 0.4], [0.2, 0.0]], dtype='float32')
        np.testing.assert_array_almost_equal(savi, expected, decimal=2)
    
    def test_msavi_calculation(self):
        """Test MSAVI calculation"""
        msavi = self.processor._calculate_msavi(self.nir, self.red)
        # MSAVI calculation is more complex, test basic properties
        self.assertTrue(np.all(msavi >= -1) and np.all(msavi <= 1))
    
    def test_evi_calculation(self):
        """Test EVI calculation"""
        evi = self.processor._calculate_evi(self.nir, self.red, self.blue)
        # EVI calculation is more complex, test basic properties
        self.assertTrue(np.all(evi >= -1) and np.all(evi <= 1))
    
    def test_zero_division_handling(self):
        """Test handling of zero division in index calculations"""
        zero_nir = np.zeros_like(self.nir)
        zero_red = np.zeros_like(self.red)
        
        ndvi = self.processor._calculate_ndvi(zero_nir, zero_red)
        self.assertTrue(np.all(ndvi == 0))
    
    def test_invalid_input_handling(self):
        """Test handling of invalid input values"""
        invalid_nir = np.array([[np.nan, np.inf], [-np.inf, 0]], dtype='float32')
        invalid_red = np.array([[0, 0], [0, 0]], dtype='float32')
        
        ndvi = self.processor._calculate_ndvi(invalid_nir, invalid_red)
        self.assertTrue(np.all(np.isfinite(ndvi)))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main() 