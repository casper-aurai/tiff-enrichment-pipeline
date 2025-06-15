"""
MicaSense Vegetation Indices Module
Calculates various vegetation indices from multispectral data
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Optional
import logging

class VegetationIndices:
    """Calculates vegetation indices from multispectral data"""
    
    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger("MicaSenseIndices")
    
    def calculate_indices(self, image_path: str, image_set: Dict) -> Dict[str, str]:
        """
        Calculate vegetation indices from aligned multispectral image
        
        Args:
            image_path: Path to aligned multispectral image
            image_set: Dictionary containing image set information
            
        Returns:
            Dictionary mapping index names to output file paths
        """
        indices_paths = {}
        
        try:
            with rasterio.open(image_path) as src:
                # Read bands (1=Blue, 2=Green, 3=Red, 4=NIR, 5=RedEdge)
                blue = src.read(1).astype('float32')
                green = src.read(2).astype('float32')
                red = src.read(3).astype('float32')
                nir = src.read(4).astype('float32')
                red_edge = src.read(5).astype('float32')
                
                profile = src.profile.copy()
                profile.update({'count': 1, 'dtype': 'float32'})
                
                # Calculate requested indices
                if self.config['vegetation_indices']['ndvi']:
                    ndvi_path = self._calculate_ndvi(nir, red, image_set, profile)
                    indices_paths["NDVI"] = ndvi_path
                
                if self.config['vegetation_indices']['ndre']:
                    ndre_path = self._calculate_ndre(nir, red_edge, image_set, profile)
                    indices_paths["NDRE"] = ndre_path
                
                if self.config['vegetation_indices']['gndvi']:
                    gndvi_path = self._calculate_gndvi(nir, green, image_set, profile)
                    indices_paths["GNDVI"] = gndvi_path
                
                if self.config['vegetation_indices']['savi']:
                    savi_path = self._calculate_savi(nir, red, image_set, profile)
                    indices_paths["SAVI"] = savi_path
                
                if self.config['vegetation_indices']['msavi']:
                    msavi_path = self._calculate_msavi(nir, red, image_set, profile)
                    indices_paths["MSAVI"] = msavi_path
                
                if self.config['vegetation_indices']['evi']:
                    evi_path = self._calculate_evi(nir, red, blue, image_set, profile)
                    indices_paths["EVI"] = evi_path
                
                self.logger.info(f"Calculated vegetation indices for {image_set['name']}")
                
        except Exception as e:
            self.logger.error(f"Failed to calculate indices for {image_set['name']}: {e}")
            
        return indices_paths
    
    def _calculate_ndvi(self, nir: np.ndarray, red: np.ndarray, 
                       image_set: Dict, profile: Dict) -> str:
        """Calculate Normalized Difference Vegetation Index"""
        # Ensure data is in float32 and handle potential division by zero
        nir = nir.astype('float32')
        red = red.astype('float32')
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        
        # Calculate NDVI with improved handling of edge cases
        denominator = nir + red + epsilon
        ndvi = np.divide(
            (nir - red),
            denominator,
            out=np.zeros_like(nir),
            where=denominator > epsilon
        )
        
        # Clip values to valid NDVI range
        ndvi = np.clip(ndvi, -1.0, 1.0)
        
        # Mask out invalid values (e.g., where denominator was too small)
        ndvi[denominator <= epsilon] = -1.0
        
        output_path = self.output_dir / "indices" / f"{image_set['name']}_NDVI.tif"
        self._save_index(ndvi, output_path, profile, "NDVI")
        
        # Log statistics about vegetation coverage
        veg_mask = (ndvi > 0.2) & (ndvi <= 1.0)
        high_veg_mask = (ndvi > 0.5) & (ndvi <= 1.0)
        
        veg_percentage = float(np.sum(veg_mask) / ndvi.size * 100)
        high_veg_percentage = float(np.sum(high_veg_mask) / ndvi.size * 100)
        
        self.logger.info(f"NDVI statistics for {image_set['name']}:")
        self.logger.info(f"  Vegetation coverage (>0.2): {veg_percentage:.2f}%")
        self.logger.info(f"  High vegetation (>0.5): {high_veg_percentage:.2f}%")
        self.logger.info(f"  Mean NDVI: {np.mean(ndvi):.3f}")
        self.logger.info(f"  Min NDVI: {np.min(ndvi):.3f}")
        self.logger.info(f"  Max NDVI: {np.max(ndvi):.3f}")
        
        return str(output_path)
    
    def _calculate_ndre(self, nir: np.ndarray, red_edge: np.ndarray,
                       image_set: Dict, profile: Dict) -> str:
        """Calculate Normalized Difference Red Edge Index"""
        ndre = np.divide(
            (nir - red_edge),
            (nir + red_edge),
            out=np.zeros_like(nir),
            where=(nir + red_edge) != 0
        )
        
        output_path = self.output_dir / "indices" / f"{image_set['name']}_NDRE.tif"
        self._save_index(ndre, output_path, profile, "NDRE")
        return str(output_path)
    
    def _calculate_gndvi(self, nir: np.ndarray, green: np.ndarray,
                        image_set: Dict, profile: Dict) -> str:
        """Calculate Green Normalized Difference Vegetation Index"""
        gndvi = np.divide(
            (nir - green),
            (nir + green),
            out=np.zeros_like(nir),
            where=(nir + green) != 0
        )
        
        output_path = self.output_dir / "indices" / f"{image_set['name']}_GNDVI.tif"
        self._save_index(gndvi, output_path, profile, "GNDVI")
        return str(output_path)
    
    def _calculate_savi(self, nir: np.ndarray, red: np.ndarray,
                       image_set: Dict, profile: Dict) -> str:
        """Calculate Soil Adjusted Vegetation Index"""
        L = 0.5  # Soil brightness correction factor
        savi = np.divide(
            (nir - red) * (1 + L),
            (nir + red + L),
            out=np.zeros_like(nir),
            where=(nir + red + L) != 0
        )
        
        output_path = self.output_dir / "indices" / f"{image_set['name']}_SAVI.tif"
        self._save_index(savi, output_path, profile, "SAVI")
        return str(output_path)
    
    def _calculate_msavi(self, nir: np.ndarray, red: np.ndarray,
                        image_set: Dict, profile: Dict) -> str:
        """Calculate Modified Soil Adjusted Vegetation Index"""
        msavi = 0.5 * (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red)))
        
        output_path = self.output_dir / "indices" / f"{image_set['name']}_MSAVI.tif"
        self._save_index(msavi, output_path, profile, "MSAVI")
        return str(output_path)
    
    def _calculate_evi(self, nir: np.ndarray, red: np.ndarray, blue: np.ndarray,
                      image_set: Dict, profile: Dict) -> str:
        """Calculate Enhanced Vegetation Index"""
        G = 2.5  # Gain factor
        L = 1.0  # Soil brightness correction factor
        C1, C2 = 6.0, 7.5  # Aerosol resistance coefficients
        
        evi = G * np.divide(
            (nir - red),
            (nir + C1 * red - C2 * blue + L),
            out=np.zeros_like(nir),
            where=(nir + C1 * red - C2 * blue + L) != 0
        )
        
        output_path = self.output_dir / "indices" / f"{image_set['name']}_EVI.tif"
        self._save_index(evi, output_path, profile, "EVI")
        return str(output_path)
    
    def _save_index(self, index_data: np.ndarray, output_path: Path,
                   profile: Dict, index_name: str):
        """Save vegetation index to file"""
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(index_data, 1)
            dst.set_band_description(1, index_name) 