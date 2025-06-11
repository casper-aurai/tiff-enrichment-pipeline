"""
MicaSense Processing Module
Handles band alignment and radiometric calibration
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from typing import Dict, Optional
import logging

class BandProcessor:
    """Processes MicaSense bands (alignment and calibration)"""
    
    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger("MicaSenseProcessor")
    
    def align_bands(self, image_set: Dict) -> Optional[str]:
        """
        Align all bands of an image set and create a multi-band TIFF
        
        Args:
            image_set: Dictionary containing band information
            
        Returns:
            Path to aligned image if successful, None otherwise
        """
        try:
            aligned_files = []
            reference_band = None
            
            # Use band 3 (Red) as reference for alignment
            ref_path = image_set["bands"][3]
            
            with rasterio.open(ref_path) as ref_src:
                ref_profile = ref_src.profile.copy()
                ref_transform = ref_src.transform
                ref_crs = ref_src.crs
                ref_bounds = ref_src.bounds
                
                # Create output path
                output_name = f"{image_set['name']}_aligned.tif"
                output_path = self.output_dir / "aligned" / output_name
                
                # Prepare for multi-band output
                ref_profile.update({
                    'count': 5,
                    'dtype': 'float32'
                })
                
                with rasterio.open(output_path, 'w', **ref_profile) as dst:
                    for band_num in sorted(image_set["bands"].keys()):
                        band_path = image_set["bands"][band_num]
                        
                        with rasterio.open(band_path) as src:
                            if band_num == 3:  # Reference band
                                data = src.read(1).astype('float32')
                            else:
                                # Reproject to match reference
                                data = np.empty((ref_src.height, ref_src.width), dtype='float32')
                                reproject(
                                    source=rasterio.band(src, 1),
                                    destination=data,
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=ref_transform,
                                    dst_crs=ref_crs,
                                    resampling=Resampling.bilinear
                                )
                            
                            # Write band to output
                            dst.write(data, band_num)
                            
                            # Set band description
                            dst.set_band_description(
                                band_num, 
                                self.config['band_config'][band_num]["name"]
                            )
                
                self.logger.info(f"Successfully aligned {image_set['name']}")
                return str(output_path)
                
        except Exception as e:
            self.logger.error(f"Failed to align {image_set['name']}: {e}")
            return None
    
    def radiometric_calibration(self, image_path: str) -> Optional[str]:
        """
        Apply radiometric calibration to convert DN to reflectance
        
        Args:
            image_path: Path to aligned multispectral image
            
        Returns:
            Path to calibrated image if successful, None otherwise
        """
        try:
            if not self.config['processing']['radiometric_calibration']:
                return image_path
            
            with rasterio.open(image_path) as src:
                # Create output path
                output_name = Path(image_path).stem + "_calibrated.tif"
                output_path = self.output_dir / "calibrated" / output_name
                
                # Prepare output profile
                profile = src.profile.copy()
                profile.update({
                    'dtype': 'float32',
                    'nodata': None
                })
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    for band_num in range(1, src.count + 1):
                        # Read band data
                        data = src.read(band_num).astype('float32')
                        
                        # Apply calibration (simplified version)
                        # In practice, would use DLS data and panel measurements
                        calibrated_data = data * 0.0001  # Convert to reflectance
                        calibrated_data = np.clip(calibrated_data, 0, 1)
                        
                        # Write calibrated band
                        dst.write(calibrated_data, band_num)
                        
                        # Copy band description
                        dst.set_band_description(
                            band_num,
                            src.descriptions[band_num - 1]
                        )
                
                self.logger.info(f"Successfully calibrated {output_name}")
                return str(output_path)
                
        except Exception as e:
            self.logger.error(f"Failed to calibrate {image_path}: {e}")
            return None 