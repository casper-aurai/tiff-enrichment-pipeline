"""
MicaSense Core Processor Module
Main processing pipeline for MicaSense data
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
import rasterio
import rasterio.warp

from pipeline.utils.rasterio_utils import RasterioManager
from .errors import (
    MicaSenseError, ProcessingError, ValidationError,
    BandError, CalibrationError, OutputError
)
from .validation import MicaSenseValidator
from .config import MicaSenseConfig

class MicaSenseProcessor:
    """Main processor for MicaSense data"""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger("MicaSenseProcessor")
        self.validator = MicaSenseValidator(config, self.logger)
        self.rasterio_manager = RasterioManager()
        
        # Create output directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary output directories"""
        dirs = [
            'aligned',
            'calibrated',
            'indices',
            'metadata',
            'quality_reports',
            'thumbnails',
            'visualizations'
        ]
        
        for dir_name in dirs:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_single_set(self, image_set: Dict) -> Dict:
        """Process a single MicaSense image set"""
        try:
            # Validate image set
            issues = self.validator.validate_image_set(image_set)
            if issues:
                raise ValidationError(f"Validation failed: {', '.join(issues)}")
            
            # Align images
            aligned_path = self._align_images(image_set)
            if not aligned_path:
                raise ProcessingError("Failed to align images")
            
            # Calibrate aligned image
            calibrated_path = self._calibrate_image(aligned_path)
            if not calibrated_path:
                raise CalibrationError("Failed to calibrate image")
            
            # Calculate indices
            indices_paths = self._calculate_indices(calibrated_path, image_set)
            if not indices_paths:
                raise ProcessingError("Failed to calculate indices")
            
            # Generate quality report
            self._generate_quality_report(image_set)
            
            # Save metadata
            self._save_metadata(image_set)
            
            return {
                'status': 'success',
                'aligned_path': aligned_path,
                'calibrated_path': calibrated_path,
                'indices_paths': indices_paths
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process image set {image_set['name']}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _align_images(self, image_set: Dict) -> Optional[str]:
        """Align images using reference band"""
        try:
            ref_path = image_set["bands"][3]  # Use band 3 as reference
            
            with self.rasterio_manager.safe_open(Path(ref_path)) as ref_src:
                ref_profile = ref_src.profile.copy()
                ref_transform = ref_src.transform
                ref_crs = ref_src.crs
                
                # Create output path
                output_name = f"{image_set['name']}_aligned.tif"
                output_path = self.output_dir / "aligned" / output_name
                
                # Prepare for multi-band output
                ref_profile.update({
                    'count': 5,
                    'dtype': 'float32'
                })
                
                # Process each band
                for band_num in sorted(image_set["bands"].keys()):
                    band_path = image_set["bands"][band_num]
                    
                    with self.rasterio_manager.safe_open(Path(band_path)) as src:
                        if band_num == 3:  # Reference band
                            data = src.read(1).astype('float32')
                        else:
                            # Reproject to match reference
                            data = np.empty((ref_src.height, ref_src.width), dtype='float32')
                            rasterio.warp.reproject(
                                source=rasterio.band(src, 1),
                                destination=data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=ref_transform,
                                dst_crs=ref_crs,
                                resampling=rasterio.warp.Resampling.bilinear
                            )
                        
                        # Write band to output
                        self.rasterio_manager.safe_write(
                            output_path,
                            data,
                            ref_profile
                        )
                
                return str(output_path)
                
        except Exception as e:
            self.logger.error(f"Failed to align {image_set['name']}: {str(e)}")
            return None
    
    def _calibrate_image(self, image_path: str) -> Optional[str]:
        """Apply radiometric calibration"""
        try:
            with self.rasterio_manager.safe_open(Path(image_path)) as src:
                # Create output path
                output_name = Path(image_path).stem + "_calibrated.tif"
                output_path = self.output_dir / "calibrated" / output_name
                
                # Prepare output profile
                profile = src.profile.copy()
                profile.update({
                    'dtype': 'float32',
                    'nodata': None
                })
                
                # Process each band
                for band_num in range(1, src.count + 1):
                    # Read band data
                    data = src.read(band_num).astype('float32')
                    
                    # Apply calibration (simplified version)
                    calibrated_data = data * 0.0001  # Convert to reflectance
                    calibrated_data = np.clip(calibrated_data, 0, 1)
                    
                    # Write calibrated band
                    self.rasterio_manager.safe_write(
                        output_path,
                        calibrated_data,
                        profile
                    )
                
                return str(output_path)
                
        except Exception as e:
            self.logger.error(f"Failed to calibrate {image_path}: {str(e)}")
            return None
    
    def _calculate_indices(self, image_path: str, image_set: Dict) -> Dict[str, str]:
        """Calculate vegetation indices"""
        indices_paths = {}
        
        try:
            with self.rasterio_manager.safe_open(Path(image_path)) as src:
                # Read bands
                blue = src.read(1).astype('float32')
                green = src.read(2).astype('float32')
                red = src.read(3).astype('float32')
                nir = src.read(4).astype('float32')
                red_edge = src.read(5).astype('float32')
                
                profile = src.profile.copy()
                profile.update({'count': 1, 'dtype': 'float32'})
                
                # Calculate indices based on configuration
                if self.config['vegetation_indices']['ndvi']:
                    ndvi = self._calculate_ndvi(nir, red)
                    indices_paths["NDVI"] = self._save_index(ndvi, image_set["name"], "NDVI", profile)
                
                if self.config['vegetation_indices']['ndre']:
                    ndre = self._calculate_ndre(nir, red_edge)
                    indices_paths["NDRE"] = self._save_index(ndre, image_set["name"], "NDRE", profile)
                
                if self.config['vegetation_indices']['gndvi']:
                    gndvi = self._calculate_gndvi(nir, green)
                    indices_paths["GNDVI"] = self._save_index(gndvi, image_set["name"], "GNDVI", profile)
                
        except Exception as e:
            self.logger.error(f"Failed to calculate indices for {image_set['name']}: {str(e)}")
        
        return indices_paths
    
    def _calculate_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Calculate NDVI"""
        return np.divide(
            (nir - red),
            (nir + red),
            out=np.zeros_like(nir),
            where=(nir + red) != 0
        )
    
    def _calculate_ndre(self, nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
        """Calculate NDRE"""
        return np.divide(
            (nir - red_edge),
            (nir + red_edge),
            out=np.zeros_like(nir),
            where=(nir + red_edge) != 0
        )
    
    def _calculate_gndvi(self, nir: np.ndarray, green: np.ndarray) -> np.ndarray:
        """Calculate GNDVI"""
        return np.divide(
            (nir - green),
            (nir + green),
            out=np.zeros_like(nir),
            where=(nir + green) != 0
        )
    
    def _save_index(self, index: np.ndarray, name: str, index_name: str, profile: Dict) -> str:
        """Save vegetation index to file"""
        output_path = self.output_dir / "indices" / f"{name}_{index_name}.tif"
        
        self.rasterio_manager.safe_write(output_path, index, profile)
        return str(output_path)
    
    def _generate_quality_report(self, image_set: Dict):
        """Generate quality control report"""
        try:
            report = {
                "name": image_set["name"],
                "timestamp": datetime.now().isoformat(),
                "bands": {},
                "quality_metrics": {}
            }
            
            for band_num, band_path in image_set["bands"].items():
                stats = self.rasterio_manager.get_statistics(Path(band_path))
                report["bands"][self.config['band_config'][band_num]["name"]] = stats
                
                # Generate histogram if enabled
                if self.config['quality_control']['generate_histograms']:
                    hist_path = self.output_dir / "quality_reports" / f"{image_set['name']}_{band_num}_histogram.png"
                    self._generate_histogram(Path(band_path), hist_path, self.config['band_config'][band_num]["name"])
            
            # Save report
            report_path = self.output_dir / "quality_reports" / f"{image_set['name']}_quality_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to generate quality report for {image_set['name']}: {str(e)}")
    
    def _generate_histogram(self, file_path: Path, output_path: Path, band_name: str):
        """Generate histogram visualization"""
        try:
            import matplotlib.pyplot as plt
            
            with self.rasterio_manager.safe_open(file_path) as src:
                data = src.read(1)
                
                plt.figure(figsize=(10, 6))
                plt.hist(data.flatten(), bins=256, range=(0, 65535))
                plt.title(f"Histogram - {band_name} Band")
                plt.xlabel("Pixel Value")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                
                plt.savefig(output_path)
                plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate histogram: {str(e)}")
    
    def _save_metadata(self, image_set: Dict):
        """Save metadata for an image set"""
        try:
            metadata = {
                "name": image_set["name"],
                "timestamp": datetime.now().isoformat(),
                "bands": {},
                "processing_info": {
                    "config": self.config
                }
            }
            
            # Extract metadata from first band
            with self.rasterio_manager.safe_open(Path(image_set["bands"][1])) as src:
                metadata["bands"] = {
                    self.config['band_config'][band_num]["name"]: {
                        "path": str(band_path),
                        "dimensions": (src.width, src.height),
                        "crs": str(src.crs) if src.crs else None,
                        "transform": list(src.transform)
                    }
                    for band_num, band_path in image_set["bands"].items()
                }
            
            # Save metadata
            metadata_path = self.output_dir / "metadata" / f"{image_set['name']}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save metadata for {image_set['name']}: {str(e)}") 