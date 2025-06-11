"""
MicaSense Core Processor Module
Main processing pipeline for MicaSense data
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from concurrent.futures import ProcessPoolExecutor
import traceback
import json
from datetime import datetime

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
    
    def find_image_sets(self, input_dir: Path) -> List[Dict]:
        """Find all MicaSense image sets in the input directory"""
        image_sets = []
        
        # Find all TIFF files
        tiff_files = list(input_dir.glob("**/*.tif")) + list(input_dir.glob("**/*.TIF"))
        
        # Group files by capture
        captures = {}
        for file_path in tiff_files:
            filename = file_path.name
            if "_" in filename:
                parts = filename.split("_")
                if len(parts) >= 3:
                    base_name = "_".join(parts[:-1])
                    band_num = parts[-1].split(".")[0]
                    
                    if base_name not in captures:
                        captures[base_name] = {}
                    
                    try:
                        band_int = int(band_num)
                        if 1 <= band_int <= 5:
                            captures[base_name][band_int] = str(file_path)
                    except ValueError:
                        continue
        
        # Filter complete sets
        for base_name, bands in captures.items():
            if len(bands) == 5 and all(i in bands for i in range(1, 6)):
                sorted_bands = {k: bands[k] for k in sorted(bands.keys())}
                image_sets.append({
                    "name": base_name,
                    "bands": sorted_bands,
                    "path": Path(bands[1]).parent
                })
        
        self.logger.info(f"Found {len(image_sets)} complete MicaSense image sets")
        return image_sets
    
    def process_single_set(self, image_set: Dict) -> Dict:
        """Process a single image set"""
        result = {
            "name": image_set["name"],
            "status": "failed",
            "error": None,
            "error_type": None,
            "failed_stage": None,
            "processing_time": 0,
            "band_stats": {},
            "index_stats": {}
        }
        
        try:
            # Validate image set
            issues = self.validator.validate_image_set(image_set)
            if issues:
                raise ValidationError(f"Validation failed: {', '.join(issues)}")
            
            # Align bands
            aligned_path = self._align_bands(image_set)
            if not aligned_path:
                raise BandError("Band alignment failed")
            
            # Apply radiometric calibration
            if self.config['processing']['radiometric_calibration']:
                calibrated_path = self._calibrate_image(aligned_path)
                if not calibrated_path:
                    raise CalibrationError("Radiometric calibration failed")
                aligned_path = calibrated_path
            
            # Calculate vegetation indices
            if self.config['processing']['generate_indices']:
                indices_paths = self._calculate_indices(aligned_path, image_set)
                result["index_stats"] = self._calculate_index_statistics(indices_paths)
            
            # Generate quality report
            if self.config['quality_control']['generate_histograms']:
                self._generate_quality_report(image_set)
            
            # Save metadata
            if self.config['output_options']['save_metadata']:
                self._save_metadata(image_set)
            
            result["status"] = "success"
            
        except MicaSenseError as e:
            result["error"] = str(e)
            result["error_type"] = e.__class__.__name__
            result["failed_stage"] = "processing"
            self.logger.error(f"Failed to process {image_set['name']}: {str(e)}")
        
        except Exception as e:
            result["error"] = str(e)
            result["error_type"] = "UnexpectedError"
            result["failed_stage"] = "processing"
            self.logger.error(f"Unexpected error processing {image_set['name']}: {str(e)}")
            self.logger.debug(traceback.format_exc())
        
        return result
    
    def _align_bands(self, image_set: Dict) -> Optional[str]:
        """Align all bands of an image set"""
        try:
            # Use band 3 (Red) as reference
            ref_path = image_set["bands"][3]
            
            with rasterio.open(ref_path) as ref_src:
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
                
                return str(output_path)
                
        except Exception as e:
            self.logger.error(f"Failed to align {image_set['name']}: {str(e)}")
            return None
    
    def _calibrate_image(self, image_path: str) -> Optional[str]:
        """Apply radiometric calibration"""
        try:
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
                        calibrated_data = data * 0.0001  # Convert to reflectance
                        calibrated_data = np.clip(calibrated_data, 0, 1)
                        
                        # Write calibrated band
                        dst.write(calibrated_data, band_num)
                        
                        # Copy band description
                        dst.set_band_description(
                            band_num,
                            src.descriptions[band_num - 1]
                        )
                
                return str(output_path)
                
        except Exception as e:
            self.logger.error(f"Failed to calibrate {image_path}: {str(e)}")
            return None
    
    def _calculate_indices(self, image_path: str, image_set: Dict) -> Dict[str, str]:
        """Calculate vegetation indices"""
        indices_paths = {}
        
        try:
            with rasterio.open(image_path) as src:
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
                
                # Add more indices as needed...
                
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
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(index, 1)
            dst.set_band_description(1, index_name)
        
        return str(output_path)
    
    def _calculate_index_statistics(self, indices_paths: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each vegetation index"""
        stats = {}
        
        for index_name, path in indices_paths.items():
            try:
                with rasterio.open(path) as src:
                    data = src.read(1)
                    stats[index_name] = {
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                        "mean": float(np.mean(data)),
                        "std": float(np.std(data))
                    }
            except Exception as e:
                self.logger.warning(f"Failed to calculate statistics for {index_name}: {str(e)}")
        
        return stats
    
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
                with rasterio.open(band_path) as src:
                    data = src.read(1)
                    
                    # Calculate basic statistics
                    stats = {
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                        "mean": float(np.mean(data)),
                        "std": float(np.std(data)),
                        "zeros_ratio": float(np.sum(data == 0) / data.size)
                    }
                    
                    report["bands"][self.config['band_config'][band_num]["name"]] = stats
                    
                    # Generate histogram if enabled
                    if self.config['quality_control']['generate_histograms']:
                        hist_path = self.output_dir / "quality_reports" / f"{image_set['name']}_{band_num}_histogram.png"
                        self._generate_histogram(data, hist_path, self.config['band_config'][band_num]["name"])
            
            # Save report
            report_path = self.output_dir / "quality_reports" / f"{image_set['name']}_quality_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to generate quality report for {image_set['name']}: {str(e)}")
    
    def _generate_histogram(self, data: np.ndarray, output_path: Path, band_name: str):
        """Generate histogram visualization"""
        try:
            import matplotlib.pyplot as plt
            
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
            with rasterio.open(image_set["bands"][1]) as src:
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