"""
MicaSense Core Processor Module
Main processing pipeline for MicaSense data
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
import rasterio
import rasterio.warp
import subprocess
import re
from dataclasses import dataclass
import math
import os
import glob

from pipeline.utils.rasterio_utils import RasterioManager
from .errors import (
    MicaSenseError, ProcessingError, ValidationError,
    BandError, CalibrationError, OutputError, GPSError
)
from .validation import MicaSenseValidator
from .config import MicaSenseConfig
from pipeline.utils.gps_utils import extract_gps_info

@dataclass
class GPSInfo:
    """GPS information extracted from TIFF files"""
    latitude: float
    longitude: float
    altitude: float
    timestamp: datetime
    crs: str = "EPSG:4326"  # Default to WGS84

class MicaSenseProcessor:
    """Main processor for MicaSense data"""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        # Force output root to /data/output/micasense
        self.output_root = Path("/data/output/micasense")
        if output_dir != self.output_root:
            self.output_root = output_dir
        self.output_dir = self.output_root
        self.logger = logging.getLogger("MicaSenseProcessor")
        self.validator = MicaSenseValidator(config, self.logger)
        self.rasterio_manager = RasterioManager()
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
            'visualizations',
            'individual_bands'
        ]
        for dir_name in dirs:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _get_output_path(self, subfolder: str, filename: str) -> Path:
        path = self.output_dir / subfolder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def _assign_crs(self, image_path: Path, gps_info: dict, camera_params: dict) -> dict:
        """Assign CRS and transform to an image based on GPS and camera parameters"""
        try:
            with rasterio.open(image_path) as src:
                # Get image dimensions
                width, height = src.width, src.height
                
                # Extract GPS info
                lat = gps_info.get('latitude')
                lon = gps_info.get('longitude')
                alt = gps_info.get('altitude', 0)
                
                if not lat or not lon:
                    self.logger.warning(f"No GPS coordinates found in {image_path}")
                    return None
                
                # Extract camera parameters
                focal_length = camera_params.get('focal_length', 5.4)  # mm
                sensor_width = camera_params.get('sensor_width', 4.8)  # mm
                sensor_height = camera_params.get('sensor_height', 3.6)  # mm
                
                # Calculate ground sample distance (GSD)
                # GSD = (sensor_width * altitude) / (focal_length * image_width)
                gsd_x = (sensor_width * alt) / (focal_length * width)
                gsd_y = (sensor_height * alt) / (focal_length * height)
                
                # Calculate pixel size in degrees using proper latitude-based conversion
                # Earth radius at equator: 6378137.0 meters
                # Earth radius at poles: 6356752.3 meters
                # Use average radius for simplicity
                earth_radius = 6371000.0  # meters
                
                # Calculate meters per degree at the given latitude
                meters_per_degree_lon = (2 * math.pi * earth_radius * math.cos(math.radians(lat))) / 360.0
                meters_per_degree_lat = (2 * math.pi * earth_radius) / 360.0
                
                # Convert GSD to degrees
                pixel_size_x = gsd_x / meters_per_degree_lon
                pixel_size_y = gsd_y / meters_per_degree_lat
                
                # Calculate the center of the image in geographic coordinates
                # The GPS coordinates represent the center of the image
                center_lon = lon
                center_lat = lat
                
                # Calculate the top-left corner coordinates
                # Move half the image width/height from the center
                top_left_lon = center_lon - (width * pixel_size_x / 2)
                top_left_lat = center_lat + (height * pixel_size_y / 2)
                
                # Create transform from the top-left corner
                transform = rasterio.transform.from_origin(
                    top_left_lon,  # west
                    top_left_lat,  # north
                    pixel_size_x,  # x pixel size
                    -pixel_size_y  # y pixel size (negative because y increases downward)
                )
                
                # Create new profile with CRS and transform
                profile = src.profile.copy()
                profile.update({
                    'crs': rasterio.crs.CRS.from_epsg(4326),  # WGS84
                    'transform': transform
                })
                
                # Create output filename
                output_path = image_path.parent / f"{image_path.stem}_georeferenced{image_path.suffix}"
                
                # Write georeferenced image
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(src.read())
                
                self.logger.debug(f"Georeferenced output: {output_path}")
                self.logger.debug(f"Transform: {transform}")
                self.logger.debug(f"CRS: {profile['crs']}")
                
                return {
                    'output_path': output_path,
                    'transform': transform,
                    'crs': profile['crs'],
                    'gsd_x': gsd_x,
                    'gsd_y': gsd_y,
                    'pixel_size_x': pixel_size_x,
                    'pixel_size_y': pixel_size_y,
                    'center_lon': center_lon,
                    'center_lat': center_lat,
                    'altitude': alt,
                    'focal_length': focal_length,
                    'sensor_width': sensor_width,
                    'sensor_height': sensor_height,
                    'image_width': width,
                    'image_height': height,
                    'meters_per_degree_lon': meters_per_degree_lon,
                    'meters_per_degree_lat': meters_per_degree_lat
                }
                
        except Exception as e:
            self.logger.error(f"Error assigning CRS to {image_path}: {str(e)}")
            return None
    
    def process_single_set(self, image_set: Dict) -> Dict:
        """Process a single MicaSense image set"""
        try:
            # Extract GPS information from reference band using shared utility
            ref_path = image_set["bands"][3]  # Use band 3 as reference
            gps_info = extract_gps_info(ref_path)
            if not gps_info:
                self.logger.warning(f"No GPS information found in {ref_path}")
            
            # Assign CRS and transform to all bands if GPS info is available
            if gps_info:
                for band_num, band_path in image_set["bands"].items():
                    with rasterio.open(band_path, 'r+') as src:
                        if not src.crs:
                            src.crs = rasterio.crs.CRS.from_epsg(4326)
                        if not src.transform or src.transform.is_identity:
                            # Calculate proper transform based on GPS and camera parameters
                            width, height = src.width, src.height
                            focal_length = self.config['camera_params'].get('focal_length', 5.4)
                            sensor_width = self.config['camera_params'].get('sensor_width', 4.8)
                            sensor_height = self.config['camera_params'].get('sensor_height', 3.6)
                            alt = gps_info.get('altitude', 0)
                            
                            # Calculate GSD
                            gsd_x = (sensor_width * alt) / (focal_length * width)
                            gsd_y = (sensor_height * alt) / (focal_length * height)
                            
                            # Calculate pixel size in degrees using proper latitude-based conversion
                            earth_radius = 6371000.0  # meters
                            lat = gps_info["latitude"]
                            
                            # Calculate meters per degree at the given latitude
                            meters_per_degree_lon = (2 * math.pi * earth_radius * math.cos(math.radians(lat))) / 360.0
                            meters_per_degree_lat = (2 * math.pi * earth_radius) / 360.0
                            
                            # Convert GSD to degrees
                            pixel_size_x = gsd_x / meters_per_degree_lon
                            pixel_size_y = gsd_y / meters_per_degree_lat
                            
                            # Calculate the center of the image in geographic coordinates
                            center_lon = gps_info["longitude"]
                            center_lat = gps_info["latitude"]
                            
                            # Calculate the top-left corner coordinates
                            top_left_lon = center_lon - (width * pixel_size_x / 2)
                            top_left_lat = center_lat + (height * pixel_size_y / 2)
                            
                            # Create transform from the top-left corner
                            transform = rasterio.transform.from_origin(
                                top_left_lon,  # west
                                top_left_lat,  # north
                                pixel_size_x,  # x pixel size
                                -pixel_size_y  # y pixel size (negative because y increases downward)
                            )
                            
                            src.transform = transform
                            self.logger.info(f"Assigned CRS EPSG:4326 and transform to {band_path}")
            
            # Align images (CRS/transform will be propagated)
            aligned_path = self._align_images(image_set)
            if not aligned_path:
                raise ProcessingError("Failed to align images")
            
            # Calibrate using aligned image (CRS/transform will be propagated)
            calibrated_path = self._calibrate_image(aligned_path)
            if not calibrated_path:
                raise CalibrationError("Failed to calibrate image")
            
            # Calculate indices using calibrated image (CRS/transform will be propagated)
            indices_paths = self._calculate_indices(calibrated_path, image_set)
            if not indices_paths:
                raise ProcessingError("Failed to calculate indices")
            
            # Generate quality report
            self._generate_quality_report(image_set)
            
            # Save metadata
            metadata = {
                'image_set': image_set,
                'gps_info': gps_info if gps_info else None,
                'processing_timestamp': datetime.now().isoformat()
            }
            self._save_metadata(metadata)
            
            # Final validation of all processed outputs
            validation_results = {}
            
            # Validate aligned image
            aligned_issues = self.validator.validate_tiff_file(Path(aligned_path))
            if aligned_issues:
                validation_results['aligned'] = aligned_issues
            
            # Validate calibrated image
            calibrated_issues = self.validator.validate_tiff_file(Path(calibrated_path))
            if calibrated_issues:
                validation_results['calibrated'] = calibrated_issues
            
            # Validate index files
            for index_name, index_path in indices_paths.items():
                index_issues = self.validator.validate_tiff_file(Path(index_path))
                if index_issues:
                    validation_results[f'index_{index_name}'] = index_issues
            
            # If any validation failed, raise error with all issues
            if validation_results:
                error_msg = "Validation failed for processed files:\n"
                for file_type, issues in validation_results.items():
                    error_msg += f"\n{file_type}:\n" + "\n".join(f"  - {issue}" for issue in issues)
                raise ValidationError(error_msg)
            
            return {
                'status': 'success',
                'aligned_path': aligned_path,
                'calibrated_path': calibrated_path,
                'indices_paths': indices_paths,
                'gps_info': gps_info if gps_info else None
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
                height, width = ref_src.height, ref_src.width
                
                # Create output path
                output_name = f"{image_set['name']}_aligned.tif"
                output_path = self._get_output_path("aligned", output_name)
                
                # Prepare for multi-band output
                ref_profile.update({
                    'count': 5,
                    'dtype': 'uint16',
                    'crs': ref_crs,
                    'transform': ref_transform
                })
                
                # Collect all bands into a single array
                all_bands = np.zeros((5, height, width), dtype='float32')
                
                for idx, band_num in enumerate(sorted(image_set["bands"].keys())):
                    band_path = image_set["bands"][band_num]
                    with self.rasterio_manager.safe_open(Path(band_path)) as src:
                        if band_num == 3:  # Reference band
                            data = src.read(1).astype('float32')
                        else:
                            # Reproject to match reference
                            data = np.empty((height, width), dtype='float32')
                            rasterio.warp.reproject(
                                source=rasterio.band(src, 1),
                                destination=data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=ref_transform,
                                dst_crs=ref_crs,
                                resampling=rasterio.warp.Resampling.bilinear
                            )
                        all_bands[idx, :, :] = data
                
                # Scale and clip to uint16
                all_bands_uint16 = np.clip(all_bands, 0, 65535).astype('uint16')
                
                # Write aligned image with preserved CRS and transform
                self.rasterio_manager.safe_write(
                    output_path,
                    all_bands_uint16,
                    ref_profile
                )
                
                self.logger.info(f"[DEBUG] Aligned output: {output_path}, CRS: {ref_profile.get('crs')}, Transform: {ref_profile.get('transform')}")
                return str(output_path)
                
        except Exception as e:
            self.logger.error(f"Failed to align {image_set['name']}: {str(e)}")
            return None
    
    def _calibrate_image(self, image_path: str) -> Optional[str]:
        """Apply radiometric calibration"""
        try:
            with self.rasterio_manager.safe_open(Path(image_path)) as src:
                output_name = Path(image_path).stem + "_calibrated.tif"
                output_path = self._get_output_path("calibrated", output_name)
                
                # Preserve CRS and transform in profile
                profile = src.profile.copy()
                profile.update({
                    'dtype': 'uint16',
                    'nodata': None,
                    'crs': src.crs,
                    'transform': src.transform
                })
                
                # Process all bands
                all_bands = np.zeros((src.count, src.height, src.width), dtype='float32')
                for band_num in range(1, src.count + 1):
                    data = src.read(band_num).astype('float32')
                    calibrated_data = data * 10000  # scale to reflectance * 10000
                    calibrated_data = np.clip(calibrated_data, 0, 65535)
                    all_bands[band_num - 1, :, :] = calibrated_data
                
                all_bands_uint16 = all_bands.astype('uint16')
                
                # Write calibrated image with preserved CRS and transform
                self.rasterio_manager.safe_write(
                    output_path,
                    all_bands_uint16,
                    profile
                )
                
                self.logger.info(f"[DEBUG] Calibrated output: {output_path}, CRS: {profile.get('crs')}, Transform: {profile.get('transform')}")
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
                profile.update({'count': 1, 'dtype': 'uint16', 'crs': src.crs, 'transform': src.transform})
                # Calculate all indices
                indices = {}
                vi_config = self.config.get('vegetation_indices', {})
                if vi_config.get('ndvi', False):
                    ndvi = self._calculate_ndvi(nir, red)
                    ndvi_uint16 = np.clip((ndvi + 1) * 32767.5, 0, 65535).astype('uint16')
                    output_path = self._get_output_path("indices", f"{image_set['name']}_NDVI.tif")
                    self.rasterio_manager.safe_write(output_path, ndvi_uint16, profile)
                    self.logger.info(f"[DEBUG] Index output: {output_path}, CRS: {profile.get('crs')}, Transform: {profile.get('transform')}")
                    indices_paths['NDVI'] = str(output_path)
                if vi_config.get('ndre', False):
                    ndre = self._calculate_ndre(nir, red_edge)
                    ndre_uint16 = np.clip((ndre + 1) * 32767.5, 0, 65535).astype('uint16')
                    output_path = self._get_output_path("indices", f"{image_set['name']}_NDRE.tif")
                    self.rasterio_manager.safe_write(output_path, ndre_uint16, profile)
                    self.logger.info(f"[DEBUG] Index output: {output_path}, CRS: {profile.get('crs')}, Transform: {profile.get('transform')}")
                    indices_paths['NDRE'] = str(output_path)
                if vi_config.get('gndvi', False):
                    gndvi = self._calculate_gndvi(nir, green)
                    gndvi_uint16 = np.clip((gndvi + 1) * 32767.5, 0, 65535).astype('uint16')
                    output_path = self._get_output_path("indices", f"{image_set['name']}_GNDVI.tif")
                    self.rasterio_manager.safe_write(output_path, gndvi_uint16, profile)
                    self.logger.info(f"[DEBUG] Index output: {output_path}, CRS: {profile.get('crs')}, Transform: {profile.get('transform')}")
                    indices_paths['GNDVI'] = str(output_path)
                if vi_config.get('savi', False):
                    savi = self._calculate_savi(nir, red)
                    savi_uint16 = np.clip((savi + 1) * 32767.5, 0, 65535).astype('uint16')
                    output_path = self._get_output_path("indices", f"{image_set['name']}_SAVI.tif")
                    self.rasterio_manager.safe_write(output_path, savi_uint16, profile)
                    self.logger.info(f"[DEBUG] Index output: {output_path}, CRS: {profile.get('crs')}, Transform: {profile.get('transform')}")
                    indices_paths['SAVI'] = str(output_path)
                if vi_config.get('msavi', False):
                    msavi = self._calculate_msavi(nir, red)
                    msavi_uint16 = np.clip((msavi + 1) * 32767.5, 0, 65535).astype('uint16')
                    output_path = self._get_output_path("indices", f"{image_set['name']}_MSAVI.tif")
                    self.rasterio_manager.safe_write(output_path, msavi_uint16, profile)
                    self.logger.info(f"[DEBUG] Index output: {output_path}, CRS: {profile.get('crs')}, Transform: {profile.get('transform')}")
                    indices_paths['MSAVI'] = str(output_path)
                if vi_config.get('evi', False):
                    evi = self._calculate_evi(nir, red, blue)
                    evi_uint16 = np.clip((evi + 1) * 32767.5, 0, 65535).astype('uint16')
                    output_path = self._get_output_path("indices", f"{image_set['name']}_EVI.tif")
                    self.rasterio_manager.safe_write(output_path, evi_uint16, profile)
                    self.logger.info(f"[DEBUG] Index output: {output_path}, CRS: {profile.get('crs')}, Transform: {profile.get('transform')}")
                    indices_paths['EVI'] = str(output_path)
                if vi_config.get('osavi', False):
                    osavi = self._calculate_osavi(nir, red)
                    osavi_uint16 = np.clip((osavi + 1) * 32767.5, 0, 65535).astype('uint16')
                    output_path = self._get_output_path("indices", f"{image_set['name']}_OSAVI.tif")
                    self.rasterio_manager.safe_write(output_path, osavi_uint16, profile)
                    self.logger.info(f"[DEBUG] Index output: {output_path}, CRS: {profile.get('crs')}, Transform: {profile.get('transform')}")
                    indices_paths['OSAVI'] = str(output_path)
                if vi_config.get('ndwi', False):
                    ndwi = self._calculate_ndwi(green, nir)
                    ndwi_uint16 = np.clip((ndwi + 1) * 32767.5, 0, 65535).astype('uint16')
                    output_path = self._get_output_path("indices", f"{image_set['name']}_NDWI.tif")
                    self.rasterio_manager.safe_write(output_path, ndwi_uint16, profile)
                    self.logger.info(f"[DEBUG] Index output: {output_path}, CRS: {profile.get('crs')}, Transform: {profile.get('transform')}")
                    indices_paths['NDWI'] = str(output_path)
                self.logger.info(f"Calculated {len(indices_paths)} vegetation indices for {image_set['name']}")
        except Exception as e:
            self.logger.error(f"Failed to calculate indices for {image_set['name']}: {str(e)}")
            self.logger.error(traceback.format_exc())
        return indices_paths
    
    def _calculate_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index"""
        return np.divide(
            (nir - red),
            (nir + red),
            out=np.zeros_like(nir),
            where=(nir + red) != 0
        )
    
    def _calculate_ndre(self, nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Red Edge Index"""
        return np.divide(
            (nir - red_edge),
            (nir + red_edge),
            out=np.zeros_like(nir),
            where=(nir + red_edge) != 0
        )
    
    def _calculate_gndvi(self, nir: np.ndarray, green: np.ndarray) -> np.ndarray:
        """Calculate Green Normalized Difference Vegetation Index"""
        return np.divide(
            (nir - green),
            (nir + green),
            out=np.zeros_like(nir),
            where=(nir + green) != 0
        )
    
    def _calculate_savi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Calculate Soil Adjusted Vegetation Index"""
        L = 0.5  # Soil brightness correction factor
        return np.divide(
            (nir - red) * (1 + L),
            (nir + red + L),
            out=np.zeros_like(nir),
            where=(nir + red + L) != 0
        )
    
    def _calculate_msavi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Calculate Modified Soil Adjusted Vegetation Index"""
        return 0.5 * (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red)))
    
    def _calculate_evi(self, nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """Calculate Enhanced Vegetation Index"""
        G = 2.5  # Gain factor
        L = 1.0  # Soil brightness correction factor
        C1, C2 = 6.0, 7.5  # Aerosol resistance coefficients
        
        return G * np.divide(
            (nir - red),
            (nir + C1 * red - C2 * blue + L),
            out=np.zeros_like(nir),
            where=(nir + C1 * red - C2 * blue + L) != 0
        )
    
    def _calculate_osavi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Calculate Optimized Soil Adjusted Vegetation Index"""
        L = 0.16  # Optimized soil brightness correction factor
        return np.divide(
            (nir - red) * (1 + L),
            (nir + red + L),
            out=np.zeros_like(nir),
            where=(nir + red + L) != 0
        )
    
    def _calculate_ndwi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Water Index"""
        return np.divide(
            (green - nir),
            (green + nir),
            out=np.zeros_like(green),
            where=(green + nir) != 0
        )
    
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
                    hist_path = self._get_output_path("quality_reports", f"{image_set['name']}_{band_num}_histogram.png")
                    self._generate_histogram(Path(band_path), hist_path, self.config['band_config'][band_num]["name"])
            
            # Save report
            report_path = self._get_output_path("quality_reports", f"{image_set['name']}_quality_report.json")
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
    
    def _save_metadata(self, metadata: Dict):
        """Save metadata for an image set"""
        try:
            # Save metadata
            metadata_path = self._get_output_path("metadata", f"{metadata['image_set']['name']}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save metadata for {metadata['image_set']['name']}: {str(e)}")
    
    def find_image_sets(self, input_dir: Path) -> List[Dict]:
        """Find complete sets of MicaSense images in the input directory"""
        image_sets = []
        
        # Group files by timestamp
        timestamp_groups = {}
        for file_path in input_dir.glob("*.tif"):
            if "_" not in file_path.stem:
                continue
                
            timestamp = file_path.stem.split("_")[0]
            if timestamp not in timestamp_groups:
                timestamp_groups[timestamp] = []
            timestamp_groups[timestamp].append(file_path)
        
        # Create image sets
        for timestamp, files in timestamp_groups.items():
            if len(files) >= 5:  # Complete set has at least 5 bands
                bands = {
                    1: str(files[0]),
                    2: str(files[1]),
                    3: str(files[2]),
                    4: str(files[3]),
                    5: str(files[4])
                }
                image_sets.append({
                    "name": timestamp,
                    "bands": bands
                })
        
        return image_sets 

    def _is_valid_georeferenced(self, tiff_path):
        import rasterio
        try:
            with rasterio.open(tiff_path) as src:
                crs = src.crs
                transform = src.transform
                pixel_width = abs(transform.a)
                pixel_height = abs(transform.e)
                # Check for valid CRS and reasonable pixel size
                if crs is None or pixel_width >= 0.01 or pixel_height >= 0.01:
                    return False
                return True
        except Exception:
            return False

    def cleanup_legacy_outputs(self):
        """Remove non-georeferenced or legacy outputs from all output subdirs."""
        import os
        import glob
        subdirs = ['aligned', 'calibrated', 'indices', 'individual_bands']
        for subdir in subdirs:
            dir_path = self.output_dir / subdir
            for tiff_path in glob.glob(str(dir_path / '*.tif')):
                if not self._is_valid_georeferenced(tiff_path):
                    os.remove(tiff_path)
                    self.logger.info(f"[CLEANUP] Removed legacy or non-georeferenced file: {tiff_path}")

    def process_all(self, image_sets: List[Dict]) -> List[Dict]:
        results = []
        for image_set in image_sets:
            try:
                result = self.process_single_set(image_set)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process image set {image_set['name']}: {e}")
                results.append({'name': image_set['name'], 'status': 'failed', 'error': str(e)})
        # Cleanup legacy outputs before validation
        self.cleanup_legacy_outputs()
        return results 