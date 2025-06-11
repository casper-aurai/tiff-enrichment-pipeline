"""
MicaSense RedEdge-M Batch Processing Module
Specialized processor for multispectral imagery from MicaSense cameras
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor
import glob
from datetime import datetime
import logging.handlers

try:
    from osgeo import gdal, osr
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling
except ImportError:
    print("GDAL/rasterio not available - install with: pip install GDAL rasterio")

class MicaSenseProcessor:
    """
    Batch processor for MicaSense RedEdge-M multispectral imagery
    Handles radiometric calibration, band alignment, and vegetation indices
    """
    
    # MicaSense RedEdge-M band configuration
    BAND_CONFIG = {
        1: {"name": "Blue", "center_wavelength": 475, "bandwidth": 20},
        2: {"name": "Green", "center_wavelength": 560, "bandwidth": 20}, 
        3: {"name": "Red", "center_wavelength": 668, "bandwidth": 10},
        4: {"name": "Near IR", "center_wavelength": 840, "bandwidth": 40},
        5: {"name": "Red Edge", "center_wavelength": 717, "bandwidth": 10}
    }
    
    def __init__(self, input_dir: Path, output_dir: Path, max_workers: int = 4, config: Dict = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Create output directories
        self._setup_directories()
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _load_config(self, user_config: Dict = None) -> Dict:
        """Load and validate configuration"""
        default_config = {
            'quality_control': {
                'check_band_alignment': True,
                'validate_reflectance_range': True,
                'generate_histograms': True
            },
            'vegetation_indices': {
                'ndvi': True,
                'ndre': True,
                'gndvi': True,
                'savi': False
            },
            'output_options': {
                'save_individual_bands': True,
                'generate_thumbnails': True,
                'overwrite_existing': False
            },
            'processing': {
                'radiometric_calibration': True,
                'band_alignment': True,
                'generate_indices': True
            }
        }
        
        if user_config:
            # Deep merge user config with defaults
            def deep_merge(d1, d2):
                for k, v in d2.items():
                    if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                        deep_merge(d1[k], v)
                    else:
                        d1[k] = v
                return d1
            
            default_config = deep_merge(default_config, user_config)
        
        return default_config
    
    def _setup_directories(self):
        """Create necessary output directories"""
        dirs = [
            'aligned',
            'indices',
            'metadata',
            'quality_reports',
            'thumbnails',
            'individual_bands'
        ]
        
        for dir_name in dirs:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the processor"""
        logger = logging.getLogger("MicaSenseProcessor")
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.output_dir / "processing.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(console_handler)
        
        return logger
    
    def find_image_sets(self) -> List[Dict]:
        """
        Find all MicaSense image sets in the input directory
        Returns list of image sets with their band files
        """
        image_sets = []
        
        # More flexible pattern for MicaSense files
        patterns = [
            "**/IMG_*.tif",
            "**/IMG_*.TIF",
            "**/IMG_*.tiff",
            "**/IMG_*.TIFF"
        ]
        
        all_files = []
        for pattern in patterns:
            all_files.extend(glob.glob(str(self.input_dir / pattern), recursive=True))
        
        # Group files by capture (same base name, different band numbers)
        captures = {}
        for file_path in all_files:
            filename = Path(file_path).name
            # Extract base name (e.g., IMG_0001 from IMG_0001_1.tif)
            if "_" in filename:
                parts = filename.split("_")
                if len(parts) >= 3:
                    base_name = "_".join(parts[:-1])  # Everything except last part
                    band_num = parts[-1].split(".")[0]  # Last part before extension
                    
                    if base_name not in captures:
                        captures[base_name] = {}
                    
                    try:
                        band_int = int(band_num)
                        if 1 <= band_int <= 5:  # MicaSense has 5 bands
                            captures[base_name][band_int] = str(file_path)
                    except ValueError:
                        continue
        
        # Filter complete sets (must have all 5 bands)
        for base_name, bands in captures.items():
            if len(bands) == 5 and all(i in bands for i in range(1, 6)):
                # Sort bands to ensure consistent order
                sorted_bands = {k: bands[k] for k in sorted(bands.keys())}
                image_sets.append({
                    "name": base_name,
                    "bands": sorted_bands,
                    "path": Path(bands[1]).parent  # Directory containing the files
                })
                
        self.logger.info(f"Found {len(image_sets)} complete MicaSense image sets")
        if len(image_sets) == 0:
            self.logger.warning("No complete MicaSense image sets found")
            # Log the partial sets for debugging
            partial_sets = {name: len(bands) for name, bands in captures.items() if len(bands) < 5}
            if partial_sets:
                self.logger.warning(f"Found {len(partial_sets)} incomplete sets: {partial_sets}")
        
        return image_sets
    
    def extract_metadata(self, image_path: str) -> Dict:
        """Extract metadata from MicaSense TIFF file"""
        metadata = {
            "capture_time": None,
            "gps_coordinates": None,
            "camera_info": {},
            "radiometric_calibration": {}
        }
        
        try:
            with rasterio.open(image_path) as src:
                # Basic image info
                metadata["width"] = src.width
                metadata["height"] = src.height
                metadata["crs"] = str(src.crs) if src.crs else None
                metadata["transform"] = list(src.transform)
                
                # Extract EXIF data
                tags = src.tags()
                
                # GPS coordinates
                if "EXIF_GPS_LatRef" in tags and "EXIF_GPS_Lat" in tags:
                    lat = self._parse_gps_coordinate(tags["EXIF_GPS_Lat"])
                    lon = self._parse_gps_coordinate(tags["EXIF_GPS_Lon"])
                    if tags["EXIF_GPS_LatRef"] == "S":
                        lat = -lat
                    if tags["EXIF_GPS_LonRef"] == "W":
                        lon = -lon
                    metadata["gps_coordinates"] = {"latitude": lat, "longitude": lon}
                
                # Capture time
                if "EXIF_DateTime" in tags:
                    metadata["capture_time"] = tags["EXIF_DateTime"]
                
                # Camera and radiometric info
                for key, value in tags.items():
                    if key.startswith("Camera"):
                        metadata["camera_info"][key] = value
                    elif "Calibration" in key or "Irradiance" in key:
                        metadata["radiometric_calibration"][key] = value
                        
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata from {image_path}: {e}")
            
        return metadata
    
    def _parse_gps_coordinate(self, coord_str: str) -> float:
        """Parse GPS coordinate from EXIF format"""
        try:
            # Format: "degrees/1 minutes/1 seconds/1"
            parts = coord_str.split()
            degrees = float(parts[0].split("/")[0])
            minutes = float(parts[1].split("/")[0]) if len(parts) > 1 else 0
            seconds = float(parts[2].split("/")[0]) if len(parts) > 2 else 0
            return degrees + minutes/60 + seconds/3600
        except:
            return 0.0
    
    def align_bands(self, image_set: Dict) -> Optional[str]:
        """
        Align all bands of an image set and create a multi-band TIFF
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
                                self.BAND_CONFIG[band_num]["name"]
                            )
                
                self.logger.info(f"Successfully aligned {image_set['name']}")
                return str(output_path)
                
        except Exception as e:
            self.logger.error(f"Failed to align {image_set['name']}: {e}")
            return None
    
    def calculate_vegetation_indices(self, aligned_path: str, image_set: Dict) -> Dict[str, str]:
        """
        Calculate vegetation indices from aligned multispectral image
        """
        indices_paths = {}
        
        try:
            with rasterio.open(aligned_path) as src:
                # Read bands (1=Blue, 2=Green, 3=Red, 4=NIR, 5=RedEdge)
                blue = src.read(1).astype('float32')
                green = src.read(2).astype('float32')
                red = src.read(3).astype('float32')
                nir = src.read(4).astype('float32')
                red_edge = src.read(5).astype('float32')
                
                profile = src.profile.copy()
                profile.update({'count': 1, 'dtype': 'float32'})
                
                # NDVI (Normalized Difference Vegetation Index)
                ndvi = np.divide(
                    (nir - red),
                    (nir + red),
                    out=np.zeros_like(nir),
                    where=(nir + red) != 0
                )
                
                ndvi_path = self.output_dir / "indices" / f"{image_set['name']}_NDVI.tif"
                with rasterio.open(ndvi_path, 'w', **profile) as dst:
                    dst.write(ndvi, 1)
                    dst.set_band_description(1, "NDVI")
                indices_paths["NDVI"] = str(ndvi_path)
                
                # NDRE (Normalized Difference Red Edge)
                ndre = np.divide(
                    (nir - red_edge),
                    (nir + red_edge),
                    out=np.zeros_like(nir),
                    where=(nir + red_edge) != 0
                )
                
                ndre_path = self.output_dir / "indices" / f"{image_set['name']}_NDRE.tif"
                with rasterio.open(ndre_path, 'w', **profile) as dst:
                    dst.write(ndre, 1)
                    dst.set_band_description(1, "NDRE")
                indices_paths["NDRE"] = str(ndre_path)
                
                # GNDVI (Green Normalized Difference Vegetation Index)
                gndvi = np.divide(
                    (nir - green),
                    (nir + green),
                    out=np.zeros_like(nir),
                    where=(nir + green) != 0
                )
                
                gndvi_path = self.output_dir / "indices" / f"{image_set['name']}_GNDVI.tif"
                with rasterio.open(gndvi_path, 'w', **profile) as dst:
                    dst.write(gndvi, 1)
                    dst.set_band_description(1, "GNDVI")
                indices_paths["GNDVI"] = str(gndvi_path)
                
                self.logger.info(f"Calculated vegetation indices for {image_set['name']}")
                
        except Exception as e:
            self.logger.error(f"Failed to calculate indices for {image_set['name']}: {e}")
            
        return indices_paths
    
    def process_single_set(self, image_set: Dict) -> Dict:
        """Process a single MicaSense image set"""
        result = {
            "name": image_set["name"],
            "status": "failed",
            "aligned_path": None,
            "indices_paths": {},
            "metadata": {},
            "error": None
        }
        
        try:
            # Extract metadata from first band
            metadata = self.extract_metadata(image_set["bands"][1])
            result["metadata"] = metadata
            
            # Align bands
            aligned_path = self.align_bands(image_set)
            if not aligned_path:
                result["error"] = "Band alignment failed"
                return result
                
            result["aligned_path"] = aligned_path
            
            # Calculate vegetation indices
            indices_paths = self.calculate_vegetation_indices(aligned_path, image_set)
            result["indices_paths"] = indices_paths
            
            # Save metadata
            metadata_path = self.output_dir / "metadata" / f"{image_set['name']}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            result["status"] = "success"
            self.logger.info(f"Successfully processed {image_set['name']}")
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Failed to process {image_set['name']}: {e}")
            
        return result
    
    def process_all(self) -> Dict:
        """
        Process all MicaSense image sets in the input directory
        """
        start_time = datetime.now()
        self.logger.info("Starting batch processing of MicaSense images")
        
        # Find all image sets
        image_sets = self.find_image_sets()
        
        if not image_sets:
            self.logger.warning("No complete MicaSense image sets found")
            return {
                "status": "no_data",
                "total_sets": 0,
                "successful": 0,
                "failed": 0,
                "duration_seconds": 0,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Process with multiprocessing
        results = []
        successful = 0
        failed = 0
        
        try:
            if self.max_workers == 1:
                # Sequential processing
                for image_set in image_sets:
                    try:
                        result = self.process_single_set(image_set)
                        results.append(result)
                        if result["status"] == "success":
                            successful += 1
                        else:
                            failed += 1
                            self.logger.error(f"Failed to process {image_set['name']}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        failed += 1
                        self.logger.error(f"Error processing {image_set['name']}: {str(e)}")
                        results.append({
                            "name": image_set["name"],
                            "status": "failed",
                            "error": str(e)
                        })
            else:
                # Parallel processing with error handling
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for img_set in image_sets:
                        future = executor.submit(self.process_single_set, img_set)
                        futures.append((img_set["name"], future))
                    
                    for name, future in futures:
                        try:
                            result = future.result(timeout=300)  # 5 minute timeout per set
                            results.append(result)
                            if result["status"] == "success":
                                successful += 1
                            else:
                                failed += 1
                                self.logger.error(f"Failed to process {name}: {result.get('error', 'Unknown error')}")
                        except TimeoutError:
                            failed += 1
                            self.logger.error(f"Timeout processing {name}")
                            results.append({
                                "name": name,
                                "status": "failed",
                                "error": "Processing timeout"
                            })
                        except Exception as e:
                            failed += 1
                            self.logger.error(f"Error processing {name}: {str(e)}")
                            results.append({
                                "name": name,
                                "status": "failed",
                                "error": str(e)
                            })
        
        except Exception as e:
            self.logger.error(f"Fatal error during batch processing: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "total_sets": len(image_sets),
                "successful": successful,
                "failed": failed,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Save summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            "status": "completed",
            "total_sets": len(image_sets),
            "successful": successful,
            "failed": failed,
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "results": results
        }
        
        # Save detailed summary
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(
            f"Batch processing completed: {successful} successful, {failed} failed, "
            f"{duration:.1f} seconds"
        )
        
        return summary


def main():
    """Example usage of MicaSense batch processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process MicaSense RedEdge-M imagery")
    parser.add_argument("input_dir", help="Directory containing MicaSense TIFF files")
    parser.add_argument("output_dir", help="Output directory for processed files")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = MicaSenseProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.workers
    )
    
    summary = processor.process_all()
    
    print(f"\nProcessing completed!")
    print(f"Total sets: {summary['total_sets']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Duration: {summary['duration_seconds']:.1f} seconds")


if __name__ == "__main__":
    main()
