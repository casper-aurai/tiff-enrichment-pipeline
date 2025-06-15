"""
MicaSense Validation Module
Validates MicaSense data and configuration
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import rasterio
from rasterio.errors import RasterioIOError
import numpy as np
from exif import Image
from datetime import datetime
import re
import subprocess
import math
import matplotlib.pyplot as plt
import pandas as pd

from .errors import ValidationError

class MicaSenseValidator:
    """Validates MicaSense data and configuration"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def _extract_exif_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract EXIF metadata from TIFF file"""
        metadata = {}
        try:
            if file_path.suffix.lower() in ['.tif', '.tiff']:
                # Use Pillow for TIFFs
                from PIL import Image as PILImage
                from PIL.TiffTags import TAGS
                with PILImage.open(file_path) as img:
                    for tag, value in img.tag_v2.items():
                        tag_name = TAGS.get(tag, tag)
                        metadata[tag_name] = value
                    # Map alternate tag names
                    if 'Date/Time Original' in metadata:
                        metadata['DateTime'] = metadata['Date/Time Original']
                    if 'Camera Model Name' in metadata:
                        metadata['CameraModel'] = metadata['Camera Model Name']
            else:
                # Use exif library for other formats (e.g., JPEG)
                with open(file_path, 'rb') as image_file:
                    exif_image = Image(image_file)
                    # Extract DateTime
                    if hasattr(exif_image, 'datetime'):
                        metadata['DateTime'] = exif_image.datetime
                    elif hasattr(exif_image, 'datetime_original'):
                        metadata['DateTime'] = exif_image.datetime_original
                    # Extract CameraModel
                    if hasattr(exif_image, 'model'):
                        metadata['CameraModel'] = exif_image.model
                    # Extract GPS information
                    if hasattr(exif_image, 'gps_latitude') and hasattr(exif_image, 'gps_longitude'):
                        metadata['GPSLatitude'] = exif_image.gps_latitude
                        metadata['GPSLongitude'] = exif_image.gps_longitude
                    # Extract other relevant metadata
                    for tag in ['make', 'software', 'exposure_time', 'f_number', 'iso_speed']:
                        if hasattr(exif_image, tag):
                            metadata[tag] = getattr(exif_image, tag)
        except Exception as e:
            self.logger.warning(f"Failed to extract EXIF metadata from {file_path}: {str(e)}")
        return metadata
    
    def _get_file_info(self, src, data, datetime_val, crs_calc_details=None) -> List[str]:
        """Get informational messages about the file"""
        info = []
        transform = src.transform
        width, height = src.width, src.height
        crs = src.crs
        
        # Get all four corners in CRS
        corners = [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height)
        ]
        corner_coords = [rasterio.transform.xy(transform, y, x, offset='center') for x, y in corners]
        
        # Calculate surface area using Haversine formula
        def haversine(lon1, lat1, lon2, lat2):
            """
            Calculate the great circle distance between two points 
            on the earth (specified in decimal degrees)
            """
            # Convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371000  # Radius of earth in meters
            return c * r
            
        if crs and crs.to_epsg() == 4326:
            # Calculate distances between corners
            lon0, lat0 = corner_coords[0]
            lon1, lat1 = corner_coords[1]
            lon2, lat2 = corner_coords[2]
            lon3, lat3 = corner_coords[3]
            
            # Calculate width and height in meters
            width_m = haversine(lon0, lat0, lon1, lat1)
            height_m = haversine(lon1, lat1, lon2, lat2)
            
            # Calculate surface area
            surface_m2 = abs(width_m * height_m)
            
            # Calculate GSD if we have the necessary information
            if crs_calc_details and 'gsd_x' in crs_calc_details and 'gsd_y' in crs_calc_details:
                gsd_x = crs_calc_details['gsd_x']
                gsd_y = crs_calc_details['gsd_y']
                info.append(f"GSD X: {gsd_x:.6f} m/pixel")
                info.append(f"GSD Y: {gsd_y:.6f} m/pixel")
                
                # Add intermediate calculation steps
                if 'altitude' in crs_calc_details:
                    info.append(f"Altitude: {crs_calc_details['altitude']:.2f} m")
                if 'focal_length' in crs_calc_details:
                    info.append(f"Focal length: {crs_calc_details['focal_length']:.2f} mm")
                if 'sensor_width' in crs_calc_details:
                    info.append(f"Sensor width: {crs_calc_details['sensor_width']:.2f} mm")
                if 'sensor_height' in crs_calc_details:
                    info.append(f"Sensor height: {crs_calc_details['sensor_height']:.2f} mm")
                
                # Calculate expected vs actual pixel sizes
                expected_pixel_width = gsd_x / (2 * math.pi * 6371000 * math.cos(math.radians(lat0)) / 360)
                expected_pixel_height = gsd_y / (2 * math.pi * 6371000 / 360)
                actual_pixel_width = abs(transform.a)
                actual_pixel_height = abs(transform.e)
                
                info.append(f"Expected pixel width (degrees): {expected_pixel_width:.10f}")
                info.append(f"Actual pixel width (degrees): {actual_pixel_width:.10f}")
                info.append(f"Expected pixel height (degrees): {expected_pixel_height:.10f}")
                info.append(f"Actual pixel height (degrees): {actual_pixel_height:.10f}")
                
                # Calculate difference percentage
                width_diff_pct = abs(expected_pixel_width - actual_pixel_width) / expected_pixel_width * 100
                height_diff_pct = abs(expected_pixel_height - actual_pixel_height) / expected_pixel_height * 100
                info.append(f"Pixel width difference: {width_diff_pct:.2f}%")
                info.append(f"Pixel height difference: {height_diff_pct:.2f}%")
        else:
            # If not in WGS84, use transform values
            surface_m2 = abs(transform.a * transform.e * width * height)
        
        # Add statistics
        stats = {
            'mean': float(np.mean(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'std': float(np.std(data)),
            'zero_ratio': float(np.sum(data == 0) / data.size)
        }
        
        # Add informational messages
        info.append(f"Surface area (m^2): {surface_m2:.2f}")
        info.append(f"Corner coordinates: {corner_coords}")
        info.append(f"Stats: {stats}")
        if datetime_val:
            info.append(f"DateTime: {datetime_val}")
        if crs_calc_details:
            info.append(f"CRS calculation details: {crs_calc_details}")
            
        return info

    def validate_tiff_file(self, file_path: Path, crs_calc_details: dict = None) -> List[str]:
        """Validate a single TIFF file, assign CRS if GPS is present and CRS is missing"""
        issues = []
        exif_metadata = self._extract_exif_metadata(file_path)
        try:
            with rasterio.open(file_path) as src:
                # Check image dimensions
                min_width, min_height = self.config['validation']['min_dimensions']
                if src.width < min_width or src.height < min_height:
                    issues.append(f"Small dimensions: {src.width}x{src.height}")
                
                # Check data type
                if src.dtypes[0] not in self.config['validation']['valid_data_types']:
                    issues.append(f"Unexpected data type: {src.dtypes[0]}")
                
                # Check for valid data range
                data = src.read(1)
                min_val, max_val = self.config['validation']['data_range']
                if np.min(data) < min_val or np.max(data) > max_val:
                    issues.append(f"Invalid data range: [{np.min(data)}, {np.max(data)}]")
                
                # Check for excessive zeros
                zero_ratio = np.sum(data == 0) / data.size
                if zero_ratio > self.config['validation']['max_zero_ratio']:
                    issues.append(f"High zero ratio: {zero_ratio:.1%}")
                
                # Check CRS and transform
                if not src.crs:
                    issues.append("Missing CRS")
                elif src.crs.to_epsg() != 4326:
                    issues.append(f"Unexpected CRS: {src.crs}")
                
                if src.transform.is_identity:
                    issues.append("Missing transform")
                
                # Check required metadata (DateTime)
                tags = src.tags()
                datetime_val = tags.get('DateTime') or exif_metadata.get('DateTime')
                if not datetime_val:
                    # Try exiftool as fallback
                    try:
                        proc = subprocess.run(["exiftool", str(file_path)], capture_output=True, text=True, timeout=10)
                        for line in proc.stdout.splitlines():
                            if 'Date/Time Original' in line:
                                datetime_val = line.split(':', 1)[1].strip()
                                break
                        if not datetime_val:
                            for line in proc.stdout.splitlines():
                                if 'Create Date' in line:
                                    datetime_val = line.split(':', 1)[1].strip()
                                    break
                    except Exception:
                        pass
                if not datetime_val:
                    issues.append("Missing required metadata: DateTime")
                
                # Get informational messages
                info = self._get_file_info(src, data, datetime_val, crs_calc_details)
                
                # Return both issues and info
                return issues + info
                
        except Exception as e:
            self.logger.warning(f"Failed to validate {file_path}: {str(e)}")
            return [f"Validation error: {str(e)}"]
    
    def validate_image_set(self, image_set: Dict) -> List[str]:
        """Validate a complete image set"""
        issues = []
        
        # Check if all required bands are present
        required_bands = set(range(1, 6))  # Bands 1-5
        present_bands = set(image_set["bands"].keys())
        missing_bands = required_bands - present_bands
        if missing_bands:
            issues.append(f"Missing bands: {missing_bands}")
        
        # Check band file consistency
        band_files = list(image_set["bands"].values())
        if len(set(Path(f).parent for f in band_files)) > 1:
            issues.append("Band files are in different directories")
        
        # Check for consistent dimensions
        dimensions = set()
        for band_path in band_files:
            try:
                with rasterio.open(band_path) as src:
                    dimensions.add((src.width, src.height))
            except Exception:
                continue
        
        if len(dimensions) > 1:
            issues.append(f"Inconsistent dimensions: {dimensions}")
        
        # Validate individual band files
        for band_num, band_path in image_set["bands"].items():
            band_issues = self.validate_tiff_file(Path(band_path))
            if band_issues:
                issues.extend([f"Band {band_num}: {issue}" for issue in band_issues])
        
        return issues
    
    def validate_input_directory(self, input_dir: Path) -> bool:
        """Validate input directory and files"""
        if not input_dir.exists():
            self.logger.error(f"Input directory not found: {input_dir}")
            return False
        
        tiff_files = list(input_dir.glob("**/*.tif")) + list(input_dir.glob("**/*.TIF"))
        if len(tiff_files) == 0:
            self.logger.error("No TIFF files found in input directory")
            return False
        
        # Validate individual files
        for tiff in tiff_files:
            issues = self.validate_tiff_file(tiff)
            if issues:
                for issue in issues:
                    self.logger.warning(f"File {tiff}: {issue}")
        
        return True
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check required sections
            required_sections = [
                'quality_control',
                'vegetation_indices',
                'output_options',
                'processing',
                'logging',
                'validation',
                'band_config'
            ]
            
            for section in required_sections:
                if section not in self.config:
                    raise ValidationError(f"Missing required section: {section}")
            
            # Validate quality control settings
            qc_settings = self.config['quality_control']
            if not isinstance(qc_settings, dict):
                raise ValidationError("quality_control must be a dictionary")
            
            # Validate vegetation indices settings
            vi_settings = self.config['vegetation_indices']
            if not isinstance(vi_settings, dict):
                raise ValidationError("vegetation_indices must be a dictionary")
            
            # Validate output options
            output_settings = self.config['output_options']
            if not isinstance(output_settings, dict):
                raise ValidationError("output_options must be a dictionary")
            
            # Validate processing settings
            proc_settings = self.config['processing']
            if not isinstance(proc_settings, dict):
                raise ValidationError("processing must be a dictionary")
            
            # Validate logging settings
            log_settings = self.config['logging']
            if not isinstance(log_settings, dict):
                raise ValidationError("logging must be a dictionary")
            
            # Validate validation settings
            val_settings = self.config['validation']
            if not isinstance(val_settings, dict):
                raise ValidationError("validation must be a dictionary")
            
            # Validate band configuration
            band_settings = self.config['band_config']
            if not isinstance(band_settings, dict):
                raise ValidationError("band_config must be a dictionary")
            
            # Check required band information
            for band_num in range(1, 6):
                if band_num not in band_settings:
                    raise ValidationError(f"Missing configuration for band {band_num}")
                
                band_info = band_settings[band_num]
                if not isinstance(band_info, dict):
                    raise ValidationError(f"Band {band_num} configuration must be a dictionary")
                
                required_fields = ['name', 'wavelength', 'description']
                for field in required_fields:
                    if field not in band_info:
                        raise ValidationError(f"Missing {field} for band {band_num}")
            
            return True
            
        except ValidationError as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during configuration validation: {str(e)}")
            return False

    def generate_indices_report(self, indices_dir: Path, output_dir: Path, reference_data: Path = None):
        """Generate a comprehensive validation report for all index files in a directory."""
        report = []
        spatial_info = {}
        summary_rows = []
        indices_dir = Path(indices_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for tif in indices_dir.glob("*.tif"):
            entry = {'file': str(tif)}
            try:
                with rasterio.open(tif) as src:
                    data = src.read(1)
                    # Basic stats
                    entry['min'] = float(np.nanmin(data))
                    entry['max'] = float(np.nanmax(data))
                    entry['mean'] = float(np.nanmean(data))
                    entry['median'] = float(np.nanmedian(data))
                    entry['std'] = float(np.nanstd(data))
                    entry['p5'] = float(np.nanpercentile(data, 5))
                    entry['p25'] = float(np.nanpercentile(data, 25))
                    entry['p75'] = float(np.nanpercentile(data, 75))
                    entry['p95'] = float(np.nanpercentile(data, 95))
                    # Histogram
                    hist, bins = np.histogram(data[~np.isnan(data)], bins=50)
                    entry['histogram_bins'] = bins.tolist()
                    entry['histogram_counts'] = hist.tolist()
                    plt.figure()
                    plt.hist(data[~np.isnan(data)].flatten(), bins=50)
                    plt.title(f"Histogram: {tif.name}")
                    plt.xlabel("Value")
                    plt.ylabel("Count")
                    hist_path = output_dir / f"{tif.stem}_hist.png"
                    plt.savefig(hist_path)
                    plt.close()
                    entry['histogram_image'] = str(hist_path)
                    # Key value ranges (NDVI example)
                    if 'NDVI' in tif.name.upper():
                        entry['pct_lt0'] = float(np.mean(data < 0) * 100)
                        entry['pct_0_02'] = float(np.mean((data >= 0) & (data < 0.2)) * 100)
                        entry['pct_02_05'] = float(np.mean((data >= 0.2) & (data < 0.5)) * 100)
                        entry['pct_gt05'] = float(np.mean(data >= 0.5) * 100)
                    # NoData/invalid
                    entry['pct_nan'] = float(np.mean(np.isnan(data)) * 100)
                    entry['pct_nodata'] = float(np.mean(data == src.nodata) * 100) if src.nodata is not None else 0.0
                    entry['pct_neg9999'] = float(np.mean(data == -9999) * 100)
                    # Outlier detection
                    entry['out_of_range'] = bool((data > 1).any() or (data < -1).any())
                    # Spatial info
                    entry['crs'] = str(src.crs)
                    entry['transform'] = str(src.transform)
                    entry['shape'] = (src.height, src.width)
                    spatial_info[tif.name] = (src.crs, src.transform, src.height, src.width)
                    # Thumbnail
                    from matplotlib import cm
                    plt.figure(figsize=(2,2))
                    plt.imshow(data, cmap=cm.viridis)
                    plt.axis('off')
                    thumb_path = output_dir / f"{tif.stem}_thumb.png"
                    plt.savefig(thumb_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    entry['thumbnail'] = str(thumb_path)
                    # Temporal (if date in filename)
                    import re
                    m = re.search(r'(\d{8})', tif.name)
                    if m:
                        entry['date'] = m.group(1)
                    # Reference comparison (if provided)
                    if reference_data:
                        # Not implemented: placeholder for future
                        entry['reference_comparison'] = 'TODO'
                report.append(entry)
                summary_rows.append(entry)
            except Exception as e:
                entry['error'] = str(e)
                report.append(entry)
        # Spatial consistency check
        crs_set = set(str(v[0]) for v in spatial_info.values())
        transform_set = set(str(v[1]) for v in spatial_info.values())
        shape_set = set(v[2:] for v in spatial_info.values())
        spatial_summary = {
            'unique_crs': list(crs_set),
            'unique_transforms': list(transform_set),
            'unique_shapes': list(shape_set),
            'crs_consistent': len(crs_set) == 1,
            'transform_consistent': len(transform_set) == 1,
            'shape_consistent': len(shape_set) == 1
        }
        # Save JSON report
        import json
        report_path = output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump({'indices': report, 'spatial_summary': spatial_summary}, f, indent=2)
        # Save CSV summary
        df = pd.DataFrame(summary_rows)
        csv_path = output_dir / 'validation_report.csv'
        df.to_csv(csv_path, index=False)
        # Log summary
        self.logger.info(f"Validation report saved: {report_path}")
        self.logger.info(f"Validation summary CSV saved: {csv_path}")
        self.logger.info(f"Spatial summary: {spatial_summary}") 