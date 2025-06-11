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

from .errors import ValidationError

class MicaSenseValidator:
    """Validates MicaSense data and configuration"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def validate_tiff_file(self, file_path: Path) -> List[str]:
        """Validate a single TIFF file"""
        issues = []
        
        try:
            # Check file size
            size = file_path.stat().st_size
            if size == 0:
                issues.append("Empty file")
            elif size < 1024:  # Less than 1KB
                issues.append("Suspiciously small file")
            
            # Try to open with rasterio
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
                
                # Check for metadata
                if not src.tags():
                    issues.append("Missing metadata")
                else:
                    # Check required metadata
                    for tag in self.config['validation']['required_metadata']:
                        if tag not in src.tags():
                            issues.append(f"Missing required metadata: {tag}")
            
        except RasterioIOError as e:
            issues.append(f"Failed to open file: {str(e)}")
        except Exception as e:
            issues.append(f"Unexpected error: {str(e)}")
        
        return issues
    
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