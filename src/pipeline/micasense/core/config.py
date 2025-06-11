"""
MicaSense Configuration Module
Manages configuration settings for the MicaSense processing pipeline
"""

import json
from pathlib import Path
from typing import Dict, Any

from .errors import ConfigurationError

class MicaSenseConfig:
    """Configuration management for MicaSense processing"""
    
    @staticmethod
    def create_default(output_dir: Path) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'quality_control': {
                'check_band_alignment': True,
                'validate_reflectance_range': True,
                'generate_histograms': True,
                'validate_dimensions': True,
                'check_zero_ratio': True,
                'validate_metadata': True,
                'check_data_range': True
            },
            'vegetation_indices': {
                'ndvi': True,
                'ndre': True,
                'gndvi': True,
                'savi': True,
                'msavi': True,
                'evi': True,
                'osavi': True,
                'ndwi': True
            },
            'output_options': {
                'save_individual_bands': True,
                'generate_thumbnails': True,
                'overwrite_existing': False,
                'save_quality_reports': True,
                'save_metadata': True,
                'save_visualizations': True,
                'save_processing_report': True
            },
            'processing': {
                'radiometric_calibration': True,
                'band_alignment': True,
                'generate_indices': True,
                'max_workers': 4,
                'batch_size': 10,
                'timeout_seconds': 300,
                'retry_attempts': 2
            },
            'logging': {
                'level': 'INFO',
                'file': str(output_dir / "processing.log"),
                'max_file_size': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5
            },
            'validation': {
                'min_dimensions': (100, 100),
                'max_zero_ratio': 0.5,
                'valid_data_types': ['uint8', 'uint16'],
                'data_range': (0, 65535),
                'required_metadata': [
                    'EXIF_DateTime',
                    'EXIF_GPS_Lat',
                    'EXIF_GPS_Lon'
                ]
            },
            'band_config': {
                1: {
                    "name": "Blue",
                    "wavelength": 475,
                    "description": "Blue band (475nm)"
                },
                2: {
                    "name": "Green",
                    "wavelength": 560,
                    "description": "Green band (560nm)"
                },
                3: {
                    "name": "Red",
                    "wavelength": 668,
                    "description": "Red band (668nm)"
                },
                4: {
                    "name": "NIR",
                    "wavelength": 840,
                    "description": "Near Infrared band (840nm)"
                },
                5: {
                    "name": "Red Edge",
                    "wavelength": 717,
                    "description": "Red Edge band (717nm)"
                }
            }
        }
    
    @staticmethod
    def load_from_file(config_path: Path) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required sections
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
                if section not in config:
                    raise ConfigurationError(f"Missing required section: {section}")
            
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    @staticmethod
    def save_to_file(config: Dict[str, Any], config_path: Path):
        """Save configuration to file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with base configuration"""
        def deep_merge(d1: Dict, d2: Dict) -> Dict:
            result = d1.copy()
            for key, value in d2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(base_config, user_config)

def validate_config(config: Dict) -> bool:
    """
    Validate configuration values
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Validate required sections
        required_sections = ['quality_control', 'vegetation_indices', 
                           'output_options', 'processing', 'band_config']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate processing options
        if config['processing']['max_workers'] < 1:
            raise ValueError("max_workers must be at least 1")
        
        if config['processing']['batch_size'] < 1:
            raise ValueError("batch_size must be at least 1")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False 