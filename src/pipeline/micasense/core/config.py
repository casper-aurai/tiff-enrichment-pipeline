"""
MicaSense Configuration Module
Handles configuration loading and validation
"""

import json
from pathlib import Path
from typing import Dict, Any
import multiprocessing as mp

DEFAULT_CONFIG = {
    'quality_control': {
        'check_band_alignment': True,
        'validate_reflectance_range': True,
        'generate_histograms': True,
        'validate_dimensions': True,
        'check_zero_ratio': True
    },
    'vegetation_indices': {
        'ndvi': True,
        'ndre': True,
        'gndvi': True,
        'savi': False,
        'msavi': False,
        'evi': False
    },
    'output_options': {
        'save_individual_bands': True,
        'generate_thumbnails': True,
        'overwrite_existing': False,
        'save_quality_reports': True,
        'save_metadata': True
    },
    'processing': {
        'radiometric_calibration': True,
        'band_alignment': True,
        'generate_indices': True,
        'max_workers': mp.cpu_count() - 1,
        'batch_size': 10
    },
    'band_config': {
        1: {"name": "Blue", "center_wavelength": 475, "bandwidth": 20},
        2: {"name": "Green", "center_wavelength": 560, "bandwidth": 20},
        3: {"name": "Red", "center_wavelength": 668, "bandwidth": 10},
        4: {"name": "Near IR", "center_wavelength": 840, "bandwidth": 40},
        5: {"name": "Red Edge", "center_wavelength": 717, "bandwidth": 10}
    }
}

def load_config(config_path: Path = None, user_config: Dict = None) -> Dict[str, Any]:
    """
    Load and validate configuration
    
    Args:
        config_path: Path to configuration JSON file
        user_config: Dictionary with user configuration
        
    Returns:
        Merged configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load from file if provided
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            config = deep_merge(config, file_config)
    
    # Override with user config if provided
    if user_config:
        config = deep_merge(config, user_config)
    
    return config

def deep_merge(d1: Dict, d2: Dict) -> Dict:
    """
    Deep merge two dictionaries
    
    Args:
        d1: Base dictionary
        d2: Dictionary to merge into d1
        
    Returns:
        Merged dictionary
    """
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            deep_merge(d1[k], v)
        else:
            d1[k] = v
    return d1

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