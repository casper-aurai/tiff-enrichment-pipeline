class MicaSenseConfig:
    """Configuration for MicaSense processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("MicaSenseConfig")
        
        # Validate required fields
        required_fields = ['input_dir', 'output_dir', 'band_order']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Set paths
        self.input_dir = Path(config['input_dir'])
        self.output_dir = Path(config['output_dir'])
        
        # Set band order
        self.band_order = config['band_order']
        
        # Set processing options
        self.align_images = config.get('align_images', True)
        self.calibrate_images = config.get('calibrate_images', True)
        self.calculate_indices = config.get('calculate_indices', True)
        
        # Set vegetation indices options
        self.indices = {
            'NDVI': config.get('calculate_ndvi', True),  # Normalized Difference Vegetation Index
            'NDRE': config.get('calculate_ndre', True),  # Normalized Difference Red Edge Index
            'GNDVI': config.get('calculate_gndvi', True),  # Green Normalized Difference Vegetation Index
            'SAVI': config.get('calculate_savi', True),  # Soil Adjusted Vegetation Index
            'MSAVI': config.get('calculate_msavi', True),  # Modified Soil Adjusted Vegetation Index
            'EVI': config.get('calculate_evi', True),  # Enhanced Vegetation Index
            'OSAVI': config.get('calculate_osavi', True),  # Optimized Soil Adjusted Vegetation Index
            'NDWI': config.get('calculate_ndwi', True),  # Normalized Difference Water Index
        }
        
        # Set visualization options
        self.generate_thumbnails = config.get('generate_thumbnails', True)
        self.generate_visualizations = config.get('generate_visualizations', True)
        
        # Set alignment parameters
        self.alignment_params = {
            'max_shift': config.get('max_shift', 50),
            'max_rotation': config.get('max_rotation', 5),
            'max_scale': config.get('max_scale', 0.1)
        }
        
        # Set calibration parameters
        self.calibration_params = {
            'dark_level': config.get('dark_level', 0),
            'gain': config.get('gain', 1.0)
        }
        
        # Set output parameters
        self.output_params = {
            'dtype': config.get('output_dtype', 'float32'),
            'compress': config.get('compress', 'lzw'),
            'tiled': config.get('tiled', True),
            'blockxsize': config.get('blockxsize', 256),
            'blockysize': config.get('blockysize', 256)
        } 