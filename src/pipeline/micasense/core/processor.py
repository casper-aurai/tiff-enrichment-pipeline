"""
MicaSense Core Processor Module
Main processor class that orchestrates the processing pipeline
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures
from datetime import datetime

from ..quality.validator import BandValidator
from ..processing.processor import BandProcessor
from ..processing.indices import VegetationIndices
from ..output.visualizer import ImageVisualizer
from ..output.metadata import MetadataGenerator
from .config import load_config, validate_config

class MicaSenseProcessor:
    """Main processor class for MicaSense image processing"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load and validate configuration
        self.config = load_config(config_path)
        validate_config(self.config)
        
        # Setup logging
        self.logger = logging.getLogger("MicaSenseProcessor")
        self._setup_logging()
        
        # Initialize output directory
        self.output_dir = Path(self.config['output']['directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validator = BandValidator(self.config, self.output_dir)
        self.processor = BandProcessor(self.config, self.output_dir)
        self.indices = VegetationIndices(self.config, self.output_dir)
        self.visualizer = ImageVisualizer(self.config, self.output_dir)
        self.metadata = MetadataGenerator(self.config, self.output_dir)
    
    def _setup_logging(self):
        """Configure logging"""
        log_level = getattr(logging, self.config['logging']['level'].upper())
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config['logging']['file'])
            ]
        )
    
    def process_image_set(self, image_set: Dict) -> Dict:
        """
        Process a single image set
        
        Args:
            image_set: Dictionary containing image set information
            
        Returns:
            Dictionary containing processing results
        """
        result = {
            "name": image_set["name"],
            "status": "failed",
            "error": None
        }
        
        try:
            # Validate bands
            is_valid, validation_results = self.validator.validate_bands(image_set)
            if not is_valid:
                result["error"] = "Band validation failed"
                return result
            
            # Align bands
            aligned_path = self.processor.align_bands(image_set)
            if not aligned_path:
                result["error"] = "Band alignment failed"
                return result
            
            # Apply radiometric calibration
            calibrated_path = self.processor.radiometric_calibration(aligned_path)
            if not calibrated_path:
                result["error"] = "Radiometric calibration failed"
                return result
            
            # Calculate vegetation indices
            index_paths = self.indices.calculate_indices(calibrated_path, image_set)
            if not index_paths:
                result["error"] = "Index calculation failed"
                return result
            
            # Generate thumbnails
            thumbnail_paths = self.visualizer.generate_thumbnails(calibrated_path, image_set)
            
            # Generate visualizations
            visualization_paths = self.visualizer.generate_index_visualizations(
                index_paths, image_set)
            
            # Generate metadata
            metadata_path = self.metadata.generate_metadata(
                calibrated_path, image_set, index_paths,
                thumbnail_paths, visualization_paths)
            
            # Update result
            result.update({
                "status": "success",
                "aligned_image": aligned_path,
                "calibrated_image": calibrated_path,
                "indices": index_paths,
                "thumbnails": thumbnail_paths,
                "visualizations": visualization_paths,
                "metadata": metadata_path
            })
            
        except Exception as e:
            self.logger.error(f"Error processing {image_set['name']}: {e}")
            result["error"] = str(e)
        
        return result
    
    def process_all(self, image_sets: List[Dict]) -> List[Dict]:
        """
        Process multiple image sets in parallel
        
        Args:
            image_sets: List of dictionaries containing image set information
            
        Returns:
            List of dictionaries containing processing results
        """
        results = []
        max_workers = self.config['processing']['max_workers']
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_set = {
                executor.submit(self.process_image_set, image_set): image_set
                for image_set in image_sets
            }
            
            for future in concurrent.futures.as_completed(future_to_set):
                image_set = future_to_set[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {image_set['name']}: {e}")
                    results.append({
                        "name": image_set["name"],
                        "status": "failed",
                        "error": str(e)
                    })
        
        # Generate processing report
        self.metadata.generate_processing_report(results)
        
        return results 