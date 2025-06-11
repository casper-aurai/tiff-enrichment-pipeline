"""
MicaSense Quality Control Module
Handles band validation and quality reporting
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import logging

class BandValidator:
    """Validates MicaSense band quality and generates reports"""
    
    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger("MicaSenseValidator")
    
    def validate_bands(self, image_set: Dict) -> Tuple[bool, Dict]:
        """
        Validate image set quality and completeness
        
        Args:
            image_set: Dictionary containing band information
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        validation_results = {
            "name": image_set["name"],
            "timestamp": datetime.now().isoformat(),
            "bands": {},
            "quality_metrics": {},
            "is_valid": True,
            "errors": []
        }
        
        try:
            # Check if all bands exist and are readable
            for band_num, band_path in image_set["bands"].items():
                band_results = self._validate_single_band(band_num, band_path)
                validation_results["bands"][self.config['band_config'][band_num]["name"]] = band_results
                
                if not band_results["is_valid"]:
                    validation_results["is_valid"] = False
                    validation_results["errors"].extend(band_results["errors"])
            
            # Generate quality report if enabled
            if self.config['quality_control']['generate_histograms']:
                self._generate_quality_report(image_set, validation_results)
            
            return validation_results["is_valid"], validation_results
            
        except Exception as e:
            self.logger.error(f"Validation failed for {image_set['name']}: {e}")
            validation_results["is_valid"] = False
            validation_results["errors"].append(str(e))
            return False, validation_results
    
    def _validate_single_band(self, band_num: int, band_path: str) -> Dict:
        """Validate a single band"""
        results = {
            "is_valid": True,
            "errors": [],
            "stats": {}
        }
        
        try:
            with rasterio.open(band_path) as src:
                # Check image dimensions
                if self.config['quality_control']['validate_dimensions']:
                    if src.width < 100 or src.height < 100:
                        results["is_valid"] = False
                        results["errors"].append(
                            f"Dimensions too small: {src.width}x{src.height}"
                        )
                
                # Read data and calculate statistics
                data = src.read(1)
                stats = {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "zeros_ratio": float(np.sum(data == 0) / data.size)
                }
                results["stats"] = stats
                
                # Check data range
                if self.config['quality_control']['validate_reflectance_range']:
                    if stats["min"] < 0 or stats["max"] > 65535:
                        results["is_valid"] = False
                        results["errors"].append(
                            f"Invalid data range: [{stats['min']}, {stats['max']}]"
                        )
                
                # Check zero ratio
                if self.config['quality_control']['check_zero_ratio']:
                    if stats["zeros_ratio"] > 0.5:
                        results["is_valid"] = False
                        results["errors"].append(
                            f"Too many zeros: {stats['zeros_ratio']:.1%}"
                        )
        
        except Exception as e:
            results["is_valid"] = False
            results["errors"].append(f"Failed to validate band: {str(e)}")
        
        return results
    
    def _generate_quality_report(self, image_set: Dict, validation_results: Dict):
        """Generate quality control report"""
        try:
            # Save validation results
            report_path = self.output_dir / "quality_reports" / f"{image_set['name']}_quality_report.json"
            with open(report_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            # Generate histograms
            for band_num, band_path in image_set["bands"].items():
                with rasterio.open(band_path) as src:
                    data = src.read(1)
                    hist_path = self.output_dir / "quality_reports" / f"{image_set['name']}_{band_num}_histogram.png"
                    self._generate_histogram(
                        data, 
                        hist_path, 
                        self.config['band_config'][band_num]["name"]
                    )
            
            self.logger.info(f"Generated quality report for {image_set['name']}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate quality report: {e}")
    
    def _generate_histogram(self, data: np.ndarray, output_path: Path, band_name: str):
        """Generate histogram visualization"""
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(data.flatten(), bins=256, range=(0, 65535))
            plt.title(f"Histogram - {band_name} Band")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            
            plt.savefig(output_path)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate histogram: {e}") 