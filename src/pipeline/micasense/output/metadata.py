"""
MicaSense Metadata Module
Handles generation and storage of processing metadata
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import rasterio
import numpy as np

class MetadataGenerator:
    """Handles generation and storage of processing metadata"""
    
    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger("MicaSenseMetadata")
        
        # Create metadata directory
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_metadata(self, image_path: str, image_set: Dict,
                         index_paths: Dict[str, str],
                         thumbnail_paths: Dict[str, str],
                         visualization_paths: Dict[str, str]) -> str:
        """
        Generate metadata for processed image set
        
        Args:
            image_path: Path to aligned multispectral image
            image_set: Dictionary containing image set information
            index_paths: Dictionary mapping index names to file paths
            thumbnail_paths: Dictionary mapping band names to thumbnail paths
            visualization_paths: Dictionary mapping index names to visualization paths
            
        Returns:
            Path to generated metadata file
        """
        try:
            # Get image statistics
            with rasterio.open(image_path) as src:
                stats = self._calculate_image_statistics(src)
                
            # Create metadata dictionary
            metadata = {
                "image_set": {
                    "name": image_set["name"],
                    "timestamp": datetime.now().isoformat(),
                    "band_files": image_set["band_files"],
                    "aligned_image": str(image_path)
                },
                "processing": {
                    "config": self.config,
                    "statistics": stats
                },
                "outputs": {
                    "indices": index_paths,
                    "thumbnails": thumbnail_paths,
                    "visualizations": visualization_paths
                }
            }
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{image_set['name']}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Generated metadata for {image_set['name']}")
            return str(metadata_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate metadata for {image_set['name']}: {e}")
            return None
    
    def _calculate_image_statistics(self, src: rasterio.DatasetReader) -> Dict:
        """Calculate statistics for each band in the image"""
        stats = {}
        
        for band_idx in range(1, src.count + 1):
            band_name = src.descriptions[band_idx - 1]
            data = src.read(band_idx)
            
            # Calculate basic statistics
            band_stats = {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "nodata_count": int(np.sum(data == src.nodata)) if src.nodata is not None else 0
            }
            
            # Calculate percentiles
            percentiles = np.percentile(data, [25, 50, 75])
            band_stats.update({
                "p25": float(percentiles[0]),
                "median": float(percentiles[1]),
                "p75": float(percentiles[2])
            })
            
            stats[band_name] = band_stats
        
        return stats
    
    def generate_processing_report(self, processed_sets: List[Dict]) -> str:
        """
        Generate a summary report of all processed image sets
        
        Args:
            processed_sets: List of dictionaries containing processing results
            
        Returns:
            Path to generated report file
        """
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "total_sets_processed": len(processed_sets),
                "processing_summary": []
            }
            
            for set_info in processed_sets:
                set_summary = {
                    "name": set_info["name"],
                    "status": set_info.get("status", "unknown"),
                    "indices_generated": list(set_info.get("indices", {}).keys()),
                    "error": set_info.get("error")
                }
                report["processing_summary"].append(set_summary)
            
            # Save report
            report_path = self.metadata_dir / "processing_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info("Generated processing report")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate processing report: {e}")
            return None 