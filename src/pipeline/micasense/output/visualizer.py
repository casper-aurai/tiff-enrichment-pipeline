"""
MicaSense Visualization Module
Handles generation of thumbnails and visualizations for processed images
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, List, Optional
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
from datetime import datetime
import traceback

class ImageVisualizer:
    """Handles visualization of MicaSense images and indices"""
    
    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger("MicaSenseVisualizer")
        
        # Create output directories
        self.thumbnails_dir = self.output_dir / "thumbnails"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        # Define colormaps for different indices
        self.ndvi_cmap = LinearSegmentedColormap.from_list(
            'ndvi', ['darkred', 'red', 'yellow', 'lightgreen', 'green', 'darkgreen'], N=256)
        self.ndre_cmap = LinearSegmentedColormap.from_list(
            'ndre', ['darkred', 'red', 'yellow', 'lightgreen', 'green', 'darkgreen'], N=256)
        self.ndwi_cmap = LinearSegmentedColormap.from_list(
            'ndwi', ['blue', 'white', 'green'], N=256)
        self.evi_cmap = LinearSegmentedColormap.from_list(
            'evi', ['red', 'yellow', 'green'], N=256)
        self.savi_cmap = LinearSegmentedColormap.from_list(
            'savi', ['red', 'yellow', 'green'], N=256)
    
    def generate_thumbnails(self, image_path: str, image_set: Dict) -> Dict[str, str]:
        """
        Generate thumbnails for all bands and indices
        
        Args:
            image_path: Path to aligned multispectral image
            image_set: Dictionary containing image set information
            
        Returns:
            Dictionary mapping band/index names to thumbnail paths
        """
        thumbnail_paths = {}
        
        try:
            with rasterio.open(image_path) as src:
                # Generate thumbnails for each band
                for band_idx in range(1, src.count + 1):
                    band_name = src.descriptions[band_idx - 1]
                    band_data = src.read(band_idx)
                    
                    thumbnail_path = self.thumbnails_dir / f"{image_set['name']}_{band_name}_thumb.png"
                    self._save_thumbnail(band_data, thumbnail_path, band_name)
                    thumbnail_paths[band_name] = str(thumbnail_path)
                
                self.logger.info(f"Generated thumbnails for {image_set['name']}")
                
        except Exception as e:
            self.logger.error(f"Failed to generate thumbnails for {image_set['name']}: {e}")
            
        return thumbnail_paths
    
    def generate_index_visualizations(self, index_paths: Dict[str, str], 
                                    image_set: Dict) -> Dict[str, str]:
        """
        Generate visualizations for vegetation indices
        
        Args:
            index_paths: Dictionary mapping index names to file paths
            image_set: Dictionary containing image set information
            
        Returns:
            Dictionary mapping index names to visualization paths
        """
        visualization_paths = {}
        
        try:
            for index_name, index_path in index_paths.items():
                with rasterio.open(index_path) as src:
                    index_data = src.read(1)
                    
                    # Select appropriate colormap and value range
                    if index_name in ['NDVI', 'GNDVI', 'NDRE']:
                        cmap = self.ndvi_cmap
                        vmin, vmax = -0.2, 1.0  # Adjusted range to better show vegetation
                    elif index_name == 'NDWI':
                        cmap = self.ndwi_cmap
                        vmin, vmax = -1, 1
                    elif index_name == 'EVI':
                        cmap = self.evi_cmap
                        vmin, vmax = -1, 1
                    elif index_name in ['SAVI', 'MSAVI', 'OSAVI']:
                        cmap = self.savi_cmap
                        vmin, vmax = -1, 1
                    else:
                        cmap = 'viridis'
                        vmin, vmax = np.min(index_data), np.max(index_data)
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(10, 10))
                    im = ax.imshow(index_data, cmap=cmap, vmin=vmin, vmax=vmax)
                    plt.colorbar(im, ax=ax, label=index_name)
                    ax.set_title(f"{index_name} - {image_set['name']}")
                    
                    # Add statistics as text
                    stats = {
                        'Mean': np.mean(index_data),
                        'Std': np.std(index_data),
                        'Min': np.min(index_data),
                        'Max': np.max(index_data),
                        'Vegetation %': float(np.sum((index_data > 0.2) & (index_data <= 1.0)) / index_data.size * 100),
                        'High Veg %': float(np.sum((index_data > 0.5) & (index_data <= 1.0)) / index_data.size * 100)
                    }
                    stats_text = '\n'.join([f"{k}: {v:.3f}" for k, v in stats.items()])
                    ax.text(0.02, 0.98, stats_text,
                           transform=ax.transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Save visualization
                    vis_path = self.visualizations_dir / f"{image_set['name']}_{index_name}_vis.png"
                    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    visualization_paths[index_name] = str(vis_path)
            
            self.logger.info(f"Generated visualizations for {image_set['name']}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualizations for {image_set['name']}: {e}")
            self.logger.error(traceback.format_exc())
            
        return visualization_paths
    
    def _save_thumbnail(self, data: np.ndarray, output_path: Path, band_name: str):
        """Save thumbnail image"""
        plt.figure(figsize=(5, 5))
        plt.imshow(data, cmap='gray')
        plt.title(f"{band_name} Band")
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close() 