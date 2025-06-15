#!/usr/bin/env python3

import sys
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Rectangle
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import os

def get_tiff_bounds(tiff_path):
    """Extract bounds and CRS from a TIFF file"""
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        crs = src.crs
        return bounds, crs

def create_location_map(tiff_files, output_path):
    """Create a map visualization of TIFF file locations"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Convert all bounds to EPSG:4326 (WGS84) for consistent plotting
    bounds_list = []
    for tiff_path in tiff_files:
        bounds, crs = get_tiff_bounds(tiff_path)
        if crs and crs.to_epsg() != 4326:
            # Convert bounds to WGS84 if needed
            from rasterio.warp import transform_bounds
            bounds = transform_bounds(crs, 'EPSG:4326', *bounds)
        bounds_list.append(bounds)
    
    # Create a GeoDataFrame with all bounds
    geometries = [box(bounds[0], bounds[1], bounds[2], bounds[3]) for bounds in bounds_list]
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    
    # Plot the bounds
    gdf.plot(ax=ax, alpha=0.5, edgecolor='red', facecolor='none')
    
    # Add basemap
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    
    # Set title and labels
    ax.set_title('TIFF File Locations in the Netherlands', fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add file names as labels
    for i, (bounds, tiff_path) in enumerate(zip(bounds_list, tiff_files)):
        x = (bounds[0] + bounds[2]) / 2
        y = bounds[3] + 0.001  # Slightly above the box
        ax.text(x, y, Path(tiff_path).name, 
                ha='center', va='bottom', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Save the map
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_tiff_locations.py <input_dir>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        sys.exit(1)

    # Always write to maps/ subdirectory of input_dir
    maps_dir = input_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    output_path = maps_dir / "tiff_locations_map.png"
    
    # Find all TIFF files
    tiff_files = list(input_dir.glob('**/*.tif')) + list(input_dir.glob('**/*.TIF'))
    
    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(tiff_files)} TIFF files")
    create_location_map(tiff_files, output_path)
    print(f"Map saved to {output_path}")

if __name__ == '__main__':
    main() 