#!/usr/bin/env python3
"""
MicaSense Batch Processing Script
Process all MicaSense RedEdge-M image sets automatically using the modular processor
"""

import sys
import os
from pathlib import Path
import json
import logging

# Add src to Python path
sys.path.append('/app/src')

from pipeline.micasense.core.processor import MicaSenseProcessor
from pipeline.micasense.core.config import load_config, validate_config

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("MicaSenseProcessor")
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = output_dir / "processing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def create_default_config(output_dir: Path) -> dict:
    """Create default configuration"""
    return {
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
            'savi': True,
            'msavi': True,
            'evi': True
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
            'max_workers': 4,  # Adjust based on your system
            'batch_size': 10
        },
        'logging': {
            'level': 'INFO',
            'file': str(output_dir / "processing.log")
        }
    }

def main():
    """Process MicaSense images from input directory"""
    
    input_dir = Path("/data/input")
    output_dir = Path("/data/output")
    
    print("🛰️  MicaSense RedEdge-M Batch Processor")
    print("=" * 50)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    # Count TIFF files
    tiff_files = list(input_dir.glob("**/*.tif")) + list(input_dir.glob("**/*.TIF"))
    logger.info(f"Found {len(tiff_files)} TIFF files in {input_dir}")
    
    if len(tiff_files) == 0:
        logger.warning("No TIFF files found. Please add your MicaSense images to data/input/")
        return 1
    
    # Create and validate configuration
    config = create_default_config(output_dir)
    if not validate_config(config):
        logger.error("Invalid configuration")
        return 1
    
    # Create processor
    processor = MicaSenseProcessor(config)
    
    # Find image sets
    image_sets = processor.find_image_sets(input_dir)
    logger.info(f"Found {len(image_sets)} complete MicaSense image sets")
    
    if len(image_sets) == 0:
        logger.warning("No complete MicaSense image sets found.")
        logger.warning("Each set needs 5 bands: IMG_XXXX_1.tif through IMG_XXXX_5.tif")
        return 1
    
    logger.info(f"Starting batch processing of {len(image_sets)} image sets...")
    logger.info("This will create:")
    logger.info("- Aligned multispectral TIFFs")
    logger.info("- Vegetation indices (NDVI, NDRE, GNDVI, SAVI, MSAVI, EVI)")
    logger.info("- Metadata extraction")
    logger.info("- Quality reports")
    logger.info("- Visualizations")
    
    # Process all sets
    results = processor.process_all(image_sets)
    
    # Generate summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    # Print results
    print("\n" + "=" * 50)
    print("🎉 Batch Processing Complete!")
    print(f"📊 Total image sets: {len(results)}")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    
    if successful > 0:
        print(f"\n📂 Output files created in:")
        print(f"   📁 Aligned images: {output_dir}/aligned/")
        print(f"   📈 Vegetation indices: {output_dir}/indices/")
        print(f"   📋 Metadata: {output_dir}/metadata/")
        print(f"   📊 Quality reports: {output_dir}/quality_reports/")
        print(f"   🖼️  Visualizations: {output_dir}/visualizations/")
        print(f"   📄 Processing log: {output_dir}/processing.log")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
