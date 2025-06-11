#!/usr/bin/env python3
"""
MicaSense Batch Processing Script
Process all 758 MicaSense RedEdge-M image sets automatically
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.append('/app/src')

from pipeline.micasense_processor import MicaSenseProcessor

def main():
    """Process MicaSense images from input directory"""
    
    input_dir = Path("/data/input")
    output_dir = Path("/data/output")
    
    print("🛰️  MicaSense RedEdge-M Batch Processor")
    print("=" * 50)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1
    
    # Count TIFF files
    tiff_files = list(input_dir.glob("**/*.tif")) + list(input_dir.glob("**/*.TIF"))
    print(f"📁 Found {len(tiff_files)} TIFF files in {input_dir}")
    
    if len(tiff_files) == 0:
        print("⚠️  No TIFF files found. Please add your MicaSense images to data/input/")
        return 1
    
    # Create processor
    processor = MicaSenseProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=4  # Adjust based on your system
    )
    
    # Find image sets first
    image_sets = processor.find_image_sets()
    print(f"🔍 Found {len(image_sets)} complete MicaSense image sets")
    
    if len(image_sets) == 0:
        print("⚠️  No complete MicaSense image sets found.")
        print("   Each set needs 5 bands: IMG_XXXX_1.tif through IMG_XXXX_5.tif")
        return 1
    
    print(f"🚀 Starting batch processing of {len(image_sets)} image sets...")
    print("   This will create:")
    print("   - Aligned multispectral TIFFs")
    print("   - Vegetation indices (NDVI, NDRE, GNDVI)")
    print("   - Metadata extraction")
    print("   - GPS coordinate enrichment")
    
    # Process all sets
    summary = processor.process_all()
    
    # Print results
    print("\n" + "=" * 50)
    print("🎉 Batch Processing Complete!")
    print(f"📊 Total image sets: {summary['total_sets']}")
    print(f"✅ Successfully processed: {summary['successful']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⏱️  Duration: {summary['duration_seconds']:.1f} seconds")
    
    if summary['successful'] > 0:
        print(f"\n📂 Output files created in:")
        print(f"   📁 Aligned images: {output_dir}/aligned/")
        print(f"   📈 Vegetation indices: {output_dir}/indices/")
        print(f"   📋 Metadata: {output_dir}/metadata/")
        print(f"   📄 Processing log: {output_dir}/processing.log")
    
    return 0 if summary['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
