"""
MicaSense Processor Main Entry Point
Command-line interface for processing MicaSense images
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict
import json

from .core.processor import MicaSenseProcessor
from .core.config import load_config, validate_config

def find_image_sets(input_dir: Path) -> List[Dict]:
    """Find complete sets of MicaSense images in the input directory"""
    image_sets = []
    
    # Group files by timestamp
    timestamp_groups = {}
    for file_path in input_dir.glob("*.tif"):
        if "_" not in file_path.stem:
            continue
            
        timestamp = file_path.stem.split("_")[0]
        if timestamp not in timestamp_groups:
            timestamp_groups[timestamp] = []
        timestamp_groups[timestamp].append(file_path)
    
    # Create image sets
    for timestamp, files in timestamp_groups.items():
        if len(files) >= 5:  # Complete set has at least 5 bands
            band_files = {
                "Blue": str(files[0]),
                "Green": str(files[1]),
                "Red": str(files[2]),
                "NIR": str(files[3]),
                "RedEdge": str(files[4])
            }
            
            image_sets.append({
                "name": timestamp,
                "band_files": band_files
            })
    
    return image_sets

def main():
    parser = argparse.ArgumentParser(description="Process MicaSense multispectral images")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Output directory")
    args = parser.parse_args()
    
    # Set default output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("/data/output/micasense")
    
    # Load configuration
    config = load_config(args.config)
    config['output'] = config.get('output', {})
    config['output']['directory'] = str(output_dir)
    validate_config(config)
    
    # Setup logging
    log_path = output_dir / "processing.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config['logging']['level'].upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )
    
    # Find image sets
    input_dir = Path(args.input_dir)
    image_sets = find_image_sets(input_dir)
    
    if not image_sets:
        logging.error("No complete image sets found")
        return
    
    # Process images
    processor = MicaSenseProcessor(config, output_dir)
    results = processor.process_all(image_sets)
    
    # Print and write summary
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\nProcessing complete:")
    print(f"Total sets: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    # Write summary JSON to both locations
    summary = {
        "total_sets": len(results),
        "successful": success_count,
        "failed": len(results) - success_count,
        "results": results
    }
    with open(output_dir / "processing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(Path("/data/output/processing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main() 