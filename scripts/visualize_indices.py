import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import json
import logging
from src.pipeline.micasense.output.visualizer import ImageVisualizer
from tqdm import tqdm
import concurrent.futures
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_index_file(index_file, visualizer):
    """Process a single index file"""
    try:
        # Create a basic image set dictionary
        image_set = {
            "name": index_file.stem
        }
        
        # Create index paths dictionary
        index_paths = {
            index_file.stem: str(index_file)
        }
        
        # Generate visualizations
        visualization_paths = visualizer.generate_index_visualizations(index_paths, image_set)
        return True, index_file.name
    except Exception as e:
        error_msg = f"Error processing {index_file.name}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return False, index_file.name

def main():
    # Define paths
    indices_dir = Path("/Users/aurai-cg/Desktop/PZG-SMG/tiff-enrichment-pipeline/data/output/micasense/indices")
    output_dir = Path("/Users/aurai-cg/Desktop/PZG-SMG/tiff-enrichment-pipeline/data/output/micasense/visualizations")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    config = {}  # Empty config as we don't need specific configuration
    visualizer = ImageVisualizer(config, output_dir)
    
    # Get all index files
    index_files = list(indices_dir.glob("*.tif"))
    total_files = len(index_files)
    logger.info(f"Found {total_files} index files to process")
    
    # Process files with progress bar
    successful = 0
    failed = 0
    
    with tqdm(total=total_files, desc="Processing indices") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_index_file, index_file, visualizer): index_file 
                for index_file in index_files
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                success, filename = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)
    
    logger.info(f"Processing complete. Successful: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main() 