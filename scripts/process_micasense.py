#!/usr/bin/env python3
"""
MicaSense Image Processing Script
Processes MicaSense images from input directory and outputs results to specified output directory
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

from pipeline.micasense.core.processor import MicaSenseProcessor
from pipeline.micasense.core.config import MicaSenseConfig
from pipeline.micasense.core.errors import MicaSenseError, ConfigurationError, InputError

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging"""
    logger = logging.getLogger("MicaSenseProcessor")
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create handlers
    log_file = output_dir / "processing.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process MicaSense multispectral images"
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing MicaSense images"
    )
    
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for processed images and results"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (optional)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)"
    )
    
    return parser.parse_args()

def main():
    """Main processing function"""
    # Parse arguments
    args = parse_args()
    
    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting MicaSense image processing")
    
    try:
        # Load configuration
        if args.config:
            config = MicaSenseConfig.load_from_file(Path(args.config))
        else:
            config = MicaSenseConfig.create_default(output_dir)
        
        # Update max workers
        config['processing']['max_workers'] = args.max_workers
        
        # Create processor
        processor = MicaSenseProcessor(config, output_dir)
        
        # Find image sets
        logger.info(f"Searching for MicaSense image sets in {input_dir}")
        image_sets = processor.find_image_sets(input_dir)
        
        if not image_sets:
            logger.error("No complete MicaSense image sets found")
            sys.exit(1)
        
        logger.info(f"Found {len(image_sets)} complete image sets")
        
        # Process image sets
        results = []
        start_time = datetime.now()
        
        for image_set in image_sets:
            logger.info(f"Processing image set: {image_set['name']}")
            result = processor.process_single_set(image_set)
            results.append(result)
            
            if result["status"] == "success":
                logger.info(f"Successfully processed {image_set['name']}")
            else:
                logger.error(
                    f"Failed to process {image_set['name']}: "
                    f"{result['error']} ({result['error_type']})"
                )
        
        # Calculate processing time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        summary = {
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration,
            "total_sets": len(image_sets),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "errors": {
                r["name"]: {
                    "type": r["error_type"],
                    "message": r["error"],
                    "stage": r["failed_stage"]
                }
                for r in results if r["status"] == "failed"
            },
            "system_info": {
                "python_version": sys.version,
                "rasterio_version": rasterio.__version__,
                "numpy_version": np.__version__
            }
        }
        
        # Save summary
        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total image sets: {summary['total_sets']}")
        logger.info(f"Successfully processed: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Total processing time: {duration:.2f} seconds")
        
        if summary['failed'] > 0:
            logger.warning("\nFailed image sets:")
            for name, error in summary['errors'].items():
                logger.warning(f"- {name}: {error['message']} ({error['type']})")
        
        logger.info(f"\nDetailed summary saved to: {summary_path}")
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except InputError as e:
        logger.error(f"Input error: {str(e)}")
        sys.exit(1)
    except MicaSenseError as e:
        logger.error(f"Processing error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
