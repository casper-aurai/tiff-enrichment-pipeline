#!/usr/bin/env python3
"""
TIFF Enrichment Pipeline - Main Processing Script
Automatically detects and processes TIFF files including MicaSense multispectral imagery
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

# Add src to Python path
sys.path.append('/app/src')

from pipeline.micasense.core.processor import MicaSenseProcessor
from pipeline.health import health_check

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class TIFFPipelineMain:
    """Main TIFF processing pipeline with auto-detection capabilities"""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = int(os.getenv('MAX_WORKERS', 4))
        self.batch_size = int(os.getenv('PROCESSING_BATCH_SIZE', 5))
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized pipeline: {self.input_dir} -> {self.output_dir}")
        logger.info(f"Workers: {self.max_workers}, Batch size: {self.batch_size}")
    
    def detect_file_types(self) -> Dict[str, List[Path]]:
        """
        Detect different types of TIFF files in the input directory
        Returns categorized file lists
        """
        file_types = {
            'micasense': [],
            'regular_tiff': [],
            'geotiff': [],
            'other': []
        }
        
        # Find all TIFF files
        tiff_patterns = ['**/*.tif', '**/*.tiff', '**/*.TIF', '**/*.TIFF']
        all_tiffs = []
        
        for pattern in tiff_patterns:
            all_tiffs.extend(self.input_dir.glob(pattern))
        
        logger.info(f"Found {len(all_tiffs)} TIFF files total")
        
        # Group MicaSense files by capture
        micasense_captures = {}
        regular_tiffs = []
        
        for tiff_file in all_tiffs:
            filename = tiff_file.name
            
            # Check if it's a MicaSense file (IMG_XXXX_Y.tif pattern)
            if self._is_micasense_file(filename):
                base_name = self._get_micasense_base_name(filename)
                if base_name:
                    if base_name not in micasense_captures:
                        micasense_captures[base_name] = []
                    micasense_captures[base_name].append(tiff_file)
                else:
                    regular_tiffs.append(tiff_file)
            else:
                regular_tiffs.append(tiff_file)
        
        # Filter complete MicaSense sets (5 bands)
        complete_micasense_sets = []
        for base_name, files in micasense_captures.items():
            if len(files) == 5:  # Complete 5-band set
                complete_micasense_sets.extend(files)
                file_types['micasense'].extend(files)
            else:
                # Incomplete sets go to regular processing
                regular_tiffs.extend(files)
                logger.warning(f"Incomplete MicaSense set {base_name}: {len(files)} bands (expected 5)")
        
        # Categorize regular TIFFs
        file_types['regular_tiff'] = regular_tiffs
        
        logger.info(f"Detected file types:")
        logger.info(f"  - MicaSense sets: {len(micasense_captures)} complete sets ({len(file_types['micasense'])} files)")
        logger.info(f"  - Regular TIFFs: {len(file_types['regular_tiff'])} files")
        
        return file_types
    
    def _is_micasense_file(self, filename: str) -> bool:
        """Check if filename matches MicaSense pattern"""
        import re
        # Pattern: IMG_XXXX_Y.tif where XXXX is capture number, Y is band number (1-5)
        pattern = r'^IMG_\d{4}_[1-5]\.(tif|TIF|tiff|TIFF)$'
        return bool(re.match(pattern, filename))
    
    def _get_micasense_base_name(self, filename: str) -> Optional[str]:
        """Extract base name from MicaSense filename"""
        import re
        match = re.match(r'^(IMG_\d{4})_[1-5]\.(tif|TIF|tiff|TIFF)$', filename)
        return match.group(1) if match else None
    
    def process_micasense_files(self, files: List[Path]) -> Dict:
        """Process MicaSense multispectral files"""
        logger.info("🛰️  Starting MicaSense RedEdge-M processing...")
        
        # Create MicaSense processor
        micasense_output = self.output_dir / "micasense"
        config = {
            'vegetation_indices': {
                'ndvi': True,
                'ndre': True,
                'gndvi': True
            },
            'quality_control': {
                'check_metadata': True,
                'check_dimensions': True,
                'check_data_range': True
            },
            'output_options': {
                'save_aligned': True,
                'save_calibrated': True,
                'save_indices': True,
                'save_metadata': True,
                'save_quality_report': True
            },
            'processing': {
                'max_workers': self.max_workers,
                'batch_size': self.batch_size
            },
            'logging': {
                'level': 'INFO',
                'save_to_file': True
            },
            'validation': {
                'min_dimensions': (100, 100),
                'valid_data_types': ['uint16', 'uint8'],
                'data_range': (0, 65535),
                'max_zero_ratio': 0.5,
                'required_metadata': ['DateTime'],  # Only require DateTime
                'known_metadata': {
                    'CameraModel': 'MicaSense RedEdge-M',
                    'GPSMapDatum': 'WGS84'
                }
            },
            'band_config': {
                1: {
                    'name': 'Blue',
                    'wavelength': 475,
                    'description': 'Blue band',
                    'camera': 'MicaSense RedEdge-M'
                },
                2: {
                    'name': 'Green',
                    'wavelength': 560,
                    'description': 'Green band',
                    'camera': 'MicaSense RedEdge-M'
                },
                3: {
                    'name': 'Red',
                    'wavelength': 668,
                    'description': 'Red band',
                    'camera': 'MicaSense RedEdge-M'
                },
                4: {
                    'name': 'NIR',
                    'wavelength': 840,
                    'description': 'Near Infrared band',
                    'camera': 'MicaSense RedEdge-M'
                },
                5: {
                    'name': 'Red Edge',
                    'wavelength': 717,
                    'description': 'Red Edge band',
                    'camera': 'MicaSense RedEdge-M'
                }
            },
            'camera_info': {
                'model': 'MicaSense RedEdge-M',
                'bands': 5,
                'gps_datum': 'WGS84',
                'band_order': [1, 2, 3, 4, 5]  # Blue, Green, Red, NIR, Red Edge
            }
        }
        processor = MicaSenseProcessor(
            config=config,
            output_dir=micasense_output
        )
        
        # Group files into sets
        file_sets = self._group_micasense_files(files)
        
        # Process first set as a test
        if file_sets:
            first_set = file_sets[0]
            summary = processor.process_single_set(first_set)
        else:
            summary = {
                'status': 'error',
                'error': 'No complete MicaSense sets found'
            }
        
        logger.info(f"MicaSense processing completed:")
        logger.info(f"  - Total sets: {len(file_sets)}")
        logger.info(f"  - Status: {summary.get('status', 'unknown')}")
        if 'error' in summary:
            logger.error(f"  - Error: {summary['error']}")
        
        return summary
    
    def _group_micasense_files(self, files: List[Path]) -> List[Dict]:
        """Group MicaSense files into complete sets"""
        sets = {}
        
        for file in files:
            base_name = self._get_micasense_base_name(file.name)
            if base_name:
                if base_name not in sets:
                    sets[base_name] = {
                        'name': base_name,
                        'bands': {}
                    }
                # Extract band number from filename
                band_num = int(file.name.split('_')[-1].split('.')[0])
                sets[base_name]['bands'][band_num] = str(file)
        
        # Filter complete sets (must have all 5 bands)
        complete_sets = []
        for set_name, set_data in sets.items():
            if len(set_data['bands']) == 5:
                complete_sets.append(set_data)
            else:
                logger.warning(f"Incomplete set {set_name}: found {len(set_data['bands'])} bands")
        
        return complete_sets
    
    def process_regular_tiffs(self, files: List[Path]) -> Dict:
        """Process regular TIFF files with standard enrichment"""
        logger.info(f"📄 Starting regular TIFF processing for {len(files)} files...")
        
        processed = 0
        failed = 0
        results = []
        
        for tiff_file in files:
            try:
                result = self._process_single_tiff(tiff_file)
                if result['status'] == 'success':
                    processed += 1
                else:
                    failed += 1
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {tiff_file}: {e}")
                failed += 1
                results.append({
                    'file': str(tiff_file),
                    'status': 'failed',
                    'error': str(e)
                })
        
        summary = {
            'type': 'regular_tiff',
            'total_files': len(files),
            'successful': processed,
            'failed': failed,
            'results': results
        }
        
        logger.info(f"Regular TIFF processing completed:")
        logger.info(f"  - Total files: {len(files)}")
        logger.info(f"  - Successful: {processed}")
        logger.info(f"  - Failed: {failed}")
        
        return summary
    
    def _process_single_tiff(self, tiff_file: Path) -> Dict:
        """Process a single TIFF file with metadata extraction and enrichment"""
        try:
            import rasterio
            from rasterio.crs import CRS
            
            result = {
                'file': str(tiff_file),
                'status': 'processing',
                'metadata': {},
                'output_files': {}
            }
            
            # Read basic metadata
            with rasterio.open(tiff_file) as src:
                metadata = {
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs) if src.crs else None,
                    'bounds': list(src.bounds),
                    'transform': list(src.transform)
                }
                
                # Extract tags
                tags = src.tags()
                if tags:
                    metadata['tags'] = tags
                
                result['metadata'] = metadata
            
            # Create output paths
            output_tiff = self.output_dir / "tiff" / tiff_file.name
            output_json = self.output_dir / "json" / f"{tiff_file.stem}_metadata.json"
            
            # Ensure output directories exist
            output_tiff.parent.mkdir(parents=True, exist_ok=True)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy enhanced TIFF (placeholder for actual enhancement)
            import shutil
            shutil.copy2(tiff_file, output_tiff)
            
            # Save metadata
            with open(output_json, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            result['output_files'] = {
                'tiff': str(output_tiff),
                'json': str(output_json)
            }
            result['status'] = 'success'
            
            logger.info(f"✅ Processed: {tiff_file.name}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"❌ Failed: {tiff_file.name} - {e}")
        
        return result
    
    def run(self) -> Dict:
        """Main processing pipeline"""
        start_time = datetime.now()
        logger.info("🚀 Starting TIFF Enrichment Pipeline...")
        
        # Check system health first
        if not health_check():
            logger.error("❌ Health check failed - aborting processing")
            return {'status': 'failed', 'error': 'Health check failed'}
        
        # Detect file types
        file_types = self.detect_file_types()
        
        if not any(files for files in file_types.values()):
            logger.warning("⚠️  No TIFF files found to process")
            return {'status': 'no_files', 'message': 'No TIFF files found'}
        
        # Process different file types
        processing_results = {}
        
        # Process MicaSense files if found
        if file_types['micasense']:
            logger.info(f"🛰️  Processing {len(file_types['micasense'])} MicaSense files...")
            micasense_result = self.process_micasense_files(file_types['micasense'])
            processing_results['micasense'] = micasense_result
        
        # Process regular TIFFs
        if file_types['regular_tiff']:
            logger.info(f"📄 Processing {len(file_types['regular_tiff'])} regular TIFF files...")
            regular_result = self.process_regular_tiffs(file_types['regular_tiff'])
            processing_results['regular_tiff'] = regular_result
        
        # Calculate overall summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        total_files = sum(len(files) for files in file_types.values())
        total_successful = sum(
            result.get('successful', 0) for result in processing_results.values()
        )
        total_failed = sum(
            result.get('failed', 0) for result in processing_results.values()
        )
        
        summary = {
            'status': 'completed',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_files': total_files,
            'successful': total_successful,
            'failed': total_failed,
            'file_types': {k: len(v) for k, v in file_types.items()},
            'processing_results': processing_results
        }
        
        # Save overall summary
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Final status
        logger.info("🎉 Pipeline processing completed!")
        logger.info(f"📊 Summary:")
        logger.info(f"   - Total files: {total_files}")
        logger.info(f"   - Successful: {total_successful}")
        logger.info(f"   - Failed: {total_failed}")
        logger.info(f"   - Duration: {duration:.1f} seconds")
        logger.info(f"   - Summary saved: {summary_path}")
        
        return summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TIFF Enrichment Pipeline")
    parser.add_argument(
        '--input-dir', 
        default='/data/input',
        help='Input directory containing TIFF files'
    )
    parser.add_argument(
        '--output-dir', 
        default='/data/output',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--health-check-only', 
        action='store_true',
        help='Only run health check and exit'
    )
    parser.add_argument(
        '--detect-only', 
        action='store_true',
        help='Only detect file types and exit'
    )
    parser.add_argument(
        '--micasense-only', 
        action='store_true',
        help='Only process MicaSense files'
    )
    
    args = parser.parse_args()
    
    # Health check only
    if args.health_check_only:
        logger.info("🏥 Running health check...")
        is_healthy = health_check()
        sys.exit(0 if is_healthy else 1)
    
    # Initialize pipeline
    pipeline = TIFFPipelineMain(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Detection only
    if args.detect_only:
        logger.info("🔍 Detecting file types...")
        file_types = pipeline.detect_file_types()
        print(json.dumps({k: len(v) for k, v in file_types.items()}, indent=2))
        sys.exit(0)
    
    # MicaSense only
    if args.micasense_only:
        logger.info("🛰️  MicaSense-only processing mode...")
        file_types = pipeline.detect_file_types()
        if file_types['micasense']:
            result = pipeline.process_micasense_files(file_types['micasense'])
            sys.exit(0 if result.get('failed', 0) == 0 else 1)
        else:
            logger.warning("No MicaSense files found")
            sys.exit(1)
    
    # Full pipeline
    try:
        summary = pipeline.run()
        
        # Exit with error code if any files failed
        exit_code = 0 if summary.get('failed', 0) == 0 else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
