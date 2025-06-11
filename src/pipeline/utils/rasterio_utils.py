"""
Rasterio Utilities Module
Provides safe and efficient rasterio operations with proper resource management
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Generator
import time
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioIOError, RasterioError
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class RasterioMetrics:
    """Metrics collection for rasterio operations"""
    def __init__(self):
        self.operation_times = {}
        self.error_counts = {}
    
    def record_operation(self, operation_name: str, duration: float):
        """Record operation duration"""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        self.operation_times[operation_name].append(duration)
    
    def record_error(self, operation_name: str):
        """Record operation error"""
        self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1

class RasterioManager:
    """Manages rasterio operations with proper resource handling"""
    
    def __init__(self):
        self.metrics = RasterioMetrics()
        self._metadata_cache = {}
    
    @contextmanager
    def safe_open(self, file_path: Path, mode: str = 'r'):
        """Safely open a rasterio dataset with proper error handling"""
        start_time = time.time()
        try:
            with rasterio.open(file_path, mode) as src:
                yield src
        except RasterioIOError as e:
            self.metrics.record_error('file_open')
            logger.error(f"Failed to open file {file_path}: {e}")
            raise
        except RasterioError as e:
            self.metrics.record_error('rasterio_operation')
            logger.error(f"Rasterio error with file {file_path}: {e}")
            raise
        except Exception as e:
            self.metrics.record_error('unexpected')
            logger.error(f"Unexpected error with file {file_path}: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.metrics.record_operation('file_open', duration)
    
    @lru_cache(maxsize=100)
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get cached metadata for a file"""
        with self.safe_open(Path(file_path)) as src:
            return {
                'crs': src.crs,
                'transform': src.transform,
                'dtype': src.dtypes[0],
                'count': src.count,
                'width': src.width,
                'height': src.height
            }
    
    def validate_dataset(self, src: rasterio.DatasetReader) -> Tuple[bool, Optional[str]]:
        """Validate a rasterio dataset"""
        try:
            if not src.crs:
                return False, "Missing CRS information"
            
            if not src.transform.is_identity:
                return False, "Invalid transform"
            
            if src.count == 0:
                return False, "No bands found"
            
            return True, None
        except Exception as e:
            return False, str(e)
    
    def process_in_chunks(self, file_path: Path, chunk_size: int = 1024,
                         callback: callable = None) -> Generator[np.ndarray, None, None]:
        """Process a rasterio dataset in chunks"""
        with self.safe_open(file_path) as src:
            is_valid, error_msg = self.validate_dataset(src)
            if not is_valid:
                raise ValueError(f"Invalid dataset: {error_msg}")
            
            for i in range(0, src.height, chunk_size):
                for j in range(0, src.width, chunk_size):
                    window = Window(
                        j, i,
                        min(chunk_size, src.width - j),
                        min(chunk_size, src.height - i)
                    )
                    chunk = src.read(window=window)
                    if callback:
                        chunk = callback(chunk)
                    yield chunk
    
    def safe_write(self, file_path: Path, data: np.ndarray,
                  profile: Dict[str, Any]) -> bool:
        """Safely write data to a rasterio dataset"""
        start_time = time.time()
        try:
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(data)
            return True
        except Exception as e:
            self.metrics.record_error('file_write')
            logger.error(f"Failed to write file {file_path}: {e}")
            return False
        finally:
            duration = time.time() - start_time
            self.metrics.record_operation('file_write', duration)
    
    def get_statistics(self, file_path: Path) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each band in a dataset"""
        stats = {}
        with self.safe_open(file_path) as src:
            for band_idx in range(1, src.count + 1):
                data = src.read(band_idx)
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
                
                stats[f"band_{band_idx}"] = band_stats 