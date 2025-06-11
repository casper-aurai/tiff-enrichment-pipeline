"""
Rasterio Utilities Module
Provides safe and efficient rasterio operations with proper resource management
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Generator, List
import time
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioIOError, RasterioError
from contextlib import contextmanager
import psutil
from filelock import FileLock
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class RasterioMetrics:
    """Metrics collection for rasterio operations"""
    def __init__(self):
        self.operation_times = {}
        self.error_counts = {}
        self.memory_usage = {}
        self._lock = threading.Lock()
    
    def record_operation(self, operation_name: str, duration: float):
        """Record operation duration"""
        with self._lock:
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
            self.operation_times[operation_name].append(duration)
    
    def record_error(self, operation_name: str):
        """Record operation error"""
        with self._lock:
            self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
    
    def record_memory_usage(self, operation_name: str):
        """Record memory usage for an operation"""
        with self._lock:
            process = psutil.Process()
            memory_info = process.memory_info()
            if operation_name not in self.memory_usage:
                self.memory_usage[operation_name] = []
            self.memory_usage[operation_name].append(memory_info.rss / 1024 / 1024)  # MB

class RasterioManager:
    """Manages rasterio operations with proper resource handling"""
    
    def __init__(self, max_memory_mb: int = 1024, max_retries: int = 3):
        self.metrics = RasterioMetrics()
        self._metadata_cache = {}
        self._lock = threading.Lock()
        self.max_memory_mb = max_memory_mb
        self.max_retries = max_retries
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def __del__(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=True)
    
    @contextmanager
    def safe_open(self, file_path: Path, mode: str = 'r'):
        """Safely open a rasterio dataset with proper error handling"""
        start_time = time.time()
        lock_path = file_path.with_suffix('.lock')
        
        try:
            with FileLock(lock_path):
                with rasterio.open(file_path, mode) as src:
                    self.metrics.record_memory_usage('file_open')
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
            if lock_path.exists():
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def safe_open_network(self, file_path: Path, mode: str = 'r'):
        """Open file with retry logic for network files"""
        return self.safe_open(file_path, mode)
    
    def validate_file_integrity(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate file integrity"""
        try:
            # Check file size
            if not file_path.exists():
                return False, "File does not exist"
            
            if file_path.stat().st_size == 0:
                return False, "File is empty"
            
            # Try to open and read a small portion
            with self.safe_open(file_path) as src:
                # Read first pixel
                src.read(1, window=Window(0, 0, 1, 1))
                
                # Calculate file hash
                file_hash = self._calculate_file_hash(file_path)
                
                # Store hash for future comparison
                self._store_file_hash(file_path, file_hash)
                
                return True, None
        except Exception as e:
            return False, str(e)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _store_file_hash(self, file_path: Path, file_hash: str):
        """Store file hash for future comparison"""
        hash_file = file_path.with_suffix('.hash')
        with open(hash_file, 'w') as f:
            f.write(file_hash)
    
    def check_memory_usage(self) -> bool:
        """Check if current memory usage is within limits"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb <= self.max_memory_mb
    
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
            
            # Check for corrupted data
            for band in range(1, src.count + 1):
                try:
                    src.read(band, window=Window(0, 0, 1, 1))
                except Exception as e:
                    return False, f"Corrupted data in band {band}: {str(e)}"
            
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
                    # Check memory usage
                    if not self.check_memory_usage():
                        logger.warning("Memory usage exceeded limit, waiting for cleanup...")
                        time.sleep(1)  # Wait for potential cleanup
                        if not self.check_memory_usage():
                            raise MemoryError("Memory usage exceeded limit")
                    
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
        lock_path = file_path.with_suffix('.lock')
        
        try:
            with FileLock(lock_path):
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
            if lock_path.exists():
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
    
    def get_statistics(self, file_path: Path) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each band in a dataset"""
        stats = {}
        with self.safe_open(file_path) as src:
            for band_idx in range(1, src.count + 1):
                # Process in chunks to manage memory
                chunk_stats = []
                for chunk in self.process_in_chunks(file_path, chunk_size=1024):
                    chunk_stats.append({
                        "min": float(np.min(chunk)),
                        "max": float(np.max(chunk)),
                        "mean": float(np.mean(chunk)),
                        "std": float(np.std(chunk)),
                        "nodata_count": int(np.sum(chunk == src.nodata)) if src.nodata is not None else 0
                    })
                
                # Combine chunk statistics
                band_stats = {
                    "min": min(s["min"] for s in chunk_stats),
                    "max": max(s["max"] for s in chunk_stats),
                    "mean": np.mean([s["mean"] for s in chunk_stats]),
                    "std": np.sqrt(np.mean([s["std"]**2 for s in chunk_stats])),
                    "nodata_count": sum(s["nodata_count"] for s in chunk_stats)
                }
                
                # Calculate percentiles using chunks
                all_data = []
                for chunk in self.process_in_chunks(file_path, chunk_size=1024):
                    all_data.extend(chunk.flatten())
                percentiles = np.percentile(all_data, [25, 50, 75])
                
                band_stats.update({
                    "p25": float(percentiles[0]),
                    "median": float(percentiles[1]),
                    "p75": float(percentiles[2])
                })
                
                stats[f"band_{band_idx}"] = band_stats
        
        return stats
    
    def get_operation_metrics(self) -> Dict[str, Any]:
        """Get operation metrics"""
        return {
            "operation_times": self.metrics.operation_times,
            "error_counts": self.metrics.error_counts,
            "memory_usage": self.metrics.memory_usage
        } 