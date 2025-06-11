import time
import numpy as np
from pathlib import Path
import tempfile
import shutil
from src.pipeline.micasense.processing.indices import VegetationIndices
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class BenchmarkVegetationIndices:
    @classmethod
    def setup_class(cls):
        """Set up benchmark environment"""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.output_dir = cls.test_dir / "output"
        cls.output_dir.mkdir()
        
        # Test configuration
        cls.config = {
            'vegetation_indices': {
                'ndvi': True,
                'ndre': True,
                'gndvi': True,
                'savi': True,
                'msavi': True,
                'evi': True
            }
        }
        
        # Create large test data (2048x2048 pixels)
        size = 2048
        cls.nir = np.random.uniform(0, 1, (size, size)).astype('float32')
        cls.red = np.random.uniform(0, 1, (size, size)).astype('float32')
        cls.green = np.random.uniform(0, 1, (size, size)).astype('float32')
        cls.red_edge = np.random.uniform(0, 1, (size, size)).astype('float32')
        cls.blue = np.random.uniform(0, 1, (size, size)).astype('float32')
        
        # Initialize processor
        cls.processor = VegetationIndices(cls.config, cls.output_dir)
    
    def benchmark_ndvi(self):
        """Benchmark NDVI calculation"""
        start_time = time.time()
        start_memory = get_memory_usage()
        
        ndvi = self.processor._calculate_ndvi(self.nir, self.red)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        return {
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'pixels_per_second': (self.nir.size) / (end_time - start_time)
        }
    
    def benchmark_ndre(self):
        """Benchmark NDRE calculation"""
        start_time = time.time()
        start_memory = get_memory_usage()
        
        ndre = self.processor._calculate_ndre(self.nir, self.red_edge)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        return {
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'pixels_per_second': (self.nir.size) / (end_time - start_time)
        }
    
    def benchmark_gndvi(self):
        """Benchmark GNDVI calculation"""
        start_time = time.time()
        start_memory = get_memory_usage()
        
        gndvi = self.processor._calculate_gndvi(self.nir, self.green)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        return {
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'pixels_per_second': (self.nir.size) / (end_time - start_time)
        }
    
    def benchmark_all_indices(self):
        """Benchmark calculation of all indices"""
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # Calculate all indices
        ndvi = self.processor._calculate_ndvi(self.nir, self.red)
        ndre = self.processor._calculate_ndre(self.nir, self.red_edge)
        gndvi = self.processor._calculate_gndvi(self.nir, self.green)
        savi = self.processor._calculate_savi(self.nir, self.red)
        msavi = self.processor._calculate_msavi(self.nir, self.red)
        evi = self.processor._calculate_evi(self.nir, self.red, self.blue)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        return {
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'pixels_per_second': (self.nir.size * 6) / (end_time - start_time)
        }
    
    @classmethod
    def teardown_class(cls):
        """Clean up benchmark environment"""
        shutil.rmtree(cls.test_dir)

def run_benchmarks():
    """Run all benchmarks and print results"""
    benchmark = BenchmarkVegetationIndices()
    benchmark.setup_class()
    
    print("\nRunning Vegetation Indices Benchmarks")
    print("=====================================")
    
    # Run individual index benchmarks
    for index_name in ['ndvi', 'ndre', 'gndvi']:
        results = getattr(benchmark, f'benchmark_{index_name}')()
        print(f"\n{index_name.upper()} Benchmark:")
        print(f"Time: {results['time']:.3f} seconds")
        print(f"Memory: {results['memory']:.1f} MB")
        print(f"Processing speed: {results['pixels_per_second']:.0f} pixels/second")
    
    # Run combined benchmark
    results = benchmark.benchmark_all_indices()
    print("\nAll Indices Benchmark:")
    print(f"Time: {results['time']:.3f} seconds")
    print(f"Memory: {results['memory']:.1f} MB")
    print(f"Processing speed: {results['pixels_per_second']:.0f} pixels/second")
    
    benchmark.teardown_class()

if __name__ == '__main__':
    run_benchmarks() 