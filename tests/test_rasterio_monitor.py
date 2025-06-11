"""
Tests for Rasterio Monitoring
"""

import unittest
import time
import tempfile
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from pipeline.monitoring.rasterio_monitor import RasterioMonitor, RasterioOperationMetrics

class TestRasterioMonitor(unittest.TestCase):
    """Test cases for RasterioMonitor"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.monitor = RasterioMonitor(metrics_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_operation_metrics(self):
        """Test operation metrics recording"""
        # Start an operation
        op_id = self.monitor.start_operation('test_op', file_path='test.tif')
        
        # Verify current operations
        current_ops = self.monitor.get_current_operations()
        self.assertIn(op_id, current_ops)
        self.assertEqual(current_ops[op_id].operation_type, 'test_op')
        
        # End operation
        self.monitor.end_operation(op_id)
        
        # Verify operation history
        history = self.monitor.get_operation_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].operation_type, 'test_op')
        self.assertIsNotNone(history[0].end_time)
        self.assertIsNotNone(history[0].memory_usage_end)
    
    def test_error_handling(self):
        """Test error handling in operations"""
        # Start operation with error
        op_id = self.monitor.start_operation('error_op')
        self.monitor.end_operation(op_id, error='Test error')
        
        # Verify error recording
        history = self.monitor.get_operation_history()
        self.assertEqual(history[0].error, 'Test error')
        
        # Verify error in summary
        summary = self.monitor.get_operation_summary()
        self.assertEqual(summary['operation_types']['error_op']['errors'], 1)
        self.assertEqual(len(summary['errors']), 1)
        self.assertEqual(summary['errors'][0]['error'], 'Test error')
    
    def test_concurrent_operations(self):
        """Test concurrent operation handling"""
        def run_operation(op_type: str):
            op_id = self.monitor.start_operation(op_type)
            time.sleep(0.1)  # Simulate work
            self.monitor.end_operation(op_id)
        
        # Run multiple operations concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_operation, f'op_{i}')
                for i in range(4)
            ]
            for future in futures:
                future.result()
        
        # Verify all operations completed
        history = self.monitor.get_operation_history()
        self.assertEqual(len(history), 4)
        
        # Verify operation types
        op_types = {m.operation_type for m in history}
        self.assertEqual(op_types, {'op_0', 'op_1', 'op_2', 'op_3'})
    
    def test_metrics_saving(self):
        """Test metrics saving to file"""
        # Create some operations
        for i in range(3):
            op_id = self.monitor.start_operation(f'test_op_{i}')
            time.sleep(0.1)
            self.monitor.end_operation(op_id)
        
        # Save metrics
        self.monitor.save_metrics()
        
        # Verify metrics file
        metrics_files = list(self.test_dir.glob('rasterio_metrics_*.json'))
        self.assertEqual(len(metrics_files), 1)
        
        # Verify metrics content
        with open(metrics_files[0]) as f:
            metrics_data = json.load(f)
        
        self.assertIn('timestamp', metrics_data)
        self.assertIn('summary', metrics_data)
        self.assertIn('operations', metrics_data)
        self.assertEqual(len(metrics_data['operations']), 3)
    
    def test_memory_tracking(self):
        """Test memory usage tracking"""
        # Start operation
        op_id = self.monitor.start_operation('memory_test')
        initial_memory = self.monitor.get_current_operations()[op_id].memory_usage_start
        
        # Simulate memory usage
        large_list = [0] * 1000000  # Allocate some memory
        
        # End operation
        self.monitor.end_operation(op_id)
        
        # Verify memory tracking
        history = self.monitor.get_operation_history()
        self.assertGreater(history[0].memory_delta, 0)
        self.assertGreater(history[0].memory_usage_end, initial_memory)
    
    def test_operation_summary(self):
        """Test operation summary generation"""
        # Create operations of different types
        for op_type in ['read', 'write', 'process']:
            for _ in range(2):
                op_id = self.monitor.start_operation(op_type)
                time.sleep(0.1)
                self.monitor.end_operation(op_id)
        
        # Add an error
        op_id = self.monitor.start_operation('read')
        self.monitor.end_operation(op_id, error='Test error')
        
        # Get summary
        summary = self.monitor.get_operation_summary()
        
        # Verify summary structure
        self.assertEqual(summary['total_operations'], 7)
        self.assertEqual(len(summary['operation_types']), 3)
        self.assertEqual(len(summary['errors']), 1)
        
        # Verify operation type summaries
        for op_type in ['read', 'write', 'process']:
            op_summary = summary['operation_types'][op_type]
            self.assertEqual(op_summary['count'], 3 if op_type == 'read' else 2)
            self.assertGreater(op_summary['total_duration'], 0)
            self.assertGreater(op_summary['avg_duration'], 0)
    
    def test_clear_history(self):
        """Test history clearing"""
        # Create some operations
        for i in range(3):
            op_id = self.monitor.start_operation(f'test_op_{i}')
            self.monitor.end_operation(op_id)
        
        # Clear history
        self.monitor.clear_history()
        
        # Verify history is cleared
        self.assertEqual(len(self.monitor.get_operation_history()), 0)
        self.assertEqual(len(self.monitor.get_current_operations()), 0)
        
        # Verify new operations work after clear
        op_id = self.monitor.start_operation('new_op')
        self.monitor.end_operation(op_id)
        self.assertEqual(len(self.monitor.get_operation_history()), 1) 