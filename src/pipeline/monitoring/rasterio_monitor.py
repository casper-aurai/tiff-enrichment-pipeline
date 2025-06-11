"""
Monitoring module for rasterio operations
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import psutil
import threading
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RasterioOperationMetrics:
    """Metrics for a single rasterio operation"""
    operation_type: str
    start_time: float
    end_time: Optional[float] = None
    memory_usage_start: float = field(default_factory=lambda: psutil.Process().memory_info().rss / 1024 / 1024)
    memory_usage_end: Optional[float] = None
    error: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    operation_details: Dict = field(default_factory=dict)

    def complete(self, error: Optional[str] = None) -> None:
        """Complete the operation and record final metrics"""
        self.end_time = time.time()
        self.memory_usage_end = psutil.Process().memory_info().rss / 1024 / 1024
        self.error = error

    @property
    def duration(self) -> float:
        """Get operation duration in seconds"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def memory_delta(self) -> float:
        """Get memory usage delta in MB"""
        if self.memory_usage_end is None:
            return psutil.Process().memory_info().rss / 1024 / 1024 - self.memory_usage_start
        return self.memory_usage_end - self.memory_usage_start

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'operation_type': self.operation_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_usage_start': self.memory_usage_start,
            'memory_usage_end': self.memory_usage_end,
            'memory_delta': self.memory_delta,
            'error': self.error,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'operation_details': self.operation_details
        }

class RasterioMonitor:
    """Monitor for rasterio operations"""
    
    def __init__(self, metrics_dir: Optional[Path] = None):
        """Initialize the monitor"""
        self.metrics_dir = metrics_dir or Path('metrics')
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._current_operations: Dict[str, RasterioOperationMetrics] = {}
        self._operation_history: List[RasterioOperationMetrics] = []
        self._lock = threading.Lock()
        self._start_time = time.time()

    def start_operation(self, operation_type: str, file_path: Optional[str] = None, **details) -> str:
        """Start monitoring an operation"""
        operation_id = f"{operation_type}_{int(time.time() * 1000)}"
        metrics = RasterioOperationMetrics(
            operation_type=operation_type,
            start_time=time.time(),
            file_path=file_path,
            operation_details=details
        )
        
        if file_path:
            try:
                metrics.file_size = Path(file_path).stat().st_size
            except (OSError, FileNotFoundError):
                pass

        with self._lock:
            self._current_operations[operation_id] = metrics
        
        logger.debug(f"Started operation {operation_id}: {operation_type}")
        return operation_id

    def end_operation(self, operation_id: str, error: Optional[str] = None) -> None:
        """End monitoring an operation"""
        with self._lock:
            if operation_id in self._current_operations:
                metrics = self._current_operations[operation_id]
                metrics.complete(error)
                self._operation_history.append(metrics)
                del self._current_operations[operation_id]
                
                if error:
                    logger.error(f"Operation {operation_id} failed: {error}")
                else:
                    logger.debug(f"Completed operation {operation_id}")

    def get_current_operations(self) -> Dict[str, RasterioOperationMetrics]:
        """Get currently running operations"""
        with self._lock:
            return self._current_operations.copy()

    def get_operation_history(self) -> List[RasterioOperationMetrics]:
        """Get operation history"""
        with self._lock:
            return self._operation_history.copy()

    def get_operation_summary(self) -> Dict:
        """Get summary of all operations"""
        with self._lock:
            summary = {
                'total_operations': len(self._operation_history),
                'current_operations': len(self._current_operations),
                'total_duration': time.time() - self._start_time,
                'operation_types': {},
                'errors': [],
                'memory_usage': {
                    'current': psutil.Process().memory_info().rss / 1024 / 1024,
                    'peak': max((m.memory_usage_end or m.memory_usage_start) 
                              for m in self._operation_history) if self._operation_history else 0
                }
            }

            # Aggregate by operation type
            for metrics in self._operation_history:
                op_type = metrics.operation_type
                if op_type not in summary['operation_types']:
                    summary['operation_types'][op_type] = {
                        'count': 0,
                        'total_duration': 0,
                        'avg_duration': 0,
                        'total_memory_delta': 0,
                        'errors': 0
                    }
                
                op_summary = summary['operation_types'][op_type]
                op_summary['count'] += 1
                op_summary['total_duration'] += metrics.duration
                op_summary['total_memory_delta'] += metrics.memory_delta
                if metrics.error:
                    op_summary['errors'] += 1
                    summary['errors'].append({
                        'operation_type': op_type,
                        'error': metrics.error,
                        'file_path': metrics.file_path,
                        'time': datetime.fromtimestamp(metrics.start_time).isoformat()
                    })

            # Calculate averages
            for op_summary in summary['operation_types'].values():
                if op_summary['count'] > 0:
                    op_summary['avg_duration'] = op_summary['total_duration'] / op_summary['count']

            return summary

    def save_metrics(self) -> None:
        """Save metrics to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = self.metrics_dir / f'rasterio_metrics_{timestamp}.json'
        
        with self._lock:
            metrics_data = {
                'timestamp': timestamp,
                'summary': self.get_operation_summary(),
                'operations': [m.to_dict() for m in self._operation_history]
            }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_file}")

    def clear_history(self) -> None:
        """Clear operation history"""
        with self._lock:
            self._operation_history.clear()
            self._start_time = time.time()

# Global monitor instance
monitor = RasterioMonitor()

def get_monitor() -> RasterioMonitor:
    """Get the global monitor instance"""
    return monitor 