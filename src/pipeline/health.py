"""Health check module for TIFF enrichment pipeline."""

import sys
import os
import json
from typing import Dict, Any

def check_database_connection() -> Dict[str, Any]:
    """Check PostgreSQL database connection."""
    try:
        import psycopg2
        db_url = os.getenv('DATABASE_URL', 'postgresql://pipeline:password@postgres:5432/tiff_pipeline')
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute('SELECT 1')
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return {
            'status': 'healthy',
            'message': 'Database connection successful',
            'details': {'result': result[0] if result else None}
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'Database connection failed: {str(e)}',
            'details': {'error': str(e)}
        }

def check_redis_connection() -> Dict[str, Any]:
    """Check Redis cache connection."""
    try:
        import redis
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
        r = redis.from_url(redis_url)
        r.ping()
        
        return {
            'status': 'healthy',
            'message': 'Redis connection successful',
            'details': {'ping': 'pong'}
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'Redis connection failed: {str(e)}',
            'details': {'error': str(e)}
        }

def check_file_system() -> Dict[str, Any]:
    """Check file system access."""
    try:
        input_dir = '/data/input'
        output_dir = '/data/output'
        
        input_readable = os.access(input_dir, os.R_OK) if os.path.exists(input_dir) else False
        output_writable = os.access(output_dir, os.W_OK) if os.path.exists(output_dir) else False
        
        if input_readable and output_writable:
            return {
                'status': 'healthy',
                'message': 'File system access successful',
                'details': {
                    'input_readable': input_readable,
                    'output_writable': output_writable
                }
            }
        else:
            return {
                'status': 'unhealthy',
                'message': 'File system access issues',
                'details': {
                    'input_readable': input_readable,
                    'output_writable': output_writable
                }
            }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'message': f'File system check failed: {str(e)}',
            'details': {'error': str(e)}
        }

def health_check() -> bool:
    """Main health check function."""
    checks = {
        'database': check_database_connection(),
        'redis': check_redis_connection(),
        'filesystem': check_file_system()
    }
    
    all_healthy = all(check['status'] == 'healthy' for check in checks.values())
    
    print(f"Health Check Results: {'HEALTHY' if all_healthy else 'UNHEALTHY'}")
    for name, result in checks.items():
        status = result['status'].upper()
        print(f"  {name}: {status} - {result['message']}")
    
    return all_healthy

if __name__ == '__main__':
    is_healthy = health_check()
    sys.exit(0 if is_healthy else 1)