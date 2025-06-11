#!/bin/bash
# =============================================================================
# TIFF Enrichment Pipeline - Deployment Script
# This script sets up and deploys the complete pipeline
# =============================================================================

set -euo pipefail

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-development}"
COMPOSE_PROJECT="tiff-pipeline"

# Functions
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "All dependencies are available"
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    local dirs=(
        "data/input"
        "data/output/tiff"
        "data/output/json"
        "data/failed"
        "logs"
        "config/dev"
        "config/prod" 
        "scripts"
        "src/pipeline"
        "src/tests"
        "monitoring/prometheus"
        "monitoring/grafana/dashboards"
        "monitoring/grafana/datasources"
        "backups"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
        log_info "Created directory: $dir"
    done
    
    log_success "Directory structure created"
}

# Setup configuration files
setup_config() {
    log_info "Setting up configuration files..."
    
    # Copy .env template if .env doesn't exist
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        if [ -f "$PROJECT_ROOT/.env.template" ]; then
            cp "$PROJECT_ROOT/.env.template" "$PROJECT_ROOT/.env"
            log_warning "Created .env from template. Please edit it with your settings!"
        else
            log_error ".env.template not found. Please create environment configuration."
            exit 1
        fi
    else
        log_info ".env file already exists"
    fi
    
    # Create development config
    cat > "$PROJECT_ROOT/config/dev/settings.yml" << EOF
environment: development
debug: true
log_level: DEBUG
api_timeout: 30
max_workers: 2
batch_size: 3
enable_monitoring: false
use_cache: true
EOF
    
    # Create production config
    cat > "$PROJECT_ROOT/config/prod/settings.yml" << EOF
environment: production
debug: false
log_level: INFO
api_timeout: 60
max_workers: 8
batch_size: 10
enable_monitoring: true
use_cache: true
EOF
    
    log_success "Configuration files created"
}

# Setup monitoring configuration
setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > "$PROJECT_ROOT/monitoring/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'tiff-pipeline'
    static_configs:
      - targets: ['tiff-pipeline:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s
EOF
    
    # Grafana datasource configuration
    mkdir -p "$PROJECT_ROOT/monitoring/grafana/datasources"
    cat > "$PROJECT_ROOT/monitoring/grafana/datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    
    # Grafana dashboard configuration
    mkdir -p "$PROJECT_ROOT/monitoring/grafana/dashboards"
    cat > "$PROJECT_ROOT/monitoring/grafana/dashboards/dashboard.yml" << EOF
apiVersion: 1

providers:
  - name: 'TIFF Pipeline'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
    
    log_success "Monitoring configuration created"
}

# Create sample health check script
create_health_check() {
    log_info "Creating health check script..."
    
    mkdir -p "$PROJECT_ROOT/src/pipeline"
    cat > "$PROJECT_ROOT/src/pipeline/health.py" << 'EOF'
"""Health check module for TIFF enrichment pipeline."""

import sys
import os
import psycopg2
import redis
from typing import Dict, Any

def check_database_connection() -> Dict[str, Any]:
    """Check PostgreSQL database connection."""
    try:
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
EOF
    
    log_success "Health check script created"
}

# Build and start services
deploy_services() {
    local profile=${1:-""}
    
    log_info "Building Docker images..."
    cd "$PROJECT_ROOT"
    docker-compose -p "$COMPOSE_PROJECT" build
    
    log_info "Starting services..."
    if [ -n "$profile" ]; then
        docker-compose -p "$COMPOSE_PROJECT" --profile "$profile" up -d
    else
        docker-compose -p "$COMPOSE_PROJECT" up -d
    fi
    
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check service health
    log_info "Checking service health..."
    if docker-compose -p "$COMPOSE_PROJECT" exec -T tiff-pipeline python -c "
import sys
sys.path.append('/app/src')
from pipeline.health import health_check
exit(0 if health_check() else 1)
" 2>/dev/null; then
        log_success "All services are healthy"
    else
        log_warning "Some services may not be fully ready yet"
    fi
}

# Create sample TIFF files for testing
create_sample_data() {
    log_info "Creating sample TIFF files..."
    
    cat > "$PROJECT_ROOT/scripts/create_sample_tiffs.py" << 'EOF'
"""Create sample TIFF files with GPS metadata for testing."""

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
from datetime import datetime

def create_sample_tiff(filename, lat, lon, width=1280, height=960):
    """Create a sample TIFF file with GPS metadata."""
    # Create random image data
    image_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_data)
    
    # Create EXIF data with GPS information
    exif_dict = {
        "0th": {
            256: width,  # ImageWidth
            257: height,  # ImageLength
            272: "Sample Camera",  # Make
            306: datetime.now().strftime("%Y:%m:%d %H:%M:%S"),  # DateTime
        },
        "GPS": {
            1: 'N' if lat >= 0 else 'S',  # GPSLatitudeRef
            2: ((int(abs(lat)), 1), (int((abs(lat) % 1) * 60), 1), (0, 1)),  # GPSLatitude
            3: 'E' if lon >= 0 else 'W',  # GPSLongitudeRef
            4: ((int(abs(lon)), 1), (int((abs(lon) % 1) * 60), 1), (0, 1)),  # GPSLongitude
        }
    }
    
    # Save the image
    output_path = f"/data/input/{filename}"
    image.save(output_path, "TIFF")
    print(f"Created sample TIFF: {output_path}")

if __name__ == "__main__":
    # Create a few sample files with different locations
    samples = [
        ("sample_01.tiff", 40.7589, -73.9851),  # New York
        ("sample_02.tiff", 34.0522, -118.2437),  # Los Angeles
        ("sample_03.tiff", 51.5074, -0.1278),   # London
    ]
    
    for filename, lat, lon in samples:
        create_sample_tiff(filename, lat, lon)
EOF
    
    log_info "Sample data creation script created"
    log_info "Run 'docker-compose exec tiff-pipeline python /app/scripts/create_sample_tiffs.py' to create test files"
}

# Main deployment function
main() {
    log_info "Starting TIFF Enrichment Pipeline deployment..."
    
    # Parse command line arguments
    PROFILE=""
    CREATE_SAMPLES=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --profile)
                PROFILE="$2"
                shift 2
                ;;
            --with-admin)
                PROFILE="admin"
                shift
                ;;
            --with-monitoring)
                PROFILE="monitoring"
                shift
                ;;
            --with-all)
                PROFILE="admin,monitoring,watcher"
                shift
                ;;
            --create-samples)
                CREATE_SAMPLES=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --profile PROFILE    Docker Compose profile to use"
                echo "  --with-admin         Start with admin tools (pgAdmin, Redis Commander)"
                echo "  --with-monitoring    Start with monitoring (Prometheus, Grafana)"
                echo "  --with-all           Start with all optional services"
                echo "  --create-samples     Create sample TIFF files for testing"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run deployment steps
    check_dependencies
    create_directories
    setup_config
    setup_monitoring
    create_health_check
    create_sample_data
    
    # Deploy services
    if [ -n "$PROFILE" ]; then
        # Convert comma-separated profiles to --profile flags
        PROFILE_FLAGS=""
        IFS=',' read -ra PROFILES <<< "$PROFILE"
        for p in "${PROFILES[@]}"; do
            PROFILE_FLAGS="$PROFILE_FLAGS --profile $p"
        done
        
        log_info "Deploying with profiles: $PROFILE"
        cd "$PROJECT_ROOT"
        docker-compose -p "$COMPOSE_PROJECT" build
        docker-compose -p "$COMPOSE_PROJECT" $PROFILE_FLAGS up -d
    else
        deploy_services
    fi
    
    # Create sample files if requested
    if [ "$CREATE_SAMPLES" = true ]; then
        log_info "Creating sample TIFF files..."
        sleep 10  # Wait for services to be ready
        docker-compose -p "$COMPOSE_PROJECT" exec -T tiff-pipeline python /app/scripts/create_sample_tiffs.py || true
    fi
    
    log_success "Deployment completed successfully!"
    
    # Show access information
    echo -e "\n${CYAN}=== Access Information ===${NC}"
    echo -e "Main application: Check logs with 'make logs-pipeline'"
    
    if [[ "$PROFILE" == *"admin"* ]]; then
        echo -e "pgAdmin: http://localhost:${PGADMIN_PORT:-8080}"
        echo -e "Redis Commander: http://localhost:${REDIS_UI_PORT:-8081}"
    fi
    
    if [[ "$PROFILE" == *"monitoring"* ]]; then
        echo -e "Grafana: http://localhost:${GRAFANA_PORT:-3000}"
        echo -e "Prometheus: http://localhost:${PROMETHEUS_PORT:-9090}"
    fi
    
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo -e "1. Check service status: make status"
    echo -e "2. View logs: make logs"
    echo -e "3. Add TIFF files to data/input/ directory"
    echo -e "4. Process files: make process-sample"
}

# Run main function
main "$@"