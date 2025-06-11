# Multi-stage build for TIFF enrichment pipeline
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.11.0 as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgdal-dev \
    libspatialindex-dev \
    libproj-dev \
    libgeos-dev \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Create application user
RUN useradd -m -u 1000 pipeline && \
    mkdir -p /app /data/input /data/output /data/failed && \
    chown -R pipeline:pipeline /app /data

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Make scripts executable
RUN find scripts/ -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

# Switch to non-root user
USER pipeline

# Set Python path
ENV PYTHONPATH=/app/src

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import sys; sys.path.append('/app/src'); from pipeline.health import health_check; health_check()" || exit 1

# Default command
CMD ["python3", "-m", "pipeline.main"]