# This Dockerfile is used for both development and production builds.
# Multi-stage build for TIFF enrichment pipeline
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.11.0 as base

# Install system dependencies including Python geospatial packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    libgdal-dev \
    libspatialindex-dev \
    libproj-dev \
    libgeos-dev \
    curl \
    postgresql-client \
    # Additional system packages for geospatial libraries
    python3-gdal \
    python3-numpy \
    python3-pandas \
    python3-shapely \
    python3-fiona \
    python3-rasterio \
    python3-geopandas \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Create application user (use different UID to avoid conflicts)
RUN groupadd -r pipeline && \
    useradd -r -g pipeline -u 1001 pipeline && \
    mkdir -p /app /data/input /data/output /data/failed && \
    chown -R pipeline:pipeline /app /data

# Set working directory
WORKDIR /app

# Create and activate virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in virtual environment
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    --find-links /usr/lib/python3/dist-packages \
    --prefer-binary \
    -r requirements.txt

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