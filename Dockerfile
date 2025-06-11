# This Dockerfile is used for both development and production builds.
# Multi-stage build for TIFF enrichment pipeline
FROM osgeo/gdal:ubuntu-small-3.6.3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GDAL_VERSION=3.6.3 \
    RASTERIO_VERSION=1.3.8 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user
RUN useradd -m -s /bin/bash pipeline && \
    mkdir -p /app && \
    chown -R pipeline:pipeline /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libgrib2c-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libwebp-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libspatialite-dev \
    libsqlite3-dev \
    libpq-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libffi-dev \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy application code
WORKDIR /app
COPY --chown=pipeline:pipeline . .

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Set Python path
ENV PYTHONPATH=/app

# Switch to non-root user
USER pipeline

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import rasterio; print('Rasterio version:', rasterio.__version__)" || exit 1

# Default command
CMD ["python3", "-m", "pipeline.main"]