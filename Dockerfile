# Build stage
FROM osgeo/gdal:ubuntu-small-3.6.3 AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GDAL_VERSION=3.6.3 \
    RASTERIO_VERSION=1.3.8 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-distutils \
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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set up Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with caching
COPY requirements/base.txt requirements/dev.txt requirements/prod.txt /tmp/requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /tmp/requirements/base.txt && \
    if [ "$ENVIRONMENT" = "development" ]; then \
        pip install --no-cache-dir -r /tmp/requirements/dev.txt; \
    elif [ "$ENVIRONMENT" = "production" ]; then \
        pip install --no-cache-dir -r /tmp/requirements/prod.txt; \
    fi && \
    rm -rf /tmp/requirements

# Final stage
FROM osgeo/gdal:ubuntu-small-3.6.3

# Copy only runtime dependencies
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user and directories
RUN useradd -m -s /bin/bash pipeline && \
    mkdir -p /data/input /data/output /app/logs /tmp/rasterio_locks && \
    chown -R pipeline:pipeline /data /app/logs /tmp/rasterio_locks && \
    chmod -R 755 /data /app/logs /tmp/rasterio_locks && \
    # Install required packages
    apt-get update && apt-get install -y --no-install-recommends \
    python3-distutils \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean && \
    rm -rf /tmp/* && \
    rm -rf /var/tmp/*

# Copy application code
WORKDIR /app
COPY --chown=pipeline:pipeline . .

# Install the package in development mode
RUN pip install --no-cache-dir -e ".[dev]"

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Set Python path
ENV PYTHONPATH=/app/src

# Switch to non-root user
USER pipeline

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import rasterio; print('Rasterio version:', rasterio.__version__)" || exit 1

# Default command
CMD ["python3", "-m", "pipeline.main"]