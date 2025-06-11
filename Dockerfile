# This Dockerfile is used for both development and production builds.
# Multi-stage build for TIFF enrichment pipeline
FROM osgeo/gdal:ubuntu-small-3.6.3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GDAL_VERSION=3.6.3

# Create non-root user
RUN useradd -m -s /bin/bash pipeline

# Set up working directory
WORKDIR /app

# Install Python virtual environment package and build dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-venv \
    python3-dev \
    build-essential \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /app/venv && \
    chown -R pipeline:pipeline /app/venv

# Copy requirements first for better caching
COPY --chown=pipeline:pipeline requirements.txt .

# Install Python dependencies in virtual environment
RUN . /app/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --find-links /usr/lib/python3/dist-packages \
    --prefer-binary \
    -r requirements.txt

# Copy application code
COPY --chown=pipeline:pipeline src/ ./src/
COPY --chown=pipeline:pipeline config/ ./config/
COPY --chown=pipeline:pipeline tests/ ./tests/
COPY --chown=pipeline:pipeline scripts/ ./scripts/

# Make scripts executable
RUN find scripts/ -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

# Set Python path
ENV PYTHONPATH=/app/src

# Switch to non-root user
USER pipeline

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import sys; sys.path.append('/app/src'); from pipeline.health import health_check; health_check()" || exit 1

# Default command
CMD ["python", "-m", "pipeline.main"]