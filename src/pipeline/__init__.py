"""TIFF Enrichment Pipeline

A geospatial data processing pipeline for enriching TIFF files with external data sources.
"""

__version__ = "1.0.0"
__author__ = "TIFF Pipeline Team"
__email__ = "team@example.com"

from .health import health_check

__all__ = ["health_check"]