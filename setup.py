from setuptools import setup, find_packages

setup(
    name="tiff-enrichment-pipeline",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "rasterio>=1.3.8",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "geopandas>=0.13.0",
        "shapely>=2.0.0",
        "fiona>=1.9.0",
        "pyproj>=3.6.0",
        "psutil>=5.9.0",
        "prometheus-client>=0.17.0",
        "python-json-logger>=2.0.0",
        "tqdm>=4.65.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
    },
    python_requires=">=3.9",
) 