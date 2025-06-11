#!/bin/bash

# Exit on error
set -e

# Build and run tests
echo "Building and running MicaSense processor tests..."
docker-compose -f docker-compose.dev.yml up --build --abort-on-container-exit

# Check test results
if [ $? -eq 0 ]; then
    echo "Tests completed successfully!"
else
    echo "Tests failed!"
    exit 1
fi 