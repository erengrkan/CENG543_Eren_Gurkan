# Dockerfile for Complete Experiment Pipeline
#
# This container runs the full benchmark pipeline in an isolated environment:
# 1. Downloads all 4 BEIR datasets
# 2. Generates embeddings for all models
# 3. Runs benchmark across all datasets
# 4. Generates analysis plots and tables
#
# Build: docker build -t vector-bench .
# Run:   docker run --rm --cpus=6.0 --memory=12g -v $(pwd)/results:/app/results vector-bench
#
# Resource Limits:
# - CPU: 6 cores
# - RAM: 12 GB

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir matplotlib tabulate

# Copy source code
COPY . .

# Create directories
RUN mkdir -p data/raw data/embeddings results

# Set Python path
ENV PYTHONPATH=/app

# Copy and set entrypoint script
COPY scripts/docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
