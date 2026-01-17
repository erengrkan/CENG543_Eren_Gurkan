#!/bin/bash
# A100 GPU Experiment Runner
#
# Run this script on a machine with A100 GPU:
#
# git clone <your-repo>
# cd vector-experiments
# ./scripts/run_a100.sh
#
# Estimated Time: ~30-60 minutes

set -e

echo "======================================================"
echo "A100 GPU EXPERIMENT RUNNER"
echo "======================================================"

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Build Docker image
echo ""
echo "Building Docker image..."
docker build -t vector-bench-gpu -f Dockerfile.gpu .

# Run experiment
echo ""
echo "Starting experiment..."
docker run --rm \
    --gpus all \
    --shm-size=16g \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/data:/app/data \
    vector-bench-gpu

echo ""
echo "======================================================"
echo "EXPERIMENT COMPLETE!"
echo "======================================================"
echo "Results saved to: results/"
