#!/bin/bash
# Docker Entrypoint for Complete Experiment Pipeline
#
# This script runs the full experiment pipeline inside Docker:
# 1. Generate embeddings for all models (with 1K limit for mock)
# 2. Run benchmark across all 4 datasets
# 3. Generate analysis plots and tables
#
# Usage: docker run --rm --cpus=6.0 --memory=12g -v $(pwd)/results:/app/results vector-bench

set -e

echo "=============================================="
echo "VECTOR RETRIEVAL BENCHMARK EXPERIMENT"
echo "=============================================="
echo "Start time: $(date)"
echo "CPU limit: $(nproc) cores"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Configuration
LIMIT=${LIMIT:-1000}  # Default to 1000 for mock, set LIMIT=0 for full
ALPHAS=${ALPHAS:-"0.25,0.5,0.75"}
DATASETS=("dbpedia-entity" "scifact" "fiqa" "arguana")
MODELS=("minilm" "splade" "bm25" "word2vec" "bge-m3" "bge-m3-all")

echo "Configuration:"
echo "  Document limit: $LIMIT (0 = full dataset)"
echo "  Alpha values: $ALPHAS"
echo "  Datasets: ${DATASETS[*]}"
echo "  Models: ${MODELS[*]}"
echo ""

# Step 1: Download datasets
echo "=============================================="
echo "Step 1/3: Downloading datasets..."
echo "=============================================="
for dataset in "${DATASETS[@]}"; do
    echo ">>> $dataset"
    python3 -m src.vector_experiments.download_datasets --dataset "$dataset" --output_dir data/raw
done

# Step 2: Generate embeddings for each dataset
echo ""
echo "=============================================="
echo "Step 2/3: Generating embeddings..."
echo "=============================================="

for dataset in "${DATASETS[@]}"; do
    echo ">>> Dataset: $dataset"
    OUTPUT_DIR="data/embeddings/${dataset}"
    mkdir -p "$OUTPUT_DIR"
    
    for model in "${MODELS[@]}"; do
        echo "  - Model: $model"
        
        LIMIT_ARG=""
        if [ "$LIMIT" -gt 0 ]; then
            LIMIT_ARG="--limit $LIMIT"
        fi
        
        # Note: generate_embeddings.py needs to support --dataset
        # For now, we use a simplified approach via benchmark.py which handles this
        python3 -c "
from src.vector_experiments.benchmark import load_dataset, filter_dataset
from src.vector_experiments.models import MiniLMEmbedder, SpladeEmbedder, BGEM3Embedder, BM25Embedder, Word2VecEmbedder
import pickle
import os

dataset = '$dataset'
model = '$model'
limit = $LIMIT
output_dir = '$OUTPUT_DIR'

print(f'Loading {dataset}...')
corpus, _, _ = load_dataset(dataset)
if limit > 0:
    corpus = {k: corpus[k] for k in list(corpus.keys())[:limit]}

print(f'Encoding with {model}...')
if model == 'minilm':
    embedder = MiniLMEmbedder()
elif model == 'splade':
    embedder = SpladeEmbedder()
elif model == 'bge-m3':
    embedder = BGEM3Embedder(return_all=False)
elif model == 'bge-m3-all':
    embedder = BGEM3Embedder(return_all=True)
elif model == 'bm25':
    embedder = BM25Embedder()
elif model == 'word2vec':
    embedder = Word2VecEmbedder()
else:
    raise ValueError(f'Unknown model: {model}')

embeddings, artifact = embedder.encode(corpus)

output_path = os.path.join(output_dir, f'{model}_embeddings.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(embeddings, f)
print(f'Saved to {output_path}')

if artifact:
    artifact_path = os.path.join(output_dir, f'{model}_artifact.pkl')
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifact, f)
"
    done
done

# Step 3: Run benchmark
echo ""
echo "=============================================="
echo "Step 3/3: Running benchmark..."
echo "=============================================="

LIMIT_ARG=""
if [ "$LIMIT" -gt 0 ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

# Run benchmark for each dataset
for dataset in "${DATASETS[@]}"; do
    echo ">>> Benchmarking: $dataset"
    python3 -m src.vector_experiments.benchmark \
        --dataset "$dataset" \
        $LIMIT_ARG \
        --alphas "$ALPHAS" \
        --embeddings-dir "data/embeddings/${dataset}" \
        --output-dir results
done

# Step 4: Generate analysis
echo ""
echo "=============================================="
echo "Step 4/4: Generating analysis..."
echo "=============================================="

# Find all result files
RESULT_FILES=$(find results -name "benchmark_*.json" -type f | tr '\n' ' ')

if [ -n "$RESULT_FILES" ]; then
    python3 -m src.vector_experiments.analyze_results --input $RESULT_FILES --output results
else
    echo "Warning: No result files found"
fi

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - results/benchmark_*.json (raw data)"
echo "  - results/benchmark_*.csv (tabular data)"
echo "  - results/tables/ (LaTeX & Markdown tables)"
echo "  - results/figures/ (PNG plots)"
echo "  - results/summary_report.txt"
