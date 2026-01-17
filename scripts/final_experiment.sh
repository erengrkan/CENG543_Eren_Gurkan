#!/bin/bash
# Final Experiment Entrypoint
#
# Configuration:
# - DBpedia: 100K documents (limited due to size)
# - SciFact, FiQA, ArguAna: Full corpus
# - Models: minilm, splade, bm25, word2vec, bge-m3 (NO bge-m3-all)
#
# Estimated Time:
# - Embedding generation: ~2-3 hours
# - Benchmarking: ~30-60 minutes
# - Total: ~3-4 hours
#
# Usage: ./scripts/final_experiment.sh
#        OR: docker run --rm --cpus=6.0 --memory=12g -v $(pwd)/results:/app/results vector-bench-final

set -e

echo "======================================================"
echo "FINAL VECTOR RETRIEVAL BENCHMARK EXPERIMENT"
echo "======================================================"
echo "Start time: $(date)"
echo ""

# Configuration - Dataset specific limits
declare -A DATASET_LIMITS
DATASET_LIMITS["dbpedia-entity"]=100000  # 100K for large corpus
DATASET_LIMITS["scifact"]=0              # Full (5183 docs)
DATASET_LIMITS["fiqa"]=0                 # Full (57638 docs)  
DATASET_LIMITS["arguana"]=0              # Full (8674 docs)

ALPHAS="0.25,0.5,0.75"
DATASETS=("scifact" "fiqa" "arguana" "dbpedia-entity")
# EXCLUDING bge-m3-all (too slow)
MODELS=("minilm" "splade" "bm25" "word2vec" "bge-m3")

echo "Configuration:"
echo "  Datasets and limits:"
for ds in "${DATASETS[@]}"; do
    limit=${DATASET_LIMITS[$ds]}
    if [ "$limit" -eq 0 ]; then
        echo "    - $ds: FULL CORPUS"
    else
        echo "    - $ds: $limit documents"
    fi
done
echo "  Alpha values: $ALPHAS"
echo "  Models: ${MODELS[*]}"
echo "  NOTE: bge-m3-all EXCLUDED (too slow for production)"
echo ""

# Time estimation
echo "======================================================"
echo "ESTIMATED TIME"
echo "======================================================"
echo "  Embedding generation:"
echo "    - MiniLM: ~10 min/dataset"
echo "    - SPLADE: ~30 min/dataset"
echo "    - BM25: ~2 min/dataset"
echo "    - Word2Vec: ~5 min/dataset"
echo "    - BGE-M3: ~60 min/dataset (especially FiQA/DBpedia)"
echo "  Total embedding: ~3 hours"
echo "  Benchmarking: ~30-60 min"
echo "  Analysis: <5 min"
echo "  ----------------------------------"
echo "  TOTAL ESTIMATED: 3-4 hours"
echo "======================================================"
echo ""

# Step 1: Download datasets
echo "======================================================"
echo "Step 1/4: Downloading datasets..."
echo "======================================================"
for dataset in "${DATASETS[@]}"; do
    echo ">>> $dataset"
    python3 -m src.vector_experiments.download_datasets --dataset "$dataset" --output_dir data/raw 2>/dev/null || true
done

# Step 2: Generate embeddings for each dataset
echo ""
echo "======================================================"
echo "Step 2/4: Generating embeddings..."
echo "======================================================"

STEP_START=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo ">>> Dataset: $dataset"
    OUTPUT_DIR="data/embeddings/${dataset}"
    mkdir -p "$OUTPUT_DIR"
    
    LIMIT=${DATASET_LIMITS[$dataset]}
    
    for model in "${MODELS[@]}"; do
        MODEL_START=$(date +%s)
        echo "  - Model: $model (started at $(date '+%H:%M:%S'))"
        
        python3 -c "
from src.vector_experiments.benchmark import load_dataset
from src.vector_experiments.models import MiniLMEmbedder, SpladeEmbedder, BGEM3Embedder, BM25Embedder, Word2VecEmbedder
import pickle
import os
import time

dataset = '$dataset'
model = '$model'
limit = $LIMIT
output_dir = '$OUTPUT_DIR'

print(f'Loading {dataset}...')
corpus, _, _ = load_dataset(dataset)
total_docs = len(corpus)
if limit > 0:
    corpus = {k: corpus[k] for k in list(corpus.keys())[:limit]}
print(f'Using {len(corpus)} of {total_docs} documents')

print(f'Encoding with {model}...')
start = time.time()

if model == 'minilm':
    embedder = MiniLMEmbedder()
elif model == 'splade':
    embedder = SpladeEmbedder()
elif model == 'bge-m3':
    embedder = BGEM3Embedder(return_all=False)
elif model == 'bm25':
    embedder = BM25Embedder()
elif model == 'word2vec':
    embedder = Word2VecEmbedder()
else:
    raise ValueError(f'Unknown model: {model}')

embeddings, artifact = embedder.encode(corpus)

elapsed = time.time() - start
print(f'Encoding completed in {elapsed/60:.1f} minutes')

output_path = os.path.join(output_dir, f'{model}_embeddings.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(embeddings, f)
print(f'Saved to {output_path}')

if artifact:
    artifact_path = os.path.join(output_dir, f'{model}_artifact.pkl')
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifact, f)
"
        MODEL_END=$(date +%s)
        MODEL_ELAPSED=$((MODEL_END - MODEL_START))
        echo "  - $model completed in $((MODEL_ELAPSED / 60))m $((MODEL_ELAPSED % 60))s"
    done
done

STEP_END=$(date +%s)
STEP_ELAPSED=$((STEP_END - STEP_START))
echo ""
echo "Embedding generation completed in $((STEP_ELAPSED / 3600))h $((STEP_ELAPSED % 3600 / 60))m"

# Step 3: Run benchmark
echo ""
echo "======================================================"
echo "Step 3/4: Running benchmark..."
echo "======================================================"

STEP_START=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    LIMIT=${DATASET_LIMITS[$dataset]}
    
    echo ">>> Benchmarking: $dataset"
    
    LIMIT_ARG=""
    if [ "$LIMIT" -gt 0 ]; then
        LIMIT_ARG="--limit $LIMIT"
    fi
    
    python3 -m src.vector_experiments.benchmark \
        --dataset "$dataset" \
        $LIMIT_ARG \
        --alphas "$ALPHAS" \
        --embeddings-dir "data/embeddings/${dataset}" \
        --output-dir results \
        --exclude-models "bge-m3-all"
done

STEP_END=$(date +%s)
STEP_ELAPSED=$((STEP_END - STEP_START))
echo ""
echo "Benchmarking completed in $((STEP_ELAPSED / 60))m $((STEP_ELAPSED % 60))s"

# Step 4: Generate analysis
echo ""
echo "======================================================"
echo "Step 4/4: Generating analysis..."
echo "======================================================"

RESULT_FILES=$(find results -name "benchmark_*.json" -type f | tr '\n' ' ')

if [ -n "$RESULT_FILES" ]; then
    python3 -m src.vector_experiments.analyze_results --input $RESULT_FILES --output results
else
    echo "Warning: No result files found"
fi

echo ""
echo "======================================================"
echo "FINAL EXPERIMENT COMPLETE!"
echo "======================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - results/benchmark_*.json (raw data)"
echo "  - results/benchmark_*.csv (tabular data)"
echo "  - results/tables/ (LaTeX & Markdown tables)"
echo "  - results/figures/ (PNG plots)"
echo "  - results/summary_report.txt"
