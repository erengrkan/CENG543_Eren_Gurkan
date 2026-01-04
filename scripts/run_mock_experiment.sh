#!/bin/bash
# Complete Mock Experiment Runner
# Runs full benchmark pipeline across all 4 datasets with 1K doc limit

set -e

echo "=============================================="
echo "COMPLETE MOCK EXPERIMENT (1000 docs per dataset)"
echo "=============================================="

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Datasets to process
DATASETS=("dbpedia-entity" "scifact" "fiqa" "arguana")

# Models to generate embeddings for
MODELS=("minilm" "splade" "bm25" "word2vec" "bge-m3" "bge-m3-all")

echo ""
echo "Step 1: Generating embeddings for all datasets..."
echo ""

for dataset in "${DATASETS[@]}"; do
    echo ">>> Dataset: $dataset"
    
    # Download dataset first
    python3 -m src.vector_experiments.download_datasets --dataset $dataset
    
    # Generate embeddings for each model
    for model in "${MODELS[@]}"; do
        echo "  - Model: $model"
        python3 -m src.vector_experiments.generate_embeddings \
            --model $model \
            --limit 1000 \
            --output_dir "data/embeddings/${dataset}" 2>&1 | tail -1
    done
done

echo ""
echo "Step 2: Running benchmark across all datasets..."
echo ""

python3 -m src.vector_experiments.benchmark \
    --dataset all \
    --limit 1000 \
    --alphas "0.25,0.5,0.75" \
    --output-dir results

echo ""
echo "=============================================="
echo "MOCK EXPERIMENT COMPLETE!"
echo "Results saved to: results/"
echo "=============================================="
