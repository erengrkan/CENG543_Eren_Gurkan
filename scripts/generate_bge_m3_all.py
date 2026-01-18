#!/usr/bin/env python3
"""
Generate BGE-M3-ALL embeddings (Dense + Sparse + ColBERT) for all datasets.
This script only generates embeddings - benchmarking is done separately.

Usage:
    python scripts/generate_bge_m3_all.py

Time estimate on A100: ~60-90 min per large dataset
"""

import os
import pickle
import time
from tqdm import tqdm

from src.vector_experiments.models import BGEM3Embedder
from src.vector_experiments.benchmark import load_dataset, filter_dataset

# Configuration
DATASETS = ["scifact", "arguana", "fiqa", "dbpedia-entity"]
DATASET_LIMITS = {
    "dbpedia-entity": 100000,
    "scifact": 0,
    "fiqa": 0,
    "arguana": 0
}

def main():
    print("=" * 60)
    print("BGE-M3-ALL EMBEDDING GENERATOR")
    print("=" * 60)
    
    # Initialize model once (loads into GPU memory)
    print("\nLoading BGE-M3 model (return_all=True)...")
    embedder = BGEM3Embedder(return_all=True)
    print("Model loaded!")
    
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset}")
        print(f"{'='*60}")
        
        # Load dataset
        corpus, queries, qrels = load_dataset(dataset)
        
        limit = DATASET_LIMITS.get(dataset, 0)
        if limit > 0:
            corpus, queries, qrels = filter_dataset(corpus, queries, qrels, limit)
        
        print(f"Corpus size: {len(corpus)} documents")
        
        # Output path
        output_dir = f"data/embeddings/{dataset}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "bge-m3-all_embeddings.pkl")
        
        # Check if already exists
        if os.path.exists(output_path):
            print(f"Already exists: {output_path}")
            print("Skipping... (delete file to regenerate)")
            continue
        
        # Generate embeddings
        print(f"Generating embeddings...")
        start_time = time.time()
        
        embeddings, artifact = embedder.encode(corpus)
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed/60:.1f} minutes")
        
        # Save
        print(f"Saving to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        if artifact:
            artifact_path = os.path.join(output_dir, "bge-m3-all_artifact.pkl")
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
        
        print(f"✓ {dataset} complete!")
    
    print("\n" + "=" * 60)
    print("ALL EMBEDDINGS GENERATED!")
    print("=" * 60)
    print("\nNow run benchmark with:")
    print("  python -m src.vector_experiments.benchmark --dataset scifact --embeddings-dir data/embeddings/scifact")

if __name__ == "__main__":
    main()
