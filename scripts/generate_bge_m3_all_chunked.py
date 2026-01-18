#!/usr/bin/env python3
"""
Generate BGE-M3-ALL embeddings with resume support and configurable chunk sizes.
"""

import os
import pickle
import time
import gc
import re
from tqdm import tqdm

from src.vector_experiments.models import BGEM3Embedder
from src.vector_experiments.benchmark import load_dataset, filter_dataset

# Configuration
DATASETS_CONFIG = {
    "fiqa": {"limit": 0, "chunk_size": 5000},
    "dbpedia-entity": {"limit": 100000, "chunk_size": 20000},  # 20k chunks for dbpedia
}

def find_resume_point(output_dir: str) -> tuple:
    """Find the latest checkpoint and return (embeddings_dict, resume_from_doc)."""
    temp_files = []
    for f in os.listdir(output_dir):
        match = re.match(r'bge-m3-all_temp_(\d+)\.pkl', f)
        if match:
            temp_files.append((int(match.group(1)), f))
    
    if not temp_files:
        return {}, 0
    
    # Get latest checkpoint
    temp_files.sort(key=lambda x: x[0], reverse=True)
    latest_count, latest_file = temp_files[0]
    
    print(f"Found checkpoint: {latest_file} (up to doc {latest_count})")
    
    # Load checkpoint
    checkpoint_path = os.path.join(output_dir, latest_file)
    with open(checkpoint_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"Loaded {len(embeddings)} embeddings from checkpoint")
    return embeddings, latest_count

def main():
    print("=" * 60)
    print("BGE-M3-ALL EMBEDDING GENERATOR (RESUME SUPPORT)")
    print("=" * 60)
    
    # Initialize model once
    print("\nLoading BGE-M3 model (return_all=True)...")
    embedder = BGEM3Embedder(return_all=True)
    print("Model loaded!")
    
    for dataset, config in DATASETS_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"Processing: {dataset}")
        print(f"Chunk size: {config['chunk_size']}")
        print(f"{'='*60}")
        
        # Load dataset
        corpus, queries, qrels = load_dataset(dataset)
        
        limit = config.get("limit", 0)
        if limit > 0:
            corpus, queries, qrels = filter_dataset(corpus, queries, qrels, limit)
        
        print(f"Corpus size: {len(corpus)} documents")
        
        # Output path
        output_dir = f"data/embeddings/{dataset}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "bge-m3-all_embeddings.pkl")
        
        # Check if already exists (final)
        if os.path.exists(output_path):
            print(f"Already exists: {output_path}")
            print("Skipping... (delete file to regenerate)")
            continue
        
        # Check for resume point
        all_embeddings, resume_from = find_resume_point(output_dir)
        
        doc_ids = list(corpus.keys())
        chunk_size = config["chunk_size"]
        total_chunks = (len(doc_ids) + chunk_size - 1) // chunk_size
        
        start_time = time.time()
        
        for chunk_start in range(resume_from, len(doc_ids), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(doc_ids))
            chunk_ids = doc_ids[chunk_start:chunk_end]
            chunk_corpus = {did: corpus[did] for did in chunk_ids}
            
            current_chunk = chunk_start // chunk_size + 1
            print(f"\nProcessing chunk {current_chunk}/{total_chunks}")
            print(f"Documents {chunk_start} - {chunk_end}")
            
            # Generate embeddings for chunk
            chunk_embeddings, _ = embedder.encode(chunk_corpus)
            
            # Merge into main dict
            all_embeddings.update(chunk_embeddings)
            
            # Clear GPU memory
            gc.collect()
            
            # Save checkpoint
            temp_path = os.path.join(output_dir, f"bge-m3-all_temp_{chunk_end}.pkl")
            print(f"Saving checkpoint: {temp_path}...")
            with open(temp_path, 'wb') as f:
                pickle.dump(all_embeddings, f)
            
            # Remove old checkpoint (keep only latest)
            old_temp = os.path.join(output_dir, f"bge-m3-all_temp_{chunk_start}.pkl")
            if os.path.exists(old_temp) and chunk_start != resume_from:
                os.remove(old_temp)
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed/60:.1f} minutes")
        
        # Save final
        print(f"Saving final to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(all_embeddings, f)
        
        # Cleanup all temp files
        for f in os.listdir(output_dir):
            if f.startswith("bge-m3-all_temp_"):
                os.remove(os.path.join(output_dir, f))
        
        # Free memory
        del all_embeddings
        gc.collect()
        
        print(f"✓ {dataset} complete!")
    
    print("\n" + "=" * 60)
    print("ALL EMBEDDINGS GENERATED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
