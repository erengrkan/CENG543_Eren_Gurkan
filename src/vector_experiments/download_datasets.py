"""
Dataset Download Script

This script downloads and prepares the 4 BEIR benchmark datasets used in
the hybrid retrieval experiments.

Datasets:
1. dbpedia-entity: Wikipedia entity descriptions (General Knowledge)
2. scifact: Scientific claim verification (Biomedical domain)
3. fiqa: Financial opinion QA (Financial domain)
4. arguana: Argument retrieval (Argumentative reasoning)

Usage:
    python -m src.vector_experiments.download_datasets --output_dir data/raw

Technical Notes:
- Uses the `beir` library for standardized dataset access
- Downloads are cached to avoid redundant network requests
- Each dataset includes: corpus.jsonl, queries.jsonl, qrels/

Data Format (BEIR Standard):
- Corpus: {doc_id: {"title": str, "text": str}}
- Queries: {query_id: str}
- Qrels: {query_id: {doc_id: relevance_score}}

References:
- Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
  of Information Retrieval Models", NeurIPS 2021 Datasets Track.
"""

import os
import argparse
from typing import Dict, Tuple, Any
from beir import util
from beir.datasets.data_loader import GenericDataLoader


# Dataset registry with metadata
DATASETS = {
    "dbpedia-entity": {
        "beir_name": "dbpedia-entity",
        "description": "Wikipedia entity retrieval (General Knowledge)",
        "corpus_size": "4.6M",
        "query_count": "467",
        "domain": "Encyclopedic"
    },
    "scifact": {
        "beir_name": "scifact",
        "description": "Scientific claim verification (Biomedical)",
        "corpus_size": "5K",
        "query_count": "300",
        "domain": "Biomedical"
    },
    "fiqa": {
        "beir_name": "fiqa",
        "description": "Financial opinion QA",
        "corpus_size": "57K",
        "query_count": "648",
        "domain": "Financial"
    },
    "arguana": {
        "beir_name": "arguana",
        "description": "Argument retrieval",
        "corpus_size": "8.7K",
        "query_count": "1406",
        "domain": "Argumentative"
    }
}


def download_dataset(
    dataset_name: str,
    output_dir: str
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Download a BEIR dataset and return its components.
    
    Args:
        dataset_name: One of the keys in DATASETS registry.
        output_dir: Directory to store downloaded data.
        
    Returns:
        Tuple of (corpus, queries, qrels)
        - corpus: {doc_id: {"title": str, "text": str}}
        - queries: {query_id: str}
        - qrels: {query_id: {doc_id: relevance}}
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    beir_name = DATASETS[dataset_name]["beir_name"]
    data_path = os.path.join(output_dir, beir_name)
    
    # Download if not exists
    if not os.path.exists(data_path):
        print(f"Downloading {dataset_name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{beir_name}.zip"
        util.download_and_unzip(url, output_dir)
    else:
        print(f"Dataset {dataset_name} already exists at {data_path}")
    
    # Load data
    print(f"Loading {dataset_name}...")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    print(f"  Corpus: {len(corpus):,} documents")
    print(f"  Queries: {len(queries):,}")
    print(f"  Qrels: {len(qrels):,} query-doc pairs")
    
    return corpus, queries, qrels


def download_all(output_dir: str) -> None:
    """
    Download all 4 benchmark datasets.
    
    Args:
        output_dir: Root directory for all datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name, metadata in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Description: {metadata['description']}")
        print(f"Domain: {metadata['domain']}")
        print(f"{'='*60}")
        
        try:
            corpus, queries, qrels = download_dataset(dataset_name, output_dir)
            print(f"✓ {dataset_name} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {e}")
            

def main():
    parser = argparse.ArgumentParser(
        description="Download BEIR benchmark datasets for retrieval experiments"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to store downloaded datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all"] + list(DATASETS.keys()),
        help="Specific dataset to download, or 'all'"
    )
    args = parser.parse_args()
    
    if args.dataset == "all":
        download_all(args.output_dir)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        download_dataset(args.dataset, args.output_dir)
        
    print("\n✓ Dataset download complete!")


if __name__ == "__main__":
    main()
