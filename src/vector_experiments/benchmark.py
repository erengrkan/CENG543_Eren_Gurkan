"""
Comprehensive Benchmark Script

This script orchestrates the full evaluation pipeline for the hybrid
retrieval experiments. It supports:

1. Multiple BEIR datasets (DBpedia, SciFact, FiQA, ArguAna)
2. Single model evaluation (MiniLM, SPLADE, BM25, Word2Vec, BGE-M3)
3. Hybrid evaluation with configurable alpha values
4. HNSW indexing for dense vectors (Faiss)
5. Structured result logging (JSON + CSV)

Usage:
    # Quick test (1K docs, single dataset)
    python -m src.vector_experiments.benchmark --dataset dbpedia-entity --limit 1000

    # Full benchmark (all datasets, all models, all alphas)
    python -m src.vector_experiments.benchmark --full

    # Specific hybrid test
    python -m src.vector_experiments.benchmark --dense minilm --sparse splade --alphas 0.25,0.5,0.75

Output:
    results/benchmark_YYYYMMDD_HHMMSS.json
    results/benchmark_YYYYMMDD_HHMMSS.csv

Technical Notes:
- Dense vectors are indexed using HNSW (Faiss) for O(log N) search
- Sparse vectors use CSR matrix dot product (exact search)
- Score combination uses min-max normalization before linear interpolation
- All metrics computed using BEIR's EvaluateRetrieval

References:
- BEIR: Thakur et al., NeurIPS 2021 Datasets Track
- HNSW: Malkov & Yashunin, IEEE TPAMI 2018
"""

import os
import sys
import json
import csv
import argparse
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import scipy.sparse
from tqdm import tqdm

# Local imports
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader
from beir import util

from src.vector_experiments.indexer import FlatIPIndexer
from src.vector_experiments.models import (
    BaseEmbedder,
    SpladeEmbedder,
    BGEM3Embedder,
    BM25Embedder,
    Word2VecEmbedder,
    MiniLMEmbedder
)


# ============================================================================
# Configuration
# ============================================================================

DATASETS = ["dbpedia-entity", "scifact", "fiqa", "arguana"]

# Models for hybrid combinations (lightweight)
DENSE_MODELS = ["minilm", "word2vec"]
SPARSE_MODELS = ["splade", "bm25"]

# Heavyweight baselines (NOT used in hybrids)
BASELINE_MODELS = ["bge-m3", "bge-m3-all"]

DEFAULT_ALPHAS = [0.25, 0.5, 0.75]

# HNSW parameters (see docs/methodology.md for justification)
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128


@dataclass
class BenchmarkResult:
    """Container for a single evaluation result."""
    dataset: str
    model: str
    model_type: str  # "single", "hybrid"
    dense_model: Optional[str]
    sparse_model: Optional[str]
    alpha: Optional[float]
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    latency_p99_ms: float
    qps: float
    num_docs: int
    num_queries: int
    timestamp: str


# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(dataset_name: str, data_dir: str = "data/raw") -> Tuple[Dict, Dict, Dict]:
    """
    Load a BEIR dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'scifact')
        data_dir: Root directory for datasets
        
    Returns:
        Tuple of (corpus, queries, qrels)
    """
    data_path = os.path.join(data_dir, dataset_name)
    
    if not os.path.exists(data_path):
        print(f"Downloading {dataset_name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        util.download_and_unzip(url, data_dir)
    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def load_embeddings(path: str) -> Dict[str, Any]:
    """Load pre-computed embeddings from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifact(path: str) -> Optional[Any]:
    """Load model artifact (e.g., BM25 IDF dict)."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def filter_dataset(
    corpus: Dict, 
    queries: Dict, 
    qrels: Dict, 
    limit: int
) -> Tuple[Dict, Dict, Dict]:
    """Filter dataset to first N documents with valid queries."""
    doc_ids = list(corpus.keys())[:limit]
    valid_doc_ids = set(doc_ids)
    
    filtered_corpus = {k: corpus[k] for k in doc_ids}
    
    # Filter qrels
    filtered_qrels = {}
    valid_query_ids = set()
    for qid, rels in qrels.items():
        new_rels = {did: score for did, score in rels.items() if did in valid_doc_ids}
        if new_rels:
            filtered_qrels[qid] = new_rels
            valid_query_ids.add(qid)
    
    # Filter queries
    filtered_queries = {qid: queries[qid] for qid in valid_query_ids if qid in queries}
    
    return filtered_corpus, filtered_queries, filtered_qrels


# ============================================================================
# Indexing
# ============================================================================

def build_hnsw_index(
    doc_ids: List[str],
    embeddings: Dict[str, np.ndarray],
    dimension: int
) -> FlatIPIndexer:
    """
    Build HNSW index for dense embeddings.
    
    Args:
        doc_ids: List of document IDs
        embeddings: {doc_id: vector}
        dimension: Vector dimensionality
        
    Returns:
        Populated HNSWIndexerIP instance
    """
    index = FlatIPIndexer(dimension=dimension)
    
    # Batch add for efficiency
    vectors = np.array([embeddings[did] for did in doc_ids], dtype=np.float32)
    index.add_batch(doc_ids, vectors)
    
    return index


def build_sparse_matrix(
    doc_ids: List[str],
    embeddings: Dict[str, Dict],
    vocab_size: int = 250002
) -> Tuple[scipy.sparse.csr_matrix, Dict[str, int]]:
    """
    Build sparse CSR matrix for sparse embeddings.
    
    Args:
        doc_ids: List of document IDs
        embeddings: {doc_id: {token_id: weight}}
        vocab_size: Maximum vocabulary size
        
    Returns:
        Tuple of (sparse_matrix, term_to_id_map)
    """
    doc_map = {did: i for i, did in enumerate(doc_ids)}
    
    # Determine if keys are strings (BM25) or ints (SPLADE)
    sample_keys = list(embeddings[doc_ids[0]].keys())
    use_string_keys = isinstance(sample_keys[0], str) if sample_keys else False
    
    if use_string_keys:
        # Build term vocabulary for BM25
        term_to_id = {}
        for vec in embeddings.values():
            for term in vec:
                if term not in term_to_id:
                    term_to_id[term] = len(term_to_id)
        vocab_size = len(term_to_id)
    else:
        term_to_id = None
    
    # Build sparse matrix
    row_ind, col_ind, data = [], [], []
    for did in doc_ids:
        vec = embeddings.get(did, {})
        idx = doc_map[did]
        for tid, w in vec.items():
            row_ind.append(idx)
            col_ind.append(term_to_id[tid] if term_to_id else int(tid))
            data.append(float(w))
    
    matrix = scipy.sparse.csr_matrix(
        (data, (row_ind, col_ind)),
        shape=(len(doc_ids), vocab_size)
    )
    
    return matrix, term_to_id


# ============================================================================
# Search Functions
# ============================================================================

def search_hnsw(
    index: HNSWIndexerIP,
    query_embeddings: Dict[str, np.ndarray],
    top_k: int = 100
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    Search HNSW index for dense vectors.
    
    Returns:
        Tuple of (results, latency_p99_ms, qps)
    """
    q_ids = list(query_embeddings.keys())
    results = {}
    latencies = []
    
    start_total = time.time()
    for qid in tqdm(q_ids, desc="HNSW Search"):
        t_start = time.perf_counter()
        hits = index.search(query_embeddings[qid], top_k=top_k)
        latencies.append(time.perf_counter() - t_start)
        results[qid] = {doc_id: score for doc_id, score in hits}
    
    total_time = time.time() - start_total
    latency_p99 = np.percentile(np.array(latencies) * 1000, 99)
    qps = len(q_ids) / total_time
    
    return results, latency_p99, qps


def search_sparse(
    doc_matrix: scipy.sparse.csr_matrix,
    doc_ids: List[str],
    query_embeddings: Dict[str, Dict],
    term_to_id: Optional[Dict[str, int]],
    top_k: int = 100
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    Search sparse matrix using dot product.
    
    Returns:
        Tuple of (results, latency_p99_ms, qps)
    """
    q_ids = list(query_embeddings.keys())
    
    # Build query matrix
    vocab_size = doc_matrix.shape[1]
    q_row, q_col, q_data = [], [], []
    for i, qid in enumerate(q_ids):
        vec = query_embeddings.get(qid, {})
        for tid, w in vec.items():
            if term_to_id and tid not in term_to_id:
                continue  # Skip unknown terms
            col = term_to_id[tid] if term_to_id else int(tid)
            if col < vocab_size:
                q_row.append(i)
                q_col.append(col)
                q_data.append(float(w))
    
    query_matrix = scipy.sparse.csr_matrix(
        (q_data, (q_row, q_col)),
        shape=(len(q_ids), vocab_size)
    )
    
    doc_matrix_T = doc_matrix.transpose()
    
    results = {}
    latencies = []
    
    start_total = time.time()
    for i, qid in enumerate(tqdm(q_ids, desc="Sparse Search")):
        t_start = time.perf_counter()
        
        q_vec = query_matrix[i]
        scores = q_vec.dot(doc_matrix_T)
        if scipy.sparse.issparse(scores):
            scores = scores.toarray().flatten()
        else:
            scores = np.array(scores).flatten()
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results[qid] = {doc_ids[idx]: float(scores[idx]) for idx in top_indices}
        
        latencies.append(time.perf_counter() - t_start)
    
    total_time = time.time() - start_total
    latency_p99 = np.percentile(np.array(latencies) * 1000, 99)
    qps = len(q_ids) / total_time
    
    return results, latency_p99, qps


def hybrid_search(
    dense_results: Dict[str, Dict[str, float]],
    sparse_results: Dict[str, Dict[str, float]],
    alpha: float,
    top_k: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Combine dense and sparse results using linear interpolation.
    
    Final_Score = alpha * Dense_Score_Norm + (1 - alpha) * Sparse_Score_Norm
    
    Uses min-max normalization per query.
    """
    combined = {}
    
    for qid in dense_results.keys():
        dense_scores = dense_results.get(qid, {})
        sparse_scores = sparse_results.get(qid, {})
        
        # Get all doc_ids
        all_docs = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Normalize dense scores
        d_vals = list(dense_scores.values()) if dense_scores else [0]
        d_min, d_max = min(d_vals), max(d_vals)
        d_range = d_max - d_min if d_max != d_min else 1
        
        # Normalize sparse scores
        s_vals = list(sparse_scores.values()) if sparse_scores else [0]
        s_min, s_max = min(s_vals), max(s_vals)
        s_range = s_max - s_min if s_max != s_min else 1
        
        # Combine
        doc_scores = {}
        for doc_id in all_docs:
            d_score = (dense_scores.get(doc_id, d_min) - d_min) / d_range
            s_score = (sparse_scores.get(doc_id, s_min) - s_min) / s_range
            doc_scores[doc_id] = alpha * d_score + (1 - alpha) * s_score
        
        # Sort and limit
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        combined[qid] = dict(sorted_docs)
    
    return combined


# ============================================================================
# Evaluation
# ============================================================================

def compute_metrics(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]]
) -> Dict[str, float]:
    """Compute retrieval metrics using BEIR."""
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 10, 100])
    
    return {
        "recall@1": recall.get("Recall@1", 0),
        "recall@10": recall.get("Recall@10", 0),
        "recall@100": recall.get("Recall@100", 0),
        "ndcg@10": ndcg.get("NDCG@10", 0)
    }


# ============================================================================
# Main Benchmark Functions
# ============================================================================

def run_bge_m3_all_benchmark(
    dataset: str,
    doc_embeddings: Dict,
    doc_ids: List[str],
    queries: Dict,
    qrels: Dict,
    embeddings_dir: str
) -> Optional[BenchmarkResult]:
    """
    Special evaluation for BGE-M3 ALL (Dense + Sparse + ColBERT).
    
    This combines three scoring signals:
    1. Dense: Cosine similarity
    2. Sparse: Dot product 
    3. ColBERT: MaxSim token-level matching
    """
    print(f"\n{'='*60}")
    print(f"Evaluating BGE-M3 ALL (Dense+Sparse+ColBERT) on {dataset}")
    print(f"{'='*60}")
    
    # Encode queries
    print("Encoding queries (ALL mode)...")
    embedder = BGEM3Embedder(return_all=True)
    query_embeddings = embedder.encode_queries(queries, None)
    
    q_ids = list(queries.keys())
    
    # Extract components
    print("Preparing dense component...")
    dense_docs = np.array([doc_embeddings[did]['dense'] for did in doc_ids], dtype=np.float32)
    dense_queries = np.array([query_embeddings[qid]['dense'] for qid in q_ids], dtype=np.float32)
    
    # Normalize
    dense_docs = dense_docs / np.linalg.norm(dense_docs, axis=1, keepdims=True)
    dense_queries = dense_queries / np.linalg.norm(dense_queries, axis=1, keepdims=True)
    
    print("Preparing sparse component...")
    # Build sparse matrices
    max_idx = 250002
    row_ind, col_ind, data = [], [], []
    for i, did in enumerate(doc_ids):
        vec = doc_embeddings[did]['sparse']
        for tid, w in vec.items():
            row_ind.append(i)
            col_ind.append(int(tid))
            data.append(float(w))
    sparse_doc_matrix = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(doc_ids), max_idx))
    
    q_row, q_col, q_data = [], [], []
    for i, qid in enumerate(q_ids):
        vec = query_embeddings[qid]['sparse']
        for tid, w in vec.items():
            q_row.append(i)
            q_col.append(int(tid))
            q_data.append(float(w))
    sparse_query_matrix = scipy.sparse.csr_matrix((q_data, (q_row, q_col)), shape=(len(q_ids), max_idx))
    sparse_doc_matrix_T = sparse_doc_matrix.transpose()
    
    # Search
    print("Running combined search (Dense + Sparse + ColBERT)...")
    latencies = []
    results = {}
    top_k = 100
    
    start_total = time.time()
    for i, qid in enumerate(tqdm(q_ids, desc="BGE-M3-ALL Search")):
        t_start = time.perf_counter()
        
        # Dense score
        score_dense = np.dot(dense_docs, dense_queries[i])
        
        # Sparse score
        sparse_q = sparse_query_matrix[i]
        score_sparse = sparse_q.dot(sparse_doc_matrix_T)
        if scipy.sparse.issparse(score_sparse):
            score_sparse = score_sparse.toarray().flatten()
        else:
            score_sparse = np.array(score_sparse).flatten()
        
        # ColBERT score (MaxSim) - simplified for speed
        q_colbert = query_embeddings[qid]['colbert']
        if hasattr(q_colbert, 'numpy'): q_colbert = q_colbert.numpy()
        q_colbert = np.array(q_colbert, dtype=np.float32)
        
        score_colbert = np.zeros(len(doc_ids))
        for j, did in enumerate(doc_ids):
            d_colbert = np.array(doc_embeddings[did]['colbert'], dtype=np.float32)
            sim_matrix = np.dot(q_colbert, d_colbert.T)
            max_sim = np.max(sim_matrix, axis=1)
            score_colbert[j] = np.sum(max_sim)
        
        # Combine (simple sum)
        total_score = score_dense + score_sparse + score_colbert
        
        top_indices = np.argsort(total_score)[::-1][:top_k]
        results[qid] = {doc_ids[idx]: float(total_score[idx]) for idx in top_indices}
        
        latencies.append(time.perf_counter() - t_start)
    
    total_time = time.time() - start_total
    latency_p99 = np.percentile(np.array(latencies) * 1000, 99)
    qps = len(q_ids) / total_time
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results, qrels)
    
    result = BenchmarkResult(
        dataset=dataset,
        model="bge-m3-all",
        model_type="single",
        dense_model="bge-m3-all",
        sparse_model=None,
        alpha=None,
        recall_at_1=metrics["recall@1"],
        recall_at_10=metrics["recall@10"],
        recall_at_100=metrics["recall@100"],
        ndcg_at_10=metrics["ndcg@10"],
        latency_p99_ms=latency_p99,
        qps=qps,
        num_docs=len(doc_ids),
        num_queries=len(q_ids),
        timestamp=datetime.now().isoformat()
    )
    
    print(f"Recall@10: {result.recall_at_10:.4f}")
    print(f"NDCG@10: {result.ndcg_at_10:.4f}")
    print(f"Latency P99: {result.latency_p99_ms:.2f} ms")
    print(f"QPS: {result.qps:.2f}")
    
    return result


def run_single_model_benchmark(
    dataset: str,
    model_name: str,
    embeddings_dir: str,
    corpus: Dict,
    queries: Dict,
    qrels: Dict,
    limit: int
) -> Optional[BenchmarkResult]:
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {dataset}")
    print(f"{'='*60}")
    
    emb_path = os.path.join(embeddings_dir, f"{model_name}_embeddings.pkl")
    if not os.path.exists(emb_path):
        print(f"Embeddings not found: {emb_path}")
        return None
    
    doc_embeddings = load_embeddings(emb_path)
    artifact_path = emb_path.replace("_embeddings.pkl", "_artifact.pkl")
    artifact = load_artifact(artifact_path)
    
    doc_ids = list(doc_embeddings.keys())[:limit]
    
    # Special handling for BGE-M3-ALL (multi-modal: dense+sparse+colbert)
    if model_name == "bge-m3-all":
        return run_bge_m3_all_benchmark(
            dataset, doc_embeddings, doc_ids, queries, qrels, embeddings_dir
        )
    
    # Determine model type
    is_dense = model_name in DENSE_MODELS or model_name in ["bge-m3"]
    
    # Encode queries
    print("Encoding queries...")
    embedder = get_embedder(model_name)
    query_embeddings = embedder.encode_queries(queries, artifact)
    
    # Search
    if is_dense:
        # Get dimension from first embedding
        sample_vec = list(doc_embeddings.values())[0]
        dim = sample_vec.shape[0] if hasattr(sample_vec, 'shape') else len(sample_vec)
        
        print("Building HNSW index...")
        index = build_hnsw_index(doc_ids, doc_embeddings, dim)
        
        print("Searching...")
        results, latency_p99, qps = search_hnsw(index, query_embeddings)
    else:
        print("Building sparse matrix...")
        doc_matrix, term_to_id = build_sparse_matrix(doc_ids, doc_embeddings)
        
        print("Searching...")
        results, latency_p99, qps = search_sparse(
            doc_matrix, doc_ids, query_embeddings, term_to_id
        )
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results, qrels)
    
    result = BenchmarkResult(
        dataset=dataset,
        model=model_name,
        model_type="single",
        dense_model=model_name if is_dense else None,
        sparse_model=model_name if not is_dense else None,
        alpha=None,
        recall_at_1=metrics["recall@1"],
        recall_at_10=metrics["recall@10"],
        recall_at_100=metrics["recall@100"],
        ndcg_at_10=metrics["ndcg@10"],
        latency_p99_ms=latency_p99,
        qps=qps,
        num_docs=len(doc_ids),
        num_queries=len(queries),
        timestamp=datetime.now().isoformat()
    )
    
    print(f"Recall@10: {result.recall_at_10:.4f}")
    print(f"NDCG@10: {result.ndcg_at_10:.4f}")
    print(f"Latency P99: {result.latency_p99_ms:.2f} ms")
    print(f"QPS: {result.qps:.2f}")
    
    return result


def run_hybrid_benchmark(
    dataset: str,
    dense_model: str,
    sparse_model: str,
    alpha: float,
    embeddings_dir: str,
    corpus: Dict,
    queries: Dict,
    qrels: Dict,
    limit: int
) -> Optional[BenchmarkResult]:
    """Evaluate a hybrid model combination."""
    print(f"\n{'='*60}")
    print(f"Hybrid: {dense_model} + {sparse_model} (α={alpha}) on {dataset}")
    print(f"{'='*60}")
    
    # Load embeddings
    dense_path = os.path.join(embeddings_dir, f"{dense_model}_embeddings.pkl")
    sparse_path = os.path.join(embeddings_dir, f"{sparse_model}_embeddings.pkl")
    
    if not os.path.exists(dense_path) or not os.path.exists(sparse_path):
        print("Embeddings not found")
        return None
    
    dense_embeddings = load_embeddings(dense_path)
    sparse_embeddings = load_embeddings(sparse_path)
    
    dense_artifact = load_artifact(dense_path.replace("_embeddings.pkl", "_artifact.pkl"))
    sparse_artifact = load_artifact(sparse_path.replace("_embeddings.pkl", "_artifact.pkl"))
    
    doc_ids = list(dense_embeddings.keys())[:limit]
    
    # Encode queries
    print("Encoding queries (dense)...")
    dense_embedder = get_embedder(dense_model)
    dense_query_emb = dense_embedder.encode_queries(queries, dense_artifact)
    
    print("Encoding queries (sparse)...")
    sparse_embedder = get_embedder(sparse_model)
    sparse_query_emb = sparse_embedder.encode_queries(queries, sparse_artifact)
    
    # Build indexes
    sample_vec = list(dense_embeddings.values())[0]
    dim = sample_vec.shape[0] if hasattr(sample_vec, 'shape') else len(sample_vec)
    
    print("Building HNSW index...")
    hnsw_index = build_hnsw_index(doc_ids, dense_embeddings, dim)
    
    print("Building sparse matrix...")
    sparse_matrix, term_to_id = build_sparse_matrix(doc_ids, sparse_embeddings)
    
    # Search both
    print("Searching (dense)...")
    dense_results, dense_lat, _ = search_hnsw(hnsw_index, dense_query_emb)
    
    print("Searching (sparse)...")
    sparse_results, sparse_lat, _ = search_sparse(
        sparse_matrix, doc_ids, sparse_query_emb, term_to_id
    )
    
    # Combine
    print("Combining results...")
    start_combine = time.time()
    hybrid_results = hybrid_search(dense_results, sparse_results, alpha)
    combine_time = time.time() - start_combine
    
    # Combined latency (approximate)
    latency_p99 = max(dense_lat, sparse_lat) + (combine_time * 1000 / len(queries))
    qps = len(queries) / (len(queries) / (1000 / dense_lat) + len(queries) / (1000 / sparse_lat) + combine_time)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(hybrid_results, qrels)
    
    result = BenchmarkResult(
        dataset=dataset,
        model=f"{dense_model}+{sparse_model}",
        model_type="hybrid",
        dense_model=dense_model,
        sparse_model=sparse_model,
        alpha=alpha,
        recall_at_1=metrics["recall@1"],
        recall_at_10=metrics["recall@10"],
        recall_at_100=metrics["recall@100"],
        ndcg_at_10=metrics["ndcg@10"],
        latency_p99_ms=latency_p99,
        qps=qps,
        num_docs=len(doc_ids),
        num_queries=len(queries),
        timestamp=datetime.now().isoformat()
    )
    
    print(f"Recall@10: {result.recall_at_10:.4f}")
    print(f"NDCG@10: {result.ndcg_at_10:.4f}")
    print(f"Latency P99: {result.latency_p99_ms:.2f} ms")
    
    return result


def get_embedder(model_name: str) -> BaseEmbedder:
    """Factory function for embedder instances."""
    if model_name == "minilm":
        return MiniLMEmbedder()
    elif model_name == "bge-m3":
        return BGEM3Embedder(return_all=False)
    elif model_name == "bge-m3-all":
        return BGEM3Embedder(return_all=True)
    elif model_name == "splade":
        return SpladeEmbedder()
    elif model_name == "bm25":
        return BM25Embedder()
    elif model_name == "word2vec":
        return Word2VecEmbedder()
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# Result Export
# ============================================================================

def save_results(results: List[BenchmarkResult], output_dir: str):
    """Save results to JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # CSV
    csv_path = os.path.join(output_dir, f"benchmark_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"Results saved to: {csv_path}")


def print_summary_table(results: List[BenchmarkResult]):
    """Print formatted summary table."""
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    
    header = f"{'Dataset':<15} {'Model':<25} {'Alpha':<8} {'R@10':<10} {'NDCG@10':<10} {'P99(ms)':<10} {'QPS':<10}"
    print(header)
    print("-"*100)
    
    for r in results:
        alpha_str = f"{r.alpha:.2f}" if r.alpha else "-"
        print(f"{r.dataset:<15} {r.model:<25} {alpha_str:<8} {r.recall_at_10:<10.4f} {r.ndcg_at_10:<10.4f} {r.latency_p99_ms:<10.2f} {r.qps:<10.2f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Hybrid Retrieval Benchmark"
    )
    parser.add_argument("--dataset", type=str, default="dbpedia-entity",
                        choices=DATASETS + ["all"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--dense", type=str, default=None, choices=DENSE_MODELS)
    parser.add_argument("--sparse", type=str, default=None, choices=SPARSE_MODELS)
    parser.add_argument("--alphas", type=str, default="0.25,0.5,0.75",
                        help="Comma-separated alpha values")
    parser.add_argument("--embeddings-dir", type=str, default="data/embeddings")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--full", action="store_true",
                        help="Run full benchmark (all datasets, models, alphas)")
    parser.add_argument("--single-only", action="store_true",
                        help="Only run single model benchmarks")
    parser.add_argument("--hybrid-only", action="store_true",
                        help="Only run hybrid benchmarks")
    args = parser.parse_args()
    
    alphas = [float(a) for a in args.alphas.split(",")]
    datasets = DATASETS if args.dataset == "all" or args.full else [args.dataset]
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*60}")
        
        # Load dataset
        corpus, queries, qrels = load_dataset(dataset)
        corpus, queries, qrels = filter_dataset(corpus, queries, qrels, args.limit)
        print(f"Loaded: {len(corpus)} docs, {len(queries)} queries")
        
        if len(queries) == 0:
            print(f"No valid queries for {dataset}, skipping...")
            continue
        
        # Single model benchmarks (includes baselines)
        if not args.hybrid_only:
            all_single_models = DENSE_MODELS + SPARSE_MODELS + BASELINE_MODELS
            for model in all_single_models:
                result = run_single_model_benchmark(
                    dataset, model, args.embeddings_dir,
                    corpus, queries, qrels, args.limit
                )
                if result:
                    all_results.append(result)
        
        # Hybrid benchmarks
        if not args.single_only:
            dense_models = [args.dense] if args.dense else DENSE_MODELS
            sparse_models = [args.sparse] if args.sparse else SPARSE_MODELS
            
            for dense in dense_models:
                for sparse in sparse_models:
                    for alpha in alphas:
                        result = run_hybrid_benchmark(
                            dataset, dense, sparse, alpha,
                            args.embeddings_dir, corpus, queries, qrels, args.limit
                        )
                        if result:
                            all_results.append(result)
    
    # Save and display results
    if all_results:
        save_results(all_results, args.output_dir)
        print_summary_table(all_results)
    else:
        print("No results collected. Check embeddings paths.")


if __name__ == "__main__":
    main()
