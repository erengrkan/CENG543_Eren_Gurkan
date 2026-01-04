# Methodology

This document describes the experimental methodology used to compare lightweight hybrid retrieval methods against heavyweight multi-modal systems.

## 1. Research Hypothesis

**H₀** (Null): BGE-M3 All (Dense + Sparse + ColBERT) provides significantly higher retrieval quality (Recall@10) than any lightweight hybrid method.

**H₁** (Alternative): A lightweight hybrid method (e.g., MiniLM + SPLADE) can achieve statistically equivalent Recall@10 to BGE-M3 All while delivering ≥10× higher throughput (QPS).

## 2. Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| **Dataset** | DBpedia, SciFact, FiQA, ArguAna | Domain diversity (general, biomedical, financial, argumentative) |
| **Embedding Model** | MiniLM, SPLADE, BM25, Word2Vec, BGE-M3 | Complexity spectrum |
| **Hybrid Alpha (α)** | 0.25, 0.5, 0.75 | Dense vs Sparse weighting |
| **Document Limit** | 1K, 10K, Full | Scalability analysis |

## 3. Dependent Variables (Metrics)

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Recall@K** | Fraction of relevant documents retrieved in top K | K ∈ {1, 10, 100} |
| **Latency P99** | 99th percentile query response time | Milliseconds |
| **QPS** | Queries processed per second | Queries/second |

## 4. Controlled Variables

| Variable | Value | Justification |
|----------|-------|---------------|
| **CPU** | 6 Cores (Docker cgroups) | Eliminate host process noise |
| **RAM** | 12 GB (Docker cgroups) | Prevent swapping |
| **Index Type** | HNSW (Faiss, ef_construction=200, M=32) | Standard ANN index |
| **Batch Size** | 8-12 (model-dependent) | GPU memory constraints |
| **Query Filtering** | Only queries with ≥1 relevant doc in subset | Validity |

## 5. Experimental Procedure

### 5.1 Data Preparation
1.  Download BEIR datasets via `beir` Python library.
2.  For each limit (1K, 10K, Full):
    - Select first N documents from corpus.
    - Filter queries: retain only those with at least one relevant document in the subset.
    - Filter qrels: remove irrelevant doc IDs.

### 5.2 Embedding Generation
For each (Dataset, Model) pair:
1.  Load corpus.
2.  Instantiate embedder (e.g., `MiniLMEmbedder`).
3.  Encode all documents → embeddings.pkl.
4.  Save any model artifacts (e.g., BM25 IDF, Word2Vec vocabulary).

### 5.3 Index Construction
1.  **Dense Models**: Build Faiss HNSW index.
    ```python
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = 200
    index.add(vectors)
    ```
2.  **Sparse Models**: Store in CSR sparse matrix (SciPy).

### 5.4 Evaluation Loop
```
For each Dataset:
    For each (Dense, Sparse) Model Pair:
        For each Alpha in [0.25, 0.5, 0.75]:
            1. Encode queries.
            2. Search dense index → dense_scores.
            3. Search sparse matrix → sparse_scores.
            4. Normalize scores (Min-Max).
            5. Combine: final = α × dense + (1-α) × sparse.
            6. Rank by final_score.
            7. Compute Recall@K, Latency, QPS.
            8. Log results.
```

### 5.5 Statistical Analysis
- **Paired t-test**: Compare BGE-M3 All Recall vs Best Hybrid Recall across datasets.
- **Effect Size (Cohen's d)**: Quantify practical significance.
- **95% Confidence Intervals**: For all reported metrics.

## 6. Implementation Details

### 6.1 Score Normalization
To combine heterogeneous scores (dense cosine similarity, sparse dot product), we apply per-query Min-Max normalization:

```
score_normalized = (score - min(scores)) / (max(scores) - min(scores))
```

This projects all scores to [0, 1] before linear combination.

### 6.2 HNSW Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| M | 32 | Connections per node |
| efConstruction | 200 | Build-time accuracy |
| efSearch | 128 | Query-time accuracy |

### 6.3 ColBERT Scoring (BGE-M3 All)
For MaxSim computation:
```python
for query_token in query_embeddings:
    max_sim = max(dot(query_token, doc_token) for doc_token in doc_embeddings)
    total_score += max_sim
```
This is O(Q × D) per document, where Q = query length, D = document length.

## 7. Reproducibility

All experiments are containerized. To reproduce:
```bash
docker build -t vector-bench .
docker run --rm --cpus=6.0 --memory=12g \
  -v $(pwd)/results:/app/results \
  vector-bench python -m src.vector_experiments.evaluate --full-benchmark
```

Results are saved to `results/benchmark_YYYYMMDD_HHMMSS.json`.
