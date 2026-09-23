```markdown
# CENG543 Vector Retrieval Benchmark

A performance comparison of Dense, Sparse, and Hybrid vector retrieval methods across BEIR benchmark datasets.

## 📊 Results Summary

| Dataset | Best Model | Recall@10 | QPS |
|---------|-------------|-----------|-----|
| SciFact | minilm+splade (α=0.5) | **0.9429** | 253 |
| ArguAna | minilm+splade (α=0.5) | **0.9226** | 249 |
| DBpedia-Entity | minilm+splade (α=0.5) | **0.4536** | 4.7 |
| FiQA | bge-m3 | **0.4833** | 61 |

---

## 📈 Figure Breakdown

### 1. Speed vs Quality Tradeoff (`speed_vs_quality.png`)

Maps models based on speed (QPS) and retrieval quality (Recall@10).

**Zones:**
- 🟢 **Green (Top):** High quality (Recall@10 > 0.8)
- 🔵 **Blue (Right):** High throughput (QPS > 100)
- **Top-right corner** is the sweet spot (fast & accurate).

**Markers:**
- ⚪ **Circles:** Dense models (MiniLM, BGE-M3)
- 🔺 **Triangles:** Sparse models (SPLADE, BM25)
- ⬛ **Squares:** Hybrid models (Dense + Sparse)

**Takeaways:**
- `minilm` and `bge-m3` hit the sweet spot.
- `bge-m3-all` is highly accurate but way too slow (stuck on the far left).
- Hybrids (`minilm+splade`) push the absolute highest recall.

---

### 2. Latency vs Recall Tradeoff (`latency_vs_recall.png`)

Looks at per-query latency (ms) against recall. 

**Takeaways:**
- **Top-left** is the ideal target (sub-millisecond + high recall).
- `minilm` sits nicely at ~1ms with great quality.
- `bge-m3-all` lags hard (1000+ ms), making it basically unusable for real-time production.
- Hybrids land in the middle (~10-50ms) but maximize recall.

---

### 3. Recall by Dataset (`recall_by_dataset.png`)

Head-to-head Recall@10 breakdown per dataset.

**Takeaways:**
- **SciFact & ArguAna:** Hybrids dominate.
- **DBpedia-Entity:** The hardest dataset in the bunch; everyone struggles here.
- **FiQA:** BGE-M3 (dense) takes the lead.

---

### 4. Alpha Sensitivity (`alpha_sensitivity.png`)

How tweaking the alpha (α) weight impacts hybrid performance.

**Weighting:**
- α = 0: 100% Sparse (SPLADE/BM25)
- α = 1: 100% Dense (MiniLM/Word2Vec)
- α = 0.5: 50/50 split

**Takeaways:**
- The optimal range is usually between **α = 0.25 and 0.5**.
- Fusing Dense and Sparse consistently beats using either one alone.

---

### 5. Model Ranking (`model_ranking.png`)

Average leaderboard across all datasets.

**Verdict:** `minilm+splade` is the overall winner.

---

### 6. BGE-M3-ALL Comparison (`bge_m3_all_comparison.png`)

Pits the heavy BGE-M3-ALL baseline against the best hybrid model.

**Verdict:**
- Hybrids actually beat BGE-M3-ALL in raw recall.
- Hybrids do it while being **100-1000x faster**.

---

## 🚀 Reproducing the Benchmark

### Prerequisites
- NVIDIA GPU (A100 recommended)
- Docker
- 250GB+ disk space (for embeddings)

### Step 1: Clone the Repo
```bash
git clone [https://github.com/erengrkan/CENG543_Eren_Gurkan.git](https://github.com/erengrkan/CENG543_Eren_Gurkan.git)
cd CENG543_Eren_Gurkan

```

### Step 2: Build the Docker Image

```bash
docker build -t vector-bench-gpu -f Dockerfile.gpu .

```

### Step 3: Run the Pipeline

```bash
./scripts/run_a100.sh

```

This script:

1. Downloads the BEIR datasets.
2. Generates embeddings for all models.
3. Runs the benchmark tests.
4. Dumps results into the `results/` folder.

### Step 4: Generate Figures

```bash
docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/scripts:/app/scripts \
    --entrypoint python vector-bench-gpu /app/scripts/generate_figures.py

```

### ETA (on an A100)

| Phase | Duration |
| --- | --- |
| Embedding Generation | 2-3 hours |
| Benchmarking | 30-60 minutes |
| **Total** | ~3-4 hours |

---

## ⚠️ The BGE-M3-ALL Bottleneck

### Why is it missing from some plots?

**BGE-M3-ALL** fuses three different retrieval signals:

1. **Dense:** Cosine similarity
2. **Sparse:** Lexical weights
3. **ColBERT:** Token-level MaxSim

**The Catch:** ColBERT's MaxSim calculation is **O(n × m × d)** complexity:

* n = Number of documents
* m = Query token count
* d = Document token count

### The Reality of Run Times

| Dataset | Docs | Query | ColBERT Time |
| --- | --- | --- | --- |
| SciFact | 5K | 300 | ~8 mins |
| ArguAna | 8K | 504 | ~15 mins |
| FiQA | 57K | 612 | **~4 hours** (est.) |
| DBpedia | 100K | 393 | **~6+ hours** (est.) |

### Conclusion

* **Small datasets** (SciFact, ArguAna): Tested and included.
* **Large datasets** (FiQA, DBpedia): Skipped. Just not practical to run.
* **Alternative:** Used base BGE-M3 (dense only) instead—way faster, similar quality.

### Note for the Paper

> "BGE-M3-ALL relies on ColBERT token-level matching, making it wildly impractical for large-scale datasets (50K+ docs). Because of this, it was only included in comparisons for the smaller datasets (SciFact, ArguAna)."

---

## 📋 Methodology Notes

### 1. Hybrid Fusion Strategy

Score fusion formula:

```
Final_Score = α × Dense_Score_Norm + (1-α) × Sparse_Score_Norm

```

Where:

* **Min-Max normalization** is applied per-query.
* **α = 0.5** was generally the optimal split.

### 2. HNSW Indexing Params

For Dense vectors in FAISS HNSW:

* `M = 32` (number of links)
* `efConstruction = 200`
* `efSearch = 128`

### 3. Evaluation Metrics

* **Recall@10:** Ratio of relevant docs found in the top 10 results.
* **NDCG@10:** Ranking quality.
* **QPS:** Queries per second.
* **Latency P99:** 99th percentile latency.

### 4. Models Evaluated

| Model | Type | Dim | Source |
| --- | --- | --- | --- |
| MiniLM | Dense | 384d | sentence-transformers |
| SPLADE | Sparse | ~30K | naver/splade-cocondenser |
| BM25 | Sparse | - | rank_bm25 |
| Word2Vec | Dense | 300d | Custom trained |
| BGE-M3 | Dense | 1024d | BAAI/bge-m3 |
| BGE-M3-ALL | Multi | 1024d + sparse + colbert | BAAI/bge-m3 |

### 5. Dataset Stats

| Dataset | Docs | Queries | Domain |
| --- | --- | --- | --- |
| SciFact | 5,183 | 300 | Scientific papers |
| ArguAna | 8,674 | 1,406 | Debate arguments |
| FiQA | 57,638 | 648 | Financial Q&A |
| DBpedia-Entity | 4.6M (100K used) | 400 | Wikipedia entities |

---

## 📁 Repo Structure

```text
CENG543_Eren_Gurkan/
├── src/vector_experiments/
│   ├── models.py          # Embedding models
│   ├── benchmark.py       # Main benchmark engine
│   ├── indexer.py         # HNSW indexing logic
│   └── analyze_results.py # Telemetry & plotting
├── scripts/
│   ├── run_a100.sh        # Master execution script
│   └── generate_figures.py # Matplotlib generators
├── data/
│   ├── raw/               # BEIR datasets
│   └── embeddings/        # Pre-computed embeddings
├── results/
│   ├── figures/           # Plots go here
│   ├── tables/            # LaTeX/MD tables
│   └── benchmark_*.json   # Raw run telemetry
├── Dockerfile.gpu         # GPU environment config
└── README.md              # You are here

```

---

## 🔗 References

1. **BEIR Benchmark:** Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", NeurIPS 2021
2. **SPLADE:** Formal et al., "SPLADE v2: Sparse Lexical and Expansion Model for First Stage Ranking", 2022
3. **BGE-M3:** Chen et al., "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity", 2024
4. **MiniLM:** Wang et al., "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression", 2020

---

**Author:** Eren Gürkan

**Course:** CENG543 - Information Retrieval

**Date:** 2026

```

```
