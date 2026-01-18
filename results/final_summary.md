# Vector Retrieval Benchmark Results

**Generated:** 2026-01-18 15:33

## Overview

- **Datasets:** dbpedia-entity, scifact, arguana, fiqa
- **Models tested:** 12
- **Total experiments:** 234

## Top Performing Models by Dataset

| Dataset | Best Model | Recall@10 | QPS |
|---------|------------|-----------|-----|
| dbpedia-entity | minilm+splade | 0.4536 | 4.8 |
| scifact | minilm+splade | 0.9429 | 669.9 |
| arguana | minilm+splade | 0.9226 | 249.2 |
| fiqa | minilm | 0.8750 | 3533.7 |

## Key Insights

1. **Hybrid models improve recall by -0.6%** over single models on average
2. **BGE-M3-ALL baseline:** Average R@10=0.7303, QPS=8.7
3. **Best overall:** minilm+splade on scifact with R@10=0.9429
