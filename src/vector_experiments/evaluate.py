import os
import argparse
import pickle
import time
import numpy as np
import scipy.sparse
from beir.retrieval.evaluation import EvaluateRetrieval
from src.vector_experiments.data_loader import load_dbpedia_data
from src.vector_experiments.models import SpladeEmbedder, BGEM3Embedder, BM25Embedder, Word2VecEmbedder
from tqdm import tqdm

def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_artifact(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def filter_dataset(corpus, queries, qrels, limit):
    """
    Filters corpus to 'limit' docs.
    Filters queries/qrels to only those relevant to the filtered corpus.
    """
    doc_ids = list(corpus.keys())[:limit]
    valid_doc_ids = set(doc_ids)
    
    # Filter corpus
    filtered_corpus = {k: corpus[k] for k in doc_ids}
    
    # Filter qrels: Keep only relevance judgements for doc_ids present in our subset
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

def evaluate_model(model_name, embeddings_path, corpus, queries, qrels):
    print(f"Evaluating {model_name}...")
    
    # Load Doc Embeddings
    try:
        doc_embeddings = load_embeddings(embeddings_path)
    except FileNotFoundError:
        print(f"Embeddings file not found: {embeddings_path}")
        return
        
    # Prepare Doc Indices
    doc_ids = list(doc_embeddings.keys())
    # Ensure ordering is consistent for matrix ops
    
    # Encode Queries
    # We need to instantiate the embedder just to encode queries
    embedder = None
    if model_name == "splade":
        embedder = SpladeEmbedder()
    elif model_name == "bge-m3":
        embedder = BGEM3Embedder()
    elif model_name == "bm25":
        embedder = BM25Embedder()
    elif model_name == "word2vec":
        embedder = Word2VecEmbedder()
        
    
    # Load Artifact (if exists)
    # Assume artifact is in same dir with suffix _artifact.pkl
    artifact_path = embeddings_path.replace("_embeddings.pkl", "_artifact.pkl")
    artifact = load_artifact(artifact_path)
    
    print("Encoding queries...")
    start_time = time.time()
    query_embeddings = embedder.encode_queries(queries, artifact)
    encoding_time = time.time() - start_time
    # Note: Query encoding latency is usually part of total latency, but often separated.
    # We will measure Search Latency here as requested (usually the vector lookup part).
    # But for end-to-end, should include encoding. Let's focus on Search Latency first as per "vector" context.
    
    # Convert to Matrix
    print("Preparing matrices...")
    
    # Check if sparse or dense
    is_sparse = model_name in ["splade", "bm25"]
    
    if is_sparse:
        # Build Sparse Matrix for Docs
        # Assume doc_embeddings is {doc_id: {token_id: weight}}
        row_ind = []
        col_ind = []
        data = []
        
        doc_map = {did: i for i, did in enumerate(doc_ids)}
        
        # Determine vocabulary size roughly or just use max index
        max_idx = 0
        
        for did, vec in doc_embeddings.items():
            if did not in doc_map: continue # Should be in doc_map
            row = doc_map[did]
            for token_id, weight in vec.items():
                # BM25 uses string tokens. We need to map them to integers globally if we want matrix op.
                # Or we use a simpler per-query dot product if vocab is huge/string-based.
                # Word2Vec/BGE/SPLADE use int or dense.
                # BM25 in our impl used strings (tokens) as keys.
                # SPLADE uses int keys.
                pass 
        
        # Handle BM25 String Keys
        if model_name == "bm25":
            # Build valid vocab maps
            term_to_id = {}
            # Collect all terms from docs and queries
            for vec in doc_embeddings.values():
                for term in vec:
                    if term not in term_to_id: term_to_id[term] = len(term_to_id)
            for vec in query_embeddings.values():
                for term in vec:
                    if term not in term_to_id: term_to_id[term] = len(term_to_id)
                    
            # Rebuild sparse structure with int IDs
            for did, vec in doc_embeddings.items():
                idx = doc_map[did]
                for term, weight in vec.items():
                    row_ind.append(idx)
                    col_ind.append(term_to_id[term])
                    data.append(weight)
            
            vocab_size = len(term_to_id)
            doc_matrix = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(doc_ids), vocab_size))
            
            # Build Query Matrix
            q_row = []
            q_col = []
            q_data = []
            q_ids = list(query_embeddings.keys())
            
            for i, qid in enumerate(q_ids):
                vec = query_embeddings[qid]
                for term, weight in vec.items():
                    if term in term_to_id:
                        q_row.append(i)
                        q_col.append(term_to_id[term])
                        q_data.append(weight)
                        
            query_matrix = scipy.sparse.csr_matrix((q_data, (q_row, q_col)), shape=(len(q_ids), vocab_size))
            
        elif model_name == "splade":
            # Splade uses ints already
            # Find max vocab id
            max_idx = 30522 # BERT vocab size usually, but let's be dynamic
            # Scan once if needed or just use 30522
            
            for vec in doc_embeddings.values():
                if vec: max_idx = max(max_idx, max(vec.keys()))
            max_idx += 1
            
            for did, vec in doc_embeddings.items():
                idx = doc_map[did]
                for tid, w in vec.items():
                    row_ind.append(idx)
                    col_ind.append(tid)
                    data.append(w)
            
            doc_matrix = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(doc_ids), max_idx))
            
            q_row = []
            q_col = []
            q_data = []
            q_ids = list(query_embeddings.keys())
             
            for i, qid in enumerate(q_ids):
                vec = query_embeddings[qid]
                for tid, w in vec.items():
                    if tid < max_idx: # Safety
                        q_row.append(i)
                        q_col.append(tid)
                        q_data.append(w)
            
            query_matrix = scipy.sparse.csr_matrix((q_data, (q_row, q_col)), shape=(len(q_ids), max_idx))
            
    else: # Dense
        # Convert to numpy arrays
        q_ids = list(query_embeddings.keys())
        query_matrix = np.array([query_embeddings[qid] for qid in q_ids])
        
        doc_map = {did: i for i, did in enumerate(doc_ids)}
        doc_matrix = np.array([doc_embeddings[did] for did in doc_ids])
        
        # Normalize for Cosine Similarity
        # (Assuming embeddings might not be normalized)
        norm_q = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        norm_q[norm_q == 0] = 1
        query_matrix = query_matrix / norm_q
        
        norm_d = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
        norm_d[norm_d == 0] = 1
        doc_matrix = doc_matrix / norm_d

    # Run Search & Measure Latency
    print("Running Search...")
    latencies = []
    results = {}
    
    # We iterate queries to measure per-query latency more naturally
    # Or batch it. QPS usually implies batch or high throughput. 
    # Latency P99 implies single query distribution.
    # So we loop.
    
    top_k = 10
    
    start_total = time.time()
    
    # Pre-transpose doc matrix for faster dot product
    doc_matrix_T = doc_matrix.transpose() # CSC if sparse
    
    for i, qid in enumerate(tqdm(q_ids, desc="Searching")):
        t_start = time.perf_counter()
        
        # Get query vector
        q_vec = query_matrix[i] # 1D or 2D (1, dim)
        
        if is_sparse:
             # q_vec is CSR row, dot with CSC docs
             scores = q_vec.dot(doc_matrix_T)
             # scores is 1xN sparse or dense. Usually dense output for search scores
             if scipy.sparse.issparse(scores):
                 scores = scores.toarray().flatten()
             else:
                 scores = scores.flatten()
        else:
            # Dense dot product
            scores = np.dot(doc_matrix, q_vec)
            
        # Top-K
        # If N=1000, sorting is fast.
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Store results
        result_dict = {}
        for idx in top_indices:
            score = scores[idx]
            original_did = doc_ids[idx]
            result_dict[original_did] = float(score)
            
        results[qid] = result_dict
        
        latencies.append(time.perf_counter() - t_start)
    
    total_time = time.time() - start_total
    
    # Metrics
    latencies_ms = np.array(latencies) * 1000
    p99 = np.percentile(latencies_ms, 99)
    avg_lat = np.mean(latencies_ms)
    qps = len(q_ids) / total_time
    
    print(f"Latency P99: {p99:.2f} ms")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"QPS: {qps:.2f}")
    
    # BEIR Evaluation (NDCG, Recall)
    # Beir expects results in {qid: {did: score}} format
    print("Calculating Quality Metrics...")
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 10, 100])
    
    print("--- Stats ---")
    print(f"Model: {model_name}")
    print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
    print(f"Recall@10: {recall['Recall@10']:.4f}")
    print(f"Latency P99: {p99:.4f} ms")
    print(f"QPS: {qps:.4f}")
    
    return {
        "model": model_name,
        "ndcg@10": ndcg['NDCG@10'],
        "recall@10": recall['Recall@10'],
        "p99": p99,
    }


def evaluate_bge_m3_all(embeddings_path, corpus, queries, qrels, limit):
    print("\n================ BGE-M3 ALL (Dense+Sparse+ColBERT) ================")
    
    # Load complex embeddings
    # Format: {doc_id: {'dense':..., 'sparse':..., 'colbert':...}}
    try:
        doc_embeddings = load_embeddings(embeddings_path)
    except FileNotFoundError:
        print(f"Embeddings not found: {embeddings_path}")
        return None
        
    doc_ids = list(doc_embeddings.keys())[:limit]
    q_ids = list(queries.keys())
    
    # Encode Queries
    # We need to instantiate with return_all=True
    embedder = BGEM3Embedder(return_all=True)
    # We don't save artifact for BGE-M3 (pretrained)
    
    print("Encoding queries...")
    start_time = time.time()
    # encode_queries calls encode which handles return_all logic
    query_embeddings = embedder.encode_queries(queries, None)
    
    # Prepare Matrices / Data Structures
    # 1. DENSE
    print("Preparing Dense...")
    dense_docs = np.array([doc_embeddings[did]['dense'] for did in doc_ids])
    dense_queries = np.array([query_embeddings[qid]['dense'] for qid in q_ids])
    
    # Normalize
    dense_docs = dense_docs / np.linalg.norm(dense_docs, axis=1, keepdims=True)
    dense_queries = dense_queries / np.linalg.norm(dense_queries, axis=1, keepdims=True)
    
    # 2. SPARSE (Lexical)
    # They are already weight dicts.
    # We need to build sparse matrix or use dot product manually?
    # Let's use sparse matrix for speed.
    print("Preparing Sparse...")
    doc_map = {did: i for i, did in enumerate(doc_ids)}
    
    # Find max vocab id
    max_idx = 250002 # BGE-M3 vocab size is 250k+
    
    # Build sparse doc matrix
    row_ind, col_ind, data = [], [], []
    for did in doc_ids:
        vec = doc_embeddings[did]['sparse']
        idx = doc_map[did]
        for tid, w in vec.items():
            row_ind.append(idx)
            col_ind.append(int(tid))
            data.append(float(w))
            # max_idx = max(max_idx, int(tid))
            
    sparse_doc_matrix = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(doc_ids), max_idx))
    
    # Build sparse query matrix
    q_row, q_col, q_data = [], [], []
    for i, qid in enumerate(q_ids):
        vec = query_embeddings[qid]['sparse']
        for tid, w in vec.items():
            q_row.append(i)
            q_col.append(int(tid))
            q_data.append(float(w))
            
    sparse_query_matrix = scipy.sparse.csr_matrix((q_data, (q_row, q_col)), shape=(len(q_ids), max_idx))
    sparse_doc_matrix_T = sparse_doc_matrix.transpose()


    # 3. COLBERT
    # Doc embeddings: list of [SeqLen, 1024] tensors/arrays
    # Query embeddings: list of [SeqLen, 1024]
    # We cannot build a single matrix easily. We iterate.
    
    print("Running Search (All Modes)...")
    latencies = []
    results = {}
    top_k = 10
    
    start_total = time.time()
    
    for i, qid in enumerate(tqdm(q_ids, desc="Searching")):
        t_start = time.perf_counter()
        
        # A. Dense Score
        dense_q = dense_queries[i] # (1024,)
        score_dense = np.dot(dense_docs, dense_q) # (N_docs,)
        
        # B. Sparse Score
        sparse_q = sparse_query_matrix[i]
        score_sparse = sparse_q.dot(sparse_doc_matrix_T)
        if scipy.sparse.issparse(score_sparse):
            score_sparse = score_sparse.toarray().flatten()
        else:
            score_sparse = np.array(score_sparse).flatten()
            
        # C. ColBERT Score (MaxSim)
        # Score = Sum_over_query_tokens( Max_over_doc_tokens( q_tok @ d_tok.T ) )
        # This is slow in Python loop.
        
        # Optimization:
        # Pre-compute doc tensor? [N_docs, MaxLen, 1024].
        # But lengths are variable.
        # We probably have to loop over docs or use torch packed sequence.
        # Given we have 1000 docs, we can stack them if we pad?
        # Or just loop. Loop 1000 times per query = 1000 * 100 = 100k iters. Too slow?
        # 100 queries * 1000 docs = 100k checks.
        # Inner op is matrix mult (Q_len x D_len).
        
        # Let's try to batch docs?
        # Or just simple loop for now (it will show latency impact strongly!).
        
        q_colbert = query_embeddings[qid]['colbert'] # (Q_len, 1024)
        # q_colbert is apt to be torch tensor or numpy?
        # BGEM3 returns numpy usually or list.
        # Let's assume numpy.
        if hasattr(q_colbert, 'numpy'): q_colbert = q_colbert.numpy()
        q_colbert = np.array(q_colbert)
        
        score_colbert = np.zeros(len(doc_ids))
        
        # Heavy loop
        for j, did in enumerate(doc_ids):
            d_colbert = doc_embeddings[did]['colbert']
            # d_colbert: (D_len, 1024)
            
            # Sim Matrix: (Q_len, D_len)
            sim_matrix = np.dot(q_colbert, d_colbert.T) 
            # Max over docs (axis 1)
            max_sim = np.max(sim_matrix, axis=1)
            # Sum over query (axis 0)
            score_colbert[j] = np.sum(max_sim)

        # FINAL COMBINATION
        # BGE-M3 default weights: Dense + Sparse + ColBERT
        # However, magnitudes differ heavily.
        # Usually Sum is fine if trained that way.
        # Official repo uses simple sum of unnormalized scores? Or weighted?
        # Weighted sum: 1.0 * Dense + 0.3 * Sparse + 1.0 * ColBERT ?
        # Let's simple sum for "All" demonstration.
        
        total_score = score_dense + score_sparse + score_colbert
        
        top_indices = np.argsort(total_score)[::-1][:top_k]
        
        result_dict = {}
        for idx in top_indices:
            result_dict[doc_ids[idx]] = float(total_score[idx])
        results[qid] = result_dict
        
        latencies.append(time.perf_counter() - t_start)

    total_time = time.time() - start_total
    
    latencies_ms = np.array(latencies) * 1000
    p99 = np.percentile(latencies_ms, 99)
    qps = len(q_ids) / total_time
    
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 10, 100])
    
    print("--- Stats (ALL) ---")
    print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
    print(f"Recall@10: {recall['Recall@10']:.4f}")
    print(f"Latency P99: {p99:.4f} ms")
    print(f"QPS: {qps:.4f}")
    
    return {
        "model": "bge-m3-all",
        "ndcg@10": ndcg['NDCG@10'],
        "recall@10": recall['Recall@10'],
        "p99": p99,
        "qps": qps
    }


def evaluate_hybrid(dense_name, sparse_name, alpha, qrels, limit):
    """
    Evaluate hybrid search combining dense and sparse scores.
    Final_Score = alpha * Dense_Score + (1 - alpha) * Sparse_Score
    """
    print(f"\n================ HYBRID: {dense_name.upper()} + {sparse_name.upper()} (alpha={alpha}) ================")
    
    # Load embeddings
    dense_emb_path = f"data/embeddings/{dense_name}_embeddings.pkl"
    sparse_emb_path = f"data/embeddings/{sparse_name}_embeddings.pkl"
    
    try:
        dense_doc_emb = load_embeddings(dense_emb_path)
        sparse_doc_emb = load_embeddings(sparse_emb_path)
    except FileNotFoundError as e:
        print(f"Embeddings not found: {e}")
        return None
    
    # Load artifacts
    dense_artifact = load_artifact(dense_emb_path.replace("_embeddings.pkl", "_artifact.pkl"))
    sparse_artifact = load_artifact(sparse_emb_path.replace("_embeddings.pkl", "_artifact.pkl"))
    
    # Get common doc_ids (should be same if generated with same limit)
    doc_ids = list(dense_doc_emb.keys())[:limit]
    
    # Load data for queries
    corpus, queries, qrels_full = load_dbpedia_data()
    corpus, queries, qrels = filter_dataset(corpus, queries, qrels_full, limit)
    
    if len(queries) == 0:
        print("No valid queries!")
        return None
    
    # Encode queries for both models
    print("Encoding queries for dense model...")
    if dense_name == "bge-m3":
        dense_embedder = BGEM3Embedder()
    else:  # word2vec
        dense_embedder = Word2VecEmbedder()
    dense_query_emb = dense_embedder.encode_queries(queries, dense_artifact)
    
    print("Encoding queries for sparse model...")
    if sparse_name == "splade":
        sparse_embedder = SpladeEmbedder()
    else:  # bm25
        sparse_embedder = BM25Embedder()
    sparse_query_emb = sparse_embedder.encode_queries(queries, sparse_artifact)
    
    q_ids = list(queries.keys())
    
    # Prepare Dense matrices
    print("Preparing dense matrices...")
    dense_doc_matrix = np.array([dense_doc_emb[did] for did in doc_ids])
    dense_query_matrix = np.array([dense_query_emb[qid] for qid in q_ids])
    
    # Normalize for cosine similarity
    norm_d = np.linalg.norm(dense_doc_matrix, axis=1, keepdims=True)
    norm_d[norm_d == 0] = 1
    dense_doc_matrix = dense_doc_matrix / norm_d
    
    norm_q = np.linalg.norm(dense_query_matrix, axis=1, keepdims=True)
    norm_q[norm_q == 0] = 1
    dense_query_matrix = dense_query_matrix / norm_q
    
    # Prepare Sparse matrices
    print("Preparing sparse matrices...")
    doc_map = {did: i for i, did in enumerate(doc_ids)}
    
    if sparse_name == "bm25":
        # Build term_to_id mapping
        term_to_id = {}
        for vec in sparse_doc_emb.values():
            for term in vec:
                if term not in term_to_id:
                    term_to_id[term] = len(term_to_id)
        for vec in sparse_query_emb.values():
            for term in vec:
                if term not in term_to_id:
                    term_to_id[term] = len(term_to_id)
        
        vocab_size = len(term_to_id)
        
        # Doc matrix
        row_ind, col_ind, data = [], [], []
        for did in doc_ids:
            vec = sparse_doc_emb.get(did, {})
            idx = doc_map[did]
            for term, weight in vec.items():
                row_ind.append(idx)
                col_ind.append(term_to_id[term])
                data.append(weight)
        sparse_doc_matrix = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(doc_ids), vocab_size))
        
        # Query matrix
        q_row, q_col, q_data = [], [], []
        for i, qid in enumerate(q_ids):
            vec = sparse_query_emb.get(qid, {})
            for term, weight in vec.items():
                if term in term_to_id:
                    q_row.append(i)
                    q_col.append(term_to_id[term])
                    q_data.append(weight)
        sparse_query_matrix = scipy.sparse.csr_matrix((q_data, (q_row, q_col)), shape=(len(q_ids), vocab_size))
        
    else:  # splade
        max_idx = 30522
        for vec in sparse_doc_emb.values():
            if vec:
                max_idx = max(max_idx, max(vec.keys()))
        max_idx += 1
        
        row_ind, col_ind, data = [], [], []
        for did in doc_ids:
            vec = sparse_doc_emb.get(did, {})
            idx = doc_map[did]
            for tid, w in vec.items():
                row_ind.append(idx)
                col_ind.append(tid)
                data.append(w)
        sparse_doc_matrix = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(doc_ids), max_idx))
        
        q_row, q_col, q_data = [], [], []
        for i, qid in enumerate(q_ids):
            vec = sparse_query_emb.get(qid, {})
            for tid, w in vec.items():
                if tid < max_idx:
                    q_row.append(i)
                    q_col.append(tid)
                    q_data.append(w)
        sparse_query_matrix = scipy.sparse.csr_matrix((q_data, (q_row, q_col)), shape=(len(q_ids), max_idx))
    
    # Run hybrid search
    print("Running hybrid search...")
    top_k = 10
    results = {}
    latencies = []
    
    sparse_doc_matrix_T = sparse_doc_matrix.transpose()
    
    start_total = time.time()
    
    for i, qid in enumerate(tqdm(q_ids, desc="Hybrid Searching")):
        t_start = time.perf_counter()
        
        # Dense scores (cosine similarity)
        dense_scores = np.dot(dense_doc_matrix, dense_query_matrix[i])
        
        # Sparse scores (dot product)
        sparse_q_vec = sparse_query_matrix[i]
        sparse_scores = sparse_q_vec.dot(sparse_doc_matrix_T)
        if scipy.sparse.issparse(sparse_scores):
            sparse_scores = sparse_scores.toarray().flatten()
        else:
            sparse_scores = np.array(sparse_scores).flatten()
        
        # Normalize scores to [0, 1] range (min-max normalization)
        # Dense
        d_min, d_max = dense_scores.min(), dense_scores.max()
        if d_max - d_min > 0:
            dense_scores_norm = (dense_scores - d_min) / (d_max - d_min)
        else:
            dense_scores_norm = np.zeros_like(dense_scores)
        
        # Sparse
        s_min, s_max = sparse_scores.min(), sparse_scores.max()
        if s_max - s_min > 0:
            sparse_scores_norm = (sparse_scores - s_min) / (s_max - s_min)
        else:
            sparse_scores_norm = np.zeros_like(sparse_scores)
        
        # Combine: alpha * dense + (1 - alpha) * sparse
        hybrid_scores = alpha * dense_scores_norm + (1 - alpha) * sparse_scores_norm
        
        # Top-K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        result_dict = {}
        for idx in top_indices:
            result_dict[doc_ids[idx]] = float(hybrid_scores[idx])
        results[qid] = result_dict
        
        latencies.append(time.perf_counter() - t_start)
    
    total_time = time.time() - start_total
    
    # Metrics
    latencies_ms = np.array(latencies) * 1000
    p99 = np.percentile(latencies_ms, 99)
    qps = len(q_ids) / total_time
    
    # Quality metrics
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 10, 100])
    
    print("--- Hybrid Stats ---")
    print(f"Combination: {dense_name} + {sparse_name}")
    print(f"Alpha: {alpha}")
    print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
    print(f"Recall@10: {recall['Recall@10']:.4f}")
    print(f"Latency P99: {p99:.4f} ms")
    print(f"QPS: {qps:.4f}")
    
    return {
        "model": f"{dense_name}+{sparse_name}",
        "alpha": alpha,
        "ndcg@10": ndcg['NDCG@10'],
        "recall@10": recall['Recall@10'],
        "p99": p99,
        "qps": qps
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--hybrid-only", action="store_true", help="Only run hybrid experiments")
    parser.add_argument("--model", type=str, default="all", help="Specific model to evaluate (optional)")
    args = parser.parse_args()
    
    print("Loading data...")
    corpus, queries, qrels = load_dbpedia_data()
    
    print(f"Filtering to {args.limit} documents...")
    corpus, queries, qrels = filter_dataset(corpus, queries, qrels, args.limit)
    print(f"Filtered: {len(corpus)} docs, {len(queries)} queries.")
    
    if len(queries) == 0:
        print("No valid queries found for this subset! Increase limit or check data.")
        return

    stats_list = []
    
    if not args.hybrid_only:
        if args.model != "all":
            models = [args.model]
        else:
            models = ["splade", "bge-m3", "bm25", "word2vec"]
            
        for model in models:
            print(f"\n================ {model.upper()} ================")
            emb_path = f"data/embeddings/{model}_embeddings.pkl"
            if model == "bge-m3-all":
                stats = evaluate_bge_m3_all(emb_path, corpus, queries, qrels, args.limit)
            else:
                stats = evaluate_model(model, emb_path, corpus, queries, qrels)
            
            if stats:
                stats_list.append(stats)
    
    # Hybrid experiments
    print("\n\n========== HYBRID EXPERIMENTS ==========")
    hybrid_combos = [
        ("bge-m3", "splade"),
        ("bge-m3", "bm25"),
        ("word2vec", "splade"),
        ("word2vec", "bm25"),
    ]
    
    hybrid_stats = []
    for dense_name, sparse_name in hybrid_combos:
        stats = evaluate_hybrid(dense_name, sparse_name, args.alpha, qrels, args.limit)
        if stats:
            hybrid_stats.append(stats)
            
    # Final Tables
    if stats_list:
        print("\n\n========== SINGLE MODEL RESULTS ==========")
        print(f"{'Model':<15} {'NDCG@10':<10} {'Recall@10':<10} {'P99 (ms)':<10} {'QPS':<10}")
        print("-" * 60)
        for s in stats_list:
            print(f"{s['model']:<15} {s['ndcg@10']:.4f}     {s['recall@10']:.4f}     {s['p99']:.2f}         {s['qps']:.2f}")
    
    print("\n\n========== HYBRID RESULTS (alpha={}) ==========".format(args.alpha))
    print(f"{'Combination':<25} {'NDCG@10':<10} {'Recall@10':<10} {'P99 (ms)':<10} {'QPS':<10}")
    print("-" * 70)
    for s in hybrid_stats:
        print(f"{s['model']:<25} {s['ndcg@10']:.4f}     {s['recall@10']:.4f}     {s['p99']:.2f}         {s['qps']:.2f}")

if __name__ == "__main__":
    main()

