"""
HNSW Indexer Wrapper using Faiss

This module provides a high-level interface for building and querying
Hierarchical Navigable Small World (HNSW) indexes for approximate
nearest neighbor search.

Technical Details:
- Library: faiss-cpu
- Index Type: IndexHNSWFlat
- Default Parameters:
  - M (connections per layer): 32
  - efConstruction (build accuracy): 200
  - efSearch (query accuracy): 128

References:
- Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate 
  nearest neighbor search using hierarchical navigable small world graphs."
  IEEE TPAMI, 2018.
"""

import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional
import time


class HNSWIndexer:
    """
    High-level wrapper for Faiss HNSW index.
    
    Attributes:
        dimension (int): Vector dimensionality.
        M (int): Number of connections per node.
        ef_construction (int): Build-time search accuracy.
        ef_search (int): Query-time search accuracy.
        index (faiss.Index): Underlying Faiss index.
        id_map (Dict[int, str]): Maps internal index IDs to original document IDs.
    """
    
    def __init__(
        self,
        dimension: int,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128
    ):
        """
        Initialize HNSW index.
        
        Args:
            dimension: Vector dimensionality.
            M: Number of bi-directional links per node. Higher = better recall, more memory.
            ef_construction: Size of dynamic candidate list during construction.
            ef_search: Size of dynamic candidate list during search.
        """
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        
        # ID mapping: internal faiss ID (int) -> original document ID (str)
        self.id_map: Dict[int, str] = {}
        self._next_id = 0
        
    def add(self, doc_id: str, vector: np.ndarray) -> None:
        """
        Add a single vector to the index.
        
        Args:
            doc_id: Original document identifier (string).
            vector: Dense embedding (1D numpy array).
        """
        vector = np.ascontiguousarray(vector.astype(np.float32).reshape(1, -1))
        self.index.add(vector)
        self.id_map[self._next_id] = doc_id
        self._next_id += 1
        
    def add_batch(self, doc_ids: List[str], vectors: np.ndarray) -> None:
        """
        Add multiple vectors to the index.
        
        Args:
            doc_ids: List of document identifiers.
            vectors: Dense embeddings matrix (N x dimension).
        """
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        start_id = self._next_id
        self.index.add(vectors)
        
        for i, doc_id in enumerate(doc_ids):
            self.id_map[start_id + i] = doc_id
        self._next_id += len(doc_ids)
        
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_vector: Query embedding (1D numpy array).
            top_k: Number of results to return.
            
        Returns:
            List of (doc_id, score) tuples, sorted by descending similarity.
        """
        query_vector = np.ascontiguousarray(
            query_vector.astype(np.float32).reshape(1, -1)
        )
        
        # Faiss returns (distances, indices)
        # For inner product / L2, lower distance = more similar
        # HNSWFlat uses L2 by default. We can use IndexFlatIP for inner product.
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 means no result
                doc_id = self.id_map.get(idx, f"unknown_{idx}")
                # Convert L2 distance to similarity (higher = better)
                # For L2: similarity = -distance (or 1/(1+distance))
                # For cosine: just use the score
                # Since we're using normalized vectors, L2 relates to cosine
                similarity = -dist  # or use 1.0 / (1.0 + dist)
                results.append((doc_id, similarity))
                
        return results
    
    def search_batch(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch search for nearest neighbors.
        
        Args:
            query_vectors: Query embeddings matrix (N x dimension).
            top_k: Number of results per query.
            
        Returns:
            List of result lists, one per query.
        """
        query_vectors = np.ascontiguousarray(
            query_vectors.astype(np.float32)
        )
        
        distances, indices = self.index.search(query_vectors, top_k)
        
        all_results = []
        for i in range(len(query_vectors)):
            results = []
            for dist, idx in zip(distances[i], indices[i]):
                if idx != -1:
                    doc_id = self.id_map.get(idx, f"unknown_{idx}")
                    similarity = -dist
                    results.append((doc_id, similarity))
            all_results.append(results)
            
        return all_results
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        import pickle
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.idmap", "wb") as f:
            pickle.dump({"id_map": self.id_map, "next_id": self._next_id}, f)
            
    def load(self, path: str) -> None:
        """Load index from disk."""
        import pickle
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.idmap", "rb") as f:
            data = pickle.load(f)
            self.id_map = data["id_map"]
            self._next_id = data["next_id"]
            
    @property
    def size(self) -> int:
        """Return number of indexed vectors."""
        return self.index.ntotal


class HNSWIndexerIP(HNSWIndexer):
    """
    HNSW Indexer using Inner Product similarity (cosine for normalized vectors).
    
    This is preferred when vectors are L2-normalized, as it directly computes
    cosine similarity without the L2 -> cosine conversion overhead.
    """
    
    def __init__(
        self,
        dimension: int,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128
    ):
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Inner Product quantizer
        quantizer = faiss.IndexFlatIP(dimension)
        # HNSW with IP similarity
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        
        self.id_map: Dict[int, str] = {}
        self._next_id = 0
        
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search using inner product (no distance conversion needed)."""
        query_vector = np.ascontiguousarray(
            query_vector.astype(np.float32).reshape(1, -1)
        )
        
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                doc_id = self.id_map.get(idx, f"unknown_{idx}")
                results.append((doc_id, float(score)))
                
        return results


class FlatIPIndexer:
    """
    Simple brute-force inner product indexer using IndexFlatIP.
    
    This is a fallback for systems where HNSW causes stability issues
    (e.g., Apple Silicon with certain Python versions).
    
    Uses O(N) search but is very stable and fast for small N (< 100K).
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.id_map: Dict[int, str] = {}
        self._next_id = 0
        
    def add_batch(self, doc_ids: List[str], vectors: np.ndarray) -> None:
        """Add multiple vectors to the index."""
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
        start_id = self._next_id
        self.index.add(vectors)
        
        for i, doc_id in enumerate(doc_ids):
            self.id_map[start_id + i] = doc_id
        self._next_id += len(doc_ids)
        
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for nearest neighbors."""
        query_vector = np.ascontiguousarray(
            query_vector.astype(np.float32).reshape(1, -1)
        )
        
        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                doc_id = self.id_map.get(idx, f"unknown_{idx}")
                results.append((doc_id, float(score)))
                
        return results
    
    @property
    def size(self) -> int:
        return self.index.ntotal
