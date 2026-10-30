"""
Embedding Cache Manager Module.

Manages persistent disk storage and fast indexed retrieval of extracted
acoustic representations across all few-shot experimental splits.
"""

import os
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


class EmbeddingCacheManager:
    """
    Persistent indexed cache for acoustic embeddings.
    Enables fast random access and batch lookups by recording_id.
    """

    def __init__(self, expected_dim: int = 384):
        self.expected_dim = expected_dim
        self._cache: Dict[str, np.ndarray] = {}

    def add(self, recording_id: str, embedding: np.ndarray) -> None:
        """Add a single embedding vector to the cache."""
        if embedding.shape != (self.expected_dim,):
            raise ValueError(f"Embedding shape {embedding.shape} does not match expected ({self.expected_dim},)")
        self._cache[recording_id] = embedding.astype(np.float32)

    def add_batch(self, recording_ids: List[str], embeddings: np.ndarray) -> None:
        """Add a batch of embeddings to the cache."""
        if len(recording_ids) != len(embeddings):
            raise ValueError(f"Length mismatch: {len(recording_ids)} IDs vs {len(embeddings)} embeddings")
        for rid, emb in zip(recording_ids, embeddings):
            self.add(rid, emb)

    def get(self, recording_id: str) -> np.ndarray:
        """Retrieve embedding vector for a given recording ID."""
        if recording_id not in self._cache:
            raise KeyError(f"Recording ID not found in embedding cache: {recording_id}")
        return self._cache[recording_id]

    def get_batch(self, recording_ids: List[str]) -> np.ndarray:
        """Retrieve 2D matrix of embeddings for a list of recording IDs."""
        vectors = [self.get(rid) for rid in recording_ids]
        return np.vstack(vectors)

    def contains(self, recording_id: str) -> bool:
        """Check if recording ID is present in cache."""
        return recording_id in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def save(self, file_path: str) -> None:
        """Save the in-memory cache to a compressed NumPy .npz archive."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        rids = np.array(list(self._cache.keys()), dtype=object)
        embs = np.vstack(list(self._cache.values())).astype(np.float32)
        np.savez_compressed(file_path, recording_ids=rids, embeddings=embs)
        print(f"Saved embedding cache to {file_path} ({len(rids)} items, shape: {embs.shape})")

    def load(self, file_path: str) -> None:
        """Load cache from a compressed NumPy .npz archive."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cache file not found: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        rids = data["recording_ids"]
        embs = data["embeddings"]
        self._cache = {str(rid): embs[i] for i, rid in enumerate(rids)}
        print(f"Loaded embedding cache from {file_path} ({len(self._cache)} items)")

    def validate_cache(self) -> Dict[str, Any]:
        """Audit the cache for integrity: finiteness, shape, zero-vectors."""
        total = len(self._cache)
        if total == 0:
            return {"total": 0, "status": "EMPTY"}

        embs = np.vstack(list(self._cache.values()))
        has_nan = bool(np.isnan(embs).any())
        has_inf = bool(np.isinf(embs).any())
        norms = np.linalg.norm(embs, axis=1)
        zero_vectors = int(np.sum(norms < 1e-6))

        status = "VALID"
        if has_nan or has_inf or zero_vectors > 0:
            status = "INVALID"

        return {
            "total_items": total,
            "embedding_dim": int(embs.shape[1]),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "zero_vectors_count": zero_vectors,
            "mean_norm": float(np.mean(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "status": status
        }
